import numpy as np
from algorithms.cost import (
    build_daily_occupancy,
    cost_function,
    delta_cost,
    total_accounting_cost,
    preference_penalty,
)
from algorithms.initial_solution import build_individual
from config import (
    GA_POPULATION_SIZE as POPULATION_SIZE,
    GA_N_GENERATIONS as N_GENERATIONS,
    GA_TOURNAMENT_SIZE as TOURNAMENT_SIZE,
    GA_CROSSOVER_RATE as CROSSOVER_RATE,
    GA_MUTATION_RATE as MUTATION_RATE,
    GA_ELITE_COUNT as ELITE_COUNT,
    GA_PLATEAU_LIMIT as PLATEAU_LIMIT,
    GA_LOCAL_SEARCH_EVERY as LOCAL_SEARCH_EVERY,
)


def _total_score(preference_cost, accounting_cost):
    """Sum a (preference_cost, accounting_cost) pair into a single score."""
    return preference_cost + accounting_cost


def init_population(
    initial_assignment,
    family_sizes,
    family_choices,
    n_days,
    min_occupancy,
    max_occupancy,
    rng,
):
    """
    Build the initial population by repeatedly applying the randomized build_individual procedure. 
    Each run produces a valid solution with different random choices, giving us a diverse starting population.
    """
    population, occupancies, per_individual_costs = [], [], []

    assignment = initial_assignment.copy()
    daily_occupancy = build_daily_occupancy(initial_assignment, family_sizes, n_days)
    while len(population) < POPULATION_SIZE:
        assignment, daily_occupancy = build_individual(
            initial_assignment,
            family_sizes,
            family_choices,
            n_days,
            min_occupancy,
            max_occupancy,
            rng,
        )

        costs = cost_function(
            assignment,
            family_sizes,
            family_choices,
            n_days,
            min_occupancy,
            max_occupancy,
        )
        if costs is not None:
            population.append(assignment)
            occupancies.append(daily_occupancy)
            per_individual_costs.append(costs)

    return population, occupancies, per_individual_costs


def _tournament_select(population, occupancies, per_individual_costs, rng):
    """Pick the best individual from a random sample (tournament selection)."""
    candidate_indices = rng.integers(0, len(population), size=TOURNAMENT_SIZE)
    winner_index = candidate_indices[
        np.argmin([_total_score(*per_individual_costs[i]) for i in candidate_indices])
    ]
    return (
        population[winner_index].copy(),
        occupancies[winner_index].copy(),
        per_individual_costs[winner_index],
    )


def _crossover(
    parent_a_assignment,
    parent_b_assignment,
    family_sizes,
    family_choices,
    n_days,
    min_occupancy,
    max_occupancy,
    rng,
):
    """
    Build a child by combining two parents.

    - Assign families in random order
    - Prefer one parent randomly per family
    - Try parent days first, then fallback to preferences
    - Enforce max occupancy constraints
    """

    n_families = len(parent_a_assignment)

    child_assignment = np.empty(n_families, dtype=parent_a_assignment.dtype)
    child_occupancy = np.zeros(n_days + 1, dtype=np.int32)
    child_preference_cost = 0.0

    # Randomly decide which parent each family prefers
    prefers_parent_a = rng.integers(0, 2, size=n_families).astype(bool)

    # Process families in random order
    for family_id in rng.permutation(n_families):

        family_size = family_sizes[family_id]

        # Build priority list of days
        if prefers_parent_a[family_id]:
            preferred_days = [
                parent_a_assignment[family_id],
                parent_b_assignment[family_id],
            ]
        else:
            preferred_days = [
                parent_b_assignment[family_id],
                parent_a_assignment[family_id],
            ]

        preferred_days += list(family_choices[family_id])

        # Try to place family
        chosen_day = None

        for day in preferred_days:
            if child_occupancy[day] + family_size <= max_occupancy:
                chosen_day = day
                break

        if chosen_day is None:
            chosen_day = min(
                family_choices[family_id],
                key=lambda d: child_occupancy[d],
            )

        # Assign family
        child_assignment[family_id] = chosen_day
        child_occupancy[chosen_day] += family_size

        # Update cost
        child_preference_cost += preference_penalty(
            family_id,
            chosen_day,
            family_sizes,
            family_choices,
        )

    # Validate occupancy constraints
    valid = np.all(child_occupancy[1:] <= max_occupancy) and np.all(
        child_occupancy[1:] >= min_occupancy
    )

    if not valid:
        return None

    # Compute accounting cost only if valid
    child_accounting_cost = total_accounting_cost(
        child_occupancy,
        n_days,
        min_occupancy,
    )

    return (
        child_assignment,
        child_occupancy,
        (child_preference_cost, child_accounting_cost),
    )


def _mutate(
    assignment,
    daily_occupancy,
    costs,
    family_sizes,
    family_choices,
    n_days,
    min_occupancy,
    max_occupancy,
    rng,
):
    """
    Randomly reassign a small fraction of families to a different preferred day.
    Uses delta_cost to update (preference_cost, accounting_cost) incrementally — no full recompute.
    A move only happens if it keeps both days within occupancy bounds.
    """
    mutated_assignment = assignment.copy()
    mutated_occupancy = daily_occupancy.copy()
    preference_cost, accounting_cost = costs

    for family_id in range(len(mutated_assignment)):
        if rng.random() >= MUTATION_RATE:
            continue
        current_day = mutated_assignment[family_id]
        candidate_day = family_choices[family_id, rng.integers(0, 10)]
        if candidate_day == current_day:
            continue

        cost_delta = delta_cost(
            family_id,
            current_day,
            candidate_day,
            accounting_cost,
            mutated_occupancy,
            family_sizes,
            family_choices,
            n_days,
            min_occupancy,
            max_occupancy,
        )
        if cost_delta is not None:
            preference_cost += cost_delta[0]
            accounting_cost += cost_delta[1]
            mutated_occupancy[current_day] -= family_sizes[family_id]
            mutated_occupancy[candidate_day] += family_sizes[family_id]
            mutated_assignment[family_id] = candidate_day

    return mutated_assignment, mutated_occupancy, (preference_cost, accounting_cost)


def _local_search(
    assignment,
    daily_occupancy,
    costs,
    family_sizes,
    family_choices,
    n_days,
    min_occupancy,
    max_occupancy,
    rng,
):
    """
    Greedily improve an assignment by trying to move each family to a better day.
    """
    preference_cost, accounting_cost = costs

    for family_id in rng.permutation(len(assignment)):
        current_day = assignment[family_id]
        for rank in range(10):
            candidate_day = family_choices[family_id, rank]
            if candidate_day == current_day:
                continue
            cost_delta = delta_cost(
                family_id,
                current_day,
                candidate_day,
                accounting_cost,
                daily_occupancy,
                family_sizes,
                family_choices,
                n_days,
                min_occupancy,
                max_occupancy,
            )
            if cost_delta is not None and sum(cost_delta) < 0:
                daily_occupancy[current_day] -= family_sizes[family_id]
                daily_occupancy[candidate_day] += family_sizes[family_id]
                assignment[family_id] = candidate_day
                preference_cost += cost_delta[0]
                accounting_cost += cost_delta[1]
                current_day = candidate_day
                break

    return assignment, daily_occupancy, (preference_cost, accounting_cost)


_callback = None
_stop = False


def emit(event: dict):
    if _callback is not None:
        _callback(event)


def run(submission, family_sizes, family_choices, n_days, min_occupancy, max_occupancy):
    """
    Run the genetic algorithm.

    1. Seed the population from the given submission.
    2. Each generation: keep the elite, breed the rest, occasionally polish the best.
    3. Stop early if no improvement for PLATEAU_LIMIT generations.
    4. Run a final local search on the best solution found.

    All cost tracking uses (preference_cost, accounting_cost) tuples updated via delta_cost;
    cost_function is called only once per individual at initialisation time.
    Invalid individuals are never stored — crossover retries up to CROSSOVER_MAX_ATTEMPTS
    times before falling back to a cloned-and-mutated parent.
    """
    initial_assignment = submission["assigned_day"].values.copy()

    initial_costs = cost_function(
        initial_assignment,
        family_sizes,
        family_choices,
        n_days,
        min_occupancy,
        max_occupancy,
    )

    if initial_costs is None:
        raise ValueError("Starting solution is invalid!")
    
    preference_cost, accounting_cost = initial_costs

    print(
        f"Initial score: {(preference_cost + accounting_cost):.2f}  "
        f"(preference: {preference_cost:.2f}  accounting: {accounting_cost:.2f})"
    )
    emit(
        {
            "type": "start",
            "score": (preference_cost + accounting_cost),
            "preference": preference_cost,
            "accounting": accounting_cost,
        }
    )

    rng = np.random.default_rng()
    population, occupancies, per_individual_costs = init_population(
        initial_assignment,
        family_sizes,
        family_choices,
        n_days,
        min_occupancy,
        max_occupancy,
        rng,
    )

    scores = [_total_score(*costs) for costs in per_individual_costs]
    best_individual_index = int(np.argmin(scores))
    best_score = scores[best_individual_index]
    best_assignment = population[best_individual_index].copy()
    best_occupancy = occupancies[best_individual_index].copy()
    best_costs = per_individual_costs[best_individual_index]
    score_history = [[0, best_score]]
    generations_without_improvement = 0
    print(f"Initial population best: {best_score:.2f}")

    for generation in range(1, N_GENERATIONS + 1):
        if _stop:
            print("Stopping: stop requested.")
            break

        # Always carry the top elite individuals into the next generation
        elite_indices = np.argsort(scores)[:ELITE_COUNT]
        next_population = [population[i].copy() for i in elite_indices]
        next_occupancies = [occupancies[i].copy() for i in elite_indices]
        next_costs = [per_individual_costs[i] for i in elite_indices]

        # Fill the rest of the population with bred/mutated children
        while len(next_population) < POPULATION_SIZE:
            if _stop:
                break

            child, child_occupancy, child_costs = None, None, None

            if rng.random() < CROSSOVER_RATE:
                parent_a, _, _ = _tournament_select(
                    population, occupancies, per_individual_costs, rng
                )
                parent_b, _, _ = _tournament_select(
                    population, occupancies, per_individual_costs, rng
                )
                
                crossover_result = _crossover(
                    parent_a,
                    parent_b,
                    family_sizes,
                    family_choices,
                    n_days,
                    min_occupancy,
                    max_occupancy,
                    rng,
                )
                if crossover_result is not None:
                    child, child_occupancy, child_costs = crossover_result

            # Fall back to cloning a tournament winner if crossover was skipped or failed
            if child is None:
                child, child_occupancy, child_costs = _tournament_select(
                    population, occupancies, per_individual_costs, rng
                )

            child, child_occupancy, child_costs = _mutate(
                child,
                child_occupancy,
                child_costs,
                family_sizes,
                family_choices,
                n_days,
                min_occupancy,
                max_occupancy,
                rng,
            )
            next_population.append(child)
            next_occupancies.append(child_occupancy)
            next_costs.append(child_costs)

        if _stop:
            break

        population, occupancies, per_individual_costs = (
            next_population,
            next_occupancies,
            next_costs,
        )
        scores = [_total_score(*costs) for costs in per_individual_costs]

        # Periodically polish the best individual with a greedy local search
        if generation % LOCAL_SEARCH_EVERY == 0:
            best_in_generation_index = int(np.argmin(scores))
            polished_assignment, polished_occupancy, polished_costs = _local_search(
                population[best_in_generation_index].copy(),
                occupancies[best_in_generation_index].copy(),
                per_individual_costs[best_in_generation_index],
                family_sizes,
                family_choices,
                n_days,
                min_occupancy,
                max_occupancy,
                rng,
            )
            polished_score = _total_score(*polished_costs)
            if polished_score < scores[best_in_generation_index]:
                population[best_in_generation_index] = polished_assignment
                occupancies[best_in_generation_index] = polished_occupancy
                per_individual_costs[best_in_generation_index] = polished_costs
                scores[best_in_generation_index] = polished_score

        # Track progress and check for plateau
        best_in_generation_index = int(np.argmin(scores))
        best_score_in_generation = scores[best_in_generation_index]
        scores = [s for s in scores if s < float("inf")]
        print(
            f"Gen {generation:4d}: best={best_score_in_generation:.2f}  "
            f"avg={np.mean(scores):.2f}  "
            f"global_best={best_score:.2f}  "
            f"plateau={generations_without_improvement}/{PLATEAU_LIMIT}"
            + ("  *** NEW BEST ***" if best_score_in_generation < best_score else "")
        )

        emit(
            {
                "type": "iteration",
                "score": best_score_in_generation,
                "best_score": best_score,
                "preference": per_individual_costs[best_in_generation_index][0],
                "accounting": per_individual_costs[best_in_generation_index][1],
                "iteration": generation,
                "improvements": 0 if best_score_in_generation >= best_score else 1,
            }
        )

        if best_score_in_generation < best_score:
            best_score = best_score_in_generation
            best_assignment = population[best_in_generation_index].copy()
            best_occupancy = occupancies[best_in_generation_index].copy()
            best_costs = per_individual_costs[best_in_generation_index]
            score_history.append([generation, best_score])
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        if generations_without_improvement >= PLATEAU_LIMIT:
            print(f"Stopping early: no improvement for {PLATEAU_LIMIT} generations.")
            break

    emit(
        {
            "type": "done",
            "score": best_score,
            "preference": best_costs[0],
            "accounting": best_costs[1],
            "occupancy": best_occupancy.tolist(),
            "assignment": best_assignment.tolist(),
        }
    )

    print(f"\nBest score found: {best_score:.2f}")
    return best_assignment, best_score, score_history
