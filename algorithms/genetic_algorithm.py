import numpy as np
from algorithms.cost import (
    build_daily_occupancy,
    cost_function,
    delta_cost,
    total_accounting_cost,
    _preference_penalty,
)
from algorithms.initial_solution import build_individual

POPULATION_SIZE = 15  # number of candidate solutions kept at once
N_GENERATIONS = 500  # max evolution cycles
TOURNAMENT_SIZE = 3  # candidates compared in each selection draw
CROSSOVER_RATE = 0.80  # chance two parents breed (vs. cloning one parent)
MUTATION_RATE = 0.04  # per-family chance of a random day reassignment
ELITE_COUNT = 2  # top solutions copied unchanged to the next generation
PLATEAU_LIMIT = 10  # stop early if best score hasn't improved this many generations
LOCAL_SEARCH_EVERY = 5  # how often (in generations) to polish the best individual
CROSSOVER_MAX_ATTEMPTS = (
    5  # retries before crossover gives up and falls back to mutation-only
)

PREFERENCE_PENALTIES = [
    (0, 0),
    (50, 0),
    (50, 9),
    (100, 9),
    (200, 9),
    (200, 18),
    (300, 18),
    (300, 36),
    (400, 36),
    (500, 235),
    (500, 434),
]


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
    parent_a_occupancy,
    parent_a_costs,
    parent_b_assignment,
    parent_b_occupancy,
    parent_b_costs,
    family_sizes,
    family_choices,
    n_days,
    min_occupancy,
    max_occupancy,
    rng,
):
    """
    Combine two parents into a child. For each family, randomly prefer one parent's
    day; if it's full, try the other; if still full, fall back to the family's own
    preferences. Families are placed in random order to avoid systematic bias.

    Preference cost is accumulated incrementally during construction.
    Accounting cost is computed once at the end (it depends on the full occupancy array).

    Returns (assignment, occupancy, costs) on success, or None if the child violates
    occupancy bounds after CROSSOVER_MAX_ATTEMPTS independent attempts.
    """
    for _ in range(CROSSOVER_MAX_ATTEMPTS):
        n_families = len(parent_a_assignment)
        child_occupancy = np.zeros(n_days + 1, dtype=np.int32)
        child_assignment = np.empty(n_families, dtype=parent_a_assignment.dtype)
        prefer_parent_a = rng.integers(0, 2, size=n_families).astype(bool)
        child_preference_cost = 0.0

        for family_id in rng.permutation(n_families):
            family_size = family_sizes[family_id]
            days_to_try = (
                [parent_a_assignment[family_id], parent_b_assignment[family_id]]
                if prefer_parent_a[family_id]
                else [parent_b_assignment[family_id], parent_a_assignment[family_id]]
            ) + list(family_choices[family_id])

            for day in days_to_try:
                if child_occupancy[day] + family_size <= max_occupancy:
                    break
            else:
                day = family_choices[
                    family_id,
                    np.argmin([child_occupancy[d] for d in family_choices[family_id]]),
                ]

            child_occupancy[day] += family_size
            child_assignment[family_id] = day
            child_preference_cost += _preference_penalty(
                family_id, day, family_sizes, family_choices
            )

        if not (
            np.any(child_occupancy[1:] > max_occupancy)
            or np.any(child_occupancy[1:] < min_occupancy)
        ):
            child_accounting_cost = total_accounting_cost(
                child_occupancy, n_days, min_occupancy
            )
            return (
                child_assignment,
                child_occupancy,
                (child_preference_cost, child_accounting_cost),
            )

    return None


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
    Uses delta_cost throughout — no cost_function calls inside the loop.
    Called sparingly (every LOCAL_SEARCH_EVERY generations) as a fine-tuning step.
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
    Run the genetic algorithm. Returns (best_assignment, best_score, score_history).

    Flow:
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
        f"Initial score: {sum(initial_costs):.2f}  "
        f"(preference: {preference_cost:.2f}  accounting: {accounting_cost:.2f})"
    )
    emit(
        {
            "type": "start",
            "score": sum(initial_costs),
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

        # Always carry the top ELITE_COUNT individuals into the next generation
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
                parent_a, parent_a_occ, parent_a_costs = _tournament_select(
                    population, occupancies, per_individual_costs, rng
                )
                parent_b, parent_b_occ, parent_b_costs = _tournament_select(
                    population, occupancies, per_individual_costs, rng
                )
                crossover_result = _crossover(
                    parent_a,
                    parent_a_occ,
                    parent_a_costs,
                    parent_b,
                    parent_b_occ,
                    parent_b_costs,
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

    # Final polish on the overall best solution
    if not _stop:
        print("\nRunning final local search polish...")
        best_assignment, best_occupancy, best_costs = _local_search(
            best_assignment,
            best_occupancy,
            best_costs,
            family_sizes,
            family_choices,
            n_days,
            min_occupancy,
            max_occupancy,
            rng,
        )
        best_score = _total_score(*best_costs)
        score_history.append([N_GENERATIONS + 1, best_score])

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
