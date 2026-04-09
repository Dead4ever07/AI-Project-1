import numpy as np
from algorithms.cost import build_daily_occupancy, cost_function, delta_cost

# ================================
# HYPERPARAMETERS
# ================================
POPULATION_SIZE = 30  # Number of individuals in the population
N_GENERATIONS = 200  # Maximum number of generations
TOURNAMENT_SIZE = 4  # Candidates per tournament selection
CROSSOVER_RATE = 0.85  # Probability of performing crossover vs. cloning
MUTATION_RATE = 0.03  # Per-family probability of mutation
ELITE_COUNT = 2  # Number of top individuals preserved each generation
PLATEAU_LIMIT = 30  # Stop if no improvement for this many generations
LOCAL_SEARCH_INTERVAL = 10  # Run a short hill-climb on the best individual every N gens
LOCAL_SEARCH_STEPS = 50  # Number of hill-climb passes per local search call


# ================================
# HELPERS
# ================================


def _score_individual(
    assignment, family_size, choices, n_days, min_occupancy, max_occupancy
):
    """Return total cost for a valid individual, or infinity if invalid."""
    result = cost_function(
        assignment, family_size, choices, n_days, min_occupancy, max_occupancy
    )
    if result is None:
        return float("inf")
    return result[0] + result[1]


def _build_valid_individual(
    base_assignment,
    family_size,
    choices,
    n_days,
    min_occupancy,
    max_occupancy,
    rng,
    perturbation=0.4,
):
    """
    Build a valid individual in O(n_families) without any repair loop.

    For each family (visited in random order):
      1. With probability `perturbation`, shuffle its preference list so we
         explore different choices across the population.
      2. Try each preference in order; commit to the first one that keeps
         the target day within [min_occupancy, max_occupancy].
      3. If no preference fits, fall back to the base assignment's day
         (which is already valid since the base solution is valid, and we
         process families one at a time so earlier commitments hold).

    Because we only ever ADD a family to a day when capacity allows, the
    resulting assignment is guaranteed valid — no repair needed.
    """
    daily_occupancy = np.zeros(n_days + 1, dtype=np.int32)
    assignment = np.empty(len(base_assignment), dtype=base_assignment.dtype)
    n_families = len(base_assignment)

    for fam_id in rng.permutation(n_families):
        n = family_size[fam_id]
        pref = choices[fam_id].copy()  # shape (10,)

        if rng.random() < perturbation:
            rng.shuffle(pref)

        placed = False
        for day in pref:
            if daily_occupancy[day] + n <= max_occupancy:
                daily_occupancy[day] += n
                assignment[fam_id] = day
                placed = True
                break

        if not placed:
            # Fall back: pick the least-occupied day among preferences
            best_day = pref[int(np.argmin([daily_occupancy[d] for d in pref]))]
            daily_occupancy[best_day] += n
            assignment[fam_id] = best_day

    # Fast min-occupancy check — violations are uncommon; fix them cheaply
    for day in range(1, n_days + 1):
        while daily_occupancy[day] < min_occupancy:
            # Pull the largest available family from the most crowded day
            donor_day = int(np.argmax(daily_occupancy[1:]) + 1)
            if donor_day == day:
                break
            # Find a family on donor_day
            for fam_id in range(n_families):
                if assignment[fam_id] == donor_day:
                    n = family_size[fam_id]
                    if daily_occupancy[donor_day] - n >= min_occupancy:
                        daily_occupancy[donor_day] -= n
                        daily_occupancy[day] += n
                        assignment[fam_id] = day
                        break
            else:
                break  # couldn't fix — cost_function will catch it

    return assignment


# ================================
# INITIALISATION
# ================================


def _init_population(
    base_assignment, family_size, choices, n_days, min_occupancy, max_occupancy, rng
):
    """
    Seed the population in O(POPULATION_SIZE * n_families):
      - Individual 0  : the provided starting solution (unchanged).
      - Individuals 1+: fresh valid assignments built directly without repair,
                        with varying perturbation levels for diversity.
    """
    population = [base_assignment.copy()]

    for i in range(1, POPULATION_SIZE):
        # Gradually increase perturbation so the population spans a wide range
        perturbation = 0.2 + 0.6 * (i / POPULATION_SIZE)
        ind = _build_valid_individual(
            base_assignment,
            family_size,
            choices,
            n_days,
            min_occupancy,
            max_occupancy,
            rng,
            perturbation=perturbation,
        )
        population.append(ind)

    return population


# ================================
# SELECTION
# ================================


def _tournament_select(population, fitness, rng):
    """Return one individual chosen by tournament selection."""
    idx = rng.integers(0, len(population), size=TOURNAMENT_SIZE)
    best_idx = idx[np.argmin([fitness[i] for i in idx])]
    return population[best_idx].copy()


# ================================
# CROSSOVER
# ================================


def _crossover(
    parent_a, parent_b, family_size, choices, n_days, min_occupancy, max_occupancy, rng
):
    """
    Occupancy-aware uniform crossover — no repair needed.

    Families are visited in random order. Each one inherits its preferred
    parent's day if that day still has capacity; otherwise it tries the
    other parent's day; otherwise it falls back through ranked preferences.
    The child is valid by construction in O(n_families).
    """
    n_families = len(parent_a)
    daily_occupancy = np.zeros(n_days + 1, dtype=np.int32)
    child = np.empty(n_families, dtype=parent_a.dtype)
    mask = rng.integers(0, 2, size=n_families).astype(bool)

    for fam_id in rng.permutation(n_families):
        n = family_size[fam_id]
        first = parent_a[fam_id] if mask[fam_id] else parent_b[fam_id]
        second = parent_b[fam_id] if mask[fam_id] else parent_a[fam_id]

        placed = False
        for day in (first, second):
            if daily_occupancy[day] + n <= max_occupancy:
                daily_occupancy[day] += n
                child[fam_id] = day
                placed = True
                break

        if not placed:
            for day in choices[fam_id]:
                if daily_occupancy[day] + n <= max_occupancy:
                    daily_occupancy[day] += n
                    child[fam_id] = day
                    placed = True
                    break

        if not placed:
            best = choices[
                fam_id, int(np.argmin([daily_occupancy[d] for d in choices[fam_id]]))
            ]
            daily_occupancy[best] += n
            child[fam_id] = best

    return child


# ================================
# MUTATION
# ================================


def _mutate(
    individual, family_size, choices, n_days, min_occupancy, max_occupancy, rng
):
    """
    Per-family mutation: reassign the family to a randomly chosen preference day.
    The individual is repaired afterwards only if a violation is introduced.
    """
    mutant = individual.copy()
    daily_occupancy = build_daily_occupancy(mutant, family_size, n_days)
    n_families = len(mutant)

    for fam_id in range(n_families):
        if rng.random() >= MUTATION_RATE:
            continue

        old_day = mutant[fam_id]
        new_day = choices[fam_id, rng.integers(0, 10)]
        if new_day == old_day:
            continue

        # Accept only if both days remain within bounds after the swap
        new_old_occ = daily_occupancy[old_day] - family_size[fam_id]
        new_new_occ = daily_occupancy[new_day] + family_size[fam_id]
        if new_old_occ >= min_occupancy and new_new_occ <= max_occupancy:
            daily_occupancy[old_day] = new_old_occ
            daily_occupancy[new_day] = new_new_occ
            mutant[fam_id] = new_day

    return mutant


# ================================
# LOCAL SEARCH (greedy hill-climb polish)
# ================================


def _local_search(
    assignment, family_size, choices, n_days, min_occupancy, max_occupancy, rng
):
    """
    A short greedy hill-climb: iterate over families in random order and
    accept any move to a preference day that strictly reduces total cost.
    Runs LOCAL_SEARCH_STEPS passes over the population.
    """
    result = cost_function(
        assignment, family_size, choices, n_days, min_occupancy, max_occupancy
    )
    if result is None:
        return assignment
    preference_penalty, accounting_cost = result
    daily_occupancy = build_daily_occupancy(assignment, family_size, n_days)
    n_families = len(assignment)

    for _ in range(LOCAL_SEARCH_STEPS):
        for fam_id in rng.permutation(n_families):
            old_day = assignment[fam_id]
            for rank in range(10):
                new_day = choices[fam_id, rank]
                if new_day == old_day:
                    continue
                result = delta_cost(
                    fam_id,
                    old_day,
                    new_day,
                    accounting_cost,
                    daily_occupancy,
                    family_size,
                    choices,
                    n_days,
                    min_occupancy,
                    max_occupancy,
                )
                if result is None:
                    continue
                pref_delta, acc_delta = result
                if pref_delta + acc_delta < 0:
                    daily_occupancy[old_day] -= family_size[fam_id]
                    daily_occupancy[new_day] += family_size[fam_id]
                    assignment[fam_id] = new_day
                    preference_penalty += pref_delta
                    accounting_cost += acc_delta
                    old_day = new_day
                    break  # move on to next family after first improvement

    return assignment


# ================================
# MAIN ENTRY POINT
# ================================


def run(submission, family_size, choices, n_days, min_occupancy, max_occupancy):
    base_assignment = submission["assigned_day"].values.copy()

    # Validate starting solution
    result = cost_function(
        base_assignment, family_size, choices, n_days, min_occupancy, max_occupancy
    )
    if result is None:
        raise ValueError("Starting solution is invalid!")
    preference_penalty, accounting_cost = result
    initial_score = preference_penalty + accounting_cost
    print(
        f"Initial score: {initial_score:.2f}  "
        f"(preference: {preference_penalty:.2f}  accounting: {accounting_cost:.2f})"
    )

    rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Initialise population
    # ------------------------------------------------------------------
    print(f"\nInitialising population of {POPULATION_SIZE} individuals...")
    population = _init_population(
        base_assignment, family_size, choices, n_days, min_occupancy, max_occupancy, rng
    )
    fitness = [
        _score_individual(
            ind, family_size, choices, n_days, min_occupancy, max_occupancy
        )
        for ind in population
    ]

    best_idx = int(np.argmin(fitness))
    best_score = fitness[best_idx]
    best_assignment = population[best_idx].copy()
    scores_trajectory = [[0, best_score]]

    print(f"Initial population best score: {best_score:.2f}")

    # ------------------------------------------------------------------
    # Evolution loop
    # ------------------------------------------------------------------
    generations_without_improvement = 0

    for generation in range(1, N_GENERATIONS + 1):

        # --- Elitism: carry top ELITE_COUNT individuals unchanged ----------
        elite_indices = np.argsort(fitness)[:ELITE_COUNT]
        new_population = [population[i].copy() for i in elite_indices]

        # --- Fill the rest of the new population --------------------------
        while len(new_population) < POPULATION_SIZE:
            if rng.random() < CROSSOVER_RATE:
                parent_a = _tournament_select(population, fitness, rng)
                parent_b = _tournament_select(population, fitness, rng)
                child = _crossover(
                    parent_a,
                    parent_b,
                    family_size,
                    choices,
                    n_days,
                    min_occupancy,
                    max_occupancy,
                    rng,
                )
            else:
                child = _tournament_select(population, fitness, rng)

            child = _mutate(
                child, family_size, choices, n_days, min_occupancy, max_occupancy, rng
            )
            new_population.append(child)

        population = new_population
        fitness = [
            _score_individual(
                ind, family_size, choices, n_days, min_occupancy, max_occupancy
            )
            for ind in population
        ]

        # --- Periodic local search on the current best --------------------
        if generation % LOCAL_SEARCH_INTERVAL == 0:
            best_local_idx = int(np.argmin(fitness))
            polished = _local_search(
                population[best_local_idx].copy(),
                family_size,
                choices,
                n_days,
                min_occupancy,
                max_occupancy,
                rng,
            )
            polished_score = _score_individual(
                polished, family_size, choices, n_days, min_occupancy, max_occupancy
            )
            if polished_score < fitness[best_local_idx]:
                population[best_local_idx] = polished
                fitness[best_local_idx] = polished_score

        # --- Track best ---------------------------------------------------
        gen_best_idx = int(np.argmin(fitness))
        gen_best_score = fitness[gen_best_idx]

        if gen_best_score < best_score:
            best_score = gen_best_score
            best_assignment = population[gen_best_idx].copy()
            scores_trajectory.append([generation, best_score])
            generations_without_improvement = 0
            improvement_marker = "  *** NEW BEST ***"
        else:
            generations_without_improvement += 1
            improvement_marker = ""

        avg_score = float(np.mean([f for f in fitness if f < float("inf")]))
        print(
            f"Gen {generation:4d}: best={gen_best_score:.2f}  "
            f"avg={avg_score:.2f}  "
            f"global_best={best_score:.2f}  "
            f"plateau={generations_without_improvement}/{PLATEAU_LIMIT}"
            f"{improvement_marker}"
        )

        if generations_without_improvement >= PLATEAU_LIMIT:
            print(f"\nStopping: no improvement for {PLATEAU_LIMIT} generations.")
            break

    # ------------------------------------------------------------------
    # Final local search polish on the best solution found
    # ------------------------------------------------------------------
    print("\nRunning final local search polish on best solution...")
    best_assignment = _local_search(
        best_assignment, family_size, choices, n_days, min_occupancy, max_occupancy, rng
    )
    result = cost_function(
        best_assignment, family_size, choices, n_days, min_occupancy, max_occupancy
    )
    if result is not None:
        best_score = result[0] + result[1]
        scores_trajectory.append([N_GENERATIONS + 1, best_score])

    print(f"\nBest score found: {best_score:.2f}")
    return best_assignment, best_score, scores_trajectory
