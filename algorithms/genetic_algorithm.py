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


def _repair(
    assignment, family_size, choices, n_days, min_occupancy, max_occupancy, rng
):
    """
    Greedily repair an assignment that violates occupancy constraints.

    Strategy: iterate over days that are over/under capacity and move families
    to their next-best available choice until all days are within bounds.
    Falls back to a random valid choice when no preference slot helps.
    """
    daily_occupancy = build_daily_occupancy(assignment, family_size, n_days)
    n_families = len(assignment)

    for _ in range(n_families * 2):  # bounded repair attempts
        # Find first violating day
        violated = -1
        for day in range(1, n_days + 1):
            if (
                daily_occupancy[day] > max_occupancy
                or daily_occupancy[day] < min_occupancy
            ):
                violated = day
                break
        if violated == -1:
            break  # all days valid

        over = daily_occupancy[violated] > max_occupancy

        # Collect families on that day
        candidates = [f for f in range(n_families) if assignment[f] == violated]
        if not candidates:
            break

        # Pick the family whose move costs least in preference
        rng.shuffle(candidates)
        moved = False
        for fam_id in candidates:
            for rank in range(10):
                candidate_day = choices[fam_id, rank]
                if candidate_day == violated:
                    continue
                new_occ = daily_occupancy[candidate_day] + family_size[fam_id]
                old_occ = daily_occupancy[violated] - family_size[fam_id]
                if new_occ <= max_occupancy and old_occ >= min_occupancy:
                    daily_occupancy[violated] -= family_size[fam_id]
                    daily_occupancy[candidate_day] += family_size[fam_id]
                    assignment[fam_id] = candidate_day
                    moved = True
                    break
            if moved:
                break

        if not moved:
            # Force-move a random family to a random valid day
            fam_id = candidates[0]
            for day in rng.permutation(np.arange(1, n_days + 1)):
                new_occ = daily_occupancy[day] + family_size[fam_id]
                old_occ = daily_occupancy[violated] - family_size[fam_id]
                if new_occ <= max_occupancy and old_occ >= min_occupancy:
                    daily_occupancy[violated] -= family_size[fam_id]
                    daily_occupancy[day] += family_size[fam_id]
                    assignment[fam_id] = day
                    break

    return assignment


# ================================
# INITIALISATION
# ================================


def _init_population(
    base_assignment, family_size, choices, n_days, min_occupancy, max_occupancy, rng
):
    """
    Seed the population:
      - Individual 0  : the provided starting solution (unchanged).
      - Individuals 1+: perturbed copies — each family is randomly reassigned
                        to one of its top-10 choices, then repaired.
    """
    population = []
    n_families = len(base_assignment)

    for i in range(POPULATION_SIZE):
        if i == 0:
            ind = base_assignment.copy()
        else:
            ind = base_assignment.copy()
            # Randomly reassign a fraction of families to a random preference
            mask = rng.random(n_families) < 0.4
            for fam_id in np.where(mask)[0]:
                ind[fam_id] = choices[fam_id, rng.integers(0, 10)]
            ind = _repair(
                ind, family_size, choices, n_days, min_occupancy, max_occupancy, rng
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
    Uniform crossover at the family level:
      - Each family independently inherits its day from parent A or B (50/50).
      - The offspring is then repaired if it violates occupancy bounds.
    """
    mask = rng.integers(0, 2, size=len(parent_a)).astype(bool)
    child = np.where(mask, parent_a, parent_b)
    child = _repair(
        child, family_size, choices, n_days, min_occupancy, max_occupancy, rng
    )
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
