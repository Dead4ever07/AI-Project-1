import numpy as np
from algorithms.cost import (
    build_daily_occupancy,
    cost_function,
    delta_cost,
    total_accounting_cost,
)

POPULATION_SIZE = 20
N_GENERATIONS = 500
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.80
MUTATION_RATE = 0.04
ELITE_COUNT = 2
PLATEAU_LIMIT = 40
LOCAL_SEARCH_EVERY = 5 


def _score_from_occupancy(
    assignment,
    daily_occupancy,
    family_size,
    choices,
    n_days,
    min_occupancy,
    max_occupancy,
):
    """
    Score an individual given its already-computed daily_occupancy.
    Returns float('inf') if any occupancy constraint is violated.
    O(n_families) preference penalty + O(n_days) accounting cost.
    """
    occ = daily_occupancy[1:]
    if np.any(occ > max_occupancy) or np.any(occ < min_occupancy):
        return float("inf")

    PENALTIES = [
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
    penalty = 0.0
    for fam_id, day in enumerate(assignment):
        n = family_size[fam_id]
        rank = 10
        for r in range(10):
            if choices[fam_id, r] == day:
                rank = r
                break
        flat, per_member = PENALTIES[rank]
        penalty += flat + per_member * n

    acc_cost = total_accounting_cost(daily_occupancy, n_days, min_occupancy)
    return penalty + acc_cost


# ================================
# POPULATION INITIALISATION
# ================================


def _build_individual(
    base_assignment,
    family_size,
    choices,
    n_days,
    min_occupancy,
    max_occupancy,
    rng,
    perturbation,
):
    """
    Build a valid individual in O(n_families) without any repair loop.
    Returns (assignment, daily_occupancy).
    """
    n_families = len(base_assignment)
    daily_occupancy = np.zeros(n_days + 1, dtype=np.int32)
    assignment = np.empty(n_families, dtype=base_assignment.dtype)

    for fam_id in rng.permutation(n_families):
        n = family_size[fam_id]
        pref = choices[fam_id].copy()
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
            best = pref[int(np.argmin([daily_occupancy[d] for d in pref]))]
            daily_occupancy[best] += n
            assignment[fam_id] = best

    # Fix under-occupied days cheaply
    for day in range(1, n_days + 1):
        attempts = 0
        while daily_occupancy[day] < min_occupancy and attempts < n_families:
            attempts += 1
            donor = int(np.argmax(daily_occupancy[1:]) + 1)
            if donor == day:
                break
            for fam_id in range(n_families):
                if assignment[fam_id] == donor:
                    n = family_size[fam_id]
                    if daily_occupancy[donor] - n >= min_occupancy:
                        daily_occupancy[donor] -= n
                        daily_occupancy[day] += n
                        assignment[fam_id] = day
                        break
            else:
                break

    return assignment, daily_occupancy


def _init_population(
    base_assignment, family_size, choices, n_days, min_occupancy, max_occupancy, rng
):
    population = []
    occupancies = []
    fitness = []

    occ0 = build_daily_occupancy(base_assignment, family_size, n_days)
    score0 = _score_from_occupancy(
        base_assignment,
        occ0,
        family_size,
        choices,
        n_days,
        min_occupancy,
        max_occupancy,
    )
    population.append(base_assignment.copy())
    occupancies.append(occ0)
    fitness.append(score0)

    for i in range(1, POPULATION_SIZE):
        perturbation = 0.15 + 0.70 * (i / POPULATION_SIZE)
        ind, occ = _build_individual(
            base_assignment,
            family_size,
            choices,
            n_days,
            min_occupancy,
            max_occupancy,
            rng,
            perturbation,
        )
        score = _score_from_occupancy(
            ind, occ, family_size, choices, n_days, min_occupancy, max_occupancy
        )
        population.append(ind)
        occupancies.append(occ)
        fitness.append(score)

    return population, occupancies, fitness


# ================================
# SELECTION
# ================================


def _tournament_select(population, occupancies, fitness, rng):
    idx = rng.integers(0, len(population), size=TOURNAMENT_SIZE)
    best_i = idx[int(np.argmin([fitness[i] for i in idx]))]
    return population[best_i].copy(), occupancies[best_i].copy()


# ================================
# CROSSOVER — occupancy-aware, no repair needed
# ================================


def _crossover(
    parent_a,
    occ_a,
    parent_b,
    occ_b,
    family_size,
    choices,
    n_days,
    min_occupancy,
    max_occupancy,
    rng,
):
    n_families = len(parent_a)
    daily_occupancy = np.zeros(n_days + 1, dtype=np.int32)
    child = np.empty(n_families, dtype=parent_a.dtype)
    prefer_a = rng.integers(0, 2, size=n_families).astype(bool)

    for fam_id in rng.permutation(n_families):
        n = family_size[fam_id]
        first = parent_a[fam_id] if prefer_a[fam_id] else parent_b[fam_id]
        second = parent_b[fam_id] if prefer_a[fam_id] else parent_a[fam_id]

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

    return child, daily_occupancy


# ================================
# MUTATION — inline occupancy update, no rebuild
# ================================


def _mutate(
    individual,
    daily_occupancy,
    family_size,
    choices,
    n_days,
    min_occupancy,
    max_occupancy,
    rng,
):
    mutant = individual.copy()
    occ = daily_occupancy.copy()

    for fam_id in range(len(mutant)):
        if rng.random() >= MUTATION_RATE:
            continue
        n = family_size[fam_id]
        old_day = mutant[fam_id]
        new_day = choices[fam_id, rng.integers(0, 10)]
        if new_day == old_day:
            continue
        if occ[old_day] - n >= min_occupancy and occ[new_day] + n <= max_occupancy:
            occ[old_day] -= n
            occ[new_day] += n
            mutant[fam_id] = new_day

    return mutant, occ


# ================================
# LOCAL SEARCH — one greedy pass, called sparingly
# ================================


def _local_search_pass(
    assignment, family_size, choices, n_days, min_occupancy, max_occupancy, rng
):
    result = cost_function(
        assignment, family_size, choices, n_days, min_occupancy, max_occupancy
    )
    if result is None:
        return assignment

    preference_penalty, accounting_cost = result
    daily_occupancy = build_daily_occupancy(assignment, family_size, n_days)

    for fam_id in rng.permutation(len(assignment)):
        old_day = assignment[fam_id]
        for rank in range(10):
            new_day = choices[fam_id, rank]
            if new_day == old_day:
                continue
            res = delta_cost(
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
            if res is None:
                continue
            pd, ad = res
            if pd + ad < 0:
                daily_occupancy[old_day] -= family_size[fam_id]
                daily_occupancy[new_day] += family_size[fam_id]
                assignment[fam_id] = new_day
                preference_penalty += pd
                accounting_cost += ad
                old_day = new_day
                break

    return assignment


# ================================
# MAIN ENTRY POINT
# ================================


def run(submission, family_size, choices, n_days, min_occupancy, max_occupancy):
    base_assignment = submission["assigned_day"].values.copy()

    result = cost_function(
        base_assignment, family_size, choices, n_days, min_occupancy, max_occupancy
    )
    if result is None:
        raise ValueError("Starting solution is invalid!")
    pref_pen, acc_cost = result
    print(
        f"Initial score: {pref_pen + acc_cost:.2f}  "
        f"(preference: {pref_pen:.2f}  accounting: {acc_cost:.2f})"
    )

    rng = np.random.default_rng()

    print(f"Initialising population of {POPULATION_SIZE}...")
    population, occupancies, fitness = _init_population(
        base_assignment, family_size, choices, n_days, min_occupancy, max_occupancy, rng
    )

    best_idx = int(np.argmin(fitness))
    best_score = fitness[best_idx]
    best_assignment = population[best_idx].copy()
    scores_trajectory = [[0, best_score]]
    print(f"Initial population best score: {best_score:.2f}")

    no_improve = 0

    for generation in range(1, N_GENERATIONS + 1):

        # Elitism
        elite_order = np.argsort(fitness)[:ELITE_COUNT]
        new_pop = [population[i].copy() for i in elite_order]
        new_occ = [occupancies[i].copy() for i in elite_order]
        new_fit = [fitness[i] for i in elite_order]

        while len(new_pop) < POPULATION_SIZE:
            if rng.random() < CROSSOVER_RATE:
                pa, oa = _tournament_select(population, occupancies, fitness, rng)
                pb, ob = _tournament_select(population, occupancies, fitness, rng)
                child, child_occ = _crossover(
                    pa,
                    oa,
                    pb,
                    ob,
                    family_size,
                    choices,
                    n_days,
                    min_occupancy,
                    max_occupancy,
                    rng,
                )
            else:
                child, child_occ = _tournament_select(
                    population, occupancies, fitness, rng
                )

            child, child_occ = _mutate(
                child,
                child_occ,
                family_size,
                choices,
                n_days,
                min_occupancy,
                max_occupancy,
                rng,
            )
            score = _score_from_occupancy(
                child,
                child_occ,
                family_size,
                choices,
                n_days,
                min_occupancy,
                max_occupancy,
            )
            new_pop.append(child)
            new_occ.append(child_occ)
            new_fit.append(score)

        population = new_pop
        occupancies = new_occ
        fitness = new_fit

        # Periodic single-pass local search on best
        if generation % LOCAL_SEARCH_EVERY == 0:
            bi = int(np.argmin(fitness))
            polished = _local_search_pass(
                population[bi].copy(),
                family_size,
                choices,
                n_days,
                min_occupancy,
                max_occupancy,
                rng,
            )
            pol_occ = build_daily_occupancy(polished, family_size, n_days)
            pol_score = _score_from_occupancy(
                polished,
                pol_occ,
                family_size,
                choices,
                n_days,
                min_occupancy,
                max_occupancy,
            )
            if pol_score < fitness[bi]:
                population[bi] = polished
                occupancies[bi] = pol_occ
                fitness[bi] = pol_score

        gen_best_i = int(np.argmin(fitness))
        gen_best_score = fitness[gen_best_i]
        valid_scores = [f for f in fitness if f < float("inf")]
        avg_score = float(np.mean(valid_scores)) if valid_scores else float("inf")
        valid_count = len(valid_scores)

        if gen_best_score < best_score:
            best_score = gen_best_score
            best_assignment = population[gen_best_i].copy()
            scores_trajectory.append([generation, best_score])
            no_improve = 0
            marker = "  *** NEW BEST ***"
        else:
            no_improve += 1
            marker = ""

        print(
            f"Gen {generation:4d}: best={gen_best_score:.2f}  avg={avg_score:.2f}  "
            f"valid={valid_count}/{POPULATION_SIZE}  global_best={best_score:.2f}  "
            f"plateau={no_improve}/{PLATEAU_LIMIT}{marker}"
        )

        if no_improve >= PLATEAU_LIMIT:
            print(f"Stopping: no improvement for {PLATEAU_LIMIT} generations.")
            break

    print("\nRunning final local search polish...")
    best_assignment = _local_search_pass(
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
