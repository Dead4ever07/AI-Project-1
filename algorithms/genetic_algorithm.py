import numpy as np
from algorithms.cost import (
    build_daily_occupancy,
    cost_function,
    delta_cost,
    total_accounting_cost,
)

POPULATION_SIZE = 20  # number of candidate solutions kept at once
N_GENERATIONS = 500  # max evolution cycles
TOURNAMENT_SIZE = 3  # candidates compared in each selection draw
CROSSOVER_RATE = 0.80  # chance two parents breed (vs. cloning one parent)
MUTATION_RATE = 0.04  # per-family chance of a random day reassignment
ELITE_COUNT = 2  # top solutions copied unchanged to the next generation
PLATEAU_LIMIT = 40  # stop early if best score hasn't improved this many generations
LOCAL_SEARCH_EVERY = 5  # how often (in generations) to polish the best individual

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


def _score(assignment, daily_occupancy, family_size, choices, n_days, min_occ, max_occ):
    """Total cost of one assignment. Returns inf if any day breaks occupancy bounds."""
    if np.any(daily_occupancy[1:] > max_occ) or np.any(daily_occupancy[1:] < min_occ):
        return float("inf")

    preference_cost = 0.0
    for fam, day in enumerate(assignment):
        rank = next((r for r in range(10) if choices[fam, r] == day), 10)
        flat, per_member = PREFERENCE_PENALTIES[rank]
        preference_cost += flat + per_member * family_size[fam]

    return preference_cost + total_accounting_cost(daily_occupancy, n_days, min_occ)


def _build_individual(
    base, family_size, choices, n_days, min_occ, max_occ, rng, shuffle_prob
):
    """
    Build one valid assignment by placing families one at a time (random order).
    Each family picks their most-preferred day that still has room.
    `shuffle_prob` controls how much we deviate from the original preferences,
    which keeps the population diverse.
    """
    n_families = len(base)
    occ = np.zeros(n_days + 1, dtype=np.int32)
    assignment = np.empty(n_families, dtype=base.dtype)

    for fam in rng.permutation(n_families):
        n = family_size[fam]
        prefs = choices[fam].copy()
        if rng.random() < shuffle_prob:
            rng.shuffle(prefs)

        # Try each preferred day in order; fall back to least-crowded if all are full
        for day in prefs:
            if occ[day] + n <= max_occ:
                break
        else:
            day = prefs[np.argmin([occ[d] for d in prefs])]

        occ[day] += n
        assignment[fam] = day

    # Repair: move families from the busiest day into any under-occupied day
    for day in range(1, n_days + 1):
        for _ in range(n_families):
            if occ[day] >= min_occ:
                break
            donor = int(np.argmax(occ[1:]) + 1)
            if donor == day:
                break
            for fam in range(n_families):
                if assignment[fam] == donor:
                    n = family_size[fam]
                    if occ[donor] - n >= min_occ:
                        occ[donor] -= n
                        occ[day] += n
                        assignment[fam] = day
                        break
            else:
                break

    return assignment, occ


def _init_population(base, family_size, choices, n_days, min_occ, max_occ, rng):
    """
    Build the starting population. The first individual is the original submission.
    The rest are variants with increasing randomness (15% to 85% shuffle probability)
    to ensure we start with a diverse range of solutions.
    """
    pop, occs, scores = [], [], []

    for i in range(POPULATION_SIZE):
        if i == 0:
            ind, occ = base.copy(), build_daily_occupancy(base, family_size, n_days)
        else:
            shuffle_prob = 0.15 + 0.70 * (i / POPULATION_SIZE)
            ind, occ = _build_individual(
                base, family_size, choices, n_days, min_occ, max_occ, rng, shuffle_prob
            )

        pop.append(ind)
        occs.append(occ)
        scores.append(_score(ind, occ, family_size, choices, n_days, min_occ, max_occ))

    return pop, occs, scores


def _tournament_select(pop, occs, scores, rng):
    """Pick the best individual from a random sample (tournament selection)."""
    candidates = rng.integers(0, len(pop), size=TOURNAMENT_SIZE)
    winner = candidates[np.argmin([scores[i] for i in candidates])]
    return pop[winner].copy(), occs[winner].copy()


def _crossover(pa, oa, pb, ob, family_size, choices, n_days, min_occ, max_occ, rng):
    """
    Combine two parents into a child. For each family, randomly prefer one parent's
    day; if it's full, try the other; if still full, fall back to the family's own
    preferences. Families are placed in random order to avoid systematic bias.
    """
    n_families = len(pa)
    occ = np.zeros(n_days + 1, dtype=np.int32)
    child = np.empty(n_families, dtype=pa.dtype)
    prefer_a = rng.integers(0, 2, size=n_families).astype(bool)

    for fam in rng.permutation(n_families):
        n = family_size[fam]
        days_to_try = (
            [pa[fam], pb[fam]] if prefer_a[fam] else [pb[fam], pa[fam]]
        ) + list(choices[fam])

        for day in days_to_try:
            if occ[day] + n <= max_occ:
                break
        else:
            day = choices[fam, np.argmin([occ[d] for d in choices[fam]])]

        occ[day] += n
        child[fam] = day

    return child, occ


def _mutate(individual, occ, family_size, choices, n_days, min_occ, max_occ, rng):
    """
    Randomly reassign a small fraction of families to a different preferred day.
    A move only happens if it keeps both days within occupancy bounds.
    Mutation prevents the population from getting stuck in a local optimum.
    """
    mutant, occ = individual.copy(), occ.copy()

    for fam in range(len(mutant)):
        if rng.random() >= MUTATION_RATE:
            continue
        n = family_size[fam]
        old_day, new_day = mutant[fam], choices[fam, rng.integers(0, 10)]
        if (
            new_day != old_day
            and occ[old_day] - n >= min_occ
            and occ[new_day] + n <= max_occ
        ):
            occ[old_day] -= n
            occ[new_day] += n
            mutant[fam] = new_day

    return mutant, occ


def _local_search(assignment, family_size, choices, n_days, min_occ, max_occ, rng):
    """
    Greedily improve an assignment by trying to move each family to a better day.
    Called sparingly (every LOCAL_SEARCH_EVERY generations) as a fine-tuning step.
    """
    result = cost_function(assignment, family_size, choices, n_days, min_occ, max_occ)
    if result is None:
        return assignment

    pref_cost, acc_cost = result
    occ = build_daily_occupancy(assignment, family_size, n_days)

    for fam in rng.permutation(len(assignment)):
        old_day = assignment[fam]
        for rank in range(10):
            new_day = choices[fam, rank]
            if new_day == old_day:
                continue
            delta = delta_cost(
                fam,
                old_day,
                new_day,
                acc_cost,
                occ,
                family_size,
                choices,
                n_days,
                min_occ,
                max_occ,
            )
            if delta is not None and sum(delta) < 0:
                occ[old_day] -= family_size[fam]
                occ[new_day] += family_size[fam]
                assignment[fam] = new_day
                pref_cost += delta[0]
                acc_cost += delta[1]
                old_day = new_day
                break

    return assignment


def run(submission, family_size, choices, n_days, min_occupancy, max_occupancy):
    """
    Run the genetic algorithm. Returns (best_assignment, best_score, score_history).

    Flow:
      1. Seed the population from the given submission.
      2. Each generation: keep the elite, breed the rest, occasionally polish the best.
      3. Stop early if no improvement for PLATEAU_LIMIT generations.
      4. Run a final local search on the best solution found.
    """
    min_occ, max_occ = min_occupancy, max_occupancy
    base = submission["assigned_day"].values.copy()

    result = cost_function(base, family_size, choices, n_days, min_occ, max_occ)
    if result is None:
        raise ValueError("Starting solution is invalid!")
    print(
        f"Initial score: {sum(result):.2f}  (preference: {result[0]:.2f}  accounting: {result[1]:.2f})"
    )

    rng = np.random.default_rng()
    pop, occs, scores = _init_population(
        base, family_size, choices, n_days, min_occ, max_occ, rng
    )

    best_score = min(scores)
    best_assignment = pop[np.argmin(scores)].copy()
    score_history = [[0, best_score]]
    no_improve = 0
    print(f"Initial population best: {best_score:.2f}")

    for gen in range(1, N_GENERATIONS + 1):

        # Always carry the top ELITE_COUNT individuals into the next generation
        elite = np.argsort(scores)[:ELITE_COUNT]
        new_pop = [pop[i].copy() for i in elite]
        new_occs = [occs[i].copy() for i in elite]
        new_scores = [scores[i] for i in elite]

        # Fill the rest of the population with bred/mutated children
        while len(new_pop) < POPULATION_SIZE:
            if rng.random() < CROSSOVER_RATE:
                pa, oa = _tournament_select(pop, occs, scores, rng)
                pb, ob = _tournament_select(pop, occs, scores, rng)
                child, child_occ = _crossover(
                    pa, oa, pb, ob, family_size, choices, n_days, min_occ, max_occ, rng
                )
            else:
                child, child_occ = _tournament_select(pop, occs, scores, rng)

            child, child_occ = _mutate(
                child, child_occ, family_size, choices, n_days, min_occ, max_occ, rng
            )
            new_pop.append(child)
            new_occs.append(child_occ)
            new_scores.append(
                _score(child, child_occ, family_size, choices, n_days, min_occ, max_occ)
            )

        pop, occs, scores = new_pop, new_occs, new_scores

        # Periodically polish the best individual with a greedy local search
        if gen % LOCAL_SEARCH_EVERY == 0:
            bi = int(np.argmin(scores))
            polished = _local_search(
                pop[bi].copy(), family_size, choices, n_days, min_occ, max_occ, rng
            )
            pol_occ = build_daily_occupancy(polished, family_size, n_days)
            pol_score = _score(
                polished, pol_occ, family_size, choices, n_days, min_occ, max_occ
            )
            if pol_score < scores[bi]:
                pop[bi], occs[bi], scores[bi] = polished, pol_occ, pol_score

        # Track progress and check for plateau
        gen_best = min(scores)
        valid = [s for s in scores if s < float("inf")]
        print(
            f"Gen {gen:4d}: best={gen_best:.2f}  avg={np.mean(valid):.2f}  "
            f"valid={len(valid)}/{POPULATION_SIZE}  global_best={best_score:.2f}  "
            f"plateau={no_improve}/{PLATEAU_LIMIT}"
            + ("  *** NEW BEST ***" if gen_best < best_score else "")
        )

        if gen_best < best_score:
            best_score = gen_best
            best_assignment = pop[int(np.argmin(scores))].copy()
            score_history.append([gen, best_score])
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PLATEAU_LIMIT:
            print(f"Stopping early: no improvement for {PLATEAU_LIMIT} generations.")
            break

    # Final polish on the overall best solution
    print("\nRunning final local search polish...")
    best_assignment = _local_search(
        best_assignment, family_size, choices, n_days, min_occ, max_occ, rng
    )
    result = cost_function(
        best_assignment, family_size, choices, n_days, min_occ, max_occ
    )
    if result is not None:
        best_score = sum(result)
        score_history.append([N_GENERATIONS + 1, best_score])

    print(f"\nBest score found: {best_score:.2f}")
    return best_assignment, best_score, score_history
