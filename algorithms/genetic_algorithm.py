import numpy as np
from algorithms.cost import (
    build_daily_occupancy,
    cost_function,
    delta_cost,
    total_accounting_cost,
    _preference_penalty,
)

POPULATION_SIZE = 20  # number of candidate solutions kept at once
N_GENERATIONS = 500  # max evolution cycles
TOURNAMENT_SIZE = 3  # candidates compared in each selection draw
CROSSOVER_RATE = 0.80  # chance two parents breed (vs. cloning one parent)
MUTATION_RATE = 0.04  # per-family chance of a random day reassignment
ELITE_COUNT = 2  # top solutions copied unchanged to the next generation
PLATEAU_LIMIT = 20  # stop early if best score hasn't improved this many generations
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


def _total_score(pref_cost, acc_cost):
    """Sum a (pref_cost, acc_cost) pair into a single score."""
    return pref_cost + acc_cost


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

    Each entry in costs is a (pref_cost, acc_cost) tuple; inf signals an invalid solution.
    """
    pop, occs, costs = [], [], []

    for i in range(POPULATION_SIZE):
        if i == 0:
            ind = base.copy()
            occ = build_daily_occupancy(base, family_size, n_days)
        else:
            shuffle_prob = 0.15 + 0.70 * (i / POPULATION_SIZE)
            ind, occ = _build_individual(
                base, family_size, choices, n_days, min_occ, max_occ, rng, shuffle_prob
            )

        result = cost_function(ind, family_size, choices, n_days, min_occ, max_occ)
        cost = result if result is not None else (float("inf"), float("inf"))

        pop.append(ind)
        occs.append(occ)
        costs.append(cost)

    return pop, occs, costs


def _tournament_select(pop, occs, costs, rng):
    """Pick the best individual from a random sample (tournament selection)."""
    candidates = rng.integers(0, len(pop), size=TOURNAMENT_SIZE)
    winner = candidates[np.argmin([_total_score(*costs[i]) for i in candidates])]
    return pop[winner].copy(), occs[winner].copy(), costs[winner]


def _crossover(
    pa, oa, ca, pb, ob, cb, family_size, choices, n_days, min_occ, max_occ, rng
):
    """
    Combine two parents into a child. For each family, randomly prefer one parent's
    day; if it's full, try the other; if still full, fall back to the family's own
    preferences. Families are placed in random order to avoid systematic bias.

    Preference cost is accumulated incrementally during construction.
    Accounting cost is computed once at the end (it depends on the full occ array).
    """
    n_families = len(pa)
    occ = np.zeros(n_days + 1, dtype=np.int32)
    child = np.empty(n_families, dtype=pa.dtype)
    prefer_a = rng.integers(0, 2, size=n_families).astype(bool)
    pref_cost = 0.0

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
        pref_cost += _preference_penalty(fam, day, family_size, choices)

    # Validate occupancy; if invalid, signal with inf costs
    if np.any(occ[1:] > max_occ) or np.any(occ[1:] < min_occ):
        return child, occ, (float("inf"), float("inf"))

    acc_cost = total_accounting_cost(occ, n_days, min_occ)
    return child, occ, (pref_cost, acc_cost)


def _mutate(individual, occ, cost, family_size, choices, n_days, min_occ, max_occ, rng):
    """
    Randomly reassign a small fraction of families to a different preferred day.
    Uses delta_cost to update (pref_cost, acc_cost) incrementally — no full recompute.
    A move only happens if it keeps both days within occupancy bounds.
    """
    mutant = individual.copy()
    occ = occ.copy()
    pref_cost, acc_cost = cost

    for fam in range(len(mutant)):
        if rng.random() >= MUTATION_RATE:
            continue
        old_day = mutant[fam]
        new_day = choices[fam, rng.integers(0, 10)]
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
        if delta is not None:
            pref_cost += delta[0]
            acc_cost += delta[1]
            occ[old_day] -= family_size[fam]
            occ[new_day] += family_size[fam]
            mutant[fam] = new_day

    return mutant, occ, (pref_cost, acc_cost)


def _local_search(
    assignment, occ, cost, family_size, choices, n_days, min_occ, max_occ, rng
):
    """
    Greedily improve an assignment by trying to move each family to a better day.
    Uses delta_cost throughout — no cost_function calls inside the loop.
    Called sparingly (every LOCAL_SEARCH_EVERY generations) as a fine-tuning step.
    """
    pref_cost, acc_cost = cost

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

    return assignment, occ, (pref_cost, acc_cost)


_callback = None
_stop = False

def emit(ev: dict):
    if _callback is not None:
        _callback(ev)

def run(submission, family_size, choices, n_days, min_occupancy, max_occupancy):
    """
    Run the genetic algorithm. Returns (best_assignment, best_score, score_history).

    Flow:
      1. Seed the population from the given submission.
      2. Each generation: keep the elite, breed the rest, occasionally polish the best.
      3. Stop early if no improvement for PLATEAU_LIMIT generations.
      4. Run a final local search on the best solution found.

    All cost tracking uses (pref_cost, acc_cost) tuples updated via delta_cost;
    cost_function is called only once per individual at initialisation time.
    """
    min_occ, max_occ = min_occupancy, max_occupancy
    base = submission["assigned_day"].values.copy()

    result = cost_function(base, family_size, choices, n_days, min_occ, max_occ)
    if result is None:
        raise ValueError("Starting solution is invalid!")
    
    print(
        f"Initial score: {sum(result):.2f}  (preference: {result[0]:.2f}  accounting: {result[1]:.2f})"
    )

    emit({"type": "start", "score": sum(result),
          "preference": result[0], "accounting": result[1]})

    rng = np.random.default_rng()
    pop, occs, costs = _init_population(
        base, family_size, choices, n_days, min_occ, max_occ, rng
    )

    scores = [_total_score(*c) for c in costs]
    best_idx = int(np.argmin(scores))
    best_score = scores[best_idx]
    best_assignment = pop[best_idx].copy()
    best_occ = occs[best_idx].copy()
    best_cost = costs[best_idx]
    score_history = [[0, best_score]]
    no_improve = 0
    print(f"Initial population best: {best_score:.2f}")

    for gen in range(1, N_GENERATIONS + 1):
        if _stop:
            print("Stopping: stop requested.")
            break

        # Always carry the top ELITE_COUNT individuals into the next generation
        elite = np.argsort(scores)[:ELITE_COUNT]
        new_pop = [pop[i].copy() for i in elite]
        new_occs = [occs[i].copy() for i in elite]
        new_costs = [costs[i] for i in elite]

        # Fill the rest of the population with bred/mutated children
        while len(new_pop) < POPULATION_SIZE:
            if _stop: break
            if rng.random() < CROSSOVER_RATE:
                pa, oa, ca = _tournament_select(pop, occs, costs, rng)
                pb, ob, cb = _tournament_select(pop, occs, costs, rng)
                child, child_occ, child_cost = _crossover(
                    pa,
                    oa,
                    ca,
                    pb,
                    ob,
                    cb,
                    family_size,
                    choices,
                    n_days,
                    min_occ,
                    max_occ,
                    rng,
                )
            else:
                child, child_occ, child_cost = _tournament_select(pop, occs, costs, rng)

            child, child_occ, child_cost = _mutate(
                child,
                child_occ,
                child_cost,
                family_size,
                choices,
                n_days,
                min_occ,
                max_occ,
                rng,
            )
            new_pop.append(child)
            new_occs.append(child_occ)
            new_costs.append(child_cost)

        if _stop: break

        pop, occs, costs = new_pop, new_occs, new_costs
        scores = [_total_score(*c) for c in costs]

        # Periodically polish the best individual with a greedy local search
        if gen % LOCAL_SEARCH_EVERY == 0:
            bi = int(np.argmin(scores))
            polished, pol_occ, pol_cost = _local_search(
                pop[bi].copy(),
                occs[bi].copy(),
                costs[bi],
                family_size,
                choices,
                n_days,
                min_occ,
                max_occ,
                rng,
            )
            pol_score = _total_score(*pol_cost)
            if pol_score < scores[bi]:
                pop[bi], occs[bi], costs[bi], scores[bi] = (
                    polished,
                    pol_occ,
                    pol_cost,
                    pol_score,
                )

        # Track progress and check for plateau
        gen_best_idx = int(np.argmin(scores))
        gen_best = scores[gen_best_idx]
        valid = [s for s in scores if s < float("inf")]
        print(
            f"Gen {gen:4d}: best={gen_best:.2f}  avg={np.mean(valid):.2f}  "
            f"valid={len(valid)}/{POPULATION_SIZE}  global_best={best_score:.2f}  "
            f"plateau={no_improve}/{PLATEAU_LIMIT}"
            + ("  *** NEW BEST ***" if gen_best < best_score else "")
        )

        emit({"type": "iteration", "score": gen_best, "best_score": best_score,
              "preference": costs[gen_best_idx][0], "accounting": costs[gen_best_idx][1],
              "iteration": gen, "improvements": 0 if gen_best >= best_score else 1})

        if gen_best < best_score:
            best_score = gen_best
            best_assignment = pop[gen_best_idx].copy()
            best_occ = occs[gen_best_idx].copy()
            best_cost = costs[gen_best_idx]
            score_history.append([gen, best_score])
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PLATEAU_LIMIT:
            print(f"Stopping early: no improvement for {PLATEAU_LIMIT} generations.")
            break

    # Final polish on the overall best solution
    if not _stop:
        print("\nRunning final local search polish...")
        best_assignment, best_occ, best_cost = _local_search(
            best_assignment,
            best_occ,
            best_cost,
            family_size,
            choices,
            n_days,
            min_occ,
            max_occ,
            rng,
        )
        best_score = _total_score(*best_cost)
        score_history.append([N_GENERATIONS + 1, best_score])

    emit({"type": "done", "score": best_score,
          "preference": best_cost[0], "accounting": best_cost[1],
          "occupancy": best_occ.tolist(), "assignment": best_assignment.tolist()})

    print(f"\nBest score found: {best_score:.2f}")
    return best_assignment, best_score, score_history
