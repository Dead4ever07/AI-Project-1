"""
Simple Genetic Algorithm for family-day scheduling.

Each "individual" is an array mapping every family to a day.
We evolve a population of individuals over generations to minimise total cost
(preference penalties + accounting cost), keeping each day within occupancy bounds.
"""

import numpy as np
from algorithms.cost import build_daily_occupancy, cost_function, delta_cost, total_accounting_cost

# --- Settings ---
POPULATION_SIZE    = 20
N_GENERATIONS      = 500
MUTATION_RATE      = 0.04   # per-family chance of moving to a random preferred day
ELITE_COUNT        = 5      # best individuals carried unchanged to the next generation
PLATEAU_LIMIT      = 20     # stop if best score hasn't improved for this many generations
LOCAL_SEARCH_EVERY = 2      # polish the best individual every N generations

# Penalty for each preference rank (0 = top choice, 10 = not in top-10).
# Format: (flat_fee, per_member_fee)
PENALTIES = [
    (0, 0), (50, 0), (50, 9), (100, 9), (200, 9),
    (200, 18), (300, 18), (300, 36), (400, 36), (500, 235), (500, 434),
]


def _score(assignment, family_size, choices, n_days, min_occ, max_occ):
    """Score an assignment. Returns inf if any day is over/under capacity."""
    occ = build_daily_occupancy(assignment, family_size, n_days)
    if np.any(occ[1:] > max_occ) or np.any(occ[1:] < min_occ):
        return float("inf"), occ

    preference_cost = 0.0
    for fam, day in enumerate(assignment):
        rank = next((r for r in range(10) if choices[fam, r] == day), 10)
        flat, per_member = PENALTIES[rank]
        preference_cost += flat + per_member * family_size[fam]

    return preference_cost + total_accounting_cost(occ, n_days, min_occ), occ


def run(submission, family_size, choices, n_days, min_occupancy, max_occupancy):
    min_occ, max_occ = min_occupancy, max_occupancy
    base = submission["assigned_day"].values.copy()
    rng  = np.random.default_rng()

    result = cost_function(base, family_size, choices, n_days, min_occ, max_occ)
    if result is None:
        raise ValueError("Starting solution is invalid!")
    print(f"Initial score: {sum(result):.2f}")

    # --- Build initial population ---
    # First individual is the original submission; the rest are increasingly shuffled variants.
    population = []
    for i in range(POPULATION_SIZE):
        assignment = base.copy()
        shuffle_prob = 0.0 if i == 0 else 0.15 + 0.70 * (i / POPULATION_SIZE)
        occ = np.zeros(n_days + 1, dtype=np.int32)

        for fam in rng.permutation(len(base)):
            n     = family_size[fam]
            prefs = choices[fam].copy()
            if rng.random() < shuffle_prob:
                rng.shuffle(prefs)
            for day in prefs:
                if occ[day] + n <= max_occ:
                    break
            else:
                day = prefs[np.argmin([occ[d] for d in prefs])]
            occ[day] += n
            assignment[fam] = day

        score, occ = _score(assignment, family_size, choices, n_days, min_occ, max_occ)
        population.append((assignment, occ, score))

    population.sort(key=lambda x: x[2])
    best_assignment, _, best_score = population[0]
    best_assignment = best_assignment.copy()
    score_history = [[0, best_score]]
    no_improve = 0
    print(f"Initial population best: {best_score:.2f}")

    # --- Evolve ---
    for gen in range(1, N_GENERATIONS + 1):

        # Keep the best ELITE_COUNT individuals, breed the rest
        next_gen = [ind for ind in population[:ELITE_COUNT]]

        while len(next_gen) < POPULATION_SIZE:
            # Crossover: pick two parents randomly from the top half, combine them
            pa, _, _ = population[rng.integers(0, POPULATION_SIZE // 2)]
            pb, _, _ = population[rng.integers(0, POPULATION_SIZE // 2)]
            occ   = np.zeros(n_days + 1, dtype=np.int32)
            child = np.empty(len(pa), dtype=pa.dtype)

            for fam in rng.permutation(len(pa)):
                n    = family_size[fam]
                days = ([pa[fam], pb[fam]] if rng.random() < 0.5 else [pb[fam], pa[fam]]) + list(choices[fam])
                for day in days:
                    if occ[day] + n <= max_occ:
                        break
                else:
                    day = choices[fam, np.argmin([occ[d] for d in choices[fam]])]
                occ[day] += n
                child[fam] = day

            # Mutation: randomly move a few families to a different preferred day
            for fam in range(len(child)):
                if rng.random() >= MUTATION_RATE:
                    continue
                n = family_size[fam]
                old_day, new_day = child[fam], choices[fam, rng.integers(0, 10)]
                if new_day != old_day and occ[old_day] - n >= min_occ and occ[new_day] + n <= max_occ:
                    occ[old_day] -= n
                    occ[new_day] += n
                    child[fam] = new_day

            score, occ = _score(child, family_size, choices, n_days, min_occ, max_occ)
            next_gen.append((child, occ, score))

        population = sorted(next_gen, key=lambda x: x[2])

        # Periodically improve the best individual with a greedy local search
        if gen % LOCAL_SEARCH_EVERY == 0:
            best, best_occ, best_gen_score = population[0]
            pref_cost, acc_cost = cost_function(best, family_size, choices, n_days, min_occ, max_occ) or (0, 0)
            occ = build_daily_occupancy(best, family_size, n_days)

            for fam in rng.permutation(len(best)):
                old_day = best[fam]
                for rank in range(10):
                    new_day = choices[fam, rank]
                    if new_day == old_day:
                        continue
                    delta = delta_cost(fam, old_day, new_day, acc_cost, occ, family_size, choices, n_days, min_occ, max_occ)
                    if delta is not None and sum(delta) < 0:
                        occ[old_day] -= family_size[fam]
                        occ[new_day] += family_size[fam]
                        best[fam] = new_day
                        pref_cost += delta[0]
                        acc_cost  += delta[1]
                        old_day = new_day
                        break

            polished_score = pref_cost + acc_cost
            if polished_score < best_gen_score:
                population[0] = (best, occ, polished_score)

        # Logging and plateau detection
        gen_best = population[0][2]
        valid    = [s for _, _, s in population if s < float("inf")]
        print(
            f"Gen {gen:4d}: best={gen_best:.2f}  avg={np.mean(valid):.2f}  "
            f"plateau={no_improve}/{PLATEAU_LIMIT}"
            + ("  *** NEW BEST ***" if gen_best < best_score else "")
        )

        if gen_best < best_score:
            best_score      = gen_best
            best_assignment = population[0][0].copy()
            score_history.append([gen, best_score])
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PLATEAU_LIMIT:
            print(f"Stopping early: no improvement for {PLATEAU_LIMIT} generations.")
            break

    print(f"\nBest score found: {best_score:.2f}")
    return best_assignment, best_score, score_history