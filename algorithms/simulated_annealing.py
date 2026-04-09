import numpy as np
from algorithms.cost import build_daily_occupancy, cost_function, delta_cost

INITIAL_TEMPERATURE = 10.0
COOLING_SCHEDULE = 0.995
PLATEAU_LIMIT = 100
MAX_ITERATIONS = 500

_callback = None
_stop = False

def emit(ev: dict):
    if _callback is not None:
        _callback(ev)

def run(submission, family_size, choices, n_days, min_occupancy, max_occupancy):
    assignment = submission["assigned_day"].values.copy()
    daily_occupancy = build_daily_occupancy(assignment, family_size, n_days)

    result = cost_function(
        assignment, family_size, choices, n_days, min_occupancy, max_occupancy
    )
    if result is None:
        raise ValueError("Starting solution is invalid!")

    preference_penalty, accounting_cost = result
    current_score = preference_penalty + accounting_cost

    print(
        f"Initial score: {current_score:.2f}  (preference: {preference_penalty:.2f}  accounting: {accounting_cost:.2f})"
    )

    emit({"type": "start", "score": current_score,
          "preference": preference_penalty, "accounting": accounting_cost,
          "occupancy": daily_occupancy.tolist(), "assignment": assignment.tolist()})

    best_assignment = assignment.copy()
    best_score = current_score
    scores_trajectory = [[0, best_score]]

    n_families = len(assignment)
    temperature = INITIAL_TEMPERATURE
    min_temperature = 1e-12
    rng = np.random.default_rng()

    iterations_since_last_improvement = 0
    iteration = 0

    while (
        not _stop
        and iteration < MAX_ITERATIONS
        and iterations_since_last_improvement < PLATEAU_LIMIT
        and temperature > min_temperature
    ):
        iteration += 1
        accepted = 0
        improved = 0

        for family_id in rng.permutation(n_families):
            old_day = assignment[family_id]

            new_day = choices[family_id, rng.integers(0, 10)]

            if new_day == old_day:
                continue

            result = delta_cost(
                family_id,
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

            preference_delta, accounting_delta = result
            total_delta = preference_delta + accounting_delta

            accepted_move = False
            if total_delta < 0 or rng.random() < np.exp(-total_delta / max(temperature, min_temperature)):
                accepted_move = True

            if accepted_move:
                daily_occupancy[old_day] -= family_size[family_id]
                daily_occupancy[new_day] += family_size[family_id]
                assignment[family_id] = new_day
                preference_penalty += preference_delta
                accounting_cost += accounting_delta
                current_score += total_delta
                accepted += 1
                if total_delta < 0:
                    improved += 1

            if current_score < best_score:
                best_score = current_score
                best_assignment = assignment.copy()
                scores_trajectory.append([iteration, best_score])
                iterations_since_last_improvement = 0
        print(
            f"Iteration {iteration}: score={current_score:.2f}  best={best_score:.2f}  "
            f"preference={preference_penalty:.2f}  accounting={accounting_cost:.2f}  "
            f"T={temperature:.4f}  accepted={accepted}  improved={improved}  "
            f"plateau={iterations_since_last_improvement}/{PLATEAU_LIMIT}"
        )
        emit({"type": "iteration", "score": current_score, "best_score": best_score,
              "preference": preference_penalty, "accounting": accounting_cost,
              "iteration": iteration, "temperature": round(temperature, 6),
              "accepted": accepted, "improved": improved})

        temperature = max(temperature * COOLING_SCHEDULE, min_temperature)
        iterations_since_last_improvement += 1

    emit({"type": "done", "score": best_score,
          "preference": preference_penalty, "accounting": accounting_cost,
          "occupancy": daily_occupancy.tolist(), "assignment": best_assignment.tolist()})

    print(f"\nBest score found: {best_score:.2f}")
    return best_assignment, best_score, scores_trajectory
