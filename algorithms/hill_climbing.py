import numpy as np
from algorithms.cost import build_daily_occupancy, cost_function, delta_cost

MAX_ITERATIONS = 100
SORT = True

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

    if SORT:
        family_ids = np.argsort(family_size)
        print("Processing smallest families first.")
    else:
        family_ids = np.arange(len(assignment))

    for iteration in range(1, MAX_ITERATIONS + 1):
        if _stop:
            print("Stopping: stop requested.")
            break
        improvements = 0

        for family_id in family_ids:
            old_day = assignment[family_id]

            best_delta = 0
            best_preference_delta = 0
            best_accounting_delta = 0
            best_new_day = None

            for pick in range(10):
                new_day = choices[family_id, pick]

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

                if total_delta < best_delta:
                    best_delta = total_delta
                    best_preference_delta = preference_delta
                    best_accounting_delta = accounting_delta
                    best_new_day = new_day

            if best_new_day is not None:
                daily_occupancy[old_day] -= family_size[family_id]
                daily_occupancy[best_new_day] += family_size[family_id]
                assignment[family_id] = best_new_day
                preference_penalty += best_preference_delta
                accounting_cost += best_accounting_delta
                current_score += best_delta
                improvements += 1

        print(
            f"Iteration {iteration}: score={current_score:.2f}  "
            f"preference={preference_penalty:.2f}  accounting={accounting_cost:.2f}  "
            f"improvements={improvements}"
        )
        emit({"type": "iteration", "score": current_score,
              "preference": preference_penalty, "accounting": accounting_cost,
              "iteration": iteration, "improvements": improvements})

        if improvements < 1:
            print("Stopping early: there were no improvements.")
            break
    else:
        print(f"Stopping: reached max iterations ({MAX_ITERATIONS}).")

    emit({"type": "done", "score": current_score,
          "preference": preference_penalty, "accounting": accounting_cost,
          "occupancy": daily_occupancy.tolist(), "assignment": assignment.tolist()})

    return assignment, current_score, []