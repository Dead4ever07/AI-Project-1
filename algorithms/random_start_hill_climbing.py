import numpy as np
import algorithms.hill_climbing as hill_climbing
from algorithms.initial_solution import build_individual
from algorithms.cost import build_daily_occupancy, cost_function

N_RESTARTS = 10

_callback = None
_stop = False


def emit(ev: dict):
    if _callback is not None:
        _callback(ev)


def run(submission, family_size, choices, n_days, min_occupancy, max_occupancy):
    initial_assignment = submission["assigned_day"].values.copy()
    rng = np.random.default_rng()

    daily_occupancy = build_daily_occupancy(initial_assignment, family_size, n_days)
    result = cost_function(
        initial_assignment, family_size, choices, n_days, min_occupancy, max_occupancy
    )
    if result is None:
        raise ValueError("Starting solution is invalid!")
    
    preference_penalty, accounting_cost = result
    current_score = preference_penalty + accounting_cost
    emit({"type": "start", "score": current_score,
          "preference": preference_penalty, "accounting": accounting_cost,
          "occupancy": daily_occupancy.tolist(), "assignment": initial_assignment.tolist()})

    best_assignment = None
    best_score = float("inf")

    for restart in range(N_RESTARTS + 1):
        if _stop:
            print("Stopping: stop requested.")
            break

        if restart == 0:
            print(
                f"\n=== Restart {restart} / {N_RESTARTS}: using provided solution ==="
            )
            assignment = initial_assignment.copy()
        else:
            print(
                f"\n=== Restart {restart} / {N_RESTARTS}: building random solution ==="
            )
            assignment, _ = build_individual(
                initial_assignment,
                family_size,
                choices,
                n_days,
                min_occupancy,
                max_occupancy,
                rng,
            )

        # Patch stop/callback through to the inner hill climber
        hill_climbing._callback = _callback
        hill_climbing._stop = _stop

        sub = submission.copy()
        sub["assigned_day"] = assignment

        climbed_assignment, score, _ = hill_climbing.run(
            sub, family_size, choices, n_days, min_occupancy, max_occupancy, emit_start=False, emit_done=False
        )

        print(
            f"  => Score after hill climbing: {score:.2f}  (best so far: {best_score:.2f})"
        )

        if score < best_score:
            best_score = score
            best_assignment = climbed_assignment.copy()
            print(f"  *** New best: {best_score:.2f} ***")

        emit(
            {
                "type": "restart",
                "restart": restart,
                "score": score,
                "best_score": best_score,
            }
        )

    print(f"\n=== Done. Best score: {best_score:.2f} ===")

    # Recompute occupancy for the final emit
    daily_occupancy = build_daily_occupancy(best_assignment, family_size, n_days)
    result = cost_function(
        best_assignment, family_size, choices, n_days, min_occupancy, max_occupancy
    )
    preference_penalty, accounting_cost = result

    emit(
        {
            "type": "done",
            "score": best_score,
            "preference": preference_penalty,
            "accounting": accounting_cost,
            "occupancy": daily_occupancy.tolist(),
            "assignment": best_assignment.tolist(),
        }
    )

    return best_assignment, best_score, []
