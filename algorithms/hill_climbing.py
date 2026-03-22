from algorithms.cost import build_daily_occupancy, cost_function, delta_cost

MAX_ITERATIONS = 100
MIN_IMPROVEMENTS = 5

# TODO: update the frontend
# TODO: sort by larger families and see if it makes difference
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
    # TODO: update the frontend

    for iteration in range(1, MAX_ITERATIONS + 1):
        improvements = 0

        for fam_id in range(len(assignment)):
            old_day = assignment[fam_id]

            best_delta = 0
            best_preference_delta = 0
            best_accounting_delta = 0
            best_new_day = None

            for pick in range(10):
                new_day = choices[fam_id, pick]

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

                preference_delta, accounting_delta = result
                total_delta = preference_delta + accounting_delta

                if total_delta < best_delta:
                    best_delta = total_delta
                    best_preference_delta = preference_delta
                    best_accounting_delta = accounting_delta
                    best_new_day = new_day

            if best_new_day is not None:
                daily_occupancy[old_day] -= family_size[fam_id]
                daily_occupancy[best_new_day] += family_size[fam_id]
                assignment[fam_id] = best_new_day
                preference_penalty += best_preference_delta
                accounting_cost += best_accounting_delta
                current_score += best_delta
                # TODO: maybe fazer aqui um threshold de improvement!
                improvements += 1
                # TODO: update the frontend

        print(
            f"  Iteration {iteration}: score={current_score:.2f}  "
            f"preference={preference_penalty:.2f}  accounting={accounting_cost:.2f}  "
            f"improvements={improvements}"
        )
        # TODO: update the frontend

        if improvements < MIN_IMPROVEMENTS:
            print(
                f"  Stopping early: fewer than {MIN_IMPROVEMENTS} improvements in this pass."
            )
            break
    else:
        print(f"  Stopping: reached max iterations ({MAX_ITERATIONS}).")

    return assignment, current_score
