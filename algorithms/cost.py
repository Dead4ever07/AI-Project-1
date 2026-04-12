import numpy as np

PREFERENCE_PENALTIES = [
    (0, 0),  # choice_0
    (50, 0),  # choice_1
    (50, 9),  # choice_2
    (100, 9),  # choice_3
    (200, 9),  # choice_4
    (200, 18),  # choice_5
    (300, 18),  # choice_6
    (300, 36),  # choice_7
    (400, 36),  # choice_8
    (500, 235),  # choice_9
    (500, 434),  # not in top 10
]


def preference_penalty(fam_id, day, family_size, choices):
    n = family_size[fam_id]
    for rank in range(10):
        if day == choices[fam_id, rank]:
            flat_cost, cost_per_member = PREFERENCE_PENALTIES[rank]
            return flat_cost + cost_per_member * n

    flat_cost, cost_per_member = PREFERENCE_PENALTIES[10]
    return flat_cost + cost_per_member * n


def _accounting_cost_last_day(today, min_occupancy):
    return max(0.0, (today - min_occupancy) / 400.0 * today**0.5)

def _accounting_cost(today, yesterday, min_occupancy):
    diff = abs(today - yesterday)
    return max(0.0, (today - min_occupancy) / 400.0 * today ** (0.5 + diff / 50.0))


def _accounting_cost_for_day(d, n_days, daily_occupancy, min_occupancy):
    if d == n_days:
        return _accounting_cost_last_day(daily_occupancy[d], min_occupancy)
    return _accounting_cost(daily_occupancy[d], daily_occupancy[d + 1], min_occupancy)


def total_accounting_cost(daily_occupancy, n_days, min_occupancy):
    """Returns the total accounting cost for the full occupancy array."""
    total = _accounting_cost_last_day(daily_occupancy[n_days], min_occupancy)
    yesterday = daily_occupancy[n_days]
    for day in range(n_days - 1, 0, -1):
        today = daily_occupancy[day]
        total += _accounting_cost(today, yesterday, min_occupancy)
        yesterday = today
    return total


def build_daily_occupancy(prediction, family_size, n_days):
    """Build the occupancy array from a prediction. Call once and store the result."""
    daily_occupancy = np.zeros(n_days + 1)
    for f, d in enumerate(prediction):
        daily_occupancy[d] += family_size[f]
    return daily_occupancy


def cost_function(
    prediction, family_size, choices, n_days, min_occupancy, max_occupancy
):
    """
    Returns (preference_penalty, accounting_cost) or None if the solution is invalid.
    """
    penalty = 0
    daily_occupancy = np.zeros(n_days + 1)

    for f, d in enumerate(prediction):
        daily_occupancy[d] += family_size[f]
        penalty += preference_penalty(f, d, family_size, choices)

    for occupancy in daily_occupancy[1:]:
        if occupancy > max_occupancy or occupancy < min_occupancy:
            return None

    accounting_cost = _accounting_cost_last_day(daily_occupancy[n_days], min_occupancy)
    yesterday = daily_occupancy[n_days]
    for day in range(n_days - 1, 0, -1):
        today = daily_occupancy[day]
        accounting_cost += _accounting_cost(today, yesterday, min_occupancy)
        yesterday = today

    return penalty, accounting_cost


def delta_cost(
    fam_id,
    old_day,
    new_day,
    old_accounting_cost,
    daily_occupancy,
    family_size,
    choices,
    n_days,
    min_occupancy,
    max_occupancy,
):
    """
    Returns (pref_delta, acc_delta) if the move is valid,
    or None if the move would violate occupancy constraints.
    """
    n = family_size[fam_id]

    # Hard validity check
    if daily_occupancy[old_day] - n < min_occupancy:
        return None
    if daily_occupancy[new_day] + n > max_occupancy:
        return None

    # Preference delta
    preference_delta = preference_penalty(
        fam_id, new_day, family_size, choices
    ) - preference_penalty(fam_id, old_day, family_size, choices)

    affected = set()
    for d in (old_day, new_day):
        affected.add(d)
        if d - 1 >= 1:
            affected.add(d - 1)

    cost_before = sum(_accounting_cost_for_day(d, n_days, daily_occupancy, min_occupancy) for d in affected)

    # Put the family in the new date to calculate the accounting cost
    daily_occupancy[old_day] -= n
    daily_occupancy[new_day] += n

    cost_after = sum(_accounting_cost_for_day(d, n_days, daily_occupancy, min_occupancy) for d in affected)


    # Put the family in the old date
    daily_occupancy[old_day] += n
    daily_occupancy[new_day] -= n

    return preference_delta, cost_after - cost_before
