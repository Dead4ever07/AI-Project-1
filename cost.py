import numpy as np

N_DAYS = 10
MAX_OCCUPANCY = 20
MIN_OCCUPANCY = 0
INVALID_PENALTY = 100_000_000_000_000_000


def cost_function(prediction, family_size, choices):

    penalty = 0
    daily_occupancy = np.zeros(N_DAYS + 1)

    for f, d in enumerate(prediction):

        n = family_size[f]
        daily_occupancy[d] += n

        if d == choices[f, 0]:
            penalty += 0
        elif d == choices[f, 1]:
            penalty += 50
        elif d == choices[f, 2]:
            penalty += 50 + 9 * n
        elif d == choices[f, 3]:
            penalty += 100 + 9 * n
        elif d == choices[f, 4]:
            penalty += 200 + 9 * n
        elif d == choices[f, 5]:
            penalty += 200 + 18 * n
        elif d == choices[f, 6]:
            penalty += 300 + 18 * n
        elif d == choices[f, 7]:
            penalty += 300 + 36 * n
        elif d == choices[f, 8]:
            penalty += 400 + 36 * n
        elif d == choices[f, 9]:
            penalty += 500 + 235 * n
        else:
            penalty += 500 + 434 * n

    for occ in daily_occupancy[1:]:
        if occ > MAX_OCCUPANCY or occ < MIN_OCCUPANCY:
            penalty += INVALID_PENALTY

    accounting_cost = max(
        0,
        (daily_occupancy[N_DAYS] - MIN_OCCUPANCY)
        / 400.0
        * daily_occupancy[N_DAYS] ** 0.5,
    )

    yesterday = daily_occupancy[N_DAYS]

    for day in range(N_DAYS - 1, 0, -1):
        today = daily_occupancy[day]
        diff = abs(today - yesterday)

        accounting_cost += max(
            0,
            (today - MIN_OCCUPANCY) / 400.0 * today ** (0.5 + diff / 50.0),
        )

        yesterday = today

    return penalty + accounting_cost
