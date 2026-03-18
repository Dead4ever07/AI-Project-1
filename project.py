import numpy as np
import pandas as pd
from pathlib import Path

# ================================
# PATHS
# ================================

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_FILE = INPUT_DIR / "easy_input.csv"
SUBMISSION_FILE = INPUT_DIR / "easy_sample_sumbmission.csv"

# ================================
# CONSTANTS
# ================================

N_DAYS = 10
MAX_OCCUPANCY = 20
MIN_OCCUPANCY = 0
INVALID_PENALTY = 100_000_000_000_000_000

# ================================
# LOAD DATA
# ================================


def load_data():
    data = pd.read_csv(DATA_FILE, index_col="family_id")
    submission = pd.read_csv(SUBMISSION_FILE, index_col="family_id")
    return data, submission


# ================================
# PREPARE ARRAYS
# ================================


def prepare_arrays(data):

    # family sizes
    family_size = data["n_people"].values

    # preference matrix (families × 10)
    choice_cols = [f"choice_{i}" for i in range(10)]
    choices = data[choice_cols].values

    return family_size, choices


# ================================
# COST FUNCTION
# ================================


def cost_function(prediction, family_size, choices):

    penalty = 0

    # occupancy array (index = day)
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

    # ================================
    # OCCUPANCY CONSTRAINT
    # ================================

    for occ in daily_occupancy[1:]:
        if occ > MAX_OCCUPANCY or occ < MIN_OCCUPANCY:
            penalty += INVALID_PENALTY

    # ================================
    # ACCOUNTING COST
    # ================================

    accounting_cost = max(
        0,
        (daily_occupancy[N_DAYS] - MIN_OCCUPANCY) / 400.0 * daily_occupancy[N_DAYS] ** 0.5,
    )

    yesterday = daily_occupancy[N_DAYS]

    for day in range(N_DAYS-1, 0, -1):

        today = daily_occupancy[day]
        diff = abs(today - yesterday)

        accounting_cost += max(
            0,
            (today - MIN_OCCUPANCY) / 400.0 * today ** (0.5 + diff / 50.0),
        )

        yesterday = today

    return penalty + accounting_cost


# ================================
# LOCAL SEARCH OPTIMIZATION
# ================================


def improve_solution(submission, family_size, choices):

    best = submission["assigned_day"].values
    best_score = cost_function(best, family_size, choices)

    new = best.copy()

    for fam_id in range(len(best)):

        for pick in range(10):

            day = choices[fam_id, pick]

            temp = new.copy()
            temp[fam_id] = day

            score = cost_function(temp, family_size, choices)

            if score < best_score:
                new = temp.copy()
                best_score = score

    return new, best_score


# ================================
# MAIN
# ================================


def main():

    data, submission = load_data()

    family_size, choices = prepare_arrays(data)

    new_solution, score = improve_solution(
        submission,
        family_size,
        choices,
    )

    submission["assigned_day"] = new_solution
    submission.loc["cost"] = [score]
    submission.to_csv(OUTPUT_DIR / "submission.csv")

if __name__ == "__main__":
    main()
