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
SUBMISSION_FILE = ( INPUT_DIR / "easy_sample_sumbmission.csv")


# ================================
# CONSTANTS
# ================================

N_DAYS = 100
MAX_OCCUPANCY = 5
MIN_OCCUPANCY = 3
DAYS = list(range(N_DAYS, 0, -1))


# ================================
# LOAD DATA
# ================================


def load_data():
    data = pd.read_csv(DATA_FILE, index_col="family_id")
    submission = pd.read_csv(SUBMISSION_FILE, index_col="family_id")

    return data, submission


# ================================
# PREPARE LOOKUP STRUCTURES
# ================================


def prepare_dicts(data):

    family_size_dict = data["n_people"].to_dict()
    choice_dict = data[[f"choice_{i}" for i in range(10)]].to_dict()

    return family_size_dict, choice_dict


# ================================
# COST FUNCTION
# ================================


def cost_function(prediction, family_size_dict, choice_dict):

    penalty = 0
    daily_occupancy = {day: 0 for day in DAYS}

    for f, d in enumerate(prediction):

        n = family_size_dict[f]
        daily_occupancy[d] += n

        # preference penalty
        choices = [choice_dict[f"choice_{i}"][f] for i in range(10)]

        if d == choices[0]:
            penalty += 0
        elif d == choices[1]:
            penalty += 50
        elif d == choices[2]:
            penalty += 50 + 9 * n
        elif d == choices[3]:
            penalty += 100 + 9 * n
        elif d == choices[4]:
            penalty += 200 + 9 * n
        elif d == choices[5]:
            penalty += 200 + 18 * n
        elif d == choices[6]:
            penalty += 300 + 18 * n
        elif d == choices[7]:
            penalty += 300 + 36 * n
        elif d == choices[8]:
            penalty += 400 + 36 * n
        elif d == choices[9]:
            penalty += 500 + 36 * n + 199 * n
        else:
            penalty += 500 + 36 * n + 398 * n

    # occupancy constraints
    for occ in daily_occupancy.values():
        if occ > MAX_OCCUPANCY or occ < MIN_OCCUPANCY:
            penalty += 100000000

    # accounting cost
    accounting_cost = max(
        0,
        (daily_occupancy[DAYS[0]] - 125.0) / 400.0 * daily_occupancy[DAYS[0]] ** 0.5,
    )

    yesterday = daily_occupancy[DAYS[0]]

    for day in DAYS[1:]:
        today = daily_occupancy[day]
        diff = abs(today - yesterday)

        accounting_cost += max(
            0,
            (today - 125.0) / 400.0 * today ** (0.5 + diff / 50.0),
        )

        yesterday = today

    return penalty + accounting_cost


# ================================
# LOCAL SEARCH OPTIMIZATION
# ================================


def improve_solution(submission, family_size_dict, choice_dict):

    best = submission["assigned_day"].tolist()
    best_score = cost_function(best, family_size_dict, choice_dict)

    new = best.copy()

    for fam_id in range(len(best)):
        for pick in range(10):

            day = choice_dict[f"choice_{pick}"][fam_id]

            temp = new.copy()
            temp[fam_id] = day

            score = cost_function(temp, family_size_dict, choice_dict)

            if score < best_score:
                new = temp.copy()
                best_score = score

    return new, best_score


# ================================
# MAIN
# ================================


def main():

    data, submission = load_data()

    family_size_dict, choice_dict = prepare_dicts(data)

    new_solution, score = improve_solution(submission, family_size_dict, choice_dict)

    submission["assigned_day"] = new_solution

    submission.to_csv(OUTPUT_DIR / f"submission_{score}.csv")

    print(f"Score: {score}")


if __name__ == "__main__":
    main()
