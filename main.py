import pandas as pd
from pathlib import Path
from algorithms import hill_climbing

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
# LOAD DATA
# ================================
def load_data():
    data = pd.read_csv(DATA_FILE, index_col="family_id")
    submission = pd.read_csv(SUBMISSION_FILE, index_col="family_id")
    return data, submission


def prepare_arrays(data):
    family_size = data["n_people"].values
    choice_cols = [f"choice_{i}" for i in range(10)]
    choices = data[choice_cols].values
    return family_size, choices


# ================================
# MAIN
# ================================
def main():
    data, submission = load_data()
    family_size, choices = prepare_arrays(data)

    algorithms = [("hill_climbing", hill_climbing.run)]

    best_score = float("inf")
    best_solution = None
    best_name = ""

    for name, algorithm in algorithms:
        print(f"\nRunning {name}...")
        solution, score = algorithm(submission.copy(), family_size, choices)
        print(f"{name} score: {score}")

        algorithm_submission = submission.copy()
        algorithm_submission["assigned_day"] = solution
        algorithm_submission.loc["cost"] = [score]
        algorithm_submission.to_csv(OUTPUT_DIR / f"{name}_submission.csv")
        print(f"Saved output for {name} -> {name}_submission.csv")

        if score < best_score:
            best_score = score
            best_solution = solution
            best_name = name

    print(f"\nBest algorithm: {best_name} ({best_score})")
    best_submission = submission.copy()
    best_submission["assigned_day"] = best_solution
    best_submission.loc["cost"] = [best_score]
    best_submission.to_csv(OUTPUT_DIR / "best_submission.csv")
    print(f"Saved best solution -> best_submission.csv")


if __name__ == "__main__":
    main()
