import pandas as pd
from algorithms import hill_climbing, simulated_annealing, genetic_algorithm

def load_data(data_file, submission_file):
    data = pd.read_csv(data_file, index_col="family_id")
    submission = pd.read_csv(submission_file, index_col="family_id")
    return data, submission


def prepare_arrays(data):
    family_size = data["n_people"].values
    choice_cols = [f"choice_{i}" for i in range(10)]
    choices = data[choice_cols].values
    return family_size, choices


def run_algorithms(
    data_file, submission_file, output_dir, n_days, min_occupancy, max_occupancy
):
    data, submission = load_data(data_file, submission_file)
    family_size, choices = prepare_arrays(data)

    algorithms = [
        ("hill_climbing", hill_climbing.run),
        ("simulated_annealing", simulated_annealing.run),
        ("genetic_algorithm", genetic_algorithm.run)
    ]

    for name, algorithm in algorithms:
        print(f"\nRunning {name}...")
        result = algorithm(
            submission.copy(),
            family_size,
            choices,
            n_days,
            min_occupancy,
            max_occupancy,
        )
        solution, score = result[0], result[1]
        print(f"{name} score: {score}")

        algorithm_submission = submission.copy()
        algorithm_submission["assigned_day"] = solution.astype(int)
        algorithm_submission.to_csv(output_dir / f"{name}_submission.csv")
        print(f"Saved output for {name} -> {name}_submission.csv")
