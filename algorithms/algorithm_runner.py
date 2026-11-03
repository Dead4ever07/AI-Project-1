import pandas as pd
from algorithms import hill_climbing, simulated_annealing, genetic_algorithm, lp_solution, hybrid_solution


def load_data(data_file, submission_file):
    """Function responsible for fetching a dataset from data_file"""
    data = pd.read_csv(data_file, index_col="family_id", skiprows=2)
    
    metadata = pd.read_csv(data_file, nrows=1)
    n_days = metadata.iloc[0, 0]
    max_occ = metadata.iloc[0, 1]
    min_occ = metadata.iloc[0, 2]

    submission = pd.read_csv(submission_file, index_col="family_id")
        
    return data, submission, (n_days, max_occ, min_occ)


def prepare_arrays(data):
    family_size = data["n_people"].values
    choice_cols = [f"choice_{i}" for i in range(10)]
    choices = data[choice_cols].values
    return family_size, choices


def run_algorithms(
    data_file, submission_file, output_dir, n_days_override, min_occ_override, max_occ_override
):
    data, submission, (n_days, max_occ, min_occ) = load_data(data_file, submission_file)
    
    n_days = n_days_override or n_days
    min_occupancy = min_occ_override or min_occ
    max_occupancy = max_occ_override or max_occ

    family_size, choices = prepare_arrays(data)

    algorithms = [
        ("hill_climbing", hill_climbing.run),
        ("simulated_annealing", simulated_annealing.run),
        ("genetic_algorithm", genetic_algorithm.run),
        ("lp_solution", lp_solution.run),
        ("hybrid_solution", hybrid_solution.run),
        ("random_start_hill_climbing", hill_climbing.run)
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
