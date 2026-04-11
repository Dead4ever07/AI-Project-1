import pandas as pd
from algorithms import hill_climbing
from algorithms import simulated_annealing
from algorithms import lp_solution
from algorithms import hybrid_solution


# ================================
# LOAD DATA
# ================================
def load_data(data_file, submission_file):
    # Data file has metadata in first 2 rows
    data = pd.read_csv(data_file, index_col="family_id", skiprows=2)
    
    # Submission file (like sample_submission.csv) usually does NOT have metadata
    # We check if it has "family_id" in the first line
    with open(submission_file, 'r') as f:
        first_line = f.readline()
    
    if "family_id" in first_line:
        submission = pd.read_csv(submission_file, index_col="family_id")
    else:
        submission = pd.read_csv(submission_file, index_col="family_id", skiprows=2)
        
    return data, submission


def prepare_arrays(data):
    family_size = data["n_people"].values
    choice_cols = [f"choice_{i}" for i in range(10)]
    choices = data[choice_cols].values
    return family_size, choices


# ================================
# RUN ALGORITHMS
# ================================
def run_algorithms(
    data_file, submission_file, output_dir, n_days, min_occupancy, max_occupancy
):
    data, submission = load_data(data_file, submission_file)
    family_size, choices = prepare_arrays(data)

    algorithms = [
        ("lp_solution", lp_solution.run),
        ("hybrid_solution", hybrid_solution.run),
        ("simulated_annealing", simulated_annealing.run),
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
