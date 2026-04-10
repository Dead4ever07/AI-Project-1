import pandas as pd
from pathlib import Path
from algorithms.algorithm_runner import run_algorithms

# ================================
# CONFIG
# ================================
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_FILE = INPUT_DIR / "family_data.csv"
SUBMISSION_FILE = (
    INPUT_DIR / "sample_submission.csv"
)
N_DAYS = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125


def main():
    run_algorithms(DATA_FILE, SUBMISSION_FILE, OUTPUT_DIR, N_DAYS, MIN_OCCUPANCY, MAX_OCCUPANCY)

if __name__ == "__main__":
    main()
