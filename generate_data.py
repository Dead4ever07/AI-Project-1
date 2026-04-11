import pandas as pd
import numpy as np
from algorithms.initial_solution import generate_initial_solution

def create_dataset(filename, n_days, n_families, min_occ, max_occ, seed=42):
    rng = np.random.default_rng(seed)
    
    # Generate family sizes (mostly between 2 and 8)
    sizes = rng.choice([2, 3, 4, 5, 6, 7, 8], size=n_families, p=[0.1, 0.25, 0.3, 0.15, 0.1, 0.05, 0.05])
    
    total_people = sizes.sum()
    print(f"Generating {filename}:")
    print(f"  Days: {n_days}, Families: {n_families}, Total people: {total_people}")
    print(f"  Capacity required: {n_days * min_occ} to {n_days * max_occ}")
    
    if total_people < n_days * min_occ or total_people > n_days * max_occ:
        print(f"  WARNING: Total people {total_people} is out of bounds for min/max occupancy.")
        # Adjust families if needed... (we'll just trust the parameters for now, or tweak them)
    
    # Generate choices (1 to n_days)
    choices = []
    for _ in range(n_families):
        # Pick 10 unique random days for each family
        fam_choices = rng.choice(np.arange(1, n_days + 1), size=10, replace=False)
        choices.append(fam_choices)
    choices = np.array(choices)
    
    # Create DataFrame
    columns = [f"choice_{i}" for i in range(10)] + ["n_people"]
    data = np.hstack((choices, sizes.reshape(-1, 1)))
    df = pd.DataFrame(data, columns=columns)
    df.index.name = "family_id"
    
    # Test with initial_solution
    try:
        sub = generate_initial_solution(sizes, choices, n_days, min_occ, max_occ)
        print(f"  SUCCESS: Generated valid initial solution.")
        
        # Save to CSV
        with open(filename, "w") as f:
            f.write("number_days,max_occupancy,min_occupancy\n")
            f.write(f"{n_days},{max_occ},{min_occ}\n")
        df.to_csv(filename, mode='a')
        print(f"  Saved to {filename}\n")
    except Exception as e:
        print(f"  FAILED to generate initial solution: {e}\n")

# Create a small dataset (10 days, approx 600 people)
create_dataset("input/small_input.csv", n_days=10, n_families=150, min_occ=40, max_occ=80)

# Create a medium dataset (50 days, approx 6000 people)
create_dataset("input/medium_input.csv", n_days=50, n_families=1500, min_occ=100, max_occ=150)

# Create a large dataset (200 days, approx 40000 people)
create_dataset("input/large_input.csv", n_days=200, n_families=10000, min_occ=150, max_occ=250)
