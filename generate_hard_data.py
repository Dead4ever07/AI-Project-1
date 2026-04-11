import pandas as pd
import numpy as np
from algorithms.initial_solution import generate_initial_solution

def create_hard_dataset(filename, n_days, n_families, min_occ, max_occ, seed=42):
    rng = np.random.default_rng(seed)
    
    # Generate family sizes (mostly between 2 and 8)
    sizes = rng.choice([2, 3, 4, 5, 6, 7, 8], size=n_families, p=[0.1, 0.25, 0.3, 0.15, 0.1, 0.05, 0.05])
    
    total_people = sizes.sum()
    print(f"Generating {filename}:")
    print(f"  Days: {n_days}, Families: {n_families}, Total people: {total_people}")
    print(f"  Capacity required: {n_days * min_occ} to {n_days * max_occ}")
    
    if total_people < n_days * min_occ or total_people > n_days * max_occ:
        print(f"  WARNING: Total people {total_people} is out of bounds for min/max occupancy.")
    
    # Generate choices with high conflict
    # We create "popular" days (e.g., days 1 to 5, and weekends if we assume 7-day cycles)
    popular_days = np.array([1, 2, 3, 8, 9, 15, 16, 22, 23, 29, 30])
    popular_days = popular_days[popular_days <= n_days]
    other_days = np.setdiff1d(np.arange(1, n_days + 1), popular_days)
    
    choices = []
    for _ in range(n_families):
        # 80% chance a family heavily prefers the popular days
        if rng.random() < 0.80 and len(popular_days) >= 5:
            # Pick 5 from popular, 5 from others
            fam_choices_pop = rng.choice(popular_days, size=min(5, len(popular_days)), replace=False)
            fam_choices_oth = rng.choice(other_days, size=10 - len(fam_choices_pop), replace=False)
            fam_choices = np.concatenate([fam_choices_pop, fam_choices_oth])
            rng.shuffle(fam_choices) # Shuffle so top choice isn't ALWAYS a popular day, but highly likely
            # Actually, let's heavily weight the top choices to be popular days
            fam_choices = np.concatenate([
                rng.choice(popular_days, size=min(3, len(popular_days)), replace=False),
                rng.choice(np.setdiff1d(popular_days, fam_choices_pop[:3]), size=0), # Placeholder for logic clarity
            ])
            
            # Re-do to ensure top choices are the popular ones
            top_pop = rng.choice(popular_days, size=min(7, len(popular_days)), replace=False)
            rem_choices = rng.choice(np.setdiff1d(np.arange(1, n_days + 1), top_pop), size=10-len(top_pop), replace=False)
            fam_choices = np.concatenate([top_pop, rem_choices])
        else:
            # Random preferences
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
        # Try to relax constraints if it fails
        print("  Attempting to relax constraints slightly to ensure solvability...")
        try:
             sub = generate_initial_solution(sizes, choices, n_days, min_occ-5, max_occ+5)
             print(f"  SUCCESS with relaxed constraints (min={min_occ-5}, max={max_occ+5}). Saving with these bounds.")
             with open(filename, "w") as f:
                 f.write("number_days,max_occupancy,min_occupancy\n")
                 f.write(f"{n_days},{max_occ+5},{min_occ-5}\n")
             df.to_csv(filename, mode='a')
             print(f"  Saved to {filename}\n")
        except Exception as e2:
             print(f"  Still FAILED: {e2}")

# Hard dataset: 30 days, 800 families. 
# Total people ~ 800 * 4.2 = 3360.
# Average people per day = 112.
# Tight bounds: min=100, max=125.
# High conflict on specific days.
create_hard_dataset("input/hard_conflicts_input.csv", n_days=30, n_families=800, min_occ=100, max_occ=125, seed=1337)
