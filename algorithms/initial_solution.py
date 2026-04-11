import numpy as np
import pandas as pd

def generate_initial_solution(family_size, choices, n_days, min_occ, max_occ, seed=42):
    """
    Generate an initial valid submission satisfying the problem constraints.
    
    Randomization Logic:
      - A random number generator is initialized using the provided `seed` (default 42).
      - Families are first sorted by size (largest first). Families of the same size
        are then randomly shuffled to prevent deterministic tie-breaking.
      - When placing a family, instead of strictly picking the single best valid day,
        the algorithm identifies the top 3 best valid days (e.g., those furthest below
        the minimum occupancy) and selects one at random.
      - This approach preserves the greedy strategy needed to satisfy constraints 
        while introducing enough variance across different seeds to explore diverse
        starting points.
    
    Args:
        family_size (np.ndarray): Array of family sizes.
        choices (np.ndarray): 2D array of top 10 choices for each family.
        n_days (int): Number of days before Christmas.
        min_occ (int): Minimum occupancy per day.
        max_occ (int): Maximum occupancy per day.
        seed (int): Random seed for reproducibility and varying initial solutions.
        
    Returns:
        pd.DataFrame: A DataFrame with index 'family_id' and column 'assigned_day'
    """
    print(f"Calculating the initial Solution (seed={seed})")
    rng = np.random.default_rng(seed)
    
    n_families = len(family_size)
    assigned_days = np.zeros(n_families, dtype=int)
    

    noisy_sizes = family_size + rng.random(n_families) * 0.5
    sorted_families = np.argsort(-noisy_sizes)
    
    day_occupancy = {day: 0 for day in range(1, n_days + 1)}
    
    for family_id in sorted_families:
        f_size = family_size[family_id]
        
        days_below_min = [day for day, occ in day_occupancy.items() if occ < min_occ]
        
        best_day = None
        
        if days_below_min:
            valid_days = [day for day in days_below_min if day_occupancy[day] + f_size <= max_occ]
            
            if valid_days:
                valid_days.sort(key=lambda d: day_occupancy[d])
                top_k = valid_days[:3]
                best_day = rng.choice(top_k)
        
        if best_day is None:
            valid_days = [day for day, occ in day_occupancy.items() if occ + f_size <= max_occ]
            
            if not valid_days:
                raise ValueError(f"Knapsack constraint failed! Could not fit family {family_id} (size {f_size}) into any day without exceeding max_occ of {max_occ}.")
                
            valid_days.sort(key=lambda d: day_occupancy[d])
            top_k = valid_days[:3]
            best_day = rng.choice(top_k)
            
        assigned_days[family_id] = best_day
        day_occupancy[best_day] += f_size
        
    for day, occ in day_occupancy.items():
        if occ < min_occ or occ > max_occ:
            raise ValueError(f"Invalid assignment: Day {day} has {occ} people (Constraints: {min_occ}-{max_occ}).")
    
    df = pd.DataFrame({
        "family_id": np.arange(n_families),
        "assigned_day": assigned_days
    })
    df.set_index("family_id", inplace=True)
    return df
