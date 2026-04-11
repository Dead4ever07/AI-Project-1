import numpy as np
import pandas as pd

def generate_initial_solution(family_size, choices, n_days, min_occ, max_occ):
    """
    Generate an initial valid submission satisfying the problem constraints.
    
    Args:
        family_size (np.ndarray): Array of family sizes.
        choices (np.ndarray): 2D array of top 10 choices for each family.
        n_days (int): Number of days before Christmas.
        min_occ (int): Minimum occupancy per day.
        max_occ (int): Maximum occupancy per day.
        
    Returns:
        pd.DataFrame: A DataFrame with index 'family_id' and column 'assigned_day'
    """
    print("Calculating the initial Solution")
    n_families = len(family_size)
    assigned_days = np.zeros(n_families, dtype=int)
    

    sorted_families = np.argsort(-family_size)
    
    day_occupancy = {day: 0 for day in range(1, n_days + 1)}
    
    for family_id in sorted_families:
        f_size = family_size[family_id]
        
        days_below_min = [day for day, occ in day_occupancy.items() if occ < min_occ]
        
        best_day = None
        
        if days_below_min:
            valid_days = [day for day in days_below_min if day_occupancy[day] + f_size <= max_occ]
            
            if valid_days:
                best_day = min(valid_days, key=lambda d: day_occupancy[d])
        
        if best_day is None:
            valid_days = [day for day, occ in day_occupancy.items() if occ + f_size <= max_occ]
            
            if not valid_days:
                raise ValueError(f"Knapsack constraint failed! Could not fit family {family_id} (size {f_size}) into any day without exceeding max_occ of {max_occ}.")
                
            best_day = max(valid_days, key=lambda d: day_occupancy[d])
            
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
