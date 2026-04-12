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
    print(f"Calculating the initial Solution")
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


def build_individual(
    initial_assignment,
    family_sizes,
    family_choices,
    n_days,
    min_occupancy,
    max_occupancy,
    rng,
):
    """
    1. In random family order, assign families to preferred days, by their preference order, respecting max occupancy.
    2. If any families are left unassigned, assign them to their least crowded preferred day that isn't already full.
    3. If any days are below minimum occupancy: Move families from the busiest day to the under-occupied day.
    """

    n_families = len(initial_assignment)

    assignment = np.empty(n_families, dtype=initial_assignment.dtype)
    daily_occupancy = np.zeros(n_days + 1, dtype=np.int32)

    families_in_day = {d: [] for d in range(1, n_days + 1)}

    unassigned_families = []

    for family_id in rng.permutation(n_families):

        size = family_sizes[family_id]
        choices = family_choices[family_id]

        assigned = False

        for day in choices:
            if daily_occupancy[day] + size <= max_occupancy:
                assignment[family_id] = day
                daily_occupancy[day] += size
                families_in_day[day].append(family_id)
                assigned = True
                break

        if not assigned:
            unassigned_families.append(family_id)

    unassigned_families.sort(key=lambda f: -family_sizes[f])

    for family_id in unassigned_families:

        size = family_sizes[family_id]

        for day in sorted(range(1, n_days + 1), key=lambda d: daily_occupancy[d]):
            if daily_occupancy[day] + size <= max_occupancy:
                assignment[family_id] = day
                daily_occupancy[day] += size
                families_in_day[day].append(family_id)
                break

    for day in range(1, n_days + 1):

        if daily_occupancy[day] >= min_occupancy:
            continue

        while daily_occupancy[day] < min_occupancy:

            busiest_day = np.argmax(daily_occupancy[1:]) + 1

            if busiest_day == day:
                break

            moved = False

            for family_id in families_in_day[busiest_day]:

                size = family_sizes[family_id]

                if (
                    daily_occupancy[busiest_day] - size >= min_occupancy
                    and daily_occupancy[day] + size <= max_occupancy
                ):
                    assignment[family_id] = day

                    daily_occupancy[busiest_day] -= size
                    daily_occupancy[day] += size

                    families_in_day[busiest_day].remove(family_id)
                    families_in_day[day].append(family_id)

                    moved = True
                    break

            if not moved:
                break

    return assignment, daily_occupancy
