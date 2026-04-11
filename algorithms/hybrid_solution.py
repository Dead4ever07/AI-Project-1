import pandas as pd
from algorithms import lp_solution, hill_climbing
from algorithms.cost import cost_function

_callback = None
_stop = False

def emit(ev: dict):
    if _callback is not None:
        _callback(ev)

def run(submission, family_size, choices, n_days, min_occ, max_occ):
    """
    Hybrid approach: LP followed by Hill Climbing.
    """
    print("Starting Hybrid Approach (LP + Hill Climbing)...")
    
    lp_assignment = lp_solution.generate_lp_solution(family_size, choices, n_days, min_occ, max_occ)
    
    if isinstance(submission, pd.DataFrame):
        lp_submission = submission.copy()
        lp_submission["assigned_day"] = lp_assignment 
    else:
        lp_submission = pd.DataFrame({"assigned_day": lp_assignment})
        lp_submission.index.name = "family_id"

    print("\nRefining LP solution with Hill Climbing...")
    final_assignment, final_score = hill_climbing.run(
        lp_submission, family_size, choices, n_days, min_occ, max_occ
    )
    
    return final_assignment, final_score
