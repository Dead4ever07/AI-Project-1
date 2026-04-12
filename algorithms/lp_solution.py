import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from algorithms.cost import build_daily_occupancy, PREFERENCE_PENALTIES, cost_function

_callback = None
_stop = False

DIFFERENCE_PENALTY_FIRST = 50.0
DIFFERENCE_PENALTY_SECOND = 10.0

def emit(ev: dict):
    if _callback is not None:
        _callback(ev)


def _base_accounting_cost(occ, min_occ, next_occ=None):
    if next_occ is None: next_occ = occ
    diff = abs(occ - next_occ)
    return (occ - min_occ) / 400.0 * (occ**(0.5 + diff / 50.0))


def _build_and_solve_lp(family_size, choices, n_days, min_occ, max_occ, fixed_occupancy=None):
    n_families = len(family_size)
    occupancy_range = np.arange(min_occ, max_occ + 1)
    n_occ_vars = len(occupancy_range)
    num_x_vars = n_families * n_days
    num_y_vars = n_days * n_occ_vars
    num_v_vars = n_days - 1
    num_vars = num_x_vars + num_y_vars + num_v_vars
    
    c = np.zeros(num_vars)
    for f in range(n_families):
        n = family_size[f]
        default_penalty = PREFERENCE_PENALTIES[10][0] + PREFERENCE_PENALTIES[10][1] * n
        for d_idx in range(n_days):
            c[f * n_days + d_idx] = default_penalty
        for rank, day in enumerate(choices[f]):
            flat_cost, cost_per_member = PREFERENCE_PENALTIES[rank]
            penalty = flat_cost + cost_per_member * n
            c[f * n_days + (day - 1)] = penalty

    if fixed_occupancy is None:
        for d in range(n_days):
            for i_idx, occ in enumerate(occupancy_range):
                c[num_x_vars + d * n_occ_vars + i_idx] = (occ - min_occ) / 400.0 * (occ**0.5)
    else:
        for d in range(n_days):
            next_occ = fixed_occupancy[d+1] if d < n_days else fixed_occupancy[d]
            for i_idx, occ in enumerate(occupancy_range):
                c[num_x_vars + (d-1) * n_occ_vars + i_idx] = _base_accounting_cost(occ, min_occ, next_occ)

    tv_weight = DIFFERENCE_PENALTY_FIRST if fixed_occupancy is None else DIFFERENCE_PENALTY_SECOND
    c[num_x_vars + num_y_vars:] = tv_weight

    rows_c1 = np.repeat(np.arange(n_families), n_days)
    cols_c1 = np.arange(num_x_vars)
    data_c1 = np.ones(num_x_vars)
    
    rows_c2 = np.repeat(np.arange(n_days), n_occ_vars) + n_families
    cols_c2 = np.arange(num_y_vars) + num_x_vars
    data_c2 = np.ones(num_y_vars)
    
    rows_c3_x = np.tile(np.arange(n_days), n_families) + n_families + n_days
    cols_c3_x = np.arange(num_x_vars)
    data_c3_x = np.repeat(family_size, n_days)
    
    rows_c3_y = np.repeat(np.arange(n_days), n_occ_vars) + n_families + n_days
    cols_c3_y = np.arange(num_y_vars) + num_x_vars
    data_c3_y = -np.tile(occupancy_range, n_days)
    
    A_eq = coo_matrix((np.concatenate([data_c1, data_c2, data_c3_x, data_c3_y]), 
                       (np.concatenate([rows_c1, rows_c2, rows_c3_x, rows_c3_y]), 
                        np.concatenate([cols_c1, cols_c2, cols_c3_x, cols_c3_y]))),
                      shape=(n_families + 2 * n_days, num_vars))
    b_eq = np.concatenate([np.ones(n_families), np.ones(n_days), np.zeros(n_days)])

    rows_ub, cols_ub, data_ub = [], [], []
    for d in range(n_days - 1):
        row_idx = 2 * d
        rows_ub.append(row_idx); cols_ub.append(num_x_vars + num_y_vars + d); data_ub.append(-1.0)
        for i_idx, occ in enumerate(occupancy_range):
            rows_ub.append(row_idx); cols_ub.append(num_x_vars + d * n_occ_vars + i_idx); data_ub.append(float(occ))
            rows_ub.append(row_idx); cols_ub.append(num_x_vars + (d + 1) * n_occ_vars + i_idx); data_ub.append(-float(occ))
        row_idx = 2 * d + 1
        rows_ub.append(row_idx); cols_ub.append(num_x_vars + num_y_vars + d); data_ub.append(-1.0)
        for i_idx, occ in enumerate(occupancy_range):
            rows_ub.append(row_idx); cols_ub.append(num_x_vars + d * n_occ_vars + i_idx); data_ub.append(-float(occ))
            rows_ub.append(row_idx); cols_ub.append(num_x_vars + (d + 1) * n_occ_vars + i_idx); data_ub.append(float(occ))

    A_ub = coo_matrix((data_ub, (rows_ub, cols_ub)), shape=(2 * (n_days - 1), num_vars))
    b_ub = np.zeros(2 * (n_days - 1))
    bounds = [(0, 1)] * (num_x_vars + num_y_vars) + [(0, max_occ)] * num_v_vars
    
    print(f"  Starting relaxation (fixed_occupancy={'None' if fixed_occupancy is None else 'Provided'})")
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return res, num_x_vars


def _fix_occupancy_bounds(assigned_days, family_size, n_days, min_occ, max_occ):
    occupancy = build_daily_occupancy(assigned_days, family_size, n_days)
    overfilled_days = [d for d in range(1, n_days + 1) if occupancy[d] > max_occ]
    underfilled_days = [d for d in range(1, n_days + 1) if occupancy[d] < min_occ]
    
    for over_d in overfilled_days:
        fams_in_day = np.where(assigned_days == over_d)[0]
        for f in fams_in_day:
            if occupancy[over_d] <= max_occ:
                break
            f_size = family_size[f]
            if underfilled_days:
                target_d = underfilled_days[0]
                if occupancy[target_d] + f_size <= max_occ:
                    assigned_days[f] = target_d
                    occupancy[over_d] -= f_size
                    occupancy[target_d] += f_size
                    if occupancy[target_d] >= min_occ:
                        underfilled_days.pop(0)
            else:
                for target_d in range(1, n_days + 1):
                    if occupancy[target_d] + f_size <= max_occ:
                        assigned_days[f] = target_d
                        occupancy[over_d] -= f_size
                        occupancy[target_d] += f_size
                        break

    for under_d in underfilled_days:
        while occupancy[under_d] < min_occ:
            if _stop: break
            moved = False
            for healthy_d in range(1, n_days + 1):
                if occupancy[healthy_d] > min_occ:
                    fams = np.where(assigned_days == healthy_d)[0]
                    for f in fams:
                        f_size = family_size[f]
                        if occupancy[healthy_d] - f_size >= min_occ and occupancy[under_d] + f_size <= max_occ:
                            assigned_days[f] = under_d
                            occupancy[healthy_d] -= f_size
                            occupancy[under_d] += f_size
                            moved = True
                            break
                if moved: break
            if not moved:
                break
                
    return assigned_days


def generate_lp_solution(family_size, choices, n_days, min_occ, max_occ):
    """
    Generate a solution using Iterative LP Relaxation and Rounding.
    """
    print("Calculating solution using Iterative LP Relaxation and Rounding...")
    n_families = len(family_size)
    
    res1, num_x_vars = _build_and_solve_lp(family_size, choices, n_days, min_occ, max_occ)
    if not res1.success:
        print(f"Warning: Pass 1 failed ({res1.message})")
        return choices[:, 0]
    
    x_sol1 = res1.x[:num_x_vars].reshape((n_families, n_days))
    assigned1 = np.argmax(x_sol1, axis=1) + 1
    occ1 = build_daily_occupancy(assigned1, family_size, n_days)
    
    res2, _ = _build_and_solve_lp(family_size, choices, n_days, min_occ, max_occ, fixed_occupancy=occ1)
    if not res2.success:
        print(f"Warning: Pass 2 failed ({res2.message}). Using Pass 1 results.")
        assigned_days = assigned1
    else:
        x_sol2 = res2.x[:num_x_vars].reshape((n_families, n_days))
        assigned_days = np.argmax(x_sol2, axis=1) + 1

    assigned_days = _fix_occupancy_bounds(assigned_days, family_size, n_days, min_occ, max_occ)
    return assigned_days


def run(submission, family_size, choices, n_days, min_occ, max_occ):
    """
    Wrapper for the algorithm runner.
    """
    if isinstance(submission, pd.DataFrame):
        initial_assignment = submission["assigned_day"].values.copy()
    else:
        initial_assignment = submission.copy()
    
    #Just to have an idea of the initial solution and to know when the algorithm started
    res_init = cost_function(initial_assignment, family_size, choices, n_days, min_occ, max_occ)
    if res_init:
        emit({"type": "start", "score": sum(res_init), "preference": res_init[0], "accounting": res_init[1]})

    assigned_days = generate_lp_solution(family_size, choices, n_days, min_occ, max_occ)
    
    score_parts = cost_function(assigned_days, family_size, choices, n_days, min_occ, max_occ)
    if score_parts is None:
        score = float('inf')
        pref, acc = 0, 0
    else:
        pref, acc = score_parts
        score = pref + acc
        
    emit({"type": "done", "score": score, "preference": pref, "accounting": acc, "iteration": 1})
        
    return assigned_days, score, []