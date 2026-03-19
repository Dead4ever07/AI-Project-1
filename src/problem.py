import numpy as np
import pandas as pd

GIFT_CARD  = [0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500]
BUFFET     = [0, 0, 9, 9, 9, 18, 18, 36, 36, 36, 36]
HELICOPTER = [0, 0, 0, 0, 0, 0, 0, 0, 0, 198, 398]


def load_data(filepath: str) -> dict:
    pass

def preference_cost(family_idx: int, assigned_day: int, data: dict) -> float:
    pass


def total_preference_cost(assignment: np.ndarray, data: dict) -> float:
    pass


def accounting_penalty(daily_occupancy: np.ndarray) -> float:
    pass


def total_cost(assignment: np.ndarray, daily_occupancy: np.ndarray, data: dict) -> float:
    pass


def is_valid(daily_occupancy: np.ndarray) -> bool:
    pass