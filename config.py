# -----------------------------------------------------------------------------
# Cost / Penalty table
# -----------------------------------------------------------------------------

PREFERENCE_PENALTIES = [
    (0, 0),  # choice_0 
    (50, 0),  # choice_1
    (50, 9),  # choice_2
    (100, 9),  # choice_3
    (200, 9),  # choice_4
    (200, 18),  # choice_5
    (300, 18),  # choice_6
    (300, 36),  # choice_7
    (400, 36),  # choice_8
    (500, 235),  # choice_9
    (500, 434),  # not in top 10
]


# -----------------------------------------------------------------------------
# Simulated Annealing
# -----------------------------------------------------------------------------

SA_INITIAL_TEMPERATURE = 5.0
SA_COOLING_SCHEDULE = 0.995
SA_PLATEAU_LIMIT = 100
SA_MAX_ITERATIONS = 500


# -----------------------------------------------------------------------------
# Hill Climbing
# -----------------------------------------------------------------------------

HC_MAX_ITERATIONS = 100
HC_SORT_FAMILIES = True 
N_RESTARTS = 5


# -----------------------------------------------------------------------------
# Genetic Algorithm
# -----------------------------------------------------------------------------

GA_POPULATION_SIZE = 15
GA_N_GENERATIONS = 500
GA_TOURNAMENT_SIZE = 3
GA_CROSSOVER_RATE = 0.5
GA_MUTATION_RATE = 0.02
GA_ELITE_COUNT = 5
GA_PLATEAU_LIMIT = 10
GA_LOCAL_SEARCH_EVERY = 3


# -----------------------------------------------------------------------------
# LP Solution
# -----------------------------------------------------------------------------

LP_DIFFERENCE_PENALTY_FIRST = 50.0
LP_DIFFERENCE_PENALTY_SECOND = 10.0

