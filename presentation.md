POPULATION_SIZE = 20  # number of candidate solutions kept at once
N_GENERATIONS = 500  # max evolution cycles
TOURNAMENT_SIZE = 3  # candidates compared in each selection draw
CROSSOVER_RATE = 0.80  # chance two parents breed (vs. cloning one parent)
MUTATION_RATE = 0.04  # per-family chance of a random day reassignment
ELITE_COUNT = 2  # top solutions copied unchanged to the next generation
PLATEAU_LIMIT = 40  # stop early if best score hasn't improved this many generations
LOCAL_SEARCH_EVERY = 5  # how often (in generations) to polish the best individual