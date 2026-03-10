# AI-Project-1 - Santa's Workshop Tour

[Problem Link](https://www.kaggle.com/competitions/santa-workshop-tour-2019/overview)

1. Design the frontend (Recomended Flask)
    - It shall have a dashbord like GUI with a modern look
    - Further information representation shall be discussed

1. Represent the problem in a state class
    - It shall include every detailed so that an evaluation function can accuratly give us a value

1. Create an evaluation function that gives us a "score" like value
    - It shall be used to infer if the new result is better than the previous one
    - It shall be made as efficient as possible
    - Maybe good to utilize the previous state as a base to recalculate the score

1. Creation of Test cases to the problem
    - It shall be created some test cases to the problem
    - Google already provided a 5000 famillies problem set
    - It shall be create another 5 small datasets(around 10-20 families) and 5 medium datasets(50-150 families)

1. Algorithms to be implemented:
    1. Meta-heuristics:
        - We shall think of a good/strong heuristc for the project
        - Choose the best one, maybe starting from reading the [Wiki](https://en.wikipedia.org/wiki/Metaheuristic)
    1. Hill-climbing:
        - It is a local search algorithm to find the optimal solution
        - We can do a multi-sorce, in order to improve the overall solution
        - Sources: [GeeksForGeeks](https://www.geeksforgeeks.org/artificial-intelligence/introduction-hill-climbing-artificial-intelligence/), [Wiki](https://en.wikipedia.org/wiki/Hill_climbing)
    1. Simulated Annealing:
        - Global Heuristic Algorithm
        - Good source might be the [Wiki](https://en.wikipedia.org/wiki/Simulated_annealing)
    1. Genetic Algorithms:
        - Generates multiple random solutions and then motates them, chooses the best N solutions and repeats the process
        - A good idea of the algorithm might be the [Wiki](https://en.wikipedia.org/wiki/Genetic_algorithm)