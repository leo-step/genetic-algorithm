import numpy as np

class GeneticAlgorithm:
    def __init__(self, num_param, size=1000, pct_best=0.01, min_param=-1.0, max_param=1.0):
        """Initializes population of given size with random parameters.
        
        Arguments:\n
        num_param -- number of parameters per individual\n
        size -- number of individuals in the population (default 1000)\n
        pct_best -- percentage of individuals selected to continue to the next generation (default 0.01)\n
        min_param -- minimum possible value for initial parameters (default -1.0)\n
        max_param -- maximum possible value for initial parameters (default 1.0)\n
        """
        self.population = np.random.uniform(min_param, max_param, (size, num_param))
        self.num_param = num_param
        self.size = size
        self.n_best = max(int(size*pct_best), 1) # a minimum of one individual is needed to 
                                               # create the next generation
        
    def _select_n_best(self, population, fitness, n, maximize=True):
        scores = np.apply_along_axis(fitness, 1, population)
        if maximize:
            return population[np.argsort(scores)[::-1][:n]]
        else:
            return population[np.argsort(scores)[:n]]

    def _crossover(self, population, p_cross=0.5):
        for index in range(self.num_param):
            if np.random.rand() < p_cross:
                np.random.shuffle(population[:,index])
        return population

    def _mutate(self, params, p_mutate=0.15, max_mutation=0.1):
        for i in range(self.num_param):
            if np.random.rand() < p_mutate:
                params[i] += np.random.uniform(-max_mutation, max_mutation)
        return params

    def run(self, fitness, epochs=1000, p_cross=0.5, p_mutate=0.15, max_mutation=0.1, maximize=True, visualize=False):
        """Runs the genetic algorithm for a given number of epochs.
        
        Arguments:\n
        fitness -- fitness function
        epochs -- number of generations to simulate (default 1000)
        p_cross -- probability of a crossover between individuals for every parameter (default 0.5)
        p_mutate -- probability of a mutation occuring in a parameters (default 0.15)
        max_mutation -- maximum change in a parameter when a mutation occurs
        maximize -- maximize/minimize the fitness function if set to True/False (default True)
        visualize -- WRITE THIS
        """
        for i in range(epochs):
            best = self._select_n_best(self.population, fitness, self.n_best)
            print(fitness(best[0]))
            cross = self._crossover(best)
            self.population = np.tile(cross, (int(self.size/self.n_best), 1))
            self.population = np.apply_along_axis(self._mutate, 1, self.population)





