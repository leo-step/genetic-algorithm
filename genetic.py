import time
import numpy as np

class GeneticAlgorithm:
    def __init__(self, num_param, size=1000, min_param=-1.0, max_param=1.0):
        """
        Initializes a population of a given size with parameters randomly generated within a range.
        
        Parameters
        ----------
        num_param : int
                The number of parameters per individual.
        size : int (default 1000)
                The number of individuals in the population.
        min_param : float (default -1.0)
                The minimum possible value for initial parameters.
        max_param : float (default 1.0)
                The maximum possible value for initial parameters.
        """
        self.population = np.random.uniform(min_param, max_param, (size, num_param))
        self.num_param = num_param
        self.size = size
        
    def _select_n_best(self, population, fitness, n, maximize):
        """
        Selects n best individuals from the population based on the fitness function.

        Parameters
        ----------
        population : np.ndarray
                An 2D array of individuals and their parameters.
        fitness : Callable
                The fitness function.
        n : int
                The number of top-performing individuals to be selected for the next generation.
        maximize: bool
                Maximize the fitness function if True and minimize the function if False.
        
        Returns
        -------
        best : np.ndarray
                A 2D array of the fittest individuals and their parameters.
        """
        scores = np.apply_along_axis(fitness, 1, population)
        if maximize:
            return population[np.argsort(scores)[::-1][:n]]
        else:
            return population[np.argsort(scores)[:n]]

    def _crossover(self, population, p_cross):
        """
        Shuffles parameters between individuals with a certain probability.

        Parameters
        ----------
        population : np.ndarray
                An 2D array of individuals and their parameters.
        p_cross : float
                The probability of a crossover between individuals for every parameter.

        Returns
        -------
        population : np.ndarray
                An 2D array of individuals and their crossed parameters.
        """
        for index in range(self.num_param):
            if np.random.rand() < p_cross:
                np.random.shuffle(population[:,index])
        return population

    def _mutate(self, params, p_mutate, max_mutation):
        """
        Mutates parameters with a certain probability.

        Parameters
        ----------
        params : np.ndarray
                An array of parameters.
        p_mutate : float
                The probability of a mutation occuring in a parameter.
        max_mutation : float
                The maximum change in a parameter when a mutation occurs.

        Returns
        -------
        params : np.ndarray
                An array of mutated parameters.
        """
        for i in range(self.num_param):
            if np.random.rand() < p_mutate:
                params[i] += np.random.uniform(-max_mutation, max_mutation)
        return params

    def run(self, fitness, epochs=1000, p_cross=0.5, p_mutate=0.15, pct_best=0.01, max_mutation=0.1, maximize=True):
        """
        Runs the genetic algorithm for a given number of epochs.
        
        Parameters
        ----------
        fitness : Callable
                The fitness function.
        epochs : int (default 1000)
                The number of generations to simulate.
        p_cross : float (default 0.5)
                The probability of a crossover between individuals for every parameter.
        p_mutate : float (default 0.15)
                The probability of a mutation occuring in a parameters.
        pct_best : float (default 0.01)
                The percentage of individuals selected to continue to the next generation.
        max_mutation : float (default 0.1)
                The maximum change in a parameter when a mutation occurs.
        maximize : bool (default True)
                Maximize the fitness function if True and minimize the function if False.

        Returns
        -------
        best_params : np.ndarray
                The best parameters found over the given number of epochs.
        """
        start = time.time()
        n_best = max(int(self.size*pct_best), 1)
        n_tiles = int(self.size/n_best)
        for i in range(epochs):
            best = self._select_n_best(self.population, fitness, n_best, maximize)
            print("Epoch {:5d} | Fitness: {:.6f} | Best Params: {}".format(i+1, fitness(best[0]), best[0]))
            cross = self._crossover(best, p_cross)
            self.population = np.tile(cross, (n_tiles, 1))
            self.population[n_best:] = np.apply_along_axis(self._mutate, 1, self.population[n_best:], p_mutate, max_mutation)
        print("Running time: {:.3f} seconds".format(time.time()-start))
        return self._select_n_best(self.population, fitness, 1, maximize)[0]