from genetic import GeneticAlgorithm
import numpy as np
from visual import visualize

# declare the fitness function
def fitness(params, max_surface_area=600.0):
    """
    Returns a value for fitness given an array of parameters.
    The fitness of a cylinder is defined as 0.0 if the radius or
    height is zero or if its surface area exceeds the maximum value.
    Otherwise, the fitness value returned is the cylinder's volume.
    
    Parameters
    ----------
    params : List
            An individual's parameters.
    max_surface_area : float (default 600.0):
            The maximum surface area allowed for the cylinder.
    
    Returns
    -------
    fitness : float
            The fitness value for the individual.
    """
    r, h = params[0], params[1]
    surface_area = 2*np.pi*(r*h+r**2)
    if r <= 0 or h <= 0 or surface_area > max_surface_area:
        return 0.0
    else:
        return np.pi*r**2*h

# initialize genetic algorithm
ga = GeneticAlgorithm(num_param=2, size=1000, min_param=0, max_param=1)

# run the genetic algorithm and print the best parameters found
best_params = ga.run(fitness, epochs=1000, verbose=10, history=True)
print("Best parameters: {}".format(best_params))

# create a visualization of the fitness landscape and parameter history
visualize("fitness.png", fitness, ga.best_params_history, x_label="radius", 
        y_label="height", z_label="fitness (volume)", x_max=12, y_max=12)