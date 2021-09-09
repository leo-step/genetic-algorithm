from genetic import GeneticAlgorithm
import numpy as np
from viz import visualize

def fitness(params, max_surface_area=600):
    r, h = params[0], params[1]
    surface_area = 2*np.pi*(r*h+r**2)
    if r <= 0 or h <= 0 or surface_area > max_surface_area:
        return 0
    else:
        return np.pi*r**2*h

ga = GeneticAlgorithm(2, 1000, 0, 1)

print(ga.run(fitness, epochs=10, verbose=1, save_best_params=True, 
                save_populations=True))

visualize("PUT PATH HERE", fitness, ga.best_param_history, 
                ga.population_history)