from genetic import GeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np
from visual import visualize_params

path = "visual.png"

def fitness(params, max_surface_area=600):
    r, h = params[0], params[1]
    surface_area = 2*np.pi*(r*h+r**2)
    if r <= 0 or h <= 0 or surface_area > max_surface_area:
        return 0
    else:
        return np.pi*r**2*h

ga = GeneticAlgorithm(2, 1000, 0, 1)

print(ga.run(fitness, epochs=500, verbose=10, save_best_params=True))

visualize_params(path, fitness, ga.best_params_history,
                x_label="radius", y_label="height", 
                z_label="fitness (volume)", x_max=12)