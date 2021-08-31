from genetic import GeneticAlgorithm
import numpy as np

def fitness(params):
    r, h = params[0], params[1]
    surface_area = 2*np.pi*(r*h+r**2)
    if r <= 0 or h <= 0 or surface_area > 600:
        return 0
    else:
        return np.pi*r**2*h

ga = GeneticAlgorithm(2, 1000, 0, 5)
print(ga.run(fitness, epochs=500, verbose=50))
    