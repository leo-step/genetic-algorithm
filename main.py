from genetic import GeneticAlgorithm

def fitness(coef):
    l, w, h = coef[0], coef[1], coef[2]
    surface_area = 2*(w*l+h*l+h*w)
    if l <= 0 or w <= 0 or h <= 0 or surface_area > 600:
        return 0
    else:
        return l*w*h

ga = GeneticAlgorithm(3, 1000, 0, 5)
print(ga.run(fitness, epochs=200, verbose=True))