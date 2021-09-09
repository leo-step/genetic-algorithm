import numpy as np

def visualize(path, fitness, best_param_history, population_history, 
                num_points=1000, x_min=0, x_max=20, y_min=0, y_max=20):
    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)

    X, Y = np.meshgrid(x, y)