import numpy as np
import matplotlib.pyplot as plt

def visualize_params(path, fitness, params_history, 
                x_label="x", y_label="x", z_label="fitness", 
                num_points=1000, x_min=0, x_max=20, y_min=0, y_max=20):
    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)
    z = []
    for j in range(num_points):
        for i in range(num_points):
            z.append(fitness(np.array([x[i], y[j]])))
    Z = np.array(z).reshape((num_points, -1))

    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50)
    ax.set_xlabel(x_label)
    ax.axes.set_xlim(x_min, x_max)
    ax.set_ylabel(y_label)
    ax.axes.set_ylim(y_min, y_max)
    ax.set_zlabel(z_label)

    ax.scatter3D(params_history[:,0], params_history[:,1], 
        np.apply_along_axis(fitness, 1, params_history), color="red", 
        depthshade=0)

    ax.view_init(30, 240)

    plt.savefig(path)