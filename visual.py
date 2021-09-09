import numpy as np
import matplotlib.pyplot as plt

def visualize(path, fitness, best_params_history, x_label="x", y_label="x",
            z_label="fitness", num_points=1000, x_min=0.0, x_max=20.0, 
            y_min=0.0, y_max=20.0, color="red", azim=20.0, elev=230.0):
    """
    Produces a 3D visualization of the fitness landscape with points 
    indicating the history of the best set of parameters. Only works
    if there are two parameters per individual.

    Parameters
    ----------
    path : string
            Output file path for the visualization.
    fitness : Callable
            The fitness function.
    best_params_history : np.ndarray
            A 2D array of parameters.
    x_label : string (default "x")
            The x-axis label (parameter 0).
    y_label : string (default "y")
            The y-axis label (parameter 1).
    z_label : string (default "fitness")
            The z-axis label (fitness value).
    num_points : int (default 1000)
            Number of points to fit across the x and y axes.
            Determines the precision of the fitness landscape.
    x_min : float (default 0.0)
            Minimum value for the x-axis.
    x_max : float (default 20.0)
            Maximum value for the x-axis.
    y_min : float (default 0.0)
            Minimum value for the y-axis.
    y_max : float (default 20.0)
            Maximum value for the y-axis.
    color : string (default "red")
            Color of the parameter points.
    azim : float (default 20.0)
            Plot azimuth angle in degrees.
    elev : float (default 230.0)
            Plot elevation angle in degrees.
    """
    # create a contour map of possible parameter values and
    # the corresponding fitness function values
    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)
    z = []
    for j in range(num_points):
        for i in range(num_points):
            z.append(fitness([x[i], y[j]]))
    Z = np.array(z).reshape((num_points, num_points))

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(azim, elev)

    ax.contour3D(X, Y, Z, 50)

    # set labels
    ax.set_xlabel(x_label)
    ax.axes.set_xlim(x_min, x_max)
    ax.set_ylabel(y_label)
    ax.axes.set_ylim(y_min, y_max)
    ax.set_zlabel(z_label)

    # add points representing the history of best parameters
    ax.scatter3D(best_params_history[:,0], best_params_history[:,1], 
        np.apply_along_axis(fitness, 1, best_params_history), 
        color=color, depthshade=0)

    # save figure
    plt.savefig(path)