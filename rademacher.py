import numpy as np
import matplotlib.pyplot as plt
import colorsys

def rademacher(A_vects):
    # rows of A are the elements
    ndim = A_vects.shape[1]
    tot = 0
    sigma_vect = np.ones(ndim)
    for sigma in range(1<<ndim):
        for i in range(ndim):
            if (1 << i) & sigma:
                sigma_vect[-i] = -1
            else:
                sigma_vect[-i] = 1
        tot += np.max(A_vects.dot(sigma_vect))
    return tot/(ndim*(2**ndim))

def plot_all_points(all_points, min_complexity, max_complexity):
    for groups,complexities in all_points:
        for points,complexity in zip(groups, complexities):
            color = colorsys.hsv_to_rgb((complexity-min_complexity)/(max_complexity-min_complexity)*360, .8, .8)
            plt.scatter(points[:,0], points[:,1], s=20.0, color=color)
    plt.show()

def circle_simulation():
    radii = np.linspace(0.1, 5, 20)
    splits = 10
    all_points = []
    min_complexity = float("inf")
    max_complexity = -float("inf")
    for radius in radii:
        angles = np.linspace(0, 2*np.pi, int(200*radius))
        points = radius*np.array([np.cos(angles), np.sin(angles)]).T  # num_points x 2
        group_size = points.shape[0]//splits
        groups = [points[i:i+group_size] for i in range(0,points.shape[0],group_size)]
        complexities = [rademacher(group) for group in groups]
        min_complexity = min(min_complexity, min(complexities))
        max_complexity = max(max_complexity, max(complexities))
        all_points.append((groups, complexities))
    return all_points, min_complexity, max_complexity

def random_simulation():
    clusters = 20
    num_points = 15
    means = np.random.uniform(-30, 30, (clusters,2))
    variances = np.random.exponential(1/4, (clusters, 2, 2))
    all_points = []
    min_complexity = float("inf")
    max_complexity = -float("inf")
    for mu,cov in zip(means, variances):
        points = np.random.multivariate_normal(mu, cov, num_points)
        complexity = rademacher(points)
        min_complexity = min(min_complexity, complexity)
        max_complexity = max(max_complexity, complexity)
        all_points.append(([points], [complexity]))
    return all_points, min_complexity, max_complexity

plot_all_points(*random_simulation())
