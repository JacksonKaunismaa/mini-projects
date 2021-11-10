import numpy as np



def sample_points(p, num):
    return np.random.multivariate_normal(np.zeros(p), np.identity(p), num)


def reduce_dist(p_arr):
    return ((p_arr)**2).sum(axis=1)


def get_avg_dist(p_arr):
    dists = reduce_dist(p_arr)
    return np.sum(dists)/np.shape(p_arr)[0]

dim = int(input("Dimension: "))
points = 10000
N = 500
sum_total = 0.0
for _ in range(N):
    sum_total += get_avg_dist(sample_points(dim, points))
print("Average distance was", float(sum_total)/N)
