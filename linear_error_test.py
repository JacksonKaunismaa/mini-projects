import numpy as np
import matplotlib.pyplot as plt

SIGMA = 2.2

def nprint(an_arr, name):
    print(f"{name}:", an_arr, "shape:", np.shape(an_arr))
    print("#"*100)


def sample_random(p, num):
    return np.random.multivariate_normal(np.zeros(p), np.identity(p)*SIGMA, size=(num)).T

def get_error_linear(points, p):
    beta_rule = sample_random(p, 1)
    epsilon_error = sample_random(points, 1)
    x_inpt = sample_random(p, points)
    y_outpt = x_inpt.T.dot(beta_rule)
    y_hat_outpt = y_outpt + epsilon_error
    return beta_rule, x_inpt, y_outpt, y_hat_outpt, epsilon_error


def get_projection(x_arr, idx):
    x_idx = x_arr[:, idx]
    symmetric_x_arr = x_arr.T.dot(x_arr)
    nprint(symmetric_x_arr, "symm")
    symmetric_x_arr = np.linalg.inv(symmetric_x_arr)
    projection_matrix = x_arr.dot(symmetric_x_arr)
    error_sum_vector = projection_matrix.dot(x_idx)
    return error_sum_vector

#def reconstruct_exact(x_arr, idx, beta_rule, error_sum_vector, errors):
#    x_idx = x_arr[idx]
#    guesstimate = x_idx.dot(beta_rule)
#    error_sum_vector_errors = error_sum_vector * errors
#    sum_errors = error_sum_vector_errors.sum()
#    reconstruction = sum_errors + guesstimate
#    return reconstruction, guesstimate, sum_errors
b, x, y, y_hat, eps = get_error_linear(5, 2)
proj = get_projection(x, 0)
#y_hopefully, guess, l_sum = reconstruct_exact(x, 0, b, proj, eps)
nprint(b, "b")
nprint(x.T, "x.T")
nprint(y, "y")
nprint(y_hat, "y_hat")
nprint(eps, "eps")
nprint(proj, "proj")
#nprint(y_hopefully, "y_hopefully")
#nprint(l_sum, "l_sum")
#nprint(guess, "guess")
#nprint(b, "b")
#plt.scatter(x, y_hat)
#plt.scatter(x, y)
#plt.show()
