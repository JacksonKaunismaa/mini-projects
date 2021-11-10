import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
import matplotlib.pyplot as plt

a = 1
b = 100

def rosen(x, y):
    return np.square(a-x) + b * np.square((y - np.square(x)))

def rosen_vec(x):
    return rosen(x[0], x[1])

def grad(x,y):
    square_diff = np.square(x) - y
    grad_x = 2*(x-a) + 4*b*x*square_diff
    grad_y = -2*b*square_diff
    return grad_x, grad_y

def grad_vec(x):
    actual_grad = np.array(grad(x[0], x[1]))
    return actual_grad

def overall(x):
    return rosen_vec(x), 100.*grad_vec(x)

count = 0
def callback_func(x):
    global count
    if count % 10 == 0:
        print(f"On iter {count}, value was {rosen_vec(x)}, gradients were {grad_vec(x)}, and x was {x}")
    count += 1

starting = np.random.uniform(low=-100.0, high=100.0, size=(2,))
result = minimize(overall, starting,
                  method='L-BFGS-B',
                  jac=True,
                  options={"maxiter": 10000},
                  bounds=Bounds(-1000,1000),
                  callback=callback_func)
final = result.x
print(result.message)
print(result.success)
print(f"Final value was {rosen_vec(final)}")
