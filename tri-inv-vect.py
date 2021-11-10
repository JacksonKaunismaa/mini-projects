import numpy as np


def mk_tri(mat):
    mat_cpy = np.zeros_like(mat)
    for i, row in enumerate(mat):
        mat_cpy[i] = np.concatenate((row[:1+i], np.zeros(row.shape[0] - i - 1)))
    return mat_cpy


def tri_inverse(mat, vect):
    vect_cpy = np.zeros_like(vect)
    for t, row in enumerate(mat):
        z_val = vect[t]
        for i in range(t):
            z_val -= mat[t,i]*vect_cpy[i]
        z_val /= mat[t,t]
        vect_cpy[t] = z_val
    return vect_cpy

my_mat = np.random.random((3,3))
my_vec = np.random.random(3)
tri_mat = mk_tri(my_mat)
res = np.matmul(tri_mat, my_vec)
inv_res = tri_inverse(tri_mat, res)
print(my_vec)
print(inv_res)
