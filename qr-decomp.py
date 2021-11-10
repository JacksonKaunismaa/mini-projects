import numpy as np
from scipy.linalg import qr as qr_scipy

def gram_schmidt(mat):
    grams = []
    mat_cpy = mat.copy().T
    new_mat = np.zeros_like(mat)
    for row_idx, row in enumerate(mat_cpy):
        row_cpy = row.copy()
        for gram in grams:
            row_cpy -= gram * row_cpy.dot(gram) #/ gram.dot(gram)  #gram.dot(gram) should always be 1
        row_cpy /= np.sqrt(row_cpy.dot(row_cpy))
        grams.append(row_cpy)
        new_mat[row_idx] = row_cpy
    return new_mat.T

def QR(mat):
    Q_part = gram_schmidt(mat)
    R_part = np.matmul(Q_part.T, mat)
    r_part_better = R_part
    r_part_better = np.zeros_like(R_part)   # somehow this give both better loss and more accurate form of R
    for i, row in enumerate(R_part):
        r_part_better[i] = np.concatenate((np.zeros(i), row[i:]))
#    loss = np.linalg.norm(mat - np.matmul(Q_part, r_part_better))
    loss = 0
    return Q_part, r_part_better, loss

def mat_error(mat, qmat, rmat):
    return np.linalg.norm(mat - np.matmul(qmat, rmat))

#my_mat = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]]).astype(np.float32)
#my_mat = np.random.random((2,2))
my_mat = np.random.random((2000,2000))
#schmidt = gram_schmidt(my_mat)
import time
start = time.time()
q, r, diff = QR(my_mat)
mine = time.time() - start
start = time.time()
q_s, r_s = qr_scipy(my_mat)
theirs = time.time() - start
#print(my_mat)
#print(q)
#print(r)
#print(diff)
#print(np.matmul(q,r))
#print(q_s)
#print(r_s)
print("my loss:", mat_error(my_mat, q, r), "time:", mine)
print("scipy loss", mat_error(my_mat, q_s, r_s), "time:", theirs)
