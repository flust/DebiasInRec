import numpy as np

raw_matrix = np.loadtxt('train.ascii')

sum = raw_matrix.sum(axis = 0)
print(sum[5:7])