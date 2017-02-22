#!/usr/bin/env python

"""
Demo of how to call low-level CUSOLVER wrappers to perform LU decomposition.
"""

import numpy as np
import scipy.linalg
import scipy as sp
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.cusolver as solver

h = solver.cusolverDnCreate()
x = np.asarray([[1.80, 2.88, 2.05, -0.89],
                [5.25, -2.95, -0.95, -3.80], 
                [1.58, -2.69, -2.90, -1.04],
                [-1.11, -0.66, -0.59, 0.80]]).astype(np.float32)

# Need to copy transposed matrix because T only returns a view:
m, n = x.shape
x_gpu = gpuarray.to_gpu(x.T.copy())

# Set up work buffers:
Lwork = solver.cusolverDnSgetrf_bufferSize(h, m, n, x_gpu.gpudata, m)
workspace_gpu = gpuarray.zeros(Lwork, np.float32)
devipiv_gpu = gpuarray.zeros(min(m, n), np.int32)
devinfo_gpu = gpuarray.zeros(1, np.int32)

# Compute:
solver.cusolverDnSgetrf(h, m, n, x_gpu.gpudata, m, workspace_gpu.gpudata, devipiv_gpu.gpudata, devinfo_gpu.gpudata)

# Confirm that solution is correct by checking against result obtained with
# scipy; set dimensions of computed lower/upper triangular matrices to facilitate
# comparison if the original matrix was not square:
l_cuda = np.tril(x_gpu.get().T, -1)
u_cuda = np.triu(x_gpu.get().T)
if m < n:
    l_cuda = l_cuda[:, :m]
else:
    u_cuda = u_cuda[:n, :]
p, l, u = sp.linalg.lu(x)

# Only check values in lower triangle starting from first off-diagonal:
print 'lower triangular matrix is correct: ', \
    np.allclose(np.tril(l, -1), l_cuda)
print 'upper triangular matrix is correct: ', \
    np.allclose(np.triu(u), u_cuda)
solver.cusolverDnDestroy(h)
