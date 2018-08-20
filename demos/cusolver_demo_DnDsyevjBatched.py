#!/usr/bin/env python

"""
Demo of how to call low-level CUSOLVER wrappers to perform eigen decomposition
for a batch of small symmetric matrices.
"""

import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.cusolver as solver

handle = solver.cusolverDnCreate()
batchSize = 100
n = 9

A = np.empty((n*batchSize, n), dtype = np.double)

for i in range(batchSize):
    x = np.random.randn(n, n)
    x = x+x.T
    A[i*n:(i+1)*n, :] = x

x_gpu = gpuarray.to_gpu(A)

# Set up output buffers:
w_gpu = gpuarray.empty((batchSize, n), dtype = A.dtype)

# Set up parameters
params = solver.cusolverDnCreateSyevjInfo()
solver.cusolverDnXsyevjSetTolerance(params, 1e-7)
solver.cusolverDnXsyevjSetMaxSweeps(params, 15)

# Set up work buffers:
lwork = solver.cusolverDnDsyevjBatched_bufferSize(handle, 'CUSOLVER_EIG_MODE_VECTOR',
                                    'u', n, x_gpu.gpudata, n,
                                    w_gpu.gpudata, params, batchSize)

workspace_gpu = gpuarray.zeros(lwork, dtype = A.dtype)
info = gpuarray.zeros(batchSize, dtype = np.int32)

# Compute:
solver.cusolverDnDsyevjBatched(handle, 'CUSOLVER_EIG_MODE_VECTOR',
                       'u', n, x_gpu.gpudata, n,
                        w_gpu.gpudata, workspace_gpu.gpudata,
                        lwork, info.gpudata, params, batchSize)

# print info
tmp = info.get()
if any(tmp):
    print("the following job did not converge: %r" % np.nonzero(tmp)[0])

# Destroy handle
solver.cusolverDnDestroySyevjInfo(params)
solver.cusolverDnDestroy(handle)

Q = x_gpu.get()
W = w_gpu.get()
print('maximum error in A * Q - Q * Lambda is:')
for i in range(batchSize):
    q = Q[i*n:(i+1)*n,:].T.copy()
    x = A[i*n:(i+1)*n,:].copy()
    w = W[i, :].copy()
    print('{}th matrix %r'.format(i) % np.abs(np.dot(x, q) - np.dot(q, np.diag(w))).max())
