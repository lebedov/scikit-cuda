#!/usr/bin/env python

"""
Demo of how to call low-level CUSOLVER wrappers to perform eigen decomposition
for Hermitian matrices.
"""

import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.cusolver as solver

handle = solver.cusolverDnCreate()
x = np.random.randn(1024,1024)+1j*np.random.rand(1024,1024)
x = x+x.conj().T

# Need to reverse dimensions because CUSOLVER expects column-major matrices:
n, m = x.shape
x_gpu = gpuarray.to_gpu(x.T.copy())

# Set up output buffers:
w = gpuarray.empty(n, dtype = np.double)

# Set up parameters
params = solver.cusolverDnCreateSyevjInfo()
solver.cusolverDnXsyevjSetTolerance(params, 1e-7)
solver.cusolverDnXsyevjSetMaxSweeps(params, 15)

# Set up work buffers:
lwork = solver.cusolverDnZheevj_bufferSize(handle, 'CUSOLVER_EIG_MODE_VECTOR',
                                    'u', n, x_gpu.gpudata, m,
                                    w.gpudata, params)

workspace_gpu = gpuarray.zeros(lwork, dtype = x.dtype)
info = gpuarray.zeros(1, dtype = np.int32)

# Compute:
solver.cusolverDnZheevj(handle, 'CUSOLVER_EIG_MODE_VECTOR',
                       'u', n, x_gpu.gpudata, m,
                        w.gpudata, workspace_gpu.gpudata,
                        lwork, info.gpudata, params)

# Print info
print(solver.cusolverDnXsyevjGetSweeps(handle, params))
print(solver.cusolverDnXsyevjGetResidual(handle, params))

# Destroy handle
solver.cusolverDnDestroySyevjInfo(params)
solver.cusolverDnDestroy(handle)

# Check error
Q = x_gpu.get().T
print('maximum error in A * Q - Q * Lambda is: %r' %
       np.abs(np.dot(x, Q) - np.dot(Q, np.diag(w.get()))).max())
