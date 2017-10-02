#!/usr/bin/env python

"""
Demo of how to call low-level CUSOLVER wrappers to perform SVD decomposition.
"""

import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.cusolver as solver

handle = solver.cusolverDnCreate()
x = np.random.randn(1024,1024).astype(np.double)
x = x+x.T

# Need to reverse dimensions because CUSOLVER expects column-major matrices:
n, m = x.shape
x_gpu = gpuarray.to_gpu(x)

# Set up output buffers:
w = gpuarray.empty(n, dtype = x.dtype)

# Set up work buffers:
params = solver.cusolverDnCreateSyevjInfo()
solver.cusolverDnXsyevjSetTolerance(params, 1e-7)
solver.cusolverDnXsyevjSetMaxSweeps(params, 15)

lwork = solver.cusolverDnDsyevj_bufferSize(handle, 'CUSOLVER_EIG_MODE_VECTOR',
                                    'u', n, x_gpu.gpudata, m,
                                    w.gpudata, params)
print lwork
workspace_gpu = gpuarray.zeros(lwork, dtype = x.dtype)
info = gpuarray.zeros(1, dtype = np.int32)
# Compute:
solver.cusolverDnDsyevj(handle, 'CUSOLVER_EIG_MODE_VECTOR',
                       'u', n, x_gpu.gpudata, m,
                        w.gpudata, workspace_gpu.gpudata,
                        lwork, info.gpudata, params)

print solver.cusolverDnXsyevjGetSweeps(handle, params)
print solver.cusolverDnXsyevjGetResidual(handle, params)

# print info
solver.cusolverDnDestroySyevjInfo(params)
solver.cusolverDnDestroy(handle)

Q = x_gpu.get().T
print np.abs(np.dot(x, Q) - np.dot(Q, np.diag(w.get()))).max()
