#!/usr/bin/env python

"""
Demo of how to call low-level CUSOLVER wrappers to perform SVD decomposition.
"""

import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.cusolver as solver

h = solver.cusolverDnCreate()
x = np.asarray([[1.80, 2.88, 2.05, -0.89],
                [5.25, -2.95, -0.95, -3.80], 
                [1.58, -2.69, -2.90, -1.04],
                [-1.11, -0.66, -0.59, 0.80]]).astype(np.float32)

# Need to reverse dimensions because CUSOLVER expects column-major matrices:
n, m = x.shape
x_gpu = gpuarray.to_gpu(x)

# Set up work buffers:
Lwork = solver.cusolverDnSgesvd_bufferSize(h, m, n)
workspace_gpu = gpuarray.zeros(Lwork, np.float32)
devInfo_gpu = gpuarray.zeros(1, np.int32)

# Set up output buffers:
s_gpu = gpuarray.zeros(min(m, n), np.float32)
u_gpu = gpuarray.zeros((m, m), np.float32)
vh_gpu = gpuarray.zeros((n, n), np.float32)

# Compute:
status = solver.cusolverDnSgesvd(h, 'A', 'A', m, n, x_gpu.gpudata, m, s_gpu.gpudata,
                                 u_gpu.gpudata, m, vh_gpu.gpudata, n,
                                 workspace_gpu.gpudata, Lwork, 0, devInfo_gpu.gpudata)

# Confirm that solution is correct by ensuring that the original matrix can be
# obtained from the decomposition:
print 'correct solution: ', np.allclose(x, np.dot(vh_gpu.get(), np.dot(np.diag(s_gpu.get()), u_gpu.get())), 1e-4)
solver.cusolverDnDestroy(h)
