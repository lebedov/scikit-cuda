#!/usr/bin/env python

"""
Demonstrates multiplication of several matrices on the GPU.
"""

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
import numpy as np
import scikits.cuda.linalg as linalg

linalg.init()

a = np.asarray(np.random.rand(8, 4), np.float32)
b = np.asarray(np.random.rand(4, 4), np.float32)
c = np.asarray(np.random.rand(4, 4), np.float32)

print 'Testing multiple matrix multiplication..'
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
c_gpu = gpuarray.to_gpu(c)
d_gpu = linalg.mdot(a_gpu, b_gpu, c_gpu)
print 'Success status: ', np.allclose(np.dot(a, np.dot(b, c)), d_gpu.get())
