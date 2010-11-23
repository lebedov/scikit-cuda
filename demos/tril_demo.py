#!/usr/bin/env python

"""
Demonstrates how to extract the lower triangle of a matrix.
"""

import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import pycuda.gpuarray as gpuarray

import scikits.cuda.linalg as culinalg
culinalg.init()

print 'Testing lower triangle extraction..'
N = 10
a = np.asarray(np.random.rand(N, N), np.float32)
a_gpu = gpuarray.to_gpu(a)
b_gpu = culinalg.tril(a_gpu, pycuda.autoinit.device, False)
print 'Success status: ', np.allclose(b_gpu.get(), np.tril(a))
