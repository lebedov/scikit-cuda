#!/usr/bin/env python

"""
Demonstrates how to transpose matrices on the GPU.
"""

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy as np

import scikits.cuda.linalg as linalg
linalg.init()

a = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], np.float32)
a_gpu = gpuarray.to_gpu(a)
at_gpu = linalg.transpose(a_gpu, pycuda.autoinit.device)
print 'Success status: ', np.all(a.T == at_gpu.get())

b = np.array([[1j, 2j, 3j, 4j, 5j, 6j], [7j, 8j, 9j, 10j, 11j, 12j]], np.complex64)
b_gpu = gpuarray.to_gpu(b)
bt_gpu = linalg.transpose(b_gpu, pycuda.autoinit.device)
print 'Success status: ', np.all(np.conj(b.T) == bt_gpu.get())

