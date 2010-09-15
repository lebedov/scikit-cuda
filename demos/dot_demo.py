#!/usr/bin/env python

"""
Demonstrates how to use the PyCUDA dot product.
"""

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np
import scikits.cuda.linalg as linalg

linalg.init()

a = np.asarray(np.random.rand(10, 5), np.complex128)
b = np.asarray(np.random.rand(5, 5), np.complex128)
c = np.asarray(np.random.rand(5, 5), np.complex128)

a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
c_gpu = gpuarray.to_gpu(c)

temp_gpu = linalg.dot(a_gpu, b_gpu)
d_gpu = linalg.dot(temp_gpu, c_gpu)
temp_gpu.gpudata.free()
del(temp_gpu)
print 'Success status: ', np.allclose(np.dot(np.dot(a, b), c) , d_gpu.get())

d = np.asarray(np.random.rand(5), np.complex64)
e = np.asarray(np.random.rand(5), np.complex64)

d_gpu = gpuarray.to_gpu(d)
e_gpu = gpuarray.to_gpu(e)

temp = linalg.dot(d_gpu, e_gpu)
print 'Success status: ', np.allclose(np.dot(d, e), temp)
