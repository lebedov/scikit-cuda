#!/usr/bin/env python

"""
Demonstrates multiplication of two matrices on the GPU.
"""
from __future__ import print_function

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np

import skcuda.linalg as culinalg
import skcuda.misc as cumisc
culinalg.init()

# Double precision is only supported by devices with compute
# capability >= 1.3:
import string
demo_types = [np.float32, np.complex64]
if cumisc.get_compute_capability(pycuda.autoinit.device) >= 1.3:
    demo_types.extend([np.float64, np.complex128])

for t in demo_types:
    print('Testing matrix multiplication for type ' + str(np.dtype(t)))
    if np.iscomplexobj(t()):
        a = np.asarray(np.random.rand(10, 5) + 1j * np.random.rand(10, 5), t)
        b = np.asarray(np.random.rand(5, 5) + 1j * np.random.rand(5, 5), t)
        c = np.asarray(np.random.rand(5, 5) + 1j * np.random.rand(5, 5), t)
    else:
        a = np.asarray(np.random.rand(10, 5), t)
        b = np.asarray(np.random.rand(5, 5), t)
        c = np.asarray(np.random.rand(5, 5), t)

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = gpuarray.to_gpu(c)

    temp_gpu = culinalg.dot(a_gpu, b_gpu)
    d_gpu = culinalg.dot(temp_gpu, c_gpu)
    temp_gpu.gpudata.free()
    del(temp_gpu)
    print('Success status: ', np.allclose(np.dot(np.dot(a, b), c), d_gpu.get()))

    print('Testing vector multiplication for type ' + str(np.dtype(t)))
    if np.iscomplexobj(t()):
        d = np.asarray(np.random.rand(5) + 1j * np.random.rand(5), t)
        e = np.asarray(np.random.rand(5) + 1j * np.random.rand(5), t)
    else:
        d = np.asarray(np.random.rand(5), t)
        e = np.asarray(np.random.rand(5), t)

    d_gpu = gpuarray.to_gpu(d)
    e_gpu = gpuarray.to_gpu(e)

    temp = culinalg.dot(d_gpu, e_gpu)
    print('Success status: ', np.allclose(np.dot(d, e), temp))
