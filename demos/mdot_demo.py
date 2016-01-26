#!/usr/bin/env python

"""
Demonstrates multiplication of several matrices on the GPU.
"""
from __future__ import print_function

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
import numpy as np

import skcuda.linalg as linalg
import skcuda.misc as cumisc
linalg.init()

# Double precision is only supported by devices with compute
# capability >= 1.3:
import string
demo_types = [np.float32, np.complex64]
if cumisc.get_compute_capability(pycuda.autoinit.device) >= 1.3:
    demo_types.extend([np.float64, np.complex128])

for t in demo_types:
    print('Testing multiple matrix multiplication for type ' + str(np.dtype(t)))
    if np.iscomplexobj(t()):
        a = np.asarray(np.random.rand(8, 4) + 1j * np.random.rand(8, 4), t)
        b = np.asarray(np.random.rand(4, 4) + 1j * np.random.rand(4, 4), t)
        c = np.asarray(np.random.rand(4, 4) + 1j * np.random.rand(4, 4), t)
    else:
        a = np.asarray(np.random.rand(8, 4), t)
        b = np.asarray(np.random.rand(4, 4), t)
        c = np.asarray(np.random.rand(4, 4), t)

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = gpuarray.to_gpu(c)
    d_gpu = linalg.mdot(a_gpu, b_gpu, c_gpu)
    print('Success status: ', np.allclose(np.dot(a, np.dot(b, c)), d_gpu.get()))
