#!/usr/bin/env python

"""
Demonstrate diagonal matrix creation on the GPU.
"""

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np

import scikits.cuda.linalg as linalg
linalg.init()

v = np.array([1, 2, 3, 4, 5, 6], np.float32)
v_gpu = gpuarray.to_gpu(v)
d_gpu = linalg.diag(v_gpu, pycuda.autoinit.device);

print 'Success status: ', np.all(d_gpu.get() == np.diag(v))

v = np.array([1j, 2j, 3j, 4j, 5j, 6j], np.complex64)
v_gpu = gpuarray.to_gpu(v)
d_gpu = linalg.diag(v_gpu, pycuda.autoinit.device);

print 'Success status: ', np.all(d_gpu.get() == np.diag(v))
