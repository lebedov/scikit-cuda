#!/usr/bin/env python

"""
Demonstrates how to extract the lower triangle of a matrix.
"""
from __future__ import print_function

import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import pycuda.gpuarray as gpuarray

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
    print('Testing lower triangle extraction for type ' + str(np.dtype(t)))
    N = 10
    if np.iscomplexobj(t()):
        a = np.asarray(np.random.rand(N, N), t)
    else:
        a = np.asarray(np.random.rand(N, N) + 1j * np.random.rand(N, N), t)
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = culinalg.tril(a_gpu, False)
    print('Success status: ', np.allclose(b_gpu.get(), np.tril(a)))
