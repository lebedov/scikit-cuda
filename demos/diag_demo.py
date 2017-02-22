#!/usr/bin/env python

"""
Demonstrate diagonal matrix creation on the GPU.
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
    print('Testing real diagonal matrix creation for type ' + str(np.dtype(t)))
    v = np.array([1, 2, 3, 4, 5, 6], t)
    v_gpu = gpuarray.to_gpu(v)
    d_gpu = culinalg.diag(v_gpu)
    print('Success status: ', np.all(d_gpu.get() == np.diag(v)))
