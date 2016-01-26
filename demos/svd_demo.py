#!/usr/bin/env python

"""
Demonstrates computation of the singular value decomposition on the GPU.
"""
from __future__ import print_function

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy as np

import skcuda.linalg as culinalg
import skcuda.misc as cumisc
culinalg.init()

# Double precision is only supported by devices with compute
# capability >= 1.3:
import string
import scikits.cuda.cula as cula
demo_types = [np.float32, np.complex64]
if cula._libcula_toolkit == 'premium' and \
        cumisc.get_compute_capability(pycuda.autoinit.device) >= 1.3:
    demo_types.extend([np.float64, np.complex128])

for t in demo_types:
    print('Testing svd for type ' + str(np.dtype(t)))
    a = np.asarray((np.random.rand(50, 50) - 0.5) / 10, t)
    a_gpu = gpuarray.to_gpu(a)
    u_gpu, s_gpu, vh_gpu = culinalg.svd(a_gpu)
    a_rec = np.dot(u_gpu.get(), np.dot(np.diag(s_gpu.get()), vh_gpu.get()))

    print('Success status: ', np.allclose(a, a_rec, atol=1e-3))
    print('Maximum error: ', np.max(np.abs(a - a_rec)))
    print('')
