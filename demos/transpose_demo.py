#!/usr/bin/env python

"""
Demonstrates how to transpose matrices on the GPU.
"""

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy as np

import scikits.cuda.linalg as culinalg
import scikits.cuda.misc as cumisc
culinalg.init()

# Double precision is only supported by devices with compute
# capability >= 1.3:
import string
demo_types = [np.float32, np.complex64]
if cumisc.get_compute_capability() >= 1.3:
    demo_types.extend([np.float64, np.complex128])

for t in demo_types:
    print 'Testing transpose for type ' + str(np.dtype(t))
    if np.iscomplexobj(t()):
        b = np.array([[1j, 2j, 3j, 4j, 5j, 6j],
                      [7j, 8j, 9j, 10j, 11j, 12j]], t)
    else:
        a = np.array([[1, 2, 3, 4, 5, 6],
                      [7, 8, 9, 10, 11, 12]], t)
    a_gpu = gpuarray.to_gpu(a)
    at_gpu = culinalg.transpose(a_gpu, pycuda.autoinit.device)
    if np.iscomplexobj(t()):
        print 'Success status: ', np.all(np.conj(a.T) == at_gpu.get())
    else:
        print 'Success status: ', np.all(a.T == at_gpu.get())



