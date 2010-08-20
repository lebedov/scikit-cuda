#!/usr/bin/env python

"""
Demonstrates how to use PyCUDA interface to CUFFT.
"""

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

import cuda_utils.fft as cu_fft

x = np.asarray(np.random.rand(256, 2), np.float32)
xf = np.fft.fft(x)
y = np.real(np.fft.ifft(xf))

x_gpu = gpuarray.to_gpu(x)

p_for = cu_fft.Plan(x_gpu.shape, np.float32, np.complex64)
xf_gpu = cu_fft.fft(x_gpu, p_for)

p_inv = cu_fft.Plan(x_gpu.shape, np.complex64, np.float32)
y_gpu = cu_fft.ifft(xf_gpu, p_inv, True)

print np.allclose(x, x_gpu.get(), 1e-3)
print np.allclose(y, y_gpu.get(), 1e-3)
