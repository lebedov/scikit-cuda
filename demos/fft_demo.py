#!/usr/bin/env python

"""
Demonstrates how to use PyCUDA interface to CUFFT.
"""

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

import cuda_utils.fft as cu_fft

# Perform usual forward/reverse transformations:
N = 4096*16

x = np.asarray(np.random.rand(N), np.float32)
xf = np.fft.fft(x)
y = np.real(np.fft.ifft(xf))

x_gpu = gpuarray.to_gpu(x)

p_for = cu_fft.Plan(x_gpu.shape, np.float32, np.complex64)
xf_gpu = cu_fft.fft(x_gpu, p_for)

p_inv = cu_fft.Plan(x_gpu.shape, np.complex64, np.float32)
y_gpu = cu_fft.ifft(xf_gpu, p_inv, True)

print np.allclose(x, x_gpu.get(), atol=1e-6)
print np.allclose(y, y_gpu.get(), atol=1e-6)

# Perform in-place transformations:
x = np.asarray(np.random.rand(N)+1j*np.random.rand(N), np.complex64)
xf = np.fft.fft(x)
y = np.fft.ifft(xf)

x_gpu = gpuarray.to_gpu(x)

p = cu_fft.Plan(x_gpu.shape, np.complex64, np.complex64)
cu_fft.fft(x_gpu, p, inplace=True)

cu_fft.ifft(x_gpu, p, True, inplace=True)

print np.allclose(x, x_gpu.get(), atol=1e-6)
