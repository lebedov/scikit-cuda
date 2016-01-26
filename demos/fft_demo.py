#!/usr/bin/env python

"""
Demonstrates how to use the PyCUDA interface to CUFFT to compute 1D FFTs.
"""
from __future__ import print_function

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

import skcuda.fft as cu_fft

print('Testing fft/ifft..')
N = 4096 * 16

x = np.asarray(np.random.rand(N), np.float32)
xf = np.fft.fft(x)
y = np.real(np.fft.ifft(xf))

x_gpu = gpuarray.to_gpu(x)
xf_gpu = gpuarray.empty(N//2+1, np.complex64)
plan_forward = cu_fft.Plan(x_gpu.shape, np.float32, np.complex64)
cu_fft.fft(x_gpu, xf_gpu, plan_forward)

y_gpu = gpuarray.empty_like(x_gpu)
plan_inverse = cu_fft.Plan(x_gpu.shape, np.complex64, np.float32)
cu_fft.ifft(xf_gpu, y_gpu, plan_inverse, True)

print('Success status: ', np.allclose(y, y_gpu.get(), atol=1e-6))

print('Testing in-place fft..')
x = np.asarray(np.random.rand(N) + 1j * np.random.rand(N), np.complex64)
x_gpu = gpuarray.to_gpu(x)

plan = cu_fft.Plan(x_gpu.shape, np.complex64, np.complex64)
cu_fft.fft(x_gpu, x_gpu, plan)

cu_fft.ifft(x_gpu, x_gpu, plan, True)

print('Success status: ', np.allclose(x, x_gpu.get(), atol=1e-6))
