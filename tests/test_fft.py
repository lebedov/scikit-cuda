#!/usr/bin/env python

"""
Unit tests for scikits.cuda.fft
"""


from unittest import main, makeSuite, TestCase, TestSuite

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

import scikits.cuda.fft as fft
import scikits.cuda.misc as misc

atol_float32 = 1e-6
atol_float64 = 1e-8

class test_fft(TestCase):
    def setUp(self):
        self.N = 128
        
    def test_fft_float32_to_complex64(self):
        x = np.asarray(np.random.rand(self.N), np.float32)
        xf = np.fft.fft(x)
        x_gpu = gpuarray.to_gpu(x)
        xf_gpu = gpuarray.empty(self.N/2+1, np.complex64)
        plan = fft.Plan(x.shape, np.float32, np.complex64)
        fft.fft(x_gpu, xf_gpu, plan)
        assert np.allclose(xf[0:self.N/2+1], xf_gpu.get(), atol=atol_float32)

    def test_fft_float64_to_complex128(self):
        x = np.asarray(np.random.rand(self.N), np.float64)
        xf = np.fft.fft(x)
        x_gpu = gpuarray.to_gpu(x)
        xf_gpu = gpuarray.empty(self.N/2+1, np.complex128)
        plan = fft.Plan(x.shape, np.float64, np.complex128)
        fft.fft(x_gpu, xf_gpu, plan)
        assert np.allclose(xf[0:self.N/2+1], xf_gpu.get(), atol=atol_float64)

    def test_ifft_complex64_to_float32(self):
        x = np.asarray(np.random.rand(self.N), np.float32)
        xf = np.asarray(np.fft.fft(x), np.complex64)
        xf_gpu = gpuarray.to_gpu(xf[0:self.N/2+1])
        x_gpu = gpuarray.empty(self.N, np.float32)
        plan = fft.Plan(x.shape, np.complex64, np.float32)
        fft.ifft(xf_gpu, x_gpu, plan, True)
        assert np.allclose(x, x_gpu.get(), atol=atol_float32)

    def test_ifft_complex128_to_float64(self):
        x = np.asarray(np.random.rand(self.N), np.float64)
        xf = np.asarray(np.fft.fft(x), np.complex128)
        xf_gpu = gpuarray.to_gpu(xf[0:self.N/2+1])
        x_gpu = gpuarray.empty(self.N, np.float64)
        plan = fft.Plan(x.shape, np.complex128, np.float64)
        fft.ifft(xf_gpu, x_gpu, plan, True)
        assert np.allclose(x, x_gpu.get(), atol=atol_float64)
        
def suite():
    s = TestSuite()
    s.addTest(test_fft('test_fft_float32_to_complex64'))
    s.addTest(test_fft('test_ifft_complex64_to_float32'))
    if misc.get_compute_capability(pycuda.autoinit.device) >= 1.3:
        s.addTest(test_fft('test_fft_float64_to_complex128'))
        s.addTest(test_fft('test_ifft_complex128_to_float64'))
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
