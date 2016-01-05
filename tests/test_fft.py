#!/usr/bin/env python

"""
Unit tests for scikits.cuda.fft
"""

from __future__ import division

from unittest import main, makeSuite, TestCase, TestSuite

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy as np

import skcuda.fft as fft
import skcuda.misc as misc

atol_float32 = 1e-6
atol_float64 = 1e-8

class test_fft(TestCase):
    def setUp(self):
        np.random.seed(0) # for reproducible tests
        self.N = 8
        self.M = 4
        self.B = 3

    def test_fft_float32_to_complex64_1d(self):
        x = np.asarray(np.random.rand(self.N), np.float32)
        xf = np.fft.rfftn(x)
        x_gpu = gpuarray.to_gpu(x)
        xf_gpu = gpuarray.empty(self.N//2+1, np.complex64)
        plan = fft.Plan(x.shape, np.float32, np.complex64)
        fft.fft(x_gpu, xf_gpu, plan)
        assert np.allclose(xf, xf_gpu.get(), atol=atol_float32)

    def test_fft_float32_to_complex64_2d(self):
        x = np.asarray(np.random.rand(self.N, self.M), np.float32)
        xf = np.fft.rfftn(x)
        x_gpu = gpuarray.to_gpu(x)
        xf_gpu = gpuarray.empty((self.N, self.M//2+1), np.complex64)
        plan = fft.Plan(x.shape, np.float32, np.complex64)
        fft.fft(x_gpu, xf_gpu, plan)
        assert np.allclose(xf, xf_gpu.get(), atol=atol_float32)

    def test_batch_fft_float32_to_complex64_1d(self):
        x = np.asarray(np.random.rand(self.B, self.N), np.float32)
        xf = np.fft.rfft(x, axis=1)
        x_gpu = gpuarray.to_gpu(x)
        xf_gpu = gpuarray.empty((self.B, self.N//2+1), np.complex64)
        plan = fft.Plan(x.shape[1], np.float32, np.complex64, batch=self.B)
        fft.fft(x_gpu, xf_gpu, plan)
        assert np.allclose(xf, xf_gpu.get(), atol=atol_float32)

    def test_batch_fft_float32_to_complex64_2d(self):
        x = np.asarray(np.random.rand(self.B, self.N, self.M), np.float32)
        xf = np.fft.rfftn(x, axes=(1,2))
        x_gpu = gpuarray.to_gpu(x)
        xf_gpu = gpuarray.empty((self.B, self.N, self.M//2+1), np.complex64)
        plan = fft.Plan([self.N, self.M], np.float32, np.complex64, batch=self.B)
        fft.fft(x_gpu, xf_gpu, plan)
        assert np.allclose(xf, xf_gpu.get(), atol=atol_float32)

    def test_fft_float64_to_complex128_1d(self):
        x = np.asarray(np.random.rand(self.N), np.float64)
        xf = np.fft.rfftn(x)
        x_gpu = gpuarray.to_gpu(x)
        xf_gpu = gpuarray.empty(self.N//2+1, np.complex128)
        plan = fft.Plan(x.shape, np.float64, np.complex128)
        fft.fft(x_gpu, xf_gpu, plan)
        assert np.allclose(xf, xf_gpu.get(), atol=atol_float64)

    def test_fft_float64_to_complex128_2d(self):
        x = np.asarray(np.random.rand(self.N, self.M), np.float64)
        xf = np.fft.rfftn(x)
        x_gpu = gpuarray.to_gpu(x)
        xf_gpu = gpuarray.empty((self.N, self.M//2+1), np.complex128)
        plan = fft.Plan(x.shape, np.float64, np.complex128)
        fft.fft(x_gpu, xf_gpu, plan)
        assert np.allclose(xf, xf_gpu.get(), atol=atol_float64)

    def test_batch_fft_float64_to_complex128_1d(self):
        x = np.asarray(np.random.rand(self.B, self.N), np.float64)
        xf = np.fft.rfft(x, axis=1)
        x_gpu = gpuarray.to_gpu(x)
        xf_gpu = gpuarray.empty((self.B, self.N//2+1), np.complex128)
        plan = fft.Plan(x.shape[1], np.float64, np.complex128, batch=self.B)
        fft.fft(x_gpu, xf_gpu, plan)
        assert np.allclose(xf, xf_gpu.get(), atol=atol_float64)

    def test_batch_fft_float64_to_complex128_2d(self):
        x = np.asarray(np.random.rand(self.B, self.N, self.M), np.float64)
        xf = np.fft.rfftn(x, axes=(1,2))
        x_gpu = gpuarray.to_gpu(x)
        xf_gpu = gpuarray.empty((self.B, self.N, self.M//2+1), np.complex128)
        plan = fft.Plan([self.N, self.M], np.float64, np.complex128, batch=self.B)
        fft.fft(x_gpu, xf_gpu, plan)
        assert np.allclose(xf, xf_gpu.get(), atol=atol_float64)

    def test_ifft_complex64_to_float32_1d(self):
        x = np.asarray(np.random.rand(self.N), np.float32)
        xf = np.asarray(np.fft.rfftn(x), np.complex64)
        xf_gpu = gpuarray.to_gpu(xf)
        x_gpu = gpuarray.empty(self.N, np.float32)
        plan = fft.Plan(x.shape, np.complex64, np.float32)
        fft.ifft(xf_gpu, x_gpu, plan, True)
        assert np.allclose(x, x_gpu.get(), atol=atol_float32)

    def test_ifft_complex64_to_float32_2d(self):

        # Note that since rfftn returns a Fortran-ordered array, it
        # needs to be reformatted as a C-ordered array before being
        # passed to gpuarray.to_gpu:
        x = np.asarray(np.random.rand(self.N, self.M), np.float32)
        xf = np.asarray(np.fft.rfftn(x), np.complex64)
        xf_gpu = gpuarray.to_gpu(np.ascontiguousarray(xf))
        x_gpu = gpuarray.empty((self.N, self.M), np.float32)
        plan = fft.Plan(x.shape, np.complex64, np.float32)
        fft.ifft(xf_gpu, x_gpu, plan, True)
        assert np.allclose(x, x_gpu.get(), atol=atol_float32)

    def test_batch_ifft_complex64_to_float32_1d(self):

        # Note that since rfftn returns a Fortran-ordered array, it
        # needs to be reformatted as a C-ordered array before being
        # passed to gpuarray.to_gpu:
        x = np.asarray(np.random.rand(self.B, self.N), np.float32)
        xf = np.asarray(np.fft.rfft(x, axis=1), np.complex64)
        xf_gpu = gpuarray.to_gpu(np.ascontiguousarray(xf))
        x_gpu = gpuarray.empty((self.B, self.N), np.float32)
        plan = fft.Plan(x.shape[1], np.complex64, np.float32, batch=self.B)
        fft.ifft(xf_gpu, x_gpu, plan, True)
        assert np.allclose(x, x_gpu.get(), atol=atol_float32)

    def test_batch_ifft_complex64_to_float32_2d(self):

        # Note that since rfftn returns a Fortran-ordered array, it
        # needs to be reformatted as a C-ordered array before being
        # passed to gpuarray.to_gpu:
        x = np.asarray(np.random.rand(self.B, self.N, self.M), np.float32)
        xf = np.asarray(np.fft.rfftn(x, axes=(1,2)), np.complex64)
        xf_gpu = gpuarray.to_gpu(np.ascontiguousarray(xf))
        x_gpu = gpuarray.empty((self.B, self.N, self.M), np.float32)
        plan = fft.Plan([self.N, self.M], np.complex64, np.float32, batch=self.B)
        fft.ifft(xf_gpu, x_gpu, plan, True)
        assert np.allclose(x, x_gpu.get(), atol=atol_float32)

    def test_ifft_complex128_to_float64_1d(self):
        x = np.asarray(np.random.rand(self.N), np.float64)
        xf = np.asarray(np.fft.rfftn(x), np.complex128)
        xf_gpu = gpuarray.to_gpu(xf)
        x_gpu = gpuarray.empty(self.N, np.float64)
        plan = fft.Plan(x.shape, np.complex128, np.float64)
        fft.ifft(xf_gpu, x_gpu, plan, True)
        assert np.allclose(x, x_gpu.get(), atol=atol_float64)

    def test_ifft_complex128_to_float64_2d(self):

        # Note that since rfftn returns a Fortran-ordered array, it
        # needs to be reformatted as a C-ordered array before being
        # passed to gpuarray.to_gpu:
        x = np.asarray(np.random.rand(self.N, self.M), np.float64)
        xf = np.asarray(np.fft.rfftn(x), np.complex128)
        xf_gpu = gpuarray.to_gpu(np.ascontiguousarray(xf))
        x_gpu = gpuarray.empty((self.N, self.M), np.float64)
        plan = fft.Plan(x.shape, np.complex128, np.float64)
        fft.ifft(xf_gpu, x_gpu, plan, True)
        assert np.allclose(x, x_gpu.get(), atol=atol_float64)

    def test_batch_ifft_complex128_to_float64_1d(self):

        # Note that since rfftn returns a Fortran-ordered array, it
        # needs to be reformatted as a C-ordered array before being
        # passed to gpuarray.to_gpu:
        x = np.asarray(np.random.rand(self.B, self.N), np.float64)
        xf = np.asarray(np.fft.rfft(x, axis=1), np.complex128)
        xf_gpu = gpuarray.to_gpu(np.ascontiguousarray(xf))
        x_gpu = gpuarray.empty((self.B, self.N), np.float64)
        plan = fft.Plan(x.shape[1], np.complex128, np.float64, batch=self.B)
        fft.ifft(xf_gpu, x_gpu, plan, True)
        assert np.allclose(x, x_gpu.get(), atol=atol_float64)

    def test_batch_ifft_complex128_to_float64_2d(self):

        # Note that since rfftn returns a Fortran-ordered array, it
        # needs to be reformatted as a C-ordered array before being
        # passed to gpuarray.to_gpu:
        x = np.asarray(np.random.rand(self.B, self.N, self.M), np.float64)
        xf = np.asarray(np.fft.rfftn(x, axes=(1,2)), np.complex128)
        xf_gpu = gpuarray.to_gpu(np.ascontiguousarray(xf))
        x_gpu = gpuarray.empty((self.B, self.N, self.M), np.float64)
        plan = fft.Plan([self.N, self.M], np.complex128, np.float64, batch=self.B)
        fft.ifft(xf_gpu, x_gpu, plan, True)
        assert np.allclose(x, x_gpu.get(), atol=atol_float64)

    def test_multiple_streams(self):
        x = np.asarray(np.random.rand(self.N), np.float32)
        xf = np.fft.rfftn(x)
        y = np.asarray(np.random.rand(self.N), np.float32)
        yf = np.fft.rfftn(y)
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.to_gpu(y)
        xf_gpu = gpuarray.empty(self.N//2+1, np.complex64)
        yf_gpu = gpuarray.empty(self.N//2+1, np.complex64)
        stream0 = drv.Stream()
        stream1 = drv.Stream()
        plan1 = fft.Plan(x.shape, np.float32, np.complex64, stream=stream0)
        plan2 = fft.Plan(y.shape, np.float32, np.complex64, stream=stream1)
        fft.fft(x_gpu, xf_gpu, plan1)
        fft.fft(y_gpu, yf_gpu, plan2)
        assert np.allclose(xf, xf_gpu.get(), atol=atol_float32)
        assert np.allclose(yf, yf_gpu.get(), atol=atol_float32)

    def test_work_area(self):
        x = np.asarray(np.random.rand(self.N), np.float32)
        xf = np.fft.rfftn(x)
        x_gpu = gpuarray.to_gpu(x)
        xf_gpu = gpuarray.empty(self.N//2+1, np.complex64)
        plan = fft.Plan(x.shape, np.float32, np.complex64, auto_allocate=False)
        work_area = gpuarray.empty((plan.worksize,), np.uint8)
        plan.set_work_area(work_area)
        fft.fft(x_gpu, xf_gpu, plan)
        assert np.allclose(xf, xf_gpu.get(), atol=atol_float32)

def suite():
    s = TestSuite()
    s.addTest(test_fft('test_fft_float32_to_complex64_1d'))
    s.addTest(test_fft('test_fft_float32_to_complex64_2d'))
    s.addTest(test_fft('test_batch_fft_float32_to_complex64_1d'))
    s.addTest(test_fft('test_batch_fft_float32_to_complex64_2d'))
    s.addTest(test_fft('test_ifft_complex64_to_float32_1d'))
    s.addTest(test_fft('test_ifft_complex64_to_float32_2d'))
    s.addTest(test_fft('test_batch_ifft_complex64_to_float32_1d'))
    s.addTest(test_fft('test_batch_ifft_complex64_to_float32_2d'))
    s.addTest(test_fft('test_multiple_streams'))
    s.addTest(test_fft('test_work_area'))
    if misc.get_compute_capability(pycuda.autoinit.device) >= 1.3:
        s.addTest(test_fft('test_fft_float64_to_complex128_1d'))
        s.addTest(test_fft('test_fft_float64_to_complex128_2d'))
        s.addTest(test_fft('test_batch_fft_float64_to_complex128_1d'))
        s.addTest(test_fft('test_batch_fft_float64_to_complex128_2d'))
        s.addTest(test_fft('test_ifft_complex128_to_float64_1d'))
        s.addTest(test_fft('test_ifft_complex128_to_float64_2d'))
        s.addTest(test_fft('test_batch_ifft_complex128_to_float64_1d'))
        s.addTest(test_fft('test_batch_ifft_complex128_to_float64_2d'))
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
