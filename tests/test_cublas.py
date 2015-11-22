#!/usr/bin/env python

"""
Unit tests for scikits.cuda.cublas
"""

from unittest import main, makeSuite, TestCase, TestSuite

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

_SEPS = np.finfo(np.float32).eps
_DEPS = np.finfo(np.float64).eps

import skcuda.cublas as cublas
import skcuda.misc as misc

def bptrs(a):
    """
    Pointer array when input represents a batch of matrices.
    """

    return gpuarray.arange(a.ptr,a.ptr+a.shape[0]*a.strides[0],a.strides[0],
                dtype=cublas.ctypes.c_void_p)

class test_cublas(TestCase):
    def setUp(self):
        np.random.seed(23)    # For reproducible tests.
        self.cublas_handle = cublas.cublasCreate()

    def tearDown(self):
        cublas.cublasDestroy(self.cublas_handle)

    # ISAMAX, IDAMAX, ICAMAX, IZAMAX
    def test_cublasIsamax(self):
        x = np.random.rand(5).astype(np.float32)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasIsamax(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.argmax(x))

    def test_cublasIdamax(self):
        x = np.random.rand(5).astype(np.float64)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasIdamax(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.argmax(x))

    def test_cublasIcamax(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasIcamax(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.argmax(np.abs(x.real) + np.abs(x.imag)))

    def test_cublasIzamax(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasIzamax(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.argmax(np.abs(x.real) + np.abs(x.imag)))

    # ISAMIN, IDAMIN, ICAMIN, IZAMIN
    def test_cublasIsamin(self):
        x = np.random.rand(5).astype(np.float32)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasIsamin(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.argmin(x))

    def test_cublasIdamin(self):
        x = np.random.rand(5).astype(np.float64)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasIdamin(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.argmin(x))

    def test_cublasIcamin(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasIcamin(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.argmin(np.abs(x.real) + np.abs(x.imag)))

    def test_cublasIzamin(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasIzamin(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.argmin(np.abs(x.real) + np.abs(x.imag)))

    # SASUM, DASUM, SCASUM, DZASUM
    def test_cublasSasum(self):
        x = np.random.rand(5).astype(np.float32)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasSasum(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.sum(np.abs(x)))

    def test_cublasDasum(self):
        x = np.random.rand(5).astype(np.float64)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasDasum(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.sum(np.abs(x)))

    def test_cublasScasum(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasScasum(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.sum(np.abs(x.real)+np.abs(x.imag)))

    def test_cublasDzasum(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasDzasum(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.sum(np.abs(x.real)+np.abs(x.imag)))

    # SAXPY, DAXPY, CAXPY, ZAXPY
    def test_cublasSaxpy(self):
        alpha = np.float32(np.random.rand())
        x = np.random.rand(5).astype(np.float32)
        x_gpu = gpuarray.to_gpu(x)
        y = np.random.rand(5).astype(np.float32)
        y_gpu = gpuarray.to_gpu(y)
        cublas.cublasSaxpy(self.cublas_handle, x_gpu.size, alpha, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), alpha*x+y)

    def test_cublasDaxpy(self):
        alpha = np.float64(np.random.rand())
        x = np.random.rand(5).astype(np.float64)
        x_gpu = gpuarray.to_gpu(x)
        y = np.random.rand(5).astype(np.float64)
        y_gpu = gpuarray.to_gpu(y)
        cublas.cublasDaxpy(self.cublas_handle, x_gpu.size, alpha, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), alpha*x+y)

    def test_cublasCaxpy(self):
        alpha = np.complex64(np.random.rand()+1j*np.random.rand())
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        y = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        y_gpu = gpuarray.to_gpu(y)
        cublas.cublasCaxpy(self.cublas_handle, x_gpu.size, alpha, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), alpha*x+y)

    def test_cublasZaxpy(self):
        alpha = np.complex128(np.random.rand()+1j*np.random.rand())
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        y = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        y_gpu = gpuarray.to_gpu(y)
        cublas.cublasZaxpy(self.cublas_handle, x_gpu.size, alpha, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), alpha*x+y)

    # SCOPY, DCOPY, CCOPY, ZCOPY
    def test_cublasScopy(self):
        x = np.random.rand(5).astype(np.float32)
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.zeros_like(x_gpu)
        cublas.cublasScopy(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), x_gpu.get())

    def test_cublasDcopy(self):
        x = np.random.rand(5).astype(np.float64)
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.zeros_like(x_gpu)
        cublas.cublasDcopy(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), x_gpu.get())

    def test_cublasCcopy(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = misc.zeros_like(x_gpu)
        cublas.cublasCcopy(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), x_gpu.get())

    def test_cublasZcopy(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = misc.zeros_like(x_gpu)
        cublas.cublasZcopy(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), x_gpu.get())

    # SDOT, DDOT, CDOTU, CDOTC, ZDOTU, ZDOTC
    def test_cublasSdot(self):
        x = np.random.rand(5).astype(np.float32)
        x_gpu = gpuarray.to_gpu(x)
        y = np.random.rand(5).astype(np.float32)
        y_gpu = gpuarray.to_gpu(y)
        result = cublas.cublasSdot(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1,
                                   y_gpu.gpudata, 1)
        assert np.allclose(result, np.dot(x, y))

    def test_cublasDdot(self):
        x = np.random.rand(5).astype(np.float64)
        x_gpu = gpuarray.to_gpu(x)
        y = np.random.rand(5).astype(np.float64)
        y_gpu = gpuarray.to_gpu(y)
        result = cublas.cublasDdot(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1,
                                   y_gpu.gpudata, 1)
        assert np.allclose(result, np.dot(x, y))

    def test_cublasCdotu(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        y = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        y_gpu = gpuarray.to_gpu(y)
        result = cublas.cublasCdotu(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1,
                                    y_gpu.gpudata, 1)
        assert np.allclose(result, np.dot(x, y))

    def test_cublasCdotc(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        y = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        y_gpu = gpuarray.to_gpu(y)
        result = cublas.cublasCdotc(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1,
                                    y_gpu.gpudata, 1)
        assert np.allclose(result, np.dot(np.conj(x), y))

    def test_cublasZdotu(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        y = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        y_gpu = gpuarray.to_gpu(y)
        result = cublas.cublasZdotu(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1,
                                    y_gpu.gpudata, 1)
        assert np.allclose(result, np.dot(x, y))

    def test_cublasZdotc(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        y = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        y_gpu = gpuarray.to_gpu(y)
        result = cublas.cublasZdotc(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1,
                                    y_gpu.gpudata, 1)
        assert np.allclose(result, np.dot(np.conj(x), y))

    # SNRM2, DNRM2, SCNRM2, DZNRM2
    def test_cublasSrnm2(self):
        x = np.random.rand(5).astype(np.float32)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasSnrm2(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.linalg.norm(x))

    def test_cublasDrnm2(self):
        x = np.random.rand(5).astype(np.float64)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasDnrm2(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.linalg.norm(x))

    def test_cublasScrnm2(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasScnrm2(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.linalg.norm(x))

    def test_cublasDzrnm2(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasDznrm2(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.linalg.norm(x))

    # SSCAL, DSCAL, CSCAL, CSSCAL, ZSCAL, ZDSCAL
    def test_cublasSscal(self):
        x = np.random.rand(5).astype(np.float32)
        x_gpu = gpuarray.to_gpu(x)
        alpha = np.float32(np.random.rand())
        cublas.cublasSscal(self.cublas_handle, x_gpu.size, alpha,
                           x_gpu.gpudata, 1)
        assert np.allclose(x_gpu.get(), alpha*x)

    def test_cublasCscal(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        alpha = np.complex64(np.random.rand()+1j*np.random.rand())
        cublas.cublasCscal(self.cublas_handle, x_gpu.size, alpha,
                           x_gpu.gpudata, 1)
        assert np.allclose(x_gpu.get(), alpha*x)

    def test_cublasCsscal(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        alpha = np.float32(np.random.rand())
        cublas.cublasCscal(self.cublas_handle, x_gpu.size, alpha,
                           x_gpu.gpudata, 1)
        assert np.allclose(x_gpu.get(), alpha*x)

    def test_cublasDscal(self):
        x = np.random.rand(5).astype(np.float64)
        x_gpu = gpuarray.to_gpu(x)
        alpha = np.float64(np.random.rand())
        cublas.cublasDscal(self.cublas_handle, x_gpu.size, alpha,
                           x_gpu.gpudata, 1)
        assert np.allclose(x_gpu.get(), alpha*x)

    def test_cublasZscal(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        alpha = np.complex128(np.random.rand()+1j*np.random.rand())
        cublas.cublasZscal(self.cublas_handle, x_gpu.size, alpha,
                           x_gpu.gpudata, 1)
        assert np.allclose(x_gpu.get(), alpha*x)

    def test_cublasZdscal(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        alpha = np.float64(np.random.rand())
        cublas.cublasZdscal(self.cublas_handle, x_gpu.size, alpha,
                           x_gpu.gpudata, 1)
        assert np.allclose(x_gpu.get(), alpha*x)

    # SROT, DROT, CROT, CSROT, ZROT, ZDROT
    def test_cublasSrot(self):
        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([4, 5, 6]).astype(np.float32)
        s = 2.0
        c = 3.0
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.to_gpu(x)
        cublas.cublasSrot(self.cublas_handle, x_gpu.size,
                          x_gpu.gpudata, 1,
                          y_gpu.gpudata, 1,
                          c, s)
        assert np.allclose(x_gpu.get(), [5, 10, 15])
        assert np.allclose(y_gpu.get(), [1, 2, 3])

    # SSWAP, DSWAP, CSWAP, ZSWAP
    def test_cublasSswap(self):
        x = np.random.rand(5).astype(np.float32)
        x_gpu = gpuarray.to_gpu(x)
        y = np.random.rand(5).astype(np.float32)
        y_gpu = gpuarray.to_gpu(y)
        cublas.cublasSswap(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(x_gpu.get(), y)

    def test_cublasDswap(self):
        x = np.random.rand(5).astype(np.float64)
        x_gpu = gpuarray.to_gpu(x)
        y = np.random.rand(5).astype(np.float64)
        y_gpu = gpuarray.to_gpu(y)
        cublas.cublasDswap(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(x_gpu.get(), y)

    def test_cublasCswap(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        y = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        y_gpu = gpuarray.to_gpu(y)
        cublas.cublasCswap(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(x_gpu.get(), y)

    def test_cublasZswap(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        y = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        y_gpu = gpuarray.to_gpu(y)
        cublas.cublasZswap(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(x_gpu.get(), y)

    # SGEMV, DGEMV, CGEMV, ZGEMV
    def test_cublasSgemv(self):
        a = np.random.rand(2, 3).astype(np.float32)
        x = np.random.rand(3, 1).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a.T.copy())
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.empty((2, 1), np.float32)
        alpha = np.float32(1.0)
        beta = np.float32(0.0)
        cublas.cublasSgemv(self.cublas_handle, 'n', 2, 3, alpha,
                           a_gpu.gpudata, 2, x_gpu.gpudata,
                           1, beta, y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), np.dot(a, x))

    def test_cublasDgemv(self):
        a = np.random.rand(2, 3).astype(np.float64)
        x = np.random.rand(3, 1).astype(np.float64)
        a_gpu = gpuarray.to_gpu(a.T.copy())
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.empty((2, 1), np.float64)
        alpha = np.float64(1.0)
        beta = np.float64(0.0)
        cublas.cublasDgemv(self.cublas_handle, 'n', 2, 3, alpha,
                           a_gpu.gpudata, 2, x_gpu.gpudata,
                           1, beta, y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), np.dot(a, x))

    def test_cublasCgemv(self):
        a = (np.random.rand(2, 3)+1j*np.random.rand(2, 3)).astype(np.complex64)
        x = (np.random.rand(3, 1)+1j*np.random.rand(3, 1)).astype(np.complex64)
        a_gpu = gpuarray.to_gpu(a.T.copy())
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.empty((2, 1), np.complex64)
        alpha = np.complex64(1.0)
        beta = np.complex64(0.0)
        cublas.cublasCgemv(self.cublas_handle, 'n', 2, 3, alpha,
                           a_gpu.gpudata, 2, x_gpu.gpudata,
                           1, beta, y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), np.dot(a, x))

    def test_cublasZgemv(self):
        a = (np.random.rand(2, 3)+1j*np.random.rand(2, 3)).astype(np.complex128)
        x = (np.random.rand(3, 1)+1j*np.random.rand(3, 1)).astype(np.complex128)
        a_gpu = gpuarray.to_gpu(a.T.copy())
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.empty((2, 1), np.complex128)
        alpha = np.complex128(1.0)
        beta = np.complex128(0.0)
        cublas.cublasZgemv(self.cublas_handle, 'n', 2, 3, alpha,
                           a_gpu.gpudata, 2, x_gpu.gpudata,
                           1, beta, y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), np.dot(a, x))

    # SGEAM, CGEAM, DGEAM, ZDGEAM
    def test_cublasSgeam(self):
        a = np.random.rand(2, 3).astype(np.float32)
        b = np.random.rand(2, 3).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a.copy())
        b_gpu = gpuarray.to_gpu(b.copy())
        c_gpu = gpuarray.zeros_like(a_gpu)
        alpha = np.float32(np.random.rand())
        beta = np.float32(np.random.rand())
        cublas.cublasSgeam(self.cublas_handle, 'n', 'n', 2, 3,
                           alpha, a_gpu.gpudata, 2,
                           beta, b_gpu.gpudata, 2,
                           c_gpu.gpudata, 2)
        assert np.allclose(c_gpu.get(), alpha*a+beta*b)

    def test_cublasCgeam(self):
        a = (np.random.rand(2, 3)+1j*np.random.rand(2, 3)).astype(np.complex64)
        b = (np.random.rand(2, 3)+1j*np.random.rand(2, 3)).astype(np.complex64)
        a_gpu = gpuarray.to_gpu(a.copy())
        b_gpu = gpuarray.to_gpu(b.copy())
        c_gpu = gpuarray.zeros_like(a_gpu)
        alpha = np.complex64(np.random.rand()+1j*np.random.rand())
        beta = np.complex64(np.random.rand()+1j*np.random.rand())
        cublas.cublasCgeam(self.cublas_handle, 'n', 'n', 2, 3,
                           alpha, a_gpu.gpudata, 2,
                           beta, b_gpu.gpudata, 2,
                           c_gpu.gpudata, 2)
        assert np.allclose(c_gpu.get(), alpha*a+beta*b)

    def test_cublasDgeam(self):
        a = np.random.rand(2, 3).astype(np.float64)
        b = np.random.rand(2, 3).astype(np.float64)
        a_gpu = gpuarray.to_gpu(a.copy())
        b_gpu = gpuarray.to_gpu(b.copy())
        c_gpu = gpuarray.zeros_like(a_gpu)
        alpha = np.float64(np.random.rand())
        beta = np.float64(np.random.rand())
        cublas.cublasDgeam(self.cublas_handle, 'n', 'n', 2, 3,
                           alpha, a_gpu.gpudata, 2,
                           beta, b_gpu.gpudata, 2,
                           c_gpu.gpudata, 2)
        assert np.allclose(c_gpu.get(), alpha*a+beta*b)

    def test_cublasZgeam(self):
        a = (np.random.rand(2, 3)+1j*np.random.rand(2, 3)).astype(np.complex128)
        b = (np.random.rand(2, 3)+1j*np.random.rand(2, 3)).astype(np.complex128)
        a_gpu = gpuarray.to_gpu(a.copy())
        b_gpu = gpuarray.to_gpu(b.copy())
        c_gpu = gpuarray.zeros_like(a_gpu)
        alpha = np.complex128(np.random.rand()+1j*np.random.rand())
        beta = np.complex128(np.random.rand()+1j*np.random.rand())
        cublas.cublasZgeam(self.cublas_handle, 'n', 'n', 2, 3,
                           alpha, a_gpu.gpudata, 2,
                           beta, b_gpu.gpudata, 2,
                           c_gpu.gpudata, 2)
        assert np.allclose(c_gpu.get(), alpha*a+beta*b)

    # CgemmBatched, ZgemmBatched
    def test_cublasCgemmBatched(self):
        l, m, k, n = 11, 7, 5, 3
        A = (np.random.rand(l, m, k)+1j*np.random.rand(l, m, k)).astype(np.complex64)
        B = (np.random.rand(l, k, n)+1j*np.random.rand(l, k, n)).astype(np.complex64)

        C_res = np.einsum('nij,njk->nik', A, B)

        a_gpu = gpuarray.to_gpu(A)
        b_gpu = gpuarray.to_gpu(B)
        c_gpu = gpuarray.empty((l, m, n), np.complex64)

        alpha = np.complex64(1.0)
        beta = np.complex64(0.0)

        a_arr = bptrs(a_gpu)
        b_arr = bptrs(b_gpu)
        c_arr = bptrs(c_gpu)

        cublas.cublasCgemmBatched(self.cublas_handle, 'n','n',
                                  n, m, k, alpha,
                                  b_arr.gpudata, n,
                                  a_arr.gpudata, k,
                                  beta, c_arr.gpudata, n, l)

        assert np.allclose(C_res, c_gpu.get())

    def test_cublasZgemmBatched(self):
        l, m, k, n = 11, 7, 5, 3
        A = (np.random.rand(l, m, k)+1j*np.random.rand(l, m, k)).astype(np.complex128)
        B = (np.random.rand(l, k, n)+1j*np.random.rand(l, k, n)).astype(np.complex128)

        C_res = np.einsum('nij,njk->nik', A, B)

        a_gpu = gpuarray.to_gpu(A)
        b_gpu = gpuarray.to_gpu(B)
        c_gpu = gpuarray.empty((l, m, n), np.complex128)

        alpha = np.complex128(1.0)
        beta = np.complex128(0.0)

        a_arr = bptrs(a_gpu)
        b_arr = bptrs(b_gpu)
        c_arr = bptrs(c_gpu)

        cublas.cublasZgemmBatched(self.cublas_handle, 'n','n',
                                  n, m, k, alpha,
                                  b_arr.gpudata, n,
                                  a_arr.gpudata, k,
                                  beta, c_arr.gpudata, n, l)

        assert np.allclose(C_res, c_gpu.get())

    # SgemmBatched, DgemmBatched
    def test_cublasSgemmBatched(self):
        l, m, k, n = 11, 7, 5, 3
        A = np.random.rand(l, m, k).astype(np.float32)
        B = np.random.rand(l, k, n).astype(np.float32)

        C_res = np.einsum('nij,njk->nik', A, B)

        a_gpu = gpuarray.to_gpu(A)
        b_gpu = gpuarray.to_gpu(B)
        c_gpu = gpuarray.empty((l, m, n), np.float32)

        alpha = np.float32(1.0)
        beta = np.float32(0.0)

        a_arr = bptrs(a_gpu)
        b_arr = bptrs(b_gpu)
        c_arr = bptrs(c_gpu)

        cublas.cublasSgemmBatched(self.cublas_handle, 'n','n',
                                  n, m, k, alpha,
                                  b_arr.gpudata, n,
                                  a_arr.gpudata, k,
                                  beta, c_arr.gpudata, n, l)

        assert np.allclose(C_res, c_gpu.get())

    def test_cublasDgemmBatched(self):
        l, m, k, n = 11, 7, 5, 3
        A = np.random.rand(l, m, k).astype(np.float64)
        B = np.random.rand(l, k, n).astype(np.float64)

        C_res = np.einsum('nij,njk->nik',A,B)

        a_gpu = gpuarray.to_gpu(A)
        b_gpu = gpuarray.to_gpu(B)
        c_gpu = gpuarray.empty((l, m, n), np.float64)

        alpha = np.float64(1.0)
        beta = np.float64(0.0)

        a_arr = bptrs(a_gpu)
        b_arr = bptrs(b_gpu)
        c_arr = bptrs(c_gpu)

        cublas.cublasDgemmBatched(self.cublas_handle, 'n','n',
                                  n, m, k, alpha,
                                  b_arr.gpudata, n,
                                  a_arr.gpudata, k,
                                  beta, c_arr.gpudata, n, l)

        assert np.allclose(C_res, c_gpu.get())

    # StrsmBatched, DtrsmBatched
    def test_cublasStrsmBatched(self):
        l, m, n = 11, 7, 5
        A = np.random.rand(l, m, m).astype(np.float32)
        B = np.random.rand(l, m, n).astype(np.float32)

        A = np.array(list(map(np.triu, A)))
        X = np.array([np.linalg.solve(a, b) for a, b in zip(A, B)])

        alpha = np.float32(1.0)

        a_gpu = gpuarray.to_gpu(A)
        b_gpu = gpuarray.to_gpu(B)

        a_arr = bptrs(a_gpu)
        b_arr = bptrs(b_gpu)

        cublas.cublasStrsmBatched(self.cublas_handle, 'r', 'l', 'n', 'n',
                                  n, m, alpha,
                                  a_arr.gpudata, m,
                                  b_arr.gpudata, n, l)

        assert np.allclose(X, b_gpu.get(), 5)

    def test_cublasDtrsmBatched(self):
        l, m, n = 11, 7, 5
        A = np.random.rand(l, m, m).astype(np.float64)
        B = np.random.rand(l, m, n).astype(np.float64)

        A = np.array(list(map(np.triu, A)))
        X = np.array([np.linalg.solve(a, b) for a, b in zip(A, B)])

        alpha = np.float64(1.0)

        a_gpu = gpuarray.to_gpu(A)
        b_gpu = gpuarray.to_gpu(B)

        a_arr = bptrs(a_gpu)
        b_arr = bptrs(b_gpu)

        cublas.cublasDtrsmBatched(self.cublas_handle, 'r', 'l', 'n', 'n',
                                  n, m, alpha,
                                  a_arr.gpudata, m,
                                  b_arr.gpudata, n, l)

        assert np.allclose(X, b_gpu.get(), 5)

    # SgetrfBatched, DgetrfBatched
    def test_cublasSgetrfBatched(self):
        from scipy.linalg import lu_factor
        l, m = 11, 7
        A = np.random.rand(l, m, m).astype(np.float32)
        A = np.array([np.matrix(a)*np.matrix(a).T for a in A])

        a_gpu = gpuarray.to_gpu(A)
        a_arr = bptrs(a_gpu)
        p_gpu = gpuarray.empty((l, m), np.int32)
        i_gpu = gpuarray.zeros(1, np.int32)
        X = np.array([ lu_factor(a)[0] for a in A])

        cublas.cublasSgetrfBatched(self.cublas_handle,
                                   m, a_arr.gpudata, m,
                                   p_gpu.gpudata, i_gpu.gpudata, l)

        X_ = np.array([a.T for a in a_gpu.get()])

        assert np.allclose(X, X_, atol=10*_SEPS)

    def test_cublasDgetrfBatched(self):
        from scipy.linalg import lu_factor
        l, m = 11, 7
        A = np.random.rand(l, m, m).astype(np.float64)
        A = np.array([np.matrix(a)*np.matrix(a).T for a in A])

        a_gpu = gpuarray.to_gpu(A)
        a_arr = bptrs(a_gpu)
        p_gpu = gpuarray.empty((l, m), np.int32)
        i_gpu = gpuarray.zeros(1, np.int32)
        X = np.array([ lu_factor(a)[0] for a in A])

        cublas.cublasDgetrfBatched(self.cublas_handle,
                                   m, a_arr.gpudata, m,
                                   p_gpu.gpudata, i_gpu.gpudata, l)

        X_ = np.array([a.T for a in a_gpu.get()])

        assert np.allclose(X,X_)


def suite():
    s = TestSuite()
    s.addTest(test_cublas('test_cublasIsamax'))
    s.addTest(test_cublas('test_cublasIcamax'))
    s.addTest(test_cublas('test_cublasIsamin'))
    s.addTest(test_cublas('test_cublasIcamin'))
    s.addTest(test_cublas('test_cublasSasum'))
    s.addTest(test_cublas('test_cublasScasum'))
    s.addTest(test_cublas('test_cublasSaxpy'))
    s.addTest(test_cublas('test_cublasCaxpy'))
    s.addTest(test_cublas('test_cublasScopy'))
    s.addTest(test_cublas('test_cublasCcopy'))
    s.addTest(test_cublas('test_cublasSdot'))
    s.addTest(test_cublas('test_cublasCdotu'))
    s.addTest(test_cublas('test_cublasCdotc'))
    s.addTest(test_cublas('test_cublasSrnm2'))
    s.addTest(test_cublas('test_cublasScrnm2'))
    s.addTest(test_cublas('test_cublasSscal'))
    s.addTest(test_cublas('test_cublasCscal'))
    s.addTest(test_cublas('test_cublasSrot'))
    s.addTest(test_cublas('test_cublasSswap'))
    s.addTest(test_cublas('test_cublasCswap'))
    s.addTest(test_cublas('test_cublasSgemv'))
    s.addTest(test_cublas('test_cublasCgemv'))
    s.addTest(test_cublas('test_cublasSgeam'))
    s.addTest(test_cublas('test_cublasCgeam'))
    s.addTest(test_cublas('test_cublasSgemmBatched'))
    s.addTest(test_cublas('test_cublasCgemmBatched'))
    s.addTest(test_cublas('test_cublasStrsmBatched'))
    s.addTest(test_cublas('test_cublasSgetrfBatched'))
    if misc.get_compute_capability(pycuda.autoinit.device) >= 1.3:
        s.addTest(test_cublas('test_cublasIdamax'))
        s.addTest(test_cublas('test_cublasIzamax'))
        s.addTest(test_cublas('test_cublasIdamin'))
        s.addTest(test_cublas('test_cublasIzamin'))
        s.addTest(test_cublas('test_cublasDasum'))
        s.addTest(test_cublas('test_cublasDzasum'))
        s.addTest(test_cublas('test_cublasDaxpy'))
        s.addTest(test_cublas('test_cublasZaxpy'))
        s.addTest(test_cublas('test_cublasDcopy'))
        s.addTest(test_cublas('test_cublasZcopy'))
        s.addTest(test_cublas('test_cublasDdot'))
        s.addTest(test_cublas('test_cublasZdotu'))
        s.addTest(test_cublas('test_cublasZdotc'))
        s.addTest(test_cublas('test_cublasDrnm2'))
        s.addTest(test_cublas('test_cublasDzrnm2'))
        s.addTest(test_cublas('test_cublasDscal'))
        s.addTest(test_cublas('test_cublasZscal'))
        s.addTest(test_cublas('test_cublasZdscal'))
        s.addTest(test_cublas('test_cublasDswap'))
        s.addTest(test_cublas('test_cublasZswap'))
        s.addTest(test_cublas('test_cublasDgemv'))
        s.addTest(test_cublas('test_cublasZgemv'))
        s.addTest(test_cublas('test_cublasDgeam'))
        s.addTest(test_cublas('test_cublasZgeam'))
        s.addTest(test_cublas('test_cublasDgemmBatched'))
        s.addTest(test_cublas('test_cublasZgemmBatched'))
        s.addTest(test_cublas('test_cublasDtrsmBatched'))
        s.addTest(test_cublas('test_cublasDgetrfBatched'))
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
