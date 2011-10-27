#!/usr/bin/env python

"""
Unit tests for scikits.cuda.cublas
"""



from unittest import main, makeSuite, TestCase, TestSuite

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

import scikits.cuda.cublas as cublas
import scikits.cuda.misc as misc

class test_cublas(TestCase):
    def setUp(self):
        cublas.cublasInit()

    # ISAMAX, IDAMAX, ICAMAX, IZAMAX
    def test_cublasIsamax(self):
        x = np.random.rand(5).astype(np.float32)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasIsamax(x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.argmax(x))

    def test_cublasIdamax(self):
        x = np.random.rand(5).astype(np.float64)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasIdamax(x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.argmax(x))

    def test_cublasIcamax(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasIcamax(x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.argmax(np.abs(x)))

    def test_cublasIzamax(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasIzamax(x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.argmax(np.abs(x)))

    # ISAMIN, IDAMIN, ICAMIN, IZAMIN
    def test_cublasIsamin(self):
        x = np.random.rand(5).astype(np.float32)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasIsamin(x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.argmin(x))

    def test_cublasIdamin(self):
        x = np.random.rand(5).astype(np.float64)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasIdamin(x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.argmin(x))

    def test_cublasIcamin(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasIcamin(x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.argmin(np.abs(x)))

    def test_cublasIzamin(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasIzamin(x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.argmin(np.abs(x)))

    # SASUM, DASUM, SCASUM, DZASUM
    def test_cublasSasum(self):
        x = np.random.rand(5).astype(np.float32)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasSasum(x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.sum(np.abs(x)))

    def test_cublasDasum(self):
        x = np.random.rand(5).astype(np.float64)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasDasum(x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.sum(np.abs(x)))

    def test_cublasScasum(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasScasum(x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.sum(np.abs(x.real)+np.abs(x.imag)))

    def test_cublasDzasum(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasDzasum(x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.sum(np.abs(x.real)+np.abs(x.imag)))

    # SAXPY, DAXPY, CAXPY, ZAXPY
    def test_cublasSaxpy(self):
        alpha = np.float32(np.random.rand())
        x = np.random.rand(5).astype(np.float32)
        x_gpu = gpuarray.to_gpu(x)
        y = np.random.rand(5).astype(np.float32)
        y_gpu = gpuarray.to_gpu(y)
        cublas.cublasSaxpy(x_gpu.size, alpha, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), alpha*x+y)

    def test_cublasDaxpy(self):
        alpha = np.float64(np.random.rand())
        x = np.random.rand(5).astype(np.float64)
        x_gpu = gpuarray.to_gpu(x)
        y = np.random.rand(5).astype(np.float64)
        y_gpu = gpuarray.to_gpu(y)
        cublas.cublasDaxpy(x_gpu.size, alpha, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), alpha*x+y)

    def test_cublasCaxpy(self):
        alpha = np.complex64(np.random.rand()+1j*np.random.rand())
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        y = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        y_gpu = gpuarray.to_gpu(y)
        cublas.cublasCaxpy(x_gpu.size, alpha, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), alpha*x+y)

    def test_cublasZaxpy(self):
        alpha = np.complex128(np.random.rand()+1j*np.random.rand())
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        y = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        y_gpu = gpuarray.to_gpu(y)
        cublas.cublasZaxpy(x_gpu.size, alpha, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), alpha*x+y)

    # SCOPY, DCOPY, CCOPY, ZCOPY
    def test_cublasScopy(self):
        x = np.random.rand(5).astype(np.float32)
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.zeros_like(x_gpu)
        cublas.cublasScopy(x_gpu.size, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), x_gpu.get())

    def test_cublasDcopy(self):
        x = np.random.rand(5).astype(np.float64)
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.zeros_like(x_gpu)
        cublas.cublasDcopy(x_gpu.size, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), x_gpu.get())

    def test_cublasCcopy(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = misc.zeros_like(x_gpu)
        cublas.cublasCcopy(x_gpu.size, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), x_gpu.get())

    def test_cublasZcopy(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = misc.zeros_like(x_gpu)
        cublas.cublasZcopy(x_gpu.size, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), x_gpu.get())

    # SDOT, DDOT, CDOTU, CDOTC, ZDOTU, ZDOTC
    def test_cublasSdot(self):
        x = np.random.rand(5).astype(np.float32)
        x_gpu = gpuarray.to_gpu(x)
        y = np.random.rand(5).astype(np.float32)
        y_gpu = gpuarray.to_gpu(y)
        result = cublas.cublasSdot(x_gpu.size, x_gpu.gpudata, 1,
                                   y_gpu.gpudata, 1)
        assert np.allclose(result, np.dot(x, y))

    def test_cublasDdot(self):
        x = np.random.rand(5).astype(np.float64)
        x_gpu = gpuarray.to_gpu(x)
        y = np.random.rand(5).astype(np.float64)
        y_gpu = gpuarray.to_gpu(y)
        result = cublas.cublasDdot(x_gpu.size, x_gpu.gpudata, 1,
                                   y_gpu.gpudata, 1)
        assert np.allclose(result, np.dot(x, y))

    def test_cublasCdotu(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        y = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        y_gpu = gpuarray.to_gpu(y)
        result = cublas.cublasCdotu(x_gpu.size, x_gpu.gpudata, 1,
                                    y_gpu.gpudata, 1)
        assert np.allclose(result, np.dot(x, y))

    def test_cublasCdotc(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        y = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        y_gpu = gpuarray.to_gpu(y)
        result = cublas.cublasCdotc(x_gpu.size, x_gpu.gpudata, 1,
                                    y_gpu.gpudata, 1)
        assert np.allclose(result, np.dot(np.conj(x), y))

    def test_cublasZdotu(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        y = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        y_gpu = gpuarray.to_gpu(y)
        result = cublas.cublasZdotu(x_gpu.size, x_gpu.gpudata, 1,
                                    y_gpu.gpudata, 1)
        assert np.allclose(result, np.dot(x, y))

    def test_cublasZdotc(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        y = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        y_gpu = gpuarray.to_gpu(y)
        result = cublas.cublasZdotc(x_gpu.size, x_gpu.gpudata, 1,
                                    y_gpu.gpudata, 1)
        assert np.allclose(result, np.dot(np.conj(x), y))

    # SNRM2, DNRM2, SCNRM2, DZNRM2
    def test_cublasSrnm2(self):
        x = np.random.rand(5).astype(np.float32)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasSnrm2(x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.linalg.norm(x))

    def test_cublasDrnm2(self):
        x = np.random.rand(5).astype(np.float64)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasDnrm2(x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.linalg.norm(x))

    def test_cublasScrnm2(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasScnrm2(x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.linalg.norm(x))

    def test_cublasDzrnm2(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        result = cublas.cublasDznrm2(x_gpu.size, x_gpu.gpudata, 1)
        assert np.allclose(result, np.linalg.norm(x))

    # SSCAL, DSCAL, CSCAL, CSSCAL, ZSCAL, ZDSCAL
    def test_cublasSscal(self):
        x = np.random.rand(5).astype(np.float32)
        x_gpu = gpuarray.to_gpu(x)
        alpha = np.float32(np.random.rand())
        cublas.cublasSscal(x_gpu.size, alpha,
                           x_gpu.gpudata, 1)
        assert np.allclose(x_gpu.get(), alpha*x)
        
    def test_cublasCscal(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        alpha = np.complex64(np.random.rand()+1j*np.random.rand())
        cublas.cublasCscal(x_gpu.size, alpha,
                           x_gpu.gpudata, 1)
        assert np.allclose(x_gpu.get(), alpha*x)

    def test_cublasCsscal(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        alpha = np.float32(np.random.rand())
        cublas.cublasCscal(x_gpu.size, alpha,
                           x_gpu.gpudata, 1)
        assert np.allclose(x_gpu.get(), alpha*x)

    def test_cublasDscal(self):
        x = np.random.rand(5).astype(np.float64)
        x_gpu = gpuarray.to_gpu(x)
        alpha = np.float64(np.random.rand())
        cublas.cublasDscal(x_gpu.size, alpha,
                           x_gpu.gpudata, 1)
        assert np.allclose(x_gpu.get(), alpha*x)

    def test_cublasZscal(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        alpha = np.complex128(np.random.rand()+1j*np.random.rand())
        cublas.cublasZscal(x_gpu.size, alpha,
                           x_gpu.gpudata, 1)
        assert np.allclose(x_gpu.get(), alpha*x)

    def test_cublasZdscal(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        alpha = np.float64(np.random.rand())
        cublas.cublasZdscal(x_gpu.size, alpha,
                           x_gpu.gpudata, 1)
        assert np.allclose(x_gpu.get(), alpha*x)

    # SSWAP, DSWAP, CSWAP, ZSWAP
    def test_cublasSswap(self):
        x = np.random.rand(5).astype(np.float32)
        x_gpu = gpuarray.to_gpu(x)
        y = np.random.rand(5).astype(np.float32)
        y_gpu = gpuarray.to_gpu(y)
        cublas.cublasSswap(x_gpu.size, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(x_gpu.get(), y)

    def test_cublasDswap(self):
        x = np.random.rand(5).astype(np.float64)
        x_gpu = gpuarray.to_gpu(x)
        y = np.random.rand(5).astype(np.float64)
        y_gpu = gpuarray.to_gpu(y)
        cublas.cublasDswap(x_gpu.size, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(x_gpu.get(), y)

    def test_cublasCswap(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        y = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
        y_gpu = gpuarray.to_gpu(y)
        cublas.cublasCswap(x_gpu.size, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
        assert np.allclose(x_gpu.get(), y)

    def test_cublasZswap(self):
        x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        y = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
        y_gpu = gpuarray.to_gpu(y)
        cublas.cublasZswap(x_gpu.size, x_gpu.gpudata, 1,
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
        cublas.cublasSgemv('n', 2, 3, alpha, a_gpu.gpudata, 2, x_gpu.gpudata,
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
        cublas.cublasDgemv('n', 2, 3, alpha, a_gpu.gpudata, 2, x_gpu.gpudata,
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
        cublas.cublasCgemv('n', 2, 3, alpha, a_gpu.gpudata, 2, x_gpu.gpudata,
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
        cublas.cublasZgemv('n', 2, 3, alpha, a_gpu.gpudata, 2, x_gpu.gpudata,
                           1, beta, y_gpu.gpudata, 1)
        assert np.allclose(y_gpu.get(), np.dot(a, x))
        
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
    s.addTest(test_cublas('test_cublasSswap'))
    s.addTest(test_cublas('test_cublasCswap'))
    s.addTest(test_cublas('test_cublasSgemv'))
    s.addTest(test_cublas('test_cublasCgemv'))
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
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
