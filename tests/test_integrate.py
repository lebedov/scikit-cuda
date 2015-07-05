"""
Unit tests for scikits.cuda.integrate
"""

from unittest import main, TestCase, TestSuite

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

import skcuda.misc as misc
import skcuda.integrate as integrate

class test_integrate(TestCase):
    def setUp(self):
        np.random.seed(0)
        integrate.init()

    def test_trapz_float32(self):
        x = np.asarray(np.random.rand(10), np.float32)
        x_gpu = gpuarray.to_gpu(x)
        z = integrate.trapz(x_gpu)
        assert np.allclose(np.trapz(x), z)

    def test_trapz_float64(self):
        x = np.asarray(np.random.rand(10), np.float64)
        x_gpu = gpuarray.to_gpu(x)
        z = integrate.trapz(x_gpu)
        assert np.allclose(np.trapz(x), z)

    def test_trapz_complex64(self):
        x = np.asarray(np.random.rand(10)+1j*np.random.rand(10), np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        z = integrate.trapz(x_gpu)
        assert np.allclose(np.trapz(x), z)

    def test_trapz_complex128(self):
        x = np.asarray(np.random.rand(10)+1j*np.random.rand(10), np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        z = integrate.trapz(x_gpu)
        assert np.allclose(np.trapz(x), z)

    def test_trapz2d_float32(self):
        x = np.asarray(np.random.rand(5, 5), np.float32)
        x_gpu = gpuarray.to_gpu(x)
        z = integrate.trapz2d(x_gpu)
        assert np.allclose(np.trapz(np.trapz(x)), z)

    def test_trapz2d_float64(self):
        x = np.asarray(np.random.rand(5, 5), np.float64)
        x_gpu = gpuarray.to_gpu(x)
        z = integrate.trapz2d(x_gpu)
        assert np.allclose(np.trapz(np.trapz(x)), z)

    def test_trapz2d_complex64(self):
        x = np.asarray(np.random.rand(5, 5)+1j*np.random.rand(5, 5), np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        z = integrate.trapz2d(x_gpu)
        assert np.allclose(np.trapz(np.trapz(x)), z)

    def test_trapz2d_complex128(self):
        x = np.asarray(np.random.rand(5, 5)+1j*np.random.rand(5, 5), np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        z = integrate.trapz2d(x_gpu)
        assert np.allclose(np.trapz(np.trapz(x)), z)

def suite():
    s = TestSuite()
    s.addTest(test_integrate('test_trapz_float32'))
    s.addTest(test_integrate('test_trapz_complex64'))
    s.addTest(test_integrate('test_trapz2d_float32'))
    s.addTest(test_integrate('test_trapz2d_complex64'))
    if misc.get_compute_capability(pycuda.autoinit.device) >= 1.3:
        s.addTest(test_integrate('test_trapz_float64'))
        s.addTest(test_integrate('test_trapz_complex128'))
        s.addTest(test_integrate('test_trapz2d_float64'))
        s.addTest(test_integrate('test_trapz2d_complex128'))
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
