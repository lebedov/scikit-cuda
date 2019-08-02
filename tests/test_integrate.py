"""
Unit tests for skcuda.integrate
"""

from unittest import main, TestCase, TestSuite

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.tools import clear_context_caches, make_default_context
import numpy as np
import skcuda.misc as misc
import skcuda.integrate as integrate
import scipy.integrate

drv.init()

class test_integrate(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = make_default_context()
        integrate.init()

    @classmethod
    def tearDownClass(cls):
        integrate.shutdown()
        cls.ctx.pop()
        clear_context_caches()

    def setUp(self):
        np.random.seed(0)

    def test_trapz_float32(self):
        x = np.asarray(np.random.rand(10), np.float32)
        x_gpu = gpuarray.to_gpu(x)
        z = integrate.trapz(x_gpu)
        np.testing.assert_allclose(np.trapz(x), z)

    def test_trapz_float64(self):
        x = np.asarray(np.random.rand(10), np.float64)
        x_gpu = gpuarray.to_gpu(x)
        z = integrate.trapz(x_gpu)
        np.testing.assert_allclose(np.trapz(x), z)

    def test_trapz_complex64(self):
        x = np.asarray(np.random.rand(10)+1j*np.random.rand(10), np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        z = integrate.trapz(x_gpu)
        np.testing.assert_allclose(np.trapz(x), z)

    def test_trapz_complex128(self):
        x = np.asarray(np.random.rand(10)+1j*np.random.rand(10), np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        z = integrate.trapz(x_gpu)
        np.testing.assert_allclose(np.trapz(x), z)

    def test_simps_float32(self):
        x = np.asarray(np.random.rand(10), np.float32)
        x_gpu = gpuarray.to_gpu(x)
        np.testing.assert_allclose(scipy.integrate.simps(x),
                        integrate.simps(x_gpu))
        np.testing.assert_allclose(scipy.integrate.simps(x, even='first'),
                        integrate.simps(x_gpu, even='first'))
        np.testing.assert_allclose(scipy.integrate.simps(x, even='last'),
                        integrate.simps(x_gpu, even='last'))

    def test_simps_float64(self):
        x = np.asarray(np.random.rand(10), np.float64)
        x_gpu = gpuarray.to_gpu(x)
        np.testing.assert_allclose(scipy.integrate.simps(x),
                        integrate.simps(x_gpu))
        np.testing.assert_allclose(scipy.integrate.simps(x, even='first'),
                        integrate.simps(x_gpu, even='first'))
        np.testing.assert_allclose(scipy.integrate.simps(x, even='last'),
                        integrate.simps(x_gpu, even='last'))

    def test_simps_complex64(self):
        x = np.asarray(np.random.rand(10)+1j*np.random.rand(10), np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        np.testing.assert_allclose(scipy.integrate.simps(x),
                        integrate.simps(x_gpu))
        np.testing.assert_allclose(scipy.integrate.simps(x, even='first'),
                        integrate.simps(x_gpu, even='first'))
        np.testing.assert_allclose(scipy.integrate.simps(x, even='last'),
                        integrate.simps(x_gpu, even='last'))

    def test_simps_complex128(self):
        x = np.asarray(np.random.rand(10)+1j*np.random.rand(10), np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        np.testing.assert_allclose(scipy.integrate.simps(x),
                        integrate.simps(x_gpu))
        np.testing.assert_allclose(scipy.integrate.simps(x, even='first'),
                        integrate.simps(x_gpu, even='first'))
        np.testing.assert_allclose(scipy.integrate.simps(x, even='last'),
                        integrate.simps(x_gpu, even='last'))

    def test_trapz2d_float32(self):
        x = np.asarray(np.random.rand(5, 5), np.float32)
        x_gpu = gpuarray.to_gpu(x)
        z = integrate.trapz2d(x_gpu)
        np.testing.assert_allclose(np.trapz(np.trapz(x)), z)

    def test_trapz2d_float64(self):
        x = np.asarray(np.random.rand(5, 5), np.float64)
        x_gpu = gpuarray.to_gpu(x)
        z = integrate.trapz2d(x_gpu)
        np.testing.assert_allclose(np.trapz(np.trapz(x)), z)

    def test_trapz2d_complex64(self):
        x = np.asarray(np.random.rand(5, 5)+1j*np.random.rand(5, 5), np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        z = integrate.trapz2d(x_gpu)
        np.testing.assert_allclose(np.trapz(np.trapz(x)), z)

    def test_trapz2d_complex128(self):
        x = np.asarray(np.random.rand(5, 5)+1j*np.random.rand(5, 5), np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        z = integrate.trapz2d(x_gpu)
        np.testing.assert_allclose(np.trapz(np.trapz(x)), z)

def suite():
    context = make_default_context()
    device = context.get_device()
    context.pop()

    s = TestSuite()
    s.addTest(test_integrate('test_trapz_float32'))
    s.addTest(test_integrate('test_trapz_complex64'))
    s.addTest(test_integrate('test_simps_float32'))
    s.addTest(test_integrate('test_simps_complex64'))
    s.addTest(test_integrate('test_trapz2d_float32'))
    s.addTest(test_integrate('test_trapz2d_complex64'))
    if misc.get_compute_capability(device) >= 1.3:
        s.addTest(test_integrate('test_trapz_float64'))
        s.addTest(test_integrate('test_trapz_complex128'))
        s.addTest(test_integrate('test_simps_float64'))
        s.addTest(test_integrate('test_simps_complex128'))
        s.addTest(test_integrate('test_trapz2d_float64'))
        s.addTest(test_integrate('test_trapz2d_complex128'))
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
