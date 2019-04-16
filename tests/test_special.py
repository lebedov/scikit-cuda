#!/usr/bin/env python

"""
Unit tests for skcuda.special
"""


from unittest import main, makeSuite, TestCase, TestSuite

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.tools import clear_context_caches, make_default_context
import numpy as np
import scipy as sp
import scipy.special
import skcuda.linalg as linalg
import skcuda.misc as misc
import skcuda.special as special

drv.init()

class test_special(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = make_default_context()

    @classmethod
    def tearDownClass(cls):
        cls.ctx.pop()
        clear_context_caches()

    def setUp(self):
        np.random.seed(0)
        linalg.init()

    def test_sici_float32(self):
        x = np.array([[1, 2], [3, 4]], np.float32)
        x_gpu = gpuarray.to_gpu(x)
        (si_gpu, ci_gpu) = special.sici(x_gpu)
        (si, ci) = scipy.special.sici(x)
        np.testing.assert_allclose(si, si_gpu.get(), atol=1e-7)
        np.testing.assert_allclose(ci, ci_gpu.get(), atol=1e-7)

    def test_sici_float64(self):
        x = np.array([[1, 2], [3, 4]], np.float64)
        x_gpu = gpuarray.to_gpu(x)
        (si_gpu, ci_gpu) = special.sici(x_gpu)
        (si, ci) = scipy.special.sici(x)
        np.testing.assert_allclose(si, si_gpu.get())
        np.testing.assert_allclose(ci, ci_gpu.get())

    def test_exp1_complex64(self):
        z = np.asarray(np.random.rand(4, 4) + 1j*np.random.rand(4, 4), np.complex64)
        z_gpu = gpuarray.to_gpu(z)
        e_gpu = special.exp1(z_gpu)
        np.testing.assert_allclose(sp.special.exp1(z), e_gpu.get())

    def test_exp1_complex128(self):
        z = np.asarray(np.random.rand(4, 4) + 1j*np.random.rand(4, 4), np.complex128)
        z_gpu = gpuarray.to_gpu(z)
        e_gpu = special.exp1(z_gpu)
        np.testing.assert_allclose(sp.special.exp1(z), e_gpu.get())

    def test_expi_complex64(self):
        z = np.asarray(np.random.rand(4, 4) + 1j*np.random.rand(4, 4), np.complex64)
        z_gpu = gpuarray.to_gpu(z)
        e_gpu = special.expi(z_gpu)
        np.testing.assert_allclose(sp.special.expi(z), e_gpu.get())

    def test_expi_complex128(self):
        z = np.asarray(np.random.rand(4, 4) + 1j*np.random.rand(4, 4), np.complex128)
        z_gpu = gpuarray.to_gpu(z)
        e_gpu = special.expi(z_gpu)
        np.testing.assert_allclose(sp.special.expi(z), e_gpu.get())

def suite():
    context = make_default_context()
    device = context.get_device()
    context.pop()

    s = TestSuite()
    s.addTest(test_special('test_sici_float32'))
    s.addTest(test_special('test_exp1_complex64'))
    s.addTest(test_special('test_expi_complex64'))
    if misc.get_compute_capability(device) >= 1.3:
        s.addTest(test_special('test_sici_float64'))
        s.addTest(test_special('test_exp1_complex128'))
        s.addTest(test_special('test_expi_complex128'))
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
