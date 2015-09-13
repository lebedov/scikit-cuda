#!/usr/bin/env python

"""
Unit tests for scikits.cuda.linalg
"""

from unittest import main, makeSuite, TestCase, TestSuite

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

from numpy.testing import assert_raises

import skcuda.linalg as linalg
import skcuda.rlinalg as rlinalg
import skcuda.misc as misc

atol_float32 = 1e-4
atol_float64 = 1e-8

class test_rlinalg(TestCase):
    def setUp(self):
        np.random.seed(0)
        linalg.init()
        rlinalg.init()

    def test_rsvd_float32(self):
        m, n = 5, 4
        a = np.array(np.random.randn(m, n), np.float32, order='F')
        a_gpu = gpuarray.to_gpu(a)
        U, s, Vt = rlinalg.rsvd(a_gpu, k=n, p=0, q=2, method='standard')
        assert np.allclose(a, np.dot(U.get(), np.dot(np.diag(s.get()), Vt.get())), atol_float32)
       
    def test_rsvd_float64(self):
        m, n = 5, 4
        a = np.array(np.random.randn(m, n), np.float64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        U, s, Vt = rlinalg.rsvd(a_gpu, k=n, p=0, q=2, method='standard')
        assert np.allclose(a, np.dot(U.get(), np.dot(np.diag(s.get()), Vt.get())), atol_float64)
    
    def test_rsvd_complex64(self):
        m, n = 5, 4
        a = np.array(np.random.randn(m, n) + 1j*np.random.randn(m, n), np.complex64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        U, s, Vt = rlinalg.rsvd(a_gpu, k=n, p=0, q=2, method='standard')
        assert np.allclose(a, np.dot(U.get(), np.dot(np.diag(s.get()), Vt.get())), atol_float32)
        
    def test_rsvd_complex128(self):
        m, n = 5, 4
        a = np.array(np.random.randn(m, n) + 1j*np.random.randn(m, n), np.complex128, order='F')
        a_gpu = gpuarray.to_gpu(a)
        U, s, Vt = rlinalg.rsvd(a_gpu, k=n, p=0, q=2, method='standard')
        assert np.allclose(a, np.dot(U.get(), np.dot(np.diag(s.get()), Vt.get())), atol_float64)
     
    def test_rsvdf_float32(self):
        m, n = 5, 4
        a = np.array(np.random.randn(m, n), np.float32, order='F')
        a_gpu = gpuarray.to_gpu(a)
        U, s, Vt = rlinalg.rsvd(a_gpu, k=n, p=0, q=2, method='fast')
        assert np.allclose(a, np.dot(U.get(), np.dot(np.diag(s.get()), Vt.get())), atol_float32)

    def test_rsvdf_float64(self):
        m, n = 5, 4
        a = np.array(np.random.randn(m, n), np.float64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        U, s, Vt = rlinalg.rsvd(a_gpu, k=n, p=0, q=2, method='fast')
        assert np.allclose(a, np.dot(U.get(), np.dot(np.diag(s.get()), Vt.get())), atol_float64)
    
    def test_rsvdf_complex64(self):
        m, n = 5, 4
        a = np.array(np.random.randn(m, n) + 1j*np.random.randn(m, n), np.complex64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        U, s, Vt = rlinalg.rsvd(a_gpu, k=n, p=0, q=2, method='fast')
        assert np.allclose(a, np.dot(U.get(), np.dot(np.diag(s.get()), Vt.get())), atol_float32)
        
    def test_rsvdf_complex128(self):
        m, n = 5, 4
        a = np.array(np.random.randn(m, n) + 1j*np.random.randn(m, n), np.complex128, order='F')
        a_gpu = gpuarray.to_gpu(a)
        U, s, Vt = rlinalg.rsvd(a_gpu, k=n, p=0, q=2, method='fast')
        assert np.allclose(a, np.dot(U.get(), np.dot(np.diag(s.get()), Vt.get())), atol_float64) 

def suite():
    s = TestSuite()
    s.addTest(test_linalg('test_rsvd_float32'))
    s.addTest(test_linalg('test_rsvd_float64'))
    s.addTest(test_linalg('test_rsvd_complex64'))
    s.addTest(test_linalg('test_rsvd_complex128'))
    s.addTest(test_linalg('test_rsvdf_float32'))
    s.addTest(test_linalg('test_rsvdf_float64'))
    s.addTest(test_linalg('test_rsvdf_complex64'))
    s.addTest(test_linalg('test_rsvdf_complex128'))
     
    
    if misc.get_compute_capability(pycuda.autoinit.device) >= 1.3:
        s.addTest(test_linalg('test_rsvd_float32'))
        s.addTest(test_linalg('test_rsvd_float64'))
        s.addTest(test_linalg('test_rsvd_complex64'))
        s.addTest(test_linalg('test_rsvd_complex128'))  
        s.addTest(test_linalg('test_rsvdf_float32'))
        s.addTest(test_linalg('test_rsvdf_float64'))
        s.addTest(test_linalg('test_rsvdf_complex64'))
        s.addTest(test_linalg('test_rsvdf_complex128'))
        
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
