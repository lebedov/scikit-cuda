#!/usr/bin/env python

"""
Unit tests for scikits.cuda.cublasxt
"""

from unittest import main, makeSuite, TestCase, TestSuite

import numpy as np

import pycuda.autoinit
import skcuda.cublasxt as cublasxt
import skcuda.misc as misc

class test_cublasxt(TestCase):
    def setUp(self):
        np.random.seed(0)
        self.handle = cublasxt.cublasXtCreate()
        self.nbDevices = 1
        self.deviceId = np.array([0], np.int32)
        cublasxt.cublasXtDeviceSelect(self.handle, self.nbDevices,
                                      self.deviceId)

    def tearDown(self):
        cublasxt.cublasXtDestroy(self.handle)

    def test_cublasXtSgemm(self):
        a = np.random.rand(4, 4).astype(np.float32)
        b = np.random.rand(4, 4).astype(np.float32)
        c = np.zeros((4, 4), np.float32)
        cublasxt.cublasXtSgemm(self.handle,
                               'N', 'N', 
                               4, 4, 4, np.float32(1.0),
                               a.ctypes.data, 4, b.ctypes.data, 4, np.float32(0.0),
                               c.ctypes.data, 4)
        np.allclose(np.dot(b.T, a.T).T, c)

    def test_cublasXtDgemm(self):
        a = np.random.rand(4, 4).astype(np.float64)
        b = np.random.rand(4, 4).astype(np.float64)
        c = np.zeros((4, 4), np.float64)
        
        cublasxt.cublasXtDgemm(self.handle,
                               'N', 'N', 4, 4, 4,
                               np.float64(1.0),
                               a.ctypes.data, 4, b.ctypes.data, 4, np.float64(0.0),                               
                               c.ctypes.data, 4)
        np.allclose(np.dot(b.T, a.T).T, c)

    def test_cublasXtCgemm(self):
        a = (np.random.rand(4, 4)+1j*np.random.rand(4, 4)).astype(np.complex128)
        b = (np.random.rand(4, 4)+1j*np.random.rand(4, 4)).astype(np.complex128)
        c = np.zeros((4, 4), np.complex128)
        
        cublasxt.cublasXtCgemm(self.handle,
                               'N', 'N', 4, 4, 4,           
                               np.complex128(1.0),
                               a.ctypes.data, 4, b.ctypes.data, 4, np.complex128(0.0),
                               c.ctypes.data, 4)
        np.allclose(np.dot(b.T, a.T).T, c)

    def test_cublasXtZgemm(self):
        a = (np.random.rand(4, 4)+1j*np.random.rand(4, 4)).astype(np.complex256)
        b = (np.random.rand(4, 4)+1j*np.random.rand(4, 4)).astype(np.complex256)
        c = np.zeros((4, 4), np.complex256)
        
        cublasxt.cublasXtZgemm(self.handle,
                               'N', 'N', 4, 4, 4,
                               np.complex256(1.0),
                               a.ctypes.data, 4, b.ctypes.data, 4, np.complex256(0.0),
                               c.ctypes.data, 4)
        np.allclose(np.dot(b.T, a.T).T, c)

def suite():
    s = TestSuite()
    s.addTest(test_cublasxt('test_cublasXtSgemm'))
    s.addTest(test_cublasxt('test_cublasXtCgemm'))
    if misc.get_compute_capability(pycuda.autoinit.device) >= 1.3:
        s.addTest(test_cublasxt('test_cublasXtDgemm'))
        s.addTest(test_cublasxt('test_cublasXtZgemm'))
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
