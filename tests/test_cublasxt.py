#!/usr/bin/env python

"""
Unit tests for scikits.cuda.cublasxt
"""

from unittest import main, makeSuite, TestCase, TestSuite

import numpy as np

import pycuda.driver as drv
from pycuda.tools import clear_context_caches, make_default_context

import skcuda.cublasxt as cublasxt
import skcuda.misc as misc

drv.init()

class test_cublasxt(TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.ctx = make_default_context()
        cls.handle = cublasxt.cublasXtCreate()
        cls.nbDevices = 1
        cls.deviceId = np.array([0], np.int32)
        cublasxt.cublasXtDeviceSelect(cls.handle, cls.nbDevices,
                                      cls.deviceId)

    @classmethod
    def tearDownClass(cls):
        cublasxt.cublasXtDestroy(cls.handle)
        cls.ctx.pop()
        clear_context_caches()

    def setUp(self):
        np.random.seed(0)

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
    context = make_default_context()
    device = context.get_device()
    context.pop()

    s = TestSuite()
    s.addTest(test_cublasxt('test_cublasXtSgemm'))
    s.addTest(test_cublasxt('test_cublasXtCgemm'))
    if misc.get_compute_capability(device) >= 1.3:
        s.addTest(test_cublasxt('test_cublasXtDgemm'))
        s.addTest(test_cublasxt('test_cublasXtZgemm'))
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
