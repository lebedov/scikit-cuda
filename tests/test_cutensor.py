#!/usr/bin/env python

"""
Unit tests for skcuda.cutensor
"""

import ctypes
from unittest import main, makeSuite, TestCase, TestSuite

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.tools import clear_context_caches, make_default_context
import numpy as np

from numpy.testing import assert_array_equal

import skcuda.cutensor as cutensor

drv.init()

CUDA_R_32F = 0

class test_cutensor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = make_default_context()

    @classmethod
    def tearDownClass(cls):
        cls.ctx.pop()
        clear_context_caches()

    def test_elementwise_binary(self):
        alpha = 1.1
        gamma = 1.2
        _modeA = ['c', 'b', 'a']
        _modeC = ['a', 'b', 'c']
        modeA = np.array([ord(c) for c in _modeA])
        nmodeA = len(modeA)
        modeC = np.array([ord(c) for c in _modeC])
        nmodeC = len(modeC)
        extent = {'a': 400, 'b': 200, 'c': 300}
        extentA = np.array([extent[mode] for mode in _modeA])
        extentC = np.array([extent[mode] for mode in _modeC])
        typeA = typeC = typeCompute = CUDA_R_32F
        descA = cutensor.cutensorCreateTensorDescriptor(nmodeA,
                                                        ctypes.c_void_p(extentA.ctypes.data),
                                                        ctypes.c_void_p(),
                                                        typeA, cutensor.CUTENSOR_OP_IDENTITY,
                                                        1, 0)
        descC = cutensor.cutensorCreateTensorDescriptor(nmodeC,
                                                        ctypes.c_void_p(extentC.ctypes.data),
                                                        ctypes.c_void_p(),
                                                        typeC, cutensor.CUTENSOR_OP_IDENTITY,
                                                        1, 0)
        A = np.random.rand(extentA.prod()).astype(np.float32)
        A_gpu = gpuarray.to_gpu(A)
        C = np.random.rand(extentC.prod()).astype(np.float32)
        C_gpu = gpuarray.to_gpu(C)
        D = np.zeros_like(C)
        handle = cutensor.cutensorCreate()
        cutensor.cutensorElementwiseBinary(handle,
                                           ctypes.POINTER(ctypes.c_float)(ctypes.c_float(alpha)),
                                           A_gpu.gpudata,
                                           ctypes.c_void_p(descA),
                                           ctypes.c_void_p(modeA.ctypes.data),
                                           ctypes.POINTER(ctypes.c_float)(ctypes.c_float(gamma)),
                                           C_gpu.gpudata,
                                           ctypes.c_void_p(descC),
                                           ctypes.c_void_p(modeC.ctypes.data),
                                           C_gpu.gpudata,
                                           ctypes.c_void_p(descC),
                                           ctypes.c_void_p(modeC.ctypes.data),
                                           cutensor.CUTENSOR_OP_ADD,
                                           typeCompute, 0)
        for a in range(extent['a']):
            for b in range(extent['b']):
                for c in range(extent['c']):
                    D[a, b, c] = alpha*A[c, b, a]+gamma*C[a, b, c]
        assert_array_equal(C_gpu.get(), D)

        cutensor.cutensorDestroy(handle)
        cutensor.cutensorDestroyTensorDescriptor(descC)
        cutensor.cutensorDestroyTensorDescriptor(descA)

    def test_elementwise_trinary(self):
        alpha = 1.1
        beta = 1.3
        gamma = 1.2
        _modeA = ['c', 'b', 'a']
        _modeB = ['c', 'a', 'b']
        _modeC = ['a', 'b', 'c']
        modeA = np.array([ord(c) for c in _modeA])
        nmodeA = len(modeA)
        modeB = np.array([ord(c) for c in _modeB])
        nmodeB = len(modeB)
        modeC = np.array([ord(c) for c in _modeC])
        nmodeC = len(modeC)
        extent = {'a': 400, 'b': 200, 'c': 300}
        extentA = np.array([extent[mode] for mode in _modeA])
        extentB = np.array([extent[mode] for mode in _modeB])
        extentC = np.array([extent[mode] for mode in _modeC])
        typeA = typeB = typeC = typeCompute = CUDA_R_32F
        descA = cutensor.cutensorCreateTensorDescriptor(nmodeA,
                                                        ctypes.c_void_p(extentA.ctypes.data),
                                                        ctypes.c_void_p(),
                                                        typeA, cutensor.CUTENSOR_OP_IDENTITY,
                                                        1, 0)
        descB = cutensor.cutensorCreateTensorDescriptor(nmodeB,
                                                        ctypes.c_void_p(extentB.ctypes.data),
                                                        ctypes.c_void_p(),
                                                        typeB, cutensor.CUTENSOR_OP_IDENTITY,
                                                        1, 0)
        descC = cutensor.cutensorCreateTensorDescriptor(nmodeC,
                                                        ctypes.c_void_p(extentC.ctypes.data),
                                                        ctypes.c_void_p(),
                                                        typeC, cutensor.CUTENSOR_OP_IDENTITY,
                                                        1, 0)
        A = np.random.rand(extentA.prod()).astype(np.float32)
        A_gpu = gpuarray.to_gpu(A)
        B = np.random.rand(extentB.prod()).astype(np.float32)
        B_gpu = gpuarray.to_gpu(B)
        C = np.random.rand(extentC.prod()).astype(np.float32)
        C_gpu = gpuarray.to_gpu(C)
        D = np.zeros_like(C)
        D_gpu = gpuarray.to_gpu(D)
        handle = cutensor.cutensorCreate()
        cutensor.cutensorElementwiseTrinary(handle,
                                            ctypes.POINTER(ctypes.c_float)(ctypes.c_float(alpha)),
                                            A_gpu.gpudata,
                                            ctypes.c_void_p(descA),
                                            ctypes.c_void_p(modeA.ctypes.data),
                                            ctypes.POINTER(ctypes.c_float)(ctypes.c_float(beta)),
                                            B_gpu.gpudata,
                                            ctypes.c_void_p(descB),
                                            ctypes.c_void_p(modeB.ctypes.data),
                                            ctypes.POINTER(ctypes.c_float)(ctypes.c_float(gamma)),
                                            C_gpu.gpudata,
                                            ctypes.c_void_p(descC),
                                            ctypes.c_void_p(modeC.ctypes.data),
                                            C_gpu.gpudata,
                                            ctypes.c_void_p(descC),
                                            ctypes.c_void_p(modeC.ctypes.data),
                                            cutensor.CUTENSOR_OP_ADD, cutensor.CUTENSOR_OP_ADD,
                                            typeCompute, 0)
        for a in range(extent['a']):
            for b in range(extent['b']):
                for c in range(extent['c']):
                    D[a, b, c] = alpha*A[c, b, a]+beta*B[c, a, b]*gamma*C[a, b, c]
        assert_array_equal(D_gpu.get(), D)

        cutensor.cutensorDestroy(handle)
        cutensor.cutensorDestroyTensorDescriptor(descC)
        cutensor.cutensorDestroyTensorDescriptor(descA)

    def test_vectorization(self):
        _modeA = ['a', 'b']
        modeA = np.array([ord(c) for c in _modeA])
        _modeB = ['a', 'b']
        modeB = np.array([ord(c) for c in _modeB])
        nmodeA = len(modeA)
        nmodeB = len(modeB)
        extent = {'a': 2, 'b': 3}
        extentA = np.array([extent[mode] for mode in _modeA])
        extentB = np.array([extent[mode] for mode in _modeB])
        typeA = typeB = typeCompute = CUDA_R_32F
        vectorWidthB = 2
        vectorModeIdxB = 1
        descA = cutensor.cutensorCreateTensorDescriptor(nmodeA,
                                                        ctypes.c_void_p(extentA.ctypes.data),
                                                        ctypes.c_void_p(),
                                                        typeA, cutensor.CUTENSOR_OP_IDENTITY,
                                                        1, 0)
        descB = cutensor.cutensorCreateTensorDescriptor(nmodeB,
                                                        ctypes.c_void_p(extentB.ctypes.data),
                                                        ctypes.c_void_p(),
                                                        typeB, cutensor.CUTENSOR_OP_IDENTITY,
                                                        vectorWidthB, vectorModeIdxB)
        elementsA = extentA.prod()
        elementsB = 1
        for mode in _modeB:
            if _modeB[vectorModeIdxB] == mode:
                elementsB *= ((extent[mode]+vectorWidthB-1)//vectorWidthB)*vectorWidthB
            else:
                elementsB *= extent[mode]
        A = np.arange(1, elementsA+1, dtype=np.float32)
        B = np.tile(-1, elementsB).astype(np.float32)
        A_gpu = gpuarray.to_gpu(A)
        B_gpu = gpuarray.to_gpu(B)
        handle = cutensor.cutensorCreate()
        one = 1.0
        cutensor.cutensorPermutation(handle,
                                     ctypes.POINTER(ctypes.c_float)(ctypes.c_float(one)),
                                     A_gpu.gpudata, ctypes.c_void_p(descA),
                                     ctypes.c_void_p(modeA.ctypes.data),
                                     B_gpu.gpudata, ctypes.c_void_p(descB),
                                     ctypes.c_void_p(modeB.ctypes.data),
                                     typeCompute, 0)
        assert_array_equal(B_gpu.get(),
                           np.array([1, 3, 2, 4, 5, 0, 6, 0], np.float32))

        cutensor.cutensorDestroy(handle)
        cutensor.cutensorDestroyTensorDescriptor(descB)
        cutensor.cutensorDestroyTensorDescriptor(descA)

def suite():
    context = make_default_context()
    device = context.get_device()
    context.pop()

    s = TestSuite()
    s.addTest(test_cutensor('test_elementwise_binary'))
    s.addTest(test_cutensor('test_elementwise_trinary'))
    s.addTest(test_cutensor('test_vectorization'))
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')

