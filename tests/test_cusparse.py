#!/usr/bin/env python

"""
Unit tests for skcuda.cusparse
"""

from unittest import main, TestCase, TestSuite

import pycuda
import pycuda.gpuarray as gpuarray
from pycuda.tools import clear_context_caches, make_default_context
import numpy as np

import skcuda.cusparse as cusparse

def check_batch_tridiagonal(dl,d,du,x, y, m,batchCount,batchStride=None,atol=1e-8):
    """
    Check all solutions from batched tridiagonal routine
    """
    if batchStride is None:
        batchStride = m

    for ii in range(batchCount):
        A_sys = np.diagflat(dl[ii*batchStride+1:ii*batchStride+m], -1) +\
            np.diagflat(d[ii*batchStride:ii*batchStride+m], 0) + \
            np.diagflat(du[ii*batchStride:ii*batchStride+m-1], 1)
        x_sys = x[ii*batchStride:ii*batchStride+m]
        y_sys = y[ii*batchStride:ii*batchStride+m]
        assert(np.allclose(np.dot(A_sys,y_sys), x_sys, atol=atol))

def tridiagonal_system(m, batchCount, batchStride=None, seed=None,
                       dtype=np.float32):
    """
    Create a tridiagonal system of a given size
    """
    if batchStride is None:
        batchStride = m
    if seed is not None:
        np.random.seed(seed)

    dl = np.zeros(batchStride*batchCount).astype(dtype)
    d = np.zeros(batchStride*batchCount).astype(dtype)
    du = np.zeros(batchStride*batchCount).astype(dtype)
    x = np.zeros(batchStride*batchCount).astype(dtype)

    for ii in range(batchCount):
        dl[ii*batchStride+1:ii*batchStride+m] = np.random.rand(m-1)
        d[ii*batchStride:ii*batchStride+m] = np.random.rand(m)
        du[ii*batchStride:ii*batchStride+m-1] = np.random.rand(m-1)
        x[ii*batchStride:ii*batchStride+m] = np.random.rand(m)

    return dl,d,du,x

class test_cusparse(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = make_default_context()
        cls.cusparse_handle = cusparse.cusparseCreate()

    @classmethod
    def tearDownClass(cls):
        cusparse.cusparseDestroy(cls.cusparse_handle)
        cls.ctx.pop()
        clear_context_caches()

    def setUp(self):
        np.random.seed(23)    # For reproducible tests.

    def test_cusparseSgtsv2StridedBatch(self):
        m = 6
        batchCount = 9
        batchStride = 11

        dl,d,du,x = tridiagonal_system(m, batchCount, batchStride, seed=23)

        dl_gpu = gpuarray.to_gpu(dl)
        d_gpu = gpuarray.to_gpu(d)
        du_gpu = gpuarray.to_gpu(du)
        x_gpu = gpuarray.to_gpu(x)

        bufferSizeInBytes = cusparse.cusparseSgtsv2StridedBatch_bufferSizeExt(
            self.cusparse_handle, m, dl_gpu.gpudata, d_gpu.gpudata,
            du_gpu.gpudata, x_gpu.gpudata, batchCount, batchStride)
        pBuffer = pycuda.driver.mem_alloc(bufferSizeInBytes)

        cusparse.cusparseSgtsv2StridedBatch(self.cusparse_handle, m,
            dl_gpu.gpudata, d_gpu.gpudata, du_gpu.gpudata, x_gpu.gpudata,
            batchCount, batchStride, pBuffer)

        sln = x_gpu.get()
        # For unstable algorithms, need to loosen atol
        check_batch_tridiagonal(dl,d,du,x, sln, m,batchCount,batchStride, 1e-4)

    def test_cusparseDgtsv2StridedBatch(self):
        m = 6
        batchCount = 9
        batchStride = 11

        dl,d,du,x = tridiagonal_system(m, batchCount, batchStride, seed=23,
                                       dtype=np.float64)

        dl_gpu = gpuarray.to_gpu(dl)
        d_gpu = gpuarray.to_gpu(d)
        du_gpu = gpuarray.to_gpu(du)
        x_gpu = gpuarray.to_gpu(x)

        bufferSizeInBytes = cusparse.cusparseDgtsv2StridedBatch_bufferSizeExt(
            self.cusparse_handle, m, dl_gpu.gpudata, d_gpu.gpudata,
            du_gpu.gpudata, x_gpu.gpudata, batchCount, batchStride)
        pBuffer = pycuda.driver.mem_alloc(bufferSizeInBytes)

        cusparse.cusparseDgtsv2StridedBatch(self.cusparse_handle, m,
            dl_gpu.gpudata, d_gpu.gpudata, du_gpu.gpudata, x_gpu.gpudata,
            batchCount, batchStride, pBuffer)

        sln = x_gpu.get()
        # For unstable algorithms, need to loosen atol
        check_batch_tridiagonal(dl,d,du,x, sln, m,batchCount,batchStride)

    def test_cusparseSgtsvInterleavedBatch(self):
        m = 6
        batchCount = 9

        dl,d,du,x = tridiagonal_system(m, batchCount, seed=23)

        # Convert to interleaved format, by switching from row-major order
        # (numpy default) to column-major
        dl_int = np.reshape(dl,(batchCount,m)).ravel('F')
        d_int = np.reshape(d, (batchCount, m)).ravel('F')
        du_int = np.reshape(du,(batchCount,m)).ravel('F')
        x_int = np.reshape(x,(batchCount,m)).ravel('F')

        for algo in range(3):
            dl_int_gpu = gpuarray.to_gpu(dl_int)
            d_int_gpu = gpuarray.to_gpu(d_int)
            du_int_gpu = gpuarray.to_gpu(du_int)
            x_int_gpu = gpuarray.to_gpu(x_int)

            pBufferSizeInBytes = cusparse.cusparseSgtsvInterleavedBatch_bufferSizeExt(self.cusparse_handle,
                algo, m, dl_int_gpu.gpudata, d_int_gpu.gpudata,
                du_int_gpu.gpudata, x_int_gpu.gpudata, batchCount)
                
            pBuffer = pycuda.driver.mem_alloc(pBufferSizeInBytes)

            cusparse.cusparseSgtsvInterleavedBatch(self.cusparse_handle, algo, m,
                dl_int_gpu.gpudata, d_int_gpu.gpudata, du_int_gpu.gpudata,
                x_int_gpu.gpudata, batchCount, pBuffer)
    
            sln_int = x_int_gpu.get()
            # Convert back from interleaved format
            sln = np.reshape(sln_int,(m,batchCount)).ravel('F')
            check_batch_tridiagonal(dl,d,du,x, sln, m,batchCount, atol=1e-6)

    def test_cusparseDgtsvInterleavedBatch(self):
        m = 6
        batchCount = 9

        dl,d,du,x = tridiagonal_system(m, batchCount, seed=23,
                                       dtype=np.float64)

        # Convert to interleaved format, by switching from row-major order
        # (numpy default) to column-major
        dl_int = np.reshape(dl,(batchCount,m)).ravel('F')
        d_int = np.reshape(d, (batchCount, m)).ravel('F')
        du_int = np.reshape(du,(batchCount,m)).ravel('F')
        x_int = np.reshape(x,(batchCount,m)).ravel('F')

        for algo in range(3):
            dl_int_gpu = gpuarray.to_gpu(dl_int)
            d_int_gpu = gpuarray.to_gpu(d_int)
            du_int_gpu = gpuarray.to_gpu(du_int)
            x_int_gpu = gpuarray.to_gpu(x_int)

            pBufferSizeInBytes = cusparse.cusparseDgtsvInterleavedBatch_bufferSizeExt(self.cusparse_handle,
                algo, m, dl_int_gpu.gpudata, d_int_gpu.gpudata,
                du_int_gpu.gpudata, x_int_gpu.gpudata, batchCount)
                
            pBuffer = pycuda.driver.mem_alloc(pBufferSizeInBytes)

            cusparse.cusparseDgtsvInterleavedBatch(self.cusparse_handle, algo, m,
                dl_int_gpu.gpudata, d_int_gpu.gpudata, du_int_gpu.gpudata,
                x_int_gpu.gpudata, batchCount, pBuffer)
    
            sln_int = x_int_gpu.get()
            # Convert back from interleaved format
            sln = np.reshape(sln_int,(m,batchCount)).ravel('F')
            check_batch_tridiagonal(dl,d,du,x, sln, m,batchCount)
        
    def test_cusparseGetSetStream(self):
        initial_stream = cusparse.cusparseGetStream(self.cusparse_handle)
        # Switch stream
        cusparse.cusparseSetStream(self.cusparse_handle, initial_stream+1)
        final_stream = cusparse.cusparseGetStream(self.cusparse_handle)
        assert(final_stream == initial_stream+1)

    def test_cusparseGetVersion(self):
        cusparse.cusparseGetVersion(self.cusparse_handle)

def suite():
    s = TestSuite()
    s.addTest(test_cusparse('test_cusparseSgtsv2StridedBatch'))
    s.addTest(test_cusparse('test_cusparseDgtsv2StridedBatch'))
    s.addTest(test_cusparse('test_cusparseSgtsvInterleavedBatch'))
    s.addTest(test_cusparse('test_cusparseDgtsvInterleavedBatch'))

    s.addTest(test_cusparse('test_cusparseGetSetStream'))
    s.addTest(test_cusparse('test_cusparseGetVersion'))
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
