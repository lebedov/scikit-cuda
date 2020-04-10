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

def check_batch_tridiagonal(dl,d,du,x, y, m,batchCount,batchStride):
    """
    Check all solutions from batched tridiagonal routine
    """
    for ii in range(batchCount):
        A_sys = np.diagflat(dl[ii*batchStride+1:ii*batchStride+m], -1) +\
            np.diagflat(d[ii*batchStride:ii*batchStride+m], 0) + \
            np.diagflat(du[ii*batchStride:ii*batchStride+m-1], 1)
        x_sys = x[ii*batchStride:ii*batchStride+m]
        y_sys = y[ii*batchStride:ii*batchStride+m]
        assert(np.allclose(np.dot(A_sys,y_sys), x_sys, atol=1e-3))

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

    # Sgtsv2StridedBatch_bufferSizeExt
    def test_cusparseSgtsv2StridedBatch(self):
        m = 6
        batchCount = 9
        batchStride = 9

        dl = np.zeros(batchStride*batchCount).astype(np.float32)
        d = np.zeros(batchStride*batchCount).astype(np.float32)
        du = np.zeros(batchStride*batchCount).astype(np.float32)
        x = np.zeros(batchStride*batchCount).astype(np.float32)

        for ii in range(batchCount):
            dl[ii*batchStride+1:ii*batchStride+m] = np.random.rand(m-1)
            d[ii*batchStride:ii*batchStride+m] = np.random.rand(m)
            du[ii*batchStride:ii*batchStride+m-1] = np.random.rand(m-1)
            x[ii*batchStride:ii*batchStride+m] = np.random.rand(m)

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

        check_batch_tridiagonal(dl,d,du,x, x_gpu.get(), m,batchCount,batchStride)

def suite():
    s = TestSuite()
    s.addTest(test_cusparse('test_cusparseSgtsv2StridedBatch'))
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
