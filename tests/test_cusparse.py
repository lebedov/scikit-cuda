#!/usr/bin/env python

"""
Unit tests for skcuda.cusparse
"""

from unittest import main, makeSuite, TestCase, TestSuite

import pycuda.gpuarray as gpuarray
from pycuda.tools import clear_context_caches, make_default_context
import numpy as np

import skcuda.cusparse as cusparse

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
    def test_cusparseSgtsv2StridedBatch_bufferSizeExt(self):
        m = 5
        batchCount = 5
        batchStride = m

        dl = np.zeros(m*batchCount).astype(np.float32)
        d = np.zeros(m*batchCount).astype(np.float32)
        du = np.zeros(m*batchCount).astype(np.float32)
        x = np.zeros(m*batchCount).astype(np.float32)

        for ii in range(batchCount):
            dl[ii*batchStride+1:ii*batchStride+batchStride] = np.random.rand(m-1)
            d[ii*batchStride:ii*batchStride+batchStride] = np.random.rand(m)
            du[ii*batchStride:ii*batchStride+batchStride-1] = np.random.rand(m-1)
            x[ii*batchStride:ii*batchStride+batchStride] = np.random.rand(m)

        dl_gpu = gpuarray.to_gpu(dl)
        d_gpu = gpuarray.to_gpu(d)
        du_gpu = gpuarray.to_gpu(du)
        x_gpu = gpuarray.to_gpu(x)

        bufferSizeInBytes = cusparse.cusparseSgtsv2StridedBatch_bufferSizeExt(
            self.cusparse_handle, m, dl_gpu.gpudata, d_gpu.gpudata,
            du_gpu.gpudata, x_gpu.gpudata, batchCount, batchStride)

def suite():
    s = TestSuite()
    s.addTest(test_cusparse('test_cusparseSgtsv2StridedBatch_bufferSizeExt'))
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
