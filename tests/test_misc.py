#!/usr/bin/env python

"""
Unit tests for scikits.cuda.misc
"""

from unittest import main, TestCase, TestSuite

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

import scikits.cuda.misc as misc

class test_misc(TestCase):

    def setUp(self):
        np.random.seed(0)
        misc.init()

    def test_maxabs_float32(self):
        x = np.array([-1, 2, -3], np.float32)
        x_gpu = gpuarray.to_gpu(x)
        m_gpu = misc.maxabs(x_gpu)
        assert np.allclose(m_gpu.get(), np.max(np.abs(x)))

    def test_maxabs_float64(self):
        x = np.array([-1, 2, -3], np.float64)
        x_gpu = gpuarray.to_gpu(x)
        m_gpu = misc.maxabs(x_gpu)
        assert np.allclose(m_gpu.get(), np.max(np.abs(x)))

    def test_maxabs_complex64(self):
        x = np.array([-1j, 2, -3j], np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        m_gpu = misc.maxabs(x_gpu)
        assert np.allclose(m_gpu.get(), np.max(np.abs(x)))

    def test_maxabs_complex128(self):
        x = np.array([-1j, 2, -3j], np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        m_gpu = misc.maxabs(x_gpu)
        assert np.allclose(m_gpu.get(), np.max(np.abs(x)))

    def test_cumsum_float32(self):
        x = np.array([1, 4, 3, 2, 8], np.float32)
        x_gpu = gpuarray.to_gpu(x)
        c_gpu = misc.cumsum(x_gpu)
        assert np.allclose(c_gpu.get(), np.cumsum(x))

    def test_cumsum_float64(self):
        x = np.array([1, 4, 3, 2, 8], np.float64)
        x_gpu = gpuarray.to_gpu(x)
        c_gpu = misc.cumsum(x_gpu)
        assert np.allclose(c_gpu.get(), np.cumsum(x))

    def test_cumsum_complex64(self):
        x = np.array([1, 4j, 3, 2j, 8], np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        c_gpu = misc.cumsum(x_gpu)
        assert np.allclose(c_gpu.get(), np.cumsum(x))

    def test_cumsum_complex128(self):
        x = np.array([1, 4j, 3, 2j, 8], np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        c_gpu = misc.cumsum(x_gpu)
        assert np.allclose(c_gpu.get(), np.cumsum(x))

    def test_diff_float32(self):
        x = np.array([1.3, 2.7, 4.9, 5.1], np.float32)
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = misc.diff(x_gpu)
        assert np.allclose(y_gpu.get(), np.diff(x))

    def test_diff_float64(self):
        x = np.array([1.3, 2.7, 4.9, 5.1], np.float64)
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = misc.diff(x_gpu)
        assert np.allclose(y_gpu.get(), np.diff(x))

    def test_diff_complex64(self):
        x = np.array([1.3+2.0j, 2.7-3.9j, 4.9+1.0j, 5.1-9.0j], np.complex64)
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = misc.diff(x_gpu)
        assert np.allclose(y_gpu.get(), np.diff(x))

    def test_diff_complex128(self):
        x = np.array([1.3+2.0j, 2.7-3.9j, 4.9+1.0j, 5.1-9.0j], np.complex128)
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = misc.diff(x_gpu)
        assert np.allclose(y_gpu.get(), np.diff(x))

    def test_get_by_index_float32(self):
        src = np.random.rand(5).astype(np.float32)
        src_gpu = gpuarray.to_gpu(src)
        ind = gpuarray.to_gpu(np.array([0, 2, 4]))
        res_gpu = misc.get_by_index(src_gpu, ind)
        assert np.allclose(res_gpu.get(), src[[0, 2, 4]])

        ind = gpuarray.to_gpu(np.array([], np.int64))
        res_gpu = misc.get_by_index(src_gpu, ind)
        assert len(res_gpu) == 0

    def test_get_by_index_float64(self):
        src = np.random.rand(5).astype(np.float64)
        src_gpu = gpuarray.to_gpu(src)
        ind = gpuarray.to_gpu(np.array([0, 2, 4]))
        res_gpu = misc.get_by_index(src_gpu, ind)
        assert np.allclose(res_gpu.get(), src[[0, 2, 4]])

        ind = gpuarray.to_gpu(np.array([], np.int64))
        res_gpu = misc.get_by_index(src_gpu, ind)
        assert len(res_gpu) == 0

    def test_set_by_index_dest_float32(self):
        dest_gpu = gpuarray.to_gpu(np.arange(5, dtype=np.float32))
        ind = gpuarray.to_gpu(np.array([0, 2, 4]))
        src_gpu = gpuarray.to_gpu(np.array([1, 1, 1], dtype=np.float32))
        misc.set_by_index(dest_gpu, ind, src_gpu, 'dest')
        assert np.allclose(dest_gpu.get(),
                           np.array([1, 1, 1, 3, 1], dtype=np.float32))

        dest_gpu = gpuarray.to_gpu(np.arange(5, dtype=np.float32))
        ind = gpuarray.to_gpu(np.array([], np.int64))
        src_gpu = gpuarray.to_gpu(np.array([1, 1, 1], dtype=np.float32))
        misc.set_by_index(dest_gpu, ind, src_gpu, 'dest')
        assert np.allclose(dest_gpu.get(),
                           np.arange(5, dtype=np.float32))

    def test_set_by_index_dest_float64(self):
        dest_gpu = gpuarray.to_gpu(np.arange(5, dtype=np.double))
        ind = gpuarray.to_gpu(np.array([0, 2, 4]))
        src_gpu = gpuarray.to_gpu(np.array([1, 1, 1], dtype=np.double))
        misc.set_by_index(dest_gpu, ind, src_gpu, 'dest')
        assert np.allclose(dest_gpu.get(),
                           np.array([1, 1, 1, 3, 1], dtype=np.double))

        dest_gpu = gpuarray.to_gpu(np.arange(5, dtype=np.double))
        ind = gpuarray.to_gpu(np.array([], np.int64))
        src_gpu = gpuarray.to_gpu(np.array([1, 1, 1], dtype=np.double))
        misc.set_by_index(dest_gpu, ind, src_gpu, 'dest')
        assert np.allclose(dest_gpu.get(),
                           np.arange(5, dtype=np.double))

    def test_set_by_index_src_float32(self):
        dest_gpu = gpuarray.to_gpu(np.zeros(3, dtype=np.float32))
        ind = gpuarray.to_gpu(np.array([0, 2, 4]))
        src_gpu = gpuarray.to_gpu(np.arange(5, dtype=np.float32))
        misc.set_by_index(dest_gpu, ind, src_gpu, 'src')
        assert np.allclose(dest_gpu.get(),
                           np.array([0, 2, 4], dtype=np.float32))

        dest_gpu = gpuarray.to_gpu(np.arange(5, dtype=np.float32))
        ind = gpuarray.to_gpu(np.array([], np.int64))
        src_gpu = gpuarray.to_gpu(np.array([1, 1, 1], dtype=np.float32))
        misc.set_by_index(dest_gpu, ind, src_gpu, 'src')
        assert np.allclose(dest_gpu.get(),
                           np.arange(5, dtype=np.float32))

    def test_set_by_index_src_float64(self):
        dest_gpu = gpuarray.to_gpu(np.zeros(3, dtype=np.double))
        ind = gpuarray.to_gpu(np.array([0, 2, 4]))
        src_gpu = gpuarray.to_gpu(np.arange(5, dtype=np.double))
        misc.set_by_index(dest_gpu, ind, src_gpu, 'src')
        assert np.allclose(dest_gpu.get(),
                           np.array([0, 2, 4], dtype=np.double))

        dest_gpu = gpuarray.to_gpu(np.arange(5, dtype=np.double))
        ind = gpuarray.to_gpu(np.array([], np.int64))
        src_gpu = gpuarray.to_gpu(np.array([1, 1, 1], dtype=np.double))
        misc.set_by_index(dest_gpu, ind, src_gpu, 'src')
        assert np.allclose(dest_gpu.get(),
                           np.arange(5, dtype=np.double))

    def impl_test_sum(self, dtype):
        x = np.random.normal(scale=5.0, size=(3, 5))
        x = x.astype(dtype=dtype, order='C')
        x_gpu = gpuarray.to_gpu(x)
        assert np.allclose(misc.sum(x_gpu), x.sum())
        assert np.allclose(misc.sum(x_gpu, axis=0).get(), x.sum(axis=0))
        assert np.allclose(misc.sum(x_gpu, axis=1).get(), x.sum(axis=1))
        x = x.astype(dtype=dtype, order='F')
        x_gpu = gpuarray.to_gpu(x)
        assert np.allclose(misc.sum(x_gpu), x.sum())
        assert np.allclose(misc.sum(x_gpu, axis=0).get(), x.sum(axis=0))
        assert np.allclose(misc.sum(x_gpu, axis=1).get(), x.sum(axis=1))

    def test_sum_float32(self):
        self.impl_test_sum(np.float32)

    def test_sum_float64(self):
        self.impl_test_sum(np.float64)

    def test_sum_complex64(self):
        self.impl_test_sum(np.complex64)

    def test_sum_complex128(self):
        self.impl_test_sum(np.complex128)

    def impl_test_mean(self, dtype):
        x = np.random.normal(scale=5.0, size=(3, 5))
        x = x.astype(dtype=dtype, order='C')
        x_gpu = gpuarray.to_gpu(x)
        assert np.allclose(misc.mean(x_gpu), x.mean())
        assert np.allclose(misc.mean(x_gpu, axis=0).get(), x.mean(axis=0))
        assert np.allclose(misc.mean(x_gpu, axis=1).get(), x.mean(axis=1))
        x = x.astype(dtype=dtype, order='F')
        x_gpu = gpuarray.to_gpu(x)
        assert np.allclose(misc.mean(x_gpu), x.mean())
        assert np.allclose(misc.mean(x_gpu, axis=-1).get(), x.mean(axis=-1))
        assert np.allclose(misc.mean(x_gpu, axis=-2).get(), x.mean(axis=-2))

    def test_mean_float32(self):
        self.impl_test_mean(np.float32)

    def test_mean_float64(self):
        self.impl_test_mean(np.float64)

    def test_mean_complex64(self):
        self.impl_test_mean(np.complex64)

    def test_mean_complex128(self):
        self.impl_test_mean(np.complex128)

def suite():
    s = TestSuite()
    s.addTest(test_misc('test_maxabs_float32'))
    s.addTest(test_misc('test_maxabs_complex64'))
    s.addTest(test_misc('test_cumsum_float32'))
    s.addTest(test_misc('test_cumsum_complex64'))
    s.addTest(test_misc('test_diff_float32'))
    s.addTest(test_misc('test_diff_complex64'))
    s.addTest(test_misc('test_get_by_index_float32'))
    s.addTest(test_misc('test_set_by_index_dest_float32'))
    s.addTest(test_misc('test_set_by_index_src_float32'))
    s.addTest(test_misc('test_sum_float32'))
    s.addTest(test_misc('test_sum_complex64'))
    s.addTest(test_misc('test_mean_float32'))
    s.addTest(test_misc('test_mean_complex64'))
    if misc.get_compute_capability(pycuda.autoinit.device) >= 1.3:
        s.addTest(test_misc('test_maxabs_float64'))
        s.addTest(test_misc('test_maxabs_complex128'))
        s.addTest(test_misc('test_cumsum_float64'))
        s.addTest(test_misc('test_cumsum_complex128'))
        s.addTest(test_misc('test_diff_float64'))
        s.addTest(test_misc('test_diff_complex128'))
        s.addTest(test_misc('test_get_by_index_float32'))
        s.addTest(test_misc('test_set_by_index_dest_float64'))
        s.addTest(test_misc('test_set_by_index_src_float64'))
        s.addTest(test_misc('test_sum_float64'))
        s.addTest(test_misc('test_sum_complex128'))
        s.addTest(test_misc('test_mean_float64'))
        s.addTest(test_misc('test_mean_complex128'))
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
