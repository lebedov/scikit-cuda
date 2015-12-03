#!/usr/bin/env python

"""
Unit tests for scikits.cuda.misc
"""

import numbers
from unittest import main, TestCase, TestSuite

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
from numpy.testing import assert_raises
import skcuda.misc as misc

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

    def impl_test_binaryop_2d(self, dtype):
        if issubclass(dtype, numbers.Integral):
            a_sca = np.array(np.random.randint(1, 10), dtype=dtype)
            b_sca = np.array(np.random.randint(1, 10), dtype=dtype)
            a_vec = np.random.randint(1, 10, 3).astype(dtype)
            b_vec = np.random.randint(1, 10, 3).astype(dtype)
            a_mat = np.random.randint(1, 10, 6).reshape((3, 2)).astype(dtype)
            b_mat = np.random.randint(1, 10, 6).reshape((3, 2)).astype(dtype)            
        else:
            a_sca = np.random.normal(scale=5.0, size=()).astype(dtype)
            b_sca = np.random.normal(scale=5.0, size=()).astype(dtype)
            a_vec = np.random.normal(scale=5.0, size=(3,)).astype(dtype)
            b_vec = np.random.normal(scale=5.0, size=(3,)).astype(dtype)
            a_mat = np.random.normal(scale=5.0, size=(3, 2)).astype(dtype)
            b_mat = np.random.normal(scale=5.0, size=(3, 2)).astype(dtype)

        a_sca_gpu = gpuarray.to_gpu(a_sca)
        b_sca_gpu = gpuarray.to_gpu(b_sca)
        a_vec_gpu = gpuarray.to_gpu(a_vec)
        b_vec_gpu = gpuarray.to_gpu(b_vec)
        a_mat_gpu = gpuarray.to_gpu(a_mat)
        b_mat_gpu = gpuarray.to_gpu(b_mat)

        # addition
        assert np.allclose(misc.add(a_sca_gpu, b_sca_gpu).get(), a_sca+b_sca)
        assert np.allclose(misc.add(a_vec_gpu, b_vec_gpu).get(), a_vec+b_vec)
        assert np.allclose(misc.add(a_mat_gpu, b_mat_gpu).get(), a_mat+b_mat)

        # subtract
        assert np.allclose(misc.subtract(a_sca_gpu, b_sca_gpu).get(), a_sca-b_sca)
        assert np.allclose(misc.subtract(a_vec_gpu, b_vec_gpu).get(), a_vec-b_vec)
        assert np.allclose(misc.subtract(a_mat_gpu, b_mat_gpu).get(), a_mat-b_mat)

        # multiplication
        assert np.allclose(misc.multiply(a_sca_gpu, b_sca_gpu).get(), a_sca*b_sca)
        assert np.allclose(misc.multiply(a_vec_gpu, b_vec_gpu).get(), a_vec*b_vec)
        assert np.allclose(misc.multiply(a_mat_gpu, b_mat_gpu).get(), a_mat*b_mat)

        # division
        assert np.allclose(misc.divide(a_sca_gpu, b_sca_gpu).get(), a_sca/b_sca)
        assert np.allclose(misc.divide(a_vec_gpu, b_vec_gpu).get(), a_vec/b_vec)
        assert np.allclose(misc.divide(a_mat_gpu, b_mat_gpu).get(), a_mat/b_mat)

    def test_binaryop_2d_int32(self):
        self.impl_test_binaryop_2d(np.int32)

    def test_binaryop_2d_float32(self):
        self.impl_test_binaryop_2d(np.float32)

    def test_binaryop_2d_float32(self):
        self.impl_test_binaryop_2d(np.float32)

    def test_binaryop_2d_float64(self):
        self.impl_test_binaryop_2d(np.float64)

    def test_binaryop_2d_complex64(self):
        self.impl_test_binaryop_2d(np.complex64)

    def test_binaryop_2d_complex128(self):
        self.impl_test_binaryop_2d(np.complex128)

    def impl_test_binaryop_matvec(self, dtype):
        if issubclass(dtype, numbers.Integral):
            x = np.random.randint(1, 10, 15).reshape((3, 5)).astype(dtype)
            a = np.random.randint(1, 10, 5).reshape((1, 5)).astype(dtype)
            b = np.random.randint(1, 10, 3).reshape((3, 1)).astype(dtype)

            # the following two test correct broadcasting on 0D vectors
            c = np.random.randint(1, 10, 5).reshape((5, )).astype(dtype)
            d = np.random.randint(1, 10, 3).reshape((3, )).astype(dtype)
        else:
            x = np.random.normal(scale=5.0, size=(3, 5)).astype(dtype)
            a = np.random.normal(scale=5.0, size=(1, 5)).astype(dtype)
            b = np.random.normal(scale=5.0, size=(3, 1)).astype(dtype)

            # the following two test correct broadcasting on 0D vectors
            c = np.random.normal(scale=5.0, size=(5, )).astype(dtype)
            d = np.random.normal(scale=5.0, size=(3, )).astype(dtype)
        x_gpu = gpuarray.to_gpu(x)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)
        d_gpu = gpuarray.to_gpu(d)
        out = gpuarray.empty(x.shape, dtype=dtype)
        # addition
        res = misc.add_matvec(x_gpu, a_gpu, out=out).get()
        assert np.allclose(res, x+a)
        assert np.allclose(misc.add_matvec(x_gpu, b_gpu).get(), x+b)
        assert np.allclose(misc.add_matvec(x_gpu, c_gpu).get(), x+c)
        assert_raises(ValueError, misc.add_matvec, x_gpu, d_gpu)
        # multiplication
        res = misc.mult_matvec(x_gpu, a_gpu, out=out).get()
        assert np.allclose(res, x*a)
        assert np.allclose(misc.mult_matvec(x_gpu, b_gpu).get(), x*b)
        assert np.allclose(misc.mult_matvec(x_gpu, c_gpu).get(), x*c)
        assert_raises(ValueError, misc.mult_matvec, x_gpu, d_gpu)
        # division
        res = misc.div_matvec(x_gpu, a_gpu, out=out).get()
        assert np.allclose(res, x/a)
        assert np.allclose(misc.div_matvec(x_gpu, b_gpu).get(), x/b)
        assert np.allclose(misc.div_matvec(x_gpu, c_gpu).get(), x/c)
        assert_raises(ValueError, misc.div_matvec, x_gpu, d_gpu)

    def test_binaryop_matvec_int32(self):
        self.impl_test_binaryop_matvec(np.int32)

    def test_binaryop_matvec_float32(self):
        self.impl_test_binaryop_matvec(np.float32)

    def test_binaryop_matvec_float64(self):
        self.impl_test_binaryop_matvec(np.float64)

    def test_binaryop_matvec_complex64(self):
        self.impl_test_binaryop_matvec(np.complex64)

    def test_binaryop_matvec_complex128(self):
        self.impl_test_binaryop_matvec(np.complex128)

    def impl_test_sum(self, dtype):
        x = np.random.normal(scale=5.0, size=(3, 5))
        x = x.astype(dtype=dtype, order='C')
        x_gpu = gpuarray.to_gpu(x)
        assert np.allclose(misc.sum(x_gpu).get(), x.sum())
        assert np.allclose(misc.sum(x_gpu, axis=0).get(), x.sum(axis=0))
        assert np.allclose(misc.sum(x_gpu, axis=1).get(), x.sum(axis=1))
        x = x.astype(dtype=dtype, order='F')
        x_gpu = gpuarray.to_gpu(x)
        assert np.allclose(misc.sum(x_gpu).get(), x.sum())
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
        assert np.allclose(misc.mean(x_gpu).get(), x.mean())
        assert np.allclose(misc.mean(x_gpu, axis=0).get(), x.mean(axis=0))
        assert np.allclose(misc.mean(x_gpu, axis=1).get(), x.mean(axis=1))
        x = x.astype(dtype=dtype, order='F')
        x_gpu = gpuarray.to_gpu(x)
        assert np.allclose(misc.mean(x_gpu).get(), x.mean())
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

    def impl_test_var(self, dtype):
        x = np.random.normal(scale=5.0, size=(3, 5))
        x = x.astype(dtype=dtype, order='C')
        x_gpu = gpuarray.to_gpu(x)
        assert np.allclose(misc.var(x_gpu).get(), x.var())
        assert np.allclose(misc.var(x_gpu, axis=0).get(), x.var(axis=0))
        assert np.allclose(misc.var(x_gpu, axis=1).get(), x.var(axis=1))

        assert np.allclose(misc.var(x_gpu, ddof=1).get(), x.var(ddof=1))
        assert np.allclose(misc.var(x_gpu, ddof=1, axis=0).get(),
                           x.var(ddof=1, axis=0))
        assert np.allclose(misc.var(x_gpu, ddof=1, axis=1).get(),
                           x.var(ddof=1, axis=1))
        # Currently not working due to a bug in PyCUDA, see Issue #92
        #x = x.astype(dtype=dtype, order='F')
        #x_gpu = gpuarray.to_gpu(x)
        #assert np.allclose(misc.var(x_gpu).get(), x.var())
        #assert np.allclose(misc.var(x_gpu, axis=-1).get(), x.var(axis=-1))
        #assert np.allclose(misc.var(x_gpu, axis=-2).get(), x.var(axis=-2))

    def test_var_float32(self):
        self.impl_test_var(np.float32)

    def test_var_float64(self):
        self.impl_test_var(np.float64)

    def test_var_complex64(self):
        self.impl_test_var(np.complex64)

    def test_var_complex128(self):
        self.impl_test_var(np.complex128)

    def impl_test_std(self, dtype):
        x = np.random.normal(scale=5.0, size=(3, 5))
        x = x.astype(dtype=dtype, order='C')
        x_gpu = gpuarray.to_gpu(x)
        assert np.allclose(misc.std(x_gpu).get(), x.std())
        assert np.allclose(misc.std(x_gpu, axis=0).get(), x.std(axis=0))
        assert np.allclose(misc.std(x_gpu, axis=1).get(), x.std(axis=1))

        assert np.allclose(misc.std(x_gpu, ddof=1).get(), x.std(ddof=1))
        assert np.allclose(misc.std(x_gpu, ddof=1, axis=0).get(),
                           x.std(ddof=1, axis=0))
        assert np.allclose(misc.std(x_gpu, ddof=1, axis=1).get(),
                           x.std(ddof=1, axis=1))
        # Currently not working due to a bug in PyCUDA, see Issue #92
        #x = x.astype(dtype=dtype, order='F')
        #x_gpu = gpuarray.to_gpu(x)
        #assert np.allclose(misc.std(x_gpu).get(), x.std())
        #assert np.allclose(misc.std(x_gpu, axis=-1).get(), x.std(axis=-1))
        #assert np.allclose(misc.std(x_gpu, axis=-2).get(), x.std(axis=-2))

    def test_std_float32(self):
        self.impl_test_std(np.float32)

    def test_std_float64(self):
        self.impl_test_std(np.float64)

    def test_std_complex64(self):
        self.impl_test_std(np.complex64)

    def test_std_complex128(self):
        self.impl_test_std(np.complex128)

    def _impl_test_minmax(self, dtype):
        x = np.random.normal(scale=5.0, size=(3, 5))
        x = x.astype(dtype=dtype, order='C')
        x_gpu = gpuarray.to_gpu(x)
        assert np.allclose(misc.max(x_gpu, axis=0).get(), x.max(axis=0))
        assert np.allclose(misc.max(x_gpu, axis=1).get(), x.max(axis=1))
        assert np.allclose(misc.min(x_gpu, axis=0).get(), x.min(axis=0))
        assert np.allclose(misc.min(x_gpu, axis=1).get(), x.min(axis=1))

        x = x.astype(dtype=dtype, order='F')
        x_gpu = gpuarray.to_gpu(x)
        assert np.allclose(misc.max(x_gpu, axis=0).get(), x.max(axis=0))
        assert np.allclose(misc.max(x_gpu, axis=1).get(), x.max(axis=1))
        assert np.allclose(misc.min(x_gpu, axis=0).get(), x.min(axis=0))
        assert np.allclose(misc.min(x_gpu, axis=1).get(), x.min(axis=1))

    def test_minmax_float32(self):
        self._impl_test_minmax(np.float32)

    def test_minmax_float64(self):
        self._impl_test_minmax(np.float64)

    def _impl_test_argminmax(self, dtype):
        x = np.random.normal(scale=5.0, size=(3, 5))
        x = x.astype(dtype=dtype, order='C')
        x_gpu = gpuarray.to_gpu(x)
        assert np.allclose(misc.argmax(x_gpu, axis=0).get(), x.argmax(axis=0))
        assert np.allclose(misc.argmax(x_gpu, axis=1).get(), x.argmax(axis=1))
        assert np.allclose(misc.argmin(x_gpu, axis=0).get(), x.argmin(axis=0))
        assert np.allclose(misc.argmin(x_gpu, axis=1).get(), x.argmin(axis=1))

        x = x.astype(dtype=dtype, order='F')
        x_gpu = gpuarray.to_gpu(x)
        assert np.allclose(misc.argmax(x_gpu, axis=0).get(), x.argmax(axis=0))
        assert np.allclose(misc.argmax(x_gpu, axis=1).get(), x.argmax(axis=1))
        assert np.allclose(misc.argmin(x_gpu, axis=0).get(), x.argmin(axis=0))
        assert np.allclose(misc.argmin(x_gpu, axis=1).get(), x.argmin(axis=1))

    def test_argminmax_float32(self):
        self._impl_test_argminmax(np.float32)

    def test_argminmax_float64(self):
        self._impl_test_argminmax(np.float64)

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
    s.addTest(test_misc('test_binaryop_2d_int32'))
    s.addTest(test_misc('test_binaryop_2d_float32'))
    s.addTest(test_misc('test_binaryop_2d_complex64'))
    s.addTest(test_misc('test_binaryop_matvec_int32'))
    s.addTest(test_misc('test_binaryop_matvec_float32'))
    s.addTest(test_misc('test_binaryop_matvec_complex64'))
    s.addTest(test_misc('test_sum_float32'))
    s.addTest(test_misc('test_sum_complex64'))
    s.addTest(test_misc('test_mean_float32'))
    s.addTest(test_misc('test_mean_complex64'))
    s.addTest(test_misc('test_var_float32'))
    s.addTest(test_misc('test_var_complex64'))
    s.addTest(test_misc('test_std_float32'))
    s.addTest(test_misc('test_std_complex64'))
    s.addTest(test_misc('test_minmax_float32'))
    s.addTest(test_misc('test_argminmax_float32'))
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
        s.addTest(test_misc('test_binaryop_2d_float64'))
        s.addTest(test_misc('test_binaryop_2d_complex128'))
        s.addTest(test_misc('test_binaryop_matvec_float64'))
        s.addTest(test_misc('test_binaryop_matvec_complex128'))
        s.addTest(test_misc('test_var_float64'))
        s.addTest(test_misc('test_var_complex128'))
        s.addTest(test_misc('test_std_float64'))
        s.addTest(test_misc('test_std_complex128'))
    s.addTest(test_misc('test_minmax_float64'))
    s.addTest(test_misc('test_argminmax_float64'))
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
