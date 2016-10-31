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
import skcuda.misc as misc

atol_float32 = 1e-6
atol_float64 = 1e-8

class test_linalg(TestCase):
    def setUp(self):
        np.random.seed(0)
        linalg.init()

    def tearDown(self):
        linalg.shutdown()

    def test_svd_ss_cula_float32(self):
        a = np.asarray(np.random.randn(9, 6), np.float32)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 's', 's')
        assert np.allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                           atol=atol_float32)

    def test_svd_ss_cula_float64(self):
        a = np.asarray(np.random.randn(9, 6), np.float64)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 's', 's')
        assert np.allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                           atol=atol_float64)

    def test_svd_ss_cula_complex64(self):
        a = np.asarray(np.random.randn(9, 6) + 1j*np.random.randn(9, 6), np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 's', 's')
        assert np.allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                           atol=atol_float32)

    def test_svd_ss_cula_complex128(self):
        a = np.asarray(np.random.randn(9, 6) + 1j*np.random.randn(9, 6), np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 's', 's')
        assert np.allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                           atol=atol_float64)

    def test_svd_so_cula_float32(self):
        a = np.asarray(np.random.randn(6, 6), np.float32)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 's', 'o')
        assert np.allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                           atol=atol_float32)

    def test_svd_so_cula_float64(self):
        a = np.asarray(np.random.randn(6, 6), np.float64)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 's', 'o')
        assert np.allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                           atol=atol_float64)

    def test_svd_so_cula_complex64(self):
        a = np.asarray(np.random.randn(6, 6) + 1j*np.random.randn(6, 6), np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 's', 'o')
        assert np.allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                           atol=atol_float32)

    def test_svd_so_cula_complex128(self):
        a = np.asarray(np.random.randn(6, 6) + 1j*np.random.randn(6, 6), np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 's', 'o')
        assert np.allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                           atol=atol_float64)

    def test_svd_aa_cusolver_float32(self):
        a = np.asarray(np.random.randn(6, 6), np.float32)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, lib='cusolver')
        assert np.allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                           atol=atol_float32)

    def test_svd_aa_cusolver_float64(self):
        a = np.asarray(np.random.randn(6, 6), np.float64)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, lib='cusolver')
        assert np.allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                           atol=atol_float64)

    def test_svd_aa_cusolver_complex64(self):
        a = np.asarray(np.random.randn(6, 6) + 1j*np.random.randn(6, 6), np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, lib='cusolver')
        assert np.allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                           atol=atol_float32)

    def test_svd_aa_cusolver_complex128(self):
        a = np.asarray(np.random.randn(6, 6) + 1j*np.random.randn(6, 6), np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, lib='cusolver')
        assert np.allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                           atol=atol_float64)

    def _dot_matrix_vector_tests(self, dtype):
        a = np.asarray(np.random.rand(4, 4), dtype)
        b = np.asarray(np.random.rand(4), dtype)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = linalg.dot(a_gpu, b_gpu)
        assert np.allclose(np.dot(a, b), c_gpu.get())

        a = np.asarray(np.random.rand(4), dtype)
        b = np.asarray(np.random.rand(4, 4), dtype)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = linalg.dot(a_gpu, b_gpu)
        assert np.allclose(np.dot(a, b), c_gpu.get())

        a = np.asarray(np.random.rand(4, 4), dtype)
        b = np.asarray(np.random.rand(4, 1), dtype)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = linalg.dot(a_gpu, b_gpu)
        assert np.allclose(np.dot(a, b), c_gpu.get())

    def test_dot_matrix_vector_float32(self):
        self._dot_matrix_vector_tests(np.float32)

    def test_dot_matrix_vector_float64(self):
        self._dot_matrix_vector_tests(np.float64)

    def test_dot_matrix_vector_complex64(self):
        self._dot_matrix_vector_tests(np.complex64)

    def test_dot_matrix_vector_complex128(self):
        self._dot_matrix_vector_tests(np.complex128)

    def _dot_matrix_tests(self, dtype, transa, transb):
        a = np.asarray(np.random.rand(4, 2), dtype)
        if transa == 'n':
            b = np.asarray(np.random.rand(2, 2), dtype)
        else:
            b = np.asarray(np.random.rand(4, 4), dtype)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = linalg.dot(a_gpu, b_gpu, transa, transb)
        aa = a if transa == 'n' else a.T
        bb = b if transb == 'n' else b.T
        assert np.allclose(np.dot(aa, bb), c_gpu.get())
        a = a.astype(dtype, order="F", copy=True)
        b = b.astype(dtype, order="F", copy=True)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = linalg.dot(a_gpu, b_gpu, transa, transb)
        assert np.allclose(np.dot(aa, bb), c_gpu.get())

    def test_dot_matrix_float32(self):
        self._dot_matrix_tests(np.float32, 'n', 'n')
        self._dot_matrix_tests(np.float32, 'n', 't')
        self._dot_matrix_tests(np.float32, 't', 'n')
        self._dot_matrix_tests(np.float32, 't', 't')

    def test_dot_matrix_float64(self):
        self._dot_matrix_tests(np.float64, 'n', 'n')
        self._dot_matrix_tests(np.float64, 'n', 't')
        self._dot_matrix_tests(np.float64, 't', 'n')
        self._dot_matrix_tests(np.float64, 't', 't')

    def test_dot_matrix_complex64(self):
        self._dot_matrix_tests(np.complex64, 'n', 'n')
        self._dot_matrix_tests(np.complex64, 'n', 't')
        self._dot_matrix_tests(np.complex64, 't', 'n')
        self._dot_matrix_tests(np.complex64, 't', 't')

    def test_dot_matrix_complex128(self):
        self._dot_matrix_tests(np.complex128, 'n', 'n')
        self._dot_matrix_tests(np.complex128, 'n', 't')
        self._dot_matrix_tests(np.complex128, 't', 'n')
        self._dot_matrix_tests(np.complex128, 't', 't')

    def test_dot_matrix_h_complex64(self):
        a = np.asarray(np.random.rand(2, 4)+1j*np.random.rand(2, 4), np.complex64)
        b = np.asarray(np.random.rand(2, 2)+1j*np.random.rand(2, 2), np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = linalg.dot(a_gpu, b_gpu, 'c')
        assert np.allclose(np.dot(a.conj().T, b), c_gpu.get())
        a = a.astype(np.complex64, order="F", copy=True)
        b = b.astype(np.complex64, order="F", copy=True)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = linalg.dot(a_gpu, b_gpu, 'c')
        assert np.allclose(np.dot(a.conj().T, b), c_gpu.get())

    def test_dot_matrix_h_complex128(self):
        a = np.asarray(np.random.rand(2, 4)+1j*np.random.rand(2, 4), np.complex128)
        b = np.asarray(np.random.rand(2, 2)+1j*np.random.rand(2, 2), np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = linalg.dot(a_gpu, b_gpu, 'c')
        assert np.allclose(np.dot(a.conj().T, b), c_gpu.get())
        a = a.astype(np.complex128, order="F", copy=True)
        b = b.astype(np.complex128, order="F", copy=True)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = linalg.dot(a_gpu, b_gpu, 'c')
        assert np.allclose(np.dot(a.conj().T, b), c_gpu.get())

    def test_dot_vector_float32(self):
        a = np.asarray(np.random.rand(5), np.float32)
        b = np.asarray(np.random.rand(5), np.float32)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c = linalg.dot(a_gpu, b_gpu)
        assert np.allclose(np.dot(a, b), c)
        a = a.astype(np.float32, order="F", copy=True)
        b = b.astype(np.float32, order="F", copy=True)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c = linalg.dot(a_gpu, b_gpu)
        assert np.allclose(np.dot(a, b), c)

    def test_dot_vector_float64(self):
        a = np.asarray(np.random.rand(5), np.float64)
        b = np.asarray(np.random.rand(5), np.float64)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c = linalg.dot(a_gpu, b_gpu)
        assert np.allclose(np.dot(a, b), c)
        a = a.astype(np.float64, order="F", copy=True)
        b = b.astype(np.float64, order="F", copy=True)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c = linalg.dot(a_gpu, b_gpu)
        assert np.allclose(np.dot(a, b), c)

    def test_dot_vector_complex64(self):
        a = np.asarray(np.random.rand(5), np.complex64)
        b = np.asarray(np.random.rand(5), np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c = linalg.dot(a_gpu, b_gpu)
        assert np.allclose(np.dot(a, b), c)
        a = a.astype(np.complex64, order="F", copy=True)
        b = b.astype(np.complex64, order="F", copy=True)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c = linalg.dot(a_gpu, b_gpu)
        assert np.allclose(np.dot(a, b), c)

    def test_dot_vector_complex128(self):
        a = np.asarray(np.random.rand(5), np.complex128)
        b = np.asarray(np.random.rand(5), np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c = linalg.dot(a_gpu, b_gpu)
        assert np.allclose(np.dot(a, b), c)
        a = a.astype(np.complex128, order="F", copy=True)
        b = b.astype(np.complex128, order="F", copy=True)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c = linalg.dot(a_gpu, b_gpu)
        assert np.allclose(np.dot(a, b), c)

    def test_mdot_matrix_float32(self):
        a = np.asarray(np.random.rand(4, 2), np.float32)
        b = np.asarray(np.random.rand(2, 2), np.float32)
        c = np.asarray(np.random.rand(2, 2), np.float32)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)
        d_gpu = linalg.mdot(a_gpu, b_gpu, c_gpu)
        assert np.allclose(np.dot(a, np.dot(b, c)), d_gpu.get())

    def test_mdot_matrix_float64(self):
        a = np.asarray(np.random.rand(4, 2), np.float64)
        b = np.asarray(np.random.rand(2, 2), np.float64)
        c = np.asarray(np.random.rand(2, 2), np.float64)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)
        d_gpu = linalg.mdot(a_gpu, b_gpu, c_gpu)
        assert np.allclose(np.dot(a, np.dot(b, c)), d_gpu.get())

    def test_mdot_matrix_complex64(self):
        a = np.asarray(np.random.rand(4, 2), np.complex64)
        b = np.asarray(np.random.rand(2, 2), np.complex64)
        c = np.asarray(np.random.rand(2, 2), np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)
        d_gpu = linalg.mdot(a_gpu, b_gpu, c_gpu)
        assert np.allclose(np.dot(a, np.dot(b, c)), d_gpu.get())

    def test_mdot_matrix_complex128(self):
        a = np.asarray(np.random.rand(4, 2), np.complex128)
        b = np.asarray(np.random.rand(2, 2), np.complex128)
        c = np.asarray(np.random.rand(2, 2), np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)
        d_gpu = linalg.mdot(a_gpu, b_gpu, c_gpu)
        assert np.allclose(np.dot(a, np.dot(b, c)), d_gpu.get())


    def __impl_test_dot_diag(self, dtype):
        d = np.asarray(np.random.rand(5), dtype)
        a = np.asarray(np.random.rand(5, 3), dtype)
        d_gpu = gpuarray.to_gpu(d)
        a_gpu = gpuarray.to_gpu(a)
        r_gpu = linalg.dot_diag(d_gpu, a_gpu)
        assert np.allclose(np.dot(np.diag(d), a), r_gpu.get())
        a = a.astype(dtype, order="F", copy=True)
        d_gpu = gpuarray.to_gpu(d)
        a_gpu = gpuarray.to_gpu(a)
        # note: due to pycuda issue #66, this will fail when overwrite=False
        r_gpu = linalg.dot_diag(d_gpu, a_gpu, overwrite=True)
        assert np.allclose(np.dot(np.diag(d), a), r_gpu.get())

    def test_dot_diag_float32(self):
        self.__impl_test_dot_diag(np.float32)

    def test_dot_diag_float64(self):
        self.__impl_test_dot_diag(np.float64)

    def test_dot_diag_complex64(self):
        self.__impl_test_dot_diag(np.complex64)

    def test_dot_diag_complex128(self):
        self.__impl_test_dot_diag(np.complex128)

    def ___impl_test_dot_diag_t(self, dtype):
        d = np.asarray(np.random.rand(5), dtype)
        a = np.asarray(np.random.rand(3, 5), dtype)
        d_gpu = gpuarray.to_gpu(d)
        a_gpu = gpuarray.to_gpu(a)
        r_gpu = linalg.dot_diag(d_gpu, a_gpu, 't')
        assert np.allclose(np.dot(np.diag(d), a.T).T, r_gpu.get())
        a = a.astype(dtype, order="F", copy=True)
        d_gpu = gpuarray.to_gpu(d)
        a_gpu = gpuarray.to_gpu(a)
        # note: due to pycuda issue #66, this will fail when overwrite=False
        r_gpu = linalg.dot_diag(d_gpu, a_gpu, 't', overwrite=True)
        assert np.allclose(np.dot(np.diag(d), a.T).T, r_gpu.get())

    def test_dot_diag_t_float32(self):
        self.___impl_test_dot_diag_t(np.float32)

    def test_dot_diag_t_float64(self):
        self.___impl_test_dot_diag_t(np.float64)

    def test_dot_diag_t_complex64(self):
        self.___impl_test_dot_diag_t(np.complex64)

    def test_dot_diag_t_complex128(self):
        self.___impl_test_dot_diag_t(np.complex128)

    def test_transpose_float32(self):
        # M < N
        a = np.array([[1, 2, 3, 4, 5, 6],
                      [7, 8, 9, 10, 11, 12]],
                     np.float32)
        a_gpu = gpuarray.to_gpu(a)
        at_gpu = linalg.transpose(a_gpu)
        assert np.all(a.T == at_gpu.get())
        # M > N
        b = a.T.copy()
        b_gpu = gpuarray.to_gpu(b)
        bt_gpu = linalg.transpose(b_gpu)
        assert np.all(b.T == bt_gpu.get())

    def test_transpose_float64(self):
        # M < N
        a = np.array([[1, 2, 3, 4, 5, 6],
                      [7, 8, 9, 10, 11, 12]],
                     np.float64)
        a_gpu = gpuarray.to_gpu(a)
        at_gpu = linalg.transpose(a_gpu)
        assert np.all(a.T == at_gpu.get())
        # M > N
        b = a.T.copy()
        b_gpu = gpuarray.to_gpu(b)
        bt_gpu = linalg.transpose(b_gpu)
        assert np.all(b.T == bt_gpu.get())

    def test_transpose_complex64(self):
        # M < N
        a = np.array([[1j, 2j, 3j, 4j, 5j, 6j],
                      [7j, 8j, 9j, 10j, 11j, 12j]],
                     np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        at_gpu = linalg.transpose(a_gpu)
        assert np.all(a.T == at_gpu.get())
        # M > N
        b = a.T.copy()
        b_gpu = gpuarray.to_gpu(b)
        bt_gpu = linalg.transpose(b_gpu)
        assert np.all(b.T == bt_gpu.get())

    def test_transpose_complex128(self):
        # M < N
        a = np.array([[1j, 2j, 3j, 4j, 5j, 6j],
                      [7j, 8j, 9j, 10j, 11j, 12j]],
                     np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        at_gpu = linalg.transpose(a_gpu)
        assert np.all(a.T == at_gpu.get())
        # M > N
        b = a.T.copy()
        b_gpu = gpuarray.to_gpu(b)
        bt_gpu = linalg.transpose(b_gpu)
        assert np.all(b.T == bt_gpu.get())

    def test_hermitian_float32(self):
        # M < N
        a = np.array([[1, 2, 3, 4, 5, 6],
                      [7, 8, 9, 10, 11, 12]],
                     np.float32)
        a_gpu = gpuarray.to_gpu(a)
        at_gpu = linalg.hermitian(a_gpu)
        assert np.all(a.T == at_gpu.get())
        # M > N
        b = a.T.copy()
        b_gpu = gpuarray.to_gpu(b)
        bt_gpu = linalg.hermitian(b_gpu)
        assert np.all(b.T == bt_gpu.get())

    def test_hermitian_complex64(self):
        # M < N
        a = np.array([[1j, 2j, 3j, 4j, 5j, 6j],
                      [7j, 8j, 9j, 10j, 11j, 12j]],
                     np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        at_gpu = linalg.hermitian(a_gpu)
        assert np.all(np.conj(a.T) == at_gpu.get())
        # M > N
        b = a.T.copy()
        b_gpu = gpuarray.to_gpu(b)
        bt_gpu = linalg.hermitian(b_gpu)
        assert np.all(np.conj(b.T) == bt_gpu.get())

    def test_hermitian_float64(self):
        # M < N
        a = np.array([[1, 2, 3, 4, 5, 6],
                      [7, 8, 9, 10, 11, 12]],
                     np.float64)
        a_gpu = gpuarray.to_gpu(a)
        at_gpu = linalg.hermitian(a_gpu)
        assert np.all(a.T == at_gpu.get())
        # M > N
        b = a.T.copy()
        b_gpu = gpuarray.to_gpu(b)
        bt_gpu = linalg.hermitian(b_gpu)
        assert np.all(b.T == bt_gpu.get())

    def test_hermitian_complex128(self):
        # M < N
        a = np.array([[1j, 2j, 3j, 4j, 5j, 6j],
                      [7j, 8j, 9j, 10j, 11j, 12j]],
                     np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        at_gpu = linalg.hermitian(a_gpu)
        assert np.all(np.conj(a.T) == at_gpu.get())
        # M > N
        b = a.T.copy()
        b_gpu = gpuarray.to_gpu(b)
        bt_gpu = linalg.hermitian(b_gpu)
        assert np.all(np.conj(b.T) == bt_gpu.get())

    def test_conj_complex64(self):
        a = np.array([[1+1j, 2-2j, 3+3j, 4-4j],
                      [5+5j, 6-6j, 7+7j, 8-8j]], np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        r_gpu = linalg.conj(a_gpu)
        assert np.all(np.conj(a) == r_gpu.get())

    def test_conj_complex128(self):
        a = np.array([[1+1j, 2-2j, 3+3j, 4-4j],
                      [5+5j, 6-6j, 7+7j, 8-8j]], np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        r_gpu = linalg.conj(a_gpu)
        assert np.all(np.conj(a) == r_gpu.get())

    def test_diag_1d_float32(self):
        v = np.array([1, 2, 3, 4, 5, 6], np.float32)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert np.all(np.diag(v) == d_gpu.get())

    def test_diag_2d_wide_float32(self):
        v = np.array(np.random.rand(32, 64), np.float32)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert np.all(np.diag(v) == d_gpu.get())

    def test_diag_2d_tall_float32(self):
        v = np.array(np.random.rand(64, 32), np.float32)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert np.all(np.diag(v) == d_gpu.get())

    def test_diag_1d_float64(self):
        v = np.array([1, 2, 3, 4, 5, 6], np.float64)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert np.all(np.diag(v) == d_gpu.get())

    def test_diag_2d_wide_float64(self):
        v = np.array(np.random.rand(32, 64), np.float64)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert np.all(np.diag(v) == d_gpu.get())

    def test_diag_2d_tall_float64(self):
        v = np.array(np.random.rand(64, 32), np.float64)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert np.all(np.diag(v) == d_gpu.get())

    def test_diag_1d_complex64(self):
        v = np.array([1j, 2j, 3j, 4j, 5j, 6j], np.complex64)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert np.all(np.diag(v) == d_gpu.get())

    def test_diag_2d_wide_complex64(self):
        v = np.array(np.random.rand(32, 64)*1j, np.complex64)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert np.all(np.diag(v) == d_gpu.get())

    def test_diag_2d_tall_complex64(self):
        v = np.array(np.random.rand(64, 32)*1j, np.complex64)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert np.all(np.diag(v) == d_gpu.get())

    def test_diag_1d_complex128(self):
        v = np.array([1j, 2j, 3j, 4j, 5j, 6j], np.complex128)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert np.all(np.diag(v) == d_gpu.get())

    def test_diag_2d_wide_complex128(self):
        v = np.array(np.random.rand(32, 64)*1j, np.complex128)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert np.all(np.diag(v) == d_gpu.get())

    def test_diag_2d_tall_complex128(self):
        v = np.array(np.random.rand(64, 32)*1j, np.complex128)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert np.all(np.diag(v) == d_gpu.get())

    def test_eye_float32(self):
        N = 10
        e_gpu = linalg.eye(N, dtype=np.float32)
        assert np.all(np.eye(N, dtype=np.float32) == e_gpu.get())

    def test_eye_float64(self):
        N = 10
        e_gpu = linalg.eye(N, dtype=np.float64)
        assert np.all(np.eye(N, dtype=np.float64) == e_gpu.get())

    def test_eye_complex64(self):
        N = 10
        e_gpu = linalg.eye(N, dtype=np.complex64)
        assert np.all(np.eye(N, dtype=np.complex64) == e_gpu.get())

    def test_eye_complex128(self):
        N = 10
        e_gpu = linalg.eye(N, dtype=np.complex128)
        assert np.all(np.eye(N, dtype=np.complex128) == e_gpu.get())

    def test_pinv_float32(self):
        a = np.asarray(np.random.rand(8, 4), np.float32)
        a_gpu = gpuarray.to_gpu(a)
        a_inv_gpu = linalg.pinv(a_gpu)
        assert np.allclose(np.linalg.pinv(a), a_inv_gpu.get(),
                           atol=atol_float32)

    def test_pinv_float64(self):
        a = np.asarray(np.random.rand(8, 4), np.float64)
        a_gpu = gpuarray.to_gpu(a)
        a_inv_gpu = linalg.pinv(a_gpu)
        assert np.allclose(np.linalg.pinv(a), a_inv_gpu.get(),
                           atol=atol_float64)

    def test_pinv_complex64(self):
        a = np.asarray(np.random.rand(8, 4) + \
                       1j*np.random.rand(8, 4), np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        a_inv_gpu = linalg.pinv(a_gpu)
        assert np.allclose(np.linalg.pinv(a), a_inv_gpu.get(),
                           atol=atol_float32)

    def test_pinv_complex128(self):
        a = np.asarray(np.random.rand(8, 4) + \
                       1j*np.random.rand(8, 4), np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        a_inv_gpu = linalg.pinv(a_gpu)
        assert np.allclose(np.linalg.pinv(a), a_inv_gpu.get(),
                           atol=atol_float64)

    def test_tril_float32(self):
        a = np.asarray(np.random.rand(4, 4), np.float32)
        a_gpu = gpuarray.to_gpu(a)
        l_gpu = linalg.tril(a_gpu)
        assert np.allclose(np.tril(a), l_gpu.get())

    def test_tril_float64(self):
        a = np.asarray(np.random.rand(4, 4), np.float64)
        a_gpu = gpuarray.to_gpu(a)
        l_gpu = linalg.tril(a_gpu)
        assert np.allclose(np.tril(a), l_gpu.get())

    def test_tril_complex64(self):
        a = np.asarray(np.random.rand(4, 4), np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        l_gpu = linalg.tril(a_gpu)
        assert np.allclose(np.tril(a), l_gpu.get())

    def test_tril_complex128(self):
        a = np.asarray(np.random.rand(4, 4), np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        l_gpu = linalg.tril(a_gpu)
        assert np.allclose(np.tril(a), l_gpu.get())

    def test_triu_float32(self):
        a = np.asarray(np.random.rand(4, 4), np.float32)
        a_gpu = gpuarray.to_gpu(a)
        l_gpu = linalg.triu(a_gpu)
        assert np.allclose(np.triu(a), l_gpu.get())

    def test_triu_float64(self):
        a = np.asarray(np.random.rand(4, 4), np.float64)
        a_gpu = gpuarray.to_gpu(a)
        l_gpu = linalg.triu(a_gpu)
        assert np.allclose(np.triu(a), l_gpu.get())

    def test_triu_complex64(self):
        a = np.asarray(np.random.rand(4, 4), np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        l_gpu = linalg.triu(a_gpu)
        assert np.allclose(np.triu(a), l_gpu.get())

    def test_triu_complex128(self):
        a = np.asarray(np.random.rand(4, 4), np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        l_gpu = linalg.triu(a_gpu)
        assert np.allclose(np.triu(a), l_gpu.get())

    def _impl_test_multiply(self, N, dtype):
        mk_matrix = lambda N, dtype: np.asarray(np.random.rand(N, N), dtype)
        x = mk_matrix(N, dtype)
        y = mk_matrix(N, dtype)
        if np.iscomplexobj(x):
            x += 1j*mk_matrix(N, dtype)
            y += 1j*mk_matrix(N, dtype)
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.to_gpu(y)
        z_gpu = linalg.multiply(x_gpu, y_gpu)
        assert np.allclose(x*y, z_gpu.get())

    def test_multiply_float32(self):
        self._impl_test_multiply(4, np.float32)

    def test_multiply_float64(self):
        self._impl_test_multiply(4, np.float64)

    def test_multiply_complex64(self):
        self._impl_test_multiply(4, np.complex64)

    def test_multiply_complex128(self):
        self._impl_test_multiply(4, np.complex128)

    def _impl_test_cho_factor(self, N, dtype, lib='cula'):
        from scipy.linalg import cho_factor as cpu_cho_factor
        x = np.asarray(np.random.rand(N, N), dtype)
        if np.iscomplexobj(x):
            x += 1j*np.asarray(np.random.rand(N, N), dtype)
            x = np.dot(np.conj(x.T), x)
        else:
            x = np.dot(x.T, x)
        x_gpu = gpuarray.to_gpu(x)
        linalg.cho_factor(x_gpu, 'L', lib)
        c = np.triu(cpu_cho_factor(x)[0])
        assert np.allclose(c, np.triu(x_gpu.get()))

    def test_cho_factor_cula_float32(self):
        self._impl_test_cho_factor(4, np.float32, 'cula')

    def test_cho_factor_cula_float64(self):
        self._impl_test_cho_factor(4, np.float64, 'cula')

    def test_cho_factor_cula_complex64(self):
        self._impl_test_cho_factor(4, np.complex64, 'cula')

    def test_cho_factor_cula_complex128(self):
        self._impl_test_cho_factor(4, np.complex128, 'cula')

    def test_cho_factor_cusolver_float32(self):
        self._impl_test_cho_factor(4, np.float32, 'cusolver')

    def test_cho_factor_cusolver_float64(self):
        self._impl_test_cho_factor(4, np.float64, 'cusolver')

    def test_cho_factor_cusolver_complex64(self):
        self._impl_test_cho_factor(4, np.complex64, 'cusolver')

    def test_cho_factor_cusolver_complex128(self):
        self._impl_test_cho_factor(4, np.complex128, 'cusolver')

    def test_cho_solve_float32(self):
        x = np.asarray(np.random.rand(4, 4), np.float32)
        x = np.dot(x.T, x)
        y = np.asarray(np.random.rand(4), np.float32)
        c = np.linalg.inv(x).dot(y)

        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.to_gpu(y)
        linalg.cho_solve(x_gpu, y_gpu)
        assert np.allclose(c, y_gpu.get(), atol=1e-4)

        x = np.asarray(np.random.rand(4, 4), np.float32)
        x = np.dot(x.T, x).astype(np.float32, order="F", copy=True)
        y = np.asarray(np.random.rand(4, 4), np.float32, order="F")
        c = np.linalg.inv(x).dot(y)

        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.to_gpu(y)
        linalg.cho_solve(x_gpu, y_gpu)
        assert np.allclose(c, y_gpu.get(), atol=1e-4)

    def _impl_test_inv(self, dtype):
        from scipy.linalg import inv as cpu_inv
        x = np.asarray(np.random.rand(4, 4), dtype)
        x = np.dot(x.T, x)
        x_gpu = gpuarray.to_gpu(x)
        xinv = cpu_inv(x)
        xinv_gpu = linalg.inv(x_gpu)
        assert np.allclose(xinv, xinv_gpu.get(), atol=1e-5)
        assert xinv_gpu is not x_gpu
        xinv_gpu = linalg.inv(x_gpu, overwrite=True)
        assert np.allclose(xinv, xinv_gpu.get(), atol=1e-5)
        assert xinv_gpu is x_gpu

    def test_inv_exceptions(self):
        x = np.asarray([[1, 2], [2, 4]], np.float32)
        x_gpu = gpuarray.to_gpu(x)
        assert_raises(linalg.LinAlgError, linalg.inv, x_gpu)

    def test_inv_float32(self):
        self._impl_test_inv(np.float32)

    def test_inv_float64(self):
        self._impl_test_inv(np.float64)

    def test_inv_complex64(self):
        self._impl_test_inv(np.complex64)

    def test_inv_complex128(self):
        self._impl_test_inv(np.complex128)


    def _impl_test_add_diag(self, dtype):
        x = np.asarray(np.random.rand(4, 4), dtype)
        d = np.asarray(np.random.rand(1, 4), dtype).reshape(-1)
        x_gpu = gpuarray.to_gpu(x)
        d_gpu = gpuarray.to_gpu(d)
        res_cpu = x + np.diag(d)
        res_gpu = linalg.add_diag(d_gpu, x_gpu, overwrite=False)
        assert np.allclose(res_cpu, res_gpu.get(), atol=1e-5)
        assert res_gpu is not x_gpu
        res_gpu = linalg.add_diag(d_gpu, x_gpu, overwrite=True)
        assert np.allclose(res_cpu, res_gpu.get(), atol=1e-5)
        assert res_gpu is x_gpu


    def test_add_diag_float32(self):
        self._impl_test_add_diag(np.float32)

    def test_add_diag_float64(self):
        self._impl_test_add_diag(np.float64)

    def test_add_diag_complex64(self):
        self._impl_test_add_diag(np.complex64)

    def test_add_diag_complex128(self):
        self._impl_test_add_diag(np.complex128)

    def test_eye_large_float32(self):
        N = 128
        e_gpu = linalg.eye(N, dtype=np.float32)
        assert np.all(np.eye(N, dtype=np.float32) == e_gpu.get())

    def _impl_test_trace(self, dtype):
        # square matrix
        x = 10*np.asarray(np.random.rand(4, 4), dtype)
        x_gpu = gpuarray.to_gpu(x)
        assert np.allclose(linalg.trace(x_gpu), np.trace(x))
        # tall matrix
        x = np.asarray(np.random.rand(5, 2), dtype)
        x_gpu = gpuarray.to_gpu(x)
        assert np.allclose(linalg.trace(x_gpu), np.trace(x))
        # fat matrix
        x = np.asarray(np.random.rand(2, 5), dtype)
        x_gpu = gpuarray.to_gpu(x)
        assert np.allclose(linalg.trace(x_gpu), np.trace(x))

    def test_trace_float32(self):
        self._impl_test_trace(np.float32)

    def test_trace_float64(self):
        self._impl_test_trace(np.float64)

    def test_trace_complex64(self):
        self._impl_test_trace(np.complex64)

    def test_trace_complex128(self):
        self._impl_test_trace(np.complex128)

    def _impl_add_dot_matrix_tests(self, dtype, transa, transb):
        a = np.asarray(np.random.rand(4, 2), dtype)
        if transa == 'n':
            b = np.asarray(np.random.rand(2, 2), dtype)
        else:
            b = np.asarray(np.random.rand(4, 4), dtype)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        aa = a if transa == 'n' else a.T
        bb = b if transb == 'n' else b.T
        c = np.asarray(np.random.rand(aa.shape[0], bb.shape[1]), dtype)
        c_gpu = gpuarray.to_gpu(c)
        c_gpu = linalg.add_dot(a_gpu, b_gpu, c_gpu, transa, transb)
        assert np.allclose(c + np.dot(aa, bb), c_gpu.get())
        a = a.astype(dtype, order="F", copy=True)
        b = b.astype(dtype, order="F", copy=True)
        c = c.astype(dtype, order="F", copy=True)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)
        c_gpu = linalg.add_dot(a_gpu, b_gpu, c_gpu, transa, transb)
        assert np.allclose(c+np.dot(aa, bb), c_gpu.get())

    def test_add_dot_matrix_float32(self):
        self._impl_add_dot_matrix_tests(np.float32, 'n', 'n')
        self._impl_add_dot_matrix_tests(np.float32, 'n', 't')
        self._impl_add_dot_matrix_tests(np.float32, 't', 'n')
        self._impl_add_dot_matrix_tests(np.float32, 't', 't')

    def test_add_dot_matrix_float64(self):
        self._impl_add_dot_matrix_tests(np.float64, 'n', 'n')
        self._impl_add_dot_matrix_tests(np.float64, 'n', 't')
        self._impl_add_dot_matrix_tests(np.float64, 't', 'n')
        self._impl_add_dot_matrix_tests(np.float64, 't', 't')

    def test_add_dot_matrix_complex64(self):
        self._impl_add_dot_matrix_tests(np.complex64, 'n', 'n')
        self._impl_add_dot_matrix_tests(np.complex64, 'n', 't')
        self._impl_add_dot_matrix_tests(np.complex64, 't', 'n')
        self._impl_add_dot_matrix_tests(np.complex64, 't', 't')

    def test_add_dot_matrix_complex128(self):
        self._impl_add_dot_matrix_tests(np.complex128, 'n', 'n')
        self._impl_add_dot_matrix_tests(np.complex128, 'n', 't')
        self._impl_add_dot_matrix_tests(np.complex128, 't', 'n')
        self._impl_add_dot_matrix_tests(np.complex128, 't', 't')

    def _impl_test_dot_strided(self, dtype):
        # n/n
        a = np.asarray(np.random.rand(4, 10), dtype)
        b = np.asarray(np.random.rand(2, 20), dtype)
        c = np.zeros((4, 30), dtype)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)
        linalg.add_dot(a_gpu[:, 4:6], b_gpu[:, 2:8], c_gpu[:, 1:7], 'n', 'n')
        res = c_gpu.get()
        assert np.allclose(np.dot(a[:, 4:6], b[:, 2:8]), res[:, 1:7])

        # t/n
        a = np.asarray(np.random.rand(4, 10), dtype)
        b = np.asarray(np.random.rand(4, 20), dtype)
        c = np.zeros((2, 30), dtype)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)
        linalg.add_dot(a_gpu[:, 4:6], b_gpu[:, 2:8], c_gpu[:, 1:7], 't', 'n')
        res = c_gpu.get()
        assert np.allclose(np.dot(a[:, 4:6].T, b[:, 2:8]), res[:, 1:7])

        # n/t
        a = np.asarray(np.random.rand(4, 10), dtype)
        b = np.asarray(np.random.rand(6, 20), dtype)
        c = np.zeros((4, 30), dtype)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)
        linalg.add_dot(a_gpu[:, 4:10], b_gpu[:, 2:8], c_gpu[:, 1:7], 'n', 't')
        res = c_gpu.get()
        assert np.allclose(np.dot(a[:, 4:10], b[:, 2:8].T), res[:, 1:7])

        # t/t
        a = np.asarray(np.random.rand(6, 10), dtype)
        b = np.asarray(np.random.rand(8, 20), dtype)
        c = np.zeros((2, 30), dtype)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)
        linalg.add_dot(a_gpu[:, 4:6], b_gpu[:, 2:8], c_gpu[:, 1:9], 't', 't')
        res = c_gpu.get()
        assert np.allclose(np.dot(a[:, 4:6].T, b[:, 2:8].T), res[:, 1:9])

    def test_dot_strided_float32(self):
        self._impl_test_dot_strided(np.float32)

    def test_dot_strided_float64(self):
        self._impl_test_dot_strided(np.float64)

    def test_dot_strided_complex64(self):
        self._impl_test_dot_strided(np.complex64)

    def test_dot_strided_complex128(self):
        self._impl_test_dot_strided(np.complex128)

    def _impl_test_det(self, dtype, lib):
        # random matrix
        x = 10*np.asarray(np.random.rand(4, 4), dtype)
        x_gpu = gpuarray.to_gpu(x)
        assert np.allclose(linalg.det(x_gpu, lib=lib), np.linalg.det(x))

        # known matrix (from http://en.wikipedia.org/wiki/Determinant )
        x = np.asarray([[-2.0, 2, -3.0], [-1, 1, 3], [2, 0, -1]], dtype)
        x_gpu = gpuarray.to_gpu(x)
        assert np.allclose(linalg.det(x_gpu, lib=lib), 18.0)

    def test_det_cula_float32(self):
        self._impl_test_det(np.float32, 'cula')

    def test_det_cula_float64(self):
        self._impl_test_det(np.float64, 'cula')

    def test_det_cula_complex64(self):
        self._impl_test_det(np.complex64, 'cula')

    def test_det_cula_complex128(self):
        self._impl_test_det(np.complex128, 'cula')

    def test_det_cusolver_float32(self):
        self._impl_test_det(np.float32, 'cusolver')

    def test_det_cusolver_float64(self):
        self._impl_test_det(np.float64, 'cusolver')

    def test_det_cusolver_complex64(self):
        self._impl_test_det(np.complex64, 'cusolver')

    def test_det_cusolver_complex128(self):
        self._impl_test_det(np.complex128, 'cusolver')
        
    
    def test_qr_reduced_float32(self):
        a = np.asarray(np.random.randn(5, 3), np.float32, order='F')
        a_gpu = gpuarray.to_gpu(a)
        q_gpu, r_gpu = linalg.qr(a_gpu, 'reduced')
        assert np.allclose(a, np.dot(q_gpu.get(), r_gpu.get()), atol=1e-4)

    def test_qr_reduced_float64(self):
        a = np.asarray(np.random.randn(5, 3), np.float32, order='F')
        a_gpu = gpuarray.to_gpu(a)
        q_gpu, r_gpu = linalg.qr(a_gpu, 'reduced')
        assert np.allclose(a, np.dot(q_gpu.get(), r_gpu.get()), atol=atol_float64)

    def test_qr_reduced_complex64(self):
        a = np.asarray(np.random.randn(9, 6) + 1j*np.random.randn(9, 6), np.complex64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        q_gpu, r_gpu = linalg.qr(a_gpu, 'reduced')
        assert np.allclose(a, np.dot(q_gpu.get(), r_gpu.get()), atol=1e-4)

    def test_qr_reduced_complex128(self):
        a = np.asarray(np.random.randn(9, 6) + 1j*np.random.randn(9, 6), np.complex64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        q_gpu, r_gpu = linalg.qr(a_gpu, 'reduced')
        assert np.allclose(a, np.dot(q_gpu.get(), r_gpu.get()), atol=atol_float64)     
        
    def test_eig_float32(self):
        a = np.asarray(np.random.rand(9, 9), np.float32, order='F')
        a_gpu = gpuarray.to_gpu(a)
        w_gpu = linalg.eig(a_gpu, 'N', 'N')
        assert np.allclose(np.trace(a), sum(w_gpu.get()), atol=1e-4)

    def test_eig_float64(self):
        a = np.asarray(np.random.rand(9, 9), np.float64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        w_gpu = linalg.eig(a_gpu, 'N', 'N')
        assert np.allclose(np.trace(a), sum(w_gpu.get()), atol=atol_float64)

    def test_eig_complex64(self):
        a = np.asarray(np.random.rand(9, 9) + 1j*np.random.rand(9, 9), np.complex64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        w_gpu = linalg.eig(a_gpu, 'N', 'N')
        assert np.allclose(np.trace(a), sum(w_gpu.get()), atol=1e-4)

    def test_eig_complex128(self):
        a = np.array(np.random.rand(9, 9) + 1j*np.random.rand(9,9), np.complex128, order='F')
        a_gpu = gpuarray.to_gpu(a)
        w_gpu = linalg.eig(a_gpu, 'N', 'N')
        assert np.allclose(np.trace(a), sum(w_gpu.get()), atol=atol_float64) 

    def test_vander_float32(self):
        a = np.array(np.random.uniform(1,2,5), np.float32, order='F')
        a_gpu = gpuarray.to_gpu(a)
        vander_gpu = linalg.vander(a_gpu)
        assert np.allclose(np.fliplr(np.vander(a)), vander_gpu.get(), atol=atol_float32)

    def test_vander_float64(self):
        a = np.array(np.random.uniform(1,2,5), np.float64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        vander_gpu = linalg.vander(a_gpu)
        assert np.allclose(np.fliplr(np.vander(a)), vander_gpu.get(), atol=atol_float64)

    def test_vander_complex64(self):
        a = np.array(np.random.uniform(1,2,5) + 1j*np.random.uniform(1,2,5), np.complex64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        vander_gpu = linalg.vander(a_gpu)
        assert np.allclose(np.fliplr(np.vander(a)), vander_gpu.get(), atol=atol_float32)

    def test_vander_complex128(self):
        a = np.array(np.random.uniform(1,2,5) + 1j*np.random.uniform(1,2,5), np.complex128, order='F')
        a_gpu = gpuarray.to_gpu(a)
        vander_gpu = linalg.vander(a_gpu)
        assert np.allclose(np.fliplr(np.vander(a)), vander_gpu.get(), atol=atol_float64)

    def test_dmd_float32(self):
        m, n = 6, 4
        a = np.array(np.fliplr(np.vander(np.random.rand(m)+1, n)), np.float32, order='F')
        a_gpu = gpuarray.to_gpu(a)
        f_gpu, b_gpu, v_gpu, omega = linalg.dmd(a_gpu, modes='standard', return_amplitudes=True, return_vandermonde=True)
        assert np.allclose(a[:,:(n-1)], np.dot(f_gpu.get(), np.dot(np.diag(b_gpu.get()), v_gpu.get()) ), 1e-4)

    def test_dmd_float64(self):
        m, n = 9, 7
        a = np.array(np.fliplr(np.vander(np.random.rand(m)+1, n)), np.float64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        f_gpu, b_gpu, v_gpu, omega = linalg.dmd(a_gpu, modes='standard', return_amplitudes=True, return_vandermonde=True)
        assert np.allclose(a[:,:(n-1)], np.dot(f_gpu.get(), np.dot(np.diag(b_gpu.get()), v_gpu.get()) ), atol_float64)
    
    def test_dmd_complex64(self):
        m, n = 9, 7
        a = np.array(np.fliplr(np.vander(np.random.rand(m)+1, n)) + 1j*np.fliplr(np.vander(np.random.rand(m), n)), 
                     np.complex64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        f_gpu, b_gpu, v_gpu, omega = linalg.dmd(a_gpu, modes='standard', return_amplitudes=True, return_vandermonde=True)
        assert np.allclose(a[:,:(n-1)], np.dot(f_gpu.get(), np.dot(np.diag(b_gpu.get()), v_gpu.get()) ), 1e-4)
        
    def test_dmd_complex128(self):
        m, n = 9, 7
        a = np.array(np.fliplr(np.vander(np.random.rand(m)+1, n)) + 1j*np.fliplr(np.vander(np.random.rand(m), n)), 
                     np.complex128, order='F')
        a_gpu = gpuarray.to_gpu(a)
        f_gpu, b_gpu, v_gpu, omega = linalg.dmd(a_gpu, modes='standard', return_amplitudes=True, return_vandermonde=True)
        assert np.allclose(a[:,:(n-1)], np.dot(f_gpu.get(), np.dot(np.diag(b_gpu.get()), v_gpu.get()) ), atol_float64)
        

def suite():
    s = TestSuite()
    s.addTest(test_linalg('test_svd_ss_cula_float32'))
    s.addTest(test_linalg('test_svd_ss_cula_complex64'))
    s.addTest(test_linalg('test_svd_so_cula_float32'))
    s.addTest(test_linalg('test_svd_so_cula_complex64'))
    s.addTest(test_linalg('test_svd_aa_cusolver_float32'))
    s.addTest(test_linalg('test_svd_aa_cusolver_complex64'))
    s.addTest(test_linalg('test_dot_matrix_vector_float32'))
    s.addTest(test_linalg('test_dot_matrix_vector_complex64'))
    s.addTest(test_linalg('test_dot_matrix_float32'))
    s.addTest(test_linalg('test_dot_matrix_complex64'))
    s.addTest(test_linalg('test_dot_matrix_h_complex64'))
    s.addTest(test_linalg('test_dot_vector_float32'))
    s.addTest(test_linalg('test_dot_vector_complex64'))
    s.addTest(test_linalg('test_mdot_matrix_float32'))
    s.addTest(test_linalg('test_mdot_matrix_complex64'))
    s.addTest(test_linalg('test_dot_diag_float32'))
    s.addTest(test_linalg('test_dot_diag_complex64'))
    s.addTest(test_linalg('test_dot_diag_t_float32'))
    s.addTest(test_linalg('test_dot_diag_t_complex64'))
    s.addTest(test_linalg('test_transpose_float32'))
    s.addTest(test_linalg('test_transpose_complex64'))
    s.addTest(test_linalg('test_hermitian_float32'))
    s.addTest(test_linalg('test_hermitian_complex64'))
    s.addTest(test_linalg('test_conj_complex64'))
    s.addTest(test_linalg('test_diag_1d_float32'))
    s.addTest(test_linalg('test_diag_2d_wide_float32'))
    s.addTest(test_linalg('test_diag_2d_tall_float32'))
    s.addTest(test_linalg('test_diag_1d_complex64'))
    s.addTest(test_linalg('test_diag_2d_wide_complex64'))
    s.addTest(test_linalg('test_diag_2d_tall_complex64'))
    s.addTest(test_linalg('test_eye_float32'))
    s.addTest(test_linalg('test_eye_complex64'))
    s.addTest(test_linalg('test_pinv_float32'))
    s.addTest(test_linalg('test_pinv_complex64'))
    s.addTest(test_linalg('test_tril_float32'))
    s.addTest(test_linalg('test_tril_complex64'))
    s.addTest(test_linalg('test_triu_float32'))
    s.addTest(test_linalg('test_triu_complex64'))
    s.addTest(test_linalg('test_multiply_float32'))
    s.addTest(test_linalg('test_multiply_complex64'))
    s.addTest(test_linalg('test_cho_factor_cula_float32'))
    s.addTest(test_linalg('test_cho_factor_cula_complex64'))
    s.addTest(test_linalg('test_cho_factor_cusolver_float32'))
    s.addTest(test_linalg('test_cho_factor_cusolver_complex64'))
    s.addTest(test_linalg('test_cho_solve_float32'))
    s.addTest(test_linalg('test_inv_float32'))
    s.addTest(test_linalg('test_inv_complex64'))
    s.addTest(test_linalg('test_add_diag_float32'))
    s.addTest(test_linalg('test_add_diag_complex64'))
    s.addTest(test_linalg('test_inv_exceptions'))
    s.addTest(test_linalg('test_eye_large_float32'))
    s.addTest(test_linalg('test_trace_float32'))
    s.addTest(test_linalg('test_trace_complex64'))
    s.addTest(test_linalg('test_add_dot_matrix_float32'))
    s.addTest(test_linalg('test_add_dot_matrix_complex64'))
    s.addTest(test_linalg('test_dot_strided_float32'))
    s.addTest(test_linalg('test_dot_strided_complex64'))
    s.addTest(test_linalg('test_det_cula_float32'))
    s.addTest(test_linalg('test_det_cula_complex64'))
    s.addTest(test_linalg('test_det_cusolver_float32'))
    s.addTest(test_linalg('test_det_cusolver_complex64'))
    s.addTest(test_linalg('test_qr_reduced_float32'))
    s.addTest(test_linalg('test_qr_reduced_float64'))
    s.addTest(test_linalg('test_qr_reduced_complex64'))
    s.addTest(test_linalg('test_qr_reduced_complex128'))
    s.addTest(test_linalg('test_eig_float32'))
    s.addTest(test_linalg('test_eig_float64'))
    s.addTest(test_linalg('test_eig_complex64'))
    s.addTest(test_linalg('test_eig_complex128'))
    s.addTest(test_linalg('test_vander_float32'))
    s.addTest(test_linalg('test_vander_float64'))
    s.addTest(test_linalg('test_vander_complex64'))
    s.addTest(test_linalg('test_vander_complex128'))
    s.addTest(test_linalg('test_dmd_float32'))
    s.addTest(test_linalg('test_dmd_float64'))
    s.addTest(test_linalg('test_dmd_complex64'))
    s.addTest(test_linalg('test_dmd_complex128'))
     
    
    if misc.get_compute_capability(pycuda.autoinit.device) >= 1.3:
        s.addTest(test_linalg('test_svd_ss_cula_float64'))
        s.addTest(test_linalg('test_svd_ss_cula_complex128'))
        s.addTest(test_linalg('test_svd_so_cula_float64'))
        s.addTest(test_linalg('test_svd_so_cula_complex128'))
        s.addTest(test_linalg('test_svd_aa_cusolver_float64'))
        s.addTest(test_linalg('test_svd_aa_cusolver_complex128'))
        s.addTest(test_linalg('test_dot_matrix_vector_float64'))
        s.addTest(test_linalg('test_dot_matrix_vector_complex128'))
        s.addTest(test_linalg('test_dot_matrix_float64'))
        s.addTest(test_linalg('test_dot_matrix_complex128'))
        s.addTest(test_linalg('test_dot_matrix_h_complex128'))
        s.addTest(test_linalg('test_dot_vector_float64'))
        s.addTest(test_linalg('test_dot_vector_complex128'))
        s.addTest(test_linalg('test_mdot_matrix_float64'))
        s.addTest(test_linalg('test_mdot_matrix_complex128'))
        s.addTest(test_linalg('test_dot_diag_t_float64'))
        s.addTest(test_linalg('test_dot_diag_t_complex128'))
        s.addTest(test_linalg('test_transpose_float64'))
        s.addTest(test_linalg('test_transpose_complex128'))
        s.addTest(test_linalg('test_hermitian_float64'))
        s.addTest(test_linalg('test_hermitian_complex64'))
        s.addTest(test_linalg('test_conj_complex128'))
        s.addTest(test_linalg('test_diag_1d_float64'))
        s.addTest(test_linalg('test_diag_2d_wide_float64'))
        s.addTest(test_linalg('test_diag_2d_tall_float64'))
        s.addTest(test_linalg('test_diag_1d_complex128'))
        s.addTest(test_linalg('test_diag_2d_wide_complex128'))
        s.addTest(test_linalg('test_diag_2d_tall_complex128'))
        s.addTest(test_linalg('test_eye_float64'))
        s.addTest(test_linalg('test_eye_complex128'))
        s.addTest(test_linalg('test_pinv_float64'))
        s.addTest(test_linalg('test_pinv_complex128'))
        s.addTest(test_linalg('test_tril_float64'))
        s.addTest(test_linalg('test_tril_complex128'))
        s.addTest(test_linalg('test_triu_float32'))
        s.addTest(test_linalg('test_triu_complex64'))
        s.addTest(test_linalg('test_multiply_float64'))
        s.addTest(test_linalg('test_multiply_complex128'))
        s.addTest(test_linalg('test_cho_factor_cula_float64'))
        s.addTest(test_linalg('test_cho_factor_cula_complex128'))
        s.addTest(test_linalg('test_cho_factor_cusolver_float64'))
        s.addTest(test_linalg('test_cho_factor_cusolver_complex128'))
        s.addTest(test_linalg('test_inv_float64'))
        s.addTest(test_linalg('test_inv_complex128'))
        s.addTest(test_linalg('test_add_diag_float64'))
        s.addTest(test_linalg('test_add_diag_complex128'))
        s.addTest(test_linalg('test_trace_float64'))
        s.addTest(test_linalg('test_trace_complex128'))
        s.addTest(test_linalg('test_add_dot_matrix_float64'))
        s.addTest(test_linalg('test_add_dot_matrix_complex128'))
        s.addTest(test_linalg('test_dot_strided_float64'))
        s.addTest(test_linalg('test_dot_strided_complex128'))
        s.addTest(test_linalg('test_det_cula_float64'))
        s.addTest(test_linalg('test_det_cula_complex128'))
        s.addTest(test_linalg('test_det_cusolver_float64'))
        s.addTest(test_linalg('test_det_cusolver_complex128'))
        s.addTest(test_linalg('test_qr_reduced_float32'))
        s.addTest(test_linalg('test_qr_reduced_float64'))
        s.addTest(test_linalg('test_qr_reduced_complex64'))
        s.addTest(test_linalg('test_qr_reduced_complex128'))
        s.addTest(test_linalg('test_eig_float32'))
        s.addTest(test_linalg('test_eig_float64'))
        s.addTest(test_linalg('test_eig_complex64'))
        s.addTest(test_linalg('test_eig_complex128'))
        s.addTest(test_linalg('test_vander_float32'))
        s.addTest(test_linalg('test_vander_float64'))
        s.addTest(test_linalg('test_vander_complex64'))
        s.addTest(test_linalg('test_vander_complex128'))
        s.addTest(test_linalg('test_dmd_float32'))
        s.addTest(test_linalg('test_dmd_float64'))
        s.addTest(test_linalg('test_dmd_complex64'))
        s.addTest(test_linalg('test_dmd_complex128'))
        
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
