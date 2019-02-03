#!/usr/bin/env python

"""
Unit tests for skcuda.linalg
"""

from unittest import main, makeSuite, skipUnless, TestCase, TestSuite

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.tools import clear_context_caches, make_default_context
import numpy as np

from numpy.testing import assert_equal, assert_allclose, assert_raises

import skcuda.linalg as linalg
import skcuda.misc as misc

drv.init()

dtype_to_atol = {np.float32: 1e-6,
                 np.complex64: 1e-6,
                 np.float64: 1e-8,
                 np.complex128: 1e-8}
dtype_to_rtol = {np.float32: 5e-5,
                 np.complex64: 5e-5,
                 np.float64: 1e-5,
                 np.complex128: 1e-5}

class test_linalg(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = make_default_context()
        linalg.init()
 
    @classmethod
    def tearDownClass(cls):
        linalg.shutdown()
        cls.ctx.pop()
        clear_context_caches()

    def setUp(self):
        np.random.seed(0)

        ### required for PCA tests ##### 
        self.M = 1000
        self.N = 100
        self.test_pca = linalg.PCA()
        self.max_sdot = np.float32(0.005)
        self.max_ddot = np.float64(0.000001)
        self.K = 2
        self.test_pca2 = linalg.PCA(n_components=self.K)
        Xd_ = np.random.rand(self.M, self.N)
        Xf_ = np.random.rand(self.M, self.N).astype(np.float32)

        self.Xd = gpuarray.GPUArray((self.M, self.N), np.float64, order="F")
        self.Xd.set(Xd_)
        self.Xf = gpuarray.GPUArray((self.M, self.N), np.float32, order="F")
        self.Xf.set(Xf_)
        
    def test_pca_ortho_type_and_shape_float64_all_comp(self):
        # test that the shape is what we think it should be
        Td_all = self.test_pca.fit_transform(self.Xd)
        self.assertIsNotNone(Td_all)
        self.assertEqual(Td_all.dtype, np.float64)
        self.assertEqual(Td_all.shape, (self.M, self.N))
        for i in range(self.N-1):
            self.assertTrue(linalg.dot(Td_all[:,i], Td_all[:,i+1]) < self.max_ddot)

    def test_pca_ortho_type_and_shape_float32_all_comp(self):
        # test that the shape is what we think it should be
        Tf_all = self.test_pca.fit_transform(self.Xf)
        self.assertIsNotNone(Tf_all)
        self.assertEqual(Tf_all.dtype, np.float32)
        self.assertEqual(Tf_all.shape, (self.M, self.N))
        self.Tf_all = Tf_all
        for i in range(self.N-1):
            self.assertTrue(linalg.dot(Tf_all[:,i], Tf_all[:,i+1]) < self.max_sdot)

    def test_pca_ortho_type_and_shape_float64(self):
        # test that the shape is what we think it should be
        Td_2 = self.test_pca2.fit_transform(self.Xd)
        self.assertIsNotNone(Td_2)
        self.assertEqual(Td_2.dtype, np.float64)
        self.assertEqual(Td_2.shape, (self.M, self.K))
        self.assertTrue(linalg.dot(Td_2[:,0], Td_2[:,1]) < self.max_ddot)

    def test_pca_ortho_type_and_shape_float32(self):
        # test that the shape is what we think it should be	
        Tf_2 = self.test_pca2.fit_transform(self.Xf)
        self.assertIsNotNone(Tf_2)
        self.assertEqual(Tf_2.dtype, np.float32)
        self.assertEqual(Tf_2.shape, (self.M, self.K))
        self.assertTrue(linalg.dot(Tf_2[:,0], Tf_2[:,1]) < self.max_sdot) 

    def test_pca_f_contiguous_check(self):
        try:
            self.test_pca2.fit_transform(self.Xf.transpose())
            fail(msg="PCA F-contiguous array check failed") # should not reach this line. The prev line should fail and go to the except block
        except ValueError:
            pass

    def test_pca_arr_2d_check(self):
        try:
            X_trash = np.random.rand(self.M, self.M, 3).astype(np.float32)
            X_gpu_trash = gpuarray.GPUArray(X_trash.shape, np.float32, order="F")	
            X_gpu_trash.set(X_trash)
            self.test_pca2.fit_transform(X_gpu_trash)
            fail(msg="PCA Array dimensions check failed") # should not reach this line. The prev line should fail and go to the except block
        except ValueError:
            pass

    def test_pca_k_bigger_than_array_dims_and_getset(self):
        self.test_pca.set_n_components(self.N+1)
        self.assertEqual(self.test_pca.get_n_components(), self.N+1)
        T1 = self.test_pca.fit_transform(self.Xf)
        self.assertEqual(T1.shape[1], self.N) # should have been reset internally once the algorithm saw K was bigger than N
        T2 = self.test_pca.fit_transform(self.Xf[0:(self.N-1), 0:(self.N-2)].transpose())
        self.assertEqual(T2.shape[1], self.N-2) # should have been reset internally once the algorithm saw K was bigger than N	

    def test_pca_type_error_check(self):
        try:
            X_trash = np.random.rand(self.M, self.M, 3).astype(np.int64)
            X_gpu_trash = gpuarray.GPUArray(X_trash.shape, np.int64, order="F")
            X_gpu_trash.set(X_trash)
            self.test_pca2.fit_transform(X_gpu_trash)
            fail(msg="PCA Array data type check failed") # should not reach this line. The prev line should fail and go to the except block
        except ValueError:
            pass

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_svd_ss_cula_float32(self):
        a = np.asarray(np.random.randn(9, 6), np.float32)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 's', 's', lib='cula')
        assert_allclose(a, np.dot(u_gpu.get(),
                                      np.dot(np.diag(s_gpu.get()),
                                                vh_gpu.get())),
                            rtol=dtype_to_rtol[np.float32],
                            atol=dtype_to_atol[np.float32])

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_svd_ss_cula_float64(self):
        a = np.asarray(np.random.randn(9, 6), np.float64)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 's', 's', lib='cula')
        assert_allclose(a, np.dot(u_gpu.get(),
                                      np.dot(np.diag(s_gpu.get()),
                                                vh_gpu.get())),
                            rtol=dtype_to_rtol[np.float64],
                            atol=dtype_to_atol[np.float64])

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_svd_ss_cula_complex64(self):
        a = np.asarray(np.random.randn(9, 6) + 1j*np.random.randn(9, 6), np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 's', 's', lib='cula')
        assert_allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                            rtol=dtype_to_rtol[np.complex64],
                            atol=dtype_to_atol[np.complex64])

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_svd_ss_cula_complex128(self):
        a = np.asarray(np.random.randn(9, 6) + 1j*np.random.randn(9, 6), np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 's', 's', lib='cula')
        assert_allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                            rtol=dtype_to_rtol[np.complex128],
                            atol=dtype_to_atol[np.complex128])

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_svd_so_cula_float32(self):
        a = np.asarray(np.random.randn(6, 6), np.float32)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 's', 'o', lib='cula')
        assert_allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                            rtol=dtype_to_rtol[np.float64],
                            atol=dtype_to_atol[np.float32])

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_svd_so_cula_float64(self):
        a = np.asarray(np.random.randn(6, 6), np.float64)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 's', 'o', lib='cula')
        assert_allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                            rtol=dtype_to_rtol[np.float64],
                            atol=dtype_to_atol[np.float64])

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_svd_so_cula_complex64(self):
        a = np.asarray(np.random.randn(6, 6) + 1j*np.random.randn(6, 6), np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 's', 'o', lib='cula')
        assert_allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                            rtol=dtype_to_rtol[np.complex64],
                            atol=dtype_to_atol[np.complex64])

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_svd_so_cula_complex128(self):
        a = np.asarray(np.random.randn(6, 6) + 1j*np.random.randn(6, 6), np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 's', 'o', lib='cula')
        assert_allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                            rtol=dtype_to_rtol[np.complex128],
                            atol=dtype_to_atol[np.complex128])

    def test_svd_aa_cusolver_float32(self):
        a = np.asarray(np.random.randn(6, 6), np.float32)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, lib='cusolver')
        assert_allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                            rtol=dtype_to_rtol[np.float32],
                            atol=dtype_to_atol[np.float32])

    def test_svd_aa_cusolver_float64(self):
        a = np.asarray(np.random.randn(6, 6), np.float64)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, lib='cusolver')
        assert_allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                            rtol=dtype_to_rtol[np.float64],
                            atol=dtype_to_atol[np.float64])

    def test_svd_aa_cusolver_complex64(self):
        a = np.asarray(np.random.randn(6, 6) + 1j*np.random.randn(6, 6), np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, lib='cusolver')
        assert_allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                            rtol=dtype_to_rtol[np.complex64],
                            atol=dtype_to_atol[np.complex64])

    def test_svd_aa_cusolver_complex128(self):
        a = np.asarray(np.random.randn(6, 6) + 1j*np.random.randn(6, 6), np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, lib='cusolver')
        assert_allclose(a, np.dot(u_gpu.get(),
                                     np.dot(np.diag(s_gpu.get()),
                                            vh_gpu.get())),
                            rtol=dtype_to_rtol[np.complex128],
                            atol=dtype_to_atol[np.complex128])

    def _dot_matrix_vector_tests(self, dtype):
        a = np.asarray(np.random.rand(4, 4), dtype)
        b = np.asarray(np.random.rand(4), dtype)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = linalg.dot(a_gpu, b_gpu)
        assert_allclose(np.dot(a, b), c_gpu.get(), 
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])

        a = np.asarray(np.random.rand(4), dtype)
        b = np.asarray(np.random.rand(4, 4), dtype)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = linalg.dot(a_gpu, b_gpu)
        assert_allclose(np.dot(a, b), c_gpu.get(),
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])

        a = np.asarray(np.random.rand(4, 4), dtype)
        b = np.asarray(np.random.rand(4, 1), dtype)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = linalg.dot(a_gpu, b_gpu)
        assert_allclose(np.dot(a, b), c_gpu.get(),
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])

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
        assert_allclose(np.dot(aa, bb), c_gpu.get(),
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])

        a = a.astype(dtype, order="F", copy=True)
        b = b.astype(dtype, order="F", copy=True)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = linalg.dot(a_gpu, b_gpu, transa, transb)
        assert_allclose(np.dot(aa, bb), c_gpu.get(),
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])

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
        assert_allclose(np.dot(a.conj().T, b), c_gpu.get(),
                            rtol=dtype_to_rtol[np.complex64],
                            atol=dtype_to_atol[np.complex64])

        a = a.astype(np.complex64, order="F", copy=True)
        b = b.astype(np.complex64, order="F", copy=True)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = linalg.dot(a_gpu, b_gpu, 'c')
        assert_allclose(np.dot(a.conj().T, b), c_gpu.get(),
                            rtol=dtype_to_rtol[np.complex64],
                            atol=dtype_to_atol[np.complex64])

    def test_dot_matrix_h_complex128(self):
        a = np.asarray(np.random.rand(2, 4)+1j*np.random.rand(2, 4), np.complex128)
        b = np.asarray(np.random.rand(2, 2)+1j*np.random.rand(2, 2), np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = linalg.dot(a_gpu, b_gpu, 'c')
        assert_allclose(np.dot(a.conj().T, b), c_gpu.get(),
                            rtol=dtype_to_rtol[np.complex128],
                            atol=dtype_to_atol[np.complex128])

        a = a.astype(np.complex128, order="F", copy=True)
        b = b.astype(np.complex128, order="F", copy=True)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = linalg.dot(a_gpu, b_gpu, 'c')
        assert_allclose(np.dot(a.conj().T, b), c_gpu.get(),
                            rtol=dtype_to_rtol[np.complex128],
                            atol=dtype_to_atol[np.complex128])

    def test_dot_vector_float32(self):
        a = np.asarray(np.random.rand(5), np.float32)
        b = np.asarray(np.random.rand(5), np.float32)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c = linalg.dot(a_gpu, b_gpu)
        assert_allclose(np.dot(a, b), c,
                            rtol=dtype_to_rtol[np.float32],
                            atol=dtype_to_atol[np.float32])

        a = a.astype(np.float32, order="F", copy=True)
        b = b.astype(np.float32, order="F", copy=True)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c = linalg.dot(a_gpu, b_gpu)
        assert_allclose(np.dot(a, b), c,
                            rtol=dtype_to_rtol[np.float32],
                            atol=dtype_to_atol[np.float32])

    def test_dot_vector_float64(self):
        a = np.asarray(np.random.rand(5), np.float64)
        b = np.asarray(np.random.rand(5), np.float64)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c = linalg.dot(a_gpu, b_gpu)
        assert_allclose(np.dot(a, b), c,
                            rtol=dtype_to_rtol[np.float64],
                            atol=dtype_to_atol[np.float64])

        a = a.astype(np.float64, order="F", copy=True)
        b = b.astype(np.float64, order="F", copy=True)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c = linalg.dot(a_gpu, b_gpu)
        assert_allclose(np.dot(a, b), c,
                            rtol=dtype_to_rtol[np.float64],
                            atol=dtype_to_atol[np.float64])

    def test_dot_vector_complex64(self):
        a = np.asarray(np.random.rand(5), np.complex64)
        b = np.asarray(np.random.rand(5), np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c = linalg.dot(a_gpu, b_gpu)
        assert_allclose(np.dot(a, b), c,
                            rtol=dtype_to_rtol[np.complex64],
                            atol=dtype_to_atol[np.complex64])

        a = a.astype(np.complex64, order="F", copy=True)
        b = b.astype(np.complex64, order="F", copy=True)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c = linalg.dot(a_gpu, b_gpu)
        assert_allclose(np.dot(a, b), c,
                            rtol=dtype_to_rtol[np.complex64],
                            atol=dtype_to_atol[np.complex64])

    def test_dot_vector_complex128(self):
        a = np.asarray(np.random.rand(5), np.complex128)
        b = np.asarray(np.random.rand(5), np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c = linalg.dot(a_gpu, b_gpu)
        assert_allclose(np.dot(a, b), c,
                            rtol=dtype_to_rtol[np.complex128],
                            atol=dtype_to_atol[np.complex128])

        a = a.astype(np.complex128, order="F", copy=True)
        b = b.astype(np.complex128, order="F", copy=True)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c = linalg.dot(a_gpu, b_gpu)
        assert_allclose(np.dot(a, b), c,
                            rtol=dtype_to_rtol[np.complex128],
                            atol=dtype_to_atol[np.complex128])

    def test_mdot_matrix_float32(self):
        a = np.asarray(np.random.rand(4, 2), np.float32)
        b = np.asarray(np.random.rand(2, 2), np.float32)
        c = np.asarray(np.random.rand(2, 2), np.float32)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)
        d_gpu = linalg.mdot(a_gpu, b_gpu, c_gpu)
        assert_allclose(np.dot(a, np.dot(b, c)), d_gpu.get(),
                            rtol=dtype_to_rtol[np.float32],
                            atol=dtype_to_atol[np.float32])

    def test_mdot_matrix_float64(self):
        a = np.asarray(np.random.rand(4, 2), np.float64)
        b = np.asarray(np.random.rand(2, 2), np.float64)
        c = np.asarray(np.random.rand(2, 2), np.float64)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)
        d_gpu = linalg.mdot(a_gpu, b_gpu, c_gpu)
        assert_allclose(np.dot(a, np.dot(b, c)), d_gpu.get(),
                            rtol=dtype_to_rtol[np.float64],
                            atol=dtype_to_atol[np.float64])

    def test_mdot_matrix_complex64(self):
        a = np.asarray(np.random.rand(4, 2), np.complex64)
        b = np.asarray(np.random.rand(2, 2), np.complex64)
        c = np.asarray(np.random.rand(2, 2), np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)
        d_gpu = linalg.mdot(a_gpu, b_gpu, c_gpu)
        assert_allclose(np.dot(a, np.dot(b, c)), d_gpu.get(),
                            rtol=dtype_to_rtol[np.complex64],
                            atol=dtype_to_atol[np.complex64])

    def test_mdot_matrix_complex128(self):
        a = np.asarray(np.random.rand(4, 2), np.complex128)
        b = np.asarray(np.random.rand(2, 2), np.complex128)
        c = np.asarray(np.random.rand(2, 2), np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)
        d_gpu = linalg.mdot(a_gpu, b_gpu, c_gpu)
        assert_allclose(np.dot(a, np.dot(b, c)), d_gpu.get(),
                            rtol=dtype_to_rtol[np.complex128],
                            atol=dtype_to_atol[np.complex128])

    def __impl_test_dot_diag(self, dtype):
        d = np.asarray(np.random.rand(5), dtype)
        a = np.asarray(np.random.rand(5, 3), dtype)
        d_gpu = gpuarray.to_gpu(d)
        a_gpu = gpuarray.to_gpu(a)
        r_gpu = linalg.dot_diag(d_gpu, a_gpu)
        assert_allclose(np.dot(np.diag(d), a), r_gpu.get(),
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])

        a = a.astype(dtype, order="F", copy=True)
        d_gpu = gpuarray.to_gpu(d)
        a_gpu = gpuarray.to_gpu(a)
        # note: due to pycuda issue #66, this will fail when overwrite=False
        r_gpu = linalg.dot_diag(d_gpu, a_gpu, overwrite=True)
        assert_allclose(np.dot(np.diag(d), a), r_gpu.get(),
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])

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
        v = np.asarray(np.random.rand(5), dtype)
        a = np.asarray(np.random.rand(3, 5), dtype)
        d_gpu = gpuarray.to_gpu(d)
        a_gpu = gpuarray.to_gpu(a)
        r_gpu = linalg.dot_diag(d_gpu, a_gpu, 't')
        assert_allclose(np.dot(np.diag(d), a.T).T, r_gpu.get(),
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])

        a = a.astype(dtype, order="F", copy=True)
        d_gpu = gpuarray.to_gpu(d)
        a_gpu = gpuarray.to_gpu(a)
        # note: due to pycuda issue #66, this will fail when overwrite=False
        r_gpu = linalg.dot_diag(d_gpu, a_gpu, 't', overwrite=True)
        assert_allclose(np.dot(np.diag(d), a.T).T, r_gpu.get(),
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])

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
        assert_equal(a.T, at_gpu.get())
        # M > N
        b = a.T.copy()
        b_gpu = gpuarray.to_gpu(b)
        bt_gpu = linalg.transpose(b_gpu)
        assert_equal(b.T, bt_gpu.get())

    def test_transpose_float64(self):
        # M < N
        a = np.array([[1, 2, 3, 4, 5, 6],
                      [7, 8, 9, 10, 11, 12]],
                     np.float64)
        a_gpu = gpuarray.to_gpu(a)
        at_gpu = linalg.transpose(a_gpu)
        assert_equal(a.T, at_gpu.get())
        # M > N
        b = a.T.copy()
        b_gpu = gpuarray.to_gpu(b)
        bt_gpu = linalg.transpose(b_gpu)
        assert_equal(b.T, bt_gpu.get())

    def test_transpose_complex64(self):
        # M < N
        a = np.array([[1j, 2j, 3j, 4j, 5j, 6j],
                      [7j, 8j, 9j, 10j, 11j, 12j]],
                     np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        at_gpu = linalg.transpose(a_gpu)
        assert_equal(a.T, at_gpu.get())
        # M > N
        b = a.T.copy()
        b_gpu = gpuarray.to_gpu(b)
        bt_gpu = linalg.transpose(b_gpu)
        assert_equal(b.T, bt_gpu.get())

    def test_transpose_complex128(self):
        # M < N
        a = np.array([[1j, 2j, 3j, 4j, 5j, 6j],
                      [7j, 8j, 9j, 10j, 11j, 12j]],
                     np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        at_gpu = linalg.transpose(a_gpu)
        assert_equal(a.T, at_gpu.get())
        # M > N
        b = a.T.copy()
        b_gpu = gpuarray.to_gpu(b)
        bt_gpu = linalg.transpose(b_gpu)
        assert_equal(b.T, bt_gpu.get())

    def test_hermitian_float32(self):
        # M < N
        a = np.array([[1, 2, 3, 4, 5, 6],
                      [7, 8, 9, 10, 11, 12]],
                     np.float32)
        a_gpu = gpuarray.to_gpu(a)
        at_gpu = linalg.hermitian(a_gpu)
        assert_equal(a.T, at_gpu.get())
        # M > N
        b = a.T.copy()
        b_gpu = gpuarray.to_gpu(b)
        bt_gpu = linalg.hermitian(b_gpu)
        assert_equal(b.T, bt_gpu.get())

    def test_hermitian_complex64(self):
        # M < N
        a = np.array([[1j, 2j, 3j, 4j, 5j, 6j],
                      [7j, 8j, 9j, 10j, 11j, 12j]],
                     np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        at_gpu = linalg.hermitian(a_gpu)
        assert_equal(np.conj(a.T), at_gpu.get())
        # M > N
        b = a.T.copy()
        b_gpu = gpuarray.to_gpu(b)
        bt_gpu = linalg.hermitian(b_gpu)
        assert_equal(np.conj(b.T), bt_gpu.get())

    def test_hermitian_float64(self):
        # M < N
        a = np.array([[1, 2, 3, 4, 5, 6],
                      [7, 8, 9, 10, 11, 12]],
                      np.float64)
        a_gpu = gpuarray.to_gpu(a)
        at_gpu = linalg.hermitian(a_gpu)
        assert_equal(a.T, at_gpu.get())
        # M > N
        b = a.T.copy()
        b_gpu = gpuarray.to_gpu(b)
        bt_gpu = linalg.hermitian(b_gpu)
        assert_equal(b.T, bt_gpu.get())

    def test_hermitian_complex128(self):
        # M < N
        a = np.array([[1j, 2j, 3j, 4j, 5j, 6j],
                      [7j, 8j, 9j, 10j, 11j, 12j]],
                      np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        at_gpu = linalg.hermitian(a_gpu)
        assert_equal(np.conj(a.T), at_gpu.get())
        # M > N
        b = a.T.copy()
        b_gpu = gpuarray.to_gpu(b)
        bt_gpu = linalg.hermitian(b_gpu)
        assert_equal(np.conj(b.T), bt_gpu.get())

    def test_conj_complex64(self):
        a = np.array([[1+1j, 2-2j, 3+3j, 4-4j],
                      [5+5j, 6-6j, 7+7j, 8-8j]], np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        r_gpu = linalg.conj(a_gpu)
        assert_equal(np.conj(a), r_gpu.get())

    def test_conj_complex128(self):
        a = np.array([[1+1j, 2-2j, 3+3j, 4-4j],
                      [5+5j, 6-6j, 7+7j, 8-8j]], np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        r_gpu = linalg.conj(a_gpu)
        assert_equal(np.conj(a), r_gpu.get())

    def test_diag_1d_float32(self):
        v = np.array([1, 2, 3, 4, 5, 6], np.float32)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

    def test_diag_2d_wide_float32(self):
        v = np.asarray(np.random.rand(32, 64), np.float32)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

        v = np.asarray(np.random.rand(32, 64), np.float32, order="F")
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

        v = np.asarray(np.random.rand(32, 64), np.float32, order="F")
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

    def test_diag_2d_tall_float32(self):
        v = np.asarray(np.random.rand(64, 32), np.float32)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

        v = np.asarray(np.random.rand(64, 32), np.float32, order="F")
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

    def test_diag_1d_float64(self):
        v = np.array([1, 2, 3, 4, 5, 6], np.float64)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

    def test_diag_2d_wide_float64(self):
        v = np.asarray(np.random.rand(32, 64), np.float64)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

        v = np.asarray(np.random.rand(32, 64), np.float64, order="F")
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

    def test_diag_2d_tall_float64(self):
        v = np.asarray(np.random.rand(64, 32), np.float64)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

        v = np.asarray(np.random.rand(64, 32), np.float64, order="F")
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

        v = np.asarray(np.random.rand(64, 32), np.float64, order="F")
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

    def test_diag_1d_complex64(self):
        v = np.array([1j, 2j, 3j, 4j, 5j, 6j], np.complex64)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

    def test_diag_2d_wide_complex64(self):
        v = np.asarray(np.random.rand(32, 64)*1j, np.complex64)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

        v = np.asarray(np.random.rand(32, 64)*1j, np.complex64, order="F")
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

        v = np.asarray(np.random.rand(32, 64)*1j, np.complex64, order="F")
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

    def test_diag_2d_tall_complex64(self):
        v = np.asarray(np.random.rand(64, 32)*1j, np.complex64)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

        v = np.asarray(np.random.rand(64, 32)*1j, np.complex64, order="F")
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

        v = np.asarray(np.random.rand(64, 32)*1j, np.complex64, order="F")
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

    def test_diag_1d_complex128(self):
        v = np.array([1j, 2j, 3j, 4j, 5j, 6j], np.complex128)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

    def test_diag_2d_wide_complex128(self):
        v = np.asarray(np.random.rand(32, 64)*1j, np.complex128)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

        v = np.asarray(np.random.rand(32, 64)*1j, np.complex128, order="F")
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

        v = np.asarray(np.random.rand(32, 64)*1j, np.complex128, order="F")
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

    def test_diag_2d_tall_complex128(self):
        v = np.asarray(np.random.rand(64, 32)*1j, np.complex128)
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

        v = np.asarray(np.random.rand(64, 32)*1j, np.complex128, order="F")
        v_gpu = gpuarray.to_gpu(v)
        d_gpu = linalg.diag(v_gpu)
        assert_equal(np.diag(v), d_gpu.get())

    def test_eye_float32(self):
        N = 10
        e_gpu = linalg.eye(N, dtype=np.float32)
        assert_equal(np.eye(N, dtype=np.float32), e_gpu.get())

    def test_eye_float64(self):
        N = 10
        e_gpu = linalg.eye(N, dtype=np.float64)
        assert_equal(np.eye(N, dtype=np.float64), e_gpu.get())

    def test_eye_complex64(self):
        N = 10
        e_gpu = linalg.eye(N, dtype=np.complex64)
        assert_equal(np.eye(N, dtype=np.complex64), e_gpu.get())

    def test_eye_complex128(self):
        N = 10
        e_gpu = linalg.eye(N, dtype=np.complex128)
        assert_equal(np.eye(N, dtype=np.complex128), e_gpu.get())

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_pinv_cula_float32(self):
        a = np.asarray(np.random.rand(8, 4), np.float32)
        a_gpu = gpuarray.to_gpu(a)
        a_inv_gpu = linalg.pinv(a_gpu, lib='cula')
        assert_allclose(np.linalg.pinv(a), a_inv_gpu.get(),
                            atol=dtype_to_atol[np.float32],
                            rtol=dtype_to_rtol[np.float32])

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_pinv_cula_float64(self):
        a = np.asarray(np.random.rand(8, 4), np.float64)
        a_gpu = gpuarray.to_gpu(a)
        a_inv_gpu = linalg.pinv(a_gpu, lib='cula')
        assert_allclose(np.linalg.pinv(a), a_inv_gpu.get(),
                            atol=dtype_to_atol[np.float64],
                            rtol=dtype_to_rtol[np.float64])

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_pinv_cula_complex64(self):
        a = np.asarray(np.random.rand(8, 4) + \
                       1j*np.random.rand(8, 4), np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        a_inv_gpu = linalg.pinv(a_gpu, lib='cula')
        assert_allclose(np.linalg.pinv(a), a_inv_gpu.get(),
                            atol=dtype_to_atol[np.complex64],
                            rtol=dtype_to_rtol[np.complex64])

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_pinv_cula_complex128(self):
        a = np.asarray(np.random.rand(8, 4) + \
                       1j*np.random.rand(8, 4), np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        a_inv_gpu = linalg.pinv(a_gpu, lib='cula')
        assert_allclose(np.linalg.pinv(a), a_inv_gpu.get(),
                            atol=dtype_to_atol[np.complex128],
                            rtol=dtype_to_rtol[np.complex128])

    def test_pinv_cusolver_float32(self):
        a = np.asarray(np.random.rand(4, 8), np.float32)
        a_gpu = gpuarray.to_gpu(a)
        a_inv_gpu = linalg.pinv(a_gpu, lib='cusolver')
        assert_allclose(np.linalg.pinv(a), a_inv_gpu.get(),
                            atol=dtype_to_atol[np.float32],
                            rtol=dtype_to_rtol[np.float32])

    def test_pinv_cusolver_float64(self):
        a = np.asarray(np.random.rand(4, 8), np.float64)
        a_gpu = gpuarray.to_gpu(a)
        a_inv_gpu = linalg.pinv(a_gpu, lib='cusolver')
        assert_allclose(np.linalg.pinv(a), a_inv_gpu.get(),
                            atol=dtype_to_atol[np.float64],
                            rtol=dtype_to_rtol[np.float64])

    def test_pinv_cusolver_complex64(self):
        a = np.asarray(np.random.rand(4, 8) + \
                       1j*np.random.rand(4, 8), np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        a_inv_gpu = linalg.pinv(a_gpu, lib='cusolver')
        assert_allclose(np.linalg.pinv(a), a_inv_gpu.get(),
                            atol=dtype_to_atol[np.complex64],
                            rtol=dtype_to_rtol[np.complex64])                          

    def test_pinv_cusolver_complex128(self):
        a = np.asarray(np.random.rand(4, 8) + \
                       1j*np.random.rand(4, 8), np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        a_inv_gpu = linalg.pinv(a_gpu, lib='cusolver')
        assert_allclose(np.linalg.pinv(a), a_inv_gpu.get(),
                            atol=dtype_to_atol[np.complex128],
                            rtol=dtype_to_rtol[np.complex128])

    def test_tril_float32(self):
        a = np.asarray(np.random.rand(4, 4), np.float32)
        a_gpu = gpuarray.to_gpu(a)
        l_gpu = linalg.tril(a_gpu)
        assert_allclose(np.tril(a), l_gpu.get(),
                            atol=dtype_to_atol[np.float32],
                            rtol=dtype_to_rtol[np.float32])

    def test_tril_float64(self):
        a = np.asarray(np.random.rand(4, 4), np.float64)
        a_gpu = gpuarray.to_gpu(a)
        l_gpu = linalg.tril(a_gpu)
        assert_allclose(np.tril(a), l_gpu.get(),
                            atol=dtype_to_atol[np.float64],
                            rtol=dtype_to_rtol[np.float64]) 

    def test_tril_complex64(self):
        a = np.asarray(np.random.rand(4, 4), np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        l_gpu = linalg.tril(a_gpu)
        assert_allclose(np.tril(a), l_gpu.get(),
                            atol=dtype_to_atol[np.complex64],
                            rtol=dtype_to_rtol[np.complex64])                          

    def test_tril_complex128(self):
        a = np.asarray(np.random.rand(4, 4), np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        l_gpu = linalg.tril(a_gpu)
        assert_allclose(np.tril(a), l_gpu.get(),
                            atol=dtype_to_atol[np.complex128],
                            rtol=dtype_to_rtol[np.complex128])                          

    def test_triu_float32(self):
        a = np.asarray(np.random.rand(4, 4), np.float32)
        a_gpu = gpuarray.to_gpu(a)
        l_gpu = linalg.triu(a_gpu)
        assert_allclose(np.triu(a), l_gpu.get(),
                            atol=dtype_to_atol[np.float32],
                            rtol=dtype_to_rtol[np.float32])                          

    def test_triu_float64(self):
        a = np.asarray(np.random.rand(4, 4), np.float64)
        a_gpu = gpuarray.to_gpu(a)
        l_gpu = linalg.triu(a_gpu)
        assert_allclose(np.triu(a), l_gpu.get(),
                            atol=dtype_to_atol[np.float64],
                            rtol=dtype_to_rtol[np.float64])                          

    def test_triu_complex64(self):
        a = np.asarray(np.random.rand(4, 4), np.complex64)
        a_gpu = gpuarray.to_gpu(a)
        l_gpu = linalg.triu(a_gpu)
        assert_allclose(np.triu(a), l_gpu.get(),
                            atol=dtype_to_atol[np.complex64],
                            rtol=dtype_to_rtol[np.complex64])                          

    def test_triu_complex128(self):
        a = np.asarray(np.random.rand(4, 4), np.complex128)
        a_gpu = gpuarray.to_gpu(a)
        l_gpu = linalg.triu(a_gpu)
        assert_allclose(np.triu(a), l_gpu.get(),
                            atol=dtype_to_atol[np.complex128],
                            rtol=dtype_to_rtol[np.complex128])                          

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
        assert_allclose(x*y, z_gpu.get(),
                            atol=dtype_to_atol[dtype],
                            rtol=dtype_to_rtol[dtype])

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
        assert_allclose(c, np.triu(x_gpu.get()),
                            atol=dtype_to_atol[dtype],
                            rtol=dtype_to_rtol[dtype])

    def _impl_test_cholesky(self, N, dtype, lib='cula'):
        from scipy.linalg import cholesky as cpu_cholesky
        x = np.asarray(np.random.rand(N, N), dtype)
        if np.iscomplexobj(x):
            x += 1j*np.asarray(np.random.rand(N, N), dtype)
            x = np.dot(np.conj(x.T), x)
        else:
            x = np.dot(x.T, x)
        x_gpu = gpuarray.to_gpu(x)
        linalg.cholesky(x_gpu, 'L', lib)
        c = np.triu(cpu_cholesky(x))
        d = np.tril(cpu_cholesky(x), -1)
        assert_allclose(c, np.triu(x_gpu.get()),
                            atol=dtype_to_atol[dtype],
                            rtol=dtype_to_rtol[dtype])
        assert_allclose(d, np.tril(x_gpu.get(), -1),
                            atol=dtype_to_atol[dtype],
                            rtol=dtype_to_rtol[dtype])

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_cho_factor_cula_float32(self):
        self._impl_test_cho_factor(4, np.float32, 'cula')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_cho_factor_cula_float64(self):
        self._impl_test_cho_factor(4, np.float64, 'cula')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_cho_factor_cula_complex64(self):
        self._impl_test_cho_factor(4, np.complex64, 'cula')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_cho_factor_cula_complex128(self):
        self._impl_test_cho_factor(4, np.complex128, 'cula')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_cholesky_cula_float32(self):
        self._impl_test_cholesky(4, np.float32, 'cula')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_cholesky_cula_float64(self):
        self._impl_test_cholesky(4, np.float64, 'cula')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_cholesky_cula_complex64(self):
        self._impl_test_cholesky(4, np.complex64, 'cula')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_cholesky_cula_complex128(self):
        self._impl_test_cholesky(4, np.complex128, 'cula')

    def test_cho_factor_cusolver_float32(self):
        self._impl_test_cho_factor(4, np.float32, 'cusolver')

    def test_cho_factor_cusolver_float64(self):
        self._impl_test_cho_factor(4, np.float64, 'cusolver')

    def test_cho_factor_cusolver_complex64(self):
        self._impl_test_cho_factor(4, np.complex64, 'cusolver')

    def test_cho_factor_cusolver_complex128(self):
        self._impl_test_cho_factor(4, np.complex128, 'cusolver')

    def test_cholesky_cusolver_float32(self):
        self._impl_test_cholesky(4, np.float32, 'cusolver')

    def test_cholesky_cusolver_float64(self):
        self._impl_test_cholesky(4, np.float64, 'cusolver')

    def test_cholesky_cusolver_complex64(self):
        self._impl_test_cholesky(4, np.complex64, 'cusolver')

    def test_cholesky_cusolver_complex128(self):
        self._impl_test_cholesky(4, np.complex128, 'cusolver')

    def _impl_test_cho_solve(self, N, dtype, lib='cula'):
        x = np.asarray(np.random.rand(N, N), dtype)
        y = np.asarray(np.random.rand(N), dtype)
        if np.iscomplexobj(x):
            x += 1j*np.asarray(np.random.rand(N, N), dtype)
            x = np.dot(np.conj(x.T), x)
            y += 1j*np.asarray(np.random.rand(N), dtype)
            c = np.linalg.inv(x.T).dot(y)
        else:
            x = np.dot(x.T, x)
            c = np.linalg.inv(x.T).dot(y)

        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.to_gpu(y)
        linalg.cho_solve(x_gpu, y_gpu, lib=lib)
        assert_allclose(c, y_gpu.get(), atol=1e-1) # need higher tolerance for
                                                   # this test

        x = np.asarray(np.random.rand(N, N), dtype)
        y = np.asarray(np.random.rand(N, N), dtype, order="F")
        if np.iscomplexobj(x):
            x = np.dot(np.conj(x.T), x).astype(dtype, order="F", copy=True)
            y += 1j*np.asarray(np.random.rand(N, N), dtype, order="F")
            c = np.linalg.inv(x.T).dot(y)
        else:
            x = np.dot(x.T, x).astype(dtype, order="F", copy=True)
            c = np.linalg.inv(x.T).dot(y)

        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.to_gpu(y)
        linalg.cho_solve(x_gpu, y_gpu, lib=lib)
        assert_allclose(c, y_gpu.get(), atol=1e-1)

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_cho_solve_cula_float32(self):
        self._impl_test_cho_solve(4, np.float32, 'cula')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_cho_solve_cula_float64(self):
        self._impl_test_cho_solve(4, np.float64, 'cula')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_cho_solve_cula_complex64(self):
        self._impl_test_cho_solve(4, np.complex64, 'cula')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_cho_solve_cula_complex128(self):
        self._impl_test_cho_solve(4, np.complex128, 'cusolver')

    def test_cho_solve_cusolver_float32(self):
        self._impl_test_cho_solve(4, np.float32, 'cusolver')

    def test_cho_solve_cusolver_float64(self):
        self._impl_test_cho_solve(4, np.float64, 'cusolver')

    def test_cho_solve_cusolver_complex64(self):
        self._impl_test_cho_solve(4, np.complex64, 'cusolver')

    def test_cho_solve_cusolver_complex128(self):
        self._impl_test_cho_solve(4, np.complex128, 'cusolver')

    def _impl_test_inv(self, dtype, lib):
        from scipy.linalg import inv as cpu_inv
        x = np.asarray(np.random.rand(4, 4), dtype)
        x = np.dot(x.T, x)
        x_gpu = gpuarray.to_gpu(x)
        xinv = cpu_inv(x)
        xinv_gpu = linalg.inv(x_gpu, lib=lib)
        assert_allclose(xinv, xinv_gpu.get(),
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])
        assert xinv_gpu is not x_gpu
        xinv_gpu = linalg.inv(x_gpu, overwrite=True, lib=lib)
        assert_allclose(xinv, xinv_gpu.get(),
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])
        assert xinv_gpu is x_gpu

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_inv_cula_exceptions(self):
        x = np.asarray([[1, 2], [2, 4]], np.float32)
        x_gpu = gpuarray.to_gpu(x)
        assert_raises(linalg.LinAlgError, linalg.inv, x_gpu, lib='cula')

    def test_inv_cusolver_exceptions(self):
        x = np.asarray([[1, 2], [2, 4]], np.float32)
        x_gpu = gpuarray.to_gpu(x)
        assert_raises(linalg.LinAlgError, linalg.inv, x_gpu, lib='cusolver')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_inv_cula_float32(self):
        self._impl_test_inv(np.float32, 'cula')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_inv_cula_float64(self):
        self._impl_test_inv(np.float64, 'cula')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_inv_cula_complex64(self):
        self._impl_test_inv(np.complex64, 'cula')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_inv_cula_complex128(self):
        self._impl_test_inv(np.complex128, 'cula')

    def test_inv_cusolver_float32(self):
        self._impl_test_inv(np.float32, 'cusolver')

    def test_inv_cusolver_float64(self):
        self._impl_test_inv(np.float64, 'cusolver')

    def test_inv_cusolver_complex64(self):
        self._impl_test_inv(np.complex64, 'cusolver')

    def test_inv_cusolver_complex128(self):
        self._impl_test_inv(np.complex128, 'cusolver')

    def _impl_test_add_diag(self, dtype):
        x = np.asarray(np.random.rand(4, 4), dtype)
        d = np.asarray(np.random.rand(1, 4), dtype).reshape(-1)
        x_gpu = gpuarray.to_gpu(x)
        d_gpu = gpuarray.to_gpu(d)
        res_cpu = x + np.diag(d)
        res_gpu = linalg.add_diag(d_gpu, x_gpu, overwrite=False)
        assert_allclose(res_cpu, res_gpu.get(),
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])
        assert res_gpu is not x_gpu
        res_gpu = linalg.add_diag(d_gpu, x_gpu, overwrite=True)
        assert_allclose(res_cpu, res_gpu.get(),
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])
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
        assert_allclose(linalg.trace(x_gpu), np.trace(x),
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])
        # tall matrix
        x = np.asarray(np.random.rand(5, 2), dtype)
        x_gpu = gpuarray.to_gpu(x)
        assert_allclose(linalg.trace(x_gpu), np.trace(x),
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])
        # fat matrix
        x = np.asarray(np.random.rand(2, 5), dtype)
        x_gpu = gpuarray.to_gpu(x)
        assert_allclose(linalg.trace(x_gpu), np.trace(x),
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])

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
        assert_allclose(c + np.dot(aa, bb), c_gpu.get(),
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])

        a = a.astype(dtype, order="F", copy=True)
        b = b.astype(dtype, order="F", copy=True)
        c = c.astype(dtype, order="F", copy=True)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)
        c_gpu = linalg.add_dot(a_gpu, b_gpu, c_gpu, transa, transb)
        assert_allclose(c+np.dot(aa, bb), c_gpu.get(),
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])

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
        assert_allclose(np.dot(a[:, 4:6], b[:, 2:8]), res[:, 1:7],
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])

        # t/n
        a = np.asarray(np.random.rand(4, 10), dtype)
        b = np.asarray(np.random.rand(4, 20), dtype)
        c = np.zeros((2, 30), dtype)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)
        linalg.add_dot(a_gpu[:, 4:6], b_gpu[:, 2:8], c_gpu[:, 1:7], 't', 'n')
        res = c_gpu.get()
        assert_allclose(np.dot(a[:, 4:6].T, b[:, 2:8]), res[:, 1:7],
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])

        # n/t
        a = np.asarray(np.random.rand(4, 10), dtype)
        b = np.asarray(np.random.rand(6, 20), dtype)
        c = np.zeros((4, 30), dtype)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)
        linalg.add_dot(a_gpu[:, 4:10], b_gpu[:, 2:8], c_gpu[:, 1:7], 'n', 't')
        res = c_gpu.get()
        assert_allclose(np.dot(a[:, 4:10], b[:, 2:8].T), res[:, 1:7],
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])

        # t/t
        a = np.asarray(np.random.rand(6, 10), dtype)
        b = np.asarray(np.random.rand(8, 20), dtype)
        c = np.zeros((2, 30), dtype)
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)
        linalg.add_dot(a_gpu[:, 4:6], b_gpu[:, 2:8], c_gpu[:, 1:9], 't', 't')
        res = c_gpu.get()
        assert_allclose(np.dot(a[:, 4:6].T, b[:, 2:8].T), res[:, 1:9],
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])

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
        assert_allclose(linalg.det(x_gpu, lib=lib), np.linalg.det(x),
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])

        # known matrix (from http://en.wikipedia.org/wiki/Determinant )
        x = np.asarray([[-2.0, 2, -3.0], [-1, 1, 3], [2, 0, -1]], dtype)
        x_gpu = gpuarray.to_gpu(x)
        assert_allclose(linalg.det(x_gpu, lib=lib), 18.0,
                            rtol=dtype_to_rtol[dtype],
                            atol=dtype_to_atol[dtype])

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_det_cula_float32(self):
        self._impl_test_det(np.float32, 'cula')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_det_cula_float64(self):
        self._impl_test_det(np.float64, 'cula')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_det_cula_complex64(self):
        self._impl_test_det(np.complex64, 'cula')

    @skipUnless(linalg._has_cula, 'CULA required')
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

    def _impl_test_qr_reduced(self, dtype, lib):
        if np.issubdtype(dtype, np.complexfloating):
            a = np.asarray(np.random.randn(9, 6) + 1j*np.random.randn(9, 6), dtype, order='F')
        else:
            a = np.asarray(np.random.randn(5, 3), dtype, order='F')
        a_gpu = gpuarray.to_gpu(a)
        q_gpu, r_gpu = linalg.qr(a_gpu, 'reduced', lib=lib)
        assert_allclose(a, np.dot(q_gpu.get(), r_gpu.get()), atol=1e-4)

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_qr_reduced_cula_float32(self):
        self._impl_test_qr_reduced(np.float32, 'cula')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_qr_reduced_cula_float64(self):
        self._impl_test_qr_reduced(np.float64, 'cula')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_qr_reduced_cula_complex64(self):
        self._impl_test_qr_reduced(np.complex64, 'cula')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_qr_reduced_cula_complex128(self):
        self._impl_test_qr_reduced(np.complex128, 'cula')

    def test_qr_reduced_cusolver_float32(self):
        self._impl_test_qr_reduced(np.float32, 'cusolver')

    def test_qr_reduced_cusolver_float64(self):
        self._impl_test_qr_reduced(np.float64, 'cusolver')

    def test_qr_reduced_cusolver_complex64(self):
        self._impl_test_qr_reduced(np.complex64, 'cusolver')

    def test_qr_reduced_cusolver_complex128(self):
        self._impl_test_qr_reduced(np.complex128, 'cusolver')

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_eig_cula_float32(self):
        a = np.asarray(np.random.rand(9, 9), np.float32, order='F')
        a_gpu = gpuarray.to_gpu(a)
        w_gpu = linalg.eig(a_gpu, 'N', 'N', lib='cula')
        assert_allclose(np.trace(a), sum(w_gpu.get()), atol=1e-4)

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_eig_cula_float64(self):
        a = np.asarray(np.random.rand(9, 9), np.float64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        w_gpu = linalg.eig(a_gpu, 'N', 'N', lib='cula')
        assert_allclose(np.trace(a), sum(w_gpu.get()),
                            rtol=dtype_to_rtol[np.float64],
                            atol=dtype_to_atol[np.float64])

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_eig_cula_complex64(self):
        a = np.asarray(np.random.rand(9, 9) + 1j*np.random.rand(9, 9), np.complex64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        w_gpu = linalg.eig(a_gpu, 'N', 'N', lib='cula')
        assert_allclose(np.trace(a), sum(w_gpu.get()), atol=1e-4)

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_eig_cula_complex128(self):
        a = np.array(np.random.rand(9, 9) + 1j*np.random.rand(9,9), np.complex128, order='F')
        a_gpu = gpuarray.to_gpu(a)
        w_gpu = linalg.eig(a_gpu, 'N', 'N', lib='cula')
        assert_allclose(np.trace(a), sum(w_gpu.get()),
                            rtol=dtype_to_rtol[np.complex128],
                            atol=dtype_to_atol[np.complex128])

    def test_eig_cusolver_float32(self):
        tmp = np.random.rand(9, 9)
        a = np.asarray(np.dot(tmp, tmp.T), np.float32, order='F')
        a_gpu = gpuarray.to_gpu(a)
        w_gpu = linalg.eig(a_gpu, 'N', 'N', lib='cusolver')
        assert_allclose(np.trace(a), sum(w_gpu.get()), atol=1e-4)

    def test_eig_cusolver_float64(self):
        tmp = np.random.rand(9, 9)
        a = np.asarray(np.dot(tmp, tmp.T), np.float64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        w_gpu = linalg.eig(a_gpu, 'N', 'N', lib='cusolver')
        assert_allclose(np.trace(a), sum(w_gpu.get()),
                            rtol=dtype_to_rtol[np.float64],
                            atol=dtype_to_atol[np.float64])

    def test_eig_cusolver_complex64(self):
        tmp = np.random.rand(9, 9)+1j*np.random.rand(9, 9)
        a = np.asarray(np.dot(tmp, tmp.T.conj()), np.complex64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        w_gpu = linalg.eig(a_gpu, 'N', 'N', lib='cusolver')
        assert_allclose(np.trace(a), sum(w_gpu.get()), atol=1e-4)

    def test_eig_cusolver_complex128(self):
        tmp = np.random.rand(9, 9)+1j*np.random.rand(9, 9)
        a = np.asarray(np.dot(tmp, tmp.T.conj()), np.complex128, order='F')
        a_gpu = gpuarray.to_gpu(a)
        w_gpu = linalg.eig(a_gpu, 'N', 'N', lib='cusolver')
        assert_allclose(np.trace(a), sum(w_gpu.get()),
                            rtol=dtype_to_rtol[np.float64],
                            atol=dtype_to_atol[np.float64])

    def test_vander_float32(self):
        a = np.array(np.random.uniform(1,2,5), np.float32, order='F')
        a_gpu = gpuarray.to_gpu(a)
        vander_gpu = linalg.vander(a_gpu)
        assert_allclose(np.fliplr(np.vander(a)), vander_gpu.get(),
                            rtol=dtype_to_rtol[np.float32],
                            atol=dtype_to_atol[np.float32])

    def test_vander_float64(self):
        a = np.array(np.random.uniform(1,2,5), np.float64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        vander_gpu = linalg.vander(a_gpu)
        assert_allclose(np.fliplr(np.vander(a)), vander_gpu.get(),
                            rtol=dtype_to_rtol[np.float64],
                            atol=dtype_to_atol[np.float64])

    def test_vander_complex64(self):
        a = np.array(np.random.uniform(1,2,5) + 1j*np.random.uniform(1,2,5), np.complex64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        vander_gpu = linalg.vander(a_gpu)
        assert_allclose(np.fliplr(np.vander(a)), vander_gpu.get(),
                            rtol=dtype_to_rtol[np.complex64],
                            atol=dtype_to_atol[np.complex64])

    def test_vander_complex128(self):
        a = np.array(np.random.uniform(1,2,5) + 1j*np.random.uniform(1,2,5), np.complex128, order='F')
        a_gpu = gpuarray.to_gpu(a)
        vander_gpu = linalg.vander(a_gpu)
        assert_allclose(np.fliplr(np.vander(a)), vander_gpu.get(),
                            rtol=dtype_to_rtol[np.complex128],
                            atol=dtype_to_atol[np.complex128])

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_dmd_float32(self):
        m, n = 6, 4
        a = np.array(np.fliplr(np.vander(np.random.rand(m)+1, n)), np.float32, order='F')
        a_gpu = gpuarray.to_gpu(a)
        f_gpu, b_gpu, v_gpu, omega = linalg.dmd(a_gpu, modes='standard', return_amplitudes=True, return_vandermonde=True)
        assert_allclose(a[:,:(n-1)], np.dot(f_gpu.get(), np.dot(np.diag(b_gpu.get()), v_gpu.get()) ), 1e-4)

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_dmd_float64(self):
        m, n = 9, 7
        a = np.array(np.fliplr(np.vander(np.random.rand(m)+1, n)), np.float64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        f_gpu, b_gpu, v_gpu, omega = linalg.dmd(a_gpu, modes='standard', return_amplitudes=True, return_vandermonde=True)
        assert_allclose(a[:,:(n-1)], np.dot(f_gpu.get(), np.dot(np.diag(b_gpu.get()), v_gpu.get()) ),
                            rtol=dtype_to_rtol[np.float64],
                            atol=dtype_to_atol[np.float64])

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_dmd_complex64(self):
        m, n = 9, 7
        a = np.array(np.fliplr(np.vander(np.random.rand(m)+1, n)) + 1j*np.fliplr(np.vander(np.random.rand(m)+1, n)),
                     np.complex64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        f_gpu, b_gpu, v_gpu, omega = linalg.dmd(a_gpu, modes='standard', return_amplitudes=True, return_vandermonde=True)
        assert_allclose(a[:,:(n-1)], np.dot(f_gpu.get(), np.dot(np.diag(b_gpu.get()), v_gpu.get()) ), 1e-4)

    @skipUnless(linalg._has_cula, 'CULA required')
    def test_dmd_complex128(self):
        m, n = 9, 7
        a = np.array(np.fliplr(np.vander(np.random.rand(m)+1, n)) + 1j*np.fliplr(np.vander(np.random.rand(m)+1, n)),
                     np.complex128, order='F')
        a_gpu = gpuarray.to_gpu(a)
        f_gpu, b_gpu, v_gpu, omega = linalg.dmd(a_gpu, modes='standard', return_amplitudes=True, return_vandermonde=True)
        assert_allclose(a[:,:(n-1)], np.dot(f_gpu.get(), np.dot(np.diag(b_gpu.get()), v_gpu.get()) ),
                            rtol=dtype_to_rtol[np.complex128],
                            atol=dtype_to_atol[np.complex128])

def suite():
    context = make_default_context()
    device = context.get_device()
    context.pop()

    s = TestSuite()
    s.addTest(test_linalg('test_pca_ortho_type_and_shape_float32_all_comp'))
    s.addTest(test_linalg('test_pca_ortho_type_and_shape_float32'))
    s.addTest(test_linalg('test_pca_f_contiguous_check'))
    s.addTest(test_linalg('test_pca_arr_2d_check'))
    s.addTest(test_linalg('test_pca_k_bigger_than_array_dims_and_getset'))
    s.addTest(test_linalg('test_pca_type_error_check'))
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
    s.addTest(test_linalg('test_pinv_cula_float32'))
    s.addTest(test_linalg('test_pinv_cula_complex64'))
    s.addTest(test_linalg('test_pinv_cusolver_float32'))
    s.addTest(test_linalg('test_pinv_cusolver_complex64'))
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
    s.addTest(test_linalg('test_cholesky_cula_float32'))
    s.addTest(test_linalg('test_cholesky_cula_complex64'))
    s.addTest(test_linalg('test_cholesky_cusolver_float32'))
    s.addTest(test_linalg('test_cholesky_cusolver_complex64'))
    s.addTest(test_linalg('test_cho_solve_cula_float32'))
    s.addTest(test_linalg('test_cho_solve_cula_complex64'))
    s.addTest(test_linalg('test_cho_solve_cusolver_float32'))
    s.addTest(test_linalg('test_cho_solve_cusolver_complex64'))
    s.addTest(test_linalg('test_inv_cula_float32'))
    s.addTest(test_linalg('test_inv_cula_complex64'))
    s.addTest(test_linalg('test_inv_cusolver_float32'))
    s.addTest(test_linalg('test_inv_cusolver_complex64'))
    s.addTest(test_linalg('test_add_diag_float32'))
    s.addTest(test_linalg('test_add_diag_complex64'))
    s.addTest(test_linalg('test_inv_cula_exceptions'))
    s.addTest(test_linalg('test_inv_cusolver_exceptions'))
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
    s.addTest(test_linalg('test_qr_reduced_cula_float32'))
    s.addTest(test_linalg('test_qr_reduced_cula_complex64'))
    s.addTest(test_linalg('test_qr_reduced_cusolver_float32'))
    s.addTest(test_linalg('test_qr_reduced_cusolver_complex64'))
    s.addTest(test_linalg('test_eig_cula_float32'))
    s.addTest(test_linalg('test_eig_cula_complex64'))
    s.addTest(test_linalg('test_eig_cusolver_float32'))
    s.addTest(test_linalg('test_eig_cusolver_complex64'))
    s.addTest(test_linalg('test_vander_float32'))
    s.addTest(test_linalg('test_vander_complex64'))
    s.addTest(test_linalg('test_dmd_float32'))
    s.addTest(test_linalg('test_dmd_complex64'))

    if misc.get_compute_capability(device) >= 1.3:
        s.addTest(test_linalg('test_pca_ortho_type_and_shape_float64_all_comp'))
        s.addTest(test_linalg('test_pca_ortho_type_and_shape_float64'))
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
        s.addTest(test_linalg('test_pinv_cula_float64'))
        s.addTest(test_linalg('test_pinv_cula_complex128'))
        s.addTest(test_linalg('test_pinv_cusolver_float64'))
        s.addTest(test_linalg('test_pinv_cusolver_complex128'))
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
        s.addTest(test_linalg('test_cholesky_cula_float64'))
        s.addTest(test_linalg('test_cholesky_cula_complex128'))
        s.addTest(test_linalg('test_cholesky_cusolver_float64'))
        s.addTest(test_linalg('test_cholesky_cusolver_complex128'))
        s.addTest(test_linalg('test_cho_solve_cula_float64'))
        s.addTest(test_linalg('test_cho_solve_cula_complex128'))
        s.addTest(test_linalg('test_cho_solve_cusolver_float64'))
        s.addTest(test_linalg('test_cho_solve_cusolver_complex128'))
        s.addTest(test_linalg('test_inv_cula_float64'))
        s.addTest(test_linalg('test_inv_cula_complex128'))
        s.addTest(test_linalg('test_inv_cusolver_float64'))
        s.addTest(test_linalg('test_inv_cusolver_complex128'))
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
        s.addTest(test_linalg('test_qr_reduced_cula_float64'))
        s.addTest(test_linalg('test_qr_reduced_cula_complex128'))
        s.addTest(test_linalg('test_qr_reduced_cusolver_float64'))
        s.addTest(test_linalg('test_qr_reduced_cusolver_complex128'))
        s.addTest(test_linalg('test_eig_cula_float64'))
        s.addTest(test_linalg('test_eig_cula_complex128'))
        s.addTest(test_linalg('test_eig_cusolver_float64'))
        s.addTest(test_linalg('test_eig_cusolver_complex128'))
        s.addTest(test_linalg('test_vander_float64'))
        s.addTest(test_linalg('test_vander_complex128'))
        s.addTest(test_linalg('test_dmd_float64'))
        s.addTest(test_linalg('test_dmd_complex128'))

    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
