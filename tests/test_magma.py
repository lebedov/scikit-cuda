"""
Unit tests for skcuda.magma
"""
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
from unittest import main, makeSuite, skipUnless, TestCase, TestSuite

# pycuda stuff
import pycuda.driver as drv
# import pycuda.gpuarray as gpuarray
from pycuda.tools import clear_context_caches, make_default_context

import skcuda.magma as magma

drv.init()

dtype_to_atol = {np.float32: 1e-5,
                 np.complex64: 1e-5,
                 np.float64: 1e-8,
                 np.complex128: 1e-8}
dtype_to_rtol = {np.float32: 5e-5,
                 np.complex64: 5e-5,
                 np.float64: 1e-5,
                 np.complex128: 1e-5}

dtype_d = { 's': np.float32,
            'd': np.float64,
            'c': np.complex64,
            'z': np.complex128 }

class test_magma(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = make_default_context()
        magma.magma_init()

    @classmethod
    def tearDownClass(cls):
        magma.magma_finalize()
        cls.ctx.pop()
        clear_context_caches()
    
    def setUp(self):
        np.random.seed(0)
        self.N = 500
        
    def _test_magma_geev_novecs(self, dtype, jobvl, jobvr):
        """
        test geev for eigenvalues only
        dtype: s, d, c, z
        """
        dtype_ = dtype_d[dtype]
        dtype_float = np.float32 if dtype in ['s', 'c'] else np.float64

        mat = np.random.rand(self.N*self.N)
        if dtype in ['c', 'z']:
            mat = mat + 1j*np.random.rand(self.N*self.N)
        
        mat = mat.astype(dtype_).reshape((self.N, self.N), order='F')
        mat_= mat.copy()

        if dtype in ['s', 'd']:
            wr = np.zeros((self.N, ), dtype=dtype_)
            wi = np.zeros((self.N, ), dtype=dtype_)
        else:
            w = np.zeros((self.N, ), dtype=dtype_)
            
        if jobvl == 'N':
            vl = np.zeros((1, 1), dtype=dtype_)
        else:
            vl = np.zeros((self.N, self.N), dtype=dtype_)

        if jobvr == 'N':
            vr = np.zeros((1, 1), dtype=dtype_)
        else:
            vr = np.zeros((self.N, self.N), dtype=dtype_)
        
        if dtype == 's':
            nb = magma.magma_get_sgeqrf_nb(self.N, self.N)
        elif dtype == 'd':
            nb = magma.magma_get_dgeqrf_nb(self.N, self.N)
        elif dtype == 'c':
            nb = magma.magma_get_cgeqrf_nb(self.N, self.N)
        elif dtype == 'z':
            nb = magma.magma_get_zgeqrf_nb(self.N, self.N)
        
        lwork = self.N * (1+2*nb)
        work = np.zeros((lwork, ), dtype=dtype_)
        if dtype in ['c', 'z']:
            rwork= np.zeros((2*self.N,), dtype=dtype_float)
        
        if dtype == 's':
            status = magma.magma_sgeev(
                        jobvl, jobvr, self.N, mat.ctypes.data, self.N,
                        wr.ctypes.data, wi.ctypes.data,
                        vl.ctypes.data, vl.shape[0], vr.ctypes.data, vr.shape[0],
                        work.ctypes.data, lwork)
        elif dtype == 'd':
            status = magma.magma_dgeev(
                        jobvl, jobvr, self.N, mat.ctypes.data, self.N,
                        wr.ctypes.data, wi.ctypes.data,
                        vl.ctypes.data, vl.shape[0], vr.ctypes.data, vr.shape[0],
                        work.ctypes.data, lwork)
        elif dtype == 'c':
            status = magma.magma_cgeev(
                        jobvl, jobvr, self.N, mat.ctypes.data, self.N,
                        w.ctypes.data, vl.ctypes.data, vl.shape[0], vr.ctypes.data, vr.shape[0],
                        work.ctypes.data, lwork, rwork.ctypes.data)

        elif dtype == 'z':
            status = magma.magma_zgeev(
                        jobvl, jobvr, self.N, mat.ctypes.data, self.N,
                        w.ctypes.data,
                        vl.ctypes.data, vl.shape[0], vr.ctypes.data, vr.shape[0],
                        work.ctypes.data, lwork, rwork.ctypes.data)

        if dtype in ['s', 'd']:
            if np.allclose(wi, np.zeros(wi.shape[0])):
                # if w is real
                w_magma = np.sort(wr)
            else:
                w_magma = np.sort(wr+1j*wi)
        else:
            w_magma = np.sort(w)

        w_numpy = np.sort(np.linalg.eigvals(mat_))

        assert_allclose(w_numpy, w_magma, 
                        rtol=dtype_to_rtol[dtype_],
                        atol=dtype_to_atol[dtype_])

    def test_magma_geev_novecs(self):
        """
        Tests for sgeev, dgeev, cgeev, zgeev
        without vector output
        """
        self._test_magma_geev_novecs('s', jobvl='N', jobvr='N')
        self._test_magma_geev_novecs('d', jobvl='N', jobvr='N')
        self._test_magma_geev_novecs('c', jobvl='N', jobvr='N')
        self._test_magma_geev_novecs('z', jobvl='N', jobvr='N')
        
def suite():
    context = make_default_context()
    device = context.get_device()
    context.pop()
    
    s = TestSuite()
    s.addTest(test_magma('test_magma_geev_novecs'))
    
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
    
    