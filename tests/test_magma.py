"""
Unit tests for skcuda.magma

Contains following tests:
General eigensolvers: magma_[s,d,c,z]geev
Real symmetric eigensolvers: magma_[s,d]syevd[,x][,_m,_gpu]
"""
from unittest import (
    main,
    # makeSuite, 
    # skipUnless, 
    TestCase,
    TestSuite
)
import numpy as np
from numpy.testing import assert_allclose


# pycuda stuff
try:
    import pycuda.autoinit
    import pycuda.driver as drv
    import pycuda.gpuarray as gpuarray
except:
    raise ImportError("Please check the pycuda installation")
from pycuda.tools import clear_context_caches, make_default_context

import skcuda.magma as magma

DTYPE_TO_ATOL = {np.float32: 1e-5,
                 np.complex64: 1e-5,
                 np.float64: 1e-8,
                 np.complex128: 1e-8}
DTYPE_TO_RTOL = {np.float32: 5e-5,
                 np.complex64: 5e-5,
                 np.float64: 1e-5,
                 np.complex128: 1e-5}

TYPEDICT = {'s': np.float32, 'd': np.float64, 'c': np.complex64, 'z': np.complex128}

class test_magma(TestCase):
    """tests for magma functions
    """
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

    def _test_magma_geev_novecs(self, type_key, jobvl, jobvr):
        """
        test geev for eigenvalues only
        dtype: s, d, c, z
        """
        dtype = TYPEDICT[type_key]
        dtype_float = np.float32 if type_key in ['s', 'c'] else np.float64

        mat = np.random.rand(self.N*self.N)
        if type_key in ['c', 'z']:
            mat = mat + 1j*np.random.rand(self.N*self.N)
        
        mat = mat.astype(dtype).reshape((self.N, self.N), order='F')
        mat_numpy = mat.copy() # cpu

        if type_key in ['s', 'd']:
            wr = np.zeros((self.N, ), dtype=dtype)
            wi = np.zeros((self.N, ), dtype=dtype)
        else:
            w = np.zeros((self.N, ), dtype=dtype)

        vl = np.zeros((1, 1), dtype=dtype) if jobvl == 'N' \
             else np.zeros((self.N, self.N), dtype=dtype)
        vr = np.zeros((1, 1), dtype=dtype) if jobvr == 'N' \
             else np.zeros((self.N, self.N), dtype=dtype)

        if type_key == 's':
            nb = magma.magma_get_sgeqrf_nb(self.N, self.N)
        elif type_key == 'd':
            nb = magma.magma_get_dgeqrf_nb(self.N, self.N)
        elif type_key == 'c':
            nb = magma.magma_get_cgeqrf_nb(self.N, self.N)
        elif type_key == 'z':
            nb = magma.magma_get_zgeqrf_nb(self.N, self.N)

        lwork = self.N * (1+2*nb)
        work = np.zeros((lwork, ), dtype=dtype)
        if type_key in ['c', 'z']:
            rwork = np.zeros((2*self.N,), dtype=dtype_float)

        if type_key == 's':
            magma.magma_sgeev(
                jobvl, jobvr, self.N, mat.ctypes.data, self.N,
                wr.ctypes.data, wi.ctypes.data,
                vl.ctypes.data, vl.shape[0], vr.ctypes.data, vr.shape[0],
                work.ctypes.data, lwork)
        elif type_key == 'd':
            magma.magma_dgeev(
                jobvl, jobvr, self.N, mat.ctypes.data, self.N,
                wr.ctypes.data, wi.ctypes.data,
                vl.ctypes.data, vl.shape[0], vr.ctypes.data, vr.shape[0],
                work.ctypes.data, lwork)
        elif type_key == 'c':
            magma.magma_cgeev(
                jobvl, jobvr, self.N, mat.ctypes.data, self.N,
                w.ctypes.data, vl.ctypes.data, vl.shape[0], vr.ctypes.data, vr.shape[0],
                work.ctypes.data, lwork, rwork.ctypes.data)

        elif type_key == 'z':
            magma.magma_zgeev(
                jobvl, jobvr, self.N, mat.ctypes.data, self.N,
                w.ctypes.data,
                vl.ctypes.data, vl.shape[0], vr.ctypes.data, vr.shape[0],
                work.ctypes.data, lwork, rwork.ctypes.data)

        if type_key in ['s', 'd']:
            if np.allclose(wi, np.zeros(wi.shape[0])):
                # if w is real
                w_magma = np.sort(wr)
            else:
                w_magma = np.sort(wr+1j*wi)
        else:
            w_magma = np.sort(w)

        w_numpy = np.sort(np.linalg.eigvals(mat_numpy))

        assert_allclose(w_numpy, w_magma,
                        rtol=DTYPE_TO_RTOL[dtype],
                        atol=DTYPE_TO_ATOL[dtype])

    def test_magma_geev_novecs(self):
        """Testing eigensolvers: [s,d,c,z]geev
        """
        self._test_magma_geev_novecs('s', jobvl='N', jobvr='N')
        self._test_magma_geev_novecs('d', jobvl='N', jobvr='N')
        self._test_magma_geev_novecs('c', jobvl='N', jobvr='N')
        self._test_magma_geev_novecs('z', jobvl='N', jobvr='N')

    def _test_syev(self, N, type_key, data_gpu=False, ngpu=1, expert=False, keigs=None):
        """A simple test of the eigenvalue solver for symmetric matrix.
        Random matrix is used.

        :param N: size of the testing matrix
        :param type_key: data type: one of 's', 'd'
        :type type_key: string
        :param data_gpu: whether data is stored on the gpu initially, i.e., `_gpu`
        :type data_gpu: bool, optional
        :param ngpu: number of gpu, use -1 for running single gpu with `_m` functions
        :type ngpu: int, optional
        :param expert: use of expert variant (```x```)
        :type expert: bool, optional
        :param keigs: number of eigenvalue to find; find all when ```keigs=-1```
        :type keigs: int, optional
        """
        dtype = TYPEDICT[type_key]
        if dtype in [np.complex64, np.complex128]:
            raise ValueError("Complex types are not supported yet.")

        # not symmetric, but only side of the diagonal is used
        np.random.seed(1234)
        mat = np.random.random((N, N)).astype(dtype)
        mat_numpy = mat.copy() # for comparison

        # GPU
        jobz = 'N' # do not compute eigenvectors
        uplo = 'U' # use upper diagonal
        if expert:
            rnge = 'I'
            vl = 0; vu = 1e10
            if keigs is None:
                keigs = int(np.ceil(N/2))
            il = 1; iu = keigs
            m = np.zeros((1,), dtype=int) # no. of eigenvalues found

        # allocate memory for eigenvalues
        w_magma = np.empty((N,), dtype, order='F')

        if type_key == 's':
            nb = magma.magma_get_ssytrd_nb(N)
        elif type_key == 'd':
            nb = magma.magma_get_dsytrd_nb(N)
        else:
            raise ValueError('unsupported type')

        lwork = N*(1 + 2*nb)
        if jobz:
            lwork = max(lwork, 1+6*N+2*N**2)
        work = np.empty(lwork, dtype)
        liwork = 3+5*N if jobz else 1
        iwork = np.empty(liwork, dtype)
        if data_gpu:
            ldwa = 2*max(1, N)
            worka = np.empty((ldwa, N), order='F')

        if data_gpu:
            ngpu = 1
            if type_key == 's':
                magma_function = magma.magma_ssyevdx_gpu if expert \
                                    else magma.magma_ssyevd_gpu
            elif type_key == 'd':
                magma_function = magma.magma_dsyevdx_gpu if expert \
                                    else magma.magma_dsyevd_gpu
        else:
            if ngpu > 1 or ngpu == -1:
                ngpu = abs(ngpu)
                if type_key == 's':
                    magma_function = magma.magma_ssyevdx_m if expert \
                                        else magma.magma_ssyevd_m
                elif type_key == 'd':
                    magma_function = magma.magma_dsyevdx_m if expert \
                                        else magma.magma_dsyevd_m
            elif ngpu == 1:
                if type_key == 's':
                    magma_function = magma.magma_ssyevdx if expert \
                                        else magma.magma_ssyevd
                elif type_key == 'd':
                    magma_function = magma.magma_dsyevdx if expert \
                                        else magma.magma_dsyevd
            else:
                raise ValueError(f"ngpu={ngpu} is not supported, it must be -1 or >= 1")

        if '_m' in magma_function.__name__:
            # multi-gpu
            if 'x' in magma_function.__name__:
                # expert
                magma_function(ngpu, jobz, rnge, uplo, N, mat.ctypes.data, N,
                               vl, vu, il, iu, m.ctypes.data,
                               w_magma.ctypes.data, work.ctypes.data,
                               lwork, iwork.ctypes.data, liwork)
            else:
                # non-expert
                status = magma_function(ngpu, jobz, uplo, N, mat.ctypes.data, N,
                                        w_magma.ctypes.data, work.ctypes.data,
                                        lwork, iwork.ctypes.data, liwork)
        elif '_gpu' in magma_function.__name__:
            # data-on-gpu
            mat_gpu = gpuarray.to_gpu(mat)
            if 'x' in magma_function.__name__:
                magma_function(jobz, rnge, uplo, N, mat_gpu.gpudata, N,
                               vl, vu, il, iu, m.ctypes.data,
                               w_magma.ctypes.data, worka.ctypes.data, ldwa,
                               work.ctypes.data, lwork, iwork.ctypes.data, liwork)
            else:
                magma_function(jobz, uplo, N, mat_gpu.gpudata, N,
                               w_magma.ctypes.data, worka.ctypes.data, ldwa,
                               work.ctypes.data, lwork, iwork.ctypes.data, liwork)
        else:
            # single-gpu
            if 'x' in magma_function.__name__:
                # expert
                magma_function(jobz, rnge, uplo, N, mat.ctypes.data, N,
                               vl, vu, il, iu, m.ctypes.data,
                               w_magma.ctypes.data, work.ctypes.data,
                               lwork, iwork.ctypes.data, liwork)
            else:
                # non-expert
                magma_function(jobz, uplo, N, mat.ctypes.data, N,
                               w_magma.ctypes.data, work.ctypes.data,
                               lwork, iwork.ctypes.data, liwork)
        w_magma = np.sort(w_magma)

        # CPU
        w_numpy, v_numpy = np.linalg.eigh(mat_numpy)
        w_numpy = np.sort(w_numpy)

        assert_allclose(w_numpy, w_magma,
                        rtol=DTYPE_TO_RTOL[dtype],
                        atol=DTYPE_TO_ATOL[dtype])

    def test_symmetric_eig_float32(self):
        """Testing eigensolvers: magma_ssyevd[,x][,_gpu,_m]
        """
        self._test_syev(700, 's', data_gpu=False, ngpu=1, expert=False)    #
        self._test_syev(700, 's', data_gpu=False, ngpu=-1, expert=False)   # _m
        self._test_syev(700, 's', data_gpu=True, ngpu=1, expert=False)    # _gpu
        # expert
        self._test_syev(700, 's', data_gpu=False, ngpu=1, expert=True)    #
        self._test_syev(700, 's', data_gpu=False, ngpu=-1, expert=True)   # _m
        self._test_syev(700, 's', data_gpu=True, ngpu=1, expert=True)    # _gpu

    def test_symmetric_eig_float64(self):
        """Testing eigensolvers: magma_dsyevd[,x][,_gpu,_m]
        """
        self._test_syev(700, 'd', data_gpu=False, ngpu=1, expert=False)    #
        self._test_syev(700, 'd', data_gpu=False, ngpu=-1, expert=False)   # _m
        self._test_syev(700, 'd', data_gpu=True, ngpu=1, expert=False)    # _gpu
        # expert
        self._test_syev(700, 'd', data_gpu=False, ngpu=1, expert=True)    #
        self._test_syev(700, 'd', data_gpu=False, ngpu=-1, expert=True)   # _m
        self._test_syev(700, 'd', data_gpu=True, ngpu=1, expert=True)    # _gpu

def suite():
    """test suite
    """
    context = make_default_context()
    device = context.get_device()
    context.pop()

    testsuite = TestSuite()
    testsuite.addTest(test_magma('test_magma_geev_novecs'))
    testsuite.addTest(test_magma('test_symmetric_eig_float32'))
    testsuite.addTest(test_magma('test_symmetric_eig_float64'))
    return testsuite

if __name__ == '__main__':
    main(defaultTest='suite')
