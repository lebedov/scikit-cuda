#!/usr/bin/env python

"""
PyCUDA-based linear algebra functions.
"""

from __future__ import absolute_import, division

from pprint import pprint
from string import Template
from pycuda.tools import context_dependent_memoize
from pycuda.compiler import SourceModule
from pycuda.reduction import ReductionKernel

from pycuda import cumath

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.elementwise as el
import pycuda.tools as tools
import numpy as np

from . import cublas
from . import cudart
from . import misc
from . import cusolver

import sys
if sys.version_info < (3,):
    range = xrange

class LinAlgError(Exception):
    """Linear Algebra Error."""
    pass

try:
    from . import cula
    _has_cula = True
except (ImportError, OSError):
    _has_cula = False

try:
    from . import cusolver
    _has_cusolver = True
except (ImportError, OSError):
    _has_cusolver = False

from .misc import init, shutdown, add_matvec, div_matvec, mult_matvec

# Get installation location of C headers:
from . import install_headers

class PCA(object):
    """
    Principal Component Analysis with similar API to sklearn.decomposition.PCA

    The algorithm implemented here was first implemented with cuda in [Andrecut, 2008]. 
    It performs nonlinear dimensionality reduction for a data matrix, mapping the data
    to a lower dimensional space of K. See references for more information.

    Parameters
    ----------
    n_components: int, default=None
       The number of principal component column vectors to compute in the output 
       matrix.

    epsilon: float, default=1e-7
       The maximum error tolerance for eigen value approximation.

    max_iter: int, default=10000
       The maximum number of iterations in approximating each eigenvalue.

    Notes
    -----
    If n_components is None, then for a NxP data matrix `K = min(N, P)`. Otherwise, `K = min(n_components, N, P)`

    References
    ----------
    `[Andrecut, 2008] <https://arxiv.org/pdf/0811.1081.pdf>`_


    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> import skcuda.linalg as linalg
    >>> from skcuda.linalg import PCA as cuPCA 
    >>> pca = cuPCA(n_components=4) # map the data to 4 dimensions
    >>> X = np.random.rand(1000,100) # 1000 samples of 100-dimensional data vectors
    >>> X_gpu = gpuarray.GPUArray((1000,100), np.float64, order="F") # note that order="F" or a transpose is necessary. fit_transform requires row-major matrices, and column-major is the default
    >>> X_gpu.set(X) # copy data to gpu
    >>> T_gpu = pca.fit_transform(X_gpu) # calculate the principal components
    >>> linalg.dot(T_gpu[:,0], T_gpu[:,1]) # show that the resulting eigenvectors are orthogonal
    0.0
    """

    def __init__(self, n_components=None, handle=None, epsilon=1e-7, max_iter=10000):
        self.n_components = n_components
        self.epsilon = epsilon
        self.max_iter = max_iter
        misc.init()
        if handle is None:
            self.h = misc._global_cublas_handle # create a handle to initialize cublas
        else:
            self.h = handle

    def fit_transform(self, X_gpu):
        """
        Fit the Principal Component Analysis model, and return the dimension-reduced matrix.

        Compute the first K principal components of R_gpu using the
        Gram-Schmidt orthogonalization algorithm provided by [Andrecut, 2008].

        Parameters
        ----------
        R_gpu: pycuda.gpuarray.GPUArray
            NxP (N = number of samples, P = number of variables) data matrix that needs 
            to be reduced. R_gpu can be of type numpy.float32 or numpy.float64.
            Note that if R_gpu is not instantiated with the kwarg 'order="F"', 
            specifying a fortran-contiguous (row-major) array structure,
            fit_transform will throw an error.	

        Returns
        -------
        T_gpu: pycuda.gpuarray.GPUArray
            `NxK` matrix of the first K principal components of R_gpu. 

        References
        ----------
        `[Andrecut, 2008] <https://arxiv.org/pdf/0811.1081.pdf>`_

        Notes
        -----
        If n_components was not set, then `K = min(N, P)`. Otherwise, `K = min(n_components, N, P)`

        Examples
        --------
        >>> import pycuda.autoinit
        >>> import pycuda.gpuarray as gpuarray
        >>> import numpy as np
        >>> import skcuda.linalg as linalg
        >>> from skcuda.linalg import PCA as cuPCA 
        >>> pca = cuPCA(n_components=4) # map the data to 4 dimensions
        >>> X = np.random.rand(1000,100) # 1000 samples of 100-dimensional data vectors
        >>> X_gpu = gpuarray.GPUArray((1000,100), np.float64, order="F") # note that order="F" or a transpose is necessary. fit_transform requires row-major matrices, and column-major is the default
        >>> X_gpu.set(X) # copy data to gpu
        >>> T_gpu = pca.fit_transform(X_gpu) # calculate the principal components
        >>> linalg.dot(T_gpu[:,0], T_gpu[:,1]) # show that the resulting eigenvectors are orthogonal
        0.0
        """

        if len(X_gpu.shape) != 2:
            raise ValueError("Array must be 2D for PCA")

        if X_gpu.flags.c_contiguous:
            raise ValueError("Array must be fortran-contiguous. Please instantiate with 'order=\"F\"' or use the transpose of a C-ordered array.")

        R_gpu = X_gpu.copy() # copy X, because it will be altered internally otherwise
        n = R_gpu.shape[0] # num samples
        p = R_gpu.shape[1] # num features
        # choose either single or double precision cublas functions
        if R_gpu.dtype == 'float32':
            inpt_dtype = np.float32
            cuAxpy = cublas.cublasSaxpy
            cuCopy = cublas.cublasScopy
            cuGemv = cublas.cublasSgemv
            cuNrm2 = cublas.cublasSnrm2
            cuScal = cublas.cublasSscal
            cuGer = cublas.cublasSger
        elif R_gpu.dtype == 'float64':
            inpt_dtype = np.float64
            cuAxpy = cublas.cublasDaxpy
            cuCopy = cublas.cublasDcopy
            cuGemv = cublas.cublasDgemv
            cuNrm2 = cublas.cublasDnrm2
            cuScal = cublas.cublasDscal
            cuGer = cublas.cublasDger
        else:
            raise TypeError("Array must be of type numpy.float32 or numpy.float64, not '" + R_gpu.dtype + "'") 

        n_components = self.n_components
        if n_components == None or n_components > n or n_components > p:
            n_components = min(n, p)

        Lambda = np.zeros((n_components,1), inpt_dtype, order="F") # kx1
        P_gpu = gpuarray.zeros((p, n_components), inpt_dtype, order="F") # pxk
        T_gpu = gpuarray.zeros((n, n_components), inpt_dtype, order="F") # nxk

        # mean centering data
        U_gpu = gpuarray.zeros((n,1), np.float32, order="F")
        U_gpu = misc.sum(R_gpu,axis=1) # nx1 sum the columns of R
        for i in range(p):
            cuAxpy(self.h, n, -1.0/p, U_gpu.gpudata, 1, R_gpu[:,i].gpudata, 1)

        # calculate principal components
        for k in range(n_components):
            mu = 0.0
            cuCopy(self.h, n, R_gpu[:,k].gpudata, 1, T_gpu[:,k].gpudata, 1)
            for j in range(self.max_iter):
                cuGemv(self.h, 't', n, p, 1.0, R_gpu.gpudata, n, T_gpu[:,k].gpudata, 1, 0.0, P_gpu[:,k].gpudata, 1)
                if k > 0:
                    cuGemv(self.h,'t', p, k, 1.0, P_gpu.gpudata, p, P_gpu[:,k].gpudata, 1, 0.0, U_gpu.gpudata, 1)  
                    cuGemv (self.h, 'n', p, k, -1.0, P_gpu.gpudata, p, U_gpu.gpudata, 1, 1.0, P_gpu[:,k].gpudata, 1)

                l2 = cuNrm2(self.h, p, P_gpu[:,k].gpudata, 1)
                cuScal(self.h, p, 1.0/l2, P_gpu[:,k].gpudata, 1)
                cuGemv(self.h, 'n', n, p, 1.0, R_gpu.gpudata, n, P_gpu[:,k].gpudata, 1, 0.0, T_gpu[:,k].gpudata, 1)
                if k > 0:
                    cuGemv(self.h, 't', n, k, 1.0, T_gpu.gpudata, n, T_gpu[:,k].gpudata, 1, 0.0, U_gpu.gpudata, 1)
                    cuGemv(self.h, 'n', n, k, -1.0, T_gpu.gpudata, n, U_gpu.gpudata, 1, 1.0, T_gpu[:,k].gpudata, 1)

                Lambda[k] = cuNrm2(self.h, n, T_gpu[:,k].gpudata, 1)
                cuScal(self.h, n, 1.0/Lambda[k], T_gpu[:,k].gpudata, 1)
                if abs(Lambda[k] - mu) < self.epsilon*Lambda[k]:
                    break

                mu = Lambda[k]
            # end for j
        cuGer(self.h, n, p, (0.0-Lambda[k]), T_gpu[:,k].gpudata, 1, P_gpu[:,k].gpudata, 1, R_gpu.gpudata, n)
        # end for k

        # last step is to multiply each component vector by the corresponding eigenvalue
        for k in range(n_components):
            cuScal(self.h, n, Lambda[k], T_gpu[:,k].gpudata, 1) 

        # free gpu memory
        P_gpu.gpudata.free()
        U_gpu.gpudata.free()

        return T_gpu # return the gpu array of principal component scores


    def set_n_components(self, n_components):
        """
        n_components setter.

        Parameters
        ----------

        n_components: int
            The new number of principal components to return in fit_transform. 
            Must be None or greater than 0
        """

        if n_components > 0 or n_components == None:
            self.n_components = n_components
        else:
            raise ValueError("n_components can only be greater than 0 or None")

    def get_n_components(self):
        """
        n_components getter.


        Returns
        -------
        n_components: int
            The current value of self.n_components
        """

        return self.n_components

def svd(a_gpu, jobu='A', jobvt='A', lib='cusolver'):
    """
    Singular Value Decomposition.

    Factors the matrix `a` into two unitary matrices, `u` and `vh`,
    and a 1-dimensional array of real, non-negative singular values,
    `s`, such that `a == dot(u.T, dot(diag(s), vh.T))`.

    Parameters
    ----------
    a : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, n)` to decompose.
    jobu : {'A', 'S', 'O', 'N'}
        If 'A', return the full `u` matrix with shape `(m, m)`.
        If 'S', return the `u` matrix with shape `(m, k)`.
        If 'O', return the `u` matrix with shape `(m, k) without
        allocating a new matrix.
        If 'N', don't return `u`.
    jobvt : {'A', 'S', 'O', 'N'}
        If 'A', return the full `vh` matrix with shape `(n, n)`.
        If 'S', return the `vh` matrix with shape `(k, n)`.
        If 'O', return the `vh` matrix with shape `(k, n) without
        allocating a new matrix.
        If 'N', don't return `vh`.
    lib : str
        Library to use. May be either 'cula' or 'cusolver'.

    Returns
    -------
    u : pycuda.gpuarray.GPUArray
        Unitary matrix of shape `(m, m)` or `(m, k)` depending on
        value of `jobu`.
    s : pycuda.gpuarray.GPUArray
        Array containing the singular values, sorted such that `s[i] >= s[i+1]`.
        `s` is of length `min(m, n)`.
    vh : pycuda.gpuarray.GPUArray
        Unitary matrix of shape `(n, n)` or `(k, n)`, depending
        on `jobvt`.

    Notes
    -----
    If using CULA, double precision is only supported if the standard version of the
    CULA Dense toolkit is installed.

    This function destroys the contents of the input matrix regardless
    of the values of `jobu` and `jobvt`.

    Only one of `jobu` or `jobvt` may be set to `O`, and then only for
    a square matrix.

    The CUSOLVER library in CUDA 7.0 only supports `jobu` == `jobvt` == 'A'.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> import skcuda.linalg as linalg
    >>> linalg.init()
    >>> a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
    >>> a = np.asarray(a, np.complex64)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 'S', 'S')
    >>> np.allclose(a, np.dot(u_gpu.get(), np.dot(np.diag(s_gpu.get()), vh_gpu.get())), 1e-4)
    True
    """

    alloc = misc._global_cublas_allocator

    # The free version of CULA only supports single precision floating
    # point numbers:
    data_type = a_gpu.dtype.type
    real_type = np.float32

    if lib == 'cula':
        if not _has_cula:
            raise NotImplementedError('CULA not installed')

        if data_type == np.complex64:
            func = cula.culaDeviceCgesvd
        elif data_type == np.float32:
            func = cula.culaDeviceSgesvd
        else:
            if cula._libcula_toolkit == 'standard':
                if data_type == np.complex128:
                    func = cula.culaDeviceZgesvd
                elif data_type == np.float64:
                    func = cula.culaDeviceDgesvd
                else:
                    raise ValueError('unsupported type')
                real_type = np.float64
            else:
                raise ValueError('double precision not supported')
    elif lib == 'cusolver':
        if not _has_cusolver:
            raise NotImplementedError('CUSOLVER not installed')

        cusolverHandle = misc._global_cusolver_handle

        if data_type == np.complex64:
            func = cusolver.cusolverDnCgesvd
            bufsize = cusolver.cusolverDnCgesvd_bufferSize
        elif data_type == np.float32:
            func = cusolver.cusolverDnSgesvd
            bufsize = cusolver.cusolverDnSgesvd_bufferSize
        elif data_type == np.complex128:
            real_type = np.float64
            func = cusolver.cusolverDnZgesvd
            bufsize = cusolver.cusolverDnZgesvd_bufferSize
        elif data_type == np.float64:
            real_type = np.float64
            func = cusolver.cusolverDnDgesvd
            bufsize = cusolver.cusolverDnDgesvd_bufferSize
        else:
            raise ValueError('unsupported type')
    else:
        raise ValueError('invalid library specified')

    # Since CUDA assumes that arrays are stored in column-major
    # format, the input matrix is assumed to be transposed:
    n, m = a_gpu.shape
    square = (n == m)

    # CUSOLVER's gesvd routines only support m >= n as of CUDA 7.5:
    if lib == 'cusolver' and m < n:
        raise ValueError('CUSOLVER only supports a_gpu.shape[1] >= a_gpu.shape[0]')

    # Since the input matrix is transposed, jobu and jobvt must also
    # be switched because the computed matrices will be returned in
    # reversed order:
    jobvt, jobu = jobu, jobvt

    # Set the leading dimension of the input matrix:
    lda = max(1, m)

    # Allocate the array of singular values:
    s_gpu = gpuarray.empty(min(m, n), real_type, allocator=alloc)

    # CUSOLVER in CUDA 7.0 only supports jobu = jobvt = 'A':
    jobu = jobu.upper()
    jobvt = jobvt.upper()
    if lib == 'cusolver' and (jobu != 'A' or jobvt != 'A') and \
      cudart._cudart_version <= 7000:
         raise ValueError("CUSOLVER 7.0 only supports jobu = jobvt = 'A'")

    # Set the leading dimension and allocate u:
    ldu = m
    if jobu == 'A':
        u_gpu = gpuarray.empty((ldu, m), data_type, allocator=alloc)
    elif jobu == 'S':
        u_gpu = gpuarray.empty((min(m, n), ldu), data_type, allocator=alloc)
    elif jobu == 'O':
        if not square:
            raise ValueError('in-place computation of singular vectors '+
                             'of non-square matrix not allowed')
        ldu = a_gpu.shape[1]
        u_gpu = a_gpu
    else:
        ldu = 1
        u_gpu = gpuarray.empty((), data_type, allocator=alloc)

    # Set the leading dimension and allocate vh:
    if jobvt == 'A':
        ldvt = n
        vh_gpu = gpuarray.empty((n, n), data_type, allocator=alloc)
    elif jobvt == 'S':
        ldvt = min(m, n)
        vh_gpu = gpuarray.empty((n, ldvt), data_type, allocator=alloc)
    elif jobvt == 'O':
        if jobu == 'O':
            raise ValueError('jobu and jobvt cannot both be O')
        if not square:
            raise ValueError('in-place computation of singular vectors '+
                             'of non-square matrix not allowed')
        ldvt = a_gpu.shape[1]
        vh_gpu = a_gpu
    else:
        ldvt = 1
        vh_gpu = gpuarray.empty((), data_type, allocator=alloc)

    # Compute SVD and check error status:
    if lib == 'cula':
        func(jobu, jobvt, m, n, int(a_gpu.gpudata),
             lda, int(s_gpu.gpudata), int(u_gpu.gpudata),
             ldu, int(vh_gpu.gpudata), ldvt)

        # Free internal CULA memory:
        cula.culaFreeBuffers()
    else:
        # Allocate working space:
        Lwork = bufsize(misc._global_cusolver_handle, m, n)

        Work = gpuarray.empty(Lwork, data_type, allocator=alloc)
        devInfo = gpuarray.empty(1, np.int32, allocator=alloc)

        # rwork is only needed for complex arrays:
        if data_type != real_type:
            rwork = np.empty(Lwork, real_type).ctypes.data
        else:
            rwork = 0
        func(misc._global_cusolver_handle,
             jobu, jobvt, m, n, int(a_gpu.gpudata),
             lda, int(s_gpu.gpudata), int(u_gpu.gpudata),
             ldu, int(vh_gpu.gpudata), ldvt,
             int(Work.gpudata), Lwork, rwork,
             int(devInfo.gpudata))

        # Free working space:
        del rwork, Work, devInfo

    # Since the input is assumed to be transposed, it is necessary to
    # return the computed matrices in reverse order:
    if jobu in ['A', 'S', 'O'] and jobvt in ['A', 'S', 'O']:
        return vh_gpu, s_gpu, u_gpu
    elif jobu == 'N' and jobvt != 'N':
        return vh_gpu, s_gpu
    elif jobu != 'N' and jobvt == 'N':
        return s_gpu, u_gpu
    else:
        return s_gpu

def cho_factor(a_gpu, uplo='L', lib='cusolver'):
    """
    Cholesky factorization.

    Performs an in-place Cholesky factorization on the matrix `a`
    such that `a = x*x.T` or `x.T*x`, if the lower='L' or upper='U'
    triangle of `a` is used, respectively.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, m)` to decompose.
    uplo : {'U', 'L'}
        Use upper or lower (default) triangle of 'a_gpu'
    lib : str
        Library to use. May be either 'cula' or 'cusolver'.

    Notes
    -----
    If using CULA, double precision is only supported if the standard version of the
    CULA Dense toolkit is installed.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import scipy.linalg
    >>> import skcuda.linalg as linalg
    >>> linalg.init()
    >>> a = np.array([[3.0,0.0],[0.0,7.0]])
    >>> a = np.asarray(a, np.float64)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> cho_factor(a_gpu)
    >>> np.allclose(a_gpu.get(), scipy.linalg.cho_factor(a)[0])
    True
    """

    alloc = misc._global_cublas_allocator

    data_type = a_gpu.dtype.type

    if lib == 'cula':
        if not _has_cula:
            raise NotImplementedError('CULA not installed')

        real_type = np.float32
        if cula._libcula_toolkit == 'standard':
            if data_type == np.complex64:
                func = cula.culaDeviceCpotrf
            elif data_type == np.float32:
                func = cula.culaDeviceSpotrf
            elif data_type == np.complex128:
                func = cula.culaDeviceZpotrf
            elif data_type == np.float64:
                func = cula.culaDeviceDpotrf
            else:
                raise ValueError('unsupported type')
            real_type = np.float64
        else:
            raise ValueError('Cholesky factorization not included in CULA Dense Free version')

    elif lib == 'cusolver':
        if not _has_cusolver:
            raise NotImplementedError('CUSOLVER not installed')

        cusolverHandle = misc._global_cusolver_handle

        if data_type == np.complex64:
            func = cusolver.cusolverDnCpotrf
            bufsize = cusolver.cusolverDnCpotrf_bufferSize
        elif data_type == np.float32:
            func = cusolver.cusolverDnSpotrf
            bufsize = cusolver.cusolverDnSpotrf_bufferSize
        elif data_type == np.complex128:
            real_type = np.float64
            func = cusolver.cusolverDnZpotrf
            bufsize = cusolver.cusolverDnZpotrf_bufferSize
        elif data_type == np.float64:
            real_type = np.float64
            func = cusolver.cusolverDnDpotrf
            bufsize = cusolver.cusolverDnDpotrf_bufferSize
        else:
            raise ValueError('unsupported type')

    else:
        raise ValueError('invalid library specified')

    # Since CUDA assumes that arrays are stored in column-major
    # format, the input matrix is assumed to be transposed:
    n, m = a_gpu.shape
    if (n!=m):
        raise ValueError('Matrix must be symmetric positive-definite')

    # Set the leading dimension of the input matrix:
    lda = max(1, m)

    # Factorize and check error status:
    if lib == 'cula':
        func(uplo, n, int(a_gpu.gpudata), lda)

        # Free internal CULA memory:
        cula.culaFreeBuffers()
    else:
        # CUSOLVER expects uplo to be an int rather than a char:
        uplo = cublas._CUBLAS_FILL_MODE[uplo]

        # Allocate working space:
        Lwork = bufsize(misc._global_cusolver_handle, uplo, n, int(a_gpu.gpudata), lda)
        Work = gpuarray.empty(Lwork, data_type, allocator=alloc)
        devInfo = gpuarray.empty(1, np.int32, allocator=alloc)

        func(misc._global_cusolver_handle, uplo, n, int(a_gpu.gpudata), lda,
             int(Work.gpudata), Lwork, int(devInfo.gpudata))

        # Free working space:
        del Work, devInfo

    # In-place operation. No return matrix. Result is stored in the input matrix.

def cholesky(a_gpu, uplo='L', lib='cusolver'):
    """
    Cholesky factorization.

    Performs an in-place Cholesky factorization on the matrix `a`
    such that `a = x*x.T` or `x.T*x`, if the lower='L' or upper='U'
    triangle of `a` is used, respectively. All other entries in `a` are set to 0.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, m)` to decompose.
    uplo : {'U', 'L'}
        Use upper or lower (default) triangle of 'a_gpu'
    lib : str
        Library to use. May be either 'cula' or 'cusolver'.

    Notes
    -----
    If using CULA, double precision is only supported if the standard version of the
    CULA Dense toolkit is installed.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import scipy.linalg
    >>> import skcuda.linalg as linalg
    >>> linalg.init()
    >>> a = np.array([[3.0,0.0],[0.0,7.0]])
    >>> a = np.asarray(a, np.float64)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> cholesky(a_gpu)
    >>> np.allclose(a_gpu.get(), scipy.linalg.cholesky(a)[0])
    True
    """

    if a_gpu.dtype == np.float32:
        use_double = 0
        use_complex = 0
    elif a_gpu.dtype == np.float64:
        use_double = 1
        use_complex = 0
    elif a_gpu.dtype == np.complex64:
        use_double = 0
        use_complex = 1
    elif a_gpu.dtype == np.complex128:
        use_double = 1
        use_complex = 1
    else:
        raise ValueError('unrecognized type')

    cho_factor(a_gpu, uplo, lib)

    N = a_gpu.shape[0]
    dev = misc.get_current_device()
    block_dim, grid_dim = misc.select_block_grid_sizes(dev, a_gpu.shape)

    # Zero out the opposite triangle of the matrix
    if cublas._CUBLAS_FILL_MODE[uplo] == 0: # 0 == L
        func = _get_triu_kernel(use_double, use_complex, cols=N)
    else:
        func = _get_tril_kernel(use_double, use_complex, cols=N)

    func(a_gpu, np.uint32(a_gpu.size),
         block=block_dim,
         grid=grid_dim)

def cho_solve(a_gpu, b_gpu, uplo='L', lib='cusolver'):
    """
    Cholesky solver.

    Solve a system of equations via Cholesky factorization,
    i.e. `a*x = b`.
    Overwrites `b` to give `inv(a)*b`, and overwrites the chosen triangle
    of `a` with factorized triangle.

    Parameters
    ----------
    a : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, m)` to decompose.
    b : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, 1)` to decompose.
    uplo: chr
        Use the upper='U' or lower='L' (default) triangle of `a`.
    lib : str
        Library to use. May be either 'cula' or 'cusolver'.

    Notes
    -----
    If using CULA, double precision is only supported if the standard version of the
    CULA Dense toolkit is installed.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import scipy.linalg
    >>> import skcuda.linalg as linalg
    >>> linalg.init()
    >>> a = np.array([[3, 0], [0, 7]]).asarray(np.float64)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> b = np.array([11, 19]).astype(np.float64)
    >>> b_gpu  = gpuarray.to_gpu(b)
    >>> cho_solve(a_gpu, b_gpu)
    >>> np.allclose(b_gpu.get(), scipy.linalg.cho_solve(scipy.linalg.cho_factor(a), b))
    True
    """

    alloc = misc._global_cublas_allocator

    data_type = a_gpu.dtype.type
    if lib == 'cula':
        if not _has_cula:
            raise NotImplementedError('CULA not installed')

        if cula._libcula_toolkit == 'standard':
            if data_type == np.complex64:
                func = cula.culaDeviceCposv
            elif data_type == np.float32:
                func = cula.culaDeviceSposv
            elif data_type == np.complex128:
                func = cula.culaDeviceZposv
            elif data_type == np.float64:
                func = cula.culaDeviceDposv
            else:
                raise ValueError('unsupported type')
        else:
            raise ValueError('Cholesky factorization not included in CULA Dense Free version')

    elif lib == 'cusolver':
        if not _has_cusolver:
            raise NotImplementedError('CUSOLVER not installed')
        cusolverHandle = misc._global_cusolver_handle

        if data_type == np.complex64:
            func = cusolver.cusolverDnCpotrs
        elif data_type == np.float32:
            func = cusolver.cusolverDnSpotrs
        elif data_type == np.complex128:
            func = cusolver.cusolverDnZpotrs
        elif data_type == np.float64:
            func = cusolver.cusolverDnDpotrs
        else:
            raise ValueError('unsupported type')

    else:
        raise ValueError('invalid library specified')

    # Since CUDA assumes that arrays are stored in column-major
    # format, the input matrix is assumed to be transposed:
    na, ma = a_gpu.shape

    if (na!=ma):
        raise ValueError('Matrix must be symmetric positive-definite')

    if a_gpu.flags.c_contiguous != b_gpu.flags.c_contiguous:
        raise ValueError('unsupported combination of input order')

    b_shape = b_gpu.shape
    if len(b_shape) == 1:
        b_shape = (b_shape[0], 1)

    if a_gpu.flags.f_contiguous:
        lda = max(1, na)
        ldb = max(1, b_shape[0])
    else:
        lda = max(1, ma)
        ldb = lda
        if b_shape[1] > 1:
            raise ValueError('only vectors allowed in c-order RHS')

    if lib == 'cula':
        # Assuming we are only solving for a vector. Hence, nrhs = 1
        func(uplo, na, b_shape[1], int(a_gpu.gpudata), lda,
             int(b_gpu.gpudata), ldb)

        # Free internal CULA memory:
        cula.culaFreeBuffers()
    else:
        # CUSOLVER expects uplo to be an int rather than a char:
        uplo = cublas._CUBLAS_FILL_MODE[uplo]

        # Since CUSOLVER doesn't implement POSV as of 8.0, we need to factor the
        # given matrix before calling POTRS:
        cho_factor(a_gpu, uplo, lib)

        # Assuming we are only solving for a vector. Hence, nrhs = 1
        devInfo = gpuarray.empty(1, np.int32, allocator=alloc)
        func(cusolverHandle, uplo, na, b_shape[1], int(a_gpu.gpudata), lda,
             int(b_gpu.gpudata), ldb, int(devInfo.gpudata))

    # In-place operation. No return matrix. Result is stored in the input matrix
    # and in the input vector.

def add_dot(a_gpu, b_gpu, c_gpu, transa='N', transb='N', alpha=1.0, beta=1.0, handle=None):
    """
    Calculates the dot product of two arrays and adds it to a third matrix.

    In essence, this computes

    C =  alpha * (A B) + beta * C

    For 2D arrays of shapes `(m, k)` and `(k, n)`, it computes the matrix
    product; the result has shape `(m, n)`.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input array.
    b_gpu : pycuda.gpuarray.GPUArray
        Input array.
    c_gpu : pycuda.gpuarray.GPUArray
        Cumulative array.
    transa : char
        If 'T', compute the product of the transpose of `a_gpu`.
        If 'C', compute the product of the Hermitian of `a_gpu`.
    transb : char
        If 'T', compute the product of the transpose of `b_gpu`.
        If 'C', compute the product of the Hermitian of `b_gpu`.
    handle : int (optional)
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.

    Returns
    -------
    c_gpu : pycuda.gpuarray.GPUArray

    Notes
    -----
    The matrices must all contain elements of the same data type.
    """

    if handle is None:
        handle = misc._global_cublas_handle

    # Get the shapes of the arguments (accounting for the
    # possibility that one of them may only have one dimension):
    a_shape = a_gpu.shape
    b_shape = b_gpu.shape
    if len(a_shape) == 1:
        a_shape = (1, a_shape[0])
    if len(b_shape) == 1:
        b_shape = (1, b_shape[0])

    # Perform matrix multiplication for 2D arrays:
    if (a_gpu.dtype == np.complex64 and b_gpu.dtype == np.complex64):
        cublas_func = cublas.cublasCgemm
        alpha = np.complex64(alpha)
        beta = np.complex64(beta)
    elif (a_gpu.dtype == np.float32 and b_gpu.dtype == np.float32):
        cublas_func = cublas.cublasSgemm
        alpha = np.float32(alpha)
        beta = np.float32(beta)
    elif (a_gpu.dtype == np.complex128 and b_gpu.dtype == np.complex128):
        cublas_func = cublas.cublasZgemm
        alpha = np.complex128(alpha)
        beta = np.complex128(beta)
    elif (a_gpu.dtype == np.float64 and b_gpu.dtype == np.float64):
        cublas_func = cublas.cublasDgemm
        alpha = np.float64(alpha)
        beta = np.float64(beta)
    else:
        raise ValueError('unsupported combination of input types')

    transa = transa.lower()
    transb = transb.lower()

    a_f_order = a_gpu.strides[1] > a_gpu.strides[0]
    b_f_order = b_gpu.strides[1] > b_gpu.strides[0]
    c_f_order = c_gpu.strides[1] > c_gpu.strides[0]

    if a_f_order != b_f_order:
        raise ValueError('unsupported combination of input order')
    if a_f_order != c_f_order:
        raise ValueError('invalid order for c_gpu')

    if a_f_order:  # F order array
        if transa in ['t', 'c']:
            k, m = a_shape
        elif transa in ['n']:
            m, k = a_shape
        else:
            raise ValueError('invalid value for transa')

        if transb in ['t', 'c']:
            n, l = b_shape
        elif transb in ['n']:
            l, n = b_shape
        else:
            raise ValueError('invalid value for transb')

        if l != k:
            raise ValueError('objects are not aligned')

        lda = max(1, a_gpu.strides[1] // a_gpu.dtype.itemsize)
        ldb = max(1, b_gpu.strides[1] // b_gpu.dtype.itemsize)
        ldc = max(1, c_gpu.strides[1] // c_gpu.dtype.itemsize)

        if c_gpu.shape != (m, n) or c_gpu.dtype != a_gpu.dtype:
            raise ValueError('invalid value for c_gpu')
        cublas_func(handle, transa, transb, m, n, k, alpha, a_gpu.gpudata,
                lda, b_gpu.gpudata, ldb, beta, c_gpu.gpudata, ldc)
    else:
        if transb in ['t', 'c']:
            m, k = b_shape
        elif transb in ['n']:
            k, m = b_shape
        else:
            raise ValueError('invalid value for transb')

        if transa in ['t', 'c']:
            l, n = a_shape
        elif transa in ['n']:
            n, l = a_shape
        else:
            raise ValueError('invalid value for transa')

        if l != k:
            raise ValueError('objects are not aligned')

        lda = max(1, a_gpu.strides[0] // a_gpu.dtype.itemsize)
        ldb = max(1, b_gpu.strides[0] // b_gpu.dtype.itemsize)
        ldc = max(1, c_gpu.strides[0] // c_gpu.dtype.itemsize)

        # Note that the desired shape of the output matrix is the transpose
        # of what CUBLAS assumes:
        if c_gpu.shape != (n, m) or c_gpu.dtype != a_gpu.dtype:
            raise ValueError('invalid value for c_gpu')
        cublas_func(handle, transb, transa, m, n, k, alpha, b_gpu.gpudata,
                ldb, a_gpu.gpudata, lda, beta, c_gpu.gpudata, ldc)
    return c_gpu


def dot(x_gpu, y_gpu, transa='N', transb='N', handle=None, out=None):
    """
    Dot product of two arrays.

    For 1D arrays, this function computes the inner product. For 2D
    arrays of shapes `(m, k)` and `(k, n)`, it computes the matrix
    product; the result has shape `(m, n)`.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Input array.
    y_gpu : pycuda.gpuarray.GPUArray
        Input array.
    transa : char
        If 'T', compute the product of the transpose of `x_gpu`.
        If 'C', compute the product of the Hermitian of `x_gpu`.
    transb : char
        If 'T', compute the product of the transpose of `y_gpu`.
        If 'C', compute the product of the Hermitian of `y_gpu`.
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.
    out : pycuda.gpuarray.GPUArray, optional
        Output argument. Will be used to store the result.

    Returns
    -------
    c_gpu : pycuda.gpuarray.GPUArray, float{32,64}, or complex{64,128}
        Inner product of `x_gpu` and `y_gpu`. When the inputs are 1D
        arrays, the result will be returned as a scalar.

    Notes
    -----
    The input matrices must all contain elements of the same data type.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> import skcuda.linalg as linalg
    >>> import skcuda.misc as misc
    >>> linalg.init()
    >>> a = np.asarray(np.random.rand(4, 2), np.float32)
    >>> b = np.asarray(np.random.rand(2, 2), np.float32)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> b_gpu = gpuarray.to_gpu(b)
    >>> c_gpu = linalg.dot(a_gpu, b_gpu)
    >>> np.allclose(np.dot(a, b), c_gpu.get())
    True
    >>> d = np.asarray(np.random.rand(5), np.float32)
    >>> e = np.asarray(np.random.rand(5), np.float32)
    >>> d_gpu = gpuarray.to_gpu(d)
    >>> e_gpu = gpuarray.to_gpu(e)
    >>> f = linalg.dot(d_gpu, e_gpu)
    >>> np.allclose(np.dot(d, e), f)
    True

    """
    if handle is None:
        handle = misc._global_cublas_handle

    x_shape = x_gpu.shape
    y_shape = y_gpu.shape

    # When one argument is a vector and the other a matrix, increase the number
    # of dimensions of the vector to 2 so that they can be multiplied using
    # GEMM, but also set the shape of the output to 1 dimension to conform with
    # the behavior of numpy.dot:
    if len(x_shape) == 1 and len(y_shape) > 1:
        out_shape = (y_shape[1],)
        x_shape = (1, x_shape[0])
        x_gpu = x_gpu.reshape(x_shape)
    elif len(x_shape) > 1 and len(y_shape) == 1:
        out_shape = (x_shape[0],)
        y_shape = (y_shape[0], 1)
        y_gpu = y_gpu.reshape(y_shape)

    if len(x_gpu.shape) == 1 and len(y_gpu.shape) == 1:
        if x_gpu.size != y_gpu.size:
            raise ValueError('arrays must be of same length')

        # Compute inner product for 1D arrays:
        if (x_gpu.dtype == np.complex64 and y_gpu.dtype == np.complex64):
            cublas_func = cublas.cublasCdotu
        elif (x_gpu.dtype == np.float32 and y_gpu.dtype == np.float32):
            cublas_func = cublas.cublasSdot
        elif (x_gpu.dtype == np.complex128 and y_gpu.dtype == np.complex128):
            cublas_func = cublas.cublasZdotu
        elif (x_gpu.dtype == np.float64 and y_gpu.dtype == np.float64):
            cublas_func = cublas.cublasDdot
        else:
            raise ValueError('unsupported combination of input types')

        return cublas_func(handle, x_gpu.size, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
    else:
        transa = transa.lower()
        transb = transb.lower()
        if out is None:
            if transa in ['t', 'c']:
                k, m = x_shape
            else:
                m, k = x_shape

            if transb in ['t', 'c']:
                n, l = y_shape
            else:
                l, n = y_shape

            alloc = misc._global_cublas_allocator
            if x_gpu.strides[1] > x_gpu.strides[0]: # F order
                out = gpuarray.empty((m, n), x_gpu.dtype, order="F", allocator=alloc)
            else:
                out = gpuarray.empty((m, n), x_gpu.dtype, order="C", allocator=alloc)

    add_dot(x_gpu, y_gpu, out, transa, transb, 1.0, 0.0, handle)
    if 'out_shape' in locals():
        return out.reshape(out_shape)
    else:
        return out

def mdot(*args, **kwargs):
    """
    Product of several matrices.

    Computes the matrix product of several arrays of shapes.

    Parameters
    ----------
    a_gpu, b_gpu, ... : pycuda.gpuarray.GPUArray
        Arrays to multiply.
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.

    Returns
    -------
    c_gpu : pycuda.gpuarray.GPUArray
        Matrix product of `a_gpu`, `b_gpu`, etc.

    Notes
    -----
    The input matrices must all contain elements of the same data type.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import skcuda.linalg as linalg
    >>> linalg.init()
    >>> a = np.asarray(np.random.rand(4, 2), np.float32)
    >>> b = np.asarray(np.random.rand(2, 2), np.float32)
    >>> c = np.asarray(np.random.rand(2, 2), np.float32)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> b_gpu = gpuarray.to_gpu(b)
    >>> c_gpu = gpuarray.to_gpu(c)
    >>> d_gpu = linalg.mdot(a_gpu, b_gpu, c_gpu)
    >>> np.allclose(np.dot(a, np.dot(b, c)), d_gpu.get())
    True

    """

    if ' handle' in kwargs and kwargs['handle'] is not None:
        handle = kwargs['handle']
    else:
        handle = misc._global_cublas_handle

    # Free the temporary matrix allocated when computing the dot
    # product:
    out_gpu = args[0]
    for next_gpu in args[1:]:
        temp_gpu = dot(out_gpu, next_gpu, handle=handle)
        out_gpu.gpudata.free()
        del(out_gpu)
        out_gpu = temp_gpu
        del(temp_gpu)
    return out_gpu

def dot_diag(d_gpu, a_gpu, trans='N', overwrite=False, handle=None):
    """
    Dot product of diagonal and non-diagonal arrays.

    Computes the matrix product of a diagonal array represented as a
    vector and a non-diagonal array.

    Parameters
    ----------
    d_gpu : pycuda.gpuarray.GPUArray
        Array of length `N` corresponding to the diagonal of the
        multiplier.
    a_gpu : pycuda.gpuarray.GPUArray
        Multiplicand array with shape `(N, M)`. Must have same data type
        as `d_gpu`.
    trans : char
        If 'T', compute the product of the transpose of `a_gpu`.
    overwrite : bool (default: False)
        If true, save the result in `a_gpu`.
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.

    Returns
    -------
    r_gpu : pycuda.gpuarray.GPUArray
        The computed matrix product.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> import skcuda.linalg as linalg
    >>> linalg.init()
    >>> d = np.random.rand(4)
    >>> a = np.random.rand(4, 4)
    >>> d_gpu = gpuarray.to_gpu(d)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> r_gpu = linalg.dot_diag(d_gpu, a_gpu)
    >>> np.allclose(np.dot(np.diag(d), a), r_gpu.get())
    True
    """

    if handle is None:
        handle = misc._global_cublas_handle

    if not (len(d_gpu.shape) == 1 or (d_gpu.shape[0] == 1 or d_gpu.shape[1] == 1)):
        raise ValueError('d_gpu must be a vector')
    if len(a_gpu.shape) != 2:
        raise ValueError('a_gpu must be a matrix')

    trans = trans.lower()
    if trans == 'n':
        rows, cols = a_gpu.shape
    else:
        cols, rows = a_gpu.shape

    N = d_gpu.size
    if N != rows:
        raise ValueError('incompatible dimensions')

    if a_gpu.dtype != d_gpu.dtype:
        raise ValueError('argument types must be the same')

    if (a_gpu.dtype == np.complex64):
        cublas_func = cublas.cublasCdgmm
    elif (a_gpu.dtype == np.float32):
        cublas_func = cublas.cublasSdgmm
    elif (a_gpu.dtype == np.complex128):
        cublas_func = cublas.cublasZdgmm
    elif (a_gpu.dtype == np.float64):
        cublas_func = cublas.cublasDdgmm
    else:
        raise ValueError('unsupported input type')

    if overwrite:
        r_gpu = a_gpu
    else:
        r_gpu = a_gpu.copy()

    if (trans == 'n' and a_gpu.flags.c_contiguous) \
        or (trans == 't' and a_gpu.flags.f_contiguous):
        side = "R"
    else:
        side = "L"

    lda = a_gpu.shape[1] if a_gpu.flags.c_contiguous else a_gpu.shape[0]
    ldr = lda

    n, m = a_gpu.shape if a_gpu.flags.f_contiguous else (a_gpu.shape[1], a_gpu.shape[0])
    cublas_func(handle, side, n, m, a_gpu.gpudata, lda,
                d_gpu.gpudata, 1, r_gpu.gpudata, ldr)
    return r_gpu

def add_diag(d_gpu, a_gpu, overwrite=False, handle=None):
    """
    Adds a vector to the diagonal of an array.

    This is the same as A + diag(D), but faster.

    Parameters
    ----------
    d_gpu : pycuda.gpuarray.GPUArray
        Array of length `N` corresponding to the vector to be added to the
        diagonal.
    a_gpu : pycuda.gpuarray.GPUArray
        Summand array with shape `(N, N)`.
    overwrite : bool (default: False)
        If true, save the result in `a_gpu`.
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.

    Returns
    -------
    r_gpu : pycuda.gpuarray.GPUArray
        The computed sum product.

    Notes
    -----
    `d_gpu` and `a_gpu` must have the same precision data type.
    """

    if handle is None:
        handle = misc._global_cublas_handle

    if not (len(d_gpu.shape) == 1 or (d_gpu.shape[0] == 1 or d_gpu.shape[1] == 1)):
        raise ValueError('d_gpu must be a vector')
    if len(a_gpu.shape) != 2:
        raise ValueError('a_gpu must be a matrix')
    if a_gpu.shape[0] != a_gpu.shape[1]:
        raise ValueError('a_gpu must be square')

    if d_gpu.size != a_gpu.shape[0]:
        raise ValueError('incompatible dimensions')

    if a_gpu.dtype != d_gpu.dtype:
        raise ValueError('precision of argument types must be the same')

    if (a_gpu.dtype == np.complex64):
        axpy = cublas.cublasCaxpy
    elif (a_gpu.dtype == np.float32):
        axpy = cublas.cublasSaxpy
    elif (a_gpu.dtype == np.complex128):
        axpy = cublas.cublasZaxpy
    elif (a_gpu.dtype == np.float64):
        axpy = cublas.cublasDaxpy
    else:
        raise ValueError('unsupported input type')

    if overwrite:
        r_gpu = a_gpu
    else:
        r_gpu = a_gpu.copy()

    n = a_gpu.shape[0]
    axpy(handle, n, 1.0, d_gpu.gpudata, int(1), r_gpu.gpudata, int(n+1))
    return r_gpu

def _transpose(a_gpu, conj=False, handle=None):
    if handle is None:
        handle = misc._global_cublas_handle

    if len(a_gpu.shape) != 2:
        raise ValueError('a_gpu must be a matrix')

    if (a_gpu.dtype == np.complex64):
        func = cublas.cublasCgeam
    elif (a_gpu.dtype == np.float32):
        func = cublas.cublasSgeam
    elif (a_gpu.dtype == np.complex128):
        func = cublas.cublasZgeam
    elif (a_gpu.dtype == np.float64):
        func = cublas.cublasDgeam
    else:
        raise ValueError('unsupported input type')

    if conj:
        transa = 'c'
    else:
        transa = 't'
    M, N = a_gpu.shape
    at_gpu = gpuarray.empty((N, M), a_gpu.dtype)
    func(handle, transa, 't', M, N,
         1.0, a_gpu.gpudata, N, 0.0, a_gpu.gpudata, N,
         at_gpu.gpudata, M)
    return at_gpu

def transpose(a_gpu, handle=None):
    """
    Matrix transpose.

    Transpose a matrix in device memory and return an object
    representing the transposed matrix.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, n)`.

    Returns
    -------
    at_gpu : pycuda.gpuarray.GPUArray
        Transposed matrix of shape `(n, m)`.
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.driver as drv
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> import skcuda.linalg as linalg
    >>> linalg.init()
    >>> a = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], np.float32)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> at_gpu = linalg.transpose(a_gpu)
    >>> np.all(a.T == at_gpu.get())
    True
    >>> b = np.array([[1j, 2j, 3j, 4j, 5j, 6j], [7j, 8j, 9j, 10j, 11j, 12j]], np.complex64)
    >>> b_gpu = gpuarray.to_gpu(b)
    >>> bt_gpu = linalg.transpose(b_gpu)
    >>> np.all(b.T == bt_gpu.get())
    True
    """

    return _transpose(a_gpu, False, handle)

def hermitian(a_gpu, handle=None):
    """
    Hermitian (conjugate) matrix transpose.

    Conjugate transpose a matrix in device memory and return an object
    representing the transposed matrix.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, n)`.
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.

    Returns
    -------
    at_gpu : pycuda.gpuarray.GPUArray
        Transposed matrix of shape `(n, m)`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.driver as drv
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> import skcuda.linalg as linalg
    >>> linalg.init()
    >>> a = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], np.float32)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> at_gpu = linalg.hermitian(a_gpu)
    >>> np.all(a.T == at_gpu.get())
    True
    >>> b = np.array([[1j, 2j, 3j, 4j, 5j, 6j], [7j, 8j, 9j, 10j, 11j, 12j]], np.complex64)
    >>> b_gpu = gpuarray.to_gpu(b)
    >>> bt_gpu = linalg.hermitian(b_gpu)
    >>> np.all(np.conj(b.T) == bt_gpu.get())
    True
    """

    return _transpose(a_gpu, True, handle)

@context_dependent_memoize
def _get_conj_kernel(dtype):
    ctype = tools.dtype_to_ctype(dtype)
    return el.ElementwiseKernel(
                "{ctype} *x, {ctype} *y".format(ctype=ctype),
                "y[i] = conj(x[i])")

def conj(x_gpu, overwrite=False):
    """
    Complex conjugate.

    Compute the complex conjugate of the array in device memory.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Input array of shape `(m, n)`.
    overwrite : bool (default: False)
        If true, save the result in the specified array.
        If false, return the result in a newly allocated array.

    Returns
    -------
    xc_gpu : pycuda.gpuarray.GPUArray
        Conjugate of the input array. If `overwrite` is true, the
        returned matrix is the same as the input array.

    Examples
    --------
    >>> import pycuda.driver as drv
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import skcuda.linalg as linalg
    >>> linalg.init()
    >>> x = np.array([[1+1j, 2-2j, 3+3j, 4-4j], [5+5j, 6-6j, 7+7j, 8-8j]], np.complex64)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = linalg.conj(x_gpu)
    >>> np.all(x == np.conj(y_gpu.get()))
    True

    """

    # Don't attempt to process non-complex matrix types:
    if x_gpu.dtype in [np.float32, np.float64]:
        return x_gpu

    func = _get_conj_kernel(x_gpu.dtype)
    if overwrite:
        func(x_gpu, x_gpu)
        return x_gpu
    else:
        y_gpu = gpuarray.empty_like(x_gpu)
        func(x_gpu, y_gpu)
        return y_gpu

@context_dependent_memoize
def _get_diag_kernel(dtype):
    ctype=tools.dtype_to_ctype(dtype)
    return el.ElementwiseKernel("{ctype} *d, {ctype} *v, int N".format(ctype=ctype),
                                "d[i*(N+1)] = v[i]")

def diag(v_gpu):
    """
    Construct a diagonal matrix if input array is one-dimensional,
    or extracts diagonal entries of a two-dimensional array.

    If input-array is one-dimensional, constructs a matrix in device
    memory whose diagonal elements correspond to the elements in the
    specified array; all non-diagonal elements are set to 0.

    If input-array is two-dimensional, constructs an array in device memory
    whose elements correspond to the elements along the main-diagonal
    of the specified array.

    Parameters
    ----------
    v_obj : pycuda.gpuarray.GPUArray
            Input array of shape `(n,m)`.

    Returns
    -------
    d_gpu : pycuda.gpuarray.GPUArray
        If v_obj has shape `(n,1)`, output is diagonal matrix of dimensions `[n, n]`.
        If v_obj has shape `(n,m)`, output is array of length `min(n,m)`.

    Examples
    --------
    >>> import pycuda.driver as drv
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import skcuda.linalg as linalg
    >>> linalg.init()
    >>> v = np.array([1, 2, 3, 4, 5, 6], np.float32)
    >>> v_gpu = gpuarray.to_gpu(v)
    >>> d_gpu = linalg.diag(v_gpu)
    >>> np.all(d_gpu.get() == np.diag(v))
    True
    >>> v = np.array([1j, 2j, 3j, 4j, 5j, 6j], np.complex64)
    >>> v_gpu = gpuarray.to_gpu(v)
    >>> d_gpu = linalg.diag(v_gpu)
    >>> np.all(d_gpu.get() == np.diag(v))
    True
    >>> v = np.array([[1., 2., 3.],[4., 5., 6.]], np.float64)
    >>> v_gpu = gpuarray.to_gpu(v)
    >>> d_gpu = linalg.diag(v_gpu)
    >>> d_gpu
    array([ 1.,  5.])
    """

    if v_gpu.dtype not in [np.float32, np.float64, np.complex64,
                           np.complex128]:
        raise ValueError('unrecognized type')

    alloc = misc._global_cublas_allocator

    if (len(v_gpu.shape) > 1) and (len(v_gpu.shape) < 3):
        if (v_gpu.dtype == np.complex64):
            func = cublas.cublasCcopy
        elif (v_gpu.dtype == np.float32):
            func = cublas.cublasScopy
        elif (v_gpu.dtype == np.complex128):
            func = cublas.cublasZcopy
        elif (v_gpu.dtype == np.float64):
            func = cublas.cublasDcopy
        else:
            raise ValueError('unsupported input type')

        n = min(v_gpu.shape)
        incx = int(np.sum(v_gpu.strides)/v_gpu.dtype.itemsize)

        # Allocate the output array
        d_gpu = gpuarray.empty(n, v_gpu.dtype.type, allocator=alloc)

        handle = misc._global_cublas_handle
        func(handle, n, v_gpu.gpudata, incx, d_gpu.gpudata, 1)
        return d_gpu
    elif len(v_gpu.shape) >= 3:
        raise ValueError('input array cannot have greater than 2-dimensions')

    # Initialize output matrix:
    N = len(v_gpu)
    if N <= 0:
        raise ValueError('N must be greater than 0')
    d_gpu = misc.zeros((N, N), v_gpu.dtype, allocator=alloc)

    func = _get_diag_kernel(v_gpu.dtype)
    func(d_gpu, v_gpu, N, slice=slice(0, N))
    return d_gpu

@context_dependent_memoize
def _get_eye_kernel(dtype):
    ctype=tools.dtype_to_ctype(dtype)
    return el.ElementwiseKernel("{ctype} *e".format(ctype=ctype), "e[i] = 1")

def eye(N, dtype=np.float32):
    """
    Construct a 2D matrix with ones on the diagonal and zeros elsewhere.

    Constructs a matrix in device memory whose diagonal elements
    are set to 1 and non-diagonal elements are set to 0.

    Parameters
    ----------
    N : int
        Number of rows or columns in the output matrix.
    dtype : type
        Matrix data type.

    Returns
    -------
    e_gpu : pycuda.gpuarray.GPUArray
        Diagonal matrix of dimensions `[N, N]` with diagonal values
        set to 1.

    Examples
    --------
    >>> import pycuda.driver as drv
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import skcuda.linalg as linalg
    >>> linalg.init()
    >>> N = 5
    >>> e_gpu = linalg.eye(N)
    >>> np.all(e_gpu.get() == np.eye(N))
    True
    >>> e_gpu = linalg.eye(N, np.complex64)
    >>> np.all(e_gpu.get() == np.eye(N, dtype=np.complex64))
    True

    """

    if dtype not in [np.float32, np.float64, np.complex64,
                     np.complex128]:
        raise ValueError('unrecognized type')
    if N <= 0:
        raise ValueError('N must be greater than 0')
    alloc = misc._global_cublas_allocator

    e_gpu = misc.zeros((N, N), dtype, allocator=alloc)
    func = _get_eye_kernel(dtype)
    func(e_gpu, slice=slice(0, N*N, N+1))
    return e_gpu

def pinv(a_gpu, rcond=1e-15, lib='cusolver'):
    """
    Moore-Penrose pseudoinverse.

    Compute the Moore-Penrose pseudoinverse of the specified matrix.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, n)`.
    rcond : float
        Singular values smaller than `rcond`*max(singular_values)`
        are set to zero.
    lib : str
        Library to use. May be either 'cula' or 'cusolver'.

    Returns
    -------
    a_inv_gpu : pycuda.gpuarray.GPUArray
        Pseudoinverse of input matrix.

    Notes
    -----
    Double precision is only supported if the standard version of the
    CULA Dense toolkit is installed.

    This function destroys the contents of the input matrix.

    If the input matrix is square, the pseudoinverse uses less memory.

    Examples
    --------
    >>> import pycuda.driver as drv
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import skcuda.linalg as linalg
    >>> linalg.init()
    >>> a = np.asarray(np.random.rand(8, 4), np.float32)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> a_inv_gpu = linalg.pinv(a_gpu)
    >>> np.allclose(np.linalg.pinv(a), a_inv_gpu.get(), 1e-4)
    True
    >>> b = np.asarray(np.random.rand(8, 4)+1j*np.random.rand(8, 4), np.complex64)
    >>> b_gpu = gpuarray.to_gpu(b)
    >>> b_inv_gpu = linalg.pinv(b_gpu)
    >>> np.allclose(np.linalg.pinv(b), b_inv_gpu.get(), 1e-4)
    True

    Notes
    -----
    The CUSOLVER backend cannot be used with CUDA 7.0.
    """

    if lib == 'cula' and not _has_cula:
        raise NotImplementedError('CULA not installed')

    # Perform in-place SVD if the matrix is square to save memory:
    if a_gpu.shape[0] == a_gpu.shape[1]:
        u_gpu, s_gpu, vh_gpu = svd(a_gpu, 's', 'o', lib)
    else:
        u_gpu, s_gpu, vh_gpu = svd(a_gpu, 's', 's', lib)

    # Suppress very small singular values and convert the singular value array
    # to complex if the original matrix is complex so that the former can be
    # handled by dot_diag():
    cutoff_gpu = gpuarray.max(s_gpu)*rcond
    real_ctype = tools.dtype_to_ctype(s_gpu.dtype)
    if a_gpu.dtype in [np.complex64, np.complex128]:
        if s_gpu.dtype == np.float32:
            complex_dtype = np.complex64
        elif s_gpu.dtype == np.float64:
            complex_dtype = np.complex128
        else:
            raise ValueError('cannot convert singular values to complex')
        s_complex_gpu = gpuarray.empty(len(s_gpu), complex_dtype)
        complex_ctype = tools.dtype_to_ctype(complex_dtype)
        cutoff_func = el.ElementwiseKernel("{real_ctype} *s_real, {complex_ctype} *s_complex,"
            " {real_ctype} *cutoff".format(real_ctype=real_ctype, complex_ctype=complex_ctype),
            "if (s_real[i] > cutoff[0]) {s_complex[i] = 1/s_real[i];} else {s_complex[i] = 0;}")
        cutoff_func(s_gpu, s_complex_gpu, cutoff_gpu)

        # Compute the pseudoinverse without allocating a new diagonal matrix:
        return dot(vh_gpu, dot_diag(s_complex_gpu, u_gpu, 't'), 'c', 'c')

    else:
        cutoff_func = el.ElementwiseKernel("{real_ctype} *s, {real_ctype} *cutoff".format(real_ctype=real_ctype),
                                           "if (s[i] > cutoff[0]) {s[i] = 1/s[i];} else {s[i] = 0;}")
        cutoff_func(s_gpu, cutoff_gpu)

        # Compute the pseudoinverse without allocating a new diagonal matrix:
        return dot(vh_gpu, dot_diag(s_gpu, u_gpu, 't'), 'c', 'c')



@context_dependent_memoize
def _get_tril_kernel(use_double, use_complex, cols):
    template = Template("""
    #include <pycuda-complex.hpp>

    #if ${use_double}
    #if ${use_complex}
    #define FLOAT pycuda::complex<double>
    #else
    #define FLOAT double
    #endif
    #else
    #if ${use_complex}
    #define FLOAT pycuda::complex<float>
    #else
    #define FLOAT float
    #endif
    #endif

    __global__ void tril(FLOAT *a, unsigned int N) {
        unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                           blockIdx.x*blockDim.x+threadIdx.x;
        unsigned int ix = idx/${cols};
        unsigned int iy = idx%${cols};

        if (idx < N) {
            if (ix < iy)
                a[idx] = 0.0;
        }
    }
    """)
    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    tmpl = template.substitute(use_double=use_double,
                               use_complex=use_complex,
                               cols=cols)
    mod = SourceModule(tmpl, cache_dir=cache_dir)
    return mod.get_function("tril")


def tril(a_gpu, overwrite=False, handle=None):
    """
    Lower triangle of a matrix.

    Return the lower triangle of a square matrix.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, m)`
    overwrite : bool (default: False)
        If true, zero out the upper triangle of the matrix.
        If false, return the result in a newly allocated matrix.
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.

    Returns
    -------
    l_gpu : pycuda.gpuarray
        The lower triangle of the original matrix.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.driver as drv
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> import skcuda.linalg as linalg
    >>> linalg.init()
    >>> a = np.asarray(np.random.rand(4, 4), np.float32)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> l_gpu = linalg.tril(a_gpu, False)
    >>> np.allclose(np.tril(a), l_gpu.get())
    True

    """

    if handle is None:
        handle = misc._global_cublas_handle

    alloc = misc._global_cublas_allocator

    if len(a_gpu.shape) != 2 or a_gpu.shape[0] != a_gpu.shape[1]:
        raise ValueError('matrix must be square')

    if a_gpu.dtype == np.float32:
        swap_func = cublas.cublasSswap
        copy_func = cublas.cublasScopy
        use_double = 0
        use_complex = 0
    elif a_gpu.dtype == np.float64:
        swap_func = cublas.cublasDswap
        copy_func = cublas.cublasDcopy
        use_double = 1
        use_complex = 0
    elif a_gpu.dtype == np.complex64:
        swap_func = cublas.cublasCswap
        copy_func = cublas.cublasCcopy
        use_double = 0
        use_complex = 1
    elif a_gpu.dtype == np.complex128:
        swap_func = cublas.cublasZswap
        copy_func = cublas.cublasZcopy
        use_double = 1
        use_complex = 1
    else:
        raise ValueError('unrecognized type')

    N = a_gpu.shape[0]

    # Get block/grid sizes:
    dev = misc.get_current_device()
    block_dim, grid_dim = misc.select_block_grid_sizes(dev, a_gpu.shape)
    tril = _get_tril_kernel(use_double, use_complex, cols=N)
    if not overwrite:
        a_orig_gpu = gpuarray.empty(a_gpu.shape, a_gpu.dtype, allocator=alloc)
        copy_func(handle, a_gpu.size, int(a_gpu.gpudata), 1, int(a_orig_gpu.gpudata), 1)

    tril(a_gpu, np.uint32(a_gpu.size),
         block=block_dim,
         grid=grid_dim)

    if overwrite:
        return a_gpu
    else:

        # Restore original contents of a_gpu:
        swap_func(handle, a_gpu.size, int(a_gpu.gpudata), 1, int(a_orig_gpu.gpudata), 1)
        return a_orig_gpu


@context_dependent_memoize
def _get_triu_kernel(use_double, use_complex, cols):
    template = Template("""
    #include <pycuda-complex.hpp>

    #if ${use_double}
    #if ${use_complex}
    #define FLOAT pycuda::complex<double>
    #else
    #define FLOAT double
    #endif
    #else
    #if ${use_complex}
    #define FLOAT pycuda::complex<float>
    #else
    #define FLOAT float
    #endif
    #endif

    __global__ void triu(FLOAT *a, unsigned int N) {
        unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                           blockIdx.x*blockDim.x+threadIdx.x;
        unsigned int ix = idx/${cols};
        unsigned int iy = idx%${cols};

        if (idx < N) {
            if (ix > iy)
                a[idx] = 0.0;
        }
    }
    """)
    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    tmpl = template.substitute(use_double=use_double,
                               use_complex=use_complex,
                               cols=cols)
    mod = SourceModule(tmpl, cache_dir=cache_dir)
    return mod.get_function("triu")


def triu(a_gpu, k=0, overwrite=False, handle=None):
    """
    Upper triangle of a matrix.

    Return the upper triangle of a square matrix.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, m)`
    overwrite : bool (default: False)
        If true, zero out the lower triangle of the matrix.
        If false, return the result in a newly allocated matrix.
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.

    Returns
    -------
    u_gpu : pycuda.gpuarray
        The upper triangle of the original matrix.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.driver as drv
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> import skcuda.linalg as linalg
    >>> linalg.init()
    >>> a = np.asarray(np.random.rand(4, 4), np.float32)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> u_gpu = linalg.triu(a_gpu, False)
    >>> np.allclose(np.triu(a), u_gpu.get())
    True

    """

    if handle is None:
        handle = misc._global_cublas_handle

    alloc = misc._global_cublas_allocator

    if len(a_gpu.shape) != 2 or a_gpu.shape[0] != a_gpu.shape[1]:
        raise ValueError('matrix must be square')

    if a_gpu.dtype == np.float32:
        swap_func = cublas.cublasSswap
        copy_func = cublas.cublasScopy
        use_double = 0
        use_complex = 0
    elif a_gpu.dtype == np.float64:
        swap_func = cublas.cublasDswap
        copy_func = cublas.cublasDcopy
        use_double = 1
        use_complex = 0
    elif a_gpu.dtype == np.complex64:
        swap_func = cublas.cublasCswap
        copy_func = cublas.cublasCcopy
        use_double = 0
        use_complex = 1
    elif a_gpu.dtype == np.complex128:
        swap_func = cublas.cublasZswap
        copy_func = cublas.cublasZcopy
        use_double = 1
        use_complex = 1
    else:
        raise ValueError('unrecognized type')

    N = a_gpu.shape[0]

    # Get block/grid sizes:
    dev = misc.get_current_device()
    block_dim, grid_dim = misc.select_block_grid_sizes(dev, a_gpu.shape)
    tril = _get_triu_kernel(use_double, use_complex, cols=N)
    if not overwrite:

        a_orig_gpu = gpuarray.empty( (N,N),
                                    a_gpu.dtype, allocator=alloc)
        copy_func(handle, a_gpu.size, int(a_gpu.gpudata), 1, int(a_orig_gpu.gpudata), 1)

    tril(a_gpu, np.uint32(a_gpu.size),
         block=block_dim,
         grid=grid_dim)

    if overwrite:
        return a_gpu
    else:

        # Restore original contents of a_gpu:
        swap_func(handle, a_gpu.size, int(a_gpu.gpudata), 1, int(a_orig_gpu.gpudata), 1)
        return a_orig_gpu

def multiply(x_gpu, y_gpu, overwrite=False):
    """
    Element-wise array multiplication (Hadamard product).

    Parameters
    ----------
    x_gpu, y_gpu : pycuda.gpuarray.GPUArray
        Input arrays to be multiplied.
    dev : pycuda.driver.Device
        Device object to be used.
    overwrite : bool (default: False)
        If true, return the result in `y_gpu`.
        is false, return the result in a newly allocated array.

    Returns
    -------
    z_gpu : pycuda.gpuarray.GPUArray
        The element-wise product of the input arrays.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> import skcuda.linalg as linalg
    >>> linalg.init()
    >>> x = np.asarray(np.random.rand(4, 4), np.float32)
    >>> y = np.asarray(np.random.rand(4, 4), np.float32)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> z_gpu = linalg.multiply(x_gpu, y_gpu)
    >>> np.allclose(x*y, z_gpu.get())
    True

    """

    alloc = misc._global_cublas_allocator

    if x_gpu.shape != y_gpu.shape:
        raise ValueError('input arrays must have the same shape')

    if x_gpu.dtype not in [np.float32, np.float64, np.complex64,
                           np.complex128]:
        raise ValueError('unrecognized type')

    x_ctype = tools.dtype_to_ctype(x_gpu.dtype)
    y_ctype = tools.dtype_to_ctype(y_gpu.dtype)

    if overwrite:
        func = el.ElementwiseKernel("{x_ctype} *x, {y_ctype} *y".format(x_ctype=x_ctype,
                                                                        y_ctype=y_ctype),
                                    "y[i] *= x[i]")
        func(x_gpu, y_gpu)
        return y_gpu
    else:
        result_type = np.result_type(x_gpu.dtype, y_gpu.dtype)
        z_gpu = gpuarray.empty(x_gpu.shape, result_type, allocator=alloc)
        func = \
               el.ElementwiseKernel("{x_ctype} *x, {y_ctype} *y, {z_type} *z".format(x_ctype=x_ctype,
                                                                                     y_ctype=y_ctype,
                                                                                     z_type=tools.dtype_to_ctype(result_type)),
                                    "z[i] = x[i]*y[i]")
        func(x_gpu, y_gpu, z_gpu)
        return z_gpu

def norm(x_gpu, handle=None):
    """
    Euclidean norm (2-norm) of real vector.

    Computes the Euclidean norm of an array.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Input array.
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.

    Returns
    -------
    nrm : real
        Euclidean norm of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> import skcuda.linalg as linalg
    >>> linalg.init()
    >>> x = np.asarray(np.random.rand(4, 4), np.float32)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> nrm = linalg.norm(x_gpu)
    >>> np.allclose(nrm, np.linalg.norm(x))
    True
    >>> x_gpu = gpuarray.to_gpu(np.array([3+4j, 12-84j]))
    >>> linalg.norm(x_gpu)
    85.0

    """

    if handle is None:
        handle = misc._global_cublas_handle

    if len(x_gpu.shape) != 1:
        x_gpu = x_gpu.ravel()

    # Compute inner product for 1D arrays:
    if (x_gpu.dtype == np.complex64):
        cublas_func = cublas.cublasScnrm2
    elif (x_gpu.dtype == np.float32):
        cublas_func = cublas.cublasSnrm2
    elif (x_gpu.dtype == np.complex128):
        cublas_func = cublas.cublasDznrm2
    elif (x_gpu.dtype == np.float64):
        cublas_func = cublas.cublasDnrm2
    else:
        raise ValueError('unsupported input type')

    return cublas_func(handle, x_gpu.size, x_gpu.gpudata, 1)

def scale(alpha, x_gpu, alpha_real=False, handle=None):
    """
    Scale a vector by a factor alpha.

    Parameters
    ----------
    alpha : scalar
        Scale parameter
    x_gpu : pycuda.gpuarray.GPUArray
        Input array.
    alpha_real : bool
        If `True` and `x_gpu` is complex, then one of the specialized versions
        `cublasCsscal` or `cublasZdscal` is used which might improve
        performance for large arrays.  (By default, `alpha` is coerced to
        the corresponding complex type.)
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> import skcuda.linalg as linalg
    >>> linalg.init()
    >>> x = np.asarray(np.random.rand(4, 4), np.float32)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> alpha = 2.4
    >>> linalg.scale(alpha, x_gpu)
    >>> np.allclose(x_gpu.get(), alpha*x)
    True
    """

    if handle is None:
        handle = misc._global_cublas_handle

    if len(x_gpu.shape) != 1:
        x_gpu = x_gpu.ravel()

    cublas_func = {
        np.float32: cublas.cublasSscal,
        np.float64: cublas.cublasDscal,
        np.complex64: cublas.cublasCsscal if alpha_real else
                      cublas.cublasCscal,
        np.complex128: cublas.cublasZdscal if alpha_real else
                       cublas.cublasZscal
    }.get(x_gpu.dtype.type, None)

    if cublas_func:
        return cublas_func(handle, x_gpu.size, alpha, x_gpu.gpudata, 1)
    else:
        raise ValueError('unsupported input type')

def inv(a_gpu, overwrite=False, ipiv_gpu=None, lib='cusolver'):
    """
    Compute the inverse of a matrix.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Square (n, n) matrix to be inverted.
    overwrite : bool (default: False)
        Discard data in `a` (may improve performance).
    ipiv_gpu : pycuda.gpuarray.GPUArray (optional)
        Temporary array of size `n`, can be supplied to save allocations.
    lib : str
        Library to use. May be either 'cula' or 'cusolver'.

    Returns
    -------
    ainv_gpu : pycuda.gpuarray.GPUArray
        Inverse of the matrix `a`.

    Raises
    ------
    LinAlgError :
        If `a` is singular.
    ValueError :
        * If `a` is not square, or not 2-dimensional.
        * If ipiv was not None but had the wrong dtype or shape.

    Notes
    -----
    When the CUSOLVER backend is selected, an extra copy will be performed if
    `overwrite` is set to transfer the result back into the input matrix.
    """

    alloc = misc._global_cublas_allocator
    data_dtype = a_gpu.dtype.type

    if len(a_gpu.shape) != 2 or a_gpu.shape[0] != a_gpu.shape[1]:
        raise ValueError('expected square matrix')

    n = a_gpu.shape[0]
    if ipiv_gpu is None:
        alloc = misc._global_cublas_allocator
        ipiv_gpu = gpuarray.empty((n, 1), np.int32, allocator=alloc)
    elif ipiv_gpu.dtype != np.int32 or np.prod(ipiv_gpu.shape) < n:
        raise ValueError('invalid ipiv provided')

    if lib == 'cula':
        if not _has_cula:
            raise NotImplementedError('CULA not installed')

        if (data_dtype == np.complex64):
            getrf = cula.culaDeviceCgetrf
            getri = cula.culaDeviceCgetri
        elif (data_dtype == np.float32):
            getrf = cula.culaDeviceSgetrf
            getri = cula.culaDeviceSgetri
        elif (data_dtype == np.complex128):
            getrf = cula.culaDeviceZgetrf
            getri = cula.culaDeviceZgetri
        elif (data_dtype == np.float64):
            getrf = cula.culaDeviceDgetrf
            getri = cula.culaDeviceDgetri

        out = a_gpu if overwrite else a_gpu.copy()
        try:
            getrf(n, n, out.gpudata, n, ipiv_gpu.gpudata)
            getri(n, out.gpudata, n, ipiv_gpu.gpudata)
        except cula.culaDataError as e:
            raise LinAlgError(e)
        return out
    elif lib == 'cusolver':
        if (data_dtype == np.complex64):
            getrf = cusolver.cusolverDnCgetrf
            bufsize = cusolver.cusolverDnCgetrf_bufferSize
            getrs = cusolver.cusolverDnCgetrs
        elif (data_dtype == np.float32):
            getrf = cusolver.cusolverDnSgetrf
            bufsize = cusolver.cusolverDnSgetrf_bufferSize
            getrs = cusolver.cusolverDnSgetrs
        elif (data_dtype == np.complex128):
            getrf = cusolver.cusolverDnZgetrf
            bufsize = cusolver.cusolverDnZgetrf_bufferSize
            getrs = cusolver.cusolverDnZgetrs
        elif (data_dtype == np.float64):
            getrf = cusolver.cusolverDnDgetrf
            bufsize = cusolver.cusolverDnDgetrf_bufferSize
            getrs = cusolver.cusolverDnDgetrs

        try:
            in_gpu = a_gpu if overwrite else a_gpu.copy()
            Lwork = bufsize(misc._global_cusolver_handle, n, n, in_gpu.gpudata, n)
            Work = gpuarray.empty(Lwork, data_dtype, allocator=alloc)
            devInfo = gpuarray.empty(1, np.int32, allocator=alloc)
            getrf(misc._global_cusolver_handle, n, n, in_gpu.gpudata, n, 
                  Work.gpudata, ipiv_gpu.gpudata, devInfo.gpudata)
        except cusolver.CUSOLVER_ERROR as e:
            raise LinAlgError(e)

        d = devInfo.get()[0]
        if d != 0:
            raise LinAlgError(d) # raised for singular matrix or bad params
        try:
            b_gpu = eye(n, data_dtype)
            getrs(misc._global_cusolver_handle, cublas._CUBLAS_OP['n'], n, n,
                  in_gpu.gpudata, n, ipiv_gpu.gpudata, b_gpu.gpudata, n,
                  devInfo.gpudata)

            # Since CUSOLVER's getrs functions save their output in b_gpu, we
            # need to copy it back to the input matrix if overwrite is requested:
            if overwrite:
                a_gpu.set(b_gpu)
                return a_gpu
            else:
                return b_gpu
        except cusolver.CUSOLVER_ERROR as e:
            raise LinAlgError(e)
    else:
        raise ValueError('invalid library specified')

def trace(x_gpu, handle=None):
    """
    Return the sum along the main diagonal of the array.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Matrix to calculate the trace of.

    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.

    Returns
    -------
    trace : number
        trace of x_gpu
    """
    if handle is None:
        handle = misc._global_cublas_handle

    if len(x_gpu.shape) != 2:
        raise ValueError('Only 2D matrices are supported')

    one = gpuarray.to_gpu(np.ones(1, dtype=x_gpu.dtype))
    if (x_gpu.dtype == np.complex64):
        cublas_func = cublas.cublasCdotu
    elif (x_gpu.dtype == np.float32):
        cublas_func = cublas.cublasSdot
    elif (x_gpu.dtype == np.complex128):
        cublas_func = cublas.cublasZdotu
    elif (x_gpu.dtype == np.float64):
        cublas_func = cublas.cublasDdot

    if not cublas_func:
        raise ValueError('unsupported input type')

    if x_gpu.flags.c_contiguous:
        incx = x_gpu.shape[1] + 1
    else:
        incx = x_gpu.shape[0] + 1
    return cublas_func(handle, np.min(x_gpu.shape),
                       x_gpu.gpudata, incx, one.gpudata, 0)


@context_dependent_memoize
def _get_det_kernel(dtype):
    ctype = tools.dtype_to_ctype(dtype)
    args = "int* ipiv, {ctype}* x, unsigned xn".format(ctype=ctype)
    return ReductionKernel(dtype, "1.0", "a*b",
                           "(ipiv[i] != i+1) ? -x[i*xn+i] : x[i*xn+i]", args)

def det(a_gpu, overwrite=False, workspace_gpu=None, ipiv_gpu=None, handle=None, lib='cusolver'):
    """
    Compute the determinant of a square matrix.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        The square n*n matrix of which to calculate the determinant.
    overwrite : bool (default: False)
        Discard data in `a` (may improve performance).
    workspace_gpu : pycuda.gpuarray.GPUArray (optional)
        Temporary array of size Lwork (typically computed by CUSOLVER helper
        functions), can be supplied to save allocations. Only used if lib == 'cusolver'.
    ipiv_gpu : pycuda.gpuarray.GPUArray (optional)
        Temporary array of size n, can be supplied to save allocations.
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.
    lib : str
        Library to use. May be either 'cula' or 'cusolver'.

    Returns
    -------
    det : number
        determinant of a_gpu
    """

    if handle is None:
        handle = misc._global_cublas_handle

    if lib == 'cula':
        if not _has_cula:
            raise NotImplementedError('CULA not installed')

        if len(a_gpu.shape) != 2:
            raise ValueError('Only 2D matrices are supported')
        if a_gpu.shape[0] != a_gpu.shape[1]:
            raise ValueError('Only square matrices are supported')

        if (a_gpu.dtype == np.complex64):
            getrf = cula.culaDeviceCgetrf
        elif (a_gpu.dtype == np.float32):
            getrf = cula.culaDeviceSgetrf
        elif (a_gpu.dtype == np.complex128):
            getrf = cula.culaDeviceZgetrf
        elif (a_gpu.dtype == np.float64):
            getrf = cula.culaDeviceDgetrf
        else:
            raise ValueError('unsupported input type')

        n = a_gpu.shape[0]
        alloc = misc._global_cublas_allocator
        if ipiv_gpu is None:
            ipiv_gpu = gpuarray.empty((n, 1), np.int32, allocator=alloc)
        elif ipiv_gpu.dtype != np.int32 or np.prod(ipiv_gpu.shape) < n:
            raise ValueError('invalid ipiv provided')

        out = a_gpu if overwrite else a_gpu.copy()
        try:
            getrf(n, n, out.gpudata, n, ipiv_gpu.gpudata)
            return _get_det_kernel(a_gpu.dtype)(ipiv_gpu, out, n).get()
        except cula.culaDataError as e:
            raise LinAlgError(e)

    elif lib == 'cusolver':
        if not _has_cusolver:
            raise NotImplementedError('CUSOLVER not installed')

        cusolverHandle = misc._global_cusolver_handle

        if (a_gpu.dtype == np.complex64):
            getrf = cusolver.cusolverDnCgetrf
            bufsize = cusolver.cusolverDnCgetrf_bufferSize
        elif (a_gpu.dtype == np.float32):
            getrf = cusolver.cusolverDnSgetrf
            bufsize = cusolver.cusolverDnSgetrf_bufferSize
        elif (a_gpu.dtype == np.complex128):
            getrf = cusolver.cusolverDnZgetrf
            bufsize = cusolver.cusolverDnZgetrf_bufferSize
        elif (a_gpu.dtype == np.float64):
            getrf = cusolver.cusolverDnDgetrf
            bufsize = cusolver.cusolverDnDgetrf_bufferSize
        else:
            raise ValueError('unsupported input type')

        out = a_gpu if overwrite else a_gpu.copy()

        n = a_gpu.shape[0]
        alloc = misc._global_cublas_allocator
        Lwork = bufsize(cusolverHandle, n, n, int(out.gpudata), n)
        if workspace_gpu is None:
            workspace_gpu = gpuarray.empty(Lwork, a_gpu.dtype, allocator=alloc)
        elif workspace_gpu.dtype != a_gpu.dtype or len(workspace_gpu) < Lwork:
            raise ValueError('invalid workspace provided')

        if ipiv_gpu is None:
            ipiv_gpu = gpuarray.empty((n, 1), np.int32, allocator=alloc)
        elif ipiv_gpu.dtype != np.int32 or np.prod(ipiv_gpu.shape) < n:
            raise ValueError('invalid ipiv provided')

        devInfo = gpuarray.empty(1, np.int32, allocator=alloc)
        try:
            getrf(cusolverHandle, n, n, out.gpudata, n, workspace_gpu.gpudata,
                  ipiv_gpu.gpudata, devInfo.gpudata)
            return _get_det_kernel(a_gpu.dtype)(ipiv_gpu, out, n).get()
        except cusolver.CUSOLVER_ERROR as e:
            raise LinAlgError(e)
    else:
        raise ValueError('invalid library specified')

def qr(a_gpu, mode='reduced', handle=None, lib='cusolver'):
    """
    QR Decomposition.

    Factor the real/complex matrix `a` as `QR`, where `Q` is an orthonormal/unitary
    matrix and `R` is an upper triangular matrix.

    Parameters
    ----------
    a_gpu: pycuda.gpuarray.GPUArray
        Real/complex input matrix  `a` with dimensions `(m, n)`.
        `a` is assumed to be `m`>=`n`.
    mode :  {'reduced', 'economic', 'r'}
        'reduced' : returns `Q`, `R` with dimensions `(m, k)` and `(k, n)` (default).
        'economic' : returns `Q` only with dimensions `(m, k)`.
        'r' : returns `R` only with dimensions `(k, n)` with `k`=min`(m,n)`.
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.
    lib : str
        Library to use. May be either 'cula' or 'cusolver'.

    Returns
    -------
    q_gpu : pycuda.gpuarray.GPUArray
        Orthonormal/unitary matrix (depending on whether or not `A` is real/complex).
    r_gpu : pycuda.gpuarray.GPUArray
        The upper-triangular matrix.

    Notes
    -----
    Double precision is only supported if the standard version of the
    CULA Dense toolkit is installed.

    This function destroys the contents of the input matrix.

    Arrays are assumed to be stored in column-major order, i.e., order='F'.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> import skcuda.linalg as linalg
    >>> linalg.init()
    >>> # Rectangular matrix A, np.float32
    >>> A = np.array(np.random.randn(9, 7), np.float32, order='F')
    >>> A_gpu = gpuarray.to_gpu(A)
    >>> Q_gpu, R_gpu = linalg.qr(A_gpu, 'reduced')
    >>> np.allclose(A, np.dot(Q_gpu.get(), R_gpu.get()), 1e-4)
    True
    >>> # Square matrix A, np.complex128
    >>> A = np.random.randn(9, 9) + 1j*np.random.randn(9, 9)
    >>> A = np.asarray(A, np.complex128, order='F')
    >>> A_gpu = gpuarray.to_gpu(A)
    >>> Q_gpu, R_gpu = linalg.qr(A_gpu, 'reduced')
    >>> np.allclose(A, np.dot(Q_gpu.get(), R_gpu.get()), 1e-4)
    True
    >>> np.allclose(np.identity(Q_gpu.shape[0]) + 1j*0, np.dot(Q_gpu.get().conj().T, Q_gpu.get()), 1e-4)
    True
    >>> # Numpy QR and CULA QR
    >>> A = np.array(np.random.randn(9, 7), np.float32, order='F')
    >>> Q, R = np.linalg.qr(A, 'reduced')
    >>> a_gpu = gpuarray.to_gpu(A)
    >>> Q_gpu, R_gpu = linalg.qr(a_gpu, 'reduced')
    >>> np.allclose(Q, Q_gpu.get(), 1e-4)
    True
    >>> np.allclose(R, R_gpu.get(), 1e-4)
    True
    """

    alloc = misc._global_cublas_allocator

    if handle is None:
         handle = misc._global_cublas_handle

    data_type = a_gpu.dtype.type

    if lib == 'cula':
        if not _has_cula:
            raise NotImplementedError('CULA not installed')

        # The free version of CULA only supports single precision floating
        # point numbers:

        real_type = np.float32
        if data_type == np.complex64:
            func_qr = cula.culaDeviceCgeqrf
            func_q = cula.culaDeviceCungqr
            copy_func = cublas.cublasCcopy
            use_double = 0
            use_complex = 1
        elif data_type == np.float32:
            func_qr = cula.culaDeviceSgeqrf
            func_q = cula.culaDeviceSorgqr
            copy_func = cublas.cublasScopy
            use_double = 0
            use_complex = 0
        else:
            if cula._libcula_toolkit == 'standard':
                if data_type == np.complex128:
                    func_qr = cula.culaDeviceZgeqrf
                    func_q = cula.culaDeviceZungqr
                    copy_func = cublas.cublasZcopy
                    use_double = 1
                    use_complex = 1
                elif data_type == np.float64:
                    func_qr = cula.culaDeviceDgeqrf
                    func_q = cula.culaDeviceDorgqr
                    copy_func = cublas.cublasDcopy
                    use_double = 1
                    use_complex = 0
                else:
                    raise ValueError('unsupported type')
                real_type = np.float64
            else:
                raise ValueError('double precision not supported')
    elif lib == 'cusolver':
        if not _has_cusolver:
            raise NotImplementedError('CUSOLVER not installed')

        cusolverHandle = misc._global_cusolver_handle
        if data_type == np.complex64:
            func_qr = cusolver.cusolverDnCgeqrf
            func_q = cusolver.cusolverDnCungqr
            bufsize_qr = cusolver.cusolverDnCgeqrf_bufferSize
            bufsize_q = cusolver.cusolverDnCungqr_bufferSize
            copy_func = cublas.cublasCcopy
            use_double = 0
            use_complex = 1
        elif data_type == np.float32:
            func_qr = cusolver.cusolverDnSgeqrf
            func_q = cusolver.cusolverDnSorgqr
            bufsize_qr = cusolver.cusolverDnSgeqrf_bufferSize
            bufsize_q = cusolver.cusolverDnSorgqr_bufferSize
            copy_func = cublas.cublasScopy
            use_double = 0
            use_complex = 0
        elif data_type == np.complex128:
            real_type = np.float64
            func_qr = cusolver.cusolverDnZgeqrf
            func_q = cusolver.cusolverDnZungqr
            bufsize_qr = cusolver.cusolverDnZgeqrf_bufferSize
            bufsize_q = cusolver.cusolverDnZungqr_bufferSize
            copy_func = cublas.cublasZcopy
            use_double = 1
            use_complex = 1
        elif data_type == np.float64:
            real_type = np.float64
            func_qr = cusolver.cusolverDnDgeqrf
            func_q = cusolver.cusolverDnDorgqr
            bufsize_qr = cusolver.cusolverDnDgeqrf_bufferSize
            bufsize_q = cusolver.cusolverDnDorgqr_bufferSize
            copy_func = cublas.cublasDcopy
            use_double = 1
            use_complex = 0
        else:
            raise ValueError('unsupported type')

    else:
        raise ValueError('invalid library specified')

    # CUDA assumes that arrays are stored in column-major order
    m, n = a_gpu.shape

    if m<n and mode != 'r':
        raise ValueError('if m < n only the mode "r" is supported')

    # Set the leading dimension of the input matrix:
    lda = max(1, m)

    # Set k:
    k = min(m, n)

    # Set the leading dimension and allocate u:
    tau_gpu = gpuarray.empty(k, data_type, allocator=alloc, order='F')

    # Compute QR and check error status:
    if lib == 'cula':
        func_qr(m, n, int(a_gpu.gpudata), lda, int(tau_gpu.gpudata))
    else:
        Lwork = bufsize_qr(cusolverHandle, m, n, int(a_gpu.gpudata), m)      
        workspace_gpu = gpuarray.empty(Lwork, data_type, allocator=alloc)
        devInfo = gpuarray.empty(1, np.int32, allocator=alloc)
        func_qr(cusolverHandle, m, n, int(a_gpu.gpudata), lda, int(tau_gpu.gpudata),
                int(workspace_gpu.gpudata), Lwork, int(devInfo.gpudata))

    if mode != 'economic':
        # Get upper triangular matrix R with dimensions (n,n)
        # Note: _get_tril_kernel returns the upper triangular
        r_gpu = gpuarray.empty((m, n), data_type, allocator=alloc, order='F')
        copy_func(handle, a_gpu.size, int(a_gpu.gpudata), 1, int(r_gpu.gpudata), 1)

        # tril
        dev = misc.get_current_device()
        block_dim, grid_dim = misc.select_block_grid_sizes(dev, r_gpu.shape)
        tril = _get_tril_kernel(use_double, use_complex, cols=m) #cols are here rows
        tril(r_gpu, np.uint32(r_gpu.size), block=block_dim, grid=grid_dim)

        # Mode r
        if mode == 'r':
            return r_gpu[:k, :n]

    # Compute Q and check error status:
    if lib == 'cula':
        func_q(m, n, k, int(a_gpu.gpudata), lda, int(tau_gpu.gpudata))

        # Free internal CULA memory:
        cula.culaFreeBuffers()
    else:
        Lwork = bufsize_q(cusolverHandle, m, n, k, int(a_gpu.gpudata), lda, int(tau_gpu.gpudata))
        workspace_gpu = gpuarray.empty(Lwork, data_type, allocator=alloc)
        # Reuse devInfo allocated earlier:
        func_q(cusolverHandle, m, n, k, int(a_gpu.gpudata), lda,
               int(tau_gpu.gpudata), int(workspace_gpu.gpudata), Lwork,
               int(devInfo.gpudata))
    q_gpu = a_gpu

    # Mode economic
    if mode == 'reduced':
        return q_gpu, r_gpu[:k, :n]
    if mode == 'economic':
        return q_gpu

def eig(a_gpu, jobvl='N', jobvr='V', imag='F', lib='cusolver'):
    """
    Eigendecomposition of a matrix.

    Compute the eigenvalues `w`  for a real/complex square matrix `a`
    and (optionally) the real left and right eigenvectors `vl`, `vr`.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Real/complex input matrix  `a` with dimensions `(m, n)`.
    jobvl :  {'V', 'N'}
        'V' : returns `vl`, the left eigenvectors of `a` with dimensions `(m, m)`.
        'N' : left eigenvectors are not computed.
    jobvr :  {'V', 'N'}
        'V' : returns `vr`, the right eigenvectors of `a` with dimensions
        `(m, m)`, (default).
        'N' : right eigenvectors are not computed.
    imag :  {'F', 'T'}
         'F' : imaginary parts of a real matrix are not returned (default).
         'T' : returns the imaginary parts of a real matrix
         (only relevant in the case of single/double precision ).
    lib : str
        Library to use. May be either 'cula' or 'cusolver'. If using
        'cusolver', only symmetric/Hermitian matrices are supported.

    Returns
    -------
    vr_gpu : pycuda.gpuarray.GPUArray
         The normalized (Euclidean norm equal to 1) right eigenvectors,
         such that the column `vr[:,i]` is the eigenvector corresponding
         to the eigenvalue `w[i]`.
    w_gpu : pycuda.gpuarray.GPUArray
        Array containing the real/complex eigenvalues, not necessarily ordered.
        `w` is of length `m`.
    vl_gpu : pycuda.gpuarray.GPUArray
         The normalized (Euclidean norm equal to 1) left eigenvectors,
         such that the column `vl[:,i]` is the eigenvector corresponding
         to the eigenvalue `w[i]`.

    Notes
    -----
    Double precision is only supported if the standard version of the
    CULA Dense toolkit is installed.

    This function destroys the contents of the input matrix.

    Arrays are expected to be stored in column-major order, i.e., order='F'.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> from skcuda import linalg
    >>> linalg.init()
    >>> # Compute right eigenvectors of a symmetric matrix A and verify A*vr = vr*w
    >>> a = np.array(([1,3],[3,5]), np.float32, order='F')
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> vr_gpu, w_gpu = linalg.eig(a_gpu, 'N', 'V')
    >>> np.allclose(np.dot(a, vr_gpu.get()), np.dot(vr_gpu.get(), np.diag(w_gpu.get())), 1e-4)
    True
    >>> # Compute left eigenvectors of a symmetric matrix A and verify vl.T*A = w*vl.T
    >>> a = np.array(([1,3],[3,5]), np.float32, order='F')
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> w_gpu, vl_gpu = linalg.eig(a_gpu, 'V', 'N')
    >>> np.allclose(np.dot(vl_gpu.get().T, a), np.dot(np.diag(w_gpu.get()), vl_gpu.get().T), 1e-4)
    True
    >>> # Compute left/right eigenvectors of a symmetric matrix A and verify A = vr*w*vl.T
    >>> a = np.array(([1,3],[3,5]), np.float32, order='F')
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> vr_gpu, w_gpu, vl_gpu = linalg.eig(a_gpu, 'V', 'V')
    >>> np.allclose(a, np.dot(vr_gpu.get(), np.dot(np.diag(w_gpu.get()), vl_gpu.get().T)), 1e-4)
    True
    >>> # Compute eigenvalues of a square matrix A and verify that trace(A)=sum(w)
    >>> a = np.array(np.random.rand(9,9), np.float32, order='F')
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> w_gpu = linalg.eig(a_gpu, 'N', 'N')
    >>> np.allclose(np.trace(a), sum(w_gpu.get()), 1e-4)
    True
    >>> # Compute eigenvalues of a real valued matrix A possessing complex e-valuesand
    >>> a = np.array(np.array(([1, -2], [1, 3])), np.float32, order='F')
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> w_gpu = linalg.eig(a_gpu, 'N', 'N', imag='T')
    True
    >>> # Compute eigenvalues of a complex valued matrix A and verify that trace(A)=sum(w)
    >>> a = np.array(np.random.rand(2,2) + 1j*np.random.rand(2,2), np.complex64, order='F')
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> w_gpu = linalg.eig(a_gpu, 'N', 'N')
    >>> np.allclose(np.trace(a), sum(w_gpu.get()), 1e-4)
    True
    """

    alloc = misc._global_cublas_allocator

    # The free version of CULA only supports single precision floating
    # point numbers:
    data_type = a_gpu.dtype.type
    real_type = np.float32
    if lib == 'cula':
        if not _has_cula:
            raise NotImplementedError('CULA not installed')

        if data_type == np.complex64:
            func = cula.culaDeviceCgeev
            imag='F'
        elif data_type == np.float32:
            func = cula.culaDeviceSgeev
        else:
            if cula._libcula_toolkit == 'standard':
                if data_type == np.complex128:
                    func = cula.culaDeviceZgeev
                    imag='F'
                elif data_type == np.float64:
                    func = cula.culaDeviceDgeev
                else:
                    raise ValueError('unsupported type')
                real_type = np.float64
            else:
                raise ValueError('double precision not supported')
    elif lib == 'cusolver':
        if not _has_cusolver:
            raise NotImplementedError('CUSOLVER not installed')

        cusolverHandle = misc._global_cusolver_handle

        # FIXME: Seems like CUSOLVER only handles symmetric or Hermitian matrices,
        # look into cusolverDn<t>sygvd
        if data_type == np.complex64:
            func = cusolver.cusolverDnCheevd
            bufsize = cusolver.cusolverDnCheevd_bufferSize
        elif data_type == np.float32:
            func = cusolver.cusolverDnSsyevd
            bufsize = cusolver.cusolverDnSsyevd_bufferSize
        elif data_type == np.complex128:
            func = cusolver.cusolverDnZheevd
            bufsize = cusolver.cusolverDnZheevd_bufferSize
        elif data_type == np.float64:
            real_type = np.float64
            func = cusolver.cusolverDnDsyevd
            bufsize = cusolver.cusolverDnDsyevd_bufferSize
        else:
            raise ValueError('unsupported type')
    else:
        raise ValueError('invalid library specified')

    # CUDA assumes that arrays are stored in column-major order
    n, m = a_gpu.shape

    #Check input
    if(m!=n): raise ValueError('matrix is not square!')
    jobvl = jobvl.upper()
    jobvr = jobvr.upper()

    if jobvl not in ['N', 'V'] :
        raise ValueError('jobvl has to  be "N" or "V" ')
    if jobvr not in ['N', 'V'] :
        raise ValueError('jobvr has to  be "N" or "V" ')
    if imag not in ['T', 'F'] :
        raise ValueError('imag has to  be "T" or "F" ')
    if lib == 'cula':
        w_gpu = gpuarray.empty(m, data_type, order="F", allocator=alloc)
        # Allocate vl, vr, and w:
        vl_gpu = gpuarray.empty((m,m), data_type, order="F", allocator=alloc)
        vr_gpu = gpuarray.empty((m,m), data_type, order="F", allocator=alloc)

        if data_type in (np.complex64, np.complex128):
            #culaDeviceCgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr)
            func(jobvl, jobvr, m, a_gpu.gpudata, m, w_gpu.gpudata, vl_gpu.gpudata , m , vr_gpu.gpudata, m )

        elif data_type in (np.float32, np.float64):
            wi_gpu = gpuarray.zeros(m, data_type, order="F", allocator=alloc)
            func(jobvl, jobvr, m, a_gpu.gpudata, m, w_gpu.gpudata, wi_gpu.gpudata, vl_gpu.gpudata , m , vr_gpu.gpudata, m )

        if imag == 'T':
            w_gpu = w_gpu + (1j)*wi_gpu

        # Free internal CULA memory:
        cula.culaFreeBuffers()

        if jobvl  == 'N' and jobvr == 'N':
            return w_gpu
        elif jobvl == 'V' and jobvr == 'V':
            return vr_gpu, w_gpu, vl_gpu
        elif jobvl == 'V' and jobvr == 'N':
            return w_gpu, vl_gpu,
        elif jobvl == 'N' and jobvr == 'V':
            return vr_gpu, w_gpu
    elif lib == 'cusolver':
        if data_type in (np.float32,np.complex64):
            eigv_data_type = np.float32
        elif data_type in ( np.float64, np.complex128):
            eigv_data_type = np.float64
        w_gpu = gpuarray.empty(m, eigv_data_type, order="F", allocator=alloc)
        if jobvl == 'V':
            raise NotImplementedError('CUSOLVER supports only right eigenvectors')

        if jobvr == 'V':
            jobz = cusolver._CUSOLVER_EIG_MODE['CUSOLVER_EIG_MODE_VECTOR']
            # Copy a_gpu, so we don't destroy it
            a_copy_gpu = a_gpu.copy()
        else:
            jobz = cusolver._CUSOLVER_EIG_MODE['CUSOLVER_EIG_MODE_NOVECTOR']
            a_copy_gpu = a_gpu

        # Since we have the full matrix and assuming symmetry, fill mode
        # hopefully doesn't matter
        uplo = cublas._CUBLAS_FILL_MODE[0]

        Lwork = bufsize(
            cusolverHandle,
            jobz,
            uplo,
            n,
            a_copy_gpu.gpudata,
            m,
            w_gpu.gpudata,
        )

        Work = gpuarray.empty(Lwork, data_type, allocator=alloc)
        devInfo = gpuarray.empty(1, np.int32, allocator=alloc)
        func(cusolverHandle, jobz, uplo,
             n, a_copy_gpu.gpudata, m, w_gpu.gpudata,
             Work.gpudata, Lwork, devInfo.gpudata)

        if jobz == cusolver._CUSOLVER_EIG_MODE['CUSOLVER_EIG_MODE_VECTOR']:
            return a_copy_gpu, w_gpu
        else:
            return w_gpu
    else:
        raise ValueError('invalid library specified')


@context_dependent_memoize
def _get_vander_kernel(use_double, use_complex, rows, cols):
     template = Template("""
     #include <pycuda-complex.hpp>
     #if ${use_double}
     #if ${use_complex}
     #define FLOAT pycuda::complex<double>
     #else
     #define FLOAT double
     #endif
     #else
     #if ${use_complex}
     #define FLOAT pycuda::complex<float>
     #else
     #define FLOAT float
     #endif
     #endif


     __global__ void vander(FLOAT *a, FLOAT *b, int m, int n) {

	  unsigned int ix;
	  unsigned int r = blockIdx.x*blockDim.x+threadIdx.x;

	  if(r < m) {
	    for(int i=1; i<n; ++i) {
	       ix = r  + m*i  ;
	       a[ix] = a[r  + m*(i-1)] * b[r];
            }
          }
     }
     """)
     # Set this to False when debugging to make sure the compiled kernel is
     # not cached:
     cache_dir=None
     tmpl = template.substitute(use_double=use_double,
                                use_complex=use_complex,
                                rows=rows,
                                cols=cols)
     mod = SourceModule(tmpl, cache_dir=cache_dir)
     return mod.get_function("vander")


def vander(a_gpu, n=None, handle=None):
     """
     Generate a Vandermonde matrix.

     A Vandermonde matrix (named for Alexandre- Theophile Vandermonde)
     is a  matrix where the columns are powers of the input vector, i.e.,
     the `i-th` column is the input vector raised element-wise to the
     power of `i`.

     Parameters
     ----------
     a_gpu : pycuda.gpuarray.GPUArray
         Real/complex 1-D input array of shape `(m, 1)`.

     n : int, optional
        Number of columns in the Vandermonde matrix.
        If `n` is not specified, a square array is returned `(m,m)`.

     Returns
     -------
     vander_gpu : pycuda.gpuarray
         Vandermonde matrix of shape `(m,n)`.

     Examples
     --------
     >>> import pycuda.autoinit
     >>> import pycuda.gpuarray as gpuarray
     >>> import numpy as np
     >>> import skcuda.linalg as linalg
     >>> a = np.array(np.array([1, 2, 3]), np.float32, order='F')
     >>> a_gpu = gpuarray.to_gpu(a)
     >>> v_gpu = linalg.vander(a_gpu, n=4)
     >>> np.allclose(v_gpu.get(), np.fliplr(np.vander(a, 4)), atol=1e-6)
     True
     """

     if handle is None:
         handle = misc._global_cublas_handle

     alloc = misc._global_cublas_allocator

     data_type = a_gpu.dtype.type
     if a_gpu.dtype == np.float32:
         use_double = 0
         use_complex = 0
     elif a_gpu.dtype == np.float64:
         use_double = 1
         use_complex = 0
     elif a_gpu.dtype == np.complex64:
         use_double = 0
         use_complex = 1
     elif a_gpu.dtype == np.complex128:
         use_double = 1
         use_complex = 1
     else:
         raise ValueError('unrecognized type')

     m = a_gpu.shape[0]
     if n == None: n = m

     vander_gpu = gpuarray.empty((m, n), data_type, order='F', allocator=alloc)
     vander_gpu[ : , 0 ] = vander_gpu[ : , 0 ] * 0  + 1

     # Get block/grid sizes:
     dev = misc.get_current_device()
     block_dim, grid_dim = misc.select_block_grid_sizes(dev, vander_gpu.shape)

     # Allocate Vandermonde matrix:
     vander = _get_vander_kernel(use_double, use_complex, rows=m, cols=n)

     # Call kernel:
     vander(vander_gpu, a_gpu,
            np.uint32(m), np.uint32(n),
            block=block_dim,
            grid=grid_dim)

     # Return
     return vander_gpu


def dmd(a_gpu, k=None, modes='exact', return_amplitudes=False, return_vandermonde=False, handle=None):
    """
    Dynamic Mode Decomposition.

    Dynamic Mode Decomposition (DMD) is a data processing algorithm which
    allows to decompose a matrix `a` in space and time.
    The matrix `a` is decomposed as `a = FBV`, where the columns of `F`
    contain the dynamic modes. The modes are ordered corresponding
    to the amplitudes stored in the diagonal matrix `B`. `V` is a Vandermonde
    matrix describing the temporal evolution.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Real/complex input matrix  `a` with dimensions `(m, n)`.
    k : int, optional
        If `k < (n-1)` low-rank Dynamic Mode Decomposition is computed.
    modes : `{'standard', 'exact'}`
        'standard' : uses the standard definition to compute the dynamic modes,
                    `F = U * W`.
        'exact' : computes the exact dynamic modes, `F = Y * V * (S**-1) * W`.
    return_amplitudes : bool `{True, False}`
        True: return amplitudes in addition to dynamic modes.
    return_vandermonde : bool `{True, False}`
        True: return Vandermonde matrix in addition to dynamic modes and amplitudes.
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.

    Returns
    -------
    f_gpu : pycuda.gpuarray.GPUArray
        Matrix containing the dynamic modes of shape `(m, n-1)`  or `(m, k)`.
    b_gpu : pycuda.gpuarray.GPUArray
        1-D array containing the amplitudes of length `min(n-1, k)`.
    v_gpu : pycuda.gpuarray.GPUArray
        Vandermonde matrix of shape `(n-1, n-1)`  or `(k, n-1)`.

    Notes
    -----
    Double precision is only supported if the standard version of the
    CULA Dense toolkit is installed.

    This function destroys the contents of the input matrix.

    Arrays are assumed to be stored in column-major order, i.e., order='F'.

    References
    ----------
    M. R. Jovanovic, P. J. Schmid, and J. W. Nichols.
    "Low-rank and sparse dynamic mode decomposition."
    Center for Turbulence Research Annual Research Briefs (2012): 139-152.

    J. H. Tu, et al.
    "On dynamic mode decomposition: theory and applications."
    arXiv preprint arXiv:1312.0041 (2013).


     Examples
     --------
     >>> #Numpy
     >>> import numpy as np
     >>> #Plot libs
     >>> import matplotlib.pyplot as plt
     >>> from mpl_toolkits.mplot3d import Axes3D
     >>> from matplotlib import cm
     >>> #GPU DMD libs
     >>> import pycuda.gpuarray as gpuarray
     >>> import pycuda.autoinit
     >>> from skcuda import linalg, rlinalg
     >>> linalg.init()

     >>> # Define time and space discretizations
     >>> x=np.linspace( -15, 15, 200)
     >>> t=np.linspace(0, 8*np.pi , 80)
     >>> dt=t[2]-t[1]
     >>> X, T = np.meshgrid(x,t)
     >>> # Create two patio-temporal patterns
     >>> F1 = 0.5* np.cos(X)*(1.+0.* T)
     >>> F2 = ( (1./np.cosh(X)) * np.tanh(X)) *(2.*np.exp(1j*2.8*T))
     >>> # Add both signals
     >>> F = (F1+F2)

     >>> #Plot dataset
     >>> fig = plt.figure()
     >>> ax = fig.add_subplot(231, projection='3d')
     >>> ax = fig.gca(projection='3d')
     >>> surf = ax.plot_surface(X, T, F, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
     >>> ax.set_zlim(-1, 1)
     >>> plt.title('F')
     >>> ax = fig.add_subplot(232, projection='3d')
     >>> ax = fig.gca(projection='3d')
     >>> surf = ax.plot_surface(X, T, F1, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
     >>> ax.set_zlim(-1, 1)
     >>> plt.title('F1')
     >>> ax = fig.add_subplot(233, projection='3d')
     >>> ax = fig.gca(projection='3d')
     >>> surf = ax.plot_surface(X, T, F2, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
     >>> ax.set_zlim(-1, 1)
     >>> plt.title('F2')

     >>> #Dynamic Mode Decomposition
     >>> F_gpu = np.array(F.T, np.complex64, order='F')
     >>> F_gpu = gpuarray.to_gpu(F_gpu)
     >>> Fmodes_gpu, b_gpu, V_gpu, omega_gpu = linalg.dmd(F_gpu, k=2, modes='exact', return_amplitudes=True, return_vandermonde=True)
     >>> omega = omega_gpu.get()
     >>> plt.scatter(omega.real, omega.imag, marker='o', c='r')


     >>> #Recover original signal
     >>> F1tilde = np.dot(Fmodes_gpu[:,0:1].get() , np.dot(b_gpu[0].get(), V_gpu[0:1,:].get() ) )
     >>> F2tilde = np.dot(Fmodes_gpu[:,1:2].get() , np.dot(b_gpu[1].get(), V_gpu[1:2,:].get() ) )

     >>> #Plot DMD modes
     >>> #Mode 0
     >>> ax = fig.add_subplot(235, projection='3d')
     >>> ax = fig.gca(projection='3d')
     >>> surf = ax.plot_surface(X[0:F1tilde.shape[1],:], T[0:F1tilde.shape[1],:], F1tilde.T, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
     >>> ax.set_zlim(-1, 1)
     >>> plt.title('F1_tilde')
     >>> #Mode 1
     >>> ax = fig.add_subplot(236, projection='3d')
     >>> ax = fig.gca(projection='3d')
     >>> surf = ax.plot_surface(X[0:F2tilde.shape[1],:], T[0:F2tilde.shape[1],:], F2tilde.T, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
     >>> ax.set_zlim(-1, 1)
     >>> plt.title('F2_tilde')
     >>> plt.show()
    """

    #*************************************************************************
    #***        Author: N. Benjamin Erichson <nbe@st-andrews.ac.uk>        ***
    #***                              <2015>                               ***
    #***                       License: BSD 3 clause                       ***
    #*************************************************************************

    if not _has_cula:
        raise NotImplementedError('CULA not installed')

    if handle is None:
        handle = misc._global_cublas_handle

    alloc = misc._global_cublas_allocator

    # The free version of CULA only supports single precision floating
    data_type = a_gpu.dtype.type
    real_type = np.float32

    if data_type == np.complex64:
        cula_func_gesvd = cula.culaDeviceCgesvd
        cublas_func_gemm = cublas.cublasCgemm
        cublas_func_dgmm = cublas.cublasCdgmm
        cula_func_gels = cula.culaDeviceCgels
        copy_func = cublas.cublasCcopy
        transpose_func = cublas.cublasCgeam
        alpha = np.complex64(1.0)
        beta = np.complex64(0.0)
        TRANS_type = 'C'
        isreal = False
    elif data_type == np.float32:
        cula_func_gesvd = cula.culaDeviceSgesvd
        cublas_func_gemm = cublas.cublasSgemm
        cublas_func_dgmm = cublas.cublasSdgmm
        cula_func_gels = cula.culaDeviceSgels
        copy_func = cublas.cublasScopy
        transpose_func = cublas.cublasSgeam
        alpha = np.float32(1.0)
        beta = np.float32(0.0)
        TRANS_type = 'T'
        isreal = True
    else:
        if cula._libcula_toolkit == 'standard':
            if data_type == np.complex128:
                cula_func_gesvd = cula.culaDeviceZgesvd
                cublas_func_gemm = cublas.cublasZgemm
                cublas_func_dgmm = cublas.cublasZdgmm
                cula_func_gels = cula.culaDeviceZgels
                copy_func = cublas.cublasZcopy
                transpose_func = cublas.cublasZgeam
                alpha = np.complex128(1.0)
                beta = np.complex128(0.0)
                TRANS_type = 'C'
                isreal = False
            elif data_type == np.float64:
                cula_func_gesvd = cula.culaDeviceDgesvd
                cublas_func_gemm = cublas.cublasDgemm
                cublas_func_dgmm = cublas.cublasDdgmm
                cula_func_gels = cula.culaDeviceDgels
                copy_func = cublas.cublasDcopy
                transpose_func = cublas.cublasDgeam
                alpha = np.float64(1.0)
                beta = np.float64(0.0)
                TRANS_type = 'T'
                isreal = True
            else:
                raise ValueError('unsupported type')
            real_type = np.float64
        else:
            raise ValueError('double precision not supported')

    #CUDA assumes that arrays are stored in column-major order
    m, n = a_gpu.shape
    nx = n-1
    #Set k
    if k == None : k = nx
    if k > nx or k < 1: raise ValueError('k is not valid')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Split data into lef and right snapshot sequence
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Note: we need a copy of X_gpu, because SVD destroys X_gpu
    #While Y_gpu is just a pointer
    X_gpu = gpuarray.empty((m, n), data_type, order="F", allocator=alloc)
    copy_func(handle, X_gpu.size, int(a_gpu.gpudata), 1, int(X_gpu.gpudata), 1)
    X_gpu = X_gpu[:, :nx]
    Y_gpu = a_gpu[:, 1:]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Singular Value Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #gesvd(jobu, jobvt, m, n, int(a), lda, int(s), int(u), ldu, int(vt), ldvt)
    #Parameters
    #----------
    #a : pycuda.gpuarray.GPUArray of shape (m, n)
    #jobu : {'A', 'S', 'O', 'N'}
    #    If 'A', return the full `u` matrix with shape `(m, m)`.
    #    If 'S', return the `u` matrix with shape `(m, nx)`.
    #    If 'O', return the `u` matrix with shape `(m, nx) without
    #    allocating a new matrix.
    #jobvt : {'A', 'S', 'O', 'N'}
    #    If 'A', return the full `vh` matrix with shape `(nx, nx)`.
    #    If 'S', return the `vh` matrix with shape `(nx, nx)`.
    #    If 'O', return the `vh` matrix with shape `(nx, nx) without
    #    allocating a new matrix.
    #
    #Returns
    #-------
    #u : pycuda.gpuarray.GPUArray
    #    Unitary matrix of shape `(m, m)` or `(m, nx)`
    #s : pycuda.gpuarray.GPUArray
    #    Array containing the singular values, sorted such that `s[i] >= s[i+1]`.
    #    `s` is of length `min(m, nx)`.
    #v : pycuda.gpuarray.GPUArray
    #    Unitary matrix of shape `(nx, nx)` or `(nx, nx)`

    #Allocate s, U, Vt for economic SVD
    #Note: singular values are always real
    #Allocate s, U, Vt for economic SVD
    #Note: singular values are always real
    s_gpu = gpuarray.empty(nx, real_type, order="F", allocator=alloc)
    U_gpu = gpuarray.empty((m,nx), data_type, order="F", allocator=alloc)
    Vh_gpu = gpuarray.empty((nx,nx), data_type, order="F", allocator=alloc)

    #Economic SVD
    cula_func_gesvd('S', 'S', m, nx, int(X_gpu.gpudata), m, int(s_gpu.gpudata),
                    int(U_gpu.gpudata), m, int(Vh_gpu.gpudata), nx)

    #Low-rank DMD: trancate SVD if k < nx

    if k != nx:
        s_gpu = s_gpu[:k]
        U_gpu = U_gpu[: , :k]
        #Vt_gpu = Vt_gpu[:k , : ]
        Vh_gpu = Vh_gpu[:k , : ]


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Solve the LS problem to find estimate for M using the pseudo-inverse
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #real: M = U.T * Y * Vt.T * S**-1
    #complex: M = U.H * Y * Vt.H * S**-1
    #Let G = Y * Vt.H * S**-1, hence M = M * G

    #Allocate G and M
    G_gpu = gpuarray.empty((m,k), data_type, order="F", allocator=alloc)
    M_gpu = gpuarray.empty((k,k), data_type, order="F", allocator=alloc)

    #i) s = s **-1 (inverse)
    if data_type == np.complex64 or data_type == np.complex128:
        s_gpu = 1/s_gpu
        s_gpu = s_gpu + 1j * gpuarray.zeros_like(s_gpu)
    else:
        s_gpu = 1.0/s_gpu


    #ii) real/complex: scale Vs =  Vt* x diag(s**-1)
    Vs_gpu = gpuarray.empty((nx,k), data_type, order="F", allocator=alloc)
    lda = max(1, Vh_gpu.strides[1] // Vh_gpu.dtype.itemsize)
    ldb = max(1, Vs_gpu.strides[1] // Vs_gpu.dtype.itemsize)
    transpose_func(handle, TRANS_type, TRANS_type, nx, k,
                   alpha, int(Vh_gpu.gpudata), lda, beta, int(Vh_gpu.gpudata), lda,
                   int(Vs_gpu.gpudata), ldb)


    cublas_func_dgmm(handle, 'r', nx, k, int(Vs_gpu.gpudata), nx,
                     int(s_gpu.gpudata), 1 , int(Vs_gpu.gpudata), nx)


    #iii) real: G = Y * Vs , complex: G = Y x Vs
    cublas_func_gemm(handle, 'n', 'n', m, k, nx, alpha,
                     int(Y_gpu.gpudata), m, int(Vs_gpu.gpudata), nx,
                        beta, int(G_gpu.gpudata), m )


    #iv) real/complex: M = U* x G
    cublas_func_gemm(handle, TRANS_type, 'n', k, k, m, alpha,
                     int(U_gpu.gpudata), m, int(G_gpu.gpudata), m,
                    beta, int(M_gpu.gpudata), k )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Eigen Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Note: If a_gpu is real the imag part is omitted
    Vr_gpu, w_gpu = eig(M_gpu, 'N', 'V', 'F')
    omega = cumath.log(w_gpu)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute DMD Modes
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    F_gpu = gpuarray.empty((m,k), data_type, order="F", allocator=alloc)
    modes = modes.lower()
    if modes == 'exact': #Compute (exact) DMD modes: F = Y * V * S**-1 * W = G * W
        cublas_func_gemm(handle, 'n', 'n', m, k, k, alpha,
                         G_gpu.gpudata, m, Vr_gpu.gpudata, k,
                         beta, G_gpu.gpudata, m  )
        F_gpu_temp = G_gpu

    elif modes == 'standard': #Compute (standard) DMD modes: F = U * W
        cublas_func_gemm(handle, 'n', 'n', m, k, k,
                         alpha, U_gpu.gpudata, m, Vr_gpu.gpudata, k,
                         beta, U_gpu.gpudata, m  )
        F_gpu_temp = U_gpu
    else:
        raise ValueError('Type of modes is not supported, choose "exact" or "standard".')

    #Copy is required, because gels destroys input
    copy_func(handle, F_gpu_temp.size, int(F_gpu_temp.gpudata),
              1, int(F_gpu.gpudata), 1)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute amplitueds b using least-squares: Fb=x1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if return_amplitudes==True:
        #x1_gpu = a_gpu[:,0].copy()
        x1_gpu = gpuarray.empty(m, data_type, order="F", allocator=alloc)
        copy_func(handle, x1_gpu.size, int(a_gpu[:,0].gpudata), 1, int(x1_gpu.gpudata), 1)
        cula_func_gels( 'N', m, k, int(1) , F_gpu_temp.gpudata, m, x1_gpu.gpudata, m)
        b_gpu = x1_gpu

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute Vandermonde matrix (CPU)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if return_vandermonde==True:
        V_gpu = vander(w_gpu, n=nx)

    # Free internal CULA memory:
    cula.culaFreeBuffers()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Return
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if return_amplitudes==True and return_vandermonde==True:
        return F_gpu, b_gpu[:k], V_gpu, omega
    elif return_amplitudes==True and return_vandermonde==False:
        return F_gpu, b_gpu[:k], omega
    elif return_amplitudes==False and return_vandermonde==True:
        return F_gpu, V_gpu, omega
    else:
        return F_gpu, omega


if __name__ == "__main__":
    import doctest
    doctest.testmod()
