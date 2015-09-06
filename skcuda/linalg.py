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

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.elementwise as el
import pycuda.tools as tools
import numpy as np

from . import cublas
from . import misc

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

from .misc import init, add_matvec, div_matvec, mult_matvec

# Get installation location of C headers:
from . import install_headers

def svd(a_gpu, jobu='A', jobvt='A'):
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
    Double precision is only supported if the standard version of the
    CULA Dense toolkit is installed.

    This function destroys the contents of the input matrix regardless
    of the values of `jobu` and `jobvt`.

    Only one of `jobu` or `jobvt` may be set to `O`, and then only for
    a square matrix.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import linalg
    >>> linalg.init()
    >>> a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
    >>> a = np.asarray(a, np.complex64)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 'S', 'S')
    >>> np.allclose(a, np.dot(u_gpu.get(), np.dot(np.diag(s_gpu.get()), vh_gpu.get())), 1e-4)
    True

    """

    if not _has_cula:
        raise NotImplementedError('CULA not installed')

    alloc = misc._global_cublas_allocator

    # The free version of CULA only supports single precision floating
    # point numbers:
    data_type = a_gpu.dtype.type
    real_type = np.float32
    if data_type == np.complex64:
        cula_func = cula.culaDeviceCgesvd
    elif data_type == np.float32:
        cula_func = cula.culaDeviceSgesvd
    else:
        if cula._libcula_toolkit == 'standard':
            if data_type == np.complex128:
                cula_func = cula.culaDeviceZgesvd
            elif data_type == np.float64:
                cula_func = cula.culaDeviceDgesvd
            else:
                raise ValueError('unsupported type')
            real_type = np.float64
        else:
            raise ValueError('double precision not supported')

    # Since CUDA assumes that arrays are stored in column-major
    # format, the input matrix is assumed to be transposed:
    n, m = a_gpu.shape
    square = (n == m)

    # Since the input matrix is transposed, jobu and jobvt must also
    # be switched because the computed matrices will be returned in
    # reversed order:
    jobvt, jobu = jobu, jobvt

    # Set the leading dimension of the input matrix:
    lda = max(1, m)

    # Allocate the array of singular values:
    s_gpu = gpuarray.empty(min(m, n), real_type, allocator=alloc)

    # Set the leading dimension and allocate u:
    jobu = jobu.upper()
    jobvt = jobvt.upper()
    ldu = m
    if jobu == 'A':
        u_gpu = gpuarray.empty((ldu, m), data_type, allocator=alloc)
    elif jobu == 'S':
        u_gpu = gpuarray.empty((min(m, n), ldu), data_type, allocator=alloc)
    elif jobu == 'O':
        if not square:
            raise ValueError('in-place computation of singular vectors '+
                             'of non-square matrix not allowed')
        ldu = 1
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
        ldvt = 1
        vh_gpu = a_gpu
    else:
        ldvt = 1
        vh_gpu = gpuarray.empty((), data_type, allocator=alloc)

    # Compute SVD and check error status:

    cula_func(jobu, jobvt, m, n, int(a_gpu.gpudata),
              lda, int(s_gpu.gpudata), int(u_gpu.gpudata),
              ldu, int(vh_gpu.gpudata), ldvt)

    # Free internal CULA memory:
    cula.culaFreeBuffers()

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


def cho_factor(a_gpu, uplo='L'):
    """
    Cholesky factorisation

    Performs an in-place cholesky factorisation on the matrix `a`
    such that `a = x*x.T` or `x.T*x`, if the lower='L' or upper='U'
    triangle of `a` is used, respectively.

    Parameters
    ----------
    a : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, m)` to decompose.
    uplo: use the upper='U' or lower='L' (default) triangle of 'a'

    Returns
    -------
    a: pycuda.gpuarray.GPUArray
        Cholesky factorised matrix

    Notes
    -----
    Double precision is only supported if the standard version of the
    CULA Dense toolkit is installed.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import scipy.linalg
    >>> import linalg
    >>> linalg.init()
    >>> a = np.array([[3.0,0.0],[0.0,7.0]])
    >>> a = np.asarray(a, np.float64)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> cho_factor(a_gpu)
    >>> np.allclose(a_gpu.get(), scipy.linalg.cho_factor(a)[0])
    True

    """

    if not _has_cula:
        raise NotImplementError('CULA not installed')

    data_type = a_gpu.dtype.type
    real_type = np.float32
    if cula._libcula_toolkit == 'standard':
        if data_type == np.complex64:
            cula_func = cula.culaDeviceCpotrf
        elif data_type == np.float32:
            cula_func = cula.culaDeviceSpotrf
        elif data_type == np.complex128:
            cula_func = cula.culaDeviceZpotrf
        elif data_type == np.float64:
            cula_func = cula.culaDeviceDpotrf
        else:
            raise ValueError('unsupported type')
        real_type = np.float64
    else:
        raise ValueError('Cholesky factorisation not included in CULA Dense Free version')

    # Since CUDA assumes that arrays are stored in column-major
    # format, the input matrix is assumed to be transposed:
    n, m = a_gpu.shape
    if (n!=m):
        raise ValueError('Matrix must be symmetric positive-definite')

    # Set the leading dimension of the input matrix:
    lda = max(1, m)

    cula_func(uplo, n, int(a_gpu.gpudata), lda)

    # Free internal CULA memory:
    cula.culaFreeBuffers()

    # In-place operation. No return matrix. Result is stored in the input matrix.


def cho_solve(a_gpu, b_gpu, uplo='L'):
    """
    Cholesky solver

    Solve a system of equations via cholesky factorization,
    i.e. `a*x = b`.
    Overwrites `b` to give `inv(a)*b`, and overwrites the chosen triangle
    of `a` with factorized triangle

    Parameters
    ----------
    a : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, m)` to decompose.
    b : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, 1)` to decompose.
    uplo: chr
        use the upper='U' or lower='L' (default) triangle of `a`.

    Returns
    -------
    a: pycuda.gpuarray.GPUArray
        Cholesky factorised matrix

    Notes
    -----
    Double precision is only supported if the standard version of the
    CULA Dense toolkit is installed.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import scipy.linalg
    >>> import linalg
    >>> linalg.init()
    >>> a = np.array([[3.0,0.0],[0.0,7.0]])
    >>> a = np.asarray(a, np.float64)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> b = np.array([11.,19.])
    >>> b = np.asarray(b, np.float64)
    >>> b_gpu  = gpuarray.to_gpu(b)
    >>> cho_solve(a_gpu,b_gpu)
    >>> np.allclose(b_gpu.get(), scipy.linalg.cho_solve(scipy.linalg.cho_factor(a), b))
    True

    """

    if not _has_cula:
        raise NotImplementError('CULA not installed')

    data_type = a_gpu.dtype.type
    real_type = np.float32
    if cula._libcula_toolkit == 'standard':
        if data_type == np.complex64:
            cula_func = cula.culaDeviceCposv
        elif data_type == np.float32:
            cula_func = cula.culaDeviceSposv
        elif data_type == np.complex128:
            cula_func = cula.culaDeviceZposv
        elif data_type == np.float64:
            cula_func = cula.culaDeviceDposv
        else:
            raise ValueError('unsupported type')
        real_type = np.float64
    else:
        raise ValueError('Cholesky factorisation not included in CULA Dense Free version')

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

    # Assuming we are only solving for a vector. Hence, nrhs = 1
    cula_func(uplo, na, b_shape[1], int(a_gpu.gpudata), lda,
              int(b_gpu.gpudata), ldb)

    # Free internal CULA memory:
    cula.culaFreeBuffers()

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
        Cummulative array.
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
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import linalg
    >>> import misc
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
    if len(x_shape) == 1:
        x_shape = (1, x_shape[0])
    if len(y_shape) == 1:
        y_shape = (1, y_shape[0])

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

    return add_dot(x_gpu, y_gpu, out, transa, transb, 1.0, 0.0, handle)


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
    >>> import linalg
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
    >>> import linalg
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
    >>> import linalg
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
    >>> import linalg
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
    >>> import linalg
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

    try:
        func = conj.cache[x_gpu.dtype]
    except KeyError:
        ctype = tools.dtype_to_ctype(x_gpu.dtype)
        func = el.ElementwiseKernel(
                "{ctype} *x, {ctype} *y".format(ctype=ctype),
                "y[i] = conj(x[i])")
        conj.cache[x_gpu.dtype] = func
    if overwrite:
        func(x_gpu, x_gpu)
        return x_gpu
    else:
        y_gpu = gpuarray.empty_like(x_gpu)
        func(x_gpu, y_gpu)
        return y_gpu
conj.cache = {}


@context_dependent_memoize
def _get_diag_kernel(use_double, use_complex):
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

    // Assumes that d already contains zeros in all positions.
    // N must contain the number of elements in v.
    __global__ void diag(FLOAT *v, FLOAT *d, int N) {
        unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                           blockIdx.x*blockDim.x+threadIdx.x;
        if (idx < N)
            d[idx*(N+1)] = v[idx];
    }
    """)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    tmpl = template.substitute(use_double=use_double, use_complex=use_complex)
    cache_dir=None
    diag_mod = SourceModule(tmpl, cache_dir=cache_dir)
    return diag_mod.get_function("diag")


def diag(v_gpu):
    """
    Construct a diagonal matrix if input array is one-dimensional,
    or extracts diagonal entries of a two-dimensional array.

    If input-array is one-dimensional: Constructs a matrix in device
    memory whose diagonal elements correspond to the elements in the
    specified array; all non-diagonal elements are set to 0.

    If input-array is two-dimensional: Constructs an array in device memory
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
    >>> import linalg
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
        incx = v_gpu.shape[1]+1

        # Allocate the output array
        d_gpu = gpuarray.empty(n, v_gpu.dtype.type, allocator=alloc)

        handle = misc._global_cublas_handle
        func(handle, n, v_gpu.gpudata, incx, d_gpu.gpudata, 1)
        return d_gpu
    elif len(v_gpu.shape) >= 3:
        raise ValueError('input array cannot have greater than 2-dimensions')

    use_double = int(v_gpu.dtype in [np.float64, np.complex128])
    use_complex = int(v_gpu.dtype in [np.complex64, np.complex128])

    # Initialize output matrix:
    d_gpu = misc.zeros((v_gpu.size, v_gpu.size), v_gpu.dtype, allocator=alloc)

    # Get block/grid sizes:
    dev = misc.get_current_device()
    block_dim, grid_dim = misc.select_block_grid_sizes(dev, d_gpu.shape)

    diag = _get_diag_kernel(use_double, use_complex)
    diag(v_gpu, d_gpu, np.uint32(v_gpu.size),
         block=block_dim,
         grid=grid_dim)

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
    >>> import linalg
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

def pinv(a_gpu, rcond=1e-15):
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
    >>> import linalg
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

    """

    if not _has_cula:
        raise NotImplementedError('CULA not installed')

    # Perform in-place SVD if the matrix is square to save memory:
    if a_gpu.shape[0] == a_gpu.shape[1]:
        u_gpu, s_gpu, vh_gpu = svd(a_gpu, 's', 'o')
    else:
        u_gpu, s_gpu, vh_gpu = svd(a_gpu, 's', 's')

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
    >>> import pycuda.driver as drv
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import linalg
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

def multiply(x_gpu, y_gpu, overwrite=False):
    """
    Multiply arguments element-wise.

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
    >>> import linalg
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
    >>> import linalg
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
    >>> import linalg
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

def inv(a_gpu, overwrite=False, ipiv_gpu=None):
    """
    Compute the inverse of a matrix.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Square (n, n) matrix to be inverted.
    overwrite : bool (default: False)
        Discard data in `a` (may improve performance).
    ipiv_gpu : pycuda.gpuarray.GPUArray (optional)
        Temporary array of size n, can be supplied to save allocations.

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
    """
    if len(a_gpu.shape) != 2 or a_gpu.shape[0] != a_gpu.shape[1]:
        raise ValueError('expected square matrix')

    if (a_gpu.dtype == np.complex64):
        getrf = cula.culaDeviceCgetrf
        getri = cula.culaDeviceCgetri
    elif (a_gpu.dtype == np.float32):
        getrf = cula.culaDeviceSgetrf
        getri = cula.culaDeviceSgetri
    elif (a_gpu.dtype == np.complex128):
        getrf = cula.culaDeviceZgetrf
        getri = cula.culaDeviceZgetri
    elif (a_gpu.dtype == np.float64):
        getrf = cula.culaDeviceDgetrf
        getri = cula.culaDeviceDgetri

    n = a_gpu.shape[0]
    if ipiv_gpu is None:
        alloc = misc._global_cublas_allocator
        ipiv_gpu = gpuarray.empty((n, 1), np.int32, allocator=alloc)
    elif ipiv_gpu.dtype != np.int32 or np.prod(ipiv_gpu.shape) < n:
        raise ValueError('invalid ipiv provided')

    out = a_gpu if overwrite else a_gpu.copy()
    try:
        getrf(n, n, out.gpudata, n, ipiv_gpu.gpudata)
        getri(n, out.gpudata, n, ipiv_gpu.gpudata)
    except cula.culaDataError as e:
        raise LinAlgError(e)
    return out


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

def det(a_gpu, overwrite=False, ipiv_gpu=None, handle=None):
    """
    Compute the determinant of a square matrix.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        The square n*n matrix of which to calculate the determinant.
    overwrite : bool (default: False)
        Discard data in `a` (may improve performance).
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.
    ipiv_gpu : pycuda.gpuarray.GPUArray (optional)
        Temporary array of size n, can be supplied to save allocations.

    Returns
    -------
    det : number
        determinant of a_gpu
    """

    if handle is None:
        handle = misc._global_cublas_handle

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
    if ipiv_gpu is None:
        alloc = misc._global_cublas_allocator
        ipiv_gpu = gpuarray.empty((n, 1), np.int32, allocator=alloc)
    elif ipiv_gpu.dtype != np.int32 or np.prod(ipiv_gpu.shape) < n:
        raise ValueError('invalid ipiv provided')

    out = a_gpu if overwrite else a_gpu.copy()
    try:
        getrf(n, n, out.gpudata, n, ipiv_gpu.gpudata)
        return _get_det_kernel(a_gpu.dtype)(ipiv_gpu, out, n).get()
    except cula.culaDataError as e:
        raise LinAlgError(e)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
