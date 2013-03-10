#!/usr/bin/env python

"""
PyCUDA-based linear algebra functions.
"""

from pprint import pprint
from string import Template, lower, upper
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np

import cuda
import cublas
import misc

try:
    import cula
    _has_cula = True
except (ImportError, OSError):
    _has_cula = False

from misc import init

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
        raise NotImplementError('CULA not installed')

    # The free version of CULA only supports single precision floating
    # point numbers:
    data_type = a_gpu.dtype.type
    real_type = np.float32
    if data_type == np.complex64:
        cula_func = cula._libcula.culaDeviceCgesvd
    elif data_type == np.float32:
        cula_func = cula._libcula.culaDeviceSgesvd
    else:
        if cula._libcula_toolkit == 'standard':
            if data_type == np.complex128:
                cula_func = cula._libcula.culaDeviceZgesvd
            elif data_type == np.float64:
                cula_func = cula._libcula.culaDeviceDgesvd
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
    s_gpu = gpuarray.empty(min(m, n), real_type)

    # Set the leading dimension and allocate u:
    jobu = upper(jobu)
    jobvt = upper(jobvt)
    ldu = m
    if jobu == 'A':
        u_gpu = gpuarray.empty((ldu, m), data_type)
    elif jobu == 'S':
        u_gpu = gpuarray.empty((min(m, n), ldu), data_type)
    elif jobu == 'O':
        if not square:
            raise ValueError('in-place computation of singular vectors '+
                             'of non-square matrix not allowed')
        ldu = 1
        u_gpu = a_gpu
    else:
        ldu = 1
        u_gpu = gpuarray.empty((), data_type)

    # Set the leading dimension and allocate vh:
    if jobvt == 'A':
        ldvt = n
        vh_gpu = gpuarray.empty((n, n), data_type)
    elif jobvt == 'S':
        ldvt = min(m, n)
        vh_gpu = gpuarray.empty((n, ldvt), data_type)
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
        vh_gpu = gpuarray.empty((), data_type)

    # Compute SVD and check error status:

    status = cula_func(jobu, jobvt, m, n, int(a_gpu.gpudata),
                       lda, int(s_gpu.gpudata), int(u_gpu.gpudata),
                       ldu, int(vh_gpu.gpudata), ldvt)

    cula.culaCheckStatus(status)

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

def dot(x_gpu, y_gpu, transa='N', transb='N', handle=None):
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
        `scikits.misc._global_cublas_handle` is used.
        
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

        # Get the shapes of the arguments (accounting for the
        # possibility that one of them may only have one dimension):
        x_shape = x_gpu.shape
        y_shape = y_gpu.shape
        if len(x_shape) == 1:
            x_shape = (1, x_shape[0])
        if len(y_shape) == 1:
            y_shape = (1, y_shape[0])

        # Perform matrix multiplication for 2D arrays:
        if (x_gpu.dtype == np.complex64 and y_gpu.dtype == np.complex64):
            cublas_func = cublas.cublasCgemm
            alpha = np.complex64(1.0)
            beta = np.complex64(0.0)
        elif (x_gpu.dtype == np.float32 and y_gpu.dtype == np.float32):
            cublas_func = cublas.cublasSgemm
            alpha = np.float32(1.0)
            beta = np.float32(0.0)
        elif (x_gpu.dtype == np.complex128 and y_gpu.dtype == np.complex128):
            cublas_func = cublas.cublasZgemm
            alpha = np.complex128(1.0)
            beta = np.complex128(0.0)
        elif (x_gpu.dtype == np.float64 and y_gpu.dtype == np.float64):
            cublas_func = cublas.cublasDgemm
            alpha = np.float64(1.0)
            beta = np.float64(0.0)
        else:
            raise ValueError('unsupported combination of input types')

        transa = lower(transa)
        transb = lower(transb)

        if transb in ['t', 'c']:
            m, k = y_shape
        elif transb in ['n']:
            k, m = y_shape
        else:
            raise ValueError('invalid value for transb')

        if transa in ['t', 'c']:
            l, n = x_shape
        elif transa in ['n']:
            n, l = x_shape
        else:
            raise ValueError('invalid value for transa')

        if l != k:
            raise ValueError('objects are not aligned')

        if transb == 'n':
            lda = max(1, m)
        else:
            lda = max(1, k)

        if transa == 'n':
            ldb = max(1, k)
        else:
            ldb = max(1, n)

        ldc = max(1, m)

        # Note that the desired shape of the output matrix is the transpose
        # of what CUBLAS assumes:
        c_gpu = gpuarray.empty((n, ldc), x_gpu.dtype)
        cublas_func(handle, transb, transa, m, n, k, alpha, y_gpu.gpudata,
                    lda, x_gpu.gpudata, ldb, beta, c_gpu.gpudata, ldc)

        return c_gpu

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
        `scikits.misc._global_cublas_handle` is used.
        
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

    if kwargs.has_key('handle') and kwargs['handle'] is not None:
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

def dot_diag(d_gpu, a_gpu, trans='N', overwrite=True, handle=None):
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
        Multiplicand array with shape `(N, M)`.
    trans : char
        If 'T', compute the product of the transpose of `a_gpu`.
    overwrite : bool
        If true (default), save the result in `a_gpu`.
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `scikits.misc._global_cublas_handle` is used.

    Returns
    -------
    r_gpu : pycuda.gpuarray.GPUArray
        The computed matrix product.

    Notes
    -----
    `d_gpu` and `a_gpu` must have the same precision data
    type. `d_gpu` may be real and `a_gpu` may be complex, but not
    vice-versa.

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
        
    if len(d_gpu.shape) != 1:
        raise ValueError('d_gpu must be a vector')
    if len(a_gpu.shape) != 2:
        raise ValueError('a_gpu must be a matrix')

    if lower(trans) == 'n':
        rows, cols = a_gpu.shape
    else:
        cols, rows = a_gpu.shape
    N = d_gpu.size
    if N != rows:
        raise ValueError('incompatible dimensions')

    float_type = a_gpu.dtype.type
    if float_type == np.float32:
        if d_gpu.dtype != np.float32:
            raise ValueError('precision of argument types must be the same')
        scal_func = cublas.cublasSscal
        copy_func = cublas.cublasScopy
    elif float_type == np.float64:
        if d_gpu.dtype != np.float64:
            raise ValueError('precision of argument types must be the same')
        scal_func = cublas.cublasDscal
        copy_func = cublas.cublasDcopy
    elif float_type == np.complex64:
        if d_gpu.dtype == np.complex64:
            scal_func = cublas.cublasCscal
        elif d_gpu.dtype == np.float32:
            scal_func = cublas.cublasCsscal
        else:
            raise ValueError('precision of argument types must be the same')
        copy_func = cublas.cublasCcopy
    elif float_type == np.complex128:
        if d_gpu.dtype == np.complex128:
            scal_func = cublas.cublasZscal
        elif d_gpu.dtype == np.float64:
            scal_func = cublas.cublasZdscal
        else:
            raise ValueError('precision of argument types must be the same')
        copy_func = cublas.cublasZcopy
    else:
        raise ValueError('unrecognized type')

    d = d_gpu.get()
    if overwrite:
        r_gpu = a_gpu
    else:
        r_gpu = gpuarray.empty_like(a_gpu)
        copy_func(handle, a_gpu.size, int(a_gpu.gpudata), 1,
                  int(r_gpu.gpudata), 1)

    if lower(trans) == 'n':
        incx = 1
        bytes_step = cols*float_type().itemsize
    else:
        incx = rows
        bytes_step = float_type().itemsize

    for i in xrange(N):
        scal_func(handle, cols, d[i], int(r_gpu.gpudata)+i*bytes_step, incx)
    return r_gpu

transpose_template = Template("""
#include <pycuda-complex.hpp>

#if ${use_double}
#if ${use_complex}
#define FLOAT pycuda::complex<double>
#define CONJ(x) conj(x)
#else
#define FLOAT double
#define CONJ(x) (x)
#endif
#else
#if ${use_complex}
#define FLOAT pycuda::complex<float>
#define CONJ(x) conj(x)
#else
#define FLOAT float
#define CONJ(x) (x)
#endif
#endif

__global__ void transpose(FLOAT *odata, FLOAT *idata, unsigned int N)
{
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int ix = idx/${cols};
    unsigned int iy = idx%${cols};

    if (idx < N)
        if (${hermitian})
            odata[iy*${rows}+ix] = CONJ(idata[ix*${cols}+iy]);
        else
            odata[iy*${rows}+ix] = idata[ix*${cols}+iy];
}
""")

def transpose(a_gpu):
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

    if a_gpu.dtype not in [np.float32, np.float64, np.complex64,
                           np.complex128]:
        raise ValueError('unrecognized type')

    use_double = int(a_gpu.dtype in [np.float64, np.complex128])
    use_complex = int(a_gpu.dtype in [np.complex64, np.complex128])

    # Get block/grid sizes:
    dev = misc.get_current_device()
    block_dim, grid_dim = misc.select_block_grid_sizes(dev, a_gpu.shape)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    transpose_mod = \
                  SourceModule(transpose_template.substitute(use_double=use_double,
                                                             use_complex=use_complex,
                                                             hermitian=0,
                               cols=a_gpu.shape[1],
                               rows=a_gpu.shape[0]),
                               cache_dir=cache_dir)

    transpose = transpose_mod.get_function("transpose")
    at_gpu = gpuarray.empty(a_gpu.shape[::-1], a_gpu.dtype)
    transpose(at_gpu, a_gpu, np.uint32(a_gpu.size),
              block=block_dim,
              grid=grid_dim)

    return at_gpu

def hermitian(a_gpu):
    """
    Hermitian (conjugate) matrix transpose.

    Conjugate transpose a matrix in device memory and return an object
    representing the transposed matrix.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, n)`.

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

    if a_gpu.dtype not in [np.float32, np.float64, np.complex64,
                           np.complex128]:
        raise ValueError('unrecognized type')

    use_double = int(a_gpu.dtype in [np.float64, np.complex128])
    use_complex = int(a_gpu.dtype in [np.complex64, np.complex128])

    # Get block/grid sizes:
    dev = misc.get_current_device()
    block_dim, grid_dim = misc.select_block_grid_sizes(dev, a_gpu.shape)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    transpose_mod = \
                  SourceModule(transpose_template.substitute(use_double=use_double,
                                                             use_complex=use_complex,
                                                             hermitian=1,
                               cols=a_gpu.shape[1],
                               rows=a_gpu.shape[0]),
                               cache_dir=cache_dir)

    transpose = transpose_mod.get_function("transpose")
    at_gpu = gpuarray.empty(a_gpu.shape[::-1], a_gpu.dtype)
    transpose(at_gpu, a_gpu, np.uint32(a_gpu.size),
              block=block_dim,
              grid=grid_dim)

    return at_gpu

conj_template = Template("""
#include <pycuda-complex.hpp>

#if ${use_double}
#define COMPLEX pycuda::complex<double>
#else
#define COMPLEX pycuda::complex<float>
#endif

__global__ void conj_inplace(COMPLEX *a, unsigned int N)
{
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < N)
        a[idx] = conj(a[idx]);
}

__global__ void conj(COMPLEX *a, COMPLEX *ac, unsigned int N)
{
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < N)
        ac[idx] = conj(a[idx]);
}
""")

def conj(a_gpu, overwrite=True):
    """
    Complex conjugate.

    Compute the complex conjugate of the array in device memory.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input array of shape `(m, n)`.
    overwrite : bool
        If true (default), save the result in the specified array.
        If false, return the result in a newly allocated array.

    Returns
    -------
    ac_gpu : pycuda.gpuarray.GPUArray
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
    >>> a = np.array([[1+1j, 2-2j, 3+3j, 4-4j], [5+5j, 6-6j, 7+7j, 8-8j]], np.complex64)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> linalg.conj(a_gpu)
    >>> np.all(a == np.conj(a_gpu.get()))
    True

    """

    # Don't attempt to process non-complex matrix types:
    if a_gpu.dtype in [np.float32, np.float64]:
        return

    if a_gpu.dtype == np.complex64:
        use_double = 0
    elif a_gpu.dtype == np.complex128:
        use_double = 1
    else:
        raise ValueError('unsupported type')

    # Get block/grid sizes:
    dev = misc.get_current_device()
    block_dim, grid_dim = misc.select_block_grid_sizes(dev, a_gpu.shape)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    conj_mod = \
             SourceModule(conj_template.substitute(use_double=use_double),
                          cache_dir=cache_dir)

    if overwrite:
        conj_inplace = conj_mod.get_function("conj_inplace")
        conj_inplace(a_gpu, np.uint32(a_gpu.size),
                     block=block_dim,
                     grid=grid_dim)
        return a_gpu
    else:
        conj = conj_mod.get_function("conj")
        ac_gpu = gpuarray.empty_like(a_gpu)
        conj(a_gpu, ac_gpu, np.uint32(a_gpu.size),
             block=block_dim,
             grid=grid_dim)
        return ac_gpu

diag_template = Template("""
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

def diag(v_gpu):
    """
    Construct a diagonal matrix.

    Constructs a matrix in device memory whose diagonal elements
    correspond to the elements in the specified array; all
    non-diagonal elements are set to 0.

    Parameters
    ----------
    v_obj : pycuda.gpuarray.GPUArray
        Input array of length `n`.

    Returns
    -------
    d_gpu : pycuda.gpuarray.GPUArray
        Diagonal matrix of dimensions `[n, n]`.

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

    """

    if v_gpu.dtype not in [np.float32, np.float64, np.complex64,
                           np.complex128]:
        raise ValueError('unrecognized type')

    if len(v_gpu.shape) > 1:
        raise ValueError('input array cannot be multidimensional')

    use_double = int(v_gpu.dtype in [np.float64, np.complex128])
    use_complex = int(v_gpu.dtype in [np.complex64, np.complex128])

    # Initialize output matrix:
    d_gpu = misc.zeros((v_gpu.size, v_gpu.size), v_gpu.dtype)

    # Get block/grid sizes:
    dev = misc.get_current_device()
    block_dim, grid_dim = misc.select_block_grid_sizes(dev, d_gpu.shape)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    diag_mod = \
             SourceModule(diag_template.substitute(use_double=use_double,
                                                   use_complex=use_complex),
                          cache_dir=cache_dir)

    diag = diag_mod.get_function("diag")
    diag(v_gpu, d_gpu, np.uint32(v_gpu.size),
         block=block_dim,
         grid=grid_dim)

    return d_gpu

eye_template = Template("""
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
// N must contain the number of rows or columns in the matrix.
__global__ void eye(FLOAT *d, int N) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < N)
        d[idx*(N+1)] = FLOAT(1.0);
}

""")

def eye(N, dtype=np.float32):
    """
    Construct a 2D matrix with ones on the diagonal and zeros elsewhere.

    Constructs a matrix in device memory whose diagonal elements
    are set to 1 and non-diagonal elements are set to 0.

    Parameters
    ----------
    N : int
        Number of rows or columns in the output matrix.

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
    >>> e_gpu = linalg.eye(v_gpu, np.complex64)
    >>> np.all(e_gpu.get() == np.eye(N, np.complex64))
    True

    """

    if dtype not in [np.float32, np.float64, np.complex64,
                     np.complex128]:
        raise ValueError('unrecognized type')
    if N <= 0:
        raise ValueError('N must be greater than 0')

    use_double = int(dtype in [np.float64, np.complex128])
    use_complex = int(dtype in [np.complex64, np.complex128])

    # Initialize output matrix:
    e_gpu = misc.zeros((N, N), dtype)

    # Get block/grid sizes:
    dev = misc.get_current_device()
    block_dim, grid_dim = misc.select_block_grid_sizes(dev, e_gpu.shape)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    eye_mod = \
             SourceModule(eye_template.substitute(use_double=use_double,
                                                   use_complex=use_complex),
                          cache_dir=cache_dir)

    eye = eye_mod.get_function("eye")
    eye(e_gpu, np.uint32(N),
        block=block_dim,
        grid=grid_dim)

    return e_gpu

cutoff_invert_s_template = Template("""
#if ${use_double}
#define FLOAT double
#else
#define FLOAT float
#endif

// N must equal the length of s:
__global__ void cutoff_invert_s(FLOAT *s, FLOAT *cutoff, unsigned int N) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < N)
        if (s[idx] > cutoff[0])
            s[idx] = 1/s[idx];
        else
            s[idx] = 0.0;
}
""")

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

    # Get block/grid sizes; the number of threads per block is limited
    # to 512 because the cutoff_invert_s kernel defined above uses too
    # many registers to be invoked in 1024 threads per block (i.e., on
    # GPUs with compute capability >= 2.x):
    dev = misc.get_current_device()
    max_threads_per_block = 512
    block_dim, grid_dim = misc.select_block_grid_sizes(dev, s_gpu.shape, max_threads_per_block)

    # Suppress very small singular values:
    use_double = 1 if s_gpu.dtype == np.float64 else 0
    cutoff_invert_s_mod = \
        SourceModule(cutoff_invert_s_template.substitute(use_double=use_double))
    cutoff_invert_s = \
                    cutoff_invert_s_mod.get_function('cutoff_invert_s')
    cutoff_gpu = gpuarray.max(s_gpu)*rcond
    cutoff_invert_s(s_gpu, cutoff_gpu,
                    np.uint32(s_gpu.size),
                    block=block_dim, grid=grid_dim)

    # Compute the pseudoinverse without allocating a new diagonal matrix:
    return dot(vh_gpu, dot_diag(s_gpu, u_gpu, 't'), 'c', 'c')

tril_template = Template("""
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

def tril(a_gpu, overwrite=True, handle=None):
    """
    Lower triangle of a matrix.

    Return the lower triangle of a square matrix.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, m)`
    overwrite : boolean
        If true (default), zero out the upper triangle of the matrix.
        If false, return the result in a newly allocated matrix.
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `scikits.misc._global_cublas_handle` is used.

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

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    tril_mod = \
             SourceModule(tril_template.substitute(use_double=use_double,
                                                   use_complex=use_complex,
                                                   cols=N),
                          cache_dir=cache_dir)
    tril = tril_mod.get_function("tril")

    if not overwrite:
        a_orig_gpu = gpuarray.empty(a_gpu.shape, a_gpu.dtype)
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

multiply_template = Template("""
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

// Stores result in y
__global__ void multiply_inplace(FLOAT *x, FLOAT *y,
                                 unsigned int N) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < N) {
        y[idx] *= x[idx];
    }
}

// Stores result in z
__global__ void multiply(FLOAT *x, FLOAT *y, FLOAT *z,
                         unsigned int N) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < N) {
        z[idx] = x[idx]*y[idx];
    }
}
""")

def multiply(x_gpu, y_gpu, overwrite=True):
    """
    Multiply arguments element-wise.

    Parameters
    ----------
    x_gpu, y_gpu : pycuda.gpuarray.GPUArray
        Input arrays to be multiplied.
    dev : pycuda.driver.Device
        Device object to be used.
    overwrite : bool
        If true (default), return the result in `y_gpu`.
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

    if x_gpu.shape != y_gpu.shape:
        raise ValueError('input arrays must have the same shape')

    if x_gpu.dtype not in [np.float32, np.float64, np.complex64,
                           np.complex128]:
        raise ValueError('unrecognized type')

    use_double = int(x_gpu.dtype in [np.float64, np.complex128])
    use_complex = int(x_gpu.dtype in [np.complex64, np.complex128])

    # Get block/grid sizes:
    dev = misc.get_current_device()
    block_dim, grid_dim = misc.select_block_grid_sizes(dev, x_gpu.shape)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    multiply_mod = \
             SourceModule(multiply_template.substitute(use_double=use_double,
                                                       use_complex=use_complex),
                          cache_dir=cache_dir)
    if overwrite:
        multiply = multiply_mod.get_function("multiply_inplace")
        multiply(x_gpu, y_gpu, np.uint32(x_gpu.size),
                 block=block_dim,
                 grid=grid_dim)
        return y_gpu
    else:
        multiply = multiply_mod.get_function("multiply")
        z_gpu = gpuarray.empty(x_gpu.shape, x_gpu.dtype)
        multiply(x_gpu, y_gpu, z_gpu, np.uint32(x_gpu.size),
                 block=block_dim,
                 grid=grid_dim)
        return z_gpu

if __name__ == "__main__":
    import doctest
    doctest.testmod()
