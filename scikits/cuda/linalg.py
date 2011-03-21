#!/usr/bin/env python

"""
PyCUDA-based linear algebra functions.
"""

from pprint import pprint
from string import Template, lower
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np

import cuda
import cublas
import cula

from misc import get_dev_attrs, select_block_grid_sizes, init, get_current_device

# Get installation location of C headers:
from __init__ import install_headers
    
def svd(a_gpu, full_matrices=1, compute_uv=1):
    """
    Singular Value Decomposition.

    Factors the matrix `a` into two unitary matrices, `u` and `vh`,
    and a 1-dimensional array of real, non-negative singular values,
    `s`, such that `a == dot(u.T, dot(diag(s), vh.T))`.

    Parameters
    ----------
    a : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, n)` to decompose.
    full_matrices : bool, optional
        If True (default), `u` and `vh` have the shapes
        `(m, m)` and `(n, n)`, respectively.  Otherwise, the shapes
        are `(m, k)` and `(k, n)`, resp., where `k = min(m, n)`.
    compute_uv : bool, optional
        If True (default), compute `u` and `vh` in addition to `s`.

    Returns
    -------
    u : pycuda.gpuarray.GPUArray
        Unitary matrix of shape `(m, m)` or `(m, k)` depending on
        value of `full_matrices`.
    s : pycuda.gpuarray.GPUArray
        Array containing the singular values, sorted such that `s[i] >= s[i+1]`.
        `s` is of length `min(m, n)`.
    vh : pycuda.gpuarray.GPUArray
        Unitary matrix of shape `(n, n)` or `(k, n)`, depending
        on `full_matrices`. 

    Notes
    -----
    Double precision is only supported if the premium version of the
    CULA toolkit is installed.

    This function destroys the contents of the input matrix.
    
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
    >>> u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 0)
    >>> np.allclose(a, np.dot(u_gpu.get(), np.dot(np.diag(s_gpu.get()), vh_gpu.get())), 1e-4)
    True

    """
    
    # The free version of CULA only supports single precision floating
    # point numbers:
    data_type = a_gpu.dtype.type
    real_type = np.float32
    if data_type == np.complex64:
        cula_func = cula._libcula.culaDeviceCgesvd        
    elif data_type == np.float32:
        cula_func = cula._libcula.culaDeviceSgesvd
    else:
        if cula._libcula_toolkit == 'premium':
            if data_type == np.complex128:
                cula_func = cula._libcula.culaDeviceZgesvd
            elif data_type == np.float64:
                cula_func = cula._libcula.culaDeviceDgesvd
            else:
                raise ValueError('unsupported type')
            real_type = np.float64
        else:
            raise ValueError('double precision not supported')
        
    # Flip the shape of the input because CUDA assumes arrays are
    # stored in column-major format:
    n, m = a_gpu.shape
    
    # Set LDA:
    lda = max(1, m)

    # Set S:
    s_gpu = gpuarray.empty(min(m, n), real_type)
    
    # Set JOBU and JOBVT:
    if compute_uv:
        if full_matrices:
            jobu = 'A'
            jobvt = 'A'
        else:
            jobu = 'S'
            jobvt = 'S'
    else:
        jobu = 'N'
        jobvt = 'N'

    # Set LDU and transpose of U:
    ldu = m
    if jobu == 'A':
        u_gpu = gpuarray.empty((ldu, m), data_type)
    elif jobu == 'S':
        u_gpu = gpuarray.empty((min(m, n), ldu), data_type)
    else:
        ldu = 1
        u_gpu = gpuarray.empty((1, 1), data_type)
        
    # Set LDVT and transpose of VT:
    if jobvt == 'A':
        ldvt = n
        vt_gpu = gpuarray.empty((n, n), data_type)
    elif jobvt == 'S':
        ldvt = min(m, n)
        vt_gpu = gpuarray.empty((n, ldvt), data_type)
    else:
        ldvt = 1
        vt_gpu = gpuarray.empty((1, 1), data_type)

    # Compute SVD and check error status:
    status = cula_func(jobu, jobvt, m, n, int(a_gpu.gpudata),
                       lda, int(s_gpu.gpudata), int(u_gpu.gpudata),
                       ldu, int(vt_gpu.gpudata), ldvt)

    cula.culaCheckStatus(status)

    # Free internal CULA memory:
    cula.culaFreeBuffers()
    
    if compute_uv:
        return vt_gpu, s_gpu, u_gpu
    else:
        return s_gpu

def dot(x_gpu, y_gpu, transa='N', transb='N'):
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

    if len(x_gpu.shape) == 1 and len(y_gpu.shape) == 1:

        # Compute inner product for 1D arrays:
        if (x_gpu.dtype == np.complex64 and y_gpu.dtype == np.complex64):
            cublas_func = cublas._libcublas.cublasCdotu
        elif (x_gpu.dtype == np.float32 and y_gpu.dtype == np.float32):
            cublas_func = cublas._libcublas.cublasSdot
        elif (x_gpu.dtype == np.complex128 and y_gpu.dtype == np.complex128):
            cublas_func = cublas._libcublas.cublasZdotu
        elif (x_gpu.dtype == np.float64 and y_gpu.dtype == np.float64):
            cublas_func = cublas._libcublas.cublasDdot
        else:
            raise ValueError('unsupported combination of input types')

        result = cublas_func(x_gpu.size, int(x_gpu.gpudata), 1,
                             int(y_gpu.gpudata), 1)

        if x_gpu.dtype == np.complex64:
            return np.float32(result.x)+1j*np.float32(result.y)
        elif x_gpu.dtype == np.complex128:
            return np.float64(result.x)+1j*np.float64(result.y)
        elif x_gpu.dtype == np.float32:
            return np.float32(result)
        else:
            return np.float64(result)
    else:

        # Perform matrix multiplication for 2D arrays:
        if (x_gpu.dtype == np.complex64 and y_gpu.dtype == np.complex64):
            cublas_func = cublas._libcublas.cublasCgemm        
            alpha = cuda.cuFloatComplex(1, 0)
            beta = cuda.cuFloatComplex(0, 0)
        elif (x_gpu.dtype == np.float32 and y_gpu.dtype == np.float32):
            cublas_func = cublas._libcublas.cublasSgemm
            alpha = np.float32(1.0)
            beta = np.float32(0.0)
        elif (x_gpu.dtype == np.complex128 and y_gpu.dtype == np.complex128):
            cublas_func = cublas._libcublas.cublasZgemm        
            alpha = cuda.cuDoubleComplex(1, 0)
            beta = cuda.cuDoubleComplex(0, 0)
        elif (x_gpu.dtype == np.float64 and y_gpu.dtype == np.float64):
            cublas_func = cublas._libcublas.cublasDgemm
            alpha = np.float64(1.0)
            beta = np.float64(0.0)
        else:
            raise ValueError('unsupported combination of input types')

        transa = lower(transa)
        transb = lower(transb)        

        if transb in ['t', 'c']:
            m, k = y_gpu.shape
        elif transb in ['n']:
            k, m = y_gpu.shape
        else:
            raise ValueError('invalid value for transb')

        if transa in ['t', 'c']:
            n = x_gpu.shape[1]
        elif transa in ['n']:
            n = x_gpu.shape[0]
        else:
            raise ValueError('invalid value for transa')

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
        cublas_func(transb, transa, m, n, k, alpha, int(y_gpu.gpudata),
                    lda, int(x_gpu.gpudata), ldb, beta, int(c_gpu.gpudata), ldc)

        status = cublas.cublasGetError()
        cublas.cublasCheckStatus(status)

        return c_gpu

def mdot(*args):
    """
    Product of several matrices.

    Computes the matrix product of several arrays of shapes.

    Parameters
    ----------
    a_gpu, b_gpu, ... : pycuda.gpuarray.GPUArray
        Arrays to multiply.

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

    # Free the temporary matrix allocated when computing the dot
    # product:
    out_gpu = args[0]
    for next_gpu in args[1:]:
        temp_gpu = dot(out_gpu, next_gpu)
        out_gpu.gpudata.free()
        del(out_gpu)
        out_gpu = temp_gpu
        del(temp_gpu)
    return out_gpu

transpose_template = Template("""
#include <pycuda/pycuda-complex.hpp>

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
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;
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

    dev = get_current_device()
    
    use_double = int(a_gpu.dtype in [np.float64, np.complex128])
    use_complex = int(a_gpu.dtype in [np.complex64, np.complex128])

    # Get block/grid sizes:
    max_threads_per_block, max_block_dim, max_grid_dim = get_dev_attrs(dev)
    block_dim, grid_dim = select_block_grid_sizes(dev, a_gpu.shape)
    max_blocks_per_grid = max(max_grid_dim)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None            
    transpose_mod = \
                  SourceModule(transpose_template.substitute(use_double=use_double,
                                                                 use_complex=use_complex,
                                                                 hermitian=0,
                               max_threads_per_block=max_threads_per_block,
                               max_blocks_per_grid=max_blocks_per_grid,
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

    dev = get_current_device()
    
    use_double = int(a_gpu.dtype in [np.float64, np.complex128])
    use_complex = int(a_gpu.dtype in [np.complex64, np.complex128])

    # Get block/grid sizes:
    max_threads_per_block, max_block_dim, max_grid_dim = get_dev_attrs(dev)
    block_dim, grid_dim = select_block_grid_sizes(dev, a_gpu.shape)
    max_blocks_per_grid = max(max_grid_dim)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None            
    transpose_mod = \
                  SourceModule(transpose_template.substitute(use_double=use_double,
                                                                 use_complex=use_complex,
                                                                 hermitian=1,
                               max_threads_per_block=max_threads_per_block,
                               max_blocks_per_grid=max_blocks_per_grid,
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
#include <pycuda/pycuda-complex.hpp>

#if ${use_double}
#define COMPLEX pycuda::complex<double>
#else
#define COMPLEX pycuda::complex<float>
#endif

__global__ void conj_inplace(COMPLEX *a, unsigned int N)
{
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;

    if (idx < N)                       
        a[idx] = conj(a[idx]);
}

__global__ void conj(COMPLEX *a, COMPLEX *ac, unsigned int N)
{
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;

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

    dev = get_current_device()
    
    # Get block/grid sizes:
    max_threads_per_block, max_block_dim, max_grid_dim = get_dev_attrs(dev)
    block_dim, grid_dim = select_block_grid_sizes(dev, a_gpu.shape)
    max_blocks_per_grid = max(max_grid_dim)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    conj_mod = \
             SourceModule(conj_template.substitute(use_double=use_double,
                          max_threads_per_block=max_threads_per_block,
                          max_blocks_per_grid=max_blocks_per_grid),
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
#include <pycuda/pycuda-complex.hpp>

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
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;
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

    dev = get_current_device()
    
    use_double = int(v_gpu.dtype in [np.float64, np.complex128])
    use_complex = int(v_gpu.dtype in [np.complex64, np.complex128])

    # Initialize output matrix:
    d_gpu = gpuarray.zeros((v_gpu.size, v_gpu.size), v_gpu.dtype)

    # Get block/grid sizes:
    max_threads_per_block, max_block_dim, max_grid_dim = get_dev_attrs(dev)
    block_dim, grid_dim = select_block_grid_sizes(dev, d_gpu.shape)
    max_blocks_per_grid = max(max_grid_dim)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    diag_mod = \
             SourceModule(diag_template.substitute(use_double=use_double,
                                                   use_complex=use_complex,
                          max_threads_per_block=max_threads_per_block,
                          max_blocks_per_grid=max_blocks_per_grid),
                          cache_dir=cache_dir)

    diag = diag_mod.get_function("diag")    
    diag(v_gpu, d_gpu, np.uint32(v_gpu.size),
         block=block_dim,
         grid=grid_dim)
    
    return d_gpu

eye_template = Template("""
#include <pycuda/pycuda-complex.hpp>

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
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;
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
    
    dev = get_current_device()
    
    use_double = int(dtype in [np.float64, np.complex128])
    use_complex = int(dtype in [np.complex64, np.complex128])

    # Initialize output matrix:
    e_gpu = gpuarray.zeros((N, N), dtype)

    # Get block/grid sizes:
    max_threads_per_block, max_block_dim, max_grid_dim = get_dev_attrs(dev)
    block_dim, grid_dim = select_block_grid_sizes(dev, e_gpu.shape)
    max_blocks_per_grid = max(max_grid_dim)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    eye_mod = \
             SourceModule(eye_template.substitute(use_double=use_double,
                                                   use_complex=use_complex,
                          max_threads_per_block=max_threads_per_block,
                          max_blocks_per_grid=max_blocks_per_grid),
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
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;

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
    Double precision is only supported if the premium version of the
    CULA toolkit is installed.

    This function destroys the contents of the input matrix.
    
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
    
    # Compute SVD:
    u_gpu, s_gpu, vh_gpu = svd(a_gpu, 0)
    
    # Get block/grid sizes; the number of threads per block is limited
    # to 512 because the cutoff_invert_s kernel defined above uses too
    # many registers to be invoked in 1024 threads per block (i.e., on
    # GPUs with compute capability >= 2.x): 
    dev = get_current_device()
    max_threads_per_block, max_block_dim, max_grid_dim = get_dev_attrs(dev)
    max_threads_per_block = 512
    block_dim, grid_dim = select_block_grid_sizes(dev, s_gpu.shape, max_threads_per_block)
    max_blocks_per_grid = max(max_grid_dim)

    # Suppress very small singular values:
    use_double = 1 if s_gpu.dtype == np.float64 else 0
    cutoff_invert_s_mod = \
        SourceModule(cutoff_invert_s_template.substitute( 
        max_threads_per_block=max_threads_per_block,
        max_blocks_per_grid=max_blocks_per_grid,
        use_double=use_double))
    cutoff_invert_s = \
                    cutoff_invert_s_mod.get_function('cutoff_invert_s')
    cutoff_gpu = gpuarray.max(s_gpu)*rcond
    cutoff_invert_s(s_gpu, cutoff_gpu,
                    np.uint32(s_gpu.size),
                    block=block_dim, grid=grid_dim)
    
    # The diagonal matrix of singular values must have the same data
    # type as u_gpu in order to compute the dot product below:
    if s_gpu.dtype == u_gpu.dtype:
        s_diag_gpu = diag(s_gpu)
    else:
        s_diag_gpu = diag(s_gpu.astype(u_gpu.dtype))
    
    # Finish pinv computation:
    suh_gpu = dot(s_diag_gpu, u_gpu, 'n', 'c')
    return dot(vh_gpu, suh_gpu, 'c')

tril_template = Template("""
#include <pycuda/pycuda-complex.hpp>

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
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;
    unsigned int ix = idx/${cols};
    unsigned int iy = idx%${cols};

    if (idx < N) {
        if (ix < iy)
            a[idx] = 0.0;
    }
}
""")

def tril(a_gpu, overwrite=True):
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

    if len(a_gpu.shape) != 2 or a_gpu.shape[0] != a_gpu.shape[1]:
        raise ValueError('matrix must be square')

    dev = get_current_device()
    
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
    max_threads_per_block, max_block_dim, max_grid_dim = get_dev_attrs(dev)
    block_dim, grid_dim = select_block_grid_sizes(dev, a_gpu.shape)
    max_blocks_per_grid = max(max_grid_dim)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    tril_mod = \
             SourceModule(tril_template.substitute(use_double=use_double,
                                                       use_complex=use_complex,
                          max_threads_per_block=max_threads_per_block,
                          max_blocks_per_grid=max_blocks_per_grid,
                          cols=N),
                          cache_dir=cache_dir)
    tril = tril_mod.get_function("tril")

    if not overwrite:
        a_orig_gpu = gpuarray.empty(a_gpu.shape, a_gpu.dtype)
        copy_func(a_gpu.size, int(a_gpu.gpudata), 1, int(a_orig_gpu.gpudata), 1)

    tril(a_gpu, np.uint32(a_gpu.size),
         block=block_dim,
         grid=grid_dim)

    if overwrite:
        return a_gpu
    else:

        # Restore original contents of a_gpu:
        swap_func(a_gpu.size, int(a_gpu.gpudata), 1, int(a_orig_gpu.gpudata), 1)
        return a_orig_gpu

multiply_template = Template("""
#include <pycuda/pycuda-complex.hpp>

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
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;
    if (idx < N) {
        y[idx] *= x[idx];
    }
}

// Stores result in z
__global__ void multiply(FLOAT *x, FLOAT *y, FLOAT *z,
                         unsigned int N) {
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;
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

    dev = get_current_device()
    
    # Get block/grid sizes:
    max_threads_per_block, max_block_dim, max_grid_dim = get_dev_attrs(dev)
    block_dim, grid_dim = select_block_grid_sizes(dev, x_gpu.shape)
    max_blocks_per_grid = max(max_grid_dim)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    multiply_mod = \
             SourceModule(multiply_template.substitute(use_double=use_double,
                                                       use_complex=use_complex,
                          max_threads_per_block=max_threads_per_block,
                          max_blocks_per_grid=max_blocks_per_grid),
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
