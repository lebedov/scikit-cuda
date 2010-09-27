#!/usr/bin/env python

"""
PyCUDA-based linear algebra functions.
"""

from pprint import pprint
from string import Template
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np

import cuda
import cublas
import cula

from misc import get_dev_attrs, select_block_grid_sizes

# Get installation location of C headers:
from __info__ import install_headers

def init():
    """
    Initialize CUDA utilities.
        
    Notes
    -----
    This function does not initialize PyCUDA; it uses whatever device
    was initialized in the current host thread.
    
    """

    # CUBLAS uses whatever device is being used by the host thread:
    cublas.cublasInit()

    # culaSelectDevice() need not (and, in fact, cannot) be called
    # here because the host thread has already been bound to a GPU
    # device:
    cula.culaInitialize()
    
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
    real_dtype = np.dtype(np.float32)
    if a_gpu.dtype == np.complex64:
        cula_func = cula._libcula.culaDeviceCgesvd        
    elif a_gpu.dtype == np.float32:
        cula_func = cula._libcula.culaDeviceSgesvd
    else:
        raise ValueError('unsupported type')

    # Transpose shape because CUDA assumes arrays are stored in
    # column-major format:
    (m, n) = a_gpu.shape[::-1]
    
    # Set LDA:
    lda = max(1, m)

    # Set S:
    s_gpu = gpuarray.empty(min(m, n), real_dtype)
    
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
        u_gpu = gpuarray.empty((ldu, m), a_gpu.dtype)
    elif jobu == 'S':
        u_gpu = gpuarray.empty((min(m, n), ldu), a_gpu.dtype)
    else:
        ldu = 1
        u_gpu = gpuarray.empty((1, 1), a_gpu.dtype)
        
    # Set LDVT and transpose of VT:
    if jobvt == 'A':
        ldvt = n
        vt_gpu = gpuarray.empty((n, n), a_gpu.dtype)
    elif jobvt == 'S':
        ldvt = min(m, n)
        vt_gpu = gpuarray.empty((n, ldvt), a_gpu.dtype)
    else:
        ldvt = 1
        vt_gpu = gpuarray.empty((1, 1), a_gpu.dtype)

    # Compute SVD and check error status:
    status = cula_func(jobu, jobvt, m, n, int(a_gpu.gpudata),
                       lda, int(s_gpu.gpudata), int(u_gpu.gpudata),
                       ldu, int(vt_gpu.gpudata), ldvt)
    cula.culaCheckStatus(status)

    if compute_uv:
        return vt_gpu, s_gpu, u_gpu
    else:
        return s_gpu

def dot(a_gpu, b_gpu):
    """
    Matrix product of two arrays.

    For 1D arrays, this function computes the inner product. For 2D
    arrays of shapes `(m, k)` and `(k, n)`, it computes the matrix
    product; the result has shape `(m, n)`.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input array.
    b_gpu : pycuda.gpuarray.GPUArray
        Input array.
        
    Returns
    -------
    c_gpu : pycuda.gpuarray.GPUArray
        Dot product of `a_gpu` and `b_gpu`.
    
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
    >>> p = np.asarray(np.random.rand(4, 2), np.complex64)
    >>> q = np.asarray(np.random.rand(2, 2), np.complex64)
    >>> p_gpu = gpuarray.to_gpu(p)
    >>> q_gpu = gpuarray.to_gpu(q)
    >>> r_gpu = linalg.dot(p_gpu, q_gpu)
    >>> np.allclose(np.dot(p, q), r_gpu.get())
    True
    >>> s = np.asarray(np.random.rand(5), np.complex128)
    >>> t = np.asarray(np.random.rand(5), np.complex128)
    >>> s_gpu = gpuarray.to_gpu(s)
    >>> t_gpu = gpuarray.to_gpu(t)
    >>> u = linalg.dot(s_gpu, t_gpu)
    >>> np.allclose(np.dot(s, t), u)
    True
    
    """

    if len(a_gpu.shape) == 1 and len(b_gpu.shape) == 1:

        # Compute inner product for 1D arrays:
        if (a_gpu.dtype == np.complex64 and b_gpu.dtype == np.complex64):
            cublas_func = cublas._libcublas.cublasCdotu
        elif (a_gpu.dtype == np.float32 and b_gpu.dtype == np.float32):
            cublas_func = cublas._libcublas.cublasSdot
        elif (a_gpu.dtype == np.complex128 and b_gpu.dtype == np.complex128):
            cublas_func = cublas._libcublas.cublasZdotu
        elif (a_gpu.dtype == np.float64 and b_gpu.dtype == np.float64):
            cublas_func = cublas._libcublas.cublasDdot
        else:
            raise ValueError('unsupported combination of input types')

        result = cublas_func(a_gpu.size, int(a_gpu.gpudata), 1,
                             int(b_gpu.gpudata), 1)

        if a_gpu.dtype == np.complex64:
            return np.float32(result.x)+1j*np.float32(result.y)
        elif a_gpu.dtype == np.complex128:
            return np.float64(result.x)+1j*np.float64(result.y)
        elif a_gpu.dtype == np.float32:
            return np.float32(result)
        else:
            return np.float64(result)
    else:

        # Perform matrix multiplication for 2D arrays:
        if (a_gpu.dtype == np.complex64 and b_gpu.dtype == np.complex64):
            cublas_func = cublas._libcublas.cublasCgemm        
            alpha = cuda.cuFloatComplex(1, 0)
            beta = cuda.cuFloatComplex(0, 0)
        elif (a_gpu.dtype == np.float32 and b_gpu.dtype == np.float32):
            cublas_func = cublas._libcublas.cublasSgemm
            alpha = np.float32(1.0)
            beta = np.float32(0.0)
        elif (a_gpu.dtype == np.complex128 and b_gpu.dtype == np.complex128):
            cublas_func = cublas._libcublas.cublasZgemm        
            alpha = cuda.cuDoubleComplex(1, 0)
            beta = cuda.cuDoubleComplex(0, 0)
        elif (a_gpu.dtype == np.float64 and b_gpu.dtype == np.float64):
            cublas_func = cublas._libcublas.cublasDgemm
            alpha = np.float64(1.0)
            beta = np.float64(0.0)
        else:
            raise ValueError('unsupported combination of input types')

        transa = 'N'
        transb = 'N'
        m = b_gpu.shape[1]
        n = a_gpu.shape[0]
        k = b_gpu.shape[0]
        lda = m
        ldb = k
        ldc = max(1, m)

        c_gpu = gpuarray.empty((a_gpu.shape[0], b_gpu.shape[1]), a_gpu.dtype)
        cublas_func(transb, transa, m, n, k, alpha, int(b_gpu.gpudata),
                    lda, int(a_gpu.gpudata), ldb, beta, int(c_gpu.gpudata), ldc)

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

transpose_mod_template = Template("""
#include <cuComplex.h>

#define USE_DOUBLE ${use_double}
#define USE_COMPLEX ${use_complex}
#if USE_DOUBLE == 1
#if USE_COMPLEX == 1
#define TYPE cuDoubleComplex
#define CONJ(x) cuConj(x)
#else
#define TYPE double
#define CONJ(x) (x)
#endif
#else
#if USE_COMPLEX == 1
#define TYPE cuFloatComplex
#define CONJ(x) cuConjf(x)
#else
#define TYPE float
#define CONJ(x) (x)
#endif
#endif

__global__ void transpose(TYPE *odata, TYPE *idata, unsigned int N)
{
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;
    unsigned int ix = idx/${cols};
    unsigned int iy = idx%${cols};

    if (idx < N)
        odata[iy*${rows}+ix] = CONJ(idata[ix*${cols}+iy]);
}
""")

def transpose(a_gpu, dev):
    """
    Matrix transpose.
    
    Transpose a matrix in device memory and return an object
    representing the transposed matrix.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, n)`.
    dev : pycuda.driver.Device
        Device object to be used.

    Returns
    -------
    at_gpu : pycuda.gpuarray.GPUArray
        Transposed matrix of shape `(n, m)`.
    dev : pycuda.driver.Device
        Device object to be used.

    Notes
    -----
    If the specified matrix type is complex, the function will return
    the Hermitian of the input matrix.
    
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
    >>> at_gpu = transpose(a_gpu, pycuda.autoinit.device)
    >>> np.all(a.T == at_gpu.get())
    True
    >>> b = np.array([[1j, 2j, 3j, 4j, 5j, 6j], [7j, 8j, 9j, 10j, 11j, 12j]], np.complex64)
    >>> b_gpu = gpuarray.to_gpu(b)
    >>> bt_gpu = transpose(b_gpu, pycuda.autoinit.device)
    >>> np.all(np.conj(b.T) == bt_gpu.get())
    True

    """

    if a_gpu.dtype not in [np.float32, np.float64, np.complex64,
                           np.complex128]:
        raise ValueError('unrecognized type')

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
                  SourceModule(transpose_mod_template.substitute(use_double=use_double,
                                                                 use_complex=use_complex,
                               max_threads_per_block=max_threads_per_block,
                               max_blocks_per_grid=max_blocks_per_grid,
                               cols=a_gpu.shape[1],
                               rows=a_gpu.shape[0]),
                               cache_dir=cache_dir)                                                                 

    transpose = transpose_mod.get_function("transpose")
    at_gpu = gpuarray.empty(a_gpu.shape[::-1], a_gpu.dtype)
    transpose(at_gpu.gpudata, a_gpu.gpudata,
              np.uint32(a_gpu.size),
              block=block_dim,
              grid=grid_dim)
                    
    return at_gpu

conj_mod_template = Template("""
#include <cuComplex.h>

#define USE_DOUBLE ${use_double}
#if USE_DOUBLE == 1
#define COMPLEX cuDoubleComplex
#define CONJ(z) cuConj(z)
#else
#define COMPLEX cuFloatComplex
#define CONJ(z) cuConjf(z)
#endif

__global__ void conj(COMPLEX *a, unsigned int N)
{
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;

    if (idx < N)                       
        a[idx] = CONJ(a[idx]);
}
""")

def conj(a_gpu, dev):
    """
    Complex conjugate.
    
    Compute the complex conjugate of the matrix in device memory.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, n)`.
    dev : pycuda.driver.Device
        Device object to be used.

    Notes
    -----
    The input matrix is modified in place.

    This function assumes that the input matrix contains complex
    numbers; undefined behavior may occur for other types.
    
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
    >>> conj(a_gpu, pycuda.autoinit.device)
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
    max_threads_per_block, max_block_dim, max_grid_dim = get_dev_attrs(dev)
    block_dim, grid_dim = select_block_grid_sizes(dev, a_gpu.shape)
    max_blocks_per_grid = max(max_grid_dim)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    conj_mod = \
             SourceModule(conj_mod_template.substitute(use_double=use_double,
                          max_threads_per_block=max_threads_per_block,
                          max_blocks_per_grid=max_blocks_per_grid),
                          cache_dir=cache_dir)

    conj = conj_mod.get_function("conj")
    conj(a_gpu.gpudata,
         np.uint32(a_gpu.size),
         block=block_dim,
         grid=grid_dim)

diag_mod_template = Template("""
#include <cuComplex.h>

#define USE_DOUBLE ${use_double}
#define USE_COMPLEX ${use_complex}
#if USE_DOUBLE == 1
#if USE_COMPLEX == 1
#define TYPE cuDoubleComplex
#define ZERO make_cuDoubleComplex(0, 0)
#else
#define TYPE double
#define ZERO 0.0
#endif
#else
#if USE_COMPLEX == 1
#define TYPE cuFloatComplex
#define ZERO make_cuFloatComplex(0, 0)
#else
#define TYPE float
#define ZERO 0.0
#endif
#endif

// N must contain the number of elements in d:
__global__ void diag(TYPE *v, TYPE *d, int N) {
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;
    unsigned int ix = idx/${cols};
    unsigned int iy = idx%${cols};
    
    if (idx < N)
        if (ix == iy) {
            d[idx] = v[ix];
        } else {
            d[idx] = ZERO;
        }
}
""")

def diag(v_gpu, dev):
    """
    Construct a diagonal matrix.

    Constructs a matrix in device memory whose diagonal elements
    correspond to the elements in the specified array; all
    non-diagonal elements are set to 0.

    Parameters
    ----------
    a_obj : pycuda.gpuarray.GPUArray
        Input array of length `n`.
    dev : pycuda.driver.Device
        Device object to be used.

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
    >>> d_gpu = diag(v_gpu, pycuda.autoinit.device);
    >>> np.all(d_gpu.get() == np.diag(v))
    True
    >>> v = np.array([1j, 2j, 3j, 4j, 5j, 6j], np.complex64)
    >>> v_gpu = gpuarray.to_gpu(v)
    >>> d_gpu = diag(v_gpu, pycuda.autoinit.device);
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

    # Allocate output matrix:
    d_gpu = gpuarray.empty((v_gpu.size, v_gpu.size), v_gpu.dtype)
    
    # Get block/grid sizes:
    max_threads_per_block, max_block_dim, max_grid_dim = get_dev_attrs(dev)
    block_dim, grid_dim = select_block_grid_sizes(dev, d_gpu.shape)
    max_blocks_per_grid = max(max_grid_dim)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    diag_mod = \
             SourceModule(diag_mod_template.substitute(use_double=use_double,
                                                       use_complex=use_complex,
                          max_threads_per_block=max_threads_per_block,
                          max_blocks_per_grid=max_blocks_per_grid,
                          cols=v_gpu.size),
                          cache_dir=cache_dir)

    diag = diag_mod.get_function("diag")    
    diag(v_gpu.gpudata, d_gpu.gpudata, np.uint32(d_gpu.size),
         block=block_dim,
         grid=grid_dim)
    
    return d_gpu

cutoff_invert_s_mod_template = Template("""
// N must equal the length of s:
__global__ void cutoff_invert_s(float *s, float *cutoff, unsigned int N) {
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;

    if (idx < N) 
        if (s[idx] > cutoff[0])
            s[idx] = 1/s[idx];
        else
            s[idx] = 0.0;
}
""")

def pinv(a_gpu, dev, rcond=1e-15):
    """
    Moore-Penrose pseudoinverse.

    Compute the Moore-Penrose pseudoinverse of the specified matrix.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, n)`.
    dev : pycuda.driver.Device
        Device object to be used.
    rcond : float
        Singular values smaller than `rcond`*max(singular_values)`
        are set to zero.
        
    Returns
    -------
    a_inv_gpu : pycuda.gpuarray.GPUArray
        Pseudoinverse of input matrix.
        
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
    >>> a_inv_gpu = pinv(a_gpu, pycuda.autoinit.device)
    >>> np.allclose(np.linalg.pinv(a), a_inv_gpu.get(), 1e-4)
    True
    >>> b = np.asarray(np.random.rand(8, 4)+1j*np.random.rand(8, 4), np.complex64)
    >>> b_gpu = gpuarray.to_gpu(b)
    >>> b_inv_gpu = pinv(b_gpu, pycuda.autoinit.device)
    >>> np.allclose(np.linalg.pinv(b), b_inv_gpu.get(), 1e-4)
    True

    """

    # Check input dtype because the SVD can only be computed in single
    # precision:
    if a_gpu.dtype not in [np.float32, np.complex64]:
        raise ValueError('unsupported type')

    # Compute SVD:
    u_gpu, s_gpu, vh_gpu = svd(a_gpu, 0)
    uh_gpu = transpose(u_gpu, dev)

    # Get block/grid sizes:
    max_threads_per_block, max_block_dim, max_grid_dim = get_dev_attrs(dev)
    block_dim, grid_dim = select_block_grid_sizes(dev, s_gpu.shape)
    max_blocks_per_grid = max(max_grid_dim)

    # Suppress very small singular values:
    cutoff_invert_s_mod = \
        SourceModule(cutoff_invert_s_mod_template.substitute( 
        max_threads_per_block=max_threads_per_block,
        max_blocks_per_grid=max_blocks_per_grid))
    cutoff_invert_s = \
                    cutoff_invert_s_mod.get_function('cutoff_invert_s')
    cutoff_gpu = gpuarray.max(s_gpu)*rcond
    cutoff_invert_s(s_gpu.gpudata, cutoff_gpu.gpudata,
                    np.uint32(s_gpu.size),
                    block=block_dim, grid=grid_dim)
    
    # The singular values must data type is in uh_gpu:
    if s_gpu.dtype == uh_gpu.dtype:
        s_diag_gpu = diag(s_gpu, dev)
    else:
        s_diag_gpu = diag(s_gpu.astype(uh_gpu.dtype), dev)

    # Finish pinv computation:
    v_gpu = transpose(vh_gpu, dev)
    suh_gpu = dot(s_diag_gpu, uh_gpu)
    return dot(v_gpu, suh_gpu)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
