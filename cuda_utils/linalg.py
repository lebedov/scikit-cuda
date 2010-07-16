#!/usr/bin/env python

"""
PyCUDA-based functions.
"""

from pprint import pprint
from string import Template
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np

import cublas
import cula

def init():
    """
    Initialize CUDA utilities.
    """
    
    status = cublas.cublasInit()
    cublas.cublasCheckStatus(status)
    status = cula.culaInitialize()
    cula.culaCheckStatus(status)
    
def svd(a_gpu, full_matrices=1, compute_uv=1):
    """
    Singular Value Decomposition.

    Factors the matrix `a` into two unitary matrices, `u` and `vh`,
    and a 1-dimensional array of real, non-negative singular values,
    `s`, such that `a == dot(u.T, dot(diag(s), vh.T))`.

    Parameters
    ----------
    a : GPUArray
        Input matrix of shape `(m, n)` to decompose.
    full_matrices : bool, optional
        If True (default), `u` and `vh` have the shapes
        `(m, m)` and `(n, n)`, respectively.  Otherwise, the shapes
        are `(m, k)` and `(k, n)`, resp., where `k = min(m, n)`.
    compute_uv : bool, optional
        If True (default), compute `u` and `vh` in addition to `s`.

    Returns
    -------
    u : GPUArray
        Unitary matrix of shape `(m, m)` or `(m, k)` depending on
        value of `full_matrices`.
    s : GPUArray
        Array containing the singular values, sorted such that `s[i] >= s[i+1]`.
        `s` is of length `min(m, n)`.
    vh : GPUArray
        Unitary matrix of shape `(n, n)` or `(k, n)`, depending
        on `full_matrices`. 

    Notes
    -----
    Because of the limitations of the free version of CULA, the
    argument is cast to single precision.

    This function destroys the contents of the input matrix.
    
    Example
    -------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import linalg
    >>> linalg.init()
    >>> a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
    >>> a = np.asarray(a, np.complex64)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> u_gpu, s_gpu, vh_gpu = linalg.svd(a_gpu, 0)
    >>> np.allclose(a, np.dot(u_gpu.get(), np.dot(np.diag(s_gpu.get()), vh_gpu.get())))
    True

    """

    # The free version of CULA only supports single precision floating
    # point numbers:
    real_dtype = np.dtype(np.float32)
    if a_gpu.dtype == np.complex64:
        cula_func = cula._culaDeviceCgesvd        
    elif a_gpu.dtype == np.float32:
        cula_func = cula._culaDeviceSgesvd
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

    Computes the matrix product of two arrays of shapes `(m, k)` and
    `(k, n)`; the result has shape `(m, n)`.

    Parameters
    ----------
    a_gpu : GPUArray
        Matrix of shape `(m, k)`.
    b_gpu : GPUArray
        Matrix of shape `(k, n)`.

    Returns
    -------
    c_ptr : c_void_p
        Pointer to device memory containing matrix product of shape
        `(m, n)`.

    Example
    -------
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
    
    """

    if (a_gpu.dtype == np.complex64 and b_gpu.dtype == np.complex64):
        cublas_func = cublas._cublasCgemm        
        alpha = np.complex64(1.0)
        beta = np.complex64(0.0)
    elif (a_gpu.dtype == np.float32 and b_gpu.dtype == np.float32):
        cublas_func = cublas._cublasSgemm
        alpha = np.float32(1.0)
        beta = np.float32(0.0)
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

    status = cublas._cublasGetError()
    cublas.cublasCheckStatus(status)
    
    return c_gpu

def mdot(*args):
    """
    Product of several matrices.

    Example
    -------
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

    out_gpu = args[0]
    for next_gpu in args[1:]:
        temp_gpu = dot(out_gpu, next_gpu)
        del(out_gpu)
        out_gpu = temp_gpu
        del(temp_gpu)
    return out_gpu

transpose_mod_template = Template("""
#include <cuComplex.h>
__global__ void transpose(${dtype} *odata, ${dtype} *idata,
                          int width, int height)
{
    __shared__ ${dtype} tile[${tile_dim}][${tile_dim}+1];
	
    unsigned int xIndex = blockIdx.x * ${tile_dim} + threadIdx.x;
    unsigned int yIndex = blockIdx.y * ${tile_dim} + threadIdx.y;
    unsigned int index_in = xIndex + (yIndex)*width;

    xIndex = blockIdx.y * ${tile_dim} + threadIdx.x;
    yIndex = blockIdx.x * ${tile_dim} + threadIdx.y;    
    unsigned int index_out = xIndex +(yIndex)*height;

    for (int i=0; i<${tile_dim}; i+=${block_rows}) {
	tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
    }

    __syncthreads();

    for (int i=0; i<${tile_dim}; i+=${block_rows}) {
        odata[index_out+i*height] = ${conjf}(tile[threadIdx.x][threadIdx.y+i]);
    }
}
""")

def transpose(a_gpu, tile_dim, block_rows):
    """
    Matrix transpose.
    
    Transpose a matrix in device memory and return an object
    representing the transposed matrix.

    Parameters
    ----------
    a_gpu : GPUArray
        Input matrix of shape `(m, n)`.
    tile_dim : int
        Each block of threads processes `tile_dim x tile_dim` elements.
    block_rows : int
        Each thread processes `tile_dim/block_rows` elements;
        `block_rows` must therefore divide `tile_dim`.

    Returns
    -------
    at_gpu : GPUArray
        Transposed matrix of shape `(n, m)`.

    Notes
    -----
    If the specified matrix type is complex, the function will return
    the Hermitian of the input matrix.
    
    Example
    -------
    >>> import pycuda.driver as drv
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import linalg
    >>> linalg.init()
    >>> a = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], np.float32)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> at_gpu = transpose(a_gpu, 2, 2)
    >>> np.all(a.T == at_gpu.get())
    True
    >>> a = np.array([[1j, 2j, 3j, 4j, 5j, 6j], [7j, 8j, 9j, 10j, 11j, 12j]], np.complex64)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> at_gpu = transpose(a_gpu, 2, 2)
    >>> np.all(np.conj(a.T) == at_gpu.get())
    True

    """

    # XXX: need to figure out how to determine which device is used when doing this check:
    # if tile_dim*block_rows > \
    #        device.get_attribute(drv.device_attribute.MAX_THREADS_PER_BLOCK):
    #     raise ValueError('tile size too large')
    if (tile_dim % block_rows != 0):
        raise ValueError('tile_dim must be an integral multiple of block_rows')
    if (a_gpu.shape[0] % tile_dim != 0) or \
           (a_gpu.shape[1] % tile_dim != 0):
        raise ValueError('tile dimensions must divide matrix dimensions')

    if a_gpu.dtype == np.float32:
        dtype = 'float'
        conjf = ''
    elif a_gpu.dtype == np.complex64:
        dtype = 'cuFloatComplex'
        conjf = 'cuConjf'        
    else:
        raise ValueError('unsupported matrix type')
    
    transpose_mod = \
                  SourceModule(transpose_mod_template.substitute(tile_dim=str(tile_dim),
                                                                 block_rows=str(block_rows),
                                                                 dtype=dtype,
                                                                 conjf=conjf))
    transpose = transpose_mod.get_function("transpose")
    at_gpu = gpuarray.empty(a_gpu.shape[::-1], a_gpu.dtype)
    transpose(at_gpu.gpudata, a_gpu.gpudata,
              np.uint32(a_gpu.shape[1]),
              np.uint32(a_gpu.shape[0]),
              block=(tile_dim, block_rows, 1),
              grid=(a_gpu.shape[1]/tile_dim,
                    a_gpu.shape[0]/tile_dim))
    return at_gpu

conj_mod_template = Template("""
#include <cuComplex.h>
__global__ void conj(cuFloatComplex *a, int width, int height)
{
    int xIndex = blockIdx.x * ${tile_dim} + threadIdx.x;
    int yIndex = blockIdx.y * ${tile_dim} + threadIdx.y;

    int index = xIndex + width*yIndex;
    for (int i=0; i<${tile_dim}; i+=${block_rows}) {
        a[index+i*width] = cuConjf(a[index+i*width]);
    }
}
""")

def conj(a_gpu, tile_dim, block_rows):
    """
    Complex conjugate.
    
    Compute the complex conjugate of the matrix in device memory.

    Parameters
    ----------
    a_gpu : GPUArray
        Input matrix of shape `(m, n)`.
    tile_dim : int
        Each block of threads processes `tile_dim x tile_dim` elements.
    block_rows : int
        Each thread processes `tile_dim/block_rows` elements;
        `block_rows` must therefore divide `tile_dim`.

    Notes
    -----
    The input matrix is modified in place.

    This function assumes that the input matrix contains complex
    numbers; undefined behavior may occur for other types.
    
    Example
    -------
    >>> import pycuda.driver as drv
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import linalg
    >>> linalg.init()
    >>> a = np.array([[1+1j, 2-2j, 3+3j, 4-4j], [5+5j, 6-6j, 7+7j, 8-8j]], np.complex64)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> conj(a_gpu, 2, 2)
    >>> np.all(a == np.conj(a_gpu.get()))
    True
    
    """

    # if tile_dim*block_rows > \
    #        device.get_attribute(drv.device_attribute.MAX_THREADS_PER_BLOCK):
    #     raise ValueError('tile size too large')
    if (tile_dim % block_rows != 0):
        raise ValueError('tile_dim must be an integral multiple of block_rows')
    if (a_gpu.shape[0] % tile_dim != 0) or \
           (a_gpu.shape[1] % tile_dim != 0):
        raise ValueError('tile dimensions must divide matrix dimensions')

    conj_mod = \
             SourceModule(conj_mod_template.substitute(tile_dim=str(tile_dim),
                                                       block_rows=str(block_rows)))

    conj = conj_mod.get_function("conj")
    conj(a_gpu.gpudata,
         np.uint32(a_gpu.shape[1]),
         np.uint32(a_gpu.shape[0]),
         block=(tile_dim, block_rows, 1),
         grid=(a_gpu.shape[1]/tile_dim,
               a_gpu.shape[0]/tile_dim))


diag_mod_template = Template("""
__global__ void diag(${dtype} *v, ${dtype} *a, int N) {
    // Index into input array:
    int vIndex = blockIdx.x * ${tile_dim} + threadIdx.x;

    // Indices into output matrix:
    int xIndex = blockIdx.x * ${tile_dim} + threadIdx.x;
    int yIndex = blockIdx.y * ${tile_dim} + threadIdx.y;
    int aIndex = xIndex + yIndex*N;

    if (xIndex < N && yIndex < N) 
        if (xIndex == yIndex) {
            a[aIndex] = v[vIndex];
        } else {
            a[aIndex] = 0.0;
        }
}
""")

def diag(v_gpu, tile_dim):
    """
    Construct a diagonal matrix.

    Construct a matrix in device memory whose diagonal elements
    correspond to the elements in the specified array; all
    non-diagonal elements are set to 0.

    Parameters
    ----------
    a_obj : GPUArray
        Input array of length `n`.
    tile_dim : int
        Each block of threads processes `tile_dim x tile_dim` elements.

    Notes
    -----
    This function assumes that the input array contains real values.

    Example
    -------
    >>> import pycuda.driver as drv
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import linalg
    >>> linalg.init()
    >>> v = np.array([1, 2, 3, 4, 5, 6], np.float32)
    >>> v_gpu = gpuarray.to_gpu(v)
    >>> d_gpu = diag(v_gpu, 3);
    >>> np.all(d_gpu.get() == np.diag(v))
    True
    
    """

    # if tile_dim**2 > \
    #        device.get_attribute(drv.device_attribute.MAX_THREADS_PER_BLOCK):
    #     raise ValueError('tile size too large')
    n = len(v_gpu)
    if (n % tile_dim != 0):
        raise ValueError('tile dimension must divide input array length')

    if v_gpu.dtype == np.float32:
        dtype = 'float'
    elif v_gpu.dtype == np.complex64:
        dtype = 'complex float'
    else:
        raise ValueError('unsupported array type')

    diag_mod = \
             SourceModule(diag_mod_template.substitute(tile_dim=str(tile_dim),
                                                       dtype=dtype))
    diag = diag_mod.get_function("diag")    
    d_gpu = gpuarray.empty((n, n), v_gpu.dtype)
    diag(v_gpu.gpudata, d_gpu.gpudata, np.uint32(n),
         block=(tile_dim, tile_dim, 1),
         grid=(n/tile_dim, n/tile_dim))
    return d_gpu

def pinv(a_gpu, tile_dim, block_rows):
    """
    Moore-Penrose pseudoinverse.

    Notes
    -----
    
    Example
    -------
    >>> import pycuda.driver as drv
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import linalg
    >>> linalg.init()
    >>> a = np.asarray(np.random.rand(4, 4), np.float32)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> a_inv_gpu = pinv(a_gpu, 2, 2)
    >>> np.allclose(np.linalg.pinv(a), a_inv_gpu.get())
    True

    """
    
    if a_gpu.dtype == np.complex64:
        conj(a_gpu.gpudata, tile_dim, block_rows)
    u_gpu, s_gpu, vh_gpu = svd(a_gpu, 0)
    uh_gpu = transpose(u_gpu, tile_dim, block_rows)
    s_gpu **= np.float32(-1)
    s_diag_gpu = diag(s_gpu, tile_dim)
    v_gpu = transpose(vh_gpu, tile_dim, block_rows)
    suh_gpu = dot(s_diag_gpu, uh_gpu)
    return dot(v_gpu, suh_gpu)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
