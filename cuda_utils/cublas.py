#!/usr/bin/env python

"""
Python interface to CUBLAS functions.
"""

import sys
import ctypes
import atexit
import numpy as np

import cuda

if sys.platform == 'linux2':
    _libcublas_libname = 'libcublas.so'
elif sys.platform == 'darwin':
    _libcublas_libname = 'libcublas.dylib'
else:
    raise RuntimeError('unsupported platform')

try:
    _libcublas = ctypes.cdll.LoadLibrary(_libcublas_libname)
except OSError:
    print '%s not found' % _libcublas_libname

# Generic CUBLAS error:
class cublasError(Exception):
    "CUBLAS error"
    
    pass

# Exceptions corresponding to different CUBLAS errors:
class cublasNotInitialized(cublasError):
    __doc__ = "CUBLAS library not initialized."
    pass

class cublasAllocFailed(cublasError):
    __doc__ = "Resource allocation failed."
    pass

class cublasInvalidValue(cublasError):
    __doc__ = "Unsupported numerical value was passed to function."
    pass

class cublasArchMismatch(cublasError):
    __doc__ = "Function requires an architectural feature absent from the device."
    pass

class cublasMappingError(cublasError):
    __doc__ = "Access to GPU memory space failed."""
    pass

class cublasExecutionFailed(cublasError):
    __doc__ = "GPU program failed to execute."
    pass

class cublasInternalError(cublasError):
    __doc__ = "An internal CUBLAS operation failed."
    pass

cublasExceptions = {
    0x1: cublasNotInitialized,
    0x3: cublasAllocFailed,
    0x7: cublasInvalidValue,
    0x8: cublasArchMismatch,
    0xb: cublasMappingError,
    0xd: cublasExecutionFailed,
    0xe: cublasInternalError,
    }

_cublasGetError = _libcublas.cublasGetError
_cublasGetError.restype = int
_cublasGetError.argtypes = []
def cublasGetError():
    """Returns and resets the current CUBLAS error code."""

    return _cublasGetError()

_cublasInit = _libcublas.cublasInit
_cublasInit.restype = int
_cublasInit.argtypes = []
def cublasInit():
    """Must be called before using any other CUBLAS functions."""
    
    return _cublasInit()

_cublasShutdown = _libcublas.cublasShutdown
_cublasShutdown.restype = int
_cublasShutdown.argtypes = []
def cublasShutdown():
    """Shuts down CUBLAS."""

    return _cublasShutdown()

atexit.register(_cublasShutdown)

def cublasCheckStatus(status):
    """Raise an exception if the specified CUBLAS status is an error."""

    if status != 0:
        try:
            raise cublasExceptions[status]
        except KeyError:
            raise cublasError

_cublasSgemm = _libcublas.cublasSgemm
_cublasSgemm.restype = None
_cublasSgemm.argtypes = [ctypes.c_char,
                         ctypes.c_char,
                         ctypes.c_int,
                         ctypes.c_int,
                         ctypes.c_int,
                         ctypes.c_float,
                         ctypes.c_void_p,
                         ctypes.c_int,
                         ctypes.c_void_p,
                         ctypes.c_int,
                         ctypes.c_float,
                         ctypes.c_void_p,
                         ctypes.c_int]
_cublasCgemm = _libcublas.cublasCgemm
_cublasCgemm.restype = None
_cublasCgemm.argtypes = [ctypes.c_char,
                         ctypes.c_char,
                         ctypes.c_int,
                         ctypes.c_int,
                         ctypes.c_int,
                         ctypes.c_float,
                         ctypes.c_void_p,
                         ctypes.c_int,
                         ctypes.c_void_p,
                         ctypes.c_int,
                         ctypes.c_float,
                         ctypes.c_void_p,
                         ctypes.c_int]

def dot_device(a_ptr, a_shape, b_ptr, b_shape, a_dtype):
    """
    Matrix product of two arrays.

    Computes the matrix product of two arrays of shapes `(m, k)` and
    `(k, n)`; the result has shape `(m, n)`.

    Parameters
    ----------
    a_ptr : c_void_p
        Pointer to device memory containing
        matrix of shape `(m, k)`.
    a_shape : tuple
        Shape of matrix `a` data `(m, k)`.
    b_ptr : c_void_p
        Pointer to device memory containing
        matrix of shape `(k, n)`.
    b_shape : tuple
        Shape of matrix `b` data `(k, n)`.
    a_dtype : {float32, complex64}
        Type of matrix.

    Returns
    -------
    c_ptr : c_void_p
        Pointer to device memory containing matrix product of shape
        `(m, n)`.

    Notes
    -----
    The input and output matrices are stored in column-major format;
    hence, they must be transposed prior to being copied
    between ndarrays and device memory.

    Example
    -------
    >>> import numpy as np
    >>> import cuda
    >>> import cublas
    >>> cublas.cublasInit()
    0
    
    >>> a = np.asarray(np.random.rand(4, 2), np.float32)
    >>> b = np.asarray(np.random.rand(2, 2), np.float32)
    >>> c = np.empty((4, 2), np.float32)
    >>> at = a.T.copy()
    >>> bt = b.T.copy()
    >>> ct = c.T.copy()
    >>> a_ptr = cuda.cudaMalloc(a.nbytes)
    >>> b_ptr = cuda.cudaMalloc(b.nbytes)
    >>> cuda.cuda_memcpy_htod(a_ptr, at.ctypes.data, a.nbytes)
    >>> cuda.cuda_memcpy_htod(b_ptr, bt.ctypes.data, b.nbytes)
    >>> c_ptr = cublas.dot_device(a_ptr, a.shape, b_ptr, b.shape, a.dtype)
    >>> cuda.cuda_memcpy_dtoh(ct.ctypes.data, c_ptr, c.nbytes)
    >>> np.allclose(np.dot(a, b), ct.T)
    True
    
    """

    if (a_dtype == np.complex64):
        cublas_func = _cublasCgemm        
        a_dtype_nbytes = np.nbytes[a_dtype]
        alpha = np.complex64(1.0)
        beta = np.complex64(0.0)
    elif a_dtype == np.float32:
        cublas_func = _cublasSgemm
        a_dtype_nbytes = np.nbytes[a_dtype]
        alpha = np.float32(1.0)
        beta = np.float32(0.0)
    else:
        raise ValueError('unsupported type')

    transa = 'N'
    transb = 'N'
    m = a_shape[0]
    n = b_shape[1]
    k = a_shape[1]
    lda = m
    ldb = k
    ldc = max(1, m)
    
    c_ptr = cuda.cudaMalloc(ldc*n*a_dtype_nbytes)
    cublas_func(transa, transb, m, n, k, alpha, a_ptr, lda, b_ptr, ldb, beta,
                c_ptr, ldc)

    status = _cublasGetError()
    cublasCheckStatus(status)
    
    return c_ptr

if __name__ == "__main__":
    import doctest
    doctest.testmod()
