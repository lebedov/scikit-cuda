#!/usr/bin/env python

"""
Python interface to CULA toolkit.
"""

import sys
import ctypes
import atexit
import numpy as np
from numpy.linalg import LinAlgError

import cuda

# Load CULA library:
if sys.platform == 'linux2':
    _libcula_libname = 'libcula.so'
elif sys.platform == 'darwin':
    _libcula_libname = 'libcula.dylib'
else:
    raise RuntimeError('unsupported platform')

try:
    _libcula = ctypes.cdll.LoadLibrary(_libcula_libname)
except OSError:
    print '%s not found' % _libcula_libname

# Needed because of how ctypes handles None on 64-bit platforms.
def POINTER(obj):
    p = ctypes.POINTER(obj)
    if not isinstance(p.from_param, classmethod):
        def from_param(cls, x):
            if x is None:
                return cls()
            else:
                return x
        p.from_param = classmethod(from_param)

    return p

# Function for retrieving string associated with specific CULA error
# code:
_culaGetStatusString = _libcula.culaGetStatusString
_culaGetStatusString.restype = ctypes.c_char_p
_culaGetStatusString.argtypes = [ctypes.c_int]
def culaGetStatusString(e):
    """Get string associated with the specified CULA error status code."""

    return _culaGetStatusString(e)

# Generic CULA error:
class culaError(Exception):
    """CULA error."""

    pass

# Exceptions corresponding to various CULA errors:
class culaNotFound(culaError):
    """CULA shared library not found"""
    pass

class culaNotInitialized(culaError):
    __doc__ = _culaGetStatusString(1)
    pass

class culaNoHardware(culaError):
    __doc__ = _culaGetStatusString(2)
    pass

class culaInsufficientRuntime(culaError):
    __doc__ = _culaGetStatusString(3)
    pass

class culaInsufficientComputeCapability(culaError):
    __doc__ = _culaGetStatusString(4)
    pass

class culaInsufficientMemory(culaError):
    __doc__ = _culaGetStatusString(5)
    pass

class culaFeatureNotImplemented(culaError):
    __doc__ = _culaGetStatusString(6)
    pass

class culaArgumentError(culaError):
    __doc__ = _culaGetStatusString(7)
    pass

class culaDataError(culaError):
    __doc__ = _culaGetStatusString(8)
    pass

class culaBlasError(culaError):
    __doc__ = _culaGetStatusString(9)
    pass

class culaRuntimeError(culaError):
    __doc__ = _culaGetStatusString(10)
    pass

culaExceptions = {
    -1: culaNotFound,
    1: culaNotInitialized,
    2: culaNoHardware,
    3: culaInsufficientRuntime,
    4: culaInsufficientComputeCapability,
    5: culaInsufficientMemory,
    6: culaFeatureNotImplemented,
    7: culaArgumentError,
    8: culaDataError,
    9: culaBlasError,
    10: culaRuntimeError,
    }

# CULA functions:
def culaCheckStatus(status):
    """Raise an exception corresponding to the specified CULA status
    code."""
    
    if status != 0:
        try:
            raise culaExceptions[status]
        except KeyError:
            raise culaError

def culaGetErrorInfo(e):
    """Returns extended information about the last CULA error."""

    return _libcula.culaGetErrorInfo(e)

def culaGetLastStatus():
    """Returns the last status code returned from a CULA function."""
    
    return _libcula.culaGetLastStatus()

def culaInitialize():
    """Must be called before using any other CULA function."""
    
    return _libcula.culaInitialize()

def culaShutdown():
    """Shuts down CULA."""
    
    return _libcula.culaShutdown()

# Shut down CULA upon exit:
atexit.register(_libcula.culaShutdown)

# Functions copied from numpy.linalg:
def _makearray(a):
    new = np.asarray(a)
    wrap = getattr(a, "__array_prepare__", new.__array_wrap__)
    return new, wrap

def _assertRank2(*arrays):
    for a in arrays:
        if len(a.shape) != 2:
            raise LinAlgError, '%d-dimensional array given. Array must be \
            two-dimensional' % len(a.shape)

def _assertNonEmpty(*arrays):
    for a in arrays:
        if np.size(a) == 0:
            raise LinAlgError("Arrays cannot be empty")

def _fastCopyAndTranspose(type, a):
    if a.dtype.type is type:
        return np.fastCopyAndTranspose(a)
    else:
        return np.fastCopyAndTranspose(a.astype(type))

_culaSgesvd = _libcula.culaSgesvd
_culaSgesvd.restype = int
_culaSgesvd.argtypes = [ctypes.c_char,
                        ctypes.c_char,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_void_p,
                        ctypes.c_int,
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_int,
                        ctypes.c_void_p,
                        ctypes.c_int]
_culaCgesvd = _libcula.culaCgesvd
_culaCgesvd.restype = int
_culaCgesvd.argtypes = [ctypes.c_char,
                        ctypes.c_char,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_void_p,
                        ctypes.c_int,
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_int,
                        ctypes.c_void_p,
                        ctypes.c_int]

def svd(a, full_matrices=1, compute_uv=1):
    """
    Singular Value Decomposition.

    Factors the matrix `a` into two unitary matrices, `u` and `vh`,
    and a 1-dimensional array of real, non-negative singular values,
    `s`, such that `a == dot(u, dot(diag(s), vh))`.

    Parameters
    ----------
    a : array_like
        Matrix of shape `(m, n)` to decompose.
    full_matrices : bool, optional
        If True (default), `u` and `vh` have the shapes
        `(m, m)` and `(n, n)`, respectively.  Otherwise, the shapes
        are `(m, k)` and `(k, n)`, respectively, where `k = min(m, n)`.
    compute_uv : bool, optional
        If True (default), compute `u` and `vh` in addition to `s`.

    Returns
    -------
    u : ndarray
        Unitary matrix of shape `(m, m)` or `(m, k)`
        depending on value of `full_matrices`.
    s : ndarray
        The singular values of `a`, sorted such that `s[i] >= s[i+1]`.
        `s` is a 1-D array of length `min(m, n)`.
    vh : ndarray
        Unitary matrix of shape `(n, n)` or `(k, n)`, depending
        on the value of `full_matrices`.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    Notes
    -----
    Because of the limitations of the free version of CULA, the
    argument is cast to single precision.
    
    If `a` is a matrix object (as opposed to an `ndarray`), then so
    are all the return values.

    Example
    -------
    >>> import numpy as np
    >>> import cula
    >>> status = cula.culaInitialize()
    >>> a = np.random.randn(6, 3) + 1j*np.random.randn(6, 3)
    >>> a = np.asarray(a, np.complex64)
    >>> u, s, vh = cula.svd(a)
    >>> u.shape, vh.shape, s.shape
    ((6, 6), (3, 3), (3,))

    >>> u, s, vh = cula.svd(a, full_matrices=False)
    >>> u.shape, vh.shape, s.shape
    ((6, 3), (3, 3), (3,))

    >>> s_mat = np.diag(s)
    >>> np.allclose(a, np.dot(u, np.dot(s_mat, vh)))
    True

    >>> s2 = cula.svd(a, compute_uv=False)
    >>> np.allclose(s, s2)
    True

    """

    a, wrap = _makearray(a)

    # Set M and N:
    (m, n) = a.shape

    # The free version of CULA only supports single precision floating
    # point numbers:
    real_dtype = np.float32
    if np.iscomplexobj(a):
        a_dtype = np.complex64
        cula_func = _culaCgesvd        
    else:
        a_dtype = np.float32
        cula_func = _culaSgesvd
                                                    
    a = _fastCopyAndTranspose(a_dtype, a)
    _assertRank2(a)
    _assertNonEmpty(a)
    
    # Set LDA:
    lda = max(1, m)

    # Set S (the singular values are never complex):
    s = np.zeros(min(m, n), real_dtype)

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
        u = np.zeros((ldu, m), a_dtype)
    elif jobu == 'S':
        u = np.zeros((min(m, n), ldu), a_dtype)
    else:
        ldu = 1
        u = np.empty((1, 1), a_dtype)

    # Set LDVT and transpose of VT:
    if jobvt == 'A':
        ldvt = n
        vt = np.zeros((n, n), a_dtype)
    elif jobvt == 'S':
        ldvt = min(m, n)
        vt = np.zeros((n, ldvt), a_dtype)
    else:
        ldvt = 1
        vt = np.empty((1, 1), a_dtype)        

    status = cula_func(jobu, jobvt, m, n, a.ctypes.data,
                       lda, s.ctypes.data, u.ctypes.data,
                       ldu, vt.ctypes.data, ldvt)
    if status != 0:
        status = culaInitialize()
        culaCheckStatus(status)
        status = cula_func(jobu, jobvt, m, n, a.ctypes.data,
                           lda, s.ctypes.data, u.ctypes.data,
                           ldu, vt.ctypes.data, ldvt)
        
    if status > 0:
        raise LinAlgError, 'SVD did not converge'

    if compute_uv:
        return wrap(u.transpose()), s, wrap(vt.transpose())
    else:
        return s

_culaDeviceSgesvd = _libcula.culaDeviceSgesvd
_culaDeviceSgesvd.restype = int
_culaDeviceSgesvd.argtypes = [ctypes.c_char,
                              ctypes.c_char,
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_void_p,
                              ctypes.c_int,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_int,
                              ctypes.c_void_p,
                              ctypes.c_int]

_culaDeviceCgesvd = _libcula.culaDeviceCgesvd
_culaDeviceCgesvd.restype = int
_culaDeviceCgesvd.argtypes = [ctypes.c_char,
                              ctypes.c_char,
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_void_p,
                              ctypes.c_int,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_int,
                              ctypes.c_void_p,
                              ctypes.c_int]

def svd_device(a, a_dtype, a_shape, full_matrices=1, compute_uv=1):
    """
    Singular Value Decomposition.

    Factors the matrix `a` into two unitary matrices, `u` and `vh`,
    and a 1-dimensional array of real, non-negative singular values,
    `s`, such that `a == dot(u.T, dot(diag(s), vh.T))`.

    Parameters
    ----------
    a : c_void_p
        Pointer to device memory containing
        matrix of shape `(m, n)` to decompose.
    a_dtype : {float32, complex64}
        Type of matrix data.
    a_shape : tuple
        Shape of matrix data `(m, n)`.
    full_matrices : bool, optional
        If True (default), `u` and `vh` have the shapes
        `(m, m)` and `(n, n)`, respectively.  Otherwise, the shapes
        are `(m, k)` and `(k, n)`, resp., where `k = min(m, n)`.
    compute_uv : bool, optional
        If True (default), compute `u` and `vh` in addition to `s`.

    Returns
    -------
    u : c_void_p
        Pointer to device memory containing unitary matrix of
        shape `(m, m)` or `(m, k)` depending on value of `full_matrices`.
    s : c_void_p
        Pointer to device memory containing 
        the singular values, sorted such that `s[i] >= s[i+1]`.
        `s` is a 1-D array of length `min(m, n)`.
    vh : c_void_p
        Pointer to device memory containing
        unitary matrix of shape `(n, n)` or `(k, n)`, depending
        on `full_matrices`. 

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    Notes
    -----
    Because of the limitations of the free version of CULA, the
    argument is cast to single precision.
    
    If `a` is a matrix object (as opposed to an `ndarray`), then so are all
    the return values.

    The input and output matrices are stored in column-major format;
    hence, they must be transposed prior to being copied
    between ndarrays and device memory.

    Example
    -------
    >>> import numpy as np
    >>> import cuda
    >>> import cula
    >>> status = cula.culaInitialize()
    >>> a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
    >>> a = np.asarray(a, np.complex64)
    >>> m, n = a.shape
    >>> at = a.T.copy()
    >>> a_ptr = cuda.cudaMalloc(a.nbytes)
    >>> cuda.cuda_memcpy_htod(a_ptr, at.ctypes.data, at.nbytes)
    >>> u_ptr, s_ptr, vh_ptr = cula.svd_device(a_ptr, a.dtype, a.shape, full_matrices=False)
    >>> k = min(m, n)
    >>> u = np.empty((k, m), a.dtype)
    >>> s = np.empty(k, np.float32)
    >>> vh = np.empty((n, k), a.dtype)
    >>> cuda.cuda_memcpy_dtoh(u.ctypes.data, u_ptr, u.nbytes)
    >>> cuda.cuda_memcpy_dtoh(s.ctypes.data, s_ptr, s.nbytes)
    >>> cuda.cuda_memcpy_dtoh(vh.ctypes.data, vh_ptr, vh.nbytes)
    >>> np.allclose(a, np.dot(u.T, np.dot(np.diag(s), vh.T)))
    True

    """

    # The free version of CULA only supports single precision floating
    # point numbers:
    real_dtype = np.float32
    real_dtype_nbytes = np.nbytes[real_dtype]
    if a_dtype == np.complex64:
        cula_func = _culaDeviceCgesvd        
        a_dtype_nbytes = np.nbytes[a_dtype]
    elif a_dtype == np.float32:
        cula_func = _culaDeviceSgesvd
        a_dtype_nbytes = np.nbytes[a_dtype]
    else:
        raise ValueError('unsupported type')

    (m, n) = a_shape
    
    # Set LDA:
    lda = max(1, m)

    # Set S:
    s = cuda.cudaMalloc(min(m, n)*real_dtype_nbytes)
    
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
        u = cuda.cudaMalloc(ldu*m*a_dtype_nbytes)
    elif jobu == 'S':
        u = cuda.cudaMalloc(min(m, n)*ldu*a_dtype_nbytes)
    else:
        ldu = 1
        u = cuda.cudaMalloc(a_dtype_nbytes)
        
    # Set LDVT and transpose of VT:
    if jobvt == 'A':
        ldvt = n
        vt = cuda.cudaMalloc(n*n*a_dtype_nbytes)
    elif jobvt == 'S':
        ldvt = min(m, n)
        vt = cuda.cudaMalloc(n*ldvt*a_dtype_nbytes)
    else:
        ldvt = 1
        vt = cuda.cudaMalloc(a_dtype_nbytes)
        
    status = cula_func(jobu, jobvt, m, n, a,
                       lda, s, u,
                       ldu, vt, ldvt)
    if status != 0:
        status = culaInitialize()
        culaCheckStatus(status)
        status = cula_func(jobu, jobvt, m, n, a,
                           lda, s, u,
                           ldu, vt, ldvt)
        
    if status > 0:
        raise LinAlgError, 'SVD did not converge'

    if compute_uv:
        return u, s, vt
    else:
        cudaFree(u)
        cudaFree(vt)
        return s
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()
