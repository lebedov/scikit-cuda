#!/usr/bin/env python

"""
Python interface to CULA toolkit.
"""

import ctypes
import atexit
import numpy as np

from numpy.linalg import LinAlgError

try:
    _libcula = ctypes.cdll.LoadLibrary('libcula.so')
except OSError:
    print 'libcula.so not found'

try:
    _libcuda = ctypes.cdll.LoadLibrary('libcuda.so')
except OSError:
    print 'libcuda.so not found'
    
try:
    _libcudart = ctypes.cdll.LoadLibrary('libcudart.so')
except OSError:
    print 'libcudart.so not found'

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
            raise culaErrors[status]
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

# Function for retrieving string associated with specific CUDA runtime
# error code:
_cudaGetErrorString = _libcudart.cudaGetErrorString
_cudaGetErrorString.restype = ctypes.c_char_p
_cudaGetErrorString.argtypes = [ctypes.c_int]
def cudaGetErrorString(e):
    """Get string associated with the specified CUDA error status
    code."""

    return _cudaGetErrorString(e)


# Generic CUDA error:
class cudaError(Exception):
    """CUDA error."""

    pass

# Exceptions corresponding to various CUDA runtime errors:
class cudaErrorMissingConfiguration(cudaError):
    __doc__ = _cudaGetErrorString(1)
    pass

class cudaErrorMemoryAllocation(cudaError):
    __doc__ = _cudaGetErrorString(2)
    pass

class cudaErrorInitializationError(cudaError):
    __doc__ = _cudaGetErrorString(3)
    pass

class cudaErrorLaunchFailure(cudaError):
    __doc__ = _cudaGetErrorString(4)
    pass

class cudaErrorPriorLaunchFailure(cudaError):
    __doc__ = _cudaGetErrorString(5)
    pass

class cudaErrorLaunchTimeout(cudaError):
    __doc__ = _cudaGetErrorString(6)
    pass

class cudaErrorLaunchOutOfResources(cudaError):
    __doc__ = _cudaGetErrorString(7)
    pass

class cudaErrorInvalidDeviceFunction(cudaError):
    __doc__ = _cudaGetErrorString(8)
    pass

class cudaErrorInvalidConfiguration(cudaError):
    __doc__ = _cudaGetErrorString(9)
    pass

class cudaErrorInvalidDevice(cudaError):
    __doc__ = _cudaGetErrorString(10)
    pass

class cudaErrorInvalidValue(cudaError):
    __doc__ = _cudaGetErrorString(11)
    pass

class cudaErrorInvalidPitchValue(cudaError):
    __doc__ = _cudaGetErrorString(12)
    pass

class cudaErrorInvalidSymbol(cudaError):
    __doc__ = _cudaGetErrorString(13)
    pass

class cudaErrorMapBufferObjectFailed(cudaError):
    __doc__ = _cudaGetErrorString(14)
    pass

class cudaErrorUnmapBufferObjectFailed(cudaError):
    __doc__ = _cudaGetErrorString(15)
    pass

class cudaErrorInvalidHostPointer(cudaError):
    __doc__ = _cudaGetErrorString(16)
    pass

class cudaErrorInvalidDevicePointer(cudaError):
    __doc__ = _cudaGetErrorString(17)
    pass

class cudaErrorInvalidTexture(cudaError):
    __doc__ = _cudaGetErrorString(18)
    pass

class cudaErrorInvalidTextureBinding(cudaError):
    __doc__ = _cudaGetErrorString(19)
    pass

class cudaErrorInvalidChannelDescriptor(cudaError):
    __doc__ = _cudaGetErrorString(20)
    pass

class cudaErrorInvalidMemcpyDirection(cudaError):
    __doc__ = _cudaGetErrorString(21)
    pass

class cudaErrorTextureFetchFailed(cudaError):
    __doc__ = _cudaGetErrorString(23)
    pass

class cudaErrorTextureNotBound(cudaError):
    __doc__ = _cudaGetErrorString(24)
    pass

class cudaErrorSynchronizationError(cudaError):
    __doc__ = _cudaGetErrorString(25)
    pass

class cudaErrorInvalidFilterSetting(cudaError):
    __doc__ = _cudaGetErrorString(26)
    pass

class cudaErrorInvalidNormSetting(cudaError):
    __doc__ = _cudaGetErrorString(27)
    pass

class cudaErrorMixedDeviceExecution(cudaError):
    __doc__ = _cudaGetErrorString(28)
    pass

class cudaErrorUnknown(cudaError):
    __doc__ = _cudaGetErrorString(30)
    pass

class cudaErrorNotYetImplemented(cudaError):
    __doc__ = _cudaGetErrorString(31)
    pass

class cudaErrorMemoryValueTooLarge(cudaError):
    __doc__ = _cudaGetErrorString(32)
    pass

class cudaErrorInvalidResourceHandle(cudaError):
    __doc__ = _cudaGetErrorString(33)
    pass

class cudaErrorNotReady(cudaError):
    __doc__ = _cudaGetErrorString(34)
    pass

class cudaErrorInsufficientDriver(cudaError):
    __doc__ = _cudaGetErrorString(35)
    pass

class cudaErrorSetOnActiveProcess(cudaError):
    __doc__ = _cudaGetErrorString(36)
    pass

class cudaErrorInvalidSurface(cudaError):
    __doc__ = _cudaGetErrorString(37)
    pass

class cudaErrorNoDevice(cudaError):
    __doc__ = _cudaGetErrorString(38)
    pass

class cudaErrorECCUncorrectable(cudaError):
    __doc__ = _cudaGetErrorString(39)
    pass

class cudaErrorSharedObjectSymbolNotFound(cudaError):
    __doc__ = _cudaGetErrorString(40)
    pass

class cudaErrorSharedObjectInitFailed(cudaError):
    __doc__ = _cudaGetErrorString(41)
    pass

class cudaErrorUnsupportedLimit(cudaError):
    __doc__ = _cudaGetErrorString(42)
    pass

class cudaErrorDuplicateVariableName(cudaError):
    __doc__ = _cudaGetErrorString(43)
    pass

class cudaErrorDuplicateTextureName(cudaError):
    __doc__ = _cudaGetErrorString(44)
    pass

class cudaErrorDuplicateSurfaceName(cudaError):
    __doc__ = _cudaGetErrorString(45)
    pass

class cudaErrorDevicesUnavailable(cudaError):
    __doc__ = _cudaGetErrorString(46)
    pass

class cudaErrorStartupFailure(cudaError):
    __doc__ = _cudaGetErrorString(127)
    pass

cudaExceptions = {
    1: cudaErrorMissingConfiguration,
    2: cudaErrorMemoryAllocation, 
    3: cudaErrorInitializationError,
    4: cudaErrorLaunchFailure,
    5: cudaErrorPriorLaunchFailure,
    6: cudaErrorLaunchTimeout,
    7: cudaErrorLaunchOutOfResources,
    8: cudaErrorInvalidDeviceFunction,
    9: cudaErrorInvalidConfiguration,
    10: cudaErrorInvalidDevice,
    11: cudaErrorInvalidValue,
    12: cudaErrorInvalidPitchValue,
    13: cudaErrorInvalidSymbol,
    14: cudaErrorMapBufferObjectFailed,
    15: cudaErrorUnmapBufferObjectFailed,
    16: cudaErrorInvalidHostPointer,
    17: cudaErrorInvalidDevicePointer,
    18: cudaErrorInvalidTexture,
    19: cudaErrorInvalidTextureBinding,
    20: cudaErrorInvalidChannelDescriptor,
    21: cudaErrorInvalidMemcpyDirection,
    22: cudaError,
    23: cudaErrorTextureFetchFailed,
    24: cudaErrorTextureNotBound,
    25: cudaErrorSynchronizationError,
    26: cudaErrorInvalidFilterSetting,
    27: cudaErrorInvalidNormSetting,
    28: cudaErrorMixedDeviceExecution,
    29: cudaError,
    30: cudaErrorUnknown,
    31: cudaErrorNotYetImplemented,
    32: cudaErrorMemoryValueTooLarge,
    33: cudaErrorInvalidResourceHandle,
    34: cudaErrorNotReady,
    35: cudaErrorInsufficientDriver,
    36: cudaErrorSetOnActiveProcess,
    37: cudaErrorInvalidSurface,
    38: cudaErrorNoDevice,
    39: cudaErrorECCUncorrectable,
    40: cudaErrorSharedObjectSymbolNotFound,
    41: cudaErrorSharedObjectInitFailed,
    42: cudaErrorUnsupportedLimit,
    43: cudaErrorDuplicateVariableName,
    44: cudaErrorDuplicateTextureName,
    45: cudaErrorDuplicateSurfaceName,
    46: cudaErrorDevicesUnavailable,
    127: cudaErrorStartupFailure
    # what about cudaErrorApiFailureBase?
    }

def cudaCheckStatus(status):
    """Raise an exception if the specified CUDA status is an error."""

    if status != 0:
        try:
            raise cudaExceptions[status]
        except KeyError:
            raise cudaError
            
# Memory allocation functions (adapted from pystream):
_cudaMalloc = _libcudart.cudaMalloc
_cudaMalloc.restype = int
_cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                        ctypes.c_size_t]
def cuda_malloc(count, ctype=None):
    """Allocate `count` bytes in GPU memory."""
    
    ptr = ctypes.c_void_p()
    status = _cudaMalloc(ctypes.byref(ptr), count)
    if ctype != None:
        ptr = ctypes.cast(ptr, ctypes.POINTER(ctype))
    return ptr

_cudaFree = _libcudart.cudaFree
_cudaFree.restype = int
_cudaFree.argtypes = [ctypes.c_void_p]
def cuda_free(ptr):
    """Free the device memory at the specified pointer."""
    
    status = _cudaFree(ptr)

_cudaMallocPitch = _libcudart.cudaMallocPitch
_cudaMallocPitch.restype = int
_cudaMallocPitch.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                             ctypes.POINTER(ctypes.c_size_t),
                             ctypes.c_size_t, ctypes.c_size_t]
def cuda_malloc_pitch(pitch, rows, cols, elesize):
    """Allocate memory on the device with a specific pitch."""
    
    ptr = ctypes.c_void_p()
    status = _cudaMallocPitch(ctypes.byref(ptr),
                              ctypes.c_size_t(pitch), cols*elesize,
                              rows)
    return ptr, pitch

_cudaMemcpy = _libcudart.cudaMemcpy
_cudaMemcpy.restype = int
_cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_size_t, ctypes.c_int]
memcpyHostToHost = 0
memcpyHostToDevice = 1
memcpyDeviceToHost = 2
memcpyDeviceToDevice = 3
def cuda_memcpy_htod(dst, src, count):
    """Copy `count` bytes of memory from the host memory pointer `src`
    to the device memory pointer `dst`."""
    
    status = _cudaMemcpy(dst, src,
                         ctypes.c_size_t(count), memcpyHostToDevice)
def cuda_memcpy_dtoh(dst, src, count):
    """Copy `count` bytes of memory from the the device memory pointer `src` 
    to the host memory pointer `dst` ."""

    status = _cudaMemcpy(dst, src,
                         ctypes.c_size_t(count), memcpyDeviceToHost)
                         
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

def svd(a, full_matrices=1, compute_uv=1):
    """
    Singular Value Decomposition.

    Factorizes the matrix `a` into two unitary matrices, ``U`` and
    ``Vh``, and a 1-dimensional array of singular values, ``s`` (real,
    non-negative), such that ``a == U S Vh``, where ``S`` is the
    diagonal matrix ``np.diag(s)``.

    Parameters
    ----------
    a : array_like
        Matrix of shape ``(M, N)`` to decompose.
    full_matrices : bool, optional
        If True (default), ``u`` and ``v.H`` have the shapes
        ``(M, M)`` and ``(N, N)``, respectively.  Otherwise, the shapes
        are ``(M, K)`` and ``(K, N)``, resp., where ``K = min(M, N)``.
    compute_uv : bool, optional
        Whether or not to compute ``u`` and ``v.H`` in addition to ``s``.
        True by default.

    Returns
    -------
    u : ndarray
        Unitary matrix. The shape of `U` is ``(M, M)`` or ``(M, K)``
        depending on value of `full_matrices`.
    s : ndarray
        The singular values, sorted so that ``s[i] >= s[i+1]``.
        `S` is a 1-D array of length ``min(M, N)``
    v.H : ndarray
        Unitary matrix of shape ``(N, N)`` or ``(K, N)``, depending
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

    Examples
    --------
    >>> import numpy as np
    >>> import cula
    >>> cula.culaInitialize()
    0

    >>> a = np.random.randn(6, 3) + 1j*np.random.randn(6, 3)
    >>> a = np.asarray(a, np.complex64)
    >>> U, s, Vh = cula.svd(a)
    >>> U.shape, Vh.shape, s.shape
    ((6, 6), (3, 3), (3,))

    >>> U, s, Vh = cula.svd(a, full_matrices=False)
    >>> U.shape, Vh.shape, s.shape
    ((6, 3), (3, 3), (3,))
    >>> S = np.diag(s)
    >>> np.allclose(a, np.dot(U, np.dot(S, Vh)))
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
        cula_func = _libcula.culaCgesvd        
    else:
        a_dtype = np.float32
        cula_func = _libcula.culaSgesvd
    cula_func.restype = int
    cula_func.argtypes = [ctypes.c_char,
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

def svd_device(a, a_dtype, a_shape, full_matrices=1, compute_uv=1):
    """
    Singular Value Decomposition.

    Factorizes the matrix `a` into two unitary matrices, ``U`` and
    ``Vh``, and a 1-dimensional array of singular values, ``s`` (real,
    non-negative), such that ``a == U.T S Vh.T``, where ``S`` is the
    diagonal matrix ``np.diag(s)``.

    Parameters
    ----------
    a : c_void_p
        Pointer to device memory containing
        matrix of shape ``(M, N)`` to decompose.
    a_dtype : type
        Type of matrix data.
    a_shape : tuple
        Shape of matrix data ``(M, N)``.
    full_matrices : bool, optional
        If True (default), ``U`` and ``Vh`` have the shapes
        ``(M, M)`` and ``(N, N)``, respectively.  Otherwise, the shapes
        are ``(K, M)`` and ``(N, K)``, resp., where ``K = min(M, N)``.
    compute_uv : bool, optional
        Whether or not to compute ``U`` and ``Vh`` in addition to ``s``.
        True by default.

    Returns
    -------
    U : c_void_p
        Pointer to device memory containing unitary matrix.
        The shape of `U` is ``(M, M)`` or ``(K, M)``
        depending on value of `full_matrices`.
    s : c_void_p
        Pointer to device memory containing 
        the singular values, sorted so that ``s[i] >= s[i+1]``.
        `s` is a 1-D array of length ``min(M, N)``
    Vh : c_void_p
        Pointer to device memory containing
        unitary matrix of shape ``(N, N)`` or ``(N, K)``, depending
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

    Examples
    --------
    >>> import numpy as np
    >>> import cula
    >>> cula.culaInitialize()
    0
    >>> a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
    >>> a = np.asarray(a, np.complex64)
    >>> m, n = a.shape
    >>> at = a.T.copy()
    >>> a_ptr = cula.cuda_malloc(a.nbytes)
    >>> cula.cuda_memcpy_htod(a_ptr, at.ctypes.data, at.nbytes)
    >>> U_ptr, s_ptr, Vh_ptr = cula.svd_device(a_ptr, a.dtype, a.shape, full_matrices=False)
    >>> U = np.empty((min(m, n), m), a.dtype)
    >>> s = np.empty(min(m, n), np.float32)
    >>> Vh = np.empty((n, min(m, n)), a.dtype)
    >>> cula.cuda_memcpy_dtoh(U.ctypes.data, U_ptr, U.nbytes)
    >>> cula.cuda_memcpy_dtoh(s.ctypes.data, s_ptr, s.nbytes)
    >>> cula.cuda_memcpy_dtoh(Vh.ctypes.data, Vh_ptr, Vh.nbytes)
    >>> S = np.diag(s)
    >>> np.allclose(a, np.dot(U.T, np.dot(S, Vh.T)))
    True

    """

    # The free version of CULA only supports single precision floating
    # point numbers:
    real_dtype = np.float32
    real_dtype_nbytes = np.nbytes[real_dtype]
    if a_dtype == np.complex64:
        cula_func = _libcula.culaDeviceCgesvd        
        a_dtype_nbytes = np.nbytes[a_dtype]
    elif a_dtype == np.float32:
        cula_func = _libcula.culaDeviceSgesvd
        a_dtype_nbytes = np.nbytes[a_dtype]
    else:
        raise ValueError('unsupported type')

    cula_func.restype = int
    cula_func.argtypes = [ctypes.c_char,
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

    (m, n) = a_shape
    
    # Set LDA:
    lda = max(1, m)

    # Set S:
    s = cuda_malloc(min(m, n)*real_dtype_nbytes)
    
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
        u = cuda_malloc(ldu*m*a_dtype_nbytes)
    elif jobu == 'S':
        u = cuda_malloc(min(m, n)*ldu*a_dtype_nbytes)
    else:
        ldu = 1
        u = cuda_malloc(a_dtype_nbytes)
        
    # Set LDVT and transpose of VT:
    if jobvt == 'A':
        ldvt = n
        vt = cuda_malloc(n*n*a_dtype_nbytes)
    elif jobvt == 'S':
        ldvt = min(m, n)
        vt = cuda_malloc(n*ldvt*a_dtype_nbytes)
    else:
        ldvt = 1
        vt = cuda_malloc(a_dtype_nbytes)
        
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
        cuda_free(u)
        cuda_free(vt)
        return s


if __name__ == "__main__":
    import doctest
    doctest.testmod()
