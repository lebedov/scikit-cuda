#!/usr/bin/env python

"""
Python interface to CULA toolkit.
"""

import sys
import ctypes
import atexit
import numpy as np

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
    raise RuntimeError('%s not found' % _libcula_libname)

# Check whether the basic or premium version of the toolkit is
# installed by trying to access a function that is only available in
# the latter:
try:
    _libcula.culaDeviceMalloc
except AttributeError:
    _libcula_toolkit = 'basic'
else:
    _libcula_toolkit = 'premium'
    
# Function for retrieving string associated with specific CULA error
# code:
_libcula.culaGetStatusString.restype = ctypes.c_char_p
_libcula.culaGetStatusString.argtypes = [ctypes.c_int]
def culaGetStatusString(e):
    """
    Get string associated with the specified CULA status code.

    Parameters
    ----------
    e : int
        Status code.

    Returns
    -------
    s : str
        Status string.
        
    """

    return _libcula.culaGetStatusString(e)

# Generic CULA error:
class culaError(Exception):
    """CULA error."""
    pass

# Exceptions corresponding to various CULA errors:
class culaNotFound(culaError):
    """CULA shared library not found"""
    pass

class culaPremiumNotFound(culaError):
    """Premium CULA toolkit unavailable"""
    pass

class culaNotInitialized(culaError):
    __doc__ = culaGetStatusString(1)
    pass

class culaNoHardware(culaError):
    __doc__ = culaGetStatusString(2)
    pass

class culaInsufficientRuntime(culaError):
    __doc__ = culaGetStatusString(3)
    pass

class culaInsufficientComputeCapability(culaError):
    __doc__ = culaGetStatusString(4)
    pass

class culaInsufficientMemory(culaError):
    __doc__ = culaGetStatusString(5)
    pass

class culaFeatureNotImplemented(culaError):
    __doc__ = culaGetStatusString(6)
    pass

class culaArgumentError(culaError):
    __doc__ = culaGetStatusString(7)
    pass

class culaDataError(culaError):
    __doc__ = culaGetStatusString(8)
    pass

class culaBlasError(culaError):
    __doc__ = culaGetStatusString(9)
    pass

class culaRuntimeError(culaError):
    __doc__ = culaGetStatusString(10)
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
_libcula.culaGetErrorInfo.restype = int
def culaGetErrorInfo():
    """
    Returns extended information code for the last CULA error.

    Returns
    -------
    err : int
        Extended information code.
        
    """

    return _libcula.culaGetErrorInfo()

_libcula.culaGetErrorInfoString.restype = int
_libcula.culaGetErrorInfoString.argtypes = [ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_int]
def culaGetErrorInfoString(e, i, bufsize=100):
    """
    Returns a readable CULA error string.

    Returns a readable error string corresponding to a given CULA
    error code and extended error information code.

    Parameters
    ----------
    e : int
        CULA error code.
    i : int
        Extended information code.
    bufsize : int
        Length of string to return.

    Returns
    -------
    s : str
        Error string.
        
    """

    buf = ctypes.create_string_buffer(bufsize)
    status = _libcula.culaGetErrorInfoString(e, i, buf, bufsize)
    culaCheckStatus(status)
    return buf.value
    
def culaGetLastStatus():
    """
    Returns the last status code returned from a CULA function.

    Returns
    -------
    s : int
        Status code.
        
    """
    
    return _libcula.culaGetLastStatus()

def culaCheckStatus(status):
    """
    Raise an exception corresponding to the specified CULA status
    code.

    Parameters
    ----------
    status : int
        CULA status code.
        
    """
    
    if status != 0:
        error = culaGetErrorInfo()
        try:
            raise culaExceptions[status](error)
        except KeyError:
            raise culaError(error)

_libcula.culaSelectDevice.restype = int
_libcula.culaSelectDevice.argtypes = [ctypes.c_int]
def culaSelectDevice(dev):
    """
    Selects a device with which CULA will operate.

    Parameters
    ----------
    dev : int
        GPU device number.
        
    Notes
    -----
    Must be called before `culaInitialize`.
    
    """

    status = _libcula.culaSelectDevice(dev)
    culaCheckStatus(status)

_libcula.culaGetExecutingDevice.restype = int
_libcula.culaGetExecutingDevice.argtypes = [ctypes.c_void_p]
def culaGetExecutingDevice():
    """
    Reports the id of the GPU device used by CULA.

    Returns
    -------
    dev : int
       Device id.

    """

    dev = ctypes.c_int()
    status = _libcula.culaGetExecutingDevice(ctypes.byref(dev))
    culaCheckStatus(status)
    return dev.value

def culaFreeBuffers():
    """
    Releases any memory buffers stored internally by CULA.

    """

    _libcula.culaFreeBuffers()
    
def culaInitialize():
    """
    Initialize CULA.

    Notes
    -----
    Must be called before using any other CULA functions.

    """
    
    status = _libcula.culaInitialize()
    culaCheckStatus(status)

def culaShutdown():
    """
    Shuts down CULA.
    """
    
    status = _libcula.culaShutdown()
    culaCheckStatus(status)

# Shut down CULA upon exit:
atexit.register(_libcula.culaShutdown)

# LAPACK functions available in CULA basic:
_libcula.culaDeviceSgesv.restype = \
_libcula.culaDeviceCgesv.restype = int
_libcula.culaDeviceSgesv.argtypes = \
_libcula.culaDeviceCgesv.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_int]
def culaDeviceSgesv(n, nrhs, a, lda, ipiv, b, ldb):
    """
    Solve linear system with LU factorization.

    """

    status = _libcula.culaDeviceSgesv(n, nrhs, int(a), lda, int(ipiv),
                                      int(b), ldb)
    culaCheckStatus(status)
def culaDeviceCgesv(n, nrhs, a, lda, ipiv, b, ldb):
    """
    Solve linear system with LU factorization.

    """

    status = _libcula.culaDeviceCgesv(n, nrhs, int(a), lda, int(ipiv),
                                      int(b), ldb)
    culaCheckStatus(status)

_libcula.culaDeviceSgetrf.restype = \
_libcula.culaDeviceCgetrf.restype = int
_libcula.culaDeviceSgetrf.argtypes = \
_libcula.culaDeviceCgetrf.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def culaDeviceSgetrf(m, n, a, lda, ipiv):
    """
    LU factorization.

    """
    
    status = _libcula.culaDeviceSgetrf(m, n, int(a), lda, int(ipiv))
    culaCheckStatus(status)
def culaDeviceCgetrf(m, n, a, lda, ipiv):
    """
    LU factorization.

    """
    
    status = _libcula.culaDeviceCgetrf(m, n, int(a), lda, int(ipiv))
    culaCheckStatus(status)

_libcula.culaDeviceSgeqrf.restype = \
_libcula.culaDeviceCgeqrf.restype = int
_libcula.culaDeviceSgeqrf.argtypes = \
_libcula.culaDeviceCgeqrf.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def culaDeviceSgeqrf(m, n, a, lda, tau):
    """
    QR factorization.

    """
    
    status = _libcula.culaDeviceSgeqrf(m, n, int(a), lda, int(tau))
    culaCheckStatus(status)
def culaDeviceCgeqrf(m, n, a, lda, tau):
    """
    QR factorization.

    """
    
    status = _libcula.culaDeviceCgeqrf(m, n, int(a), lda, int(tau))
    culaCheckStatus(status)

_libcula.culaDeviceSgels.restype = \
_libcula.culaDeviceCgels.restype = int
_libcula.culaDeviceSgels.argtypes = \
_libcula.culaDeviceCgels.argtypes = [ctypes.c_char,                           
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,                              
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int]
def culaDeviceSgels(trans, m, n, nrhs, a, lda, b, ldb):
    """
    Solve linear system with QR or LQ factorization.

    """
    
    status = _libcula.culaDeviceSgels(trans, m, n, nrhs, int(a),
                                      lda, int(b), ldb)
    culaCheckStatus(status)
def culaDeviceCgels(trans, m, n, nrhs, a, lda, b, ldb):
    """
    Solve linear system with QR or LQ factorization.

    """

    status = _libcula.culaDeviceCgels(trans, m, n, nrhs, int(a),
                                      lda, int(b), ldb)
    culaCheckStatus(status)

_libcula.culaDeviceSgglse.restype = \
_libcula.culaDeviceCgglse.restype = int
_libcula.culaDeviceSgglse.argtypes = \
_libcula.culaDeviceCgglse.argtypes = [ctypes.c_int,                             
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p]
def culaDeviceSgglse(m, n, p, a, lda, b, ldb, c, d, x):
    """
    Solve linear equality-constrained least squares problem.

    """
    
    status = _libcula.culaDeviceSgglse(m, n, p, int(a), lda, int(b),
                                       ldb, int(c), int(d), int(x))
    culaCheckStatus(status)
def culaDeviceCgglse(m, n, p, a, lda, b, ldb, c, d, x):
    """
    Solve linear equality-constrained least squares problem.

    """

    status = _libcula.culaDeviceCgglse(m, n, p, int(a), lda, int(b),
                                       ldb, int(c), int(d), int(x))
    culaCheckStatus(status)
    
_libcula.culaDeviceSgesvd.restype = \
_libcula.culaDeviceCgesvd.restype = int
_libcula.culaDeviceSgesvd.argtypes = \
_libcula.culaDeviceCgesvd.argtypes = [ctypes.c_char,
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
def culaDeviceSgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt):
    """
    SVD decomposition.

    """
    
    status = _libcula.culaDeviceSgesvd(jobu, jobvt, m, n, int(a), lda,
                                       int(s), int(u), ldu, int(vt),
                                       ldvt)
    culaCheckStatus(status)
def culaDeviceCgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt):
    """
    SVD decomposition.

    """

    status = _libcula.culaDeviceCgesvd(jobu, jobvt, m, n, int(a), lda,
                                       int(s), int(u), ldu, int(vt),
                                       ldvt)
    culaCheckStatus(status)

# LAPACK functions available in CULA premium:
try:
    _libcula.culaDeviceDgesv.restype = \
    _libcula.culaDeviceZgesv.restype = int
    _libcula.culaDeviceDgesv.argtypes = \
    _libcula.culaDeviceZgesv.argtypes = [ctypes.c_int,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_void_p,
                                         ctypes.c_int]
except AttributeError:
    pass
def culaDeviceDgesv(n, nrhs, a, lda, ipiv, b, ldb):
    """
    Solve linear system with LU factorization.

    """

    status = _libcula.culaDeviceDgesv(n, nrhs, int(a), lda, int(ipiv),
                                      int(b), ldb)
    culaCheckStatus(status)
def culaDeviceZgesv(n, nrhs, a, lda, ipiv, b, ldb):
    """
    Solve linear system with LU factorization.

    """

    status = _libcula.culaDeviceZgesv(n, nrhs, int(a), lda, int(ipiv),
                                      int(b), ldb)
    culaCheckStatus(status)

try:
    _libcula.culaDeviceDgetrf.restype = \
    _libcula.culaDeviceZgetrf.restype = int
    _libcula.culaDeviceDgetrf.argtypes = \
    _libcula.culaDeviceZgetrf.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
except AttributeError:
    pass
def culaDeviceDgetrf(m, n, a, lda, ipiv):
    """
    LU factorization.

    """
    
    status = _libcula.culaDeviceDgetrf(m, n, int(a), lda, int(ipiv))
    culaCheckStatus(status)
def culaDeviceZgetrf(m, n, a, lda, ipiv):
    """
    LU factorization.

    """
    
    status = _libcula.culaDeviceZgetrf(m, n, int(a), lda, int(ipiv))
    culaCheckStatus(status)

try:
    _libcula.culaDeviceDgeqrf.restype = \
    _libcula.culaDeviceZgeqrf.restype = int
    _libcula.culaDeviceDgeqrf.argtypes = \
    _libcula.culaDeviceZgeqrf.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
except AttributeError:
    pass
def culaDeviceDgeqrf(m, n, a, lda, tau):
    """
    QR factorization.

    """
    
    status = _libcula.culaDeviceDgeqrf(m, n, int(a), lda, int(tau))
    culaCheckStatus(status)
def culaDeviceZgeqrf(m, n, a, lda, tau):
    """
    QR factorization.

    """
    
    status = _libcula.culaDeviceZgeqrf(m, n, int(a), lda, int(tau))
    culaCheckStatus(status)

try:
    _libcula.culaDeviceDgels.restype = \
    _libcula.culaDeviceZgels.restype = int
    _libcula.culaDeviceDgels.argtypes = \
    _libcula.culaDeviceZgels.argtypes = [ctypes.c_char,                           
                                         ctypes.c_int,
                                         ctypes.c_int,
                                         ctypes.c_int,
                                         ctypes.c_void_p,                              
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int]
except AttributeError:
    pass
def culaDeviceDgels(trans, m, n, nrhs, a, lda, b, ldb):
    """
    Solve linear system with QR or LQ factorization.

    """
    
    status = _libcula.culaDeviceDgels(trans, m, n, nrhs, int(a),
                                      lda, int(b), ldb)
    culaCheckStatus(status)
def culaDeviceZgels(trans, m, n, nrhs, a, lda, b, ldb):
    """
    Solve linear system with QR or LQ factorization.

    """

    status = _libcula.culaDeviceZgels(trans, m, n, nrhs, int(a),
                                      lda, int(b), ldb)
    culaCheckStatus(status)

try:
    _libcula.culaDeviceDgglse.restype = \
    _libcula.culaDeviceZgglse.restype = int
    _libcula.culaDeviceDgglse.argtypes = \
    _libcula.culaDeviceZgglse.argtypes = [ctypes.c_int,                             
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
except AttributeError:
    pass
def culaDeviceDgglse(m, n, p, a, lda, b, ldb, c, d, x):
    """
    Solve linear equality-constrained least squares problem.

    """
    
    status = _libcula.culaDeviceDgglse(m, n, p, int(a), lda, int(b),
                                       ldb, int(c), int(d), int(x))
    culaCheckStatus(status)
def culaDeviceZgglse(m, n, p, a, lda, b, ldb, c, d, x):
    """
    Solve linear equality-constrained least squares problem.

    """

    status = _libcula.culaDeviceZgglse(m, n, p, int(a), lda, int(b),
                                       ldb, int(c), int(d), int(x))
    culaCheckStatus(status)

try:
    _libcula.culaDeviceDgesvd.restype = \
    _libcula.culaDeviceZgesvd.restype = int
    _libcula.culaDeviceDgesvd.argtypes = \
    _libcula.culaDeviceZgesvd.argtypes = [ctypes.c_char,
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
except AttributeError:
    pass
def culaDeviceDgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt):
    """
    SVD decomposition.
    
    """
    
    status = _libcula.culaDeviceDgesvd(jobu, jobvt, m, n, int(a), lda,
                                       int(s), int(u), ldu, int(vt),
                                       ldvt)
    culaCheckStatus(status)
def culaDeviceZgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt):
    """
    SVD decomposition.
    
    """
            
    status = _libcula.culaDeviceZgesvd(jobu, jobvt, m, n, int(a), lda,
                                       int(s), int(u), ldu, int(vt),
                                       ldvt)
    culaCheckStatus(status)


try:
    _libcula.culaDeviceSposv.restype = \
    _libcula.culaDeviceCposv.restype = \
    _libcula.culaDeviceDposv.restype = \
    _libcula.culaDeviceZposv.restype = int
    _libcula.culaDeviceSposv.argtypes = \
    _libcula.culaDeviceCposv.argtypes = \
    _libcula.culaDeviceDposv.argtypes = \
    _libcula.culaDeviceZposv.argtypes = [ctypes.c_char,
                                         ctypes.c_int,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int]
except AttributeError:
    pass
def culaDeviceSposv(upio, n, nrhs, a, lda, b, ldb):
    """
    Solve positive definite linear system with Cholesky factorization.

    """

    status = _libcula.culaDeviceSposv(upio, n, nrhs, int(a), lda, int(b),
                                      ldb)
    culaCheckStatus(status)
def culaDeviceCposv(upio, n, nrhs, a, lda, b, ldb):
    """
    Solve positive definite linear system with Cholesky factorization.

    """

    status = _libcula.culaDeviceCposv(upio, n, nrhs, int(a), lda, int(b),
                                      ldb)
    culaCheckStatus(status)
def culaDeviceDposv(upio, n, nrhs, a, lda, b, ldb):
    """
    Solve positive definite linear system with Cholesky factorization.

    """

    status = _libcula.culaDeviceDposv(upio, n, nrhs, int(a), lda, int(b),
                                      ldb)
    culaCheckStatus(status)
def culaDeviceZposv(upio, n, nrhs, a, lda, b, ldb):
    """
    Solve positive definite linear system with Cholesky factorization.

    """

    status = _libcula.culaDeviceZposv(upio, n, nrhs, int(a), lda, int(b),
                                      ldb)
    culaCheckStatus(status)

try:
    _libcula.culaDeviceSpotrf.restype = \
    _libcula.culaDeviceCpotrf.restype = \
    _libcula.culaDeviceDpotrf.restype = \
    _libcula.culaDeviceZpotrf.restype = int
    _libcula.culaDeviceSpotrf.argtypes = \
    _libcula.culaDeviceCpotrf.argtypes = \
    _libcula.culaDeviceDpotrf.argtypes = \
    _libcula.culaDeviceZpotrf.argtypes = [ctypes.c_char,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
except AttributeError:
    pass
def culaDeviceSpotrf(uplo, n, a, lda):
    """
    Cholesky factorization.

    """
    
    status = _libcula.culaDeviceSpotrf(uplo, n, int(a), lda)
    culaCheckStatus(status)
def culaDeviceCpotrf(uplo, n, a, lda):
    """
    Cholesky factorization.

    """

    status = _libcula.culaDeviceCpotrf(uplo, n, int(a), lda)
    culaCheckStatus(status)
def culaDeviceDpotrf(uplo, n, a, lda):
    """
    Cholesky factorization.

    """

    status = _libcula.culaDeviceDpotrf(uplo, n, int(a), lda)
    culaCheckStatus(status)
def culaDeviceZpotrf(uplo, n, a, lda):
    """
    Cholesky factorization.

    """

    status = _libcula.culaDeviceZpotrf(uplo, n, int(a), lda)
    culaCheckStatus(status)
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()
