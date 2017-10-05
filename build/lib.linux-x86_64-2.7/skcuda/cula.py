#!/usr/bin/env python

"""
Python interface to CULA toolkit.
"""

from __future__ import absolute_import

import sys
import ctypes
import atexit
import numpy as np

from . import cuda

# Load CULA library:
if 'linux' in sys.platform:
    _libcula_libname_list = ['libcula_lapack.so',
                             'libcula_lapack_basic.so',
                             'libcula.so']
elif sys.platform == 'darwin':
    _libcula_libname_list = ['libcula_lapack.dylib',
                             'libcula.dylib']
elif sys.platform == 'win32':
    _libcula_libname_list = ['cula_lapack.dll',
                             'cula_lapack_basic.dll']
else:
    raise RuntimeError('unsupported platform')

_load_err = ''
for _lib in  _libcula_libname_list:
    try:
        _libcula = ctypes.cdll.LoadLibrary(_lib)
    except OSError:
        _load_err += ('' if _load_err == '' else ', ') + _lib
    else:
        _load_err = ''
        break
if _load_err:
    raise OSError('%s not found' % _load_err)

# Check whether the free or standard version of the toolkit is
# installed by trying to access a function that is only available in
# the latter:
try:
    _libcula.culaDeviceMalloc
except AttributeError:
    _libcula_toolkit = 'free'
else:
    _libcula_toolkit = 'standard'

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

class culaStandardNotFound(culaError):
    """Standard CULA Dense toolkit unavailable"""
    pass

class culaNotInitialized(culaError):
    try:
        __doc__ = culaGetStatusString(1)
    except:
        pass
    pass

class culaNoHardware(culaError):
    try:
        __doc__ = culaGetStatusString(2)
    except:
        pass
    pass

class culaInsufficientRuntime(culaError):
    try:
        __doc__ = culaGetStatusString(3)
    except:
        pass
    pass

class culaInsufficientComputeCapability(culaError):
    try:
        __doc__ = culaGetStatusString(4)
    except:
        pass
    pass

class culaInsufficientMemory(culaError):
    try:
        __doc__ = culaGetStatusString(5)
    except:
        pass
    pass

class culaFeatureNotImplemented(culaError):
    try:
        __doc__ = culaGetStatusString(6)
    except:
        pass
    pass

class culaArgumentError(culaError):
    try:
        __doc__ = culaGetStatusString(7)
    except:
        pass
    pass

class culaDataError(culaError):
    try:
        __doc__ = culaGetStatusString(8)
    except:
        pass
    pass

class culaBlasError(culaError):
    try:
        __doc__ = culaGetStatusString(9)
    except:
        pass
    pass

class culaRuntimeError(culaError):
    try:
        __doc__ = culaGetStatusString(10)
    except:
        pass
    pass

class culaBadStorageFormat(culaError):
    try:
        __doc__ = culaGetStatusString(11)
    except:
        pass
    pass

class culaInvalidReferenceHandle(culaError):
    try:
        __doc__ = culaGetStatusString(12)
    except:
        pass
    pass

class culaUnspecifiedError(culaError):
    try:
        __doc__ = culaGetStatusString(13)
    except:
        pass
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
    11: culaBadStorageFormat,
    12: culaInvalidReferenceHandle,
    13: culaUnspecifiedError,
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

_libcula.culaGetVersion.restype = int
def culaGetVersion():
    """
    Report the version number of CULA.

    """

    return _libcula.culaGetVersion()

_libcula.culaGetCudaMinimumVersion.restype = int
def culaGetCudaMinimumVersion():
    """
    Report the minimum version of CUDA required by CULA.

    """

    return _libcula.culaGetCudaMinimumVersion()

_libcula.culaGetCudaRuntimeVersion.restype = int
def culaGetCudaRuntimeVersion():
    """
    Report the version of the CUDA runtime linked to by the CULA library.

    """

    return _libcula.culaGetCudaRuntimeVersion()

_libcula.culaGetCudaDriverVersion.restype = int
def culaGetCudaDriverVersion():
    """
    Report the version of the CUDA driver installed on the system.

    """

    return _libcula.culaGetCudaDriverVersion()

_libcula.culaGetCublasMinimumVersion.restype = int
def culaGetCublasMinimumVersion():
    """
    Report the version of CUBLAS required by CULA.

    """

    return _libcula.culaGetCublasMinimumVersion()

_libcula.culaGetCublasRuntimeVersion.restype = int
def culaGetCublasRuntimeVersion():
    """
    Report the version of CUBLAS linked to by CULA.

    """

    return _libcula.culaGetCublasRuntimeVersion()

_libcula.culaGetDeviceCount.restype = int
def culaGetDeviceCount():
    """
    Report the number of available GPU devices.

    """
    return _libcula.culaGetDeviceCount()

_libcula.culaInitialize.restype = int
def culaInitialize():
    """
    Initialize CULA.

    Notes
    -----
    Must be called before using any other CULA functions.

    """

    status = _libcula.culaInitialize()
    culaCheckStatus(status)

_libcula.culaShutdown.restype = int
def culaShutdown():
    """
    Shuts down CULA.
    """

    status = _libcula.culaShutdown()
    culaCheckStatus(status)

# Shut down CULA upon exit:
atexit.register(_libcula.culaShutdown)

# LAPACK functions available in CULA Dense Free:

# SGESV, CGESV
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

# SGETRF, CGETRF
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

# SGEQRF, CGEQRF
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

# SORGQR, CUNGQR
try:
    _libcula.culaDeviceSorgqr.restype = \
    _libcula.culaDeviceCungqr.restype = int
    _libcula.culaDeviceSorgqr.argtypes = \
    _libcula.culaDeviceCungqr.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
except AttributeError:
    def culaDeviceSorgqr(m, n, k, a, lda, tau):
        """
        QR factorization - Generate Q from QR factorization
        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceCungqr(m, n, k, a, lda, tau):
        """
        QR factorization - Generate Q from QR factorization
        """
        
        raise NotImplementedError('CULA Dense required')
else:
    def culaDeviceSorgqr(m, n, k, a, lda, tau):
        """
        QR factorization - Generate Q from QR factorization
        """

        status = _libcula.culaDeviceSorgqr(m, n, k, int(a), lda, int(tau))
        culaCheckStatus(status)

    def culaDeviceCungqr(m, n, k, a, lda, tau):
        """
        QR factorization - Generate Q from QR factorization
        """

        status = _libcula.culaDeviceCungqr(m, n, k, int(a), lda, int(tau))
        culaCheckStatus(status)

# SGELS, CGELS
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
    trans = trans.encode('ascii')
    status = _libcula.culaDeviceSgels(trans, m, n, nrhs, int(a),
                                      lda, int(b), ldb)
    culaCheckStatus(status)

def culaDeviceCgels(trans, m, n, nrhs, a, lda, b, ldb):
    """
    Solve linear system with QR or LQ factorization.

    """
    trans = trans.encode('ascii')
    status = _libcula.culaDeviceCgels(trans, m, n, nrhs, int(a),
                                      lda, int(b), ldb)
    culaCheckStatus(status)

# SGGLSE, CGGLSE
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

# SGESVD, CGESVD
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
    jobu = jobu.encode('ascii')
    jobvt = jobvt.encode('ascii')
    status = _libcula.culaDeviceSgesvd(jobu, jobvt, m, n, int(a), lda,
                                       int(s), int(u), ldu, int(vt),
                                       ldvt)
    culaCheckStatus(status)

def culaDeviceCgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt):
    """
    SVD decomposition.

    """
    jobu = jobu.encode('ascii')
    jobvt = jobvt.encode('ascii')
    status = _libcula.culaDeviceCgesvd(jobu, jobvt, m, n, int(a), lda,
                                       int(s), int(u), ldu, int(vt),
                                       ldvt)
    culaCheckStatus(status)

# LAPACK functions available in CULA Dense:

# DGESV, ZGESV
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
    def culaDeviceDgesv(n, nrhs, a, lda, ipiv, b, ldb):
        """
        Solve linear system with LU factorization.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgesv(n, nrhs, a, lda, ipiv, b, ldb):
        """
        Solve linear system with LU factorization.

        """

        raise NotImplementedError('CULA Dense required')
else:
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

# DGETRF, ZGETRF
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
    def culaDeviceDgetrf(m, n, a, lda, ipiv):
        """
        LU factorization.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgetrf(m, n, a, lda, ipiv):
        """
        LU factorization.

        """

        raise NotImplementedError('CULA Dense required')
else:
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


# DGEQRF, ZGEQRF
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
    def culaDeviceDgeqrf(m, n, a, lda, tau):
        """
        QR factorization.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgeqrf(m, n, a, lda, tau):
        """
        QR factorization.

        """
        raise NotImplementedError('CULA Dense required')
else:
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

# DORGQR, ZUNGQR
try:
    _libcula.culaDeviceDorgqr.restype = \
    _libcula.culaDeviceZungqr.restype = int
    _libcula.culaDeviceDorgqr.argtypes = \
    _libcula.culaDeviceZungqr.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
except AttributeError:
    def culaDeviceDorgqr(m, n, k, a, lda, tau):
        """
        QR factorization - Generate Q from QR factorization
        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceDorgqr(m, n, k, a, lda, tau):
        """
        QR factorization - Generate Q from QR factorization
        """
        raise NotImplementedError('CULA Dense required')
else:
    def culaDeviceDorgqr(m, n, k, a, lda, tau):
        """
        QR factorization.
        """

        status = _libcula.culaDeviceDorgqr(m, n, k, int(a), lda, int(tau))
        culaCheckStatus(status)

    def culaDeviceZungqr(m, n, k, a, lda, tau):
        """
        QR factorization.
        """

        status = _libcula.culaDeviceZungqr(m, n, k, int(a), lda, int(tau))
        culaCheckStatus(status)

# SGETRI, CGETRI, DGETRI, ZGETRI
try:
    _libcula.culaDeviceSgetri.restype = \
    _libcula.culaDeviceCgetri.restype = \
    _libcula.culaDeviceDgetri.restype = \
    _libcula.culaDeviceZgetri.restype = int
    _libcula.culaDeviceSgetri.argtypes = \
    _libcula.culaDeviceCgetri.argtypes = \
    _libcula.culaDeviceDgetri.argtypes = \
    _libcula.culaDeviceZgetri.argtypes = [ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
except AttributeError:
    def culaDeviceSgetri(n, a, lda, ipiv):
        """
        Compute Inverse Matrix.
        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceCgetri(n, a, lda, ipiv):
        """
        Compute Inverse Matrix.
        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceDgetri(n, a, lda, ipiv):
        """
        Compute Inverse Matrix.
        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgetri(n, a, lda, ipiv):
        """
        Compute Inverse Matrix.
        """

        raise NotImplementedError('CULA Dense required')
else:
    def culaDeviceSgetri(n, a, lda, ipiv):
        """
        Compute Inverse Matrix.
        """

        status = _libcula.culaDeviceSgetri(n, int(a), lda, int(ipiv))
        culaCheckStatus(status)

    def culaDeviceCgetri(n, a, lda, ipiv):
        """
        Compute Inverse Matrix.
        """

        status = _libcula.culaDeviceCgetri(n, int(a), lda, int(ipiv))
        culaCheckStatus(status)

    def culaDeviceDgetri(n, a, lda, ipiv):
        """
        Compute Inverse Matrix.
        """

        status = _libcula.culaDeviceDgetri(n, int(a), lda, int(ipiv))
        culaCheckStatus(status)

    def culaDeviceZgetri(n, a, lda, ipiv):
        """
        Compute Inverse Matrix.
        """

        status = _libcula.culaDeviceZgetri(n, int(a), lda, int(ipiv))
        culaCheckStatus(status)


# DGELS, ZGELS
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
    def culaDeviceDgels(trans, m, n, nrhs, a, lda, b, ldb):
        """
        Solve linear system with QR or LQ factorization.

        """
        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgels(trans, m, n, nrhs, a, lda, b, ldb):
        """
        Solve linear system with QR or LQ factorization.

        """
        raise NotImplementedError('CULA Dense required')
else:
    def culaDeviceDgels(trans, m, n, nrhs, a, lda, b, ldb):
        """
        Solve linear system with QR or LQ factorization.

        """
        trans = trans.encode('ascii')
        status = _libcula.culaDeviceDgels(trans, m, n, nrhs, int(a),
                                          lda, int(b), ldb)
        culaCheckStatus(status)

    def culaDeviceZgels(trans, m, n, nrhs, a, lda, b, ldb):
        """
        Solve linear system with QR or LQ factorization.

        """
        trans = trans.encode('ascii')
        status = _libcula.culaDeviceZgels(trans, m, n, nrhs, int(a),
                                          lda, int(b), ldb)
        culaCheckStatus(status)

# DGGLSE, ZGGLSE
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
    def culaDeviceDgglse(m, n, p, a, lda, b, ldb, c, d, x):
        """
        Solve linear equality-constrained least squares problem.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgglse(m, n, p, a, lda, b, ldb, c, d, x):
        """
        Solve linear equality-constrained least squares problem.

        """

        raise NotImplementedError('CULA Dense required')
else:
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

# DGESVD, ZGESVD
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
    def culaDeviceDgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt):
        """
        SVD decomposition.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt):
        """
        SVD decomposition.

        """

        raise NotImplementedError('CULA Dense required')
else:
    def culaDeviceDgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt):
        """
        SVD decomposition.

        """
        jobu = jobu.encode('ascii')
        jobvt = jobvt.encode('ascii')
        status = _libcula.culaDeviceDgesvd(jobu, jobvt, m, n, int(a), lda,
                                           int(s), int(u), ldu, int(vt),
                                           ldvt)
        culaCheckStatus(status)

    def culaDeviceZgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt):
        """
        SVD decomposition.

        """
        jobu = jobu.encode('ascii')
        jobvt = jobvt.encode('ascii')
        status = _libcula.culaDeviceZgesvd(jobu, jobvt, m, n, int(a), lda,
                                           int(s), int(u), ldu, int(vt),
                                           ldvt)
        culaCheckStatus(status)

# SPOSV, CPOSV, DPOSV, ZPOSV
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
    def culaDeviceSposv(upio, n, nrhs, a, lda, b, ldb):
        """
        Solve positive definite linear system with Cholesky factorization.

        """
        raise NotImplementedError('CULA Dense required')

    def culaDeviceCposv(upio, n, nrhs, a, lda, b, ldb):
        """
        Solve positive definite linear system with Cholesky factorization.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceDposv(upio, n, nrhs, a, lda, b, ldb):
        """
        Solve positive definite linear system with Cholesky factorization.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceDposv(upio, n, nrhs, a, lda, b, ldb):
        """
        Solve positive definite linear system with Cholesky factorization.

        """

        raise NotImplementedError('CULA Dense required')
else:
    def culaDeviceSposv(upio, n, nrhs, a, lda, b, ldb):
        """
        Solve positive definite linear system with Cholesky factorization.

        """
        upio = upio.encode('ascii')
        status = _libcula.culaDeviceSposv(upio, n, nrhs, int(a), lda, int(b),
                                          ldb)
        culaCheckStatus(status)

    def culaDeviceCposv(upio, n, nrhs, a, lda, b, ldb):
        """
        Solve positive definite linear system with Cholesky factorization.

        """
        upio = upio.encode('ascii')
        status = _libcula.culaDeviceCposv(upio, n, nrhs, int(a), lda, int(b),
                                          ldb)
        culaCheckStatus(status)

    def culaDeviceDposv(upio, n, nrhs, a, lda, b, ldb):
        """
        Solve positive definite linear system with Cholesky factorization.

        """
        upio = upio.encode('ascii')
        status = _libcula.culaDeviceDposv(upio, n, nrhs, int(a), lda, int(b),
                                          ldb)
        culaCheckStatus(status)

    def culaDeviceZposv(upio, n, nrhs, a, lda, b, ldb):
        """
        Solve positive definite linear system with Cholesky factorization.

        """
        upio = upio.encode('ascii')
        status = _libcula.culaDeviceZposv(upio, n, nrhs, int(a), lda, int(b),
                                          ldb)
        culaCheckStatus(status)

# SPOTRF, CPOTRF, DPOTRF, ZPOTRF
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
    def culaDeviceSpotrf(uplo, n, a, lda):
        """
        Cholesky factorization.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceCpotrf(uplo, n, a, lda):
        """
        Cholesky factorization.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceDpotrf(uplo, n, a, lda):
        """
        Cholesky factorization.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceZpotrf(uplo, n, a, lda):
        """
        Cholesky factorization.

        """

        raise NotImplementedError('CULA Dense required')
else:
    def culaDeviceSpotrf(uplo, n, a, lda):
        """
        Cholesky factorization.

        """
        uplo = uplo.encode('ascii')
        status = _libcula.culaDeviceSpotrf(uplo, n, int(a), lda)
        culaCheckStatus(status)

    def culaDeviceCpotrf(uplo, n, a, lda):
        """
        Cholesky factorization.

        """
        uplo = uplo.encode('ascii')
        status = _libcula.culaDeviceCpotrf(uplo, n, int(a), lda)
        culaCheckStatus(status)

    def culaDeviceDpotrf(uplo, n, a, lda):
        """
        Cholesky factorization.

        """
        uplo = uplo.encode('ascii')
        status = _libcula.culaDeviceDpotrf(uplo, n, int(a), lda)
        culaCheckStatus(status)

    def culaDeviceZpotrf(uplo, n, a, lda):
        """
        Cholesky factorization.

        """
        uplo = uplo.encode('ascii')
        status = _libcula.culaDeviceZpotrf(uplo, n, int(a), lda)
        culaCheckStatus(status)

# SSYEV, DSYEV, CHEEV, ZHEEV
try:
    _libcula.culaDeviceSsyev.restype = \
    _libcula.culaDeviceDsyev.restype = \
    _libcula.culaDeviceCheev.restype = \
    _libcula.culaDeviceZheev.restype = int
    _libcula.culaDeviceSsyev.argtypes = \
    _libcula.culaDeviceDsyev.argtypes = \
    _libcula.culaDeviceCheev.argtypes = \
    _libcula.culaDeviceZheev.argtypes = [ctypes.c_char,
                                         ctypes.c_char,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p]
except AttributeError:
    def culaDeviceSsyev(jobz, uplo, n, a, lda, w):
        """
        Symmetric eigenvalue decomposition.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceDsyev(jobz, uplo, n, a, lda, w):
        """
        Symmetric eigenvalue decomposition.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceCheev(jobz, uplo, n, a, lda, w):
        """
        Hermitian eigenvalue decomposition.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceZheev(jobz, uplo, n, a, lda, w):
        """
        Hermitian eigenvalue decomposition.

        """

        raise NotImplementedError('CULA Dense required')
else:

    def culaDeviceSsyev(jobz, uplo, n, a, lda, w):
        """
        Symmetric eigenvalue decomposition.

        """
        jobz = jobz.encode('ascii')
        uplo = uplo.encode('ascii')
        status = _libcula.culaDeviceSsyev(jobz, uplo, n, int(a), lda, int(w))
        culaCheckStatus(status)

    def culaDeviceDsyev(jobz, uplo, n, a, lda, w):
        """
        Symmetric eigenvalue decomposition.

        """
        jobz = jobz.encode('ascii')
        uplo = uplo.encode('ascii')
        status = _libcula.culaDeviceDsyev(jobz, uplo, n, int(a), lda, int(w))
        culaCheckStatus(status)

    def culaDeviceCheev(jobz, uplo, n, a, lda, w):
        """
        Hermitian eigenvalue decomposition.

        """
        jobz = jobz.encode('ascii')
        uplo = uplo.encode('ascii')
        status = _libcula.culaDeviceCheev(jobz, uplo, n, int(a), lda, int(w))
        culaCheckStatus(status)

    def culaDeviceZheev(jobz, uplo, n, a, lda, w):
        """
        Hermitian eigenvalue decomposition.

        """
        jobz = jobz.encode('ascii')
        uplo = uplo.encode('ascii')
        status = _libcula.culaDeviceZheev(jobz, uplo, n, int(a), lda, int(w))
        culaCheckStatus(status)

# BLAS routines provided by CULA:

# SGEMM, DGEMM, CGEMM, ZGEMM
_libcula.culaDeviceSgemm.restype = \
_libcula.culaDeviceDgemm.restype = \
_libcula.culaDeviceCgemm.restype = \
_libcula.culaDeviceZgemm.restype = int

_libcula.culaDeviceSgemm.argtypes = [ctypes.c_char,
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

_libcula.culaDeviceDgemm.argtypes = [ctypes.c_char,
                                     ctypes.c_char,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_double,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_double,
                                     ctypes.c_void_p,
                                     ctypes.c_int]

_libcula.culaDeviceCgemm.argtypes = [ctypes.c_char,
                                     ctypes.c_char,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     cuda.cuFloatComplex,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     cuda.cuFloatComplex,
                                     ctypes.c_void_p,
                                     ctypes.c_int]

_libcula.culaDeviceZgemm.argtypes = [ctypes.c_char,
                                     ctypes.c_char,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     cuda.cuDoubleComplex,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     cuda.cuDoubleComplex,
                                     ctypes.c_void_p,
                                     ctypes.c_int]

def culaDeviceSgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for general matrix.

    """
    transa = transa.encode('ascii')
    transb = transb.encode('ascii')
    status = _libcula.culaDeviceSgemm(transa, transb, m, n, k, alpha,
                           int(A), lda, int(B), ldb, beta, int(C), ldc)
    culaCheckStatus(status)

def culaDeviceDgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for general matrix.

    """
    transa = transa.encode('ascii')
    transb = transb.encode('ascii')
    status = _libcula.culaDeviceDgemm(transa, transb, m, n, k, alpha,
                           int(A), lda, int(B), ldb, beta, int(C), ldc)
    culaCheckStatus(status)

def culaDeviceCgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for complex general matrix.

    """
    transa = transa.encode('ascii')
    transb = transb.encode('ascii')
    status = _libcula.culaDeviceCgemm(transa, transb, m, n, k,
                                      cuda.cuFloatComplex(alpha.real,
                                                        alpha.imag),
                                      int(A), lda, int(B), ldb,
                                      cuda.cuFloatComplex(beta.real,
                                                        beta.imag),
                                      int(C), ldc)
    culaCheckStatus(status)

def culaDeviceZgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for complex general matrix.

    """
    transa = transa.encode('ascii')
    transb = transb.encode('ascii')
    status = _libcula.culaDeviceZgemm(transa, transb, m, n, k,
                                      cuda.cuDoubleComplex(alpha.real,
                                                        alpha.imag),
                                      int(A), lda, int(B), ldb,
                                      cuda.cuDoubleComplex(beta.real,
                                                        beta.imag),
                                      int(C), ldc)
    culaCheckStatus(status)

# SGEMV, DGEMV, CGEMV, ZGEMV
_libcula.culaDeviceSgemv.restype = \
_libcula.culaDeviceDgemv.restype = \
_libcula.culaDeviceCgemv.restype = \
_libcula.culaDeviceZgemv.restype = int

_libcula.culaDeviceSgemv.argtypes = [ctypes.c_char,
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

_libcula.culaDeviceDgemv.argtypes = [ctypes.c_char,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_double,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_double,
                                     ctypes.c_void_p,
                                     ctypes.c_int]

_libcula.culaDeviceCgemv.argtypes = [ctypes.c_char,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     cuda.cuFloatComplex,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     cuda.cuFloatComplex,
                                     ctypes.c_void_p,
                                     ctypes.c_int]

_libcula.culaDeviceZgemv.argtypes = [ctypes.c_char,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     cuda.cuDoubleComplex,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     cuda.cuDoubleComplex,
                                     ctypes.c_void_p,
                                     ctypes.c_int]

def culaDeviceSgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for real general matrix.

    """
    trans = trans.encode('ascii')
    status = _libcula.culaDeviceSgemv(trans, m, n, alpha, int(A), lda,
                           int(x), incx, beta, int(y), incy)
    culaCheckStatus(status)

def culaDeviceDgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for real general matrix.

    """
    trans = trans.encode('ascii')
    status = _libcula.culaDeviceDgemv(trans, m, n, alpha, int(A), lda,
                           int(x), incx, beta, int(y), incy)
    culaCheckStatus(status)


def culaDeviceCgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for complex general matrix.

    """
    trans = trans.encode('ascii')
    status = _libcula.culaDeviceCgemv(trans, m, n,
                           cuda.cuFloatComplex(alpha.real,
                                               alpha.imag),
                           int(A), lda, int(x), incx,
                           cuda.cuFloatComplex(beta.real,
                                               beta.imag),
                           int(y), incy)
    culaCheckStatus(status)

def culaDeviceZgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for complex general matrix.

    """
    trans = trans.encode('ascii')
    status = _libcula.culaDeviceZgemv(trans, m, n,
                           cuda.cuDoubleComplex(alpha.real,
                                               alpha.imag),
                           int(A), lda, int(x), incx,
                           cuda.cuDoubleComplex(beta.real,
                                               beta.imag),
                           int(y), incy)
    culaCheckStatus(status)
    
# SGEEV, DGEEV, CGEEV, ZGEEV
try:
    _libcula.culaDeviceSgeev.restype = \
    _libcula.culaDeviceDgeev.restype = \
    _libcula.culaDeviceCgeev.restype = \
    _libcula.culaDeviceZgeev.restype = int

    _libcula.culaDeviceSgeev.argtypes = \
    _libcula.culaDeviceDgeev.argtypes = [ctypes.c_char, #jobvl
                                         ctypes.c_char, #jobvr
                                         ctypes.c_int, #n,  the order of the matrix
                                         ctypes.c_void_p, #a
                                         ctypes.c_int, #lda
                                         ctypes.c_void_p, #wr
                                         ctypes.c_void_p, #wi
                                         ctypes.c_void_p, #vl
                                         ctypes.c_int, #ldvl
                                         ctypes.c_void_p, #vr
                                         ctypes.c_int] #ldvr
    _libcula.culaDeviceCgeev.argtypes = \
    _libcula.culaDeviceZgeev.argtypes = [ctypes.c_char, #jobvl
                                         ctypes.c_char, #jobvr
                                         ctypes.c_int, #n,  the order of the matrix
                                         ctypes.c_void_p, #a
                                         ctypes.c_int, #lda
                                         ctypes.c_void_p, #w
                                         ctypes.c_void_p, #vl
                                         ctypes.c_int, #ldvl
                                         ctypes.c_void_p, #vr
                                         ctypes.c_int] #ldvr
except AttributeError:
    def culaDeviceSgeev(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr):
        """
        General Eigenproblem solver.
        """
        raise NotImplementedError('CULA Dense required')

    def culaDeviceDgeev(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr):
        """
        General Eigenproblem solver.
        """
        raise NotImplementedError('CULA Dense required')

    def culaDeviceCgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr):
        """
        General Eigenproblem solver.
        """
        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr):
        """
        General Eigenproblem solver.
        """
        raise NotImplementedError('CULA Dense required')
else:
    def culaDeviceSgeev(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr):
        """
        General Eigenproblem solver.
        """
        jobvl = jobvl.encode('ascii')
        jobvr = jobvr.encode('ascii')
        status = _libcula.culaDeviceSgeev(jobvl, jobvr, n, int(a), lda, int(wr), int(wi),
                               int(vl), ldvl, int(vr), ldvr)
        culaCheckStatus(status)

    def culaDeviceDgeev(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr):
        """
        General Eigenproblem solver.
        """
        jobvl = jobvl.encode('ascii')
        jobvr = jobvr.encode('ascii')
        status = _libcula.culaDeviceDgeev(jobvl, jobvr, n, int(a), lda, int(wr), int(wi),
                               int(vl), ldvl, int(vr), ldvr)
        culaCheckStatus(status)

    def culaDeviceCgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr):
        """
        General Eigenproblem solver.
        """
        jobvl = jobvl.encode('ascii')
        jobvr = jobvr.encode('ascii')
        status = _libcula.culaDeviceCgeev(jobvl, jobvr, n, int(a), lda, int(w),
                               int(vl), ldvl, int(vr), ldvr)
        culaCheckStatus(status)

    def culaDeviceZgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr):
        """
        General Eigenproblem solver.

        """
        jobvl = jobvl.encode('ascii')
        jobvr = jobvr.encode('ascii')
        status = _libcula.culaDeviceZgeev(jobvl, jobvr, n, int(a), lda, int(w),
                               int(vl), ldvl, int(vr), ldvr)
        culaCheckStatus(status)   
    
# Auxiliary routines:

try:
    _libcula.culaDeviceSgeTranspose.restype = \
    _libcula.culaDeviceDgeTranspose.restype = \
    _libcula.culaDeviceCgeTranspose.restype = \
    _libcula.culaDeviceZgeTranspose.restype = int
    _libcula.culaDeviceSgeTranspose.argtypes = \
    _libcula.culaDeviceDgeTranspose.argtypes = \
    _libcula.culaDeviceCgeTranspose.argtypes = \
    _libcula.culaDeviceZgeTranspose.argtypes = [ctypes.c_int,
                                                ctypes.c_int,
                                                ctypes.c_void_p,
                                                ctypes.c_int,
                                                ctypes.c_void_p,
                                                ctypes.c_int]
except AttributeError:
    def culaDeviceSgeTranspose(m, n, A, lda, B, ldb):
        """
        Transpose of real general matrix.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceDgeTranspose(m, n, A, lda, B, ldb):
        """
        Transpose of real general matrix.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceCgeTranspose(m, n, A, lda, B, ldb):
        """
        Transpose of complex general matrix.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgeTranspose(m, n, A, lda, B, ldb):
        """
        Transpose of complex general matrix.

        """

        raise NotImplementedError('CULA Dense required')
else:
    def culaDeviceSgeTranspose(m, n, A, lda, B, ldb):
        """
        Transpose of real general matrix.

        """

        status = _libcula.culaDeviceSgeTranspose(m, n, int(A), lda, int(B), ldb)
        culaCheckStatus(status)

    def culaDeviceDgeTranspose(m, n, A, lda, B, ldb):
        """
        Transpose of real general matrix.

        """

        status = _libcula.culaDeviceDgeTranspose(m, n, int(A), lda, int(B), ldb)
        culaCheckStatus(status)

    def culaDeviceCgeTranspose(m, n, A, lda, B, ldb):
        """
        Transpose of complex general matrix.

        """

        status = _libcula.culaDeviceCgeTranspose(m, n, int(A), lda, int(B), ldb)
        culaCheckStatus(status)

    def culaDeviceZgeTranspose(m, n, A, lda, B, ldb):
        """
        Transpose of complex general matrix.

        """

        status = _libcula.culaDeviceZgeTranspose(m, n, int(A), lda, int(B), ldb)
        culaCheckStatus(status)


try:
    _libcula.culaDeviceSgeTransposeInplace.restype = \
    _libcula.culaDeviceDgeTransposeInplace.restype = \
    _libcula.culaDeviceCgeTransposeInplace.restype = \
    _libcula.culaDeviceZgeTransposeInplace.restype = int
    _libcula.culaDeviceSgeTransposeInplace.argtypes = \
    _libcula.culaDeviceDgeTransposeInplace.argtypes = \
    _libcula.culaDeviceCgeTransposeInplace.argtypes = \
    _libcula.culaDeviceZgeTransposeInplace.argtypes = [ctypes.c_int,
                                                    ctypes.c_void_p,
                                                    ctypes.c_int]
except AttributeError:
    def culaDeviceSgeTransposeInplace(n, A, lda):
        """
        Inplace transpose of real square matrix.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceDgeTransposeInplace(n, A, lda):
        """
        Inplace transpose of real square matrix.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceCgeTransposeInplace(n, A, lda):
        """
        Inplace transpose of complex square matrix.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgeTransposeInplace(n, A, lda):
        """
        Inplace transpose of complex square matrix.

        """

        raise NotImplementedError('CULA Dense required')
else:
    def culaDeviceSgeTransposeInplace(n, A, lda):
        """
        Inplace transpose of real square matrix.

        """

        status = _libcula.culaDeviceSgeTransposeInplace(n, int(A), lda)
        culaCheckStatus(status)

    def culaDeviceDgeTransposeInplace(n, A, lda):
        """
        Inplace transpose of real square matrix.

        """

        status = _libcula.culaDeviceDgeTransposeInplace(n, int(A), lda)
        culaCheckStatus(status)

    def culaDeviceCgeTransposeInplace(n, A, lda):
        """
        Inplace transpose of complex square matrix.

        """

        status = _libcula.culaDeviceCgeTransposeInplace(n, int(A), lda)
        culaCheckStatus(status)

    def culaDeviceZgeTransposeInplace(n, A, lda):
        """
        Inplace transpose of complex square matrix.

        """

        status = _libcula.culaDeviceZgeTransposeInplace(n, int(A), lda)
        culaCheckStatus(status)

try:

    _libcula.culaDeviceCgeTransposeConjugate.restype = \
    _libcula.culaDeviceZgeTransposeConjugate.restype = int
    _libcula.culaDeviceCgeTransposeConjugate.argtypes = \
    _libcula.culaDeviceZgeTransposeConjugate.argtypes = [ctypes.c_int,
                                                        ctypes.c_int,
                                                        ctypes.c_void_p,
                                                        ctypes.c_int,
                                                        ctypes.c_void_p,
                                                        ctypes.c_int]
except AttributeError:
    def culaDeviceCgeTransposeConjugate(m, n, A, lda, B, ldb):
        """
        Conjugate transpose of complex general matrix.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgeTransposeConjugate(m, n, A, lda, B, ldb):
        """
        Conjugate transpose of complex general matrix.

        """
        raise NotImplementedError('CULA Dense required')
else:
    def culaDeviceCgeTransposeConjugate(m, n, A, lda, B, ldb):
        """
        Conjugate transpose of complex general matrix.

        """

        status = _libcula.culaDeviceCgeTransposeConjugate(m, n, int(A), lda, int(B), ldb)
        culaCheckStatus(status)

    def culaDeviceZgeTransposeConjugate(m, n, A, lda, B, ldb):
        """
        Conjugate transpose of complex general matrix.

        """

        status = _libcula.culaDeviceZgeTransposeConjugate(m, n, int(A), lda, int(B), ldb)
        culaCheckStatus(status)

try:
    _libcula.culaDeviceCgeTransposeConjugateInplace.restype = \
    _libcula.culaDeviceZgeTransposeConjugateInplace.restype = int
    _libcula.culaDeviceCgeTransposeConjugateInplace.argtypes = \
    _libcula.culaDeviceZgeTransposeConjugateInplace.argtypes = [ctypes.c_int,
                                                                ctypes.c_void_p,
                                                                ctypes.c_int]
except AttributeError:
    def culaDeviceCgeTransposeConjugateInplace(n, A, lda):
        """
        Inplace conjugate transpose of complex square matrix.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgeTransposeConjugateInplace(n, A, lda):
        """
        Inplace conjugate transpose of complex square matrix.

        """

        raise NotImplementedError('CULA Dense required')
else:
    def culaDeviceCgeTransposeConjugateInplace(n, A, lda):
        """
        Inplace conjugate transpose of complex square matrix.

        """

        status = _libcula.culaDeviceCgeTransposeConjugateInplace(n, int(A), lda)
        culaCheckStatus(status)

    def culaDeviceZgeTransposeConjugateInplace(n, A, lda):
        """
        Inplace conjugate transpose of complex square matrix.

        """

        status = _libcula.culaDeviceZgeTransposeConjugateInplace(n, int(A), lda)
        culaCheckStatus(status)

try:
    _libcula.culaDeviceCgeConjugate.restype = \
    _libcula.culaDeviceZgeConjugate.restype = int
    _libcula.culaDeviceCgeConjugate.argtypes = \
    _libcula.culaDeviceZgeConjugate.argtypes = [ctypes.c_int,
                                                ctypes.c_int,
                                                ctypes.c_void_p,
                                                ctypes.c_int]
except AttributeError:
    def culaDeviceCgeConjugate(m, n, A, lda):
        """
        Conjugate of complex general matrix.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgeConjugate(m, n, A, lda):
        """
        Conjugate of complex general matrix.

        """

        raise NotImplementedError('CULA Dense required')
else:
    def culaDeviceCgeConjugate(m, n, A, lda):
        """
        Conjugate of complex general matrix.

        """

        status = _libcula.culaDeviceCgeConjugate(m, n, int(A), lda)
        culaCheckStatus(status)

    def culaDeviceZgeConjugate(m, n, A, lda):
        """
        Conjugate of complex general matrix.

        """

        status = _libcula.culaDeviceZgeConjugate(m, n, int(A), lda)
        culaCheckStatus(status)

try:
    _libcula.culaDeviceCtrConjugate.restype = \
    _libcula.culaDeviceZtrConjugate.restype = int
    _libcula.culaDeviceCtrConjugate.argtypes = \
    _libcula.culaDeviceZtrConjugate.argtypes = [ctypes.c_char,
                                                ctypes.c_char,
                                                ctypes.c_int,
                                                ctypes.c_int,
                                                ctypes.c_void_p,
                                                ctypes.c_int]
except AttributeError:
    def culaDeviceCtrConjugate(uplo, diag, m, n, A, lda):
        """
        Conjugate of complex upper or lower triangle matrix.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceZtrConjugate(uplo, diag, m, n, A, lda):
        """
        Conjugate of complex upper or lower triangle matrix.

        """

        raise NotImplementedError('CULA Dense required')
else:
    def culaDeviceCtrConjugate(uplo, diag, m, n, A, lda):
        """
        Conjugate of complex upper or lower triangle matrix.

        """
        uplo = uplo.encode('ascii')
        status = _libcula.culaDeviceCtrConjugate(uplo, diag, m, n, int(A), lda)
        culaCheckStatus(status)

    def culaDeviceZtrConjugate(uplo, diag, m, n, A, lda):
        """
        Conjugate of complex upper or lower triangle matrix.

        """
        uplo = uplo.encode('ascii')
        status = _libcula.culaDeviceZtrConjugate(uplo, diag, m, n, int(A), lda)
        culaCheckStatus(status)

try:
    _libcula.culaDeviceSgeNancheck.restype = \
    _libcula.culaDeviceDgeNancheck.restype = \
    _libcula.culaDeviceCgeNancheck.restype = \
    _libcula.culaDeviceZgeNancheck.restype = int
    _libcula.culaDeviceSgeNancheck.argtypes = \
    _libcula.culaDeviceDgeNancheck.argtypes = \
    _libcula.culaDeviceCgeNancheck.argtypes = \
    _libcula.culaDeviceZgeNancheck.argtypes = [ctypes.c_int,
                                                ctypes.c_int,
                                                ctypes.c_void_p,
                                                ctypes.c_int]
except AttributeError:
    def culaDeviceSgeNancheck(m, n, A, lda):
        """
        Check a real general matrix for invalid entries

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceDgeNancheck(m, n, A, lda):
        """
        Check a real general matrix for invalid entries

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceCgeNancheck(m, n, A, lda):
        """
        Check a complex general matrix for invalid entries

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgeNancheck(m, n, A, lda):
        """
        Check a complex general matrix for invalid entries

        """

        raise NotImplementedError('CULA Dense required')
else:
    def culaDeviceSgeNancheck(m, n, A, lda):
        """
        Check a real general matrix for invalid entries

        """

        status = _libcula.culaDeviceSgeNancheck(m, n, int(A), lda)
        try:
            culaCheckStatus(status)
        except culaDataError:
            return True
        return False

    def culaDeviceDgeNancheck(m, n, A, lda):
        """
        Check a real general matrix for invalid entries

        """

        status = _libcula.culaDeviceDgeNancheck(m, n, int(A), lda)
        try:
            culaCheckStatus(status)
        except culaDataError:
            return True
        return False

    def culaDeviceCgeNancheck(m, n, A, lda):
        """
        Check a complex general matrix for invalid entries

        """

        status = _libcula.culaDeviceCgeNancheck(m, n, int(A), lda)
        try:
            culaCheckStatus(status)
        except culaDataError:
            return True
        return False

    def culaDeviceZgeNancheck(m, n, A, lda):
        """
        Check a complex general matrix for invalid entries

        """

        status = _libcula.culaDeviceZgeNancheck(m, n, int(A), lda)
        try:
            culaCheckStatus(status)
        except culaDataError:
            return True
        return False


if __name__ == "__main__":
    import doctest
    doctest.testmod()
