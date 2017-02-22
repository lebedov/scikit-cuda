#!/usr/bin/env python

"""
Python interface to CUSOLVER functions.

Note: this module does not explicitly depend on PyCUDA.
"""

from . import cudart

if int(cudart._cudart_version) < 7000:
    raise ImportError('CUSOLVER library only available in CUDA 7.0 and later')

import ctypes
import sys

import numpy as np

from . import cuda

# Load library:
_version_list = [8.0, 7.5, 7.0]
if 'linux' in sys.platform:
    _libcusolver_libname_list = ['libcusolver.so'] + \
                                ['libcusolver.so.%s' % v for v in _version_list]

    # Fix for GOMP weirdness with CUDA 8.0 on Fedora (#171):
    try:
        ctypes.CDLL('libgomp.so.1', mode=ctypes.RTLD_GLOBAL)
    except:
        pass
elif sys.platform == 'darwin':
    _libcusolver_libname_list = ['libcusolver.dylib']
elif sys.platform == 'win32':
    if sys.maxsize > 2**32:
        _libcusolver_libname_list = ['cusolver.dll'] + \
                                    ['cusolver64_%s.dll' % int(10*v) for v in _version_list]
    else:
        _libcusolver_libname_list = ['cusolver.dll'] + \
                                    ['cusolver32_%s.dll' % int(10*v) for v in _version_list]
else:
    raise RuntimeError('unsupported platform')

# Print understandable error message when library cannot be found:
_libcusolver = None
for _libcusolver_libname in _libcusolver_libname_list:
    try:
        if sys.platform == 'win32':
            _libcusolver = ctypes.windll.LoadLibrary(_libcusolver_libname)
        else:
            _libcusolver = ctypes.cdll.LoadLibrary(_libcusolver_libname)
    except OSError:
        pass
    else:
        break
if _libcusolver == None:
    raise OSError('cusolver library not found')

class CUSOLVER_ERROR(Exception):
    """CUSOLVER error."""
    pass

class CUSOLVER_STATUS_NOT_INITIALIZED(CUSOLVER_ERROR):
    """CUSOLVER library not initialized."""
    pass

class CUSOLVER_STATUS_ALLOC_FAILED(CUSOLVER_ERROR):
    """CUSOLVER memory allocation failed."""
    pass

class CUSOLVER_STATUS_INVALID_VALUE(CUSOLVER_ERROR):
    """Invalid value passed to CUSOLVER function."""
    pass

class CUSOLVER_STATUS_ARCH_MISMATCH(CUSOLVER_ERROR):
    """CUSOLVER architecture mismatch."""
    pass

class CUSOLVER_STATUS_MAPPING_ERROR(CUSOLVER_ERROR):
    """CUSOLVER mapping error."""
    pass

class CUSOLVER_STATUS_EXECUTION_FAILED(CUSOLVER_ERROR):
    """CUSOLVER execution failed."""
    pass

class CUSOLVER_STATUS_INTERNAL_ERROR(CUSOLVER_ERROR):
    """CUSOLVER internal error."""
    pass

class CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED(CUSOLVER_ERROR):
    """Matrix type not supported by CUSOLVER."""
    pass

class CUSOLVER_STATUS_NOT_SUPPORTED(CUSOLVER_ERROR):
    """Operation not supported by CUSOLVER."""
    pass

class CUSOLVER_STATUS_ZERO_PIVOT(CUSOLVER_ERROR):
    """Zero pivot encountered by CUSOLVER."""
    pass

class CUSOLVER_STATUS_INVALID_LICENSE(CUSOLVER_ERROR):
    """Invalid CUSOLVER license."""
    pass

CUSOLVER_EXCEPTIONS = {
    1: CUSOLVER_STATUS_NOT_INITIALIZED,
    2: CUSOLVER_STATUS_ALLOC_FAILED,
    3: CUSOLVER_STATUS_INVALID_VALUE,
    4: CUSOLVER_STATUS_ARCH_MISMATCH,
    5: CUSOLVER_STATUS_MAPPING_ERROR,
    6: CUSOLVER_STATUS_EXECUTION_FAILED,
    7: CUSOLVER_STATUS_INTERNAL_ERROR,
    8: CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED,
    9: CUSOLVER_STATUS_NOT_SUPPORTED,
    10: CUSOLVER_STATUS_ZERO_PIVOT,
    11: CUSOLVER_STATUS_INVALID_LICENSE
}

def cusolverCheckStatus(status):
    """
    Raise CUSOLVER exception.

    Raise an exception corresponding to the specified CUSOLVER error
    code.

    Parameters
    ----------
    status : int
        CUSOLVER error code.

    See Also
    --------
    CUSOLVER_EXCEPTIONS
    """

    if status != 0:
        try:
            raise CUSOLVER_EXCEPTIONS[status]
        except KeyError:
            raise CUSOLVER_ERROR

# Helper functions:

_libcusolver.cusolverDnCreate.restype = int
_libcusolver.cusolverDnCreate.argtypes = [ctypes.c_void_p]
def cusolverDnCreate():
    """
    Create cuSolverDn context.

    Returns
    -------
    handle : int
        cuSolverDn context.

    References
    ----------
    `cusolverDnCreate <http://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDNcreate>`_
    """

    handle = ctypes.c_void_p()
    status = _libcusolver.cusolverDnCreate(ctypes.byref(handle))
    cusolverCheckStatus(status)
    return handle.value

_libcusolver.cusolverDnDestroy.restype = int
_libcusolver.cusolverDnDestroy.argtypes = [ctypes.c_void_p]
def cusolverDnDestroy(handle):
    """
    Destroy cuSolverDn context.

    Parameters
    ----------
    handle : int
        cuSolverDn context.

    References
    ----------
    `cusolverDnDestroy <http://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDNdestroy>`_
    """

    status = _libcusolver.cusolverDnDestroy(handle)
    cusolverCheckStatus(status)

_libcusolver.cusolverDnSetStream.restype = int
_libcusolver.cusolverDnSetStream.argtypes = [ctypes.c_int,
                                             ctypes.c_int]
def cusolverDnSetStream(handle, stream):
    """
    Set stream used by cuSolverDN library.

    Parameters
    ----------
    handle : int
        cuSolverDN context.
    stream : int
        Stream to be used.

    References
    ----------
    `cusolverDnSetStream <http://docs.nvidia.com/cuda/cusolver/index.html#cudssetstream>`_
    """

    status = _libcusolver.cusolverDnSetStream(handle, stream)
    cusolverCheckStatus(status)

_libcusolver.cusolverDnGetStream.restype = int
_libcusolver.cusolverDnGetStream.argtypes = [ctypes.c_int,
                                             ctypes.c_void_p]
def cusolverDnGetStream(handle):
    """
    Get stream used by cuSolverDN library.

    Parameters
    ----------
    handle : int
        cuSolverDN context.

    Returns
    -------
    stream : int
        Stream used by context.

    References
    ----------
    `cusolverDnGetStream <http://docs.nvidia.com/cuda/cusolver/index.html#cudsgetstream>`_
    """

    stream = ctypes.c_int()
    status = _libcusolver.cusolverDnGetStream(handle, ctypes.byref(stream))
    cusolverCheckStatus(status)
    return status.value

# Dense solver functions:

# SPOTRF, DPOTRF, CPOTRF, ZPOTRF
_libcusolver.cusolverDnSpotrf_bufferSize.restype = int
_libcusolver.cusolverDnSpotrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda):
    """
    Calculate size of work buffer used by cusolverDnSpotrf.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """

    Lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnSpotrf_bufferSize(handle, uplo, n,
                                                      int(A),
                                                      lda, ctypes.byref(Lwork))
    cusolverCheckStatus(status)
    return Lwork.value

_libcusolver.cusolverDnSpotrf.restype = int
_libcusolver.cusolverDnSpotrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnSpotrf(handle, uplo, n, A, lda, Workspace, devIpiv, devInfo):
    """
    Compute Cholesky factorization of a real single precision Hermitian positive-definite matrix.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """

    status = _libcusolver.cusolverDnSpotrf(handle, uplo, n, int(A), lda,
                                           int(Workspace),
                                           int(devIpiv),
                                           int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnDpotrf_bufferSize.restype = int
_libcusolver.cusolverDnDpotrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda):
    """
    Calculate size of work buffer used by cusolverDnDpotrf.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """

    Lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnDpotrf_bufferSize(handle, uplo, n,
                                                      int(A),
                                                      lda, ctypes.byref(Lwork))
    cusolverCheckStatus(status)
    return Lwork.value

_libcusolver.cusolverDnDpotrf.restype = int
_libcusolver.cusolverDnDpotrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, devIpiv, devInfo):
    """
    Compute Cholesky factorization of a real double precision Hermitian positive-definite matrix.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """

    status = _libcusolver.cusolverDnDpotrf(handle, uplo, n, int(A), lda,
                                           int(Workspace),
                                           int(devIpiv),
                                           int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnCpotrf_bufferSize.restype = int
_libcusolver.cusolverDnCpotrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnCpotrf_bufferSize(handle, uplo, n, A, lda):
    """
    Calculate size of work buffer used by cusolverDnCpotrf.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """

    Lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnCpotrf_bufferSize(handle, uplo, n,
                                                      int(A),
                                                      lda, ctypes.byref(Lwork))
    cusolverCheckStatus(status)
    return Lwork.value

_libcusolver.cusolverDnCpotrf.restype = int
_libcusolver.cusolverDnCpotrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnCpotrf(handle, uplo, n, A, lda, Workspace, devIpiv, devInfo):
    """
    Compute Cholesky factorization of a complex single precision Hermitian positive-definite matrix.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """

    status = _libcusolver.cusolverDnCpotrf(handle, uplo, n, int(A), lda,
                                           int(Workspace),
                                           int(devIpiv),
                                           int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnZpotrf_bufferSize.restype = int
_libcusolver.cusolverDnZpotrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnZpotrf_bufferSize(handle, uplo, n, A, lda):
    """
    Calculate size of work buffer used by cusolverDnZpotrf.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """

    Lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnZpotrf_bufferSize(handle, uplo, n,
                                                      int(A),
                                                      lda, ctypes.byref(Lwork))
    cusolverCheckStatus(status)
    return Lwork.value

_libcusolver.cusolverDnZpotrf.restype = int
_libcusolver.cusolverDnZpotrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnZpotrf(handle, uplo, n, A, lda, Workspace, devIpiv, devInfo):
    """
    Compute Cholesky factorization of a complex double precision Hermitian positive-definite matrix.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """

    status = _libcusolver.cusolverDnZpotrf(handle, uplo, n, int(A), lda,
                                           int(Workspace),
                                           int(devIpiv),
                                           int(devInfo))
    cusolverCheckStatus(status)

# SGETRF, DGETRF, CGETRF, ZGETRF
_libcusolver.cusolverDnSgetrf_bufferSize.restype = int
_libcusolver.cusolverDnSgetrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnSgetrf_bufferSize(handle, m, n, A, lda):
    """
    Calculate size of work buffer used by cusolverDnSgetrf.

    References
    ----------
    `cusolverDn<t>getrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf>`_
    """

    Lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnSgetrf_bufferSize(handle, m, n,
                                                      int(A),
                                                      n, ctypes.byref(Lwork))
    cusolverCheckStatus(status)
    return Lwork.value

_libcusolver.cusolverDnSgetrf.restype = int
_libcusolver.cusolverDnSgetrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo):
    """
    Compute LU factorization of a real single precision m x n matrix.

    References
    ----------
    `cusolverDn<t>getrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf>`_
    """

    status = _libcusolver.cusolverDnSgetrf(handle, m, n, int(A), lda,
                                          int(Workspace),
                                          int(devIpiv),
                                          int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnDgetrf_bufferSize.restype = int
_libcusolver.cusolverDnDgetrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnDgetrf_bufferSize(handle, m, n, A, lda):
    """
    Calculate size of work buffer used by cusolverDnDgetrf.

    References
    ----------
    `cusolverDn<t>getrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf>`_
    """

    Lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnDgetrf_bufferSize(handle, m, n,
                                                      int(A),
                                                      n, ctypes.byref(Lwork))
    cusolverCheckStatus(status)
    return Lwork.value

_libcusolver.cusolverDnDgetrf.restype = int
_libcusolver.cusolverDnDgetrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo):
    """
    Compute LU factorization of a real double precision m x n matrix.

    References
    ----------
    `cusolverDn<t>getrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf>`_
    """

    status = _libcusolver.cusolverDnDgetrf(handle, m, n, int(A), lda,
                                          int(Workspace),
                                          int(devIpiv),
                                          int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnCgetrf_bufferSize.restype = int
_libcusolver.cusolverDnCgetrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnCgetrf_bufferSize(handle, m, n, A, lda):
    """
    Calculate size of work buffer used by cusolverDnCgetrf.

    References
    ----------
    `cusolverDn<t>getrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf>`_
    """

    Lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnCgetrf_bufferSize(handle, m, n,
                                                      int(A),
                                                      n, ctypes.byref(Lwork))
    cusolverCheckStatus(status)
    return Lwork.value

_libcusolver.cusolverDnCgetrf.restype = int
_libcusolver.cusolverDnCgetrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnCgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo):
    """
    Compute LU factorization of a complex single precision m x n matrix.

    References
    ----------
    `cusolverDn<t>getrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf>`_
    """

    status = _libcusolver.cusolverDnCgetrf(handle, m, n, int(A), lda,
                                           int(Workspace),
                                           int(devIpiv),
                                           int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnZgetrf_bufferSize.restype = int
_libcusolver.cusolverDnZgetrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnZgetrf_bufferSize(handle, m, n, A, lda):
    """
    Calculate size of work buffer used by cusolverDnZgetrf.

    References
    ----------
    `cusolverDn<t>getrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf>`_
    """

    Lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnZgetrf_bufferSize(handle, m, n,
                                                      int(A),
                                                      n, ctypes.byref(Lwork))
    cusolverCheckStatus(status)
    return Lwork.value

_libcusolver.cusolverDnZgetrf.restype = int
_libcusolver.cusolverDnZgetrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnZgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo):
    """
    Compute LU factorization of a complex double precision m x n matrix.

    References
    ----------
    `cusolverDn<t>getrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf>`_
    """

    status = _libcusolver.cusolverDnZgetrf(handle, m, n, int(A), lda,
                                           int(Workspace),
                                           int(devIpiv),
                                           int(devInfo))
    cusolverCheckStatus(status)

# SGETRS, DGETRS, CGETRS, ZGETRS
_libcusolver.cusolverDnSgetrs.restype = int
_libcusolver.cusolverDnSgetrs.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
def cusolverDnSgetrs(handle, trans, n, nrhs, A, lda,
                     devIpiv, B, ldb, devInfo):
    """
    Solve real single precision linear system.

    References
    ----------
    `cusolverDn<t>getrs <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrs>`_
    """

    status = _libcusolver.cusolverDnSgetrs(handle, trans, n, nrhs,
                                           int(A), lda,
                                           int(devIpiv), int(B),
                                           ldb, int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnDgetrs.restype = int
_libcusolver.cusolverDnDgetrs.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
def cusolverDnDgetrs(handle, trans, n, nrhs, A, lda,
                     devIpiv, B, ldb, devInfo):
    """
    Solve real double precision linear system.

    References
    ----------
    `cusolverDn<t>getrs <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrs>`_
    """

    status = _libcusolver.cusolverDnDgetrs(handle, trans, n, nrhs,
                                           int(A), lda,
                                           int(devIpiv), int(B),
                                           ldb, int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnCgetrs.restype = int
_libcusolver.cusolverDnCgetrs.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
def cusolverDnCgetrs(handle, trans, n, nrhs, A, lda,
                     devIpiv, B, ldb, devInfo):
    """
    Solve complex single precision linear system.

    References
    ----------
    `cusolverDn<t>getrs <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrs>`_
    """

    status = _libcusolver.cusolverDnCgetrs(handle, trans, n, nrhs,
                                           int(A), lda,
                                           int(devIpiv), int(B),
                                           ldb, int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnZgetrs.restype = int
_libcusolver.cusolverDnZgetrs.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
def cusolverDnZgetrs(handle, trans, n, nrhs, A, lda,
                     devIpiv, B, ldb, devInfo):
    """
    Solve complex double precision linear system.

    References
    ----------
    `cusolverDn<t>getrs <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrs>`_
    """

    status = _libcusolver.cusolverDnZgetrs(handle, trans, n, nrhs,
                                           int(A), lda,
                                           int(devIpiv), int(B),
                                           ldb, int(devInfo))
    cusolverCheckStatus(status)

# SGESVD, DGESVD, CGESVD, ZGESVD
_libcusolver.cusolverDnSgesvd_bufferSize.restype = int
_libcusolver.cusolverDnSgesvd_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnSgesvd_bufferSize(handle, m, n):
    """
    Calculate size of work buffer used by cusolverDnSgesvd.

    References
    ----------
    `cusolverDn<t>gesvd <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesvd>`_
    """

    Lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnSgesvd_bufferSize(handle, m, n, ctypes.byref(Lwork))
    cusolverCheckStatus(status)
    return Lwork.value

_libcusolver.cusolverDnSgesvd.restype = int
_libcusolver.cusolverDnSgesvd.argtypes = [ctypes.c_void_p,
                                          ctypes.c_char,
                                          ctypes.c_char,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U,
        ldu, VT, ldvt, Work, Lwork, rwork, devInfo):
    """
    Compute real single precision singular value decomposition.

    References
    ----------
    `cusolverDn<t>gesvd <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesvd>`_
    """

    jobu = jobu.encode('ascii')
    jobvt = jobvt.encode('ascii')
    status = _libcusolver.cusolverDnSgesvd(handle, jobu, jobvt, m, n,
                                           int(A), lda, int(S), int(U),
                                           ldu, int(VT), ldvt, int(Work),
                                           Lwork, int(rwork), int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnDgesvd_bufferSize.restype = int
_libcusolver.cusolverDnDgesvd_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnDgesvd_bufferSize(handle, m, n):
    """
    Calculate size of work buffer used by cusolverDnDgesvd.

    References
    ----------
    `cusolverDn<t>gesvd <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesvd>`_
    """

    Lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnDgesvd_bufferSize(handle, m, n, ctypes.byref(Lwork))
    cusolverCheckStatus(status)
    return Lwork.value

_libcusolver.cusolverDnDgesvd.restype = int
_libcusolver.cusolverDnDgesvd.argtypes = [ctypes.c_void_p,
                                          ctypes.c_char,
                                          ctypes.c_char,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U,
                     ldu, VT, ldvt, Work, Lwork, rwork, devInfo):
    """
    Compute real double precision singular value decomposition.

    References
    ----------
    `cusolverDn<t>gesvd <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesvd>`_
    """

    jobu = jobu.encode('ascii')
    jobvt = jobvt.encode('ascii')
    status = _libcusolver.cusolverDnDgesvd(handle, jobu, jobvt, m, n,
                                           int(A), lda, int(S), int(U),
                                           ldu, int(VT), ldvt, int(Work),
                                           Lwork, int(rwork), int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnCgesvd_bufferSize.restype = int
_libcusolver.cusolverDnCgesvd_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnCgesvd_bufferSize(handle, m, n):
    """
    Calculate size of work buffer used by cusolverDnCgesvd.

    References
    ----------
    `cusolverDn<t>gesvd <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesvd>`_
    """

    Lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnCgesvd_bufferSize(handle, m, n, ctypes.byref(Lwork))
    cusolverCheckStatus(status)
    return Lwork.value

_libcusolver.cusolverDnCgesvd.restype = int
_libcusolver.cusolverDnCgesvd.argtypes = [ctypes.c_void_p,
                                          ctypes.c_char,
                                          ctypes.c_char,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnCgesvd(handle, jobu, jobvt, m, n, A, lda, S, U,
                     ldu, VT, ldvt, Work, Lwork, rwork, devInfo):
    """
    Compute complex single precision singular value decomposition.

    References
    ----------
    `cusolverDn<t>gesvd <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesvd>`_
    """

    jobu = jobu.encode('ascii')
    jobvt = jobvt.encode('ascii')
    status = _libcusolver.cusolverDnCgesvd(handle, jobu, jobvt, m, n,
                                           int(A), lda, int(S), int(U),
                                           ldu, int(VT), ldvt, int(Work),
                                           Lwork, int(rwork), int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnZgesvd_bufferSize.restype = int
_libcusolver.cusolverDnZgesvd_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnZgesvd_bufferSize(handle, m, n):
    """
    Calculate size of work buffer used by cusolverDnZgesvd.

    References
    ----------
    `cusolverDn<t>gesvd <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesvd>`_
    """

    Lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnZgesvd_bufferSize(handle, m, n, ctypes.byref(Lwork))
    cusolverCheckStatus(status)
    return Lwork.value

_libcusolver.cusolverDnZgesvd.restype = int
_libcusolver.cusolverDnZgesvd.argtypes = [ctypes.c_void_p,
                                          ctypes.c_char,
                                          ctypes.c_char,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnZgesvd(handle, jobu, jobvt, m, n, A, lda, S, U,
                     ldu, VT, ldvt, Work, Lwork, rwork, devInfo):
    """
    Compute complex double precision singular value decomposition.

    References
    ----------
    `cusolverDn<t>gesvd <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesvd>`_
    """

    jobu = jobu.encode('ascii')
    jobvt = jobvt.encode('ascii')
    status = _libcusolver.cusolverDnZgesvd(handle, jobu, jobvt, m, n,
                                           int(A), lda, int(S), int(U),
                                           ldu, int(VT), ldvt, int(Work),
                                           Lwork, int(rwork), int(devInfo))
    cusolverCheckStatus(status)

# SGEQRF, DGEQRF, CGEQRF, ZGEQRF
_libcusolver.cusolverDnSgeqrf_bufferSize.restype = int
_libcusolver.cusolverDnSgeqrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda):
    """
    Calculate size of work buffer used by cusolverDnSgeqrf.

    References
    ----------
    `cusolverDn<t>geqrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf>`_
    """

    Lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnSgeqrf_bufferSize(handle, m, n, int(A),
                                                      n, ctypes.byref(Lwork))
    cusolverCheckStatus(status)
    return Lwork.value

_libcusolver.cusolverDnSgeqrf.restype = int
_libcusolver.cusolverDnSgeqrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
def cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo):
    """
    Compute QR factorization of a real single precision m x n matrix.

    References
    ----------
    `cusolverDn<t>geqrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf>`_
    """

    status = _libcusolver.cusolverDnSgeqrf(handle, m, n, int(A), lda,
                                           int(TAU),
                                           int(Workspace),
                                           Lwork,
                                           int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnDgeqrf_bufferSize.restype = int
_libcusolver.cusolverDnDgeqrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda):
    """
    Calculate size of work buffer used by cusolverDnDgeqrf.

    References
    ----------
    `cusolverDn<t>geqrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf>`_
    """

    Lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnDgeqrf_bufferSize(handle, m, n, int(A),
                                                      n, ctypes.byref(Lwork))
    cusolverCheckStatus(status)
    return Lwork.value

_libcusolver.cusolverDnDgeqrf.restype = int
_libcusolver.cusolverDnDgeqrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
def cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo):
    """
    Compute QR factorization of a real double precision m x n matrix.

    References
    ----------
    `cusolverDn<t>geqrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf>`_
    """

    status = _libcusolver.cusolverDnDgeqrf(handle, m, n, int(A), lda,
                                           int(TAU),
                                           int(Workspace),
                                           Lwork,
                                           int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnCgeqrf_bufferSize.restype = int
_libcusolver.cusolverDnCgeqrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnCgeqrf_bufferSize(handle, m, n, A, lda):
    """
    Calculate size of work buffer used by cusolverDnCgeqrf.

    References
    ----------
    `cusolverDn<t>geqrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf>`_
    """

    Lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnCgeqrf_bufferSize(handle, m, n, int(A),
                                                      n, ctypes.byref(Lwork))
    cusolverCheckStatus(status)
    return Lwork.value

_libcusolver.cusolverDnCgeqrf.restype = int
_libcusolver.cusolverDnCgeqrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
def cusolverDnCgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo):
    """
    Compute QR factorization of a complex single precision m x n matrix.

    References
    ----------
    `cusolverDn<t>geqrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf>`_
    """

    status = _libcusolver.cusolverDnCgeqrf(handle, m, n, int(A), lda,
                                           int(TAU),
                                           int(Workspace),
                                           Lwork,
                                           int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnZgeqrf_bufferSize.restype = int
_libcusolver.cusolverDnZgeqrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnZgeqrf_bufferSize(handle, m, n, A, lda):
    """
    Calculate size of work buffer used by cusolverDnZgeqrf.

    References
    ----------
    `cusolverDn<t>geqrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf>`_
    """

    Lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnZgeqrf_bufferSize(handle, m, n, int(A),
                                                      n, ctypes.byref(Lwork))
    cusolverCheckStatus(status)
    return Lwork.value

_libcusolver.cusolverDnZgeqrf.restype = int
_libcusolver.cusolverDnZgeqrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
def cusolverDnZgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo):
    """
    Compute QR factorization of a complex double precision m x n matrix.

    References
    ----------
    `cusolverDn<t>geqrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf>`_
    """

    status = _libcusolver.cusolverDnZgeqrf(handle, m, n, int(A), lda,
                                           int(TAU),
                                           int(Workspace),
                                           Lwork,
                                           int(devInfo))
    cusolverCheckStatus(status)
