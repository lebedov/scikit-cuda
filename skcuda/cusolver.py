#!/usr/bin/env python

"""
Python interface to CUSOLVER functions.

Note: this module does not explicitly depend on PyCUDA.
"""

import re

from . import cudart

if int(cudart._cudart_version) < 7000:
    raise ImportError('CUSOLVER library only available in CUDA 7.0 and later')

import ctypes
import sys

import numpy as np

from . import cuda
from . import cublas

# Load library:
_linux_version_list = [10.1, 10.0, 9.2, 9.1, 9.0, 8.0, 7.5, 7.0]
_win32_version_list = [10, 100, 92, 91, 90, 80, 75, 70]
if 'linux' in sys.platform:
    _libcusolver_libname_list = ['libcusolver.so'] + \
                                ['libcusolver.so.%s' % v for v in _linux_version_list]

    # Fix for GOMP weirdness with CUDA 8.0 on Fedora (#171):
    try:
        ctypes.CDLL('libgomp.so.1', mode=ctypes.RTLD_GLOBAL)
    except:
        pass
    try:
        ctypes.CDLL('libgomp.so', mode=ctypes.RTLD_GLOBAL)
    except:
        pass
elif sys.platform == 'darwin':
    _libcusolver_libname_list = ['libcusolver.dylib']
elif sys.platform == 'win32':
    if sys.maxsize > 2**32:
        _libcusolver_libname_list = ['cusolver.dll'] + \
            ['cusolver64_%s.dll' % v for v in _win32_version_list]
    else:
        _libcusolver_libname_list = ['cusolver.dll'] + \
            ['cusolver32_%s.dll' % v for v in _win32_version_list]
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

# Values copied from cusolver_common.h
_CUSOLVER_EIG_TYPE = {
    1: 1,
    2: 2,
    3: 3,
    'CUSOLVER_EIG_TYPE_1': 1,
    'CUSOLVER_EIG_TYPE_2': 2,
    'CUSOLVER_EIG_TYPE_1': 3
}

_CUSOLVER_EIG_MODE = {
    0: 0,
    1: 1,
    'CUSOLVER_EIG_MODE_NOVECTOR': 0,
    'CUSOLVER_EIG_MODE_VECTOR': 1,
    'novector': 0,
    'vector': 1
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
            e = CUSOLVER_EXCEPTIONS[status]
        except KeyError:
            raise CUSOLVER_ERROR
        else:
            raise e

class _cusolver_version_req(object):
    """
    Decorator to replace function with a placeholder that raises an exception
    if the installed CUSOLVER version is not greater than `v`.
    """

    def __init__(self, v):
        self.vs = str(v)
        if isinstance(v, int):
            major = str(v)
            minor = '0'
        else:
            major, minor = re.search(r'(\d+)\.(\d+)', self.vs).groups()
        self.vi = major.ljust(len(major)+1, '0')+minor.ljust(2, '0')

    def __call__(self,f):
        def f_new(*args,**kwargs):
            raise NotImplementedError('CUSOLVER '+self.vs+' required')
        f_new.__doc__ = f.__doc__

        # Assumes that the CUSOLVER version is the same as that of the CUDART version:
        if int(cudart._cudart_version) >= int(self.vi):
            return f
        else:
            return f_new

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

if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnCreateSyevjInfo.restype = int
    _libcusolver.cusolverDnCreateSyevjInfo.argtypes = [ctypes.c_void_p]

@_cusolver_version_req(9.0)
def cusolverDnCreateSyevjInfo():
    info = ctypes.c_void_p()
    status = _libcusolver.cusolverDnCreateSyevjInfo(ctypes.byref(info))
    cusolverCheckStatus(status)
    return info.value

if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnDestroySyevjInfo.restype = int
    _libcusolver.cusolverDnDestroySyevjInfo.argtypes = [ctypes.c_void_p]

@_cusolver_version_req(9.0)
def cusolverDnDestroySyevjInfo(info):
    status = _libcusolver.cusolverDnDestroySyevjInfo(info)
    cusolverCheckStatus(status)

if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnXsyevjSetTolerance.restype = int
    _libcusolver.cusolverDnXsyevjSetTolerance.argtypes = [ctypes.c_void_p,
                                                          ctypes.c_void_p]

@_cusolver_version_req(9.0)
def cusolverDnXsyevjSetTolerance(info, tolerance):
    status = _libcusolver.cusolverDnXsyevjSetTolerance(
        info,
        ctypes.byref(ctypes.c_double(tolerance))
    )
    cusolverCheckStatus(status)

if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnXsyevjSetMaxSweeps.restype = int
    _libcusolver.cusolverDnXsyevjSetMaxSweeps.argtypes = [ctypes.c_void_p,
                                                          ctypes.c_int]

@_cusolver_version_req(9.0)
def cusolverDnXsyevjSetMaxSweeps(info, max_sweeps):
    status = _libcusolver.cusolverDnXsyevjSetMaxSweeps(info, max_sweeps)
    cusolverCheckStatus(status)

if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnXsyevjSetSortEig.restype = int
    _libcusolver.cusolverDnXsyevjSetSortEig.argtypes = [ctypes.c_void_p,
                                                        ctypes.c_int]

@_cusolver_version_req(9.0)
def cusolverDnXsyevjSetSortEig(info, sort_eig):
    status = _libcusolver.cusolverDnXsyevjSetSortEig(info, sort_eig)
    cusolverCheckStatus(status)

if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnXsyevjGetResidual.restype = int
    _libcusolver.cusolverDnXsyevjGetResidual.argtypes = [ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p]

@_cusolver_version_req(9.0)
def cusolverDnXsyevjGetResidual(handle, info):
    residual = ctypes.c_double()
    status = _libcusolver.cusolverDnXsyevjGetResidual(
        handle, info, ctypes.byref(residual))
    cusolverCheckStatus(status)
    return residual.value

if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnXsyevjGetSweeps.restype = int
    _libcusolver.cusolverDnXsyevjGetSweeps.argtypes = [ctypes.c_void_p,
                                                       ctypes.c_void_p,
                                                       ctypes.c_void_p]

@_cusolver_version_req(9.0)
def cusolverDnXsyevjGetSweeps(handle, info):
    executed_sweeps = ctypes.c_int()
    status = _libcusolver.cusolverDnXsyevjGetSweeps(
        handle, info, ctypes.byref(executed_sweeps))
    cusolverCheckStatus(status)
    return executed_sweeps.value


# Dense solver functions:

# SPOTRF, DPOTRF, CPOTRF, ZPOTRF
_libcusolver.cusolverDnSpotrf_bufferSize.restype = int
_libcusolver.cusolverDnSpotrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnSpotrf_bufferSize(handle, uplo, n, a, lda):
    """
    Calculate size of work buffer used by cusolverDnSpotrf.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnSpotrf_bufferSize(handle, uplo, n,
                                                      int(a),
                                                      lda, ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

_libcusolver.cusolverDnSpotrf.restype = int
_libcusolver.cusolverDnSpotrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnSpotrf(handle, uplo, n, a, lda, workspace, devIpiv, devInfo):
    """
    Compute Cholesky factorization of a real single precision Hermitian positive-definite matrix.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """

    status = _libcusolver.cusolverDnSpotrf(handle, uplo, n, int(a), lda,
                                           int(workspace),
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
def cusolverDnDpotrf_bufferSize(handle, uplo, n, a, lda):
    """
    Calculate size of work buffer used by cusolverDnDpotrf.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnDpotrf_bufferSize(handle, uplo, n,
                                                      int(a),
                                                      lda, ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

_libcusolver.cusolverDnDpotrf.restype = int
_libcusolver.cusolverDnDpotrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnDpotrf(handle, uplo, n, a, lda, workspace, devIpiv, devInfo):
    """
    Compute Cholesky factorization of a real double precision Hermitian positive-definite matrix.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """

    status = _libcusolver.cusolverDnDpotrf(handle, uplo, n, int(a), lda,
                                           int(workspace),
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
def cusolverDnCpotrf_bufferSize(handle, uplo, n, a, lda):
    """
    Calculate size of work buffer used by cusolverDnCpotrf.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnCpotrf_bufferSize(handle, uplo, n,
                                                      int(a),
                                                      lda, ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

_libcusolver.cusolverDnCpotrf.restype = int
_libcusolver.cusolverDnCpotrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnCpotrf(handle, uplo, n, a, lda, workspace, devIpiv, devInfo):
    """
    Compute Cholesky factorization of a complex single precision Hermitian positive-definite matrix.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """

    status = _libcusolver.cusolverDnCpotrf(handle, uplo, n, int(a), lda,
                                           int(workspace),
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
def cusolverDnZpotrf_bufferSize(handle, uplo, n, a, lda):
    """
    Calculate size of work buffer used by cusolverDnZpotrf.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnZpotrf_bufferSize(handle, uplo, n,
                                                      int(a),
                                                      lda, ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

_libcusolver.cusolverDnZpotrf.restype = int
_libcusolver.cusolverDnZpotrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnZpotrf(handle, uplo, n, a, lda, workspace, devIpiv, devInfo):
    """
    Compute Cholesky factorization of a complex double precision Hermitian positive-definite matrix.

    References
    ----------
    `cusolverDn<t>potrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf>`_
    """

    status = _libcusolver.cusolverDnZpotrf(handle, uplo, n, int(a), lda,
                                           int(workspace),
                                           int(devIpiv),
                                           int(devInfo))
    cusolverCheckStatus(status)

# SPOTRS, DPOTRS, CPOTRS, ZPOTRS
_libcusolver.cusolverDnSpotrs.restype = int
_libcusolver.cusolverDnSpotrs.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
def cusolverDnSpotrs(handle, uplo, n, nrhs, a, lda, B, ldb, devInfo):
    """
    Solve real single precision Hermitian positive-definite system.

    References
    ----------
    `cusolverDn<t>potrs <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrs>`_
    """

    status = _libcusolver.cusolverDnSpotrs(handle, uplo, n, nrhs, int(a), lda,
                                           int(B), ldb, int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnDpotrs.restype = int
_libcusolver.cusolverDnDpotrs.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
def cusolverDnDpotrs(handle, uplo, n, nrhs, a, lda, B, ldb, devInfo):
    """
    Solve real double precision Hermitian positive-definite system.

    References
    ----------
    `cusolverDn<t>potrs <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrs>`_
    """

    status = _libcusolver.cusolverDnDpotrs(handle, uplo, n, nrhs, int(a), lda,
                                           int(B), ldb, int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnCpotrs.restype = int
_libcusolver.cusolverDnCpotrs.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
def cusolverDnCpotrs(handle, uplo, n, nrhs, a, lda, B, ldb, devInfo):
    """
    Solve complex single precision Hermitian positive-definite system.

    References
    ----------
    `cusolverDn<t>potrs <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrs>`_
    """

    status = _libcusolver.cusolverDnCpotrs(handle, uplo, n, nrhs, int(a), lda,
                                           int(B), ldb, int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnZpotrs.restype = int
_libcusolver.cusolverDnZpotrs.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
def cusolverDnZpotrs(handle, uplo, n, nrhs, a, lda, B, ldb, devInfo):
    """
    Solve complex double precision Hermitian positive-definite system.

    References
    ----------
    `cusolverDn<t>potrs <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrs>`_
    """

    status = _libcusolver.cusolverDnZpotrs(handle, uplo, n, nrhs, int(a), lda,
                                           int(B), ldb, int(devInfo))
    cusolverCheckStatus(status)

# SGETRF, DGETRF, CGETRF, ZGETRF
_libcusolver.cusolverDnSgetrf_bufferSize.restype = int
_libcusolver.cusolverDnSgetrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnSgetrf_bufferSize(handle, m, n, a, lda):
    """
    Calculate size of work buffer used by cusolverDnSgetrf.

    References
    ----------
    `cusolverDn<t>getrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnSgetrf_bufferSize(handle, m, n,
                                                      int(a),
                                                      n, ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

_libcusolver.cusolverDnSgetrf.restype = int
_libcusolver.cusolverDnSgetrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnSgetrf(handle, m, n, a, lda, workspace, devIpiv, devInfo):
    """
    Compute LU factorization of a real single precision m x n matrix.

    References
    ----------
    `cusolverDn<t>getrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf>`_
    """

    status = _libcusolver.cusolverDnSgetrf(handle, m, n, int(a), lda,
                                          int(workspace),
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
def cusolverDnDgetrf_bufferSize(handle, m, n, a, lda):
    """
    Calculate size of work buffer used by cusolverDnDgetrf.

    References
    ----------
    `cusolverDn<t>getrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnDgetrf_bufferSize(handle, m, n,
                                                      int(a),
                                                      n, ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

_libcusolver.cusolverDnDgetrf.restype = int
_libcusolver.cusolverDnDgetrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnDgetrf(handle, m, n, a, lda, workspace, devIpiv, devInfo):
    """
    Compute LU factorization of a real double precision m x n matrix.

    References
    ----------
    `cusolverDn<t>getrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf>`_
    """

    status = _libcusolver.cusolverDnDgetrf(handle, m, n, int(a), lda,
                                          int(workspace),
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
def cusolverDnCgetrf_bufferSize(handle, m, n, a, lda):
    """
    Calculate size of work buffer used by cusolverDnCgetrf.

    References
    ----------
    `cusolverDn<t>getrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnCgetrf_bufferSize(handle, m, n,
                                                      int(a),
                                                      n, ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

_libcusolver.cusolverDnCgetrf.restype = int
_libcusolver.cusolverDnCgetrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnCgetrf(handle, m, n, a, lda, workspace, devIpiv, devInfo):
    """
    Compute LU factorization of a complex single precision m x n matrix.

    References
    ----------
    `cusolverDn<t>getrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf>`_
    """

    status = _libcusolver.cusolverDnCgetrf(handle, m, n, int(a), lda,
                                           int(workspace),
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
def cusolverDnZgetrf_bufferSize(handle, m, n, a, lda):
    """
    Calculate size of work buffer used by cusolverDnZgetrf.

    References
    ----------
    `cusolverDn<t>getrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnZgetrf_bufferSize(handle, m, n,
                                                      int(a),
                                                      n, ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

_libcusolver.cusolverDnZgetrf.restype = int
_libcusolver.cusolverDnZgetrf.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
def cusolverDnZgetrf(handle, m, n, a, lda, workspace, devIpiv, devInfo):
    """
    Compute LU factorization of a complex double precision m x n matrix.

    References
    ----------
    `cusolverDn<t>getrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf>`_
    """

    status = _libcusolver.cusolverDnZgetrf(handle, m, n, int(a), lda,
                                           int(workspace),
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
def cusolverDnSgetrs(handle, trans, n, nrhs, a, lda,
                     devIpiv, B, ldb, devInfo):
    """
    Solve real single precision linear system.

    References
    ----------
    `cusolverDn<t>getrs <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrs>`_
    """

    status = _libcusolver.cusolverDnSgetrs(handle, cublas._CUBLAS_OP[trans], n, nrhs,
                                           int(a), lda,
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
def cusolverDnDgetrs(handle, trans, n, nrhs, a, lda,
                     devIpiv, B, ldb, devInfo):
    """
    Solve real double precision linear system.

    References
    ----------
    `cusolverDn<t>getrs <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrs>`_
    """

    status = _libcusolver.cusolverDnDgetrs(handle, cublas._CUBLAS_OP[trans], n, nrhs,
                                           int(a), lda,
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
def cusolverDnCgetrs(handle, trans, n, nrhs, a, lda,
                     devIpiv, B, ldb, devInfo):
    """
    Solve complex single precision linear system.

    References
    ----------
    `cusolverDn<t>getrs <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrs>`_
    """

    status = _libcusolver.cusolverDnCgetrs(handle, cublas._CUBLAS_OP[trans], n, nrhs,
                                           int(a), lda,
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
def cusolverDnZgetrs(handle, trans, n, nrhs, a, lda,
                     devIpiv, B, ldb, devInfo):
    """
    Solve complex double precision linear system.

    References
    ----------
    `cusolverDn<t>getrs <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrs>`_
    """

    status = _libcusolver.cusolverDnZgetrs(handle, cublas._CUBLAS_OP[trans], n, nrhs,
                                           int(a), lda,
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

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnSgesvd_bufferSize(handle, m, n, ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

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
def cusolverDnSgesvd(handle, jobu, jobvt, m, n, a, lda, s, U,
        ldu, vt, ldvt, work, lwork, rwork, devInfo):
    """
    Compute real single precision singular value decomposition.

    References
    ----------
    `cusolverDn<t>gesvd <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesvd>`_
    """

    jobu = jobu.encode('ascii')
    jobvt = jobvt.encode('ascii')
    status = _libcusolver.cusolverDnSgesvd(handle, jobu, jobvt, m, n,
                                           int(a), lda, int(s), int(U),
                                           ldu, int(vt), ldvt, int(work),
                                           lwork, int(rwork), int(devInfo))
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

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnDgesvd_bufferSize(handle, m, n, ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

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
def cusolverDnDgesvd(handle, jobu, jobvt, m, n, a, lda, s, U,
                     ldu, vt, ldvt, work, lwork, rwork, devInfo):
    """
    Compute real double precision singular value decomposition.

    References
    ----------
    `cusolverDn<t>gesvd <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesvd>`_
    """

    jobu = jobu.encode('ascii')
    jobvt = jobvt.encode('ascii')
    status = _libcusolver.cusolverDnDgesvd(handle, jobu, jobvt, m, n,
                                           int(a), lda, int(s), int(U),
                                           ldu, int(vt), ldvt, int(work),
                                           lwork, int(rwork), int(devInfo))
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

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnCgesvd_bufferSize(handle, m, n, ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

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
def cusolverDnCgesvd(handle, jobu, jobvt, m, n, a, lda, s, U,
                     ldu, vt, ldvt, work, lwork, rwork, devInfo):
    """
    Compute complex single precision singular value decomposition.

    References
    ----------
    `cusolverDn<t>gesvd <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesvd>`_
    """

    jobu = jobu.encode('ascii')
    jobvt = jobvt.encode('ascii')
    status = _libcusolver.cusolverDnCgesvd(handle, jobu, jobvt, m, n,
                                           int(a), lda, int(s), int(U),
                                           ldu, int(vt), ldvt, int(work),
                                           lwork, int(rwork), int(devInfo))
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

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnZgesvd_bufferSize(handle, m, n, ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

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
def cusolverDnZgesvd(handle, jobu, jobvt, m, n, a, lda, s, U,
                     ldu, vt, ldvt, work, lwork, rwork, devInfo):
    """
    Compute complex double precision singular value decomposition.

    References
    ----------
    `cusolverDn<t>gesvd <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesvd>`_
    """

    jobu = jobu.encode('ascii')
    jobvt = jobvt.encode('ascii')
    status = _libcusolver.cusolverDnZgesvd(handle, jobu, jobvt, m, n,
                                           int(a), lda, int(s), int(U),
                                           ldu, int(vt), ldvt, int(work),
                                           lwork, int(rwork), int(devInfo))
    cusolverCheckStatus(status)

# SGEQRF, DGEQRF, CGEQRF, ZGEQRF
_libcusolver.cusolverDnSgeqrf_bufferSize.restype = int
_libcusolver.cusolverDnSgeqrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnSgeqrf_bufferSize(handle, m, n, a, lda):
    """
    Calculate size of work buffer used by cusolverDnSgeqrf.

    References
    ----------
    `cusolverDn<t>geqrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnSgeqrf_bufferSize(handle, m, n, int(a),
                                                      lda, ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

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
def cusolverDnSgeqrf(handle, m, n, a, lda, tau, workspace, lwork, devInfo):
    """
    Compute QR factorization of a real single precision m x n matrix.

    References
    ----------
    `cusolverDn<t>geqrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf>`_
    """

    status = _libcusolver.cusolverDnSgeqrf(handle, m, n, int(a), lda,
                                           int(tau),
                                           int(workspace),
                                           lwork,
                                           int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnDgeqrf_bufferSize.restype = int
_libcusolver.cusolverDnDgeqrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnDgeqrf_bufferSize(handle, m, n, a, lda):
    """
    Calculate size of work buffer used by cusolverDnDgeqrf.

    References
    ----------
    `cusolverDn<t>geqrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnDgeqrf_bufferSize(handle, m, n, int(a),
                                                      lda, ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

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
def cusolverDnDgeqrf(handle, m, n, a, lda, tau, workspace, lwork, devInfo):
    """
    Compute QR factorization of a real double precision m x n matrix.

    References
    ----------
    `cusolverDn<t>geqrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf>`_
    """

    status = _libcusolver.cusolverDnDgeqrf(handle, m, n, int(a), lda,
                                           int(tau),
                                           int(workspace),
                                           lwork,
                                           int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnCgeqrf_bufferSize.restype = int
_libcusolver.cusolverDnCgeqrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnCgeqrf_bufferSize(handle, m, n, a, lda):
    """
    Calculate size of work buffer used by cusolverDnCgeqrf.

    References
    ----------
    `cusolverDn<t>geqrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnCgeqrf_bufferSize(handle, m, n, int(a),
                                                      lda, ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

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
def cusolverDnCgeqrf(handle, m, n, a, lda, tau, workspace, lwork, devInfo):
    """
    Compute QR factorization of a complex single precision m x n matrix.

    References
    ----------
    `cusolverDn<t>geqrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf>`_
    """

    status = _libcusolver.cusolverDnCgeqrf(handle, m, n, int(a), lda,
                                           int(tau),
                                           int(workspace),
                                           lwork,
                                           int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnZgeqrf_bufferSize.restype = int
_libcusolver.cusolverDnZgeqrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p]
def cusolverDnZgeqrf_bufferSize(handle, m, n, a, lda):
    """
    Calculate size of work buffer used by cusolverDnZgeqrf.

    References
    ----------
    `cusolverDn<t>geqrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnZgeqrf_bufferSize(handle, m, n, int(a),
                                                      lda, ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

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
def cusolverDnZgeqrf(handle, m, n, a, lda, tau, workspace, lwork, devInfo):
    """
    Compute QR factorization of a complex double precision m x n matrix.

    References
    ----------
    `cusolverDn<t>geqrf <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf>`_
    """

    status = _libcusolver.cusolverDnZgeqrf(handle, m, n, int(a), lda,
                                           int(tau),
                                           int(workspace),
                                           lwork,
                                           int(devInfo))
    cusolverCheckStatus(status)

# SORGQR, DORGQR, CUNGQR, ZUNGQR
_libcusolver.cusolverDnSorgqr_bufferSize.restype = int
_libcusolver.cusolverDnSorgqr_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_void_p]
def cusolverDnSorgqr_bufferSize(handle, m, n, k, a, lda, tau):
    """
    Calculate size of work buffer used by cusolverDnSorgqr.

    References
    ----------
    `cusolverDn<t>orgqr <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-orgqr>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnSorgqr_bufferSize(handle, m, n, k, int(a),
                                                      lda, int(tau), ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

_libcusolver.cusolverDnSorgqr.restype = int
_libcusolver.cusolverDnSorgqr.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
def cusolverDnSorgqr(handle, m, n, k, a, lda, tau, work, lwork, devInfo):
    """
    Create unitary m x n matrix from single precision real reflection vectors.

    References
    ----------
    `cusolverDn<t>orgqr <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-orgqr>`_
    """

    status = _libcusolver.cusolverDnSorgqr(handle, m, n, k, int(a), lda,
                                           int(tau), int(work), lwork, int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnDorgqr_bufferSize.restype = int
_libcusolver.cusolverDnDorgqr_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_void_p]
def cusolverDnDorgqr_bufferSize(handle, m, n, k, a, lda, tau):
    """
    Calculate size of work buffer used by cusolverDnDorgqr.

    References
    ----------
    `cusolverDn<t>orgqr <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-orgqr>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnDorgqr_bufferSize(handle, m, n, k, int(a),
                                                      lda, int(tau), ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

_libcusolver.cusolverDnDorgqr.restype = int
_libcusolver.cusolverDnDorgqr.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
def cusolverDnDorgqr(handle, m, n, k, a, lda, tau, work, lwork, devInfo):
    """
    Create unitary m x n matrix from double precision real reflection vectors.

    References
    ----------
    `cusolverDn<t>orgqr <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-orgqr>`_
    """

    status = _libcusolver.cusolverDnDorgqr(handle, m, n, k, int(a), lda,
                                           int(tau), int(work), lwork, int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnCungqr_bufferSize.restype = int
_libcusolver.cusolverDnCungqr_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_void_p]
def cusolverDnCungqr_bufferSize(handle, m, n, k, a, lda, tau):
    """
    Calculate size of work buffer used by cusolverDnCungqr.

    References
    ----------
    `cusolverDn<t>orgqr <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-orgqr>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnCungqr_bufferSize(handle, m, n, k, int(a),
                                                      lda, int(tau), ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

_libcusolver.cusolverDnCungqr.restype = int
_libcusolver.cusolverDnCungqr.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
def cusolverDnCungqr(handle, m, n, k, a, lda, tau, work, lwork, devInfo):
    """
    Create unitary m x n matrix from single precision complex reflection vectors.

    References
    ----------
    `cusolverDn<t>orgqr <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-orgqr>`_
    """

    status = _libcusolver.cusolverDnCungqr(handle, m, n, k, int(a), lda,
                                           int(tau), int(work), lwork, int(devInfo))
    cusolverCheckStatus(status)

_libcusolver.cusolverDnZungqr_bufferSize.restype = int
_libcusolver.cusolverDnZungqr_bufferSize.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_void_p]
def cusolverDnZungqr_bufferSize(handle, m, n, k, a, lda, tau):
    """
    Calculate size of work buffer used by cusolverDnZungqr.

    References
    ----------
    `cusolverDn<t>orgqr <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-orgqr>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnZungqr_bufferSize(handle, m, n, k, int(a),
                                                      lda, int(tau), ctypes.byref(lwork))
    cusolverCheckStatus(status)
    return lwork.value

_libcusolver.cusolverDnZungqr.restype = int
_libcusolver.cusolverDnZungqr.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
def cusolverDnZungqr(handle, m, n, k, a, lda, tau, work, lwork, devInfo):
    """
    Create unitary m x n matrix from double precision complex reflection vectors.

    References
    ----------
    `cusolverDn<t>orgqr <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-orgqr>`_
    """

    status = _libcusolver.cusolverDnZungqr(handle, m, n, k, int(a), lda,
                                           int(tau), int(work), lwork, int(devInfo))
    cusolverCheckStatus(status)

# SYEVD
if cudart._cudart_version >= 8000:
    _libcusolver.cusolverDnSsyevd_bufferSize.restype = int
    _libcusolver.cusolverDnSsyevd_bufferSize.argtypes = [ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p]
@_cusolver_version_req(8.0)
def cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, a, lda, w):
    """
    Calculate size of work buffer used by culsolverDnSsyevd.

    References
    ----------
    `cusolverDn<t>gebrd <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-eigensolver-reference>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnSsyevd_bufferSize(
        handle,
        jobz,
        uplo,
        n,
        int(a),
        lda,
        int(w),
        ctypes.byref(lwork)
    )
    cusolverCheckStatus(status)
    return lwork.value

if cudart._cudart_version >= 8000:
    _libcusolver.cusolverDnSsyevd.restype = int
    _libcusolver.cusolverDnSsyevd.argtypes = [ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p]
@_cusolver_version_req(8.0)
def cusolverDnSsyevd(handle, jobz, uplo, n, a, lda, w, workspace, lwork,
                     devInfo):
    status = _libcusolver.cusolverDnSsyevd(
        handle,
        jobz,
        uplo,
        n,
        int(a),
        lda,
        int(w),
        int(workspace),
        lwork,
        int(devInfo)
    )
    cusolverCheckStatus(status)

if cudart._cudart_version >= 8000:
    _libcusolver.cusolverDnDsyevd_bufferSize.restype = int
    _libcusolver.cusolverDnDsyevd_bufferSize.argtypes = [ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p]
@_cusolver_version_req(8.0)
def cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, a, lda, w):
    """
    Calculate size of work buffer used by culsolverDnDsyevd.

    References
    ----------
    `cusolverDn<t>gebrd <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-eigensolver-reference>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnDsyevd_bufferSize(
        handle,
        jobz,
        uplo,
        n,
        int(a),
        lda,
        int(w),
        ctypes.byref(lwork)
    )
    cusolverCheckStatus(status)
    return lwork.value

if cudart._cudart_version >= 8000:
    _libcusolver.cusolverDnDsyevd.restype = int
    _libcusolver.cusolverDnDsyevd.argtypes = [ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p]
@_cusolver_version_req(8.0)
def cusolverDnDsyevd(handle, jobz, uplo, n, a, lda, w, workspace, lwork,
                     devInfo):
    status = _libcusolver.cusolverDnDsyevd(
        handle,
        jobz,
        uplo,
        n,
        int(a),
        lda,
        int(w),
        int(workspace),
        lwork,
        int(devInfo)
    )
    cusolverCheckStatus(status)


if cudart._cudart_version >= 8000:
    _libcusolver.cusolverDnCheevd_bufferSize.restype = int
    _libcusolver.cusolverDnCheevd_bufferSize.argtypes = [ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p]
@_cusolver_version_req(8.0)
def cusolverDnCheevd_bufferSize(handle, jobz, uplo, n, a, lda, w):
    """
    Calculate size of work buffer used by culsolverDnCheevd.

    References
    ----------
    `cusolverDn<t>gebrd <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-eigensolver-reference>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnCheevd_bufferSize(
        handle,
        jobz,
        uplo,
        n,
        int(a),
        lda,
        int(w),
        ctypes.byref(lwork)
    )
    cusolverCheckStatus(status)
    return lwork.value

if cudart._cudart_version >= 8000:
    _libcusolver.cusolverDnCheevd.restype = int
    _libcusolver.cusolverDnCheevd.argtypes = [ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p]
@_cusolver_version_req(8.0)
def cusolverDnCheevd(handle, jobz, uplo, n, a, lda, w, workspace, lwork,
                     devInfo):
    status = _libcusolver.cusolverDnCheevd(
        handle,
        jobz,
        uplo,
        n,
        int(a),
        lda,
        int(w),
        int(workspace),
        lwork,
        int(devInfo)
    )
    cusolverCheckStatus(status)


if cudart._cudart_version >= 8000:
    _libcusolver.cusolverDnZheevd_bufferSize.restype = int
    _libcusolver.cusolverDnZheevd_bufferSize.argtypes = [ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p]
@_cusolver_version_req(8.0)
def cusolverDnZheevd_bufferSize(handle, jobz, uplo, n, a, lda, w):
    """
    Calculate size of work buffer used by culsolverDnZheevd.

    References
    ----------
    `cusolverDn<t>gebrd <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-eigensolver-reference>`_
    """

    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnZheevd_bufferSize(
        handle,
        jobz,
        uplo,
        n,
        int(a),
        lda,
        int(w),
        ctypes.byref(lwork)
    )
    cusolverCheckStatus(status)
    return lwork.value

if cudart._cudart_version >= 8000:
    _libcusolver.cusolverDnZheevd.restype = int
    _libcusolver.cusolverDnZheevd.argtypes = [ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p]
@_cusolver_version_req(8.0)
def cusolverDnZheevd(handle, jobz, uplo, n, a, lda, w, workspace, lwork,
                     devInfo):
    status = _libcusolver.cusolverDnZheevd(
        handle,
        jobz,
        uplo,
        n,
        int(a),
        lda,
        int(w),
        int(workspace),
        lwork,
        int(devInfo)
    )
    cusolverCheckStatus(status)


# DnSsyevj and DnDsyevj
if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnSsyevj_bufferSize.restype = int
    _libcusolver.cusolverDnSsyevj_bufferSize.argtypes = [ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p]

@_cusolver_version_req(9.0)
def cusolverDnSsyevj_bufferSize(handle, jobz, uplo,
                                n, a, lda, w, params):
    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnSsyevj_bufferSize(
        handle,
        _CUSOLVER_EIG_MODE[jobz],
        cublas._CUBLAS_FILL_MODE[uplo],
        n,
        int(a),
        lda,
        int(w),
        ctypes.byref(lwork),
        params
    )
    cusolverCheckStatus(status)
    return lwork.value

if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnSsyevj.restype = int
    _libcusolver.cusolverDnSsyevj.argtypes = [ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p]

@_cusolver_version_req(9.0)
def cusolverDnSsyevj(handle, jobz, uplo,
                     n, a, lda, w, work,
                     lwork, info, params):
    status = _libcusolver.cusolverDnSsyevj(
        handle,
        _CUSOLVER_EIG_MODE[jobz],
        cublas._CUBLAS_FILL_MODE[uplo],
        n,
        int(a),
        lda,
        int(w),
        int(work),
        lwork,
        int(info),
        params
    )
    cusolverCheckStatus(status)

if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnDsyevj_bufferSize.restype = int
    _libcusolver.cusolverDnDsyevj_bufferSize.argtypes = [ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p]

@_cusolver_version_req(9.0)
def cusolverDnDsyevj_bufferSize(handle, jobz, uplo,
                                n, a, lda, w, params):
    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnDsyevj_bufferSize(
        handle,
        _CUSOLVER_EIG_MODE[jobz],
        cublas._CUBLAS_FILL_MODE[uplo],
        n,
        int(a),
        lda,
        int(w),
        ctypes.byref(lwork),
        params
    )
    cusolverCheckStatus(status)
    return lwork.value

if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnDsyevj.restype = int
    _libcusolver.cusolverDnDsyevj.argtypes = [ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p]

@_cusolver_version_req(9.0)
def cusolverDnDsyevj(handle, jobz, uplo,
                     n, a, lda, w, work,
                     lwork, info, params):
    status = _libcusolver.cusolverDnDsyevj(
        handle,
        _CUSOLVER_EIG_MODE[jobz],
        cublas._CUBLAS_FILL_MODE[uplo],
        n,
        int(a),
        lda,
        int(w),
        int(work),
        lwork,
        int(info),
        params
    )
    cusolverCheckStatus(status)

# DnCheevj and DnZheevj
if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnCheevj_bufferSize.restype = int
    _libcusolver.cusolverDnCheevj_bufferSize.argtypes = [ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p]

@_cusolver_version_req(9.0)
def cusolverDnCheevj_bufferSize(handle, jobz, uplo,
                                n, a, lda, w, params):
    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnCheevj_bufferSize(
        handle,
        _CUSOLVER_EIG_MODE[jobz],
        cublas._CUBLAS_FILL_MODE[uplo],
        n,
        int(a),
        lda,
        int(w),
        ctypes.byref(lwork),
        params
    )
    cusolverCheckStatus(status)
    return lwork.value

if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnCheevj.restype = int
    _libcusolver.cusolverDnCheevj.argtypes = [ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p]

@_cusolver_version_req(9.0)
def cusolverDnCheevj(handle, jobz, uplo,
                     n, a, lda, w, work,
                     lwork, info, params):
    status = _libcusolver.cusolverDnCheevj(
        handle,
        _CUSOLVER_EIG_MODE[jobz],
        cublas._CUBLAS_FILL_MODE[uplo],
        n,
        int(a),
        lda,
        int(w),
        int(work),
        lwork,
        int(info),
        params
    )
    cusolverCheckStatus(status)

if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnZheevj_bufferSize.restype = int
    _libcusolver.cusolverDnZheevj_bufferSize.argtypes = [ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p]

@_cusolver_version_req(9.0)
def cusolverDnZheevj_bufferSize(handle, jobz, uplo,
                                n, a, lda, w, params):
    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnZheevj_bufferSize(
        handle,
        _CUSOLVER_EIG_MODE[jobz],
        cublas._CUBLAS_FILL_MODE[uplo],
        n,
        int(a),
        lda,
        int(w),
        ctypes.byref(lwork),
        params
    )
    cusolverCheckStatus(status)
    return lwork.value

if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnZheevj.restype = int
    _libcusolver.cusolverDnZheevj.argtypes = [ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p]

@_cusolver_version_req(9.0)
def cusolverDnZheevj(handle, jobz, uplo,
                     n, a, lda, w, work,
                     lwork, info, params):
    status = _libcusolver.cusolverDnZheevj(
        handle,
        _CUSOLVER_EIG_MODE[jobz],
        cublas._CUBLAS_FILL_MODE[uplo],
        n,
        int(a),
        lda,
        int(w),
        int(work),
        lwork,
        int(info),
        params
    )
    cusolverCheckStatus(status)

# DnSsyevjBatched and DnDsyevjBatched
if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnSsyevjBatched_bufferSize.restype = int
    _libcusolver.cusolverDnSsyevjBatched_bufferSize.argtypes = [ctypes.c_void_p,
                                                                ctypes.c_int,
                                                                ctypes.c_int,
                                                                ctypes.c_int,
                                                                ctypes.c_void_p,
                                                                ctypes.c_int,
                                                                ctypes.c_void_p,
                                                                ctypes.c_void_p,
                                                                ctypes.c_void_p,
                                                                ctypes.c_int]

@_cusolver_version_req(9.0)
def cusolverDnSsyevjBatched_bufferSize(handle, jobz, uplo,
                                       n, a, lda, w, params, batchSize):
    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnSsyevjBatched_bufferSize(
        handle,
        _CUSOLVER_EIG_TYPE[jobz],
        cublas._CUBLAS_FILL_MODE[uplo],
        n,
        int(a),
        lda,
        int(w),
        ctypes.byref(lwork),
        params,
        batchSize
    )
    cusolverCheckStatus(status)
    return lwork.value

if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnSsyevjBatched.restype = int
    _libcusolver.cusolverDnSsyevjBatched.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int]

@_cusolver_version_req(9.0)
def cusolverDnSsyevjBatched(handle, jobz, uplo,
                             n, a, lda, w, work,
                             lwork, params, batchSize):
    info = ctypes.c_int()
    status = _libcusolver.cusolverDnSsyevjBatched(
        handle,
        _CUSOLVER_EIG_TYPE[jobz],
        cublas._CUBLAS_FILL_MODE[uplo],
        n,
        int(a),
        lda,
        int(w),
        int(work),
        lwork,
        ctypes.byref(info),
        params,
        batchSize
    )
    cusolverCheckStatus(status)
    return info

if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnDsyevjBatched_bufferSize.restype = int
    _libcusolver.cusolverDnDsyevjBatched_bufferSize.argtypes = [ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_int]

@_cusolver_version_req(9.0)
def cusolverDnDsyevjBatched_bufferSize(handle, jobz, uplo,
                                n, a, lda, w, params, batchSize):
    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnDsyevjBatched_bufferSize(
        handle,
        _CUSOLVER_EIG_MODE[jobz],
        cublas._CUBLAS_FILL_MODE[uplo],
        n,
        int(a),
        lda,
        int(w),
        ctypes.byref(lwork),
        params,
        batchSize
    )
    cusolverCheckStatus(status)
    return lwork.value

if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnDsyevjBatched.restype = int
    _libcusolver.cusolverDnDsyevjBatched.argtypes = [ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                              ctypes.c_int]

@_cusolver_version_req(9.0)
def cusolverDnDsyevjBatched(handle, jobz, uplo,
                     n, a, lda, w, work,
                     lwork, info, params, batchSize):
    status = _libcusolver.cusolverDnDsyevjBatched(
        handle,
        _CUSOLVER_EIG_MODE[jobz],
        cublas._CUBLAS_FILL_MODE[uplo],
        n,
        int(a),
        lda,
        int(w),
        int(work),
        lwork,
        int(info),
        params,
        batchSize
    )
    cusolverCheckStatus(status)

# DnCheevj and DnZheevj
if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnCheevjBatched_bufferSize.restype = int
    _libcusolver.cusolverDnCheevjBatched_bufferSize.argtypes = [ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_int]

@_cusolver_version_req(9.0)
def cusolverDnCheevjBatched_bufferSize(handle, jobz, uplo,
                                n, a, lda, w, params, batchSize):
    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnCheevjBatched_bufferSize(
        handle,
        _CUSOLVER_EIG_MODE[jobz],
        cublas._CUBLAS_FILL_MODE[uplo],
        n,
        int(a),
        lda,
        int(w),
        ctypes.byref(lwork),
        params,
        batchSize
    )
    cusolverCheckStatus(status)
    return lwork.value

if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnCheevjBatched.restype = int
    _libcusolver.cusolverDnCheevjBatched.argtypes = [ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                              ctypes.c_int]

@_cusolver_version_req(9.0)
def cusolverDnCheevjBatched(handle, jobz, uplo,
                     n, a, lda, w, work,
                     lwork, info, params, batchSize):
    status = _libcusolver.cusolverDnCheevjBatched(
        handle,
        _CUSOLVER_EIG_MODE[jobz],
        cublas._CUBLAS_FILL_MODE[uplo],
        n,
        int(a),
        lda,
        int(w),
        int(work),
        lwork,
        int(info),
        params,
        batchSize
    )
    cusolverCheckStatus(status)

if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnZheevjBatched_bufferSize.restype = int
    _libcusolver.cusolverDnZheevjBatched_bufferSize.argtypes = [ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_int]

@_cusolver_version_req(9.0)
def cusolverDnZheevjBatched_bufferSize(handle, jobz, uplo,
                                n, a, lda, w, params, batchSize):
    lwork = ctypes.c_int()
    status = _libcusolver.cusolverDnZheevjBatched_bufferSize(
        handle,
        _CUSOLVER_EIG_MODE[jobz],
        cublas._CUBLAS_FILL_MODE[uplo],
        n,
        int(a),
        lda,
        int(w),
        ctypes.byref(lwork),
        params,
        batchSize
    )
    cusolverCheckStatus(status)
    return lwork.value

if cudart._cudart_version >= 9000:
    _libcusolver.cusolverDnZheevjBatched.restype = int
    _libcusolver.cusolverDnZheevjBatched.argtypes = [ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                              ctypes.c_int]

@_cusolver_version_req(9.0)
def cusolverDnZheevjBatched(handle, jobz, uplo,
                     n, a, lda, w, work,
                     lwork, info, params, batchSize):
    status = _libcusolver.cusolverDnZheevjBatched(
        handle,
        _CUSOLVER_EIG_MODE[jobz],
        cublas._CUBLAS_FILL_MODE[uplo],
        n,
        int(a),
        lda,
        int(w),
        int(work),
        lwork,
        int(info),
        params,
        batchSize
    )
    cusolverCheckStatus(status)
