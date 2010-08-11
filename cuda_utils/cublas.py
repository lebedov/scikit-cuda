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

_libcublas.cublasGetError.restype = int
_libcublas.cublasGetError.argtypes = []
def cublasGetError():
    """Returns and resets the current CUBLAS error code."""

    return _libcublas.cublasGetError()

def cublasCheckStatus(status):
    """Raise an exception if the specified CUBLAS status is an error."""

    if status != 0:
        try:
            raise cublasExceptions[status]
        except KeyError:
            raise cublasError


_libcublas.cublasInit.restype = int
_libcublas.cublasInit.argtypes = []
def cublasInit():
    """Must be called before using any other CUBLAS functions."""
    
    status = _libcublas.cublasInit()
    cublasCheckStatus(status)

_libcublas.cublasShutdown.restype = int
_libcublas.cublasShutdown.argtypes = []
def cublasShutdown():
    """Shuts down CUBLAS."""

    status = _libcublas.cublasShutdown()
    cublasCheckStatus(status)

atexit.register(_libcublas.cublasShutdown)

# BLAS functions implemented by CUBLAS:
_libcublas.cublasSgemm.restype = None
_libcublas.cublasSgemm.argtypes = [ctypes.c_char,
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
_libcublas.cublasCgemm.restype = None
_libcublas.cublasCgemm.argtypes = [ctypes.c_char,
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

_cublasDgemm = _libcublas.cublasDgemm
_cublasDgemm.restype = None
_cublasDgemm.argtypes = [ctypes.c_char,
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
_cublasZgemm = _libcublas.cublasZgemm
_cublasZgemm.restype = None
_cublasZgemm.argtypes = [ctypes.c_char,
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

if __name__ == "__main__":
    import doctest
    doctest.testmod()
