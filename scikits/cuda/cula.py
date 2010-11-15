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
    """Get string associated with the specified CULA error status code."""

    return _libcula.culaGetStatusString(e)

# Generic CULA error:
class culaError(Exception):
    """CULA error."""
    pass

# Exceptions corresponding to various CULA errors:
class culaNotFound(culaError):
    """CULA shared library not found"""
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
def culaCheckStatus(status):
    """Raise an exception corresponding to the specified CULA status
    code."""
    
    if status != 0:
        try:
            raise culaExceptions[status]
        except KeyError:
            raise culaError

_libcula.culaGetErrorInfo.restype = int
_libcula.culaGetErrorInfo.argtype = [ctypes.c_int]
def culaGetErrorInfo(e):
    """Returns extended information about the last CULA error."""

    return _libcula.culaGetErrorInfo(e)

def culaGetLastStatus():
    """Returns the last status code returned from a CULA function."""
    
    return _libcula.culaGetLastStatus()

_libcula.culaSelectDevice.restype = int
_libcula.culaSelectDevice.argtype = [ctypes.c_int]
def culaSelectDevice(dev):
    """Selects a device with which CULA will operate. Must be called
    before culaInitialize()."""

    status = _libcula.culaSelectDevice(dev)
    culaCheckStatus(status)

def culaInitialize():
    """Must be called before using any other CULA function."""
    
    status = _libcula.culaInitialize()
    culaCheckStatus(status)

def culaShutdown():
    """Shuts down CULA."""
    
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

_libcula.culaSgetrf.restype = \
_libcula.culaCgetrf.restype = int

_libcula.culaSgetrf.argtypes = \
_libcula.culaCgetrf.argtypes = [ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_void_p,
                                ctypes.c_int,
                                ctypes.c_void_p]

_libcula.culaSgeqrf.restype = \
_libcula.culaCgeqrf.restype = int

_libcula.culaSgeqrf.argtypes = \
_libcula.culaCgeqrf.argtypes = [ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_void_p,
                                ctypes.c_int,
                                ctypes.c_void_p]


_libcula.culaSgels.restype = \
_libcula.culaCgels.restype = int

_libcula.culaSgels.argtypes = \
_libcula.culaCgels.argtypes = [ctypes.c_char,                           
                               ctypes.c_int,
                               ctypes.c_int,
                               ctypes.c_int,
                               ctypes.c_void_p,                              
                               ctypes.c_int,
                               ctypes.c_void_p,
                               ctypes.c_int]

_libcula.culaSgglse.restype = \
_libcula.culaCgglse.restype = int

_libcula.culaSgglse.argtypes = \
_libcula.culaCgglse.argtypes = [ctypes.c_int,                             
                                ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_void_p,
                                ctypes.c_int,
                                ctypes.c_void_p,
                                ctypes.c_int,
                                ctypes.c_void_p,
                                ctypes.c_void_p,
                                ctypes.c_void_p]

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

# LAPACK functions available in CULA premium:
if _libcula_toolkit == 'premium':
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
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
