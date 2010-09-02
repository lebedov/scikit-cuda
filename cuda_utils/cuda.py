#!/usr/bin/env python

"""
Python interface to CUDA functions.

Note: this module does not explicitly depend on PyCUDA.
"""

import sys
import ctypes
import atexit

# Load CUDA libraries:
if sys.platform == 'linux2':
    _libcuda_libname = 'libcuda.so'
    _libcudart_libname = 'libcudart.so'
elif sys.platform == 'darwin':
    _libcuda_libname = 'libcuda.dylib'
    _libcudart_libname = 'libcudart.dylib'
else:
    raise RuntimeError('unsupported platform')

try:
    _libcuda = ctypes.cdll.LoadLibrary(_libcuda_libname)
except OSError:
    print '%s not found' % _libcuda_libname
    
try:
    _libcudart = ctypes.cdll.LoadLibrary(_libcudart_libname)
except OSError:
    print '% not found' % _libcudart_libname

# Code adapted from PARRET:
def POINTER(obj):
    """
    Create ctypes pointer to object.

    Notes
    -----
    This function converts None to a real NULL pointer because of bug
    in how ctypes handles None on 64-bit platforms.

    """
    
    p = ctypes.POINTER(obj)
    if not isinstance(p.from_param, classmethod):
        def from_param(cls, x):
            if x is None:
                return cls()
            else:
                return x
        p.from_param = classmethod(from_param)

    return p

# Classes corresponding to CUDA vector structures:
class float2(ctypes.Structure):
    pass

float2._fields_ = [
    ('x', ctypes.c_float),
    ('y', ctypes.c_float)
    ]

class double2(ctypes.Structure):
    pass

double2._fields_ = [
    ('x', ctypes.c_double),
    ('y', ctypes.c_double)
    ]

cuFloatComplex = float2
cuDoubleComplex = double2

_libcudart.cudaGetErrorString.restype = ctypes.c_char_p
_libcudart.cudaGetErrorString.argtypes = [ctypes.c_int]
def cudaGetErrorString(e):
    """
    Retrieve CUDA error string.
    
    Return the string associated with the specified CUDA error status
    code.

    Parameters
    ----------
    e : int
        Error number.

    Returns
    -------
    s : str
        Error string.
        
    """

    return _libcudart.cudaGetErrorString(e)

# Generic CUDA error:
class cudaError(Exception):
    """CUDA error."""
    pass

# Exceptions corresponding to various CUDA runtime errors:
class cudaErrorMissingConfiguration(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(1)
    pass

class cudaErrorMemoryAllocation(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(2)
    pass

class cudaErrorInitializationError(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(3)
    pass

class cudaErrorLaunchFailure(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(4)
    pass

class cudaErrorPriorLaunchFailure(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(5)
    pass

class cudaErrorLaunchTimeout(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(6)
    pass

class cudaErrorLaunchOutOfResources(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(7)
    pass

class cudaErrorInvalidDeviceFunction(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(8)
    pass

class cudaErrorInvalidConfiguration(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(9)
    pass

class cudaErrorInvalidDevice(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(10)
    pass

class cudaErrorInvalidValue(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(11)
    pass

class cudaErrorInvalidPitchValue(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(12)
    pass

class cudaErrorInvalidSymbol(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(13)
    pass

class cudaErrorMapBufferObjectFailed(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(14)
    pass

class cudaErrorUnmapBufferObjectFailed(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(15)
    pass

class cudaErrorInvalidHostPointer(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(16)
    pass

class cudaErrorInvalidDevicePointer(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(17)
    pass

class cudaErrorInvalidTexture(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(18)
    pass

class cudaErrorInvalidTextureBinding(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(19)
    pass

class cudaErrorInvalidChannelDescriptor(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(20)
    pass

class cudaErrorInvalidMemcpyDirection(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(21)
    pass

class cudaErrorTextureFetchFailed(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(23)
    pass

class cudaErrorTextureNotBound(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(24)
    pass

class cudaErrorSynchronizationError(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(25)
    pass

class cudaErrorInvalidFilterSetting(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(26)
    pass

class cudaErrorInvalidNormSetting(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(27)
    pass

class cudaErrorMixedDeviceExecution(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(28)
    pass

class cudaErrorUnknown(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(30)
    pass

class cudaErrorNotYetImplemented(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(31)
    pass

class cudaErrorMemoryValueTooLarge(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(32)
    pass

class cudaErrorInvalidResourceHandle(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(33)
    pass

class cudaErrorNotReady(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(34)
    pass

class cudaErrorInsufficientDriver(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(35)
    pass

class cudaErrorSetOnActiveProcess(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(36)
    pass

class cudaErrorInvalidSurface(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(37)
    pass

class cudaErrorNoDevice(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(38)
    pass

class cudaErrorECCUncorrectable(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(39)
    pass

class cudaErrorSharedObjectSymbolNotFound(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(40)
    pass

class cudaErrorSharedObjectInitFailed(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(41)
    pass

class cudaErrorUnsupportedLimit(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(42)
    pass

class cudaErrorDuplicateVariableName(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(43)
    pass

class cudaErrorDuplicateTextureName(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(44)
    pass

class cudaErrorDuplicateSurfaceName(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(45)
    pass

class cudaErrorDevicesUnavailable(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(46)
    pass

class cudaErrorStartupFailure(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(127)
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
    """
    Raise CUDA exception.

    Raise an exception corresponding to the specified CUDA error
    code.
    
    Parameters
    ----------
    status : int
        CUDA error code.

    See Also
    --------
    cudaExceptions
    
    """

    if status != 0:
        try:
            raise cudaExceptions[status]
        except KeyError:
            raise cudaError

# Memory allocation functions (adapted from pystream):
_libcudart.cudaMalloc.restype = int
_libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                  ctypes.c_size_t]
def cudaMalloc(count, ctype=None):
    """
    Allocate device memory.

    Allocate memory on the device associated with the current active
    context.

    Parameters
    ----------
    count : int
        Number of bytes of memory to allocate
    ctype : _ctypes.SimpleType, optional
        ctypes type to cast returned pointer.
        
    Returns
    -------
    ptr : ctypes pointer
        Pointer to allocated device memory.

    """
    
    ptr = ctypes.c_void_p()
    status = _libcudart.cudaMalloc(ctypes.byref(ptr), count)
    cudaCheckStatus(status)
    if ctype != None:
        ptr = ctypes.cast(ptr, ctypes.POINTER(ctype))
    return ptr

_libcudart.cudaFree.restype = int
_libcudart.cudaFree.argtypes = [ctypes.c_void_p]
def cudaFree(ptr):
    """
    Free device memory.

    Free allocated memory on the device associated with the current active
    context.

    Parameters
    ----------
    ptr : ctypes pointer
        Pointer to allocated device memory.

    """
    
    status = _libcudart.cudaFree(ptr)
    cudaCheckStatus(status)

_libcudart.cudaMallocPitch.restype = int
_libcudart.cudaMallocPitch.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.POINTER(ctypes.c_size_t),
                                       ctypes.c_size_t, ctypes.c_size_t]
def cudaMallocPitch(pitch, rows, cols, elesize):
    """
    Allocate pitched device memory.

    Allocate pitched memory on the device associated with the current active
    context.
    
    Parameters
    ----------
    pitch : int
        Pitch for allocation.
    rows : int
        Requested pitched allocation height.
    cols : int
        Requested pitched allocation width.
    elesize : int
        Size of memory element.

    Returns
    -------
    ptr : ctypes pointer
        Pointer to allocated device memory.
        
    """
    
    ptr = ctypes.c_void_p()
    status = _libcudart.cudaMallocPitch(ctypes.byref(ptr),
                                        ctypes.c_size_t(pitch), cols*elesize,
                                        rows)
    cudaCheckStatus(status)
    return ptr, pitch

# Memory copy modes:
cudaMemcpyHostToHost = 0
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaMemcpyDeviceToDevice = 3

_libcudart.cudaMemcpy.restype = int
_libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_size_t, ctypes.c_int]
def cudaMemcpy_htod(dst, src, count):
    """
    Copy memory from host to device.

    Copy data from host memory to device memory.

    Parameters
    ----------
    dst : ctypes pointer
        Device memory pointer.
    src : ctypes pointer
        Host memory pointer.
    count : int
        Number of bytes to copy.
    
    """
    
    status = _libcudart.cudaMemcpy(dst, src,
                                   ctypes.c_size_t(count),
                                   cudaMemcpyHostToDevice)
    cudaCheckStatus(status)
    
def cudaMemcpy_dtoh(dst, src, count):
    """
    Copy memory from device to host.

    Copy data from device memory to host memory.

    Parameters
    ----------
    dst : ctypes pointer
        Host memory pointer.
    src : ctypes pointer
        Device memory pointer.
    count : int
        Number of bytes to copy.

    """

    status = _libcudart.cudaMemcpy(dst, src,
                                   ctypes.c_size_t(count),
                                   cudaMemcpyDeviceToHost)
    cudaCheckStatus(status)

_libcudart.cudaSetDevice.restype = int
_libcudart.cudaSetDevice.argtypes = [ctypes.c_int]
def cudaSetDevice(dev):
    """
    Set current CUDA device.

    Select a device to use for subsequent CUDA operations.
    
    Parameters
    ----------
    dev : int
        Device number.

    """

    status = _libcudart.cudaSetDevice(dev)
    cudaCheckStatus(status)
    
_libcudart.cudaGetDevice.restype = int
_libcudart.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
def cudaGetDevice():
    """
    Get current CUDA device.

    Return the identifying number of the device currently used to
    process CUDA operations.

    Returns
    -------
    dev : int
        Device number.

    """    

    dev = ctypes.c_int()
    status = _libcudart.cudaGetDevice(ctypes.byref(dev))
    cudaCheckStatus(status)
    return dev.value
