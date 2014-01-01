#!/usr/bin/env python

"""
Python interface to CUDA runtime functions.
"""

import re

import cffi
_ffi = cffi.FFI()

_ffi.cdef("""
enum cudaError {
    cudaSuccess                           =      0,
    cudaErrorMissingConfiguration         =      1,
    cudaErrorMemoryAllocation             =      2,
    cudaErrorInitializationError          =      3,
    cudaErrorLaunchFailure                =      4,
    cudaErrorPriorLaunchFailure           =      5,
    cudaErrorLaunchTimeout                =      6,
    cudaErrorLaunchOutOfResources         =      7,
    cudaErrorInvalidDeviceFunction        =      8,
    cudaErrorInvalidConfiguration         =      9,
    cudaErrorInvalidDevice                =     10,
    cudaErrorInvalidValue                 =     11,
    cudaErrorInvalidPitchValue            =     12,
    cudaErrorInvalidSymbol                =     13,
    cudaErrorMapBufferObjectFailed        =     14,
    cudaErrorUnmapBufferObjectFailed      =     15,
    cudaErrorInvalidHostPointer           =     16,
    cudaErrorInvalidDevicePointer         =     17,
    cudaErrorInvalidTexture               =     18,
    cudaErrorInvalidTextureBinding        =     19,
    cudaErrorInvalidChannelDescriptor     =     20,
    cudaErrorInvalidMemcpyDirection       =     21,
    cudaErrorAddressOfConstant            =     22,
    cudaErrorTextureFetchFailed           =     23,
    cudaErrorTextureNotBound              =     24,
    cudaErrorSynchronizationError         =     25,
    cudaErrorInvalidFilterSetting         =     26,
    cudaErrorInvalidNormSetting           =     27,
    cudaErrorMixedDeviceExecution         =     28,
    cudaErrorCudartUnloading              =     29,
    cudaErrorUnknown                      =     30,
    cudaErrorNotYetImplemented            =     31,
    cudaErrorMemoryValueTooLarge          =     32,
    cudaErrorInvalidResourceHandle        =     33,
    cudaErrorNotReady                     =     34,
    cudaErrorInsufficientDriver           =     35,
    cudaErrorSetOnActiveProcess           =     36,
    cudaErrorInvalidSurface               =     37,
    cudaErrorNoDevice                     =     38,
    cudaErrorECCUncorrectable             =     39,
    cudaErrorSharedObjectSymbolNotFound   =     40,
    cudaErrorSharedObjectInitFailed       =     41,
    cudaErrorUnsupportedLimit             =     42,
    cudaErrorDuplicateVariableName        =     43,
    cudaErrorDuplicateTextureName         =     44,
    cudaErrorDuplicateSurfaceName         =     45,
    cudaErrorDevicesUnavailable           =     46,
    cudaErrorInvalidKernelImage           =     47,
    cudaErrorNoKernelImageForDevice       =     48,
    cudaErrorIncompatibleDriverContext    =     49,
    cudaErrorPeerAccessAlreadyEnabled     =     50,
    cudaErrorPeerAccessNotEnabled         =     51,
    cudaErrorDeviceAlreadyInUse           =     54,
    cudaErrorProfilerDisabled             =     55,
    cudaErrorProfilerNotInitialized       =     56,
    cudaErrorProfilerAlreadyStarted       =     57,
    cudaErrorProfilerAlreadyStopped       =     58,
    cudaErrorAssert                       =     59,
    cudaErrorTooManyPeers                 =     60,
    cudaErrorHostMemoryAlreadyRegistered  =     61,
    cudaErrorHostMemoryNotRegistered      =     62,
    cudaErrorOperatingSystem              =     63,
    cudaErrorPeerAccessUnsupported        =     64,
    cudaErrorLaunchMaxDepthExceeded       =     65,
    cudaErrorLaunchFileScopedTex          =     66,
    cudaErrorLaunchFileScopedSurf         =     67,
    cudaErrorSyncDepthExceeded            =     68,
    cudaErrorLaunchPendingCountExceeded   =     69,
    cudaErrorNotPermitted                 =     70,
    cudaErrorNotSupported                 =     71,
    cudaErrorStartupFailure               =   0x7f,
    cudaErrorApiFailureBase               =  10000   
};
typedef enum cudaError cudaError_t;
const char* cudaGetErrorString(cudaError_t error);
""")

_ffi_lib = _ffi.verify("""
#include <cuda_runtime_api.h>
#include <driver_types.h>
""", libraries=['cudart'])

# Import all cudaError* definitions directly into module namespace:
for k, v in _ffi_lib.__dict__.iteritems():
    if re.match('cudaError.*', k):
        vars()[k] = v

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

    return _ffi.string(_ffi_lib.cudaGetErrorString(e))

# Generic CUDA error:
class cudaError(Exception):
    """CUDA error."""
    pass

# Exceptions corresponding to various CUDA runtime errors:
class cudaErrorMissingConfiguration(cudaError):
    __doc__ = cudaGetErrorString(1)
    pass

class cudaErrorMemoryAllocation(cudaError):
    __doc__ = cudaGetErrorString(2)
    pass

class cudaErrorInitializationError(cudaError):
    __doc__ = cudaGetErrorString(3)
    pass

class cudaErrorLaunchFailure(cudaError):
    __doc__ = cudaGetErrorString(4)
    pass

class cudaErrorPriorLaunchFailure(cudaError):
    __doc__ = cudaGetErrorString(5)
    pass

class cudaErrorLaunchTimeout(cudaError):
    __doc__ = cudaGetErrorString(6)
    pass

class cudaErrorLaunchOutOfResources(cudaError):
    __doc__ = cudaGetErrorString(7)
    pass

class cudaErrorInvalidDeviceFunction(cudaError):
    __doc__ = cudaGetErrorString(8)
    pass

class cudaErrorInvalidConfiguration(cudaError):
    __doc__ = cudaGetErrorString(9)
    pass

class cudaErrorInvalidDevice(cudaError):
    __doc__ = cudaGetErrorString(10)
    pass

class cudaErrorInvalidValue(cudaError):
    __doc__ = cudaGetErrorString(11)
    pass

class cudaErrorInvalidPitchValue(cudaError):
    __doc__ = cudaGetErrorString(12)
    pass

class cudaErrorInvalidSymbol(cudaError):
    __doc__ = cudaGetErrorString(13)
    pass

class cudaErrorMapBufferObjectFailed(cudaError):
    __doc__ = cudaGetErrorString(14)
    pass

class cudaErrorUnmapBufferObjectFailed(cudaError):
    __doc__ = cudaGetErrorString(15)
    pass

class cudaErrorInvalidHostPointer(cudaError):
    __doc__ = cudaGetErrorString(16)
    pass

class cudaErrorInvalidDevicePointer(cudaError):
    __doc__ = cudaGetErrorString(17)
    pass

class cudaErrorInvalidTexture(cudaError):
    __doc__ = cudaGetErrorString(18)
    pass

class cudaErrorInvalidTextureBinding(cudaError):
    __doc__ = cudaGetErrorString(19)
    pass

class cudaErrorInvalidChannelDescriptor(cudaError):
    __doc__ = cudaGetErrorString(20)
    pass

class cudaErrorInvalidMemcpyDirection(cudaError):
    __doc__ = cudaGetErrorString(21)
    pass

class cudaErrorTextureFetchFailed(cudaError):
    __doc__ = cudaGetErrorString(23)
    pass

class cudaErrorTextureNotBound(cudaError):
    __doc__ = cudaGetErrorString(24)
    pass

class cudaErrorSynchronizationError(cudaError):
    __doc__ = cudaGetErrorString(25)
    pass

class cudaErrorInvalidFilterSetting(cudaError):
    __doc__ = cudaGetErrorString(26)
    pass

class cudaErrorInvalidNormSetting(cudaError):
    __doc__ = cudaGetErrorString(27)
    pass

class cudaErrorMixedDeviceExecution(cudaError):
    __doc__ = cudaGetErrorString(28)
    pass

class cudaErrorCudartUnloading(cudaError):
    __doc__ = cudaGetErrorString(29)
    pass

class cudaErrorUnknown(cudaError):
    __doc__ = cudaGetErrorString(30)
    pass

class cudaErrorNotYetImplemented(cudaError):
    __doc__ = cudaGetErrorString(31)
    pass

class cudaErrorMemoryValueTooLarge(cudaError):
    __doc__ = cudaGetErrorString(32)
    pass

class cudaErrorInvalidResourceHandle(cudaError):
    __doc__ = cudaGetErrorString(33)
    pass

class cudaErrorNotReady(cudaError):
    __doc__ = cudaGetErrorString(34)
    pass

class cudaErrorInsufficientDriver(cudaError):
    __doc__ = cudaGetErrorString(35)
    pass

class cudaErrorSetOnActiveProcess(cudaError):
    __doc__ = cudaGetErrorString(36)
    pass

class cudaErrorInvalidSurface(cudaError):
    __doc__ = cudaGetErrorString(37)
    pass

class cudaErrorNoDevice(cudaError):
    __doc__ = cudaGetErrorString(38)
    pass

class cudaErrorECCUncorrectable(cudaError):
    __doc__ = cudaGetErrorString(39)
    pass

class cudaErrorSharedObjectSymbolNotFound(cudaError):
    __doc__ = cudaGetErrorString(40)
    pass

class cudaErrorSharedObjectInitFailed(cudaError):
    __doc__ = cudaGetErrorString(41)
    pass

class cudaErrorUnsupportedLimit(cudaError):
    __doc__ = cudaGetErrorString(42)
    pass

class cudaErrorDuplicateVariableName(cudaError):
    __doc__ = cudaGetErrorString(43)
    pass

class cudaErrorDuplicateTextureName(cudaError):
    __doc__ = cudaGetErrorString(44)
    pass

class cudaErrorDuplicateSurfaceName(cudaError):
    __doc__ = cudaGetErrorString(45)
    pass

class cudaErrorDevicesUnavailable(cudaError):
    __doc__ = cudaGetErrorString(46)
    pass

class cudaErrorInvalidKernelImage(cudaError):
    __doc__ = cudaGetErrorString(47)
    pass

class cudaErrorNoKernelImageForDevice(cudaError):
    __doc__ = cudaGetErrorString(48)
    pass

class cudaErrorIncompatibleDriverContext(cudaError):
    __doc__ = cudaGetErrorString(49)
    pass

class cudaErrorPeerAccessAlreadyEnabled(cudaError):
    __doc__ = cudaGetErrorString(50)
    pass

class cudaErrorPeerAccessNotEnabled(cudaError):
    __doc__ = cudaGetErrorString(51)
    pass

class cudaErrorDeviceAlreadyInUse(cudaError):
    __doc__ = cudaGetErrorString(54)
    pass

class cudaErrorProfilerDisabled(cudaError):
    __doc__ = cudaGetErrorString(55)
    pass

class cudaErrorProfilerNotInitialized(cudaError):
    __doc__ = cudaGetErrorString(56)
    pass

class cudaErrorProfilerAlreadyStarted(cudaError):
    __doc__ = cudaGetErrorString(57)
    pass

class cudaErrorProfilerAlreadyStopped(cudaError):
    __doc__ = cudaGetErrorString(58)
    pass

class cudaErrorStartupFailure(cudaError):
    __doc__ = cudaGetErrorString(127)
    pass

cudaExceptions = \
    {k:v for k,v in _ffi_lib.__dict__.iteritems() if re.match('cudaError.*', k)}

def cudaCheckStatus(status):
    """
    Raise CUDA exception.

    Raise an exception corresponding to the specified CUDA runtime error
    code.

    Parameters
    ----------
    status : int
        CUDA runtime error code.

    See Also
    --------
    cudaExceptions

    """

    if status != 0:
        try:
            raise cudaExceptions[status]
        except KeyError:
            raise cudaError
                  
