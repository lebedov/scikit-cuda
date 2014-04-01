#!/usr/bin/env python

"""
Python interface to CUDA runtime functions.
"""

import re
import struct

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

enum cudaMemoryType {
    cudaMemoryTypeHost   = 1, /**< Host memory */
    cudaMemoryTypeDevice = 2  /**< Device memory */
};
enum cudaMemcpyKind {
    cudaMemcpyHostToHost          =   0,
    cudaMemcpyHostToDevice        =   1,
    cudaMemcpyDeviceToHost        =   2,
    cudaMemcpyDeviceToDevice      =   3,
    cudaMemcpyDefault             =   4
};
struct cudaPointerAttributes {
    enum cudaMemoryType memoryType;
    int device;
    void *devicePointer;
    void *hostPointer;
};

cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       enum cudaMemcpyKind kind);
cudaError_t cudaMemGetInfo(size_t *free, size_t *total);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaDriverGetVersion(int *device);
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attributes,
                                     const void *ptr);
""")

_ffi_lib = _ffi.verify("""
#include <cuda_runtime_api.h>
#include <driver_types.h>
""", libraries=['cudart'])

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

# Generic CUDA runtime error:
class cudaError(Exception):
    """CUDA error."""
    pass

# Use cudaError* definitions to dynamically create corresponding exception
# classes and populate dictionary used to raise appropriate exception in
# response to the corresponding runtime error code:
cudaExceptions = {-1: cudaError}
for k, v in _ffi_lib.__dict__.iteritems():
    if re.match('cudaError.*', k):
        cudaExceptions[v] = vars()[k] = type(k, (cudaError,), 
                                             {'__doc__': cudaGetErrorString(v)})

def cudaCheckStatus(status):
    """
    Raise CUDA runtime exception.

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
                  
def cudaMalloc(count):
    """
    Allocate device memory.

    Allocate memory on the device associated with the current active
    context.

    Parameters
    ----------
    count : int
        Number of bytes of memory to allocate

    Returns
    -------
    ptr : int
        Pointer to allocated device memory.
    """

    ptr = _ffi.new('void **')
    status = _ffi_lib.cudaMalloc(ptr, count)
    cudaCheckStatus(status)
    return struct.Struct('L').unpack(_ffi.buffer(ptr))[0]

def cudaFree(ptr):
    """
    Free device memory.

    Free allocated memory on the device associated with the current active
    context.

    Parameters
    ----------
    ptr : int
        Pointer to allocated device memory.
    """

    status = _ffi_lib.cudaFree(_ffi.cast('void *', ptr))
    cudaCheckStatus(status)


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
    dev = _ffi.new('int[1]')
    status = _ffi_lib.cudaGetDevice(dev)
    cudaCheckStatus(status)
    return dev[0]
