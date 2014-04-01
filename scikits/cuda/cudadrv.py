#!/usr/bin/env python

"""
Python interface to CUDA driver functions.
"""

import re

import cffi
_ffi = cffi.FFI()

_ffi.cdef("""
typedef enum cudaError_enum {
    CUDA_SUCCESS                              = 0,
    CUDA_ERROR_INVALID_VALUE                  = 1,
    CUDA_ERROR_OUT_OF_MEMORY                  = 2,
    CUDA_ERROR_NOT_INITIALIZED                = 3,
    CUDA_ERROR_DEINITIALIZED                  = 4,
    CUDA_ERROR_PROFILER_DISABLED              = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED       = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED       = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED       = 8,
    CUDA_ERROR_NO_DEVICE                      = 100,
    CUDA_ERROR_INVALID_DEVICE                 = 101,
    CUDA_ERROR_INVALID_IMAGE                  = 200,
    CUDA_ERROR_INVALID_CONTEXT                = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT        = 202,
    CUDA_ERROR_MAP_FAILED                     = 205,
    CUDA_ERROR_UNMAP_FAILED                   = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED                = 207,
    CUDA_ERROR_ALREADY_MAPPED                 = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU              = 209,
    CUDA_ERROR_ALREADY_ACQUIRED               = 210,
    CUDA_ERROR_NOT_MAPPED                     = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY            = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER          = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE              = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT              = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE         = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED        = 217,
    CUDA_ERROR_INVALID_SOURCE                 = 300,
    CUDA_ERROR_FILE_NOT_FOUND                 = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      = 303,
    CUDA_ERROR_OPERATING_SYSTEM               = 304,
    CUDA_ERROR_INVALID_HANDLE                 = 400,
    CUDA_ERROR_NOT_FOUND                      = 500,
    CUDA_ERROR_NOT_READY                      = 600,
    CUDA_ERROR_LAUNCH_FAILED                  = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT                 = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED           = 709,
    CUDA_ERROR_ASSERT                         = 710,
    CUDA_ERROR_TOO_MANY_PEERS                 = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     = 713,
    CUDA_ERROR_NOT_PERMITTED                  = 800,
    CUDA_ERROR_NOT_SUPPORTED                  = 801,
    CUDA_ERROR_UNKNOWN                        = 999
} CUresult;

typedef enum CUpointer_attribute_enum {
    CU_POINTER_ATTRIBUTE_CONTEXT = 1,
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
    CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
    CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5
} CUpointer_attribute;

typedef enum CUmemorytype_enum {
    CU_MEMORYTYPE_HOST    = 0x01,
    CU_MEMORYTYPE_DEVICE  = 0x02,
    CU_MEMORYTYPE_ARRAY   = 0x03,
    CU_MEMORYTYPE_UNIFIED = 0x04
} CUmemorytype;

typedef struct CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st {
    unsigned long long p2pToken;
    unsigned int vaSpaceToken;
} CUDA_POINTER_ATTRIBUTE_P2P_TOKENS;

typedef unsigned long long CUdeviceptr;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;

CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute,
                               CUdeviceptr ptr);
""")

_ffi_lib = _ffi.verify("""
#include <cuda.h>
""", libraries=['cuda', 'cudart'], library_dirs=['/usr/local/cuda/lib64/'], include_dirs=['/usr/local/cuda/include/'])

# Generic CUDA driver error:
class CUDA_ERROR(Exception):
    """CUDA error."""
    pass

# Use CUDA_ERROR* definitions to dynamically create corresponding exception
# classes and populate dictionary used to raise appropriate exception in
# response to the corresponding runtime error code:
CUDA_EXCEPTIONS = {-1: CUDA_ERROR}
for k, v in _ffi_lib.__dict__.iteritems():
    if re.match('CUDA_ERROR.*', k):        
        CUDA_EXCEPTIONS[v] = vars()[k] = type(k, (CUDA_ERROR,), {})

# Import various enum values into module namespace:
for k, v in _ffi_lib.__dict__.iteritems():
    if re.match('(?:CU_POINTER_ATTRIBUTE|CU_MEMORYTYPE).*', k):
        vars()[k] = v

def cuCheckStatus(status):
    """
    Raise CUDA driver exception.

    Raise an exception corresponding to the specified CUDA driver
    error code.

    Parameters
    ----------
    status : int
        CUDA driver error code.

    See Also
    --------
    CUDA_EXCEPTIONS
    """

    if status != 0:
        try:
            raise CUDA_EXCEPTIONS[status]
        except KeyError:
            raise CUDA_ERROR
        
def cuPointerGetAttribute(attribute, ptr):
    """
    Retrieve attribute of specified pointer.

    Parameters
    ----------
    attribute : int
        Attribute to fetch.
    ptr : int
        GPU pointer.
    
    Returns
    -------
    value : various
        Attribute value.
    """

    if attribute == CU_POINTER_ATTRIBUTE_CONTEXT:
        data = _ffi.new('CUcontext *')
    elif attribute == CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
        data = _ffi.new('CUmemorytype *')
    elif attribute == CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
        data = _ffi.new('CUdeviceptr *')
    elif attribute == CU_POINTER_ATTRIBUTE_HOST_POINTER:
        data = _ffi.new('void **')
    elif attribute == CU_POINTER_ATTRIBUTE_P2P_TOKENS:
        data = _ffi.new('CUDA_POINTER_ATTRIBUTE_P2P_TOKENS')
    else:
        raise CUDA_ERROR
    status = _ffi_lib.cuPointerGetAttribute(data, attribute, ptr)
    cuCheckStatus(status)
    return data
