#!/usr/bin/env python

"""
Python interface to CUTENSOR functions.

Note: this module does not explicitly depend on PyCUDA.
"""

from __future__ import absolute_import

import ctypes
import sys

if 'linux' in sys.platform:
    _libcutensor_libname_list = ['libcutensor.so']
elif sys.platform == 'darwin':
    _libcutensor_libname_list = ['libcutensor.dylib']
else:
    raise RuntimeError('unsupported platform')

# Print understandable error message when library cannot be found:
_libcutensor = None
for _libcutensor_libname in _libcutensor_libname_list:
    try:
        _libcutensor = ctypes.cdll.LoadLibrary(_libcutensor_libname)
    except OSError:
        pass
    else:
        break
if _libcutensor == None:
    raise OSError('cutensor library not found')

# Tensor contraction algorithms
CUTENSOR_ALGO_TGETT          = -7 # Transpose (A or B) + GETT
CUTENSOR_ALGO_GETT           = -6 # Choose the GETT algorithm
CUTENSOR_ALGO_LOG_TENSOR_OP  = -5 # Loop-over-GEMM approach using tensor cores
CUTENSOR_ALGO_LOG            = -4 # Loop-over-GEMM approach
CUTENSOR_ALGO_TTGT_TENSOR_OP = -3 # Transpose-Transpose-GEMM-Transpose using tensor cores (requires additional memory)
CUTENSOR_ALGO_TTGT           = -2 # Transpose-Transpose-GEMM-Transpose (requires additional memory)
CUTENSOR_ALGO_DEFAULT        = -1 # Lets the internal heuristic choose

# Supported unary/binary point-wise operations:
CUTENSOR_OP_IDENTITY = 1
CUTENSOR_OP_SQRT     = 2
CUTENSOR_OP_RELU     = 8
CUTENSOR_OP_CONJ     = 9
CUTENSOR_OP_RCP      = 10

CUTENSOR_OP_ADD      = 3
CUTENSOR_OP_MUL      = 5
CUTENSOR_OP_MAX      = 6
CUTENSOR_OP_MIN      = 7

CUTENSOR_OP_UNKNOWN  = 126

# Worksize preference:
CUTENSOR_WORKSPACE_MIN = 1         # at least one algorithm will be available
CUTENSOR_WORKSPACE_RECOMMENDED = 2 # the most suitable algorithm will be available
CUTENSOR_WORKSPACE_MAX = 3         # all algorithms will be available

class cutensorError(Exception):
    """CUTENSOR error"""
    pass

class cutensorNotInitialized(cutensorError):
    """CUTENSOR library not initialized."""
    pass

class cutensorAllocFailed(cutensorError):
    """CUTENSOR resource allocation failed."""
    pass

class cutensorInvalidValue(cutensorError):
    """CUTENSOR invalid value."""
    pass

class cutensorArchMismatch(cutensorError):
    """CUTENSOR architecture mismatch."""
    pass

class cutensorMappingError(cutensorError):
    """CUTENSOR mapping error."""
    pass

class cutensorExecutionFailed(cutensorError):
    """CUTENSOR execution failed."""
    pass

class cutensorInternalError(cutensorError):
    """CUTENSOR internal error."""
    pass

class cutensorNotSupported(cutensorError):
    """CUTENSOR feature not supported."""
    pass

class cutensorLicenseError(cutensorError):
    """CUTENSOR license error."""
    pass

class cutensorCublasError(cutensorError):
    """CUTENSOR CUBLAS error."""
    pass

class cutensorCudaError(cutensorError):
    """CUTENSOR CUDA error."""
    pass

class cutensorInsufficientWorkspace(cutensorError):
    """CUTENSOR insufficient workspace."""
    pass

cutensorExceptions = {
    1: cutensorNotInitialized,
    3: cutensorAllocFailed,
    7: cutensorInvalidValue,
    8: cutensorArchMismatch,
    11: cutensorMappingError,
    13: cutensorExecutionFailed,
    14: cutensorInternalError,
    15: cutensorNotSupported,
    16: cutensorLicenseError,
    17: cutensorCublasError,
    18: cutensorCudaError,
    19: cutensorInsufficientWorkspace
    }

def cutensorCheckStatus(status):
    if status != 0:
        try:
            e = cutensorExceptions[status]
        except KeyError:
            raise cutensorError
        else:
            raise e

_libcutensor.cutensorCreate.argtype = int
_libcutensor.cutensorCreate.restypes = [ctypes.c_void_p]
def cutensorCreate():
    handle = ctypes.c_void_p()
    status = _libcutensor.cutensorCreate(ctypes.byref(handle))
    cutensorCheckStatus(status)
    return handle.value

_libcutensor.cutensorDestroy.restype = int
_libcutensor.cutensorDestroy.argtypes = [ctypes.c_void_p]
def cutensorDestroy(handle):
    status = _libcutensor.cutensorDestroy(handle)
    cutensorCheckStatus(status)

_libcutensor.cutensorCreateTensorDescriptor.restype = int
_libcutensor.cutensorCreateTensorDescriptor.argtypes = [ctypes.c_void_p,
                                                        ctypes.c_uint,
                                                        ctypes.c_void_p,
                                                        ctypes.c_void_p,
                                                        ctypes.c_uint,
                                                        ctypes.c_uint,
                                                        ctypes.c_uint,
                                                        ctypes.c_uint]
def cutensorCreateTensorDescriptor(numModes, extent, stride, dataType,
                                   unaryOp, vectorWidth, vectorModeIndex):
    desc = ctypes.c_void_p()
    status = _libcutensor.cutensorCreateTensorDescriptor(ctypes.byref(desc),
                                                         numModes, extent, stride,
                                                         dataType, unaryOp, vectorWidth,
                                                         vectorModeIndex)
    cutensorCheckStatus(status)
    return desc.value

_libcutensor.cutensorDestroyTensorDescriptor.restype = int
_libcutensor.cutensorDestroyTensorDescriptor.argtypes = [ctypes.c_void_p]
def cutensorDestroyTensorDescriptor(desc):
    status = _libcutensor.cutensorDestroyTensorDescriptor(desc)
    cutensorCheckStatus(status)

_libcutensor.cutensorElementwiseTrinary.restype = int
_libcutensor.cutensorElementwiseTrinary.argtypes = [ctypes.c_void_p,
                                                    ctypes.c_void_p,
                                                    ctypes.c_void_p,
                                                    ctypes.c_void_p,
                                                    ctypes.c_void_p,
                                                    ctypes.c_void_p,
                                                    ctypes.c_void_p,
                                                    ctypes.c_void_p,
                                                    ctypes.c_void_p,
                                                    ctypes.c_void_p,
                                                    ctypes.c_void_p,
                                                    ctypes.c_void_p,
                                                    ctypes.c_void_p,
                                                    ctypes.c_void_p,
                                                    ctypes.c_void_p,
                                                    ctypes.c_void_p,
                                                    ctypes.c_uint,
                                                    ctypes.c_uint,
                                                    ctypes.c_uint,
                                                    ctypes.c_uint]
def cutensorElementwiseTrinary(handle, alpha, A, descA, modeA,
                               beta, B, descB, modeB,
                               gamma, C, descC, modeC,
                               D, descD, modeD, opAB, opAC, typeCompute, stream):
    status = _libcutensor.cutensorElementwiseTrinary(handle,
                                                     alpha, int(A), descA, modeA,                                                     
                                                     beta, int(B), descB, modeB,                                                     
                                                     gamma, int(C), descC, modeC,                                                     
                                                     int(D), descD, modeD,
                                                     opAB, opAC, typeCompute, stream)
    cutensorCheckStatus(status)

_libcutensor.cutensorElementwiseBinary.restype = int
_libcutensor.cutensorElementwiseBinary.argtypes = [ctypes.c_void_p,
                                                   ctypes.c_void_p,
                                                   ctypes.c_void_p,
                                                   ctypes.c_void_p,
                                                   ctypes.c_void_p,
                                                   ctypes.c_void_p,
                                                   ctypes.c_void_p,
                                                   ctypes.c_void_p,
                                                   ctypes.c_void_p,
                                                   ctypes.c_void_p,
                                                   ctypes.c_void_p,
                                                   ctypes.c_void_p,
                                                   ctypes.c_uint,
                                                   ctypes.c_uint,
                                                   ctypes.c_uint]
def cutensorElementwiseBinary(handle, alpha, A, descA, modeA,
                              gamma, C, descC, modeC,
                              D, descD, modeD, opAC, typeCompute, stream):
    status = _libcutensor.cutensorElementwiseBinary(handle,
                                                    alpha,
                                                    int(A), descA, modeA,
                                                    gamma,
                                                    int(C), descC, modeC,
                                                    int(D), descD, modeD,
                                                    opAC, typeCompute, stream)
    cutensorCheckStatus(status)

_libcutensor.cutensorPermutation.restype = int
_libcutensor.cutensorPermutation.argtypes = [ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_uint,
                                             ctypes.c_uint]
def cutensorPermutation(handle, alpha, A, descA, modeA,
                        B, descB, modeB,
                        typeCompute, stream):
    status = _libcutensor.cutensorPermutation(handle,
                                              alpha, int(A), descA, modeA,
                                              beta, int(B), descB, modeB,
                                              typeCompute, stream)
    cutensorCheckStatus(status)
 
_libcutensor.cutensorContraction.restype = int
_libcutensor.cutensorContraction.argtypes = [ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p,
                                             ctypes.c_uint,
                                             ctypes.c_uint,
                                             ctypes.c_uint,
                                             ctypes.c_void_p,
                                             ctypes.c_uint,
                                             ctypes.c_uint]
def cutensorContraction(handle, alpha, A, descA, modeA,
                        B, descB, modeB,
                        beta, C, descC, modeC,
                        D, descD, modeD,
                        opOut, typeCompute, algo,
                        workspace, workspaceSize, stream):                        
    status = _libcutensor.cutensorContraction(handle,
                                              alpha, int(A), descA, modeA,
                                              int(B), descB, modeB,
                                              beta, int(C), descC, modeC,
                                              int(D), descD, modeD,
                                              opOut, typeCompute, algo,
                                              workspace, workspaceSize, stream)
    cutensorCheckStatus(status)

_libcutensor.cutensorContractionGetWorkspace.restype = int
_libcutensor.cutensorContractionGetWorkspace.argtypes = [ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_void_p,
                                                         ctypes.c_uint,
                                                         ctypes.c_uint,
                                                         ctypes.c_uint,
                                                         ctypes.c_uint,
                                                         ctypes.c_void_p]
def cutensorContractionGetWorkspace(handle,
                                    A, descA, modeA,
                                    B, descB, modeB,
                                    C, descC, modeC,
                                    D, descD, modeD,
                                    opOut, typeCompute, algo,
                                    pref, workspaceSize):                        
    status = _libcutensor.cutensorContractionGetWorkspace(handle,
                                                          int(A), descA, modeA,
                                                          int(B), descB, modeB,
                                                          int(C), descC, modeC,
                                                          int(D), descD, modeD,
                                                          opOut, typeCompute, algo,
                                                          pref, workspaceSize)
    cutensorCheckStatus(status)

_libcutensor.cutensorContractionMaxAlgos.restype = int
_libcutensor.cutensorContractionMaxAlgos.argtypes = [ctypes.c_void_p]
def cutensorContractionMaxAlgos():
    maxNumAlgos = ctypes.c_int()
    status = _libcutensor.cutensorContractionMaxAlgos(ctypes.byref(maxNumAlgos))
    cutensorCheckStatus(status)
    return maxNumAlgos.value
