#!/usr/bin/env python

"""
Python interface to CUBLAS-XT functions.

Note: this module does not explicitly depend on PyCUDA.
"""

import ctypes

from .cublas import cublasCheckStatus, _libcublas, _CUBLAS_OP, _types
from . import cuda

_CUBLAS_XT_OP_TYPE = {
    0: 0, # CUBLASXT_FLOAT
    1: 1, # CUBLASXT_DOUBLE
    2: 2, # CUBLASXT_COMPLEX
    3: 3, # CUBLASXT_DOUBLECOMPLEX
}

_CUBLAS_XT_BLAS_OP = {
    0 : 0,  # CUBLASXT_GEMM
    1: 1,   # CUBLASXT_SYRK
    2: 2,   # CUBLASXT_HERK
    3: 3,   # CUBLASXT_SYMM
    4: 4,   # CUBLASXT_HEMM
    5: 5,   # CUBLASXT_TRSM
    6: 6,   # CUBLASXT_SYR2K
    7: 7,   # CUBLASXT_HER2K
    8: 8,   # CUBLASXT_SPMM
    9: 9,   # CUBLASXT_SYRKX
    10: 10, # CUBLASXT_HERKX
    11: 11, # CUBLASXT_TRMM
    12: 12, # CUBLASXT_ROUTINE_MAX
}

_CUBLAS_XT_PINNING_MEM_MODE = {
    0: 0, # CUBLASXT_PINNING_DISABLED
    1: 1, # CUBLASXT_PINNING_ENABLED
}

_libcublas.cublasXtCreate.restype = int
_libcublas.cublasXtCreate.argtypes = [_types.handle]
def cublasXtCreate():
    handle = _types.handle()
    status = _libcublas.cublasXtCreate(ctypes.byref(handle))
    cublasCheckStatus(status)
    return handle.value

_libcublas.cublasXtDestroy.restype = int
_libcublas.cublasXtDestroy.argtypes = [_types.handle]
def cublasXtDestroy(handle):
    status = _libcublas.cublasXtDestroy(handle)
    cublasCheckStatus(status)

_libcublas.cublasXtGetNumBoards.restype = int
_libcublas.cublasXtGetNumBoards.argtypes = [ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p]
def cublasXtGetNumBoards(nbDevices, deviceId):
    nbBoards = ctypes.c_int()
    status = _libcublas.cublasXtGetNumBoards(nbDevices, deviceId, ctypes.byref(nbBoards))
    cublasCheckStatus(status)
    return nbBoards.value

_libcublas.cublasXtMaxBoards.restype = int
_libcublas.cublasXtMaxBoards.argtypes = [ctypes.c_void_p]
def cublasXtMaxBoards():
    nbGpuBoards = ctypes.c_int()
    status = _libcublas.cublasXtMaxBoards(ctypes.byref(nbGpuBoards))
    cublasCheckStatus(status)
    return nbGpuBoards.value

_libcublas.cublasXtDeviceSelect.restype = int
_libcublas.cublasXtDeviceSelect.argtypes = [_types.handle,
                                            ctypes.c_int,
                                            ctypes.c_void_p]
def cublasXtDeviceSelect(handle, nbDevices, deviceId):
    status = _libcublas.cublasXtDeviceSelect(handle, nbDevices, deviceId.ctypes.data)
    cublasCheckStatus(status)

_libcublas.cublasXtSetBlockDim.restype = int
_libcublas.cublasXtSetBlockDim.argtypes = [_types.handle,
                                           ctypes.c_void_p]
def cublasXtSetBlockDim(handle, blockDim):
    status = _libcublas.cublasXtSetBlockDim(handle, blockDim)
    cublasCheckStatus(status)

_libcublas.cublasXtGetBlockDim.restype = int
_libcublas.cublasXtGetBlockDim.argtypes = [_types.handle,
                                           ctypes.c_void_p]
def cublasXtGetBlockDim(handle):
    blockDim = ctypes.c_void_p()
    status = _libcublas.cublasXtSetBlockDim(handle, ctypes.byref(blockDim))
    cublasCheckStatus(status)
    return blockDim.value

_libcublas.cublasXtSetCpuRoutine.restype = int
_libcublas.cublasXtSetCpuRoutine.argtypes = [_types.handle,
                                             ctypes.c_int,
                                             ctypes.c_int,
                                             ctypes.c_void_p]
def cublasXtSetCpuRoutine(handle, blasOp, type, blasFunctor):
    status = _libcublas.cublasXtSetCpuRoutine(handle, blasOp, type, blasFunctor)
    cublasCheckStatus(status)

_libcublas.cublasXtSetCpuRatio.restype = int
_libcublas.cublasXtSetCpuRatio.argtypes = [_types.handle,
                                           ctypes.c_int,
                                           ctypes.c_int,
                                           ctypes.c_float]
def cublasXtSetCpuRatio(handle, blasOp, type, ratio):
    status = _libcublas.cublasXtSetCpuRatio(handle, blasOp, type, ratio)
    cublasCheckStatus(status)

_libcublas.cublasXtSetPinningMemMode.restype = int
_libcublas.cublasXtSetPinningMemMode.argtypes = [_types.handle,
                                                 ctypes.c_int]
def cublasXtSetPinningMemMode(handle, mode):
    status = _libcublas.cublasXtSetPinningMemMode(handle, mode)
    cublasCheckStatus(status)

_libcublas.cublasXtSgemm.restype = int
_libcublas.cublasXtSgemm.argtypes = [_types.handle,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_size_t,
                                     ctypes.c_size_t,
                                     ctypes.c_size_t,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_size_t,
                                     ctypes.c_void_p,
                                     ctypes.c_size_t,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_size_t]
def cublasXtSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                  beta, C, ldc):
    status = _libcublas.cublasXtSgemm(handle,
                                      _CUBLAS_OP[transa],
                                      _CUBLAS_OP[transb], m, n, k,
                                      ctypes.byref(ctypes.c_float(alpha)),
                                      int(A), lda, int(B), ldb,
                                      ctypes.byref(ctypes.c_float(beta)),
                                      int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasXtDgemm.restype = int
_libcublas.cublasXtDgemm.argtypes = [_types.handle,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_size_t,
                                     ctypes.c_size_t,
                                     ctypes.c_size_t,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_size_t,
                                     ctypes.c_void_p,
                                     ctypes.c_size_t,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_size_t]
def cublasXtDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                  beta, C, ldc):
    status = _libcublas.cublasXtDgemm(handle,
                                      _CUBLAS_OP[transa],
                                      _CUBLAS_OP[transb], m, n, k,
                                      ctypes.byref(ctypes.c_double(alpha)),
                                      int(A), lda, int(B), ldb,
                                      ctypes.byref(ctypes.c_double(beta)),
                                      int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasXtCgemm.restype = int
_libcublas.cublasXtCgemm.argtypes = [_types.handle,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_size_t,
                                     ctypes.c_size_t,
                                     ctypes.c_size_t,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_size_t,
                                     ctypes.c_void_p,
                                     ctypes.c_size_t,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_size_t]
def cublasXtCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                  beta, C, ldc):
    status = _libcublas.cublasXtCgemm(handle,
                                      _CUBLAS_OP[transa],
                                      _CUBLAS_OP[transb], m, n, k,
                                      ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                       alpha.imag)),
                                      int(A), lda, int(B), ldb,
                                      ctypes.byref(cuda.cuFloatComplex(beta.real,
                                                                       beta.imag)),
                                      int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasXtZgemm.restype = int
_libcublas.cublasXtZgemm.argtypes = [_types.handle,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_size_t,
                                     ctypes.c_size_t,
                                     ctypes.c_size_t,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_size_t,
                                     ctypes.c_void_p,
                                     ctypes.c_size_t,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_size_t]
def cublasXtZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                  beta, C, ldc):
    status = _libcublas.cublasXtZgemm(handle,
                                      _CUBLAS_OP[transa],
                                      _CUBLAS_OP[transb], m, n, k,
                                      ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                               alpha.imag)),
                                      int(A), lda, int(B), ldb,
                                      ctypes.byref(cuda.cuDoubleComplex(beta.real,
                                                               beta.imag)),
                                      int(C), ldc)
    cublasCheckStatus(status)
