#!/usr/bin/env python

"""
Python interface to CUBLAS-XT functions.

Note: this module does not explicitly depend on PyCUDA.
"""

import ctypes

from cublas import cublasCheckStatus, _libcublas, _CUBLAS_OP
from . import cuda

CUBLASXT_FLOAT = 0
CUBLASXT_DOUBLE = 1
CUBLASXT_COMPLEX = 2
CUBLASXT_DOUBLECOMPLEX = 3

CUBLASXT_GEMM = 0
CUBLASXT_SYRK = 1
CUBLASXT_HERK = 2
CUBLASXT_SYMM = 3
CUBLASXT_HEMM = 4
CUBLASXT_TRSM = 5
CUBLASXT_SYR2K = 6
CUBLASXT_HER2K = 7
CUBLASXT_SPMM = 8
CUBLASXT_SYRKX = 9,
CUBLASXT_HERKX = 10
CUBLASXT_TRMM = 11
CUBLASXT_ROUTINE_MAX = 12

_libcublas.cublasXtCreate.restype = int
_libcublas.cublasXtCreate.argtypes = [ctypes.c_void_p]
def cublasXtCreate():
    handle = ctypes.c_void_p()
    status = _libcublas.cublasXtCreate(ctypes.byref(handle))
    cublasCheckStatus(status)
    return handle.value

_libcublas.cublasXtDestroy.restype = int
_libcublas.cublasXtDestroy.argtypes = [ctypes.c_int]
def cublasXtDestroy(handle):
    status = _libcublas.cublasXtDestroy(handle)
    cublasCheckStatus(status)

_libcublas.cublasXtGetNumBoards.restype = int
_libcublas.cublasXtGetNumBoards.argtypes = [ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p]
def cublasXtGetNumBoards(handle, deviceId):
    nbBoards = ctypes.c_int()
    status = _libcublas.cublasXtGetNumBoards(handle, deviceId, ctypes.byref(nbBoards))
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
_libcublas.cublasXtDeviceSelect.argtypes = [ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_void_p]
def cublasXtDeviceSelect(handle, nbDevices, deviceId):
    status = _libcublas.cublasXtDeviceSelect(handle, nbDevices, deviceId)
    cublasCheckStatus(status)

_libcublas.cublasXtSetBlockDim.restype = int
_libcublas.cublasXtSetBlockDim.argtypes = [ctypes.c_int,
                                           ctypes.c_int]
def cublasXtSetBlockDim(handle, blockDim):
    status = _libcublas.cublasXtSetBlockDim(handle, blockDim)
    cublasCheckStatus(status)

_libcublas.cublasXtGetBlockDim.restype = int
_libcublas.cublasXtGetBlockDim.argtypes = [ctypes.c_int,
                                           ctypes.c_int]
def cublasXtGetBlockDim(handle):
    blockDim = ctypes.c_void_p()
    status = _libcublas.cublasXtSetBlockDim(handle, ctypes.byref(blockDim))
    cublasCheckStatus(status)
    return blockDim.value

_libcublas.cublasXtSetCpuRoutine.restype = int
_libcublas.cublasXtSetCpuRoutine.argtypes = [ctypes.c_int,
                                             ctypes.c_int,
                                             ctypes.c_int,
                                             ctypes.c_void_p]
def cublasXtSetCpuRoutine(handle, blasOp, type, blasFunctor):
    status = _libcublas.cublasXtSetCpuRoutine(handle, blasOp, type, blasFunctor)
    cublasCheckStatus(status)

_libcublas.cublasXtSetCpuRatio.restype = int
_libcublas.cublasXtSetCpuRatio.argtypes = [ctypes.c_int,
                                           ctypes.c_int,
                                           ctypes.c_int,
                                           ctypes.c_float]
def cublasXtSetCpuRatio(handle, blasOp, type, ratio):
    status = _libcublas.cublasXtSetCpuRatio(handle, blasOp, type, ratio)
    cublasCheckStatus(status)

_libcublas.cublasXtSgemm.restype = int
_libcublas.cublasXtSgemm.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_int]
def cublasXtSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                  beta, C, ldc):
    status = _libcublas.cublasXtSgemm(handle, transa, transb, m, n, k,
                                      ctypes.byref(ctypes.c_float(alpha)),
                                      int(A), lda, int(B), ldb,
                                      ctypes.byref(ctypes.c_float(beta)),
                                      int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasXtDgemm.restype = int
_libcublas.cublasXtDgemm.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_int]
def cublasXtDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                  beta, C, ldc):
    status = _libcublas.cublasXtDgemm(handle, transa, transb, m, n, k,
                                      ctypes.byref(ctypes.c_double(alpha)),
                                      int(A), lda, int(B), ldb,
                                      ctypes.byref(ctypes.c_double(beta)),
                                      int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasXtCgemm.restype = int
_libcublas.cublasXtCgemm.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_int]
def cublasXtCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                  beta, C, ldc):
    status = _libcublas.cublasXtCgemm(handle, transa, transb, m, n, k,
                                      ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                       alpha.imag)),
                                      int(A), lda, int(B), ldb,
                                      ctypes.byref(cuda.cuFloatComplex(beta.real,
                                                                       beta.imag)),
                                      int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasXtZgemm.restype = int
_libcublas.cublasXtZgemm.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_int]
def cublasXtZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                  beta, C, ldc):
    status = _libcublas.cublasXtZgemm(handle, transa, transb, m, n, k,
                                      ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                               alpha.imag)),
                                      int(A), lda, int(B), ldb,
                                      ctypes.byref(cuda.cuDoubleComplex(beta.real,
                                                               beta.imag)),
                                      int(C), ldc)
    cublasCheckStatus(status)
