#!/usr/bin/env/python

"""
Python interface to multi-GPU CULA toolkit functions.
"""

import ctypes
import sys

import cuda
from cula import culaCheckStatus

if 'linux' in sys.platform:
    _libpcula_libname_list = ['libcula_scalapack.so']
elif sys.platform == 'darwin':
    _libpcula_libname_list = ['libcula_scalapack.dylib']
else:
    raise RuntimeError('unsupported platform')

_load_err = ''
for _lib in  _libpcula_libname_list:
    try:
        _libpcula = ctypes.cdll.LoadLibrary(_lib)
    except OSError:
        _load_err += ('' if _load_err == '' else ', ') + _lib
    else:
        _load_err = ''
        break
if _load_err:
    raise OSError('%s not found' % _load_err)

class pculaConfig(ctypes.Structure):
    _fields_ = [
        ('ncuda', ctypes.c_int),
        ('cudaDeviceList', ctypes.c_void_p),
        ('maxCudaMemoryUsage', ctypes.c_void_p),
        ('preserveTuningResult', ctypes.c_int),
        ('dotFileName', ctypes.c_char_p),
        ('timelineFileName', ctypes.c_char_p)]

_libpcula.pculaConfigInit.restype = int
_libpcula.pculaConfigInit.argtypes = [ctypes.c_void_p]
def pculaConfigInit(config):
    """
    Initialize pCULA configuration structure to sensible defaults.
    """

    status = _libpcula.pculaConfigInit(ctypes.byref(config))
    culaCheckStatus(status)

# SGEMM, DGEMM, CGEMM, ZGEMM
_libpcula.pculaSgemm.restype = int
_libpcula.pculaSgemm.argtypes = [ctypes.c_void_p,
                                 ctypes.c_char,
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
def pculaSgemm(config, transa, transb, m, n, k, alpha, A, lda, B, ldb,
               beta, C, ldc):
    """
    Matrix-matrix product for general matrix.

    """

    status = _libpcula.pculaSgemm(ctypes.byref(config), transa, transb, m, n, k, alpha, 
                                  int(A), lda, int(B), ldb, beta, int(C), ldc)
    culaCheckStatus(status)

_libpcula.pculaDgemm.restype = int
_libpcula.pculaDgemm.argtypes = [ctypes.c_void_p,
                                 ctypes.c_char,
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
def pculaDgemm(config, transa, transb, m, n, k, alpha, A, lda, B, ldb,
               beta, C, ldc):
    """
    Matrix-matrix product for general matrix.

    """

    status = _libpcula.pculaDgemm(ctypes.byref(config), transa, transb, m, n, k, alpha, 
                                  int(A), lda, int(B), ldb, beta, int(C), ldc)
    culaCheckStatus(status)

_libpcula.pculaCgemm.restype = int
_libpcula.pculaCgemm.argtypes = [ctypes.c_void_p,
                                 ctypes.c_char,
                                 ctypes.c_char,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 cuda.cuFloatComplex,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 cuda.cuFloatComplex,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def pculaCgemm(config, transa, transb, m, n, k, alpha, A, lda, B, ldb,
               beta, C, ldc):
    """
    Matrix-matrix product for general matrix.

    """

    status = _libpcula.pculaCgemm(ctypes.byref(config), transa, transb, m, n, k, alpha, 
                                  int(A), lda, int(B), ldb, beta, int(C), ldc)
    culaCheckStatus(status)

_libpcula.pculaZgemm.restype = int
_libpcula.pculaZgemm.argtypes = [ctypes.c_void_p,
                                 ctypes.c_char,
                                 ctypes.c_char,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 cuda.cuDoubleComplex,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 cuda.cuDoubleComplex,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def pculaZgemm(config, transa, transb, m, n, k, alpha, A, lda, B, ldb,
               beta, C, ldc):
    """
    Matrix-matrix product for general matrix.

    """

    status = _libpcula.pculaZgemm(ctypes.byref(config), transa, transb, m, n, k, alpha, 
                                  int(A), lda, int(B), ldb, beta, int(C), ldc)
    culaCheckStatus(status)

# STRSM, DTRSM, CTRSM, ZTRSM
_libpcula.pculaStrsm.restype = int
_libpcula.pculaStrsm.argtypes = [ctypes.c_void_p,
                                 ctypes.c_char,
                                 ctypes.c_char,
                                 ctypes.c_char,
                                 ctypes.c_char,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_float,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def pculaStrsm(config, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb):
    """
    Triangular system solve.

    """

    status = _libpcula.pculaStrsm(ctypes.byref(config), side, uplo, transa,
                                  diag, m, n, alpha, int(a), lda, int(b), ldb)                                  
    culaCheckStatus(status)

_libpcula.pculaDtrsm.restype = int
_libpcula.pculaDtrsm.argtypes = [ctypes.c_void_p,
                                 ctypes.c_char,
                                 ctypes.c_char,
                                 ctypes.c_char,
                                 ctypes.c_char,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_double,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def pculaDtrsm(config, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb):
    """
    Triangular system solve.

    """

    status = _libpcula.pculaDtrsm(ctypes.byref(config), side, uplo, transa,
                                  diag, m, n, alpha, int(a), lda, int(b), ldb)                                  
    culaCheckStatus(status)

_libpcula.pculaCtrsm.restype = int
_libpcula.pculaCtrsm.argtypes = [ctypes.c_void_p,
                                 ctypes.c_char,
                                 ctypes.c_char,
                                 ctypes.c_char,
                                 ctypes.c_char,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 cuda.cuFloatComplex,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def pculaCtrsm(config, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb):
    """
    Triangular system solve.

    """

    status = _libpcula.pculaCtrsm(ctypes.byref(config), side, uplo, transa,
                                  diag, m, n, alpha, int(a), lda, int(b), ldb)                                  
    culaCheckStatus(status)

_libpcula.pculaZtrsm.restype = int
_libpcula.pculaZtrsm.argtypes = [ctypes.c_void_p,
                                 ctypes.c_char,
                                 ctypes.c_char,
                                 ctypes.c_char,
                                 ctypes.c_char,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 cuda.cuDoubleComplex,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def pculaZtrsm(config, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb):
    """
    Triangular system solve.

    """

    status = _libpcula.pculaZtrsm(ctypes.byref(config), side, uplo, transa,
                                  diag, m, n, alpha, int(a), lda, int(b), ldb)                                  
    culaCheckStatus(status)

# SGESV, DGESV, CGESV, ZGESV
_libpcula.pculaSgesv.restype = int
_libpcula.pculaSgesv.argtypes = [ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def pculaSgesv(config, n, nrhs, a, lda, ipiv, b, ldb):
    """
    General system solve using LU decomposition.

    """

    status = _libpcula.pculaSgesv(ctypes.byref(config), n, nrhs, int(a), lda,
                                  int(ipiv), int(b), ldb)
    culaCheckStatus(status)

_libpcula.pculaDgesv.restype = int
_libpcula.pculaDgesv.argtypes = [ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def pculaDgesv(config, n, nrhs, a, lda, ipiv, b, ldb):
    """
    General system solve using LU decomposition.

    """

    status = _libpcula.pculaDgesv(ctypes.byref(config), n, nrhs, int(a), lda,
                                  int(ipiv), int(b), ldb)
    culaCheckStatus(status)

_libpcula.pculaCgesv.restype = int
_libpcula.pculaCgesv.argtypes = [ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def pculaCgesv(config, n, nrhs, a, lda, ipiv, b, ldb):
    """
    General system solve using LU decomposition.

    """

    status = _libpcula.pculaCgesv(ctypes.byref(config), n, nrhs, int(a), lda,
                                  int(ipiv), int(b), ldb)
    culaCheckStatus(status)

_libpcula.pculaZgesv.restype = int
_libpcula.pculaZgesv.argtypes = [ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def pculaZgesv(config, n, nrhs, a, lda, ipiv, b, ldb):
    """
    General system solve using LU decomposition.

    """

    status = _libpcula.pculaZgesv(ctypes.byref(config), n, nrhs, int(a), lda,
                                  int(ipiv), int(b), ldb)
    culaCheckStatus(status)

# SGETRF, DGETRF, CGETRF, ZGETRF
_libpcula.pculaSgetrf.restype = int
_libpcula.pculaSgetrf.argtypes = [ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p]
def pculaSgetrf(config, m, n, a, lda, ipiv):
    """
    LU decomposition.

    """

    status = _libpcula.pculaSgetrf(ctypes.byref(config), m, n, int(a), lda,
                                  int(ipiv))
    culaCheckStatus(status)

_libpcula.pculaDgetrf.restype = int
_libpcula.pculaDgetrf.argtypes = [ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p]
def pculaDgetrf(config, m, n, a, lda, ipiv):
    """
    LU decomposition.

    """

    status = _libpcula.pculaDgetrf(ctypes.byref(config), m, n, int(a), lda,
                                  int(ipiv))
    culaCheckStatus(status)

_libpcula.pculaCgetrf.restype = int
_libpcula.pculaCgetrf.argtypes = [ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p]
def pculaCgetrf(config, m, n, a, lda, ipiv):
    """
    LU decomposition.

    """

    status = _libpcula.pculaCgetrf(ctypes.byref(config), m, n, int(a), lda,
                                  int(ipiv))
    culaCheckStatus(status)

_libpcula.pculaZgetrf.restype = int
_libpcula.pculaZgetrf.argtypes = [ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p]
def pculaZgetrf(config, m, n, a, lda, ipiv):
    """
    LU decomposition.

    """

    status = _libpcula.pculaZgetrf(ctypes.byref(config), m, n, int(a), lda,
                                  int(ipiv))
    culaCheckStatus(status)

# SGETRS, DGETRS, CGETRS, ZGETRS
_libpcula.pculaSgetrs.restype = int
_libpcula.pculaSgetrs.argtypes = [ctypes.c_void_p,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def pculaSgetrs(config, trans, n, nrhs, a, lda, ipiv, b, ldb):
    """
    LU solve.

    """

    status = _libpcula.pculaSgetrs(ctypes.byref(config), trans, n, nrhs, int(a), lda,
                                  int(ipiv), int(b), ldb)
    culaCheckStatus(status)

_libpcula.pculaDgetrs.restype = int
_libpcula.pculaDgetrs.argtypes = [ctypes.c_void_p,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def pculaDgetrs(config, trans, n, nrhs, a, lda, ipiv, b, ldb):
    """
    LU solve.

    """

    status = _libpcula.pculaDgetrs(ctypes.byref(config), trans, n, nrhs, int(a), lda,
                                  int(ipiv), int(b), ldb)
    culaCheckStatus(status)

_libpcula.pculaCgetrs.restype = int
_libpcula.pculaCgetrs.argtypes = [ctypes.c_void_p,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def pculaCgetrs(config, trans, n, nrhs, a, lda, ipiv, b, ldb):
    """
    LU solve.

    """

    status = _libpcula.pculaCgetrs(ctypes.byref(config), trans, n, nrhs, int(a), lda,
                                  int(ipiv), int(b), ldb)
    culaCheckStatus(status)

_libpcula.pculaZgetrs.restype = int
_libpcula.pculaZgetrs.argtypes = [ctypes.c_void_p,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def pculaZgetrs(config, trans, n, nrhs, a, lda, ipiv, b, ldb):
    """
    LU solve.

    """

    status = _libpcula.pculaZgetrs(ctypes.byref(config), trans, n, nrhs, int(a), lda,
                                  int(ipiv), int(b), ldb)
    culaCheckStatus(status)

# SPOSV, DPOSV, CPOSV, ZPOSV
_libpcula.pculaSposv.restype = int
_libpcula.pculaSposv.argtypes = [ctypes.c_void_p,
                                 ctypes.c_char,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def pculaSposv(config, uplo, n, nrhs, a, lda, b, ldb):
    """
    QR factorization.

    """

    status = _libpcula.pculaSposv(ctypes.byref(config), uplo, n, nrhs, int(a), lda,
                                   int(b), ldb)
    culaCheckStatus(status)

_libpcula.pculaDposv.restype = int
_libpcula.pculaDposv.argtypes = [ctypes.c_void_p,
                                 ctypes.c_char,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def pculaDposv(config, uplo, n, nrhs, a, lda, b, ldb):
    """
    QR factorization.

    """

    status = _libpcula.pculaDposv(ctypes.byref(config), uplo, n, nrhs, int(a), lda,
                                   int(b), ldb)
    culaCheckStatus(status)

_libpcula.pculaCposv.restype = int
_libpcula.pculaCposv.argtypes = [ctypes.c_void_p,
                                 ctypes.c_char,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def pculaCposv(config, uplo, n, nrhs, a, lda, b, ldb):
    """
    QR factorization.

    """

    status = _libpcula.pculaCposv(ctypes.byref(config), uplo, n, nrhs, int(a), lda,
                                   int(b), ldb)
    culaCheckStatus(status)

_libpcula.pculaZposv.restype = int
_libpcula.pculaZposv.argtypes = [ctypes.c_void_p,
                                 ctypes.c_char,
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def pculaZposv(config, uplo, n, nrhs, a, lda, b, ldb):
    """
    QR factorization.

    """

    status = _libpcula.pculaZposv(ctypes.byref(config), uplo, n, nrhs, int(a), lda,
                                   int(b), ldb)
    culaCheckStatus(status)

# SPOTRF, DPOTRF, CPOTRF, ZPOTRF
_libpcula.pculaSpotrf.restype = int
_libpcula.pculaSpotrf.argtypes = [ctypes.c_void_p,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def pculaSpotrf(config, uplo, n, a, lda):
    """
    Cholesky decomposition.

    """

    status = _libpcula.pculaSpotrf(ctypes.byref(config), uplo, n, int(a), lda)
    culaCheckStatus(status)

_libpcula.pculaDpotrf.restype = int
_libpcula.pculaDpotrf.argtypes = [ctypes.c_void_p,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def pculaDpotrf(config, uplo, n, a, lda):
    """
    Cholesky decomposition.

    """

    status = _libpcula.pculaDpotrf(ctypes.byref(config), uplo, n, int(a), lda)
    culaCheckStatus(status)

_libpcula.pculaCpotrf.restype = int
_libpcula.pculaCpotrf.argtypes = [ctypes.c_void_p,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def pculaCpotrf(config, uplo, n, a, lda):
    """
    Cholesky decomposition.

    """

    status = _libpcula.pculaCpotrf(ctypes.byref(config), uplo, n, int(a), lda)
    culaCheckStatus(status)

_libpcula.pculaZpotrf.restype = int
_libpcula.pculaZpotrf.argtypes = [ctypes.c_void_p,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def pculaZpotrf(config, uplo, n, a, lda):
    """
    Cholesky decomposition.

    """

    status = _libpcula.pculaZpotrf(ctypes.byref(config), uplo, n, int(a), lda)
    culaCheckStatus(status)

# SPOTRS, DPOTRS, CPOTRS, ZPOTRS
_libpcula.pculaSpotrs.restype = int
_libpcula.pculaSpotrs.argtypes = [ctypes.c_void_p,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def pculaSpotrs(config, uplo, n, nrhs, a, lda, b, ldb):
    """
    Cholesky solve.

    """

    status = _libpcula.pculaSpotrs(ctypes.byref(config), uplo, n, nrhs, int(a),
                                   lda, int(b), ldb)
    culaCheckStatus(status)

_libpcula.pculaDpotrs.restype = int
_libpcula.pculaDpotrs.argtypes = [ctypes.c_void_p,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def pculaDpotrs(config, uplo, n, nrhs, a, lda, b, ldb):
    """
    Cholesky solve.

    """

    status = _libpcula.pculaDpotrs(ctypes.byref(config), uplo, n, nrhs, int(a),
                                   lda, int(b), ldb)
    culaCheckStatus(status)

_libpcula.pculaCpotrs.restype = int
_libpcula.pculaCpotrs.argtypes = [ctypes.c_void_p,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def pculaCpotrs(config, uplo, n, nrhs, a, lda, b, ldb):
    """
    Cholesky solve.

    """

    status = _libpcula.pculaCpotrs(ctypes.byref(config), uplo, n, nrhs, int(a),
                                   lda, int(b), ldb)
    culaCheckStatus(status)

_libpcula.pculaZpotrs.restype = int
_libpcula.pculaZpotrs.argtypes = [ctypes.c_void_p,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def pculaZpotrs(config, uplo, n, nrhs, a, lda, b, ldb):
    """
    Cholesky solve.

    """

    status = _libpcula.pculaZpotrs(ctypes.byref(config), uplo, n, nrhs, int(a),
                                   lda, int(b), ldb)
    culaCheckStatus(status)
