#!/usr/bin/env python

"""
Python interface to CULA toolkit.
"""

import re
import cffi
_ffi = cffi.FFI()

import sys
import ctypes
import atexit
import numpy as np


#####FIXME: needed for compatibility with linalg.py
import sys
import ctypes
import atexit
import numpy as np
import cuda
if sys.platform == 'linux2':
    _libcula_libname_list = ['libcula_lapack.so', 'libcula_lapack_basic.so', 'libcula.so']
elif sys.platform == 'darwin':
    _libcula_libname_list = ['libcula_lapack.so', 'libcula.dylib']
else:
    raise RuntimeError('unsupported platform')
_load_err = ''
for _lib in _libcula_libname_list:
    try:
        _libcula = ctypes.cdll.LoadLibrary(_lib)
    except OSError:
        _load_err += ('' if _load_err == '' else ', ') + _lib
    else:
        _load_err = ''
        break
if _load_err:
    raise OSError('%s not found' % _load_err)


import cuda

# Check whether the free or standard version of the toolkit is
# installed by trying to access a function that is only available in
# the latter:
_cula_type_str = """

typedef enum
{
    culaNoError,                       // No error
    culaNotInitialized,                // CULA has not been initialized
    culaNoHardware,                    // No hardware is available to run
    culaInsufficientRuntime,           // CUDA runtime or driver is not supported
    culaInsufficientComputeCapability, // Available GPUs do not support the requested operation
    culaInsufficientMemory,            // There is insufficient memory to continue
    culaFeatureNotImplemented,         // The requested feature has not been implemented
    culaArgumentError,                 // An invalid argument was passed to a function
    culaDataError,                     // An operation could not complete because of singular data
    culaBlasError,                     // A blas error was encountered
    culaRuntimeError,                  // A runtime error has occurred
    culaBadStorageFormat,              // An invalid storage format was used fora parameter
    culaInvalidReferenceHandle,        // An invalid reference handle was passedto a function
    culaUnspecifiedError               // An unspecified internal error has occurred
}culaStatus;

typedef int culaInfo;
typedef int culaVersion;

typedef int culaInt;
typedef culaInt culaDeviceInt;

typedef float culaFloat;
typedef culaFloat culaDeviceFloat;

typedef float culaDouble;
typedef culaDouble culaDeviceDouble;

typedef ... culaFloatComplex;
typedef ... culaDoubleComplex;

typedef culaFloatComplex culaDeviceFloatComplex;
typedef culaDoubleComplex culaDeviceDoubleComplex;
"""
_ffi.cdef(_cula_type_str + """
culaStatus culaDeviceMalloc(void **mem, int *pitch, int rows, int cols, int elesize);
""")
try:
    _ffi_lib = _ffi.verify("""
#include <cuComplex.h>
#include <cula.h>
""", libraries=['cula_lapack'])
except cffi.ffiplatform.VerificationError:
    _libcula_toolkit = 'free'
else:
    _libcula_toolkit = 'standard'

# Generic CULA error:
class culaError(Exception):
    """CULA error."""
    pass

# Errors that don't correspond to CULA status codes:
class culaNotFound(culaError):
    """CULA shared library not found"""
    pass

class culaStandardNotFound(culaError):
    """Standard CULA Dense toolkit unavailable"""
    pass

# Import all CULA status definitions directly into module namespace:
culaExceptions = {}
_ffi = FFI()
_ffi_lib = _ffi.verify("""
#include <cuComplex.h>
#include <cula.h>
""", libraries=['cula_lapack'])
                       
for k, v in _ffi_lib.__dict__.iteritems():
    culaExceptions[v] = type(k, (culaError,), {})

# Functions defined in the free version of CULA:
_cula_dense_free_str = """
// cula_status.h functions
culaStatus culaInitialize();
void culaShutdown();

const char *culaGetStatusString(culaStatus e);
//const char *culaGetStatusAsString(culaStatus e);
culaInfo culaGetErrorInfo();
culaStatus culaGetErrorInfoString(culaStatus e, culaInfo i, char* buf, int bufsize);
void culaFreeBuffers();
culaVersion culaGetVersion();
culaVersion culaGetCudaMinimumVersion();
culaVersion culaGetCudaRuntimeVersion();
culaVersion culaGetCudaDriverVersion();
culaVersion culaGetCublasMinimumVersion();
culaVersion culaGetCublasRuntimeVersion();

// cula_device.h functions:
culaStatus culaGetDeviceCount(int* dev);
culaStatus culaSelectDevice(int dev);
culaStatus culaGetExecutingDevice(int* dev);
culaStatus culaGetDeviceInfo(int dev, char* buf, int bufsize);

// cula_device_lapack.h functions:
culaStatus culaDeviceCgels(char trans, int m, int n, int nrhs, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCgeqrf(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCgesv(int n, int nrhs, culaDeviceFloatComplex* a, int lda, culaDeviceInt* ipiv, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCgesvd(char jobu, char jobvt, int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloat* s, culaDeviceFloatComplex* u, int ldu, culaDeviceFloatComplex* vt, int ldvt);
culaStatus culaDeviceCgetrf(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceInt* ipiv);
culaStatus culaDeviceCgglse(int m, int n, int p, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb, culaDeviceFloatComplex* c, culaDeviceFloatComplex* d, culaDeviceFloatComplex* x);
culaStatus culaDeviceSgels(char trans, int m, int n, int nrhs, culaDeviceFloat* a, int lda, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceSgeqrf(int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSgesv(int n, int nrhs, culaDeviceFloat* a, int lda, culaDeviceInt* ipiv, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceSgesvd(char jobu, char jobvt, int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* s, culaDeviceFloat* u, int ldu, culaDeviceFloat* vt, int ldvt);
culaStatus culaDeviceSgetrf(int m, int n, culaDeviceFloat* a, int lda, culaDeviceInt* ipiv);
culaStatus culaDeviceSgglse(int m, int n, int p, culaDeviceFloat* a, int lda, culaDeviceFloat* b, int ldb, culaDeviceFloat* c, culaDeviceFloat* d, culaDeviceFloat* x);
"""

# Functions defined in the full version of CULA dense:
_cula_dense_standard_str = """
// cula_device.h functions:
culaStatus culaGetOptimalPitch(int* pitch, int rows, int cols, int elesize);
culaStatus culaDeviceMalloc(void** mem, int* pitch, int rows, int cols, int elesize);
culaStatus culaDeviceFree(void* mem);

// cula_device_lapack.h functions:
culaStatus culaDeviceCbdsqr(char uplo, int n, int ncvt, int nru, int ncc, culaDeviceFloat* d, culaDeviceFloat* e, culaDeviceFloatComplex* vt, int ldvt, culaDeviceFloatComplex* u, int ldu, culaDeviceFloatComplex* c, int ldc);
culaStatus culaDeviceCgbtrf(int m, int n, int kl, int ku, culaDeviceFloatComplex* a, int lda, culaInt* ipiv);
culaStatus culaDeviceCgeConjugate(int m, int n, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceCgeNancheck(int m, int n, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceCgeTranspose(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCgeTransposeConjugate(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCgeTransposeConjugateInplace(int n, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceCgeTransposeInplace(int n, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceCgebrd(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloat* d, culaDeviceFloat* e, culaDeviceFloatComplex* tauq, culaDeviceFloatComplex* taup);
culaStatus culaDeviceCgeev(char jobvl, char jobvr, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* w, culaDeviceFloatComplex* vl, int ldvl, culaDeviceFloatComplex* vr, int ldvr);
culaStatus culaDeviceCgehrd(int n, int ilo, int ihi, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCgelqf(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCgels(char trans, int m, int n, int nrhs, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCgeqlf(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCgeqrf(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCgeqrfp(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCgeqrs(int m, int n, int nrhs, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCgerqf(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCgesdd(char jobz, int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloat* s, culaDeviceFloatComplex* u, int ldu, culaDeviceFloatComplex* vt, int ldvt);
culaStatus culaDeviceCgesv(int n, int nrhs, culaDeviceFloatComplex* a, int lda, culaDeviceInt* ipiv, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCgesvd(char jobu, char jobvt, int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloat* s, culaDeviceFloatComplex* u, int ldu, culaDeviceFloatComplex* vt, int ldvt);
culaStatus culaDeviceCgetrf(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceInt* ipiv);
culaStatus culaDeviceCgetri(int n, culaDeviceFloatComplex* a, int lda, culaDeviceInt* ipiv);
culaStatus culaDeviceCgetrs(char trans, int n, int nrhs, culaDeviceFloatComplex* a, int lda, culaDeviceInt* ipiv, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCgglse(int m, int n, int p, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb, culaDeviceFloatComplex* c, culaDeviceFloatComplex* d, culaDeviceFloatComplex* x);
culaStatus culaDeviceCggrqf(int m, int p, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* taua, culaDeviceFloatComplex* b, int ldb, culaDeviceFloatComplex* taub);
culaStatus culaDeviceCheev(char jobz, char uplo, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloat* w);
culaStatus culaDeviceCheevx(char jobz, char range, char uplo, int n, culaDeviceFloatComplex* a, int lda, culaFloat vl, culaFloat vu, int il, int iu, culaFloat abstol, culaInt* m, culaDeviceFloat* w, culaDeviceFloatComplex* z, int ldz, culaInt* ifail);
culaStatus culaDeviceChegv(int itype, char jobz, char uplo, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb, culaDeviceFloat* w);
culaStatus culaDeviceCherdb(char jobz, char uplo, int n, int kd, culaDeviceFloatComplex* a, int lda, culaDeviceFloat* d, culaDeviceFloat* e, culaDeviceFloatComplex* tau, culaDeviceFloatComplex* z, int ldz);
culaStatus culaDeviceClacpy(char uplo, int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceClag2z(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceDoubleComplex* sa, int ldsa);
culaStatus culaDeviceClar2v(int n, culaDeviceFloatComplex* x, culaDeviceFloatComplex* y, culaDeviceFloatComplex* z, int incx, culaDeviceFloat* c, culaDeviceFloatComplex* s, int incc);
culaStatus culaDeviceClarfb(char side, char trans, char direct, char storev, int m, int n, int k, culaDeviceFloatComplex* v, int ldv, culaDeviceFloatComplex* t, int ldt, culaDeviceFloatComplex* c, int ldc);
culaStatus culaDeviceClarfg(int n, culaDeviceFloatComplex* alpha, culaDeviceFloatComplex* x, int incx, culaDeviceFloatComplex* tau);
culaStatus culaDeviceClargv(int n, culaDeviceFloatComplex* x, int incx, culaDeviceFloatComplex* y, int incy, culaDeviceFloat* c, int incc);
culaStatus culaDeviceClartv(int n, culaDeviceFloatComplex* x, int incx, culaDeviceFloatComplex* y, int incy, culaDeviceFloat* c, culaDeviceFloatComplex* s, int incc);
culaStatus culaDeviceClascl(char type, int kl, int ku, culaFloat cfrom, culaFloat cto, int m, int n, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceClaset(char uplo, int m, int n, culaFloatComplex alpha, culaFloatComplex beta, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceClasr(char side, char pivot, char direct, int m, int n, culaDeviceFloat* c, culaDeviceFloat* s, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceClat2z(char uplo, int n, culaDeviceFloatComplex* a, int lda, culaDeviceDoubleComplex* sa, int ldsa);
culaStatus culaDeviceCpbtrf(char uplo, int n, int kd, culaDeviceFloatComplex* ab, int ldab);
culaStatus culaDeviceCposv(char uplo, int n, int nrhs, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCpotrf(char uplo, int n, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceCpotri(char uplo, int n, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceCpotrs(char uplo, int n, int nrhs, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCsteqr(char compz, int n, culaDeviceFloat* d, culaDeviceFloat* e, culaDeviceFloatComplex* z, int ldz);
culaStatus culaDeviceCtrConjugate(char uplo, char diag, int m, int n, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceCtrtri(char uplo, char diag, int n, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceCtrtrs(char uplo, char trans, char diag, int n, int nrhs, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCungbr(char vect, int m, int n, int k, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCunghr(int n, int ilo, int ihi, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCunglq(int m, int n, int k, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCungql(int m, int n, int k, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCungqr(int m, int n, int k, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCungrq(int m, int n, int k, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCunmlq(char side, char trans, int m, int n, int k, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau, culaDeviceFloatComplex* c, int ldc);
culaStatus culaDeviceCunmql(char side, char trans, int m, int n, int k, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau, culaDeviceFloatComplex* c, int ldc);
culaStatus culaDeviceCunmqr(char side, char trans, int m, int n, int k, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau, culaDeviceFloatComplex* c, int ldc);
culaStatus culaDeviceCunmrq(char side, char trans, int m, int n, int k, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau, culaDeviceFloatComplex* c, int ldc);
culaStatus culaDeviceDbdsqr(char uplo, int n, int ncvt, int nru, int ncc, culaDeviceDouble* d, culaDeviceDouble* e, culaDeviceDouble* vt, int ldvt, culaDeviceDouble* u, int ldu, culaDeviceDouble* c, int ldc);
culaStatus culaDeviceDgbtrf(int m, int n, int kl, int ku, culaDeviceDouble* a, int lda, culaInt* ipiv);
culaStatus culaDeviceDgeNancheck(int m, int n, culaDeviceDouble* a, int lda);
culaStatus culaDeviceDgeTranspose(int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* b, int ldb);
culaStatus culaDeviceDgeTransposeInplace(int n, culaDeviceDouble* a, int lda);
culaStatus culaDeviceDgebrd(int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* d, culaDeviceDouble* e, culaDeviceDouble* tauq, culaDeviceDouble* taup);
culaStatus culaDeviceDgeev(char jobvl, char jobvr, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* wr, culaDeviceDouble* wi, culaDeviceDouble* vl, int ldvl, culaDeviceDouble* vr, int ldvr);
culaStatus culaDeviceDgehrd(int n, int ilo, int ihi, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDgelqf(int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDgels(char trans, int m, int n, int nrhs, culaDeviceDouble* a, int lda, culaDeviceDouble* b, int ldb);
culaStatus culaDeviceDgeqlf(int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDgeqrf(int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDgeqrfp(int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDgeqrs(int m, int n, int nrhs, culaDeviceDouble* a, int lda, culaDeviceDouble* tau, culaDeviceDouble* b, int ldb);
culaStatus culaDeviceDgerqf(int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDgesdd(char jobz, int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* s, culaDeviceDouble* u, int ldu, culaDeviceDouble* vt, int ldvt);
culaStatus culaDeviceDgesv(int n, int nrhs, culaDeviceDouble* a, int lda, culaDeviceInt* ipiv, culaDeviceDouble* b, int ldb);
culaStatus culaDeviceDgesvd(char jobu, char jobvt, int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* s, culaDeviceDouble* u, int ldu, culaDeviceDouble* vt, int ldvt);
culaStatus culaDeviceDgetrf(int m, int n, culaDeviceDouble* a, int lda, culaDeviceInt* ipiv);
culaStatus culaDeviceDgetri(int n, culaDeviceDouble* a, int lda, culaDeviceInt* ipiv);
culaStatus culaDeviceDgetrs(char trans, int n, int nrhs, culaDeviceDouble* a, int lda, culaDeviceInt* ipiv, culaDeviceDouble* b, int ldb);
culaStatus culaDeviceDgglse(int m, int n, int p, culaDeviceDouble* a, int lda, culaDeviceDouble* b, int ldb, culaDeviceDouble* c, culaDeviceDouble* d, culaDeviceDouble* x);
culaStatus culaDeviceDggrqf(int m, int p, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* taua, culaDeviceDouble* b, int ldb, culaDeviceDouble* taub);
culaStatus culaDeviceDlacpy(char uplo, int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* b, int ldb);
culaStatus culaDeviceDlag2s(int m, int n, culaDeviceDouble* a, int lda, culaDeviceFloat* sa, int ldsa);
culaStatus culaDeviceDlar2v(int n, culaDeviceDouble* x, culaDeviceDouble* y, culaDeviceDouble* z, int incx, culaDeviceDouble* c, culaDeviceDouble* s, int incc);
culaStatus culaDeviceDlarfb(char side, char trans, char direct, char storev, int m, int n, int k, culaDeviceDouble* v, int ldv, culaDeviceDouble* t, int ldt, culaDeviceDouble* c, int ldc);
culaStatus culaDeviceDlarfg(int n, culaDeviceDouble* alpha, culaDeviceDouble* x, int incx, culaDeviceDouble* tau);
culaStatus culaDeviceDlargv(int n, culaDeviceDouble* x, int incx, culaDeviceDouble* y, int incy, culaDeviceDouble* c, int incc);
culaStatus culaDeviceDlartv(int n, culaDeviceDouble* x, int incx, culaDeviceDouble* y, int incy, culaDeviceDouble* c, culaDeviceDouble* s, int incc);
culaStatus culaDeviceDlascl(char type, int kl, int ku, culaDouble cfrom, culaDouble cto, int m, int n, culaDeviceDouble* a, int lda);
culaStatus culaDeviceDlaset(char uplo, int m, int n, culaDouble alpha, culaDouble beta, culaDeviceDouble* a, int lda);
culaStatus culaDeviceDlasr(char side, char pivot, char direct, int m, int n, culaDeviceDouble* c, culaDeviceDouble* s, culaDeviceDouble* a, int lda);
culaStatus culaDeviceDlat2s(char uplo, int n, culaDeviceDouble* a, int lda, culaDeviceFloat* sa, int ldsa);
culaStatus culaDeviceDorgbr(char vect, int m, int n, int k, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDorghr(int n, int ilo, int ihi, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDorglq(int m, int n, int k, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDorgql(int m, int n, int k, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDorgqr(int m, int n, int k, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDorgrq(int m, int n, int k, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDormlq(char side, char trans, int m, int n, int k, culaDeviceDouble* a, int lda, culaDeviceDouble* tau, culaDeviceDouble* c, int ldc);
culaStatus culaDeviceDormql(char side, char trans, int m, int n, int k, culaDeviceDouble* a, int lda, culaDeviceDouble* tau, culaDeviceDouble* c, int ldc);
culaStatus culaDeviceDormqr(char side, char trans, int m, int n, int k, culaDeviceDouble* a, int lda, culaDeviceDouble* tau, culaDeviceDouble* c, int ldc);
culaStatus culaDeviceDormrq(char side, char trans, int m, int n, int k, culaDeviceDouble* a, int lda, culaDeviceDouble* tau, culaDeviceDouble* c, int ldc);
culaStatus culaDeviceDpbtrf(char uplo, int n, int kd, culaDeviceDouble* ab, int ldab);
culaStatus culaDeviceDposv(char uplo, int n, int nrhs, culaDeviceDouble* a, int lda, culaDeviceDouble* b, int ldb);
culaStatus culaDeviceDpotrf(char uplo, int n, culaDeviceDouble* a, int lda);
culaStatus culaDeviceDpotri(char uplo, int n, culaDeviceDouble* a, int lda);
culaStatus culaDeviceDpotrs(char uplo, int n, int nrhs, culaDeviceDouble* a, int lda, culaDeviceDouble* b, int ldb);
culaStatus culaDeviceDsgesv(int n, int nrhs, culaDeviceDouble* a, int lda, culaInt* ipiv, culaDeviceDouble* b, int ldb, culaDeviceDouble* x, int ldx, int* iter);
culaStatus culaDeviceDsposv(char uplo, int n, int nrhs, culaDeviceDouble* a, int lda, culaDeviceDouble* b, int ldb, culaDeviceDouble* x, int ldx, int* iter);
culaStatus culaDeviceDstebz(char range, char order, int n, double vl, double vu, int il, int iu, double abstol, culaDeviceDouble* d, culaDeviceDouble* e, int* m, int* nsplit, culaDeviceDouble* w, culaDeviceInt* iblock, culaDeviceInt* isplit);
culaStatus culaDeviceDsteqr(char compz, int n, culaDeviceDouble* d, culaDeviceDouble* e, culaDeviceDouble* z, int ldz);
culaStatus culaDeviceDsyev(char jobz, char uplo, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* w);
culaStatus culaDeviceDsyevx(char jobz, char range, char uplo, int n, culaDeviceDouble* a, int lda, culaDouble vl, culaDouble vu, int il, int iu, culaDouble abstol, culaInt* m, culaDeviceDouble* w, culaDeviceDouble* z, int ldz, culaInt* ifail);
culaStatus culaDeviceDsygv(int itype, char jobz, char uplo, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* b, int ldb, culaDeviceDouble* w);
culaStatus culaDeviceDsyrdb(char jobz, char uplo, int n, int kd, culaDeviceDouble* a, int lda, culaDeviceDouble* d, culaDeviceDouble* e, culaDeviceDouble* tau, culaDeviceDouble* z, int ldz);
culaStatus culaDeviceDtrtri(char uplo, char diag, int n, culaDeviceDouble* a, int lda);
culaStatus culaDeviceDtrtrs(char uplo, char trans, char diag, int n, int nrhs, culaDeviceDouble* a, int lda, culaDeviceDouble* b, int ldb);
culaStatus culaDeviceSbdsqr(char uplo, int n, int ncvt, int nru, int ncc, culaDeviceFloat* d, culaDeviceFloat* e, culaDeviceFloat* vt, int ldvt, culaDeviceFloat* u, int ldu, culaDeviceFloat* c, int ldc);
culaStatus culaDeviceSgbtrf(int m, int n, int kl, int ku, culaDeviceFloat* a, int lda, culaInt* ipiv);
culaStatus culaDeviceSgeNancheck(int m, int n, culaDeviceFloat* a, int lda);
culaStatus culaDeviceSgeTranspose(int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceSgeTransposeInplace(int n, culaDeviceFloat* a, int lda);
culaStatus culaDeviceSgebrd(int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* d, culaDeviceFloat* e, culaDeviceFloat* tauq, culaDeviceFloat* taup);
culaStatus culaDeviceSgeev(char jobvl, char jobvr, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* wr, culaDeviceFloat* wi, culaDeviceFloat* vl, int ldvl, culaDeviceFloat* vr, int ldvr);
culaStatus culaDeviceSgehrd(int n, int ilo, int ihi, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSgelqf(int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSgels(char trans, int m, int n, int nrhs, culaDeviceFloat* a, int lda, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceSgeqlf(int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSgeqrf(int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSgeqrfp(int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSgeqrs(int m, int n, int nrhs, culaDeviceFloat* a, int lda, culaDeviceFloat* tau, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceSgerqf(int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSgesdd(char jobz, int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* s, culaDeviceFloat* u, int ldu, culaDeviceFloat* vt, int ldvt);
culaStatus culaDeviceSgesv(int n, int nrhs, culaDeviceFloat* a, int lda, culaDeviceInt* ipiv, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceSgesvd(char jobu, char jobvt, int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* s, culaDeviceFloat* u, int ldu, culaDeviceFloat* vt, int ldvt);
culaStatus culaDeviceSgetrf(int m, int n, culaDeviceFloat* a, int lda, culaDeviceInt* ipiv);
culaStatus culaDeviceSgetri(int n, culaDeviceFloat* a, int lda, culaDeviceInt* ipiv);
culaStatus culaDeviceSgetrs(char trans, int n, int nrhs, culaDeviceFloat* a, int lda, culaDeviceInt* ipiv, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceSgglse(int m, int n, int p, culaDeviceFloat* a, int lda, culaDeviceFloat* b, int ldb, culaDeviceFloat* c, culaDeviceFloat* d, culaDeviceFloat* x);
culaStatus culaDeviceSggrqf(int m, int p, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* taua, culaDeviceFloat* b, int ldb, culaDeviceFloat* taub);
culaStatus culaDeviceSlacpy(char uplo, int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceSlag2d(int m, int n, culaDeviceFloat* a, int lda, culaDeviceDouble* sa, int ldsa);
culaStatus culaDeviceSlar2v(int n, culaDeviceFloat* x, culaDeviceFloat* y, culaDeviceFloat* z, int incx, culaDeviceFloat* c, culaDeviceFloat* s, int incc);
culaStatus culaDeviceSlarfb(char side, char trans, char direct, char storev, int m, int n, int k, culaDeviceFloat* v, int ldv, culaDeviceFloat* t, int ldt, culaDeviceFloat* c, int ldc);
culaStatus culaDeviceSlarfg(int n, culaDeviceFloat* alpha, culaDeviceFloat* x, int incx, culaDeviceFloat* tau);
culaStatus culaDeviceSlargv(int n, culaDeviceFloat* x, int incx, culaDeviceFloat* y, int incy, culaDeviceFloat* c, int incc);
culaStatus culaDeviceSlartv(int n, culaDeviceFloat* x, int incx, culaDeviceFloat* y, int incy, culaDeviceFloat* c, culaDeviceFloat* s, int incc);
culaStatus culaDeviceSlascl(char type, int kl, int ku, culaFloat cfrom, culaFloat cto, int m, int n, culaDeviceFloat* a, int lda);
culaStatus culaDeviceSlaset(char uplo, int m, int n, culaFloat alpha, culaFloat beta, culaDeviceFloat* a, int lda);
culaStatus culaDeviceSlasr(char side, char pivot, char direct, int m, int n, culaDeviceFloat* c, culaDeviceFloat* s, culaDeviceFloat* a, int lda);
culaStatus culaDeviceSlat2d(char uplo, int n, culaDeviceFloat* a, int lda, culaDeviceDouble* sa, int ldsa);
culaStatus culaDeviceSorgbr(char vect, int m, int n, int k, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSorghr(int n, int ilo, int ihi, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSorglq(int m, int n, int k, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSorgql(int m, int n, int k, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSorgqr(int m, int n, int k, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSorgrq(int m, int n, int k, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSormlq(char side, char trans, int m, int n, int k, culaDeviceFloat* a, int lda, culaDeviceFloat* tau, culaDeviceFloat* c, int ldc);
culaStatus culaDeviceSormql(char side, char trans, int m, int n, int k, culaDeviceFloat* a, int lda, culaDeviceFloat* tau, culaDeviceFloat* c, int ldc);
culaStatus culaDeviceSormqr(char side, char trans, int m, int n, int k, culaDeviceFloat* a, int lda, culaDeviceFloat* tau, culaDeviceFloat* c, int ldc);
culaStatus culaDeviceSormrq(char side, char trans, int m, int n, int k, culaDeviceFloat* a, int lda, culaDeviceFloat* tau, culaDeviceFloat* c, int ldc);
culaStatus culaDeviceSpbtrf(char uplo, int n, int kd, culaDeviceFloat* ab, int ldab);
culaStatus culaDeviceSposv(char uplo, int n, int nrhs, culaDeviceFloat* a, int lda, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceSpotrf(char uplo, int n, culaDeviceFloat* a, int lda);
culaStatus culaDeviceSpotri(char uplo, int n, culaDeviceFloat* a, int lda);
culaStatus culaDeviceSpotrs(char uplo, int n, int nrhs, culaDeviceFloat* a, int lda, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceSstebz(char range, char order, int n, float vl, float vu, int il, int iu, float abstol, culaDeviceFloat* d, culaDeviceFloat* e, int* m, int* nsplit, culaDeviceFloat* w, culaDeviceInt* iblock, culaDeviceInt* isplit);
culaStatus culaDeviceSsteqr(char compz, int n, culaDeviceFloat* d, culaDeviceFloat* e, culaDeviceFloat* z, int ldz);
culaStatus culaDeviceSsyev(char jobz, char uplo, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* w);
culaStatus culaDeviceSsyevx(char jobz, char range, char uplo, int n, culaDeviceFloat* a, int lda, culaFloat vl, culaFloat vu, int il, int iu, culaFloat abstol, culaInt* m, culaDeviceFloat* w, culaDeviceFloat* z, int ldz, culaInt* ifail);
culaStatus culaDeviceSsygv(int itype, char jobz, char uplo, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* b, int ldb, culaDeviceFloat* w);
culaStatus culaDeviceSsyrdb(char jobz, char uplo, int n, int kd, culaDeviceFloat* a, int lda, culaDeviceFloat* d, culaDeviceFloat* e, culaDeviceFloat* tau, culaDeviceFloat* z, int ldz);
culaStatus culaDeviceStrtri(char uplo, char diag, int n, culaDeviceFloat* a, int lda);
culaStatus culaDeviceStrtrs(char uplo, char trans, char diag, int n, int nrhs, culaDeviceFloat* a, int lda, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceZbdsqr(char uplo, int n, int ncvt, int nru, int ncc, culaDeviceDouble* d, culaDeviceDouble* e, culaDeviceDoubleComplex* vt, int ldvt, culaDeviceDoubleComplex* u, int ldu, culaDeviceDoubleComplex* c, int ldc);
culaStatus culaDeviceZcgesv(int n, int nrhs, culaDeviceDoubleComplex* a, int lda, culaInt* ipiv, culaDeviceDoubleComplex* b, int ldb, culaDeviceDoubleComplex* x, int ldx, int* iter);
culaStatus culaDeviceZcposv(char uplo, int n, int nrhs, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb, culaDeviceDoubleComplex* x, int ldx, int* iter);
culaStatus culaDeviceZgbtrf(int m, int n, int kl, int ku, culaDeviceDoubleComplex* a, int lda, culaInt* ipiv);
culaStatus culaDeviceZgeConjugate(int m, int n, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZgeNancheck(int m, int n, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZgeTranspose(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZgeTransposeConjugate(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZgeTransposeConjugateInplace(int n, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZgeTransposeInplace(int n, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZgebrd(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDouble* d, culaDeviceDouble* e, culaDeviceDoubleComplex* tauq, culaDeviceDoubleComplex* taup);
culaStatus culaDeviceZgeev(char jobvl, char jobvr, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* w, culaDeviceDoubleComplex* vl, int ldvl, culaDeviceDoubleComplex* vr, int ldvr);
culaStatus culaDeviceZgehrd(int n, int ilo, int ihi, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZgelqf(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZgels(char trans, int m, int n, int nrhs, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZgeqlf(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZgeqrf(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZgeqrfp(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZgeqrs(int m, int n, int nrhs, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZgerqf(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZgesdd(char jobz, int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDouble* s, culaDeviceDoubleComplex* u, int ldu, culaDeviceDoubleComplex* vt, int ldvt);
culaStatus culaDeviceZgesv(int n, int nrhs, culaDeviceDoubleComplex* a, int lda, culaDeviceInt* ipiv, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZgesvd(char jobu, char jobvt, int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDouble* s, culaDeviceDoubleComplex* u, int ldu, culaDeviceDoubleComplex* vt, int ldvt);
culaStatus culaDeviceZgetrf(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceInt* ipiv);
culaStatus culaDeviceZgetri(int n, culaDeviceDoubleComplex* a, int lda, culaDeviceInt* ipiv);
culaStatus culaDeviceZgetrs(char trans, int n, int nrhs, culaDeviceDoubleComplex* a, int lda, culaDeviceInt* ipiv, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZgglse(int m, int n, int p, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb, culaDeviceDoubleComplex* c, culaDeviceDoubleComplex* d, culaDeviceDoubleComplex* x);
culaStatus culaDeviceZggrqf(int m, int p, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* taua, culaDeviceDoubleComplex* b, int ldb, culaDeviceDoubleComplex* taub);
culaStatus culaDeviceZheev(char jobz, char uplo, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDouble* w);
culaStatus culaDeviceZheevx(char jobz, char range, char uplo, int n, culaDeviceDoubleComplex* a, int lda, culaDouble vl, culaDouble vu, int il, int iu, culaDouble abstol, culaInt* m, culaDeviceDouble* w, culaDeviceDoubleComplex* z, int ldz, culaInt* ifail);
culaStatus culaDeviceZhegv(int itype, char jobz, char uplo, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb, culaDeviceDouble* w);
culaStatus culaDeviceZherdb(char jobz, char uplo, int n, int kd, culaDeviceDoubleComplex* a, int lda, culaDeviceDouble* d, culaDeviceDouble* e, culaDeviceDoubleComplex* tau, culaDeviceDoubleComplex* z, int ldz);
culaStatus culaDeviceZlacpy(char uplo, int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZlag2c(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceFloatComplex* sa, int ldsa);
culaStatus culaDeviceZlar2v(int n, culaDeviceDoubleComplex* x, culaDeviceDoubleComplex* y, culaDeviceDoubleComplex* z, int incx, culaDeviceDouble* c, culaDeviceDoubleComplex* s, int incc);
culaStatus culaDeviceZlarfb(char side, char trans, char direct, char storev, int m, int n, int k, culaDeviceDoubleComplex* v, int ldv, culaDeviceDoubleComplex* t, int ldt, culaDeviceDoubleComplex* c, int ldc);
culaStatus culaDeviceZlarfg(int n, culaDeviceDoubleComplex* alpha, culaDeviceDoubleComplex* x, int incx, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZlargv(int n, culaDeviceDoubleComplex* x, int incx, culaDeviceDoubleComplex* y, int incy, culaDeviceDouble* c, int incc);
culaStatus culaDeviceZlartv(int n, culaDeviceDoubleComplex* x, int incx, culaDeviceDoubleComplex* y, int incy, culaDeviceDouble* c, culaDeviceDoubleComplex* s, int incc);
culaStatus culaDeviceZlascl(char type, int kl, int ku, culaDouble cfrom, culaDouble cto, int m, int n, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZlaset(char uplo, int m, int n, culaDoubleComplex alpha, culaDoubleComplex beta, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZlasr(char side, char pivot, char direct, int m, int n, culaDeviceDouble* c, culaDeviceDouble* s, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZlat2c(char uplo, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceFloatComplex* sa, int ldsa);
culaStatus culaDeviceZpbtrf(char uplo, int n, int kd, culaDeviceDoubleComplex* ab, int ldab);
culaStatus culaDeviceZposv(char uplo, int n, int nrhs, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZpotrf(char uplo, int n, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZpotri(char uplo, int n, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZpotrs(char uplo, int n, int nrhs, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZsteqr(char compz, int n, culaDeviceDouble* d, culaDeviceDouble* e, culaDeviceDoubleComplex* z, int ldz);
culaStatus culaDeviceZtrConjugate(char uplo, char diag, int m, int n, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZtrtri(char uplo, char diag, int n, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZtrtrs(char uplo, char trans, char diag, int n, int nrhs, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZungbr(char vect, int m, int n, int k, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZunghr(int n, int ilo, int ihi, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZunglq(int m, int n, int k, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZungql(int m, int n, int k, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZungqr(int m, int n, int k, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZungrq(int m, int n, int k, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZunmlq(char side, char trans, int m, int n, int k, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau, culaDeviceDoubleComplex* c, int ldc);
culaStatus culaDeviceZunmql(char side, char trans, int m, int n, int k, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau, culaDeviceDoubleComplex* c, int ldc);
culaStatus culaDeviceZunmqr(char side, char trans, int m, int n, int k, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau, culaDeviceDoubleComplex* c, int ldc);
culaStatus culaDeviceZunmrq(char side, char trans, int m, int n, int k, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau, culaDeviceDoubleComplex* c, int ldc);
"""

# Access CULA functions:
_ffi = FFI()
_ffi.cdef(_cula_type_str + _cula_dense_free_str)
_ffi_lib = _ffi.verify('#include <cula.h>',
                       libraries=['cula_lapack'])

# Function for retrieving string associated with specific CULA error
# code:
def culaGetStatusString(e):
    """
    Get string associated with the specified CULA status code.

    Parameters
    ----------
    e : int
        Status code.

    Returns
    -------
    s : str
        Status string.

    """

    return _ffi_lib.culaGetStatusString(e)

# Set CULA exception docstrings and add all of the exceptions to the module namespace:
for k, v in culaExceptions.iteritems():
    culaExceptions[v].__doc__ = cudaGetErrorString(v)
    vars()[k] = culaExceptions[v]

# CULA functions:
def culaGetErrorInfo():
    """
    Returns extended information code for the last CULA error.

    Returns
    -------
    err : int
        Extended information code.
    """

    return _ffi_lib.culaGetErrorInfo()

def culaGetErrorInfoString(e, i, bufsize=100):
    """
    Returns a readable CULA error string.

    Returns a readable error string corresponding to a given CULA
    error code and extended error information code.

    Parameters
    ----------
    e : int
        CULA error code.
    i : int
        Extended information code.
    bufsize : int
        Length of string to return.

    Returns
    -------
    s : str
        Error string.        
    """

    buf = _ffi.buffer(_ffi.new('char *[%s]' % bufsize))
    status = _ffi_lib.culaGetErrorInfoString(e, i, buf, bufsize)
    culaCheckStatus(status)
    return buf[:]
    
def culaCheckStatus(status):
    """
    Raise an exception corresponding to the specified CULA status
    code.

    Parameters
    ----------
    status : int
        CULA status code.
        
    """
    
    if status != 0:
        error = culaGetErrorInfo()
        try:
            raise culaExceptions[status](error)
        except KeyError:
            raise culaError(error)

def culaSelectDevice(dev):
    """
    Selects a device with which CULA will operate.

    Parameters
    ----------
    dev : int
        GPU device number.
        
    Notes
    -----
    Must be called before `culaInitialize`.
    """

    status = _ffi_lib.culaSelectDevice(dev)
    culaCheckStatus(status)

def culaGetExecutingDevice():
    """
    Reports the id of the GPU device used by CULA.

    Returns
    -------
    dev : int
       Device id.
    """

    dev = _ffi.new('int *')
    status = _ffi_lib.culaGetExecutingDevice(dev)
    culaCheckStatus(status)
    return dev[0]

def culaFreeBuffers():
    """
    Releases any memory buffers stored internally by CULA.
    """

    _ffi_lib.culaFreeBuffers()

def culaGetVersion():
    """
    Report the version number of CULA.
    """

    return _ffi_lib.culaGetVersion()

def culaGetCudaMinimumVersion():
    """
    Report the minimum version of CUDA required by CULA.
    """

    return _ffi_lib.culaGetCudaMinimumVersion()

def culaGetCudaRuntimeVersion():
    """
    Report the version of the CUDA runtime linked to by the CULA library.
    """

    return _ffi_lib.culaGetCudaRuntimeVersion()

def culaGetCudaDriverVersion():
    """
    Report the version of the CUDA driver installed on the system.
    """

    return _ffi_lib.culaGetCudaDriverVersion()

def culaGetCublasMinimumVersion():
    """
    Report the version of CUBLAS required by CULA.
    """

    return _ffi_lib.culaGetCublasMinimumVersion()

def culaGetCublasRuntimeVersion():
    """
    Report the version of CUBLAS linked to by CULA.
    """

    return _ffi_lib.culaGetCublasRuntimeVersion()

def culaGetDeviceCount():
    """
    Report the number of available GPU devices.
    """

    return _ffi_lib.culaGetDeviceCount()

def culaInitialize():
    """
    Initialize CULA.

    Notes
    -----
    Must be called before using any other CULA functions.
    """
    
    status = _ffi_lib.culaInitialize()
    culaCheckStatus(status)

def culaShutdown():
    """
    Shuts down CULA.
    """
    
    status = _ffi_lib.culaShutdown()
    culaCheckStatus(status)

# Shut down CULA upon exit:
atexit.register(_ffi_lib.culaShutdown)

# LAPACK functions available in CULA Dense Free:

# SGESV, CGESV
def culaDeviceSgesv(n, nrhs, a, lda, ipiv, b, ldb):
    """
    Solve linear system with LU factorization.
    """

    status = _ffi_lib.culaDeviceSgesv(n, nrhs, 
                                      _ffi.cast('culaDeviceFloat *', a), lda, 
                                      _ffi.cast('culaDeviceInt *', ipiv),
                                      _ffi.cast('culaDeviceFloat *', b), ldb)
    culaCheckStatus(status)

def culaDeviceCgesv(n, nrhs, a, lda, ipiv, b, ldb):
    """
    Solve linear system with LU factorization.
    """

    status = _ffi_lib.culaDeviceCgesv(n, nrhs,
                                      _ffi.cast('culaDeviceFloatComplex *', a), lda,
                                      _ffi.cast('culaDeviceInt *', ipiv),
                                      _ffi.cast('culaDeviceFloatComplex *', b), ldb)
    culaCheckStatus(status)

# SGETRF, CGETRF    
def culaDeviceSgetrf(m, n, a, lda, ipiv):
    """
    LU factorization.
    """
    
    status = _ffi_lib.culaDeviceSgetrf(m, n, 
                                       _ffi.cast('culaDeviceFloat *', a), lda, 
                                       _ffi.cast('culaDeviceInt *', ipiv))
    culaCheckStatus(status)

def culaDeviceCgetrf(m, n, a, lda, ipiv):
    """
    LU factorization.
    """
    
    status = _ffi_lib.culaDeviceCgetrf(m, n,
                                       _ffi.cast('culaDeviceFloatComplex *', a), lda, 
                                       _ffi.cast('culaDeviceInt *', ipiv))
    culaCheckStatus(status)

# SGEQRF, CGEQRF    
def culaDeviceSgeqrf(m, n, a, lda, tau):
    """
    QR factorization.
    """
    
    status = _ffi_lib.culaDeviceSgeqrf(m, n,
                                       _ffi.cast('culaDeviceFloat *', a), lda,
                                       _ffi.cast('culaDeviceFloat *', tau))
    culaCheckStatus(status)

def culaDeviceCgeqrf(m, n, a, lda, tau):
    """
    QR factorization.
    """
    
    status = _ffi_lib.culaDeviceCgeqrf(m, n,
                                       _ffi.cast('culaDeviceFloatComplex *', a), lda,
                                       _ffi.cast('culaDeviceFloatComplex *', tau))
    culaCheckStatus(status)

# SGELS, CGELS    
def culaDeviceSgels(trans, m, n, nrhs, a, lda, b, ldb):
    """
    Solve linear system with QR or LQ factorization.
    """
    
    status = _ffi_lib.culaDeviceSgels(trans, m, n, nrhs, 
                                      _ffi.cast('culaDeviceFloat *', a), lda,
                                      _ffi.cast('culaDeviceFloat *', b), ldb)
    culaCheckStatus(status)

def culaDeviceCgels(trans, m, n, nrhs, a, lda, b, ldb):
    """
    Solve linear system with QR or LQ factorization.
    """

    status = _ffi_lib.culaDeviceCgels(trans, m, n, nrhs,
                                      _ffi.cast('culaDeviceFloatComplex *', a), lda,
                                      _ffi.cast('culaDeviceFloatComplex *', b), ldb)
    culaCheckStatus(status)

# SGGLSE, CGGLSE    
def culaDeviceSgglse(m, n, p, a, lda, b, ldb, c, d, x):
    """
    Solve linear equality-constrained least squares problem.
    """
    
    status = _ffi_lib.culaDeviceSgglse(m, n, p, 
                                       _ffi.cast('culaDeviceFloat *', a), lda, 
                                       _ffi.cast('culaDeviceFloat *', b), ldb,
                                       _ffi.cast('culaDeviceFloat *', c), 
                                       _ffi.cast('culaDeviceFloat *', d),
                                       _ffi.cast('culaDeviceFloat *', x))
    culaCheckStatus(status)

def culaDeviceCgglse(m, n, p, a, lda, b, ldb, c, d, x):
    """
    Solve linear equality-constrained least squares problem.
    """

    status = _ffi_lib.culaDeviceCgglse(m, n, p,
                                       _ffi.cast('culaDeviceFloatComplex *', a), lda, 
                                       _ffi.cast('culaDeviceFloatComplex *', b), ldb,
                                       _ffi.cast('culaDeviceFloatComplex *', c), 
                                       _ffi.cast('culaDeviceFloatComplex *', d),
                                       _ffi.cast('culaDeviceFloatComplex *', x))
    culaCheckStatus(status)

# SGESVD, CGESVD    
def culaDeviceSgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt):
    """
    SVD decomposition.
    """
    
    status = _ffi_lib.culaDeviceSgesvd(jobu, jobvt, m, n, 
                                       _ffi.cast('culaDeviceFloat *', a), lda,
                                       _ffi.cast('culaDeviceFloat *', s), 
                                       _ffi.cast('culaDeviceFloat *', u), ldu, 
                                       _ffi.cast('culaDeviceFloat *', vt), ldvt)
    culaCheckStatus(status)

def culaDeviceCgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt):
    """
    SVD decomposition.
    """

    status = _ffi_lib.culaDeviceCgesvd(jobu, jobvt, m, n,
                                       _ffi.cast('culaDeviceFloatComplex *', a), lda,
                                       _ffi.cast('culaDeviceFloatComplex *', s), 
                                       _ffi.cast('culaDeviceFloatComplex *', u), ldu, 
                                       _ffi.cast('culaDeviceFloatComplex *', vt), ldvt)
    culaCheckStatus(status)

# LAPACK functions available in CULA Dense:

def _cula_standard_req(f):
    """
    Decorator to replace function with a placeholder that raises an exception
    if the standard version of CULA is not installed:
    """
    
    def f_new(*args,**kwargs):
        raise NotImplementedError('CULA Dense required')
    f_new.__doc__ = f.__doc__

    if _libcula_toolkit == 'standard':
        return f
    else:
        return f_new

# DGESV, ZGESV
@_cula_standard_req
def culaDeviceDgesv(n, nrhs, a, lda, ipiv, b, ldb):
    """
    Solve linear system with LU factorization.
    """

    status = _ffi_lib.culaDeviceDgesv(n, nrhs, 
                                      _ffi.cast('culaDeviceDouble *', a), lda, 
                                      _ffi.cast('culaDeviceInt *', ipiv),
                                      _ffi.cast('culaDeviceDouble *', b), ldb)
    culaCheckStatus(status)

@_cula_standard_req
def culaDeviceZgesv(n, nrhs, a, lda, ipiv, b, ldb):
    """
    Solve linear system with LU factorization.
    """

    status = _ffi_lib.culaDeviceZgesv(n, nrhs,
                                      _ffi.cast('culaDeviceDoubleComplex *', a), lda,
                                      _ffi.cast('culaDeviceInt *', ipiv),
                                      _ffi.cast('culaDeviceDoubleComplex *', b), ldb)
    culaCheckStatus(status)

# DGETRF, ZGETRF        
@_cula_standard_req
def culaDeviceDgetrf(m, n, a, lda, ipiv):
    """
    LU factorization.
    """

    status = _ffi_lib.culaDeviceDgetrf(m, n,
                                       _ffi.cast('culaDeviceDouble *', a), lda,
                                       _ffi.cast('culaDeviceInt *', ipiv))
    culaCheckStatus(status)

@_cula_standard_req
def culaDeviceZgetrf(m, n, a, lda, ipiv):
    """
    LU factorization.
    """

    status = _ffi_lib.culaDeviceZgetrf(m, n,
                                       _ffi.cast('culaDeviceDoubleComplex *', a), lda,
                                       _ffi.cast('culaDeviceInt *', ipiv))
    culaCheckStatus(status)

# DGEQRF, ZGEQRF        
@_cula_standard_req
def culaDeviceDgeqrf(m, n, a, lda, tau):
    """
    QR factorization.
    """

    status = _ffi_lib.culaDeviceDgeqrf(m, n,
                                       _ffi.cast('culaDeviceDouble *', a), lda,
                                       _ffi.cast('culaDeviceDouble *', tau))
    culaCheckStatus(status)

@_cula_standard_req
def culaDeviceZgeqrf(m, n, a, lda, tau):
    """
    QR factorization.
    """

    status = _ffi_lib.culaDeviceZgeqrf(m, n,
                                       _ffi.cast('culaDeviceDoubleComplex *', a), lda,
                                       _ffi.cast('culaDeviceDoubleComplex *', tau))
    culaCheckStatus(status)

# DGELS, ZGELS        
@_cula_standard_req
def culaDeviceDgels(trans, m, n, nrhs, a, lda, b, ldb):
    """
    Solve linear system with QR or LQ factorization.
    """

    status = _ffi_lib.culaDeviceDgels(trans, m, n, nrhs, 
                                      _ffi.cast('culaDeviceDouble *', a), lda,
                                      _ffi.cast('culaDeviceDouble *', b), ldb)
    culaCheckStatus(status)

@_cula_standard_req
def culaDeviceZgels(trans, m, n, nrhs, a, lda, b, ldb):
    """
    Solve linear system with QR or LQ factorization.
    """

    status = _ffi_lib.culaDeviceZgels(trans, m, n, nrhs,
                                      _ffi.cast('culaDeviceDoubleComplex *', a), lda,
                                      _ffi.cast('culaDeviceDoubleComplex *', b), ldb)
    culaCheckStatus(status)

# DGGLSE, ZGGLSE        
@_cula_standard_req
def culaDeviceDgglse(m, n, p, a, lda, b, ldb, c, d, x):
    """
    Solve linear equality-constrained least squares problem.
    """

    status = _ffi_lib.culaDeviceDgglse(m, n, p,
                                       _ffi.cast('culaDeviceDouble *', a), lda,
                                       _ffi.cast('culaDeviceDouble *', b), ldb,
                                       _ffi.cast('culaDeviceDouble *', c), 
                                       _ffi.cast('culaDeviceDouble *', d),
                                       _ffi.cast('culaDeviceDouble *', x))                                       
    culaCheckStatus(status)

@_cula_standard_req
def culaDeviceZgglse(m, n, p, a, lda, b, ldb, c, d, x):
    """
    Solve linear equality-constrained least squares problem.
    """

    status = _ffi_lib.culaDeviceZgglse(m, n, p,
                                       _ffi.cast('culaDeviceDoubleComplex *', a), lda,
                                       _ffi.cast('culaDeviceDoubleComplex *', b), ldb,
                                       _ffi.cast('culaDeviceDoubleComplex *', c),
                                       _ffi.cast('culaDeviceDoubleComplex *', d),
                                       _ffi.cast('culaDeviceDoubleComplex *', x))
    culaCheckStatus(status)

# DGESVD, ZGESVD        
@_cula_standard_req
def culaDeviceDgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt):
    """
    SVD decomposition.
    """

    status = _ffi_lib.culaDeviceDgesvd(jobu, jobvt, m, n, 
                                       _ffi.cast('culaDeviceDouble *', a), lda,
                                       _ffi.cast('culaDeviceDouble *', s),
                                       _ffi.cast('culaDeviceDouble *', u), ldu,
                                       _ffi.cast('culaDeviceDouble *', vt), ldvt)
    culaCheckStatus(status)

@_cula_standard_req
def culaDeviceZgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt):
    """
    SVD decomposition.
    """

    status = _ffi_lib.culaDeviceZgesvd(jobu, jobvt, m, n,
                                       _ffi.cast('culaDeviceDoubleComplex *', a), lda,
                                       _ffi.cast('culaDeviceDoubleComplex *', s),
                                       _ffi.cast('culaDeviceDoubleComplex *', u), ldu,
                                       _ffi.cast('culaDeviceDoubleComplex *', vt), ldvt)
    culaCheckStatus(status)

# SPOSV, CPOSV, DPOSV, ZPOSV        
@_cula_standard_req
def culaDeviceSposv(upio, n, nrhs, a, lda, b, ldb):
    """
    Solve positive definite linear system with Cholesky factorization.
    """

    status = _ffi_lib.culaDeviceSposv(upio, n, nrhs,
                                      _ffi.cast('culaDeviceFloat *', a), lda,
                                      _ffi.cast('culaDeviceFloat *', b), ldb)
    culaCheckStatus(status)

@_cula_standard_req
def culaDeviceCposv(upio, n, nrhs, a, lda, b, ldb):
    """
    Solve positive definite linear system with Cholesky factorization.
    """

    status = _ffi_lib.culaDeviceCposv(upio, n, nrhs,
                                      _ffi.cast('culaDeviceFloatComplex *', a), lda, 
                                      _ffi.cast('culaDeviceFloatComplex *', b), ldb)
    culaCheckStatus(status)

@_cula_standard_req
def culaDeviceDposv(upio, n, nrhs, a, lda, b, ldb):
    """
    Solve positive definite linear system with Cholesky factorization.
    """

    status = _ffi_lib.culaDeviceDposv(upio, n, nrhs,
                                      _ffi.cast('culaDeviceDouble *', a), lda,
                                      _ffi.cast('culaDeviceDouble *', b), ldb)
    culaCheckStatus(status)

@_cula_standard_req
def culaDeviceZposv(upio, n, nrhs, a, lda, b, ldb):
    """
    Solve positive definite linear system with Cholesky factorization.
    """

    status = _ffi_lib.culaDeviceZposv(upio, n, nrhs,
                                      _ffi.cast('culaDeviceDoubleComplex *', a), lda,
                                      _ffi.cast('culaDeviceDoubleComplex *', b), ldb)
    culaCheckStatus(status)

# SPOTRF, CPOTRF, DPOTRF, ZPOTRF        
try:
    _ffi_lib.culaDeviceSpotrf.restype = \
    _ffi_lib.culaDeviceCpotrf.restype = \
    _ffi_lib.culaDeviceDpotrf.restype = \
    _ffi_lib.culaDeviceZpotrf.restype = int
    _ffi_lib.culaDeviceSpotrf.argtypes = \
    _ffi_lib.culaDeviceCpotrf.argtypes = \
    _ffi_lib.culaDeviceDpotrf.argtypes = \
    _ffi_lib.culaDeviceZpotrf.argtypes = [ctypes.c_char,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
except AttributeError:
    def culaDeviceSpotrf(uplo, n, a, lda):
        """
        Cholesky factorization.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceCpotrf(uplo, n, a, lda):
        """
        Cholesky factorization.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceDpotrf(uplo, n, a, lda):
        """
        Cholesky factorization.

        """
        
        raise NotImplementedError('CULA Dense required')

    def culaDeviceZpotrf(uplo, n, a, lda):
        """
        Cholesky factorization.

        """
        
        raise NotImplementedError('CULA Dense required')    
else:            
    def culaDeviceSpotrf(uplo, n, a, lda):
        """
        Cholesky factorization.

        """

        status = _ffi_lib.culaDeviceSpotrf(uplo, n, int(a), lda)
        culaCheckStatus(status)

    def culaDeviceCpotrf(uplo, n, a, lda):
        """
        Cholesky factorization.

        """

        status = _ffi_lib.culaDeviceCpotrf(uplo, n, int(a), lda)
        culaCheckStatus(status)

    def culaDeviceDpotrf(uplo, n, a, lda):
        """
        Cholesky factorization.

        """

        status = _ffi_lib.culaDeviceDpotrf(uplo, n, int(a), lda)
        culaCheckStatus(status)

    def culaDeviceZpotrf(uplo, n, a, lda):
        """
        Cholesky factorization.

        """

        status = _ffi_lib.culaDeviceZpotrf(uplo, n, int(a), lda)
        culaCheckStatus(status)

# SSYEV, DSYEV, CHEEV, ZHEEV        
try:
    _ffi_lib.culaDeviceSsyev.restype = \
    _ffi_lib.culaDeviceDsyev.restype = \
    _ffi_lib.culaDeviceCheev.restype = \
    _ffi_lib.culaDeviceZheev.restype = int
    _ffi_lib.culaDeviceSsyev.argtypes = \
    _ffi_lib.culaDeviceDsyev.argtypes = \
    _ffi_lib.culaDeviceCheev.argtypes = \
    _ffi_lib.culaDeviceZheev.argtypes = [ctypes.c_char,
                                         ctypes.c_char,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p]
except AttributeError:
    def culaDeviceSsyev(jobz, uplo, n, a, lda, w):
        """
        Symmetric eigenvalue decomposition.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceDsyev(jobz, uplo, n, a, lda, w):
        """
        Symmetric eigenvalue decomposition.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceCheev(jobz, uplo, n, a, lda, w):
        """
        Hermitian eigenvalue decomposition.

        """
        
        raise NotImplementedError('CULA Dense required')

    def culaDeviceZheev(jobz, uplo, n, a, lda, w):
        """
        Hermitian eigenvalue decomposition.

        """
        
        raise NotImplementedError('CULA Dense required')    
else:
    
    def culaDeviceSsyev(jobz, uplo, n, a, lda, w):
        """
        Symmetric eigenvalue decomposition.

        """

        status = _ffi_lib.culaDeviceSsyev(jobz, uplo, n, int(a), lda, int(w))
        culaCheckStatus(status)

    def culaDeviceDsyev(jobz, uplo, n, a, lda, w):
        """
        Symmetric eigenvalue decomposition.

        """

        status = _ffi_lib.culaDeviceDsyev(jobz, uplo, n, int(a), lda, int(w))
        culaCheckStatus(status)

    def culaDeviceCheev(jobz, uplo, n, a, lda, w):
        """
        Hermitian eigenvalue decomposition.

        """

        status = _ffi_lib.culaDeviceCheev(jobz, uplo, n, int(a), lda, int(w))
        culaCheckStatus(status)

    def culaDeviceZheev(jobz, uplo, n, a, lda, w):
        """
        Hermitian eigenvalue decomposition.

        """

        status = _ffi_lib.culaDeviceZheev(jobz, uplo, n, int(a), lda, int(w))
        culaCheckStatus(status)

# BLAS routines provided by CULA:

# SGEMM, DGEMM, CGEMM, ZGEMM
_ffi_lib.culaDeviceSgemm.restype = \
_ffi_lib.culaDeviceDgemm.restype = \
_ffi_lib.culaDeviceCgemm.restype = \
_ffi_lib.culaDeviceZgemm.restype = int

_ffi_lib.culaDeviceSgemm.argtypes = [ctypes.c_char,
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

_ffi_lib.culaDeviceDgemm.argtypes = [ctypes.c_char,
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

_ffi_lib.culaDeviceCgemm.argtypes = [ctypes.c_char,
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

_ffi_lib.culaDeviceZgemm.argtypes = [ctypes.c_char,
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

def culaDeviceSgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for general matrix.

    """
    
    status = _ffi_lib.culaDeviceSgemm(transa, transb, m, n, k, alpha,
                           int(A), lda, int(B), ldb, beta, int(C), ldc)
    culaCheckStatus(status)

def culaDeviceDgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for general matrix.

    """
    
    status = _ffi_lib.culaDeviceDgemm(transa, transb, m, n, k, alpha,
                           int(A), lda, int(B), ldb, beta, int(C), ldc)
    culaCheckStatus(status)

def culaDeviceCgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for complex general matrix.

    """
    
    status = _ffi_lib.culaDeviceCgemm(transa, transb, m, n, k,
                                      cuda.cuFloatComplex(alpha.real,
                                                        alpha.imag),
                                      int(A), lda, int(B), ldb,
                                      cuda.cuFloatComplex(beta.real,
                                                        beta.imag),
                                      int(C), ldc)
    culaCheckStatus(status)

def culaDeviceZgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for complex general matrix.

    """
    
    status = _ffi_lib.culaDeviceZgemm(transa, transb, m, n, k,
                                      cuda.cuDoubleComplex(alpha.real,
                                                        alpha.imag),
                                      int(A), lda, int(B), ldb,
                                      cuda.cuDoubleComplex(beta.real,
                                                        beta.imag),
                                      int(C), ldc)
    culaCheckStatus(status)

# SGEMV, DGEMV, CGEMV, ZGEMV
_ffi_lib.culaDeviceSgemv.restype = \
_ffi_lib.culaDeviceDgemv.restype = \
_ffi_lib.culaDeviceCgemv.restype = \
_ffi_lib.culaDeviceZgemv.restype = int

_ffi_lib.culaDeviceSgemv.argtypes = [ctypes.c_char,
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

_ffi_lib.culaDeviceDgemv.argtypes = [ctypes.c_char,
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

_ffi_lib.culaDeviceCgemv.argtypes = [ctypes.c_char,
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

_ffi_lib.culaDeviceZgemv.argtypes = [ctypes.c_char,
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

def culaDeviceSgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for real general matrix.

    """
    
    status = _ffi_lib.culaDeviceSgemv(trans, m, n, alpha, int(A), lda,
                           int(x), incx, beta, int(y), incy)
    culaCheckStatus(status)

def culaDeviceDgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for real general matrix.

    """
    
    status = _ffi_lib.culaDeviceDgemv(trans, m, n, alpha, int(A), lda,
                           int(x), incx, beta, int(y), incy)
    culaCheckStatus(status)
    

def culaDeviceCgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for complex general matrix.

    """
    
    status = _ffi_lib.culaDeviceCgemv(trans, m, n,
                           cuda.cuFloatComplex(alpha.real,
                                               alpha.imag),
                           int(A), lda, int(x), incx,
                           cuda.cuFloatComplex(beta.real,
                                               beta.imag),
                           int(y), incy)
    culaCheckStatus(status)

def culaDeviceZgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for complex general matrix.

    """
    
    status = _ffi_lib.culaDeviceZgemv(trans, m, n,
                           cuda.cuDoubleComplex(alpha.real,
                                               alpha.imag),
                           int(A), lda, int(x), incx,
                           cuda.cuDoubleComplex(beta.real,
                                               beta.imag),
                           int(y), incy)
    culaCheckStatus(status)
    
# Auxiliary routines:
    
try:
    _ffi_lib.culaDeviceSgeTranspose.restype = \
    _ffi_lib.culaDeviceDgeTranspose.restype = \
    _ffi_lib.culaDeviceCgeTranspose.restype = \
    _ffi_lib.culaDeviceZgeTranspose.restype = int
    _ffi_lib.culaDeviceSgeTranspose.argtypes = \
    _ffi_lib.culaDeviceDgeTranspose.argtypes = \
    _ffi_lib.culaDeviceCgeTranspose.argtypes = \
    _ffi_lib.culaDeviceZgeTranspose.argtypes = [ctypes.c_int,
                                                ctypes.c_int,
                                                ctypes.c_void_p,
                                                ctypes.c_int,
                                                ctypes.c_void_p,
                                                ctypes.c_int]
except AttributeError:
    def culaDeviceSgeTranspose(m, n, A, lda, B, ldb):
        """
        Transpose of real general matrix.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceDgeTranspose(m, n, A, lda, B, ldb):
        """
        Transpose of real general matrix.

        """
        
        raise NotImplementedError('CULA Dense required')

    def culaDeviceCgeTranspose(m, n, A, lda, B, ldb):
        """
        Transpose of complex general matrix.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgeTranspose(m, n, A, lda, B, ldb):
        """
        Transpose of complex general matrix.

        """

        raise NotImplementedError('CULA Dense required')
else:
    def culaDeviceSgeTranspose(m, n, A, lda, B, ldb):
        """
        Transpose of real general matrix.

        """
        
        status = _ffi_lib.culaDeviceSgeTranspose(m, n, int(A), lda, int(B), ldb)
        culaCheckStatus(status)

    def culaDeviceDgeTranspose(m, n, A, lda, B, ldb):
        """
        Transpose of real general matrix.

        """
        
        status = _ffi_lib.culaDeviceDgeTranspose(m, n, int(A), lda, int(B), ldb)
        culaCheckStatus(status)

    def culaDeviceCgeTranspose(m, n, A, lda, B, ldb):
        """
        Transpose of complex general matrix.

        """
        
        status = _ffi_lib.culaDeviceCgeTranspose(m, n, int(A), lda, int(B), ldb)
        culaCheckStatus(status)

    def culaDeviceZgeTranspose(m, n, A, lda, B, ldb):
        """
        Transpose of complex general matrix.

        """
        
        status = _ffi_lib.culaDeviceZgeTranspose(m, n, int(A), lda, int(B), ldb)
        culaCheckStatus(status)
    
    
try:
    _ffi_lib.culaDeviceSgeTransposeInplace.restype = \
    _ffi_lib.culaDeviceDgeTransposeInplace.restype = \
    _ffi_lib.culaDeviceCgeTransposeInplace.restype = \
    _ffi_lib.culaDeviceZgeTransposeInplace.restype = int
    _ffi_lib.culaDeviceSgeTransposeInplace.argtypes = \
    _ffi_lib.culaDeviceDgeTransposeInplace.argtypes = \
    _ffi_lib.culaDeviceCgeTransposeInplace.argtypes = \
    _ffi_lib.culaDeviceZgeTransposeInplace.argtypes = [ctypes.c_int,
                                                    ctypes.c_void_p,
                                                    ctypes.c_int]
except AttributeError:
    def culaDeviceSgeTransposeInplace(n, A, lda):
        """
        Inplace transpose of real square matrix.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceDgeTransposeInplace(n, A, lda):
        """
        Inplace transpose of real square matrix.

        """
        
        raise NotImplementedError('CULA Dense required')

    def culaDeviceCgeTransposeInplace(n, A, lda):
        """
        Inplace transpose of complex square matrix.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgeTransposeInplace(n, A, lda):
        """
        Inplace transpose of complex square matrix.

        """

        raise NotImplementedError('CULA Dense required')
else:
    def culaDeviceSgeTransposeInplace(n, A, lda):
        """
        Inplace transpose of real square matrix.

        """
        
        status = _ffi_lib.culaDeviceSgeTransposeInplace(n, int(A), lda)
        culaCheckStatus(status)

    def culaDeviceDgeTransposeInplace(n, A, lda):
        """
        Inplace transpose of real square matrix.

        """
        
        status = _ffi_lib.culaDeviceDgeTransposeInplace(n, int(A), lda)
        culaCheckStatus(status)

    def culaDeviceCgeTransposeInplace(n, A, lda):
        """
        Inplace transpose of complex square matrix.

        """
        
        status = _ffi_lib.culaDeviceCgeTransposeInplace(n, int(A), lda)
        culaCheckStatus(status)

    def culaDeviceZgeTransposeInplace(n, A, lda):
        """
        Inplace transpose of complex square matrix.

        """
        
        status = _ffi_lib.culaDeviceZgeTransposeInplace(n, int(A), lda)
        culaCheckStatus(status)

try:

    _ffi_lib.culaDeviceCgeTransposeConjugate.restype = \
    _ffi_lib.culaDeviceZgeTransposeConjugate.restype = int
    _ffi_lib.culaDeviceCgeTransposeConjugate.argtypes = \
    _ffi_lib.culaDeviceZgeTransposeConjugate.argtypes = [ctypes.c_int,
                                                        ctypes.c_int,
                                                        ctypes.c_void_p,
                                                        ctypes.c_int,
                                                        ctypes.c_void_p,
                                                        ctypes.c_int]
except AttributeError:
    def culaDeviceCgeTransposeConjugate(m, n, A, lda, B, ldb):
        """
        Conjugate transpose of complex general matrix.

        """
        
        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgeTransposeConjugate(m, n, A, lda, B, ldb):
        """
        Conjugate transpose of complex general matrix.

        """
        raise NotImplementedError('CULA Dense required')    
else:
    def culaDeviceCgeTransposeConjugate(m, n, A, lda, B, ldb):
        """
        Conjugate transpose of complex general matrix.

        """
        
        status = _ffi_lib.culaDeviceCgeTransposeConjugate(m, n, int(A), lda, int(B), ldb)
        culaCheckStatus(status)

    def culaDeviceZgeTransposeConjugate(m, n, A, lda, B, ldb):
        """
        Conjugate transpose of complex general matrix.

        """
        
        status = _ffi_lib.culaDeviceZgeTransposeConjugate(m, n, int(A), lda, int(B), ldb)
        culaCheckStatus(status)

try:
    _ffi_lib.culaDeviceCgeTransposeConjugateInplace.restype = \
    _ffi_lib.culaDeviceZgeTransposeConjugateInplace.restype = int
    _ffi_lib.culaDeviceCgeTransposeConjugateInplace.argtypes = \
    _ffi_lib.culaDeviceZgeTransposeConjugateInplace.argtypes = [ctypes.c_int,
                                                                ctypes.c_void_p,
                                                                ctypes.c_int]
except AttributeError:
    def culaDeviceCgeTransposeConjugateInplace(n, A, lda):
        """
        Inplace conjugate transpose of complex square matrix.

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgeTransposeConjugateInplace(n, A, lda):
        """
        Inplace conjugate transpose of complex square matrix.

        """
        
        raise NotImplementedError('CULA Dense required')    
else:
    def culaDeviceCgeTransposeConjugateInplace(n, A, lda):
        """
        Inplace conjugate transpose of complex square matrix.

        """
        
        status = _ffi_lib.culaDeviceCgeTransposeConjugateInplace(n, int(A), lda)
        culaCheckStatus(status)

    def culaDeviceZgeTransposeConjugateInplace(n, A, lda):
        """
        Inplace conjugate transpose of complex square matrix.

        """
        
        status = _ffi_lib.culaDeviceZgeTransposeConjugateInplace(n, int(A), lda)
        culaCheckStatus(status)

try:
    _ffi_lib.culaDeviceCgeConjugate.restype = \
    _ffi_lib.culaDeviceZgeConjugate.restype = int
    _ffi_lib.culaDeviceCgeConjugate.argtypes = \
    _ffi_lib.culaDeviceZgeConjugate.argtypes = [ctypes.c_int,
                                                ctypes.c_int,
                                                ctypes.c_void_p,
                                                ctypes.c_int]
except AttributeError:
    def culaDeviceCgeConjugate(m, n, A, lda):
        """
        Conjugate of complex general matrix.

        """
    
        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgeConjugate(m, n, A, lda):
        """
        Conjugate of complex general matrix.

        """

        raise NotImplementedError('CULA Dense required')
else:
    def culaDeviceCgeConjugate(m, n, A, lda):
        """
        Conjugate of complex general matrix.

        """
        
        status = _ffi_lib.culaDeviceCgeConjugate(m, n, int(A), lda)
        culaCheckStatus(status)

    def culaDeviceZgeConjugate(m, n, A, lda):
        """
        Conjugate of complex general matrix.

        """
        
        status = _ffi_lib.culaDeviceZgeConjugate(m, n, int(A), lda)
        culaCheckStatus(status)

try:
    _ffi_lib.culaDeviceCtrConjugate.restype = \
    _ffi_lib.culaDeviceZtrConjugate.restype = int
    _ffi_lib.culaDeviceCtrConjugate.argtypes = \
    _ffi_lib.culaDeviceZtrConjugate.argtypes = [ctypes.c_char,
                                                ctypes.c_char,
                                                ctypes.c_int,
                                                ctypes.c_int,
                                                ctypes.c_void_p,
                                                ctypes.c_int]
except AttributeError:
    def culaDeviceCtrConjugate(uplo, diag, m, n, A, lda):
        """
        Conjugate of complex upper or lower triangle matrix.

        """
    
        raise NotImplementedError('CULA Dense required')

    def culaDeviceZtrConjugate(uplo, diag, m, n, A, lda):
        """
        Conjugate of complex upper or lower triangle matrix.

        """
        
        raise NotImplementedError('CULA Dense required')    
else:
    def culaDeviceCtrConjugate(uplo, diag, m, n, A, lda):
        """
        Conjugate of complex upper or lower triangle matrix.

        """
        
        status = _ffi_lib.culaDeviceCtrConjugate(uplo, diag, m, n, int(A), lda)
        culaCheckStatus(status)

    def culaDeviceZtrConjugate(uplo, diag, m, n, A, lda):
        """
        Conjugate of complex upper or lower triangle matrix.

        """
        
        status = _ffi_lib.culaDeviceZtrConjugate(uplo, diag, m, n, int(A), lda)
        culaCheckStatus(status)

try:
    _ffi_lib.culaDeviceSgeNancheck.restype = \
    _ffi_lib.culaDeviceDgeNancheck.restype = \
    _ffi_lib.culaDeviceCgeNancheck.restype = \
    _ffi_lib.culaDeviceZgeNancheck.restype = int
    _ffi_lib.culaDeviceSgeNancheck.argtypes = \
    _ffi_lib.culaDeviceDgeNancheck.argtypes = \
    _ffi_lib.culaDeviceCgeNancheck.argtypes = \
    _ffi_lib.culaDeviceZgeNancheck.argtypes = [ctypes.c_int,
                                                ctypes.c_int,
                                                ctypes.c_void_p,
                                                ctypes.c_int]
except AttributeError:
    def culaDeviceSgeNancheck(m, n, A, lda):
        """
        Check a real general matrix for invalid entries

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceDgeNancheck(m, n, A, lda):
        """
        Check a real general matrix for invalid entries

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceCgeNancheck(m, n, A, lda):
        """
        Check a complex general matrix for invalid entries

        """

        raise NotImplementedError('CULA Dense required')

    def culaDeviceZgeNancheck(m, n, A, lda):
        """
        Check a complex general matrix for invalid entries

        """

        raise NotImplementedError('CULA Dense required')        
else:
    def culaDeviceSgeNancheck(m, n, A, lda):
        """
        Check a real general matrix for invalid entries

        """
        
        status = _ffi_lib.culaDeviceSgeNancheck(m, n, int(A), lda)
        try:
            culaCheckStatus(status)
        except culaDataError:
            return True
        return False

    def culaDeviceDgeNancheck(m, n, A, lda):
        """
        Check a real general matrix for invalid entries

        """
        
        status = _ffi_lib.culaDeviceDgeNancheck(m, n, int(A), lda)
        try:
            culaCheckStatus(status)
        except culaDataError:
            return True
        return False

    def culaDeviceCgeNancheck(m, n, A, lda):
        """
        Check a complex general matrix for invalid entries

        """
        
        status = _ffi_lib.culaDeviceCgeNancheck(m, n, int(A), lda)
        try:
            culaCheckStatus(status)
        except culaDataError:
            return True
        return False

    def culaDeviceZgeNancheck(m, n, A, lda):
        """
        Check a complex general matrix for invalid entries

        """
        
        status = _ffi_lib.culaDeviceZgeNancheck(m, n, int(A), lda)
        try:
            culaCheckStatus(status)
        except culaDataError:
            return True
        return False

        
if __name__ == "__main__":
    import doctest
    doctest.testmod()
