#!/usr/bin/env python

"""
Python interface to MAGMA toolkit.
"""

import sys
import ctypes
import atexit
import numpy as np

import cuda

# Load MAGMA library:
if 'linux' in sys.platform:
    _libmagma_libname_list = ['libmagma.so']
elif sys.platform == 'darwin':
    _libmagma_libname_list = ['magma.so', 'libmagma.dylib']
elif sys.platform == 'win32':
    _libmagma_libname_list = ['magma.dll']
else:
    raise RuntimeError('unsupported platform')

_load_err = ''
for _lib in _libmagma_libname_list:
    try:
        _libmagma = ctypes.cdll.LoadLibrary(_lib)
    except OSError:
        _load_err += ('' if _load_err == '' else ', ') + _lib
    else:
        _load_err = ''
        break
if _load_err:
    raise OSError('%s not found' % _load_err)

# Exceptions corresponding to various MAGMA errors:
_libmagma.magma_strerror.restype = ctypes.c_char_p
_libmagma.magma_strerror.argtypes = [ctypes.c_int]
# MAGMA below 1.4.0 uses "L" and "U" to select upper/lower triangular
# matrices, MAGMA 1.5+ uses numeric constants. This dict will be filled
# in magma_init() and will convert between the two modes accordingly
_uplo_conversion = {}
def magma_strerror(error):
    """
    Return string corresponding to specified MAGMA error code.
    """

    return _libmagma.magma_strerror(error)

class magmaError(Exception):
    try:
        __doc__ = magma_strerror(-100)
    except:
        pass
    pass

class MagmaError(Exception):
	def __init__(self, status, info=None):
		self._status = status
		self._info = info
		errstr = "%s (Code: %d)" % (magma_strerror(status), status)
		super(MagmaError,self).__init__(errstr)

class MagmaNotInitializedError(MagmaError):
    pass

class MagmaReinitializedError(MagmaError):
    pass

class MagmaNotSupportedError(MagmaError):
    pass

class MagmaIllegalValueError(MagmaError):
    pass

class MagmaIllegalValueError(MagmaError):
    pass

class MagmaNotFoundError(MagmaError):
    pass

class MagmaAllocationError(MagmaError):
    pass

class MagmaInternalLimitError(MagmaError):
    pass

class MagmaUnallocatedError(MagmaError):
    pass

class MagmaFilesystemError(MagmaError):
    pass

class MagmaUnexpectedError(MagmaError):
    pass

class MagmaSequenceFlushedError(MagmaError):
    pass

class MagmaHostAllocError(MagmaError):
    pass

class MagmaDeviceAllocError(MagmaError):
    pass

class MagmaCUDAStreamError(MagmaError):
    pass

class MagmaInvalidPtrError(MagmaError):
    pass

class MagmaUnknownError(MagmaError):
    pass

magmaExceptions = {
    -100: MagmaError,
    -101: MagmaNotInitializedError,
    -102: MagmaReinitializedError,
    -103: MagmaNotSupportedError,
    -104: MagmaIllegalValueError,
    -105: MagmaNotFoundError,
    -106: MagmaAllocationError,
    -107: MagmaInternalLimitError,
    -108: MagmaUnallocatedError,
    -109: MagmaFilesystemError,
    -110: MagmaUnexpectedError,
    -111: MagmaSequenceFlushedError,
    -112: MagmaHostAllocError,
    -113: MagmaDeviceAllocError,
    -114: MagmaCUDAStreamError,
    -115: MagmaInvalidPtrError,
    -116: MagmaUnknownError
}

_libmagma.magma_uplo_const.restype = ctypes.c_int
_libmagma.magma_uplo_const.argtypes = [ctypes.c_char_p]
def magmaCheckStatus(status):
    """
    Raise an exception corresponding to the specified MAGMA status code.
    """

    if status != 0:
        try:
            raise magmaExceptions[status](status)
        except KeyError:
            raise MagmaError(status)

# Utility functions:
_libmagma.magma_version.argtypes = [ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p]


def magma_version():
    """
    Get MAGMA version.
    """
    majv = ctypes.c_int()
    minv = ctypes.c_int()
    micv = ctypes.c_int()
    _libmagma.magma_version(ctypes.byref(majv),
        ctypes.byref(minv), ctypes.byref(micv))
    return (majv.value, minv.value, micv.value)

_libmagma.magma_init.restype = int
def magma_init():
    """
    Initialize MAGMA.
    """

    global _uplo_conversion
    status = _libmagma.magma_init()
    magmaCheckStatus(status)
    v = magma_version()
    if v >= (1, 5, 0):
        _uplo_conversion.update({"L": _libmagma.magma_uplo_const("L"),
                                 "l": _libmagma.magma_uplo_const("l"),
                                 "U": _libmagma.magma_uplo_const("U"),
                                 "u": _libmagma.magma_uplo_const("u")})
    else:
       _uplo_conversion.update({"L": "L", "l": "l", "U": "u", "u": "u"})

_libmagma.magma_finalize.restype = int
def magma_finalize():
    """
    Finalize MAGMA.
    """

    status = _libmagma.magma_finalize()
    magmaCheckStatus(status)

_libmagma.magma_getdevice_arch.restype = int
def magma_getdevice_arch():
    """
    Get device architecture.
    """

    return _libmagma.magma_getdevice_arch()

_libmagma.magma_getdevice.argtypes = [ctypes.c_void_p]
def magma_getdevice():
    """
    Get current device used by MAGMA.
    """

    dev = ctypes.c_int()
    _libmagma.magma_getdevice(ctypes.byref(dev))
    return dev.value

_libmagma.magma_setdevice.argtypes = [ctypes.c_int]
def magma_setdevice(dev):
    """
    Get current device used by MAGMA.
    """

    _libmagma.magma_setdevice(dev)

def magma_device_sync():
    """
    Synchronize device used by MAGMA.
    """

    _libmagma.magma_device_sync()

# BLAS routines

# ISAMAX, IDAMAX, ICAMAX, IZAMAX
_libmagma.magma_isamax.restype = int
_libmagma.magma_isamax.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_isamax(n, dx, incx):
    """
    Index of maximum magnitude element.
    """

    return _libmagma.magma_isamax(n, int(dx), incx)

_libmagma.magma_idamax.restype = int
_libmagma.magma_idamax.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_idamax(n, dx, incx):
    """
    Index of maximum magnitude element.
    """

    return _libmagma.magma_idamax(n, int(dx), incx)

_libmagma.magma_icamax.restype = int
_libmagma.magma_icamax.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_icamax(n, dx, incx):
    """
    Index of maximum magnitude element.
    """

    return _libmagma.magma_icamax(n, int(dx), incx)

_libmagma.magma_izamax.restype = int
_libmagma.magma_izamax.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_izamax(n, dx, incx):
    """
    Index of maximum magnitude element.
    """

    return _libmagma.magma_izamax(n, int(dx), incx)

# ISAMIN, IDAMIN, ICAMIN, IZAMIN
_libmagma.magma_isamin.restype = int
_libmagma.magma_isamin.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_isamin(n, dx, incx):
    """
    Index of minimum magnitude element.
    """

    return _libmagma.magma_isamin(n, int(dx), incx)

_libmagma.magma_idamin.restype = int
_libmagma.magma_idamin.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_idamin(n, dx, incx):
    """
    Index of minimum magnitude element.
    """

    return _libmagma.magma_idamin(n, int(dx), incx)

_libmagma.magma_icamin.restype = int
_libmagma.magma_icamin.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_icamin(n, dx, incx):
    """
    Index of minimum magnitude element.
    """

    return _libmagma.magma_icamin(n, int(dx), incx)

_libmagma.magma_izamin.restype = int
_libmagma.magma_izamin.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_izamin(n, dx, incx):
    """
    Index of minimum magnitude element.
    """

    return _libmagma.magma_izamin(n, int(dx), incx)

# SASUM, DASUM, SCASUM, DZASUM
_libmagma.magma_sasum.restype = int
_libmagma.magma_sasum.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_sasum(n, dx, incx):
    """
    Sum of absolute values of vector.
    """

    return _libmagma.magma_sasum(n, int(dx), incx)

_libmagma.magma_dasum.restype = int
_libmagma.magma_dasum.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_dasum(n, dx, incx):
    """
    Sum of absolute values of vector.
    """

    return _libmagma.magma_dasum(n, int(dx), incx)

_libmagma.magma_scasum.restype = int
_libmagma.magma_scasum.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_scasum(n, dx, incx):
    """
    Sum of absolute values of vector.
    """

    return _libmagma.magma_scasum(n, int(dx), incx)

_libmagma.magma_dzasum.restype = int
_libmagma.magma_dzasum.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_dzasum(n, dx, incx):
    """
    Sum of absolute values of vector.
    """

    return _libmagma.magma_dzasum(n, int(dx), incx)

# SAXPY, DAXPY, CAXPY, ZAXPY
_libmagma.magma_saxpy.restype = int
_libmagma.magma_saxpy.argtypes = [ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_saxpy(n, alpha, dx, incx, dy, incy):
    """
    Vector addition.
    """

    _libmagma.magma_saxpy(n, alpha, int(dx), incx, int(dy), incy)

_libmagma.magma_daxpy.restype = int
_libmagma.magma_daxpy.argtypes = [ctypes.c_int,
                                  ctypes.c_double,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_daxpy(n, alpha, dx, incx, dy, incy):
    """
    Vector addition.
    """

    _libmagma.magma_daxpy(n, alpha, int(dx), incx, int(dy), incy)

_libmagma.magma_caxpy.restype = int
_libmagma.magma_caxpy.argtypes = [ctypes.c_int,
                                  cuda.cuFloatComplex,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_caxpy(n, alpha, dx, incx, dy, incy):
    """
    Vector addition.
    """

    _libmagma.magma_caxpy(n, ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                              alpha.imag)),
                          int(dx), incx, int(dy), incy)

_libmagma.magma_zaxpy.restype = int
_libmagma.magma_zaxpy.argtypes = [ctypes.c_int,
                                  cuda.cuDoubleComplex,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_zaxpy(n, alpha, dx, incx, dy, incy):
    """
    Vector addition.
    """

    _libmagma.magma_zaxpy(n, ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                               alpha.imag)),
                          int(dx), incx, int(dy), incy)

# SCOPY, DCOPY, CCOPY, ZCOPY
_libmagma.magma_scopy.restype = int
_libmagma.magma_scopy.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_scopy(n, dx, incx, dy, incy):
    """
    Vector copy.
    """

    _libmagma.magma_scopy(n, int(dx), incx, int(dy), incy)

_libmagma.magma_dcopy.restype = int
_libmagma.magma_dcopy.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_dcopy(n, dx, incx, dy, incy):
    """
    Vector copy.
    """

    _libmagma.magma_dcopy(n, int(dx), incx, int(dy), incy)

_libmagma.magma_ccopy.restype = int
_libmagma.magma_ccopy.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_ccopy(n, dx, incx, dy, incy):
    """
    Vector copy.
    """

    _libmagma.magma_ccopy(n, int(dx), incx, int(dy), incy)

_libmagma.magma_zcopy.restype = int
_libmagma.magma_zcopy.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_zcopy(n, dx, incx, dy, incy):
    """
    Vector copy.
    """

    _libmagma.magma_zcopy(n, int(dx), incx, int(dy), incy)

# SDOT, DDOT, CDOTU, CDOTC, ZDOTU, ZDOTC
_libmagma.magma_sdot.restype = ctypes.c_float
_libmagma.magma_sdot.argtypes = [ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def magma_sdot(n, dx, incx, dy, incy):
    """
    Vector dot product.
    """

    return _libmagma.magma_sdot(n, int(dx), incx, int(dy), incy)

_libmagma.magma_ddot.restype = ctypes.c_double
_libmagma.magma_ddot.argtypes = [ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def magma_ddot(n, dx, incx, dy, incy):
    """
    Vector dot product.
    """

    return _libmagma.magma_ddot(n, int(dx), incx, int(dy), incy)

_libmagma.magma_cdotc.restype = cuda.cuFloatComplex
_libmagma.magma_cdotc.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_cdotc(n, dx, incx, dy, incy):
    """
    Vector dot product.
    """

    return _libmagma.magma_cdotc(n, int(dx), incx, int(dy), incy)

_libmagma.magma_cdotu.restype = cuda.cuFloatComplex
_libmagma.magma_cdotu.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_cdotu(n, dx, incx, dy, incy):
    """
    Vector dot product.
    """

    return _libmagma.magma_cdotu(n, int(dx), incx, int(dy), incy)

_libmagma.magma_zdotc.restype = cuda.cuDoubleComplex
_libmagma.magma_zdotc.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_zdotc(n, dx, incx, dy, incy):
    """
    Vector dot product.
    """

    return _libmagma.magma_zdotc(n, int(dx), incx, int(dy), incy)

_libmagma.magma_zdotu.restype = cuda.cuDoubleComplex
_libmagma.magma_zdotu.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_zdotu(n, dx, incx, dy, incy):
    """
    Vector dot product.
    """

    return _libmagma.magma_zdotu(n, int(dx), incx, int(dy), incy)

# SNRM2, DNRM2, SCNRM2, DZNRM2
_libmagma.magma_snrm2.restype = ctypes.c_float
_libmagma.magma_snrm2.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_snrm2(n, dx, incx):
    """
    Euclidean norm (2-norm) of vector.
    """

    return _libmagma.magma_snrm2(n, int(dx), incx)

_libmagma.magma_dnrm2.restype = ctypes.c_double
_libmagma.magma_dnrm2.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_dnrm2(n, dx, incx):
    """
    Euclidean norm (2-norm) of vector.
    """

    return _libmagma.magma_dnrm2(n, int(dx), incx)

_libmagma.magma_scnrm2.restype = ctypes.c_float
_libmagma.magma_scnrm2.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_scnrm2(n, dx, incx):
    """
    Euclidean norm (2-norm) of vector.
    """

    return _libmagma.magma_scnrm2(n, int(dx), incx)

_libmagma.magma_dznrm2.restype = ctypes.c_double
_libmagma.magma_dznrm2.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_dznrm2(n, dx, incx):
    """
    Euclidean norm (2-norm) of vector.
    """

    return _libmagma.magma_dznrm2(n, int(dx), incx)

# SROT, DROT, CROT, CSROT, ZROT, ZDROT
_libmagma.magma_srot.argtypes = [ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_float,
                                 ctypes.c_float]
def magma_srot(n, dx, incx, dy, incy, dc, ds):
    """
    Apply a rotation to vectors.
    """

    _libmagma.magma_srot(n, int(dx), incx, int(dy), incy, dc, ds)

# SROTM, DROTM
_libmagma.magma_srotm.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p]
def magma_srotm(n, dx, incx, dy, incy, param):
    """
    Apply a real modified Givens rotation.
    """

    _libmagma.magma_srotm(n, int(dx), incx, int(dy), incy, param)

# SROTMG, DROTMG
_libmagma.magma_srotmg.argtypes = [ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_srotmg(d1, d2, x1, y1, param):
    """
    Construct a real modified Givens rotation matrix.
    """

    _libmagma.magma_srotmg(int(d1), int(d2), int(x1), int(y1), param)

# SSCAL, DSCAL, CSCAL, CSCAL, CSSCAL, ZSCAL, ZDSCAL
_libmagma.magma_sscal.argtypes = [ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_sscal(n, alpha, dx, incx):
    """
    Scale a vector by a scalar.
    """

    _libmagma.magma_sscal(n, alpha, int(dx), incx)

# SSWAP, DSWAP, CSWAP, ZSWAP
_libmagma.magma_sswap.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_sswap(n, dA, ldda, dB, lddb):
    """
    Swap vectors.
    """

    _libmagma.magma_sswap(n, int(dA), ldda, int(dB), lddb)

# SGEMV, DGEMV, CGEMV, ZGEMV
_libmagma.magma_sgemv.argtypes = [ctypes.c_char,
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
def magma_sgemv(trans, m, n, alpha, dA, ldda, dx, incx, beta,
                dy, incy):
    """
    Matrix-vector product for general matrix.
    """

    _libmagma.magma_sgemv(trans, m, n, alpha, int(dA), ldda, dx, incx,
                          beta, int(dy), incy)

# SGER, DGER, CGERU, CGERC, ZGERU, ZGERC
_libmagma.magma_sger.argtypes = [ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_float,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def magma_sger(m, n, alpha, dx, incx, dy, incy, dA, ldda):
    """
    Rank-1 operation on real general matrix.
    """

    _libmagma.magma_sger(m, n, alpha, int(dx), incx, int(dy), incy,
                         int(dA), ldda)

# SSYMV, DSYMV, CSYMV, ZSYMV
_libmagma.magma_ssymv.argtypes = [ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_ssymv(uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy):
    _libmagma.magma_ssymv(uplo, n, alpha, int(dA), ldda, int(dx), incx, beta,
                          int(dy), incy)

# SSYR, DSYR, CSYR, ZSYR
_libmagma.magma_ssyr.argtypes = [ctypes.c_char,
                                 ctypes.c_int,
                                 ctypes.c_float,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def magma_ssyr(uplo, n, alpha, dx, incx, dA, ldda):
    _libmagma.magma_ssyr(uplo, n, alpha, int(dx), incx, int(dA), ldda)

# SSYR2, DSYR2, CSYR2, ZSYR2
_libmagma.magma_ssyr2.argtypes = [ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_ssyr2(uplo, n, alpha, dx, incx, dy, incy, dA, ldda):
    _libmagma.magma_ssyr2(uplo, n, alpha, int(dx), incx,
                          int(dy), incy, int(dA), ldda)

# STRMV, DTRMV, CTRMV, ZTRMV
_libmagma.magma_strmv.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_strmv(uplo, trans, diag, n,
                dA, ldda, dx, incx):
    _libmagma.magma_strmv(uplo, trans, diag, n,
                          int(dA), ldda, int(dx), incx)

# STRSV, DTRSV, CTRSV, ZTRSV
_libmagma.magma_strsv.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_strsv(uplo, trans, diag, n,
                dA, ldda, dx, incx):
    _libmagma.magma_strsv(uplo, trans, diag, n,
                          int(dA), ldda, int(dx), incx)

# SGEMM, DGEMM, CGEMM, ZGEMM
_libmagma.magma_sgemm.argtypes = [ctypes.c_char,
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
def magma_sgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta,
                dC, lddc):
    _libmagma.magma_sgemm(transA, transB, m, n, k, alpha,
                          int(dA), ldda, int(dB), lddb,
                          beta, int(dC), lddc)

_libmagma.magma_zgemm.argtypes = [ctypes.c_char,
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
def magma_zgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta,
                dC, lddc):
    _libmagma.magma_zgemm(transA, transB, m, n, k, alpha,
                          int(dA), ldda, int(dB), lddb,
                          beta, int(dC), lddc)


# SSYMM, DSYMM, CSYMM, ZSYMM
_libmagma.magma_ssymm.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
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
def magma_ssymm(side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta,
                dC, lddc):
    _libmagma.magma_ssymm(side, uplo, m, n, alpha,
                          int(dA), ldda, int(dB), lddb,
                          beta, int(dC), lddc)

# SSYRK, DSYRK, CSYRK, ZSYRK
_libmagma.magma_ssyrk.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_ssyrk(uplo, trans, n, k, alpha, dA, ldda, beta,
                dC, lddc):
    _libmagma.magma_ssyrk(uplo, trans, n, k, alpha,
                          int(dA), ldda, beta, int(dC), lddc)

# SSYR2K, DSYR2K, CSYR2K, ZSYR2K
_libmagma.magma_ssyr2k.argtypes = [ctypes.c_char,
                                   ctypes.c_char,
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
def magma_ssyr2k(uplo, trans, n, k, alpha, dA, ldda,
                 dB, lddb, beta, dC, lddc):
    _libmagma.magma_ssyr2k(uplo, trans, n, k, alpha,
                           int(dA), ldda, int(dB), lddb,
                           beta, int(dC), lddc)

# STRMM, DTRMM, CTRMM, ZTRMM
_libmagma.magma_strmm.argtypes = [ctypes.c_char,
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
def magma_strmm(side, uplo, trans, diag, m, n, alpha, dA, ldda,
                dB, lddb):
    _libmagma.magma_strmm(uplo, trans, diag, m, n, alpha,
                          int(dA), ldda, int(dB), lddb)

# STRSM, DTRSM, CTRSM, ZTRSM
_libmagma.magma_strsm.argtypes = [ctypes.c_char,
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
def magma_strsm(side, uplo, trans, diag, m, n, alpha, dA, ldda,
                dB, lddb):
    _libmagma.magma_strsm(uplo, trans, diag, m, n, alpha,
                          int(dA), ldda, int(dB), lddb)


# Auxiliary routines:
_libmagma.magma_vec_const.restype = int
_libmagma.magma_vec_const.argtypes = [ctypes.c_char]
def magma_vec_const(job):
    return _libmagma.magma_vec_const(job)

_libmagma.magma_get_spotrf_nb.restype = int
_libmagma.magma_get_spotrf_nb.argtypes = [ctypes.c_int]
def magma_get_spotrf_nb(m):
    return _libmagma.magma_get_spotrf_nb(m)

_libmagma.magma_get_sgetrf_nb.restype = int
_libmagma.magma_get_sgetrf_nb.argtypes = [ctypes.c_int]
def magma_get_sgetrf_nb(m):
    return _libmagma.magma_get_sgetrf_nb(m)

_libmagma.magma_get_sgetri_nb.restype = int
_libmagma.magma_get_sgetri_nb.argtypes = [ctypes.c_int]
def magma_get_sgetri_nb(m):
    return _libmagma.magma_get_sgetri_nb(m)

_libmagma.magma_get_sgeqp3_nb.restype = int
_libmagma.magma_get_sgeqp3_nb.argtypes = [ctypes.c_int]
def magma_get_sgeqp3_nb(m):
    return _libmagma.magma_get_sgeqp3_nb(m)

_libmagma.magma_get_sgeqrf_nb.restype = int
_libmagma.magma_get_sgeqrf_nb.argtypes = [ctypes.c_int]
def magma_get_sgeqrf_nb(m):
    return _libmagma.magma_get_sgeqrf_nb(m)

_libmagma.magma_get_sgeqlf_nb.restype = int
_libmagma.magma_get_sgeqlf_nb.argtypes = [ctypes.c_int]
def magma_get_sgeqlf_nb(m):
    return _libmagma.magma_get_sgeqlf_nb(m)

_libmagma.magma_get_sgehrd_nb.restype = int
_libmagma.magma_get_sgehrd_nb.argtypes = [ctypes.c_int]
def magma_get_sgehrd_nb(m):
    return _libmagma.magma_get_sgehrd_nb(m)

_libmagma.magma_get_ssytrd_nb.restype = int
_libmagma.magma_get_ssytrd_nb.argtypes = [ctypes.c_int]
def magma_get_ssytrd_nb(m):
    return _libmagma.magma_get_ssytrd_nb(m)

_libmagma.magma_get_sgelqf_nb.restype = int
_libmagma.magma_get_sgelqf_nb.argtypes = [ctypes.c_int]
def magma_get_sgelqf_nb(m):
    return _libmagma.magma_get_sgelqf_nb(m)

_libmagma.magma_get_sgebrd_nb.restype = int
_libmagma.magma_get_sgebrd_nb.argtypes = [ctypes.c_int]
def magma_get_sgebrd_nb(m):
    return _libmagma.magma_get_sgebrd_nb(m)

_libmagma.magma_get_ssygst_nb.restype = int
_libmagma.magma_get_ssygst_nb.argtypes = [ctypes.c_int]
def magma_get_ssygst_nb(m):
    return _libmagma.magma_get_ssgyst_nb(m)

_libmagma.magma_get_sgesvd_nb.restype = int
_libmagma.magma_get_sgesvd_nb.argtypes = [ctypes.c_int]
def magma_get_sgesvd_nb(m):
    return _libmagma.magma_get_sgesvd_nb(m)

_libmagma.magma_get_dgesvd_nb.restype = int
_libmagma.magma_get_dgesvd_nb.argtypes = [ctypes.c_int]
def magma_get_dgesvd_nb(m):
    return _libmagma.magma_get_dgesvd_nb(m)

_libmagma.magma_get_cgesvd_nb.restype = int
_libmagma.magma_get_cgesvd_nb.argtypes = [ctypes.c_int]
def magma_get_cgesvd_nb(m):
    return _libmagma.magma_get_cgesvd_nb(m)

_libmagma.magma_get_zgesvd_nb.restype = int
_libmagma.magma_get_zgesvd_nb.argtypes = [ctypes.c_int]
def magma_get_zgesvd_nb(m):
    return _libmagma.magma_get_zgesvd_nb(m)

_libmagma.magma_get_ssygst_nb_m.restype = int
_libmagma.magma_get_ssygst_nb_m.argtypes = [ctypes.c_int]
def magma_get_ssygst_nb_m(m):
    return _libmagma.magma_get_ssgyst_nb_m(m)

_libmagma.magma_get_sbulge_nb.restype = int
_libmagma.magma_get_sbulge_nb.argtypes = [ctypes.c_int]
def magma_get_sbulge_nb(m):
    return _libmagma.magma_get_sbulge_nb(m)

_libmagma.magma_get_sbulge_nb_mgpu.restype = int
_libmagma.magma_get_sbulge_nb_mgpu.argtypes = [ctypes.c_int]
def magma_get_sbulge_nb_mgpu(m):
    return _libmagma.magma_get_sbulge_nb_mgpu(m)

# LAPACK routines

# SGEBRD, DGEBRD, CGEBRD, ZGEBRD
_libmagma.magma_sgebrd.restype = int
_libmagma.magma_sgebrd.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p]
def magma_sgebrd(m, n, A, lda, d, e, tauq, taup, work, lwork, info):
    """
    Reduce matrix to bidiagonal form.
    """

    status = _libmagma.magma_sgebrd.argtypes(m, n, int(A), lda,
                                             int(d), int(e),
                                             int(tauq), int(taup),
                                             int(work), int(lwork),
                                             int(info))
    magmaCheckStatus(status)

# SGEHRD2, DGEHRD2, CGEHRD2, ZGEHRD2
_libmagma.magma_sgehrd2.restype = int
_libmagma.magma_sgehrd2.argtypes = [ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p]
def magma_sgehrd2(n, ilo, ihi, A, lda, tau,
                  work, lwork, info):
    """
    Reduce matrix to upper Hessenberg form.
    """

    status = _libmagma.magma_sgehrd2(n, ilo, ihi, int(A), lda,
                                     int(tau), int(work),
                                     lwork, int(info))
    magmaCheckStatus(status)

# SGEHRD, DGEHRD, CGEHRD, ZGEHRD
_libmagma.magma_sgehrd.restype = int
_libmagma.magma_sgehrd.argtypes = [ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p]
def magma_sgehrd(n, ilo, ihi, A, lda, tau,
                 work, lwork, dT, info):
    """
    Reduce matrix to upper Hessenberg form (fast algorithm).
    """

    status = _libmagma.magma_sgehrd(n, ilo, ihi, int(A), lda,
                                    int(tau), int(work),
                                    lwork, int(dT), int(info))
    magmaCheckStatus(status)

# SGELQF, DGELQF, CGELQF, ZGELQF
_libmagma.magma_sgelqf.restype = int
_libmagma.magma_sgelqf.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p]
def magma_sgelqf(m, n, A, lda, tau, work, lwork, info):

    """
    LQ factorization.
    """

    status = _libmagma.magma_sgelqf(m, n, int(A), lda,
                                    int(tau), int(work),
                                    lwork, int(info))
    magmaCheckStatus(status)

# SGEQRF, DGEQRF, CGEQRF, ZGEQRF
_libmagma.magma_sgeqrf.restype = int
_libmagma.magma_sgeqrf.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p]
def magma_sgeqrf(m, n, A, lda, tau, work, lwork, info):

    """
    QR factorization.
    """

    status = _libmagma.magma_sgeqrf(m, n, int(A), lda,
                                    int(tau), int(work),
                                    lwork, int(info))
    magmaCheckStatus(status)

# SGEQRF4, DGEQRF4, CGEQRF4, ZGEQRF4
_libmagma.magma_sgeqrf4.restype = int
_libmagma.magma_sgeqrf4.argtypes = [ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p]
def magma_sgeqrf4(num_gpus, m, n, a, lda, tau, work, lwork, info):

    """

    """

    status = _libmagma.magma_sgeqrf4(num_gpus, m, n, int(a), lda,
                                    int(tau), int(work),
                                    lwork, int(info))
    magmaCheckStatus(status)

# SGEQRF, DGEQRF, CGEQRF, ZGEQRF (ooc)
_libmagma.magma_sgeqrf_ooc.restype = int
_libmagma.magma_sgeqrf_ooc.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def magma_sgeqrf_ooc(m, n, A, lda, tau, work, lwork, info):

    """
    QR factorization (ooc).
    """

    status = _libmagma.magma_sgeqrf_ooc(m, n, int(A), lda,
                                        int(tau), int(work),
                                        lwork, int(info))
    magmaCheckStatus(status)

# SGESV, DGESV, CGESV, ZGESV
_libmagma.magma_sgesv.restype = int
_libmagma.magma_sgesv.argtypes = [ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p]
def magma_sgesv(n, nhrs, A, lda, ipiv, B, ldb, info):

    """
    Solve system of linear equations.
    """

    status = _libmagma.magma_sgesv(n, nhrs, int(A), lda,
                                   int(ipiv), int(B),
                                   ldb, int(info))
    magmaCheckStatus(status)

# SGETRF, DGETRF, CGETRF, ZGETRF
_libmagma.magma_sgetrf.restype = int
_libmagma.magma_sgetrf.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_sgetrf(m, n, A, lda, ipiv, info):

    """
    LU factorization.
    """

    status = _libmagma.magma_sgetrf(m, n, int(A), lda,
                                    int(ipiv), int(info))
    magmaCheckStatus(status)

## SGETRF2, DGETRF2, CGETRF2, ZGETRF2
#_libmagma.magma_sgetrf2.restype = int
#_libmagma.magma_sgetrf2.argtypes = [ctypes.c_int,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_void_p]
#def magma_sgetrf2(m, n, A, lda, ipiv, info):
#
#    """
#    LU factorization (multi-GPU).
#    """
#
#    status = _libmagma.magma_sgetrf2(m, n, int(A), lda,
#                                    int(ipiv), int(info))
#    magmaCheckStatus(status)

# SGEEV, DGEEV, CGEEV, ZGEEV
_libmagma.magma_sgeev.restype = int
_libmagma.magma_sgeev.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p]
def magma_sgeev(jobvl, jobvr, n, a, lda,
                w, vl, ldvl, vr, ldvr, work, lwork, rwork, info):

    """
    Compute eigenvalues and eigenvectors.
    """

    status = _libmagma.magma_sgeev(jobvl, jobvr, n, int(a), lda,
                                   int(w), int(vl), ldvl, int(vr), ldvr,
                                   int(work), lwork, int(rwork), int(info))
    magmaCheckStatus(status)

# SGESVD, DGESVD, CGESVD, ZGESVD
_libmagma.magma_sgesvd.restype = int
_libmagma.magma_sgesvd.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p]
def magma_sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 info):
    """
    SVD decomposition.
    """

    status = _libmagma.magma_sgesvd(jobu, jobvt, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(info))
    magmaCheckStatus(status)

_libmagma.magma_dgesvd.restype = int
_libmagma.magma_dgesvd.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p]
def magma_dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 info):
    """
    SVD decomposition.
    """

    status = _libmagma.magma_dgesvd(jobu, jobvt, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(info))
    magmaCheckStatus(status)

_libmagma.magma_cgesvd.restype = int
_libmagma.magma_cgesvd.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p]
def magma_cgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 rwork, info):
    """
    SVD decomposition.
    """

    status = _libmagma.magma_cgesvd(jobu, jobvt, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(rwork),
                                    int(info))
    magmaCheckStatus(status)

_libmagma.magma_zgesvd.restype = int
_libmagma.magma_zgesvd.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_zgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 rwork, info):
    """
    SVD decomposition.
    """

    status = _libmagma.magma_zgesvd(jobu, jobvt, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(rwork), int(info))
    magmaCheckStatus(status)

# SGESDD, DGESDD, CGESDD, ZGESDD
_libmagma.magma_sgesdd.restype = int
_libmagma.magma_sgesdd.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_sgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 iwork, info):
    """
    SDD decomposition.
    """

    status = _libmagma.magma_sgesdd(jobz, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(iwork), int(info))
    magmaCheckStatus(status)

_libmagma.magma_dgesdd.restype = int
_libmagma.magma_dgesdd.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 iwork, info):
    """
    SDD decomposition.
    """

    status = _libmagma.magma_dgesdd(jobz, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(iwork), int(info))
    magmaCheckStatus(status)

_libmagma.magma_cgesdd.restype = int
_libmagma.magma_cgesdd.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_cgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 rwork, iwork, info):
    """
    SDD decomposition.
    """

    status = _libmagma.magma_cgesdd(jobz, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(rwork), int(iwork), int(info))
    magmaCheckStatus(status)

_libmagma.magma_zgesdd.restype = int
_libmagma.magma_zgesdd.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_zgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 rwork, iwork, info):
    """
    SDD decomposition.
    """

    status = _libmagma.magma_zgesdd(jobz, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(rwork), int(iwork), int(info))
    magmaCheckStatus(status)



# SPOSV, DPOSV, CPOSV, ZPOSV
_libmagma.magma_sposv_gpu.restype = int
_libmagma.magma_sposv_gpu.argtypes = [ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p]
def magma_sposv_gpu(uplo, n, nhrs, a_gpu, lda, b_gpu, ldb):
    """
    Solve linear system with positive semidefinite coefficient matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_sposv_gpu(uplo, n, nhrs, int(a_gpu), lda,
                                       int(b_gpu), ldb, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dposv_gpu.restype = int
_libmagma.magma_dposv_gpu.argtypes = _libmagma.magma_sposv_gpu.argtypes
def magma_dposv_gpu(uplo, n, nhrs, a_gpu, lda, b_gpu, ldb):
    """
    Solve linear system with positive semidefinite coefficient matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_dposv_gpu(uplo, n, nhrs, int(a_gpu), lda,
                                       int(b_gpu), ldb, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_cposv_gpu.restype = int
_libmagma.magma_cposv_gpu.argtypes = _libmagma.magma_sposv_gpu.argtypes
def magma_cposv_gpu(uplo, n, nhrs, a_gpu, lda, b_gpu, ldb):
    """
    Solve linear system with positive semidefinite coefficient matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_cposv_gpu(uplo, n, nhrs, int(a_gpu), lda,
                                       int(b_gpu), ldb, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_zposv_gpu.restype = int
_libmagma.magma_zposv_gpu.argtypes = _libmagma.magma_sposv_gpu.argtypes
def magma_zposv_gpu(uplo, n, nhrs, a_gpu, lda, b_gpu, ldb):
    """
    Solve linear system with positive semidefinite coefficient matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_zposv_gpu(uplo, n, nhrs, int(a_gpu), lda,
                                       int(b_gpu), ldb, ctypes.byref(info))
    magmaCheckStatus(status)


# SGESV, DGESV, CGESV, ZGESV
_libmagma.magma_sgesv_gpu.restype = int
_libmagma.magma_sgesv_gpu.argtypes = [ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p]
def magma_sgesv_gpu(n, nhrs, A, lda, ipiv, B, ldb):
    """
    Solve system of linear equations.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_sgesv_gpu(n, nhrs, int(A), lda,
                                   int(ipiv), int(B),
                                   ldb, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_dgesv_gpu.restype = int
_libmagma.magma_dgesv_gpu.argtypes = _libmagma.magma_sgesv_gpu.argtypes
def magma_dgesv_gpu(n, nhrs, A, lda, ipiv, B, ldb):
    """
    Solve system of linear equations.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_dgesv_gpu(n, nhrs, int(A), lda,
                                   int(ipiv), int(B),
                                   ldb, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_cgesv_gpu.restype = int
_libmagma.magma_cgesv_gpu.argtypes = _libmagma.magma_sgesv_gpu.argtypes
def magma_cgesv_gpu(n, nhrs, A, lda, ipiv, B, ldb):
    """
    Solve system of linear equations.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_cgesv_gpu(n, nhrs, int(A), lda,
                                   int(ipiv), int(B),
                                   ldb, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_zgesv_gpu.restype = int
_libmagma.magma_zgesv_gpu.argtypes = _libmagma.magma_sgesv_gpu.argtypes
def magma_zgesv_gpu(n, nhrs, A, lda, ipiv, B, ldb):
    """
    Solve system of linear equations.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_zgesv_gpu(n, nhrs, int(A), lda,
                                   int(ipiv), int(B),
                                   ldb, ctypes.byref(info))
    magmaCheckStatus(status)


# SPOTRF, DPOTRF, CPOTRF, ZPOTRF
_libmagma.magma_spotrf_gpu.restype = int
_libmagma.magma_spotrf_gpu.argtypes = [ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p]
def magma_spotrf_gpu(uplo, n, A, lda):
    """
    Cholesky factorization of positive symmetric matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_spotrf_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_dpotrf_gpu.restype = int
_libmagma.magma_dpotrf_gpu.argtypes = _libmagma.magma_spotrf_gpu.argtypes
def magma_dpotrf_gpu(uplo, n, A, lda):
    """
    Cholesky factorization of positive symmetric matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_dpotrf_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_cpotrf_gpu.restype = int
_libmagma.magma_cpotrf_gpu.argtypes = _libmagma.magma_spotrf_gpu.argtypes
def magma_cpotrf_gpu(uplo, n, A, lda):
    """
    Cholesky factorization of positive symmetric matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_cpotrf_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_zpotrf_gpu.restype = int
_libmagma.magma_zpotrf_gpu.argtypes = _libmagma.magma_zpotrf_gpu.argtypes
def magma_zpotrf_gpu(uplo, n, A, lda):
    """
    Cholesky factorization of positive symmetric matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_zpotrf_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)


# SPOTRI, DPOTRI, CPOTRI, ZPOTRI
_libmagma.magma_spotri_gpu.restype = int
_libmagma.magma_spotri_gpu.argtypes = [ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p]
def magma_spotri_gpu(uplo, n, A, lda):
    """
    Inverse using the Cholesky factorization of positive symmetric matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_spotri_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_dpotri_gpu.restype = int
_libmagma.magma_dpotri_gpu.argtypes = _libmagma.magma_spotri_gpu.argtypes
def magma_dpotri_gpu(uplo, n, A, lda):
    """
    Inverse using the Cholesky factorization of positive symmetric matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_dpotri_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_cpotri_gpu.restype = int
_libmagma.magma_cpotri_gpu.argtypes = _libmagma.magma_spotri_gpu.argtypes
def magma_cpotri_gpu(uplo, n, A, lda):
    """
    Inverse using the Cholesky factorization of positive symmetric matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_cpotri_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_zpotri_gpu.restype = int
_libmagma.magma_zpotri_gpu.argtypes = _libmagma.magma_spotri_gpu.argtypes
def magma_zpotri_gpu(uplo, n, A, lda):
    """
    Inverse using the Cholesky factorization of positive symmetric matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_zpotri_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)


# SGETRF, DGETRF, CGETRF, ZGETRF
_libmagma.magma_sgetrf_gpu.restype = int
_libmagma.magma_sgetrf_gpu.argtypes = [ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p]
def magma_sgetrf_gpu(n, m, A, lda, ipiv):
    """
    LU factorization.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_sgetrf_gpu(n, m, int(A), lda,
                                   int(ipiv), ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_dgetrf_gpu.restype = int
_libmagma.magma_dgetrf_gpu.argtypes = _libmagma.magma_sgetrf_gpu.argtypes
def magma_dgetrf_gpu(n, m, A, lda, ipiv):
    """
    LU factorization.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_dgetrf_gpu(n, m, int(A), lda,
                                   int(ipiv), ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_cgetrf_gpu.restype = int
_libmagma.magma_cgetrf_gpu.argtypes = _libmagma.magma_sgetrf_gpu.argtypes
def magma_cgetrf_gpu(n, m, A, lda, ipiv):
    """
    LU factorization.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_cgetrf_gpu(n, m, int(A), lda,
                                   int(ipiv), ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_zgetrf_gpu.restype = int
_libmagma.magma_zgetrf_gpu.argtypes = _libmagma.magma_sgetrf_gpu.argtypes
def magma_zgetrf_gpu(n, m, A, lda, ipiv):
    """
    LU factorization.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_zgetrf_gpu(n, m, int(A), lda,
                                   int(ipiv), ctypes.byref(info))
    magmaCheckStatus(status)
