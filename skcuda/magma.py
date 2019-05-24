#!/usr/bin/env python

"""
Python interface to MAGMA toolkit.
"""

from __future__ import absolute_import, division, print_function

import sys
import ctypes
import atexit
import numpy as np

from . import cuda

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

c_int_type = ctypes.c_longlong

# Exceptions corresponding to various MAGMA errors:
_libmagma.magma_strerror.restype = ctypes.c_char_p
_libmagma.magma_strerror.argtypes = [c_int_type]
def magma_strerror(error):
    """
    Return string corresponding to specified MAGMA error code.
    """

    return _libmagma.magma_strerror(error)

class MagmaError(Exception):
    def __init__(self, status, info=None):
        self._status = status
        self._info = info
        errstr = "%s (Code: %d)" % (magma_strerror(status), status)
        super(MagmaError,self).__init__(errstr)


def magmaCheckStatus(status):
    """
    Raise an exception corresponding to the specified MAGMA status code.
    """

    if status != 0:
        raise MagmaError(status)

# Utility functions:
_libmagma.magma_version.argtypes = [ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p]


def magma_version():
    """
    Get MAGMA version.
    """
    majv = c_int_type()
    minv = c_int_type()
    micv = c_int_type()
    _libmagma.magma_version(ctypes.byref(majv),
        ctypes.byref(minv), ctypes.byref(micv))
    return (majv.value, minv.value, micv.value)

# MAGMA below 1.4.0 uses LAPACK-style char constants, while MAGMA 1.5+ uses
# numeric constants. These dicts are filled in magma_init() and can convert
# between the two modes accordingly:
_bool_conversion = {}
_order_conversion = {}
_trans_conversion = {}
_uplo_conversion = {}
_diag_conversion = {}
_side_conversion = {}
_norm_conversion = {}
_dist_conversion = {}
_sym_conversion = {}
_pack_conversion = {}
_vec_conversion = {}
_range_conversion = {}
_vect_conversion = {}
_direct_conversion = {}
_storev_conversion = {}


_libmagma.magma_bool_const.restype = c_int_type
_libmagma.magma_bool_const.argtypes = [ctypes.c_char]
_libmagma.magma_order_const.restype = c_int_type
_libmagma.magma_order_const.argtypes = [ctypes.c_char]
_libmagma.magma_norm_const.restype = c_int_type
_libmagma.magma_norm_const.argtypes = [ctypes.c_char]
_libmagma.magma_dist_const.restype = c_int_type
_libmagma.magma_dist_const.argtypes = [ctypes.c_char]
_libmagma.magma_sym_const.restype = c_int_type
_libmagma.magma_sym_const.argtypes = [ctypes.c_char]
_libmagma.magma_pack_const.restype = c_int_type
_libmagma.magma_pack_const.argtypes = [ctypes.c_char]
_libmagma.magma_vect_const.restype = c_int_type
_libmagma.magma_vect_const.argtypes = [ctypes.c_char]
_libmagma.magma_range_const.restype = c_int_type
_libmagma.magma_range_const.argtypes = [ctypes.c_char]
_libmagma.magma_direct_const.restype = c_int_type
_libmagma.magma_direct_const.argtypes = [ctypes.c_char]
_libmagma.magma_storev_const.restype = c_int_type
_libmagma.magma_storev_const.argtypes = [ctypes.c_char]


_libmagma.magma_vec_const.restype = c_int_type
_libmagma.magma_vec_const.argtypes = [ctypes.c_char]
_libmagma.magma_uplo_const.restype = c_int_type
_libmagma.magma_uplo_const.argtypes = [ctypes.c_char]

_libmagma.magma_side_const.restype = c_int_type
_libmagma.magma_side_const.argtypes = [ctypes.c_char]
_libmagma.magma_trans_const.restype = c_int_type
_libmagma.magma_trans_const.argtypes = [ctypes.c_char]
_libmagma.magma_diag_const.restype = c_int_type
_libmagma.magma_diag_const.argtypes = [ctypes.c_char]

_libmagma.magma_init.restype = int
def magma_init():
    """
    Initialize MAGMA.
    """

    global _bool_conversion
    global _order_conversion
    global _trans_conversion
    global _uplo_conversion
    global _diag_conversion
    global _side_conversion
    global _norm_conversion
    global _dist_conversion
    global _sym_conversion
    global _pack_conversion
    global _vec_conversion
    global _range_conversion
    global _vect_conversion
    global _direct_conversion
    global _storev_conversion
    status = _libmagma.magma_init()
    magmaCheckStatus(status)
    v = magma_version()
    if v >= (1, 5, 0):
        for c in [b'n', b'N', b'y', b'Y']:
            _bool_conversion.update({c: _libmagma.magma_bool_const(c)})
            _bool_conversion.update({c.decode(): _libmagma.magma_bool_const(c)})
        for c in [b'r', b'R', b'c', b'C']:
            _order_conversion.update({c: _libmagma.magma_order_const(c)})
            _order_conversion.update({c.decode(): _libmagma.magma_order_const(c)})
        for c in [b'O', b'o', b'1', b'2', b'F', b'f', b'E', b'e', b'I', b'i',b'M',b'm']:
            _norm_conversion.update({c: _libmagma.magma_norm_const(c)})
            _norm_conversion.update({c.decode(): _libmagma.magma_norm_const(c)})
        for c in [b'U', b'u', b'S', b's', b'N', b'n']:
            _dist_conversion.update({c: _libmagma.magma_dist_const(c)})
            _dist_conversion.update({c.decode(): _libmagma.magma_dist_const(c)})
        for c in [b'H', b'h', b'S', b's', b'N', b'n', b'P', b'p']:
            _sym_conversion.update({c: _libmagma.magma_sym_const(c)})
            _sym_conversion.update({c.decode(): _libmagma.magma_sym_const(c)})
        for c in [b'N', b'n', b'U', b'U', b'L', b'l', b'C', b'c', b'R', b'r',b'B',b'b', b'Q', b'q', b'Z', b'z']:
            _pack_conversion.update({c: _libmagma.magma_pack_const(c)})
            _pack_conversion.update({c.decode(): _libmagma.magma_pack_const(c)})
        for c in [b'N', b'n', b'V', b'v', b'I', b'i', b'A', b'a', b'S', b's',b'O',b'o']:
            _vec_conversion.update({c: _libmagma.magma_vec_const(c)})
            _vec_conversion.update({c.decode(): _libmagma.magma_vec_const(c)})
        for c in [ b'V', b'v', b'I', b'i', b'A', b'a']:
            _range_conversion.update({c: _libmagma.magma_range_const(c)})
            _range_conversion.update({c.decode(): _libmagma.magma_range_const(c)})
        for c in [b'q', b'Q', b'p', b'P']:
            _vect_conversion.update({c: _libmagma.magma_vect_const(c)})
            _vect_conversion.update({c.decode(): _libmagma.magma_vect_const(c)})
        for c in [b'f', b'F', b'B', b'b']:
            _direct_conversion.update({c: _libmagma.magma_direct_const(c)})
            _direct_conversion.update({c.decode(): _libmagma.magma_direct_const(c)})
        for c in [b'c', b'C', b'r', b'R']:
            _storev_conversion.update({c: _libmagma.magma_storev_const(c)})
            _storev_conversion.update({c.decode(): _libmagma.magma_storev_const(c)})
        for c in [b'l', b'L', b'u', b'U']:
            _uplo_conversion.update({c: _libmagma.magma_uplo_const(c)})
            _uplo_conversion.update({c.decode(): _libmagma.magma_uplo_const(c)})
        for c in [b'l', b'L', b'r', b'R', b'b', b'B']:
            _side_conversion.update({c: _libmagma.magma_side_const(c)})
            _side_conversion.update({c.decode(): _libmagma.magma_side_const(c)})
        for c in [b'n', b'N', b't', b'T', b'c', b'C']:
            _trans_conversion.update({c: _libmagma.magma_trans_const(c)})
            _trans_conversion.update({c.decode(): _libmagma.magma_trans_const(c)})
        for c in [b'N', b'n', b'U', b'u']:
            _diag_conversion.update({c: _libmagma.magma_diag_const(c)})
            _diag_conversion.update({c.decode(): _libmagma.magma_diag_const(c)})
    else:
        for c in ['l', 'L', 'u', 'U']:
            _uplo_conversion.update({c: c})
        for c in ['n', 'N', 'a', 'A', 'o', 'O', 's', 'S', 'i', 'I', 'v', 'V']:
            _vec_conversion.update({c: c})
        for c in ['l', 'L', 'r', 'R', 'b', 'B']:
            _sides_conversion.update({c: c})
        for c in ['n', 'N', 't', 'T', 'c', 'C']:
            _trans_conversion.update({c:c})
        for c in ['n', 'N', 'u', 'U']:
            _diag_conversion.update({c:c})

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

    dev = c_int_type()
    _libmagma.magma_getdevice(ctypes.byref(dev))
    return dev.value

_libmagma.magma_setdevice.argtypes = [c_int_type]
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
_libmagma.magma_isamax.argtypes = [c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_isamax(n, dx, incx, queue):
    """
    Index of maximum magnitude element.
    """

    return _libmagma.magma_isamax(n, int(dx), incx, queue)

_libmagma.magma_idamax.restype = int
_libmagma.magma_idamax.argtypes = [c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_idamax(n, dx, incx, queue):
    """
    Index of maximum magnitude element.
    """

    return _libmagma.magma_idamax(n, int(dx), incx, queue)

_libmagma.magma_icamax.restype = int
_libmagma.magma_icamax.argtypes = [c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_icamax(n, dx, incx, queue):
    """
    Index of maximum magnitude element.
    """

    return _libmagma.magma_icamax(n, int(dx), incx, queue)

_libmagma.magma_izamax.restype = int
_libmagma.magma_izamax.argtypes = [c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_izamax(n, dx, incx, queue):
    """
    Index of maximum magnitude element.
    """

    return _libmagma.magma_izamax(n, int(dx), incx, queue)

# ISAMIN, IDAMIN, ICAMIN, IZAMIN
_libmagma.magma_isamin.restype = int
_libmagma.magma_isamin.argtypes = [c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_isamin(n, dx, incx, queue):
    """
    Index of minimum magnitude element.
    """

    return _libmagma.magma_isamin(n, int(dx), incx, queue)

_libmagma.magma_idamin.restype = int
_libmagma.magma_idamin.argtypes = [c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_idamin(n, dx, incx, queue):
    """
    Index of minimum magnitude element.
    """

    return _libmagma.magma_idamin(n, int(dx), incx, queue)

_libmagma.magma_icamin.restype = int
_libmagma.magma_icamin.argtypes = [c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_icamin(n, dx, incx, queue):
    """
    Index of minimum magnitude element.
    """

    return _libmagma.magma_icamin(n, int(dx), incx, queue)

_libmagma.magma_izamin.restype = int
_libmagma.magma_izamin.argtypes = [c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_izamin(n, dx, incx, queue):
    """
    Index of minimum magnitude element.
    """

    return _libmagma.magma_izamin(n, int(dx), incx, queue)

# SASUM, DASUM, SCASUM, DZASUM
_libmagma.magma_sasum.restype = int
_libmagma.magma_sasum.argtypes = [c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_sasum(n, dx, incx, queue):
    """
    Sum of absolute values of vector.
    """

    return _libmagma.magma_sasum(n, int(dx), incx, queue)

_libmagma.magma_dasum.restype = int
_libmagma.magma_dasum.argtypes = [c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_dasum(n, dx, incx, queue):
    """
    Sum of absolute values of vector.
    """

    return _libmagma.magma_dasum(n, int(dx), incx, queue)

_libmagma.magma_scasum.restype = int
_libmagma.magma_scasum.argtypes = [c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_scasum(n, dx, incx, queue):
    """
    Sum of absolute values of vector.
    """

    return _libmagma.magma_scasum(n, int(dx), incx, queue)

_libmagma.magma_dzasum.restype = int
_libmagma.magma_dzasum.argtypes = [c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_dzasum(n, dx, incx, queue):
    """
    Sum of absolute values of vector.
    """

    return _libmagma.magma_dzasum(n, int(dx), incx, queue)

# SAXPY, DAXPY, CAXPY, ZAXPY
_libmagma.magma_saxpy.restype = int
_libmagma.magma_saxpy.argtypes = [c_int_type,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_saxpy(n, alpha, dx, incx, dy, incy, queue):
    """
    Vector addition.
    """

    _libmagma.magma_saxpy(n, alpha, int(dx), incx, int(dy), incy, queue)

_libmagma.magma_daxpy.restype = int
_libmagma.magma_daxpy.argtypes = [c_int_type,
                                  ctypes.c_double,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_daxpy(n, alpha, dx, incx, dy, incy, queue):
    """
    Vector addition.
    """

    _libmagma.magma_daxpy(n, alpha, int(dx), incx, int(dy), incy, queue)

_libmagma.magma_caxpy.restype = int
_libmagma.magma_caxpy.argtypes = [c_int_type,
                                  cuda.cuFloatComplex,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_caxpy(n, alpha, dx, incx, dy, incy, queue):
    """
    Vector addition.
    """

    _libmagma.magma_caxpy(n, ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                              alpha.imag)),
                          int(dx), incx, int(dy), incy, queue)

_libmagma.magma_zaxpy.restype = int
_libmagma.magma_zaxpy.argtypes = [c_int_type,
                                  cuda.cuDoubleComplex,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_zaxpy(n, alpha, dx, incx, dy, incy, queue):
    """
    Vector addition.
    """

    _libmagma.magma_zaxpy(n, ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                               alpha.imag)),
                          int(dx), incx, int(dy), incy, queue)

# SCOPY, DCOPY, CCOPY, ZCOPY
_libmagma.magma_scopy.restype = int
_libmagma.magma_scopy.argtypes = [c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_scopy(n, dx, incx, dy, incy, queue):
    """
    Vector copy.
    """

    _libmagma.magma_scopy(n, int(dx), incx, int(dy), incy, queue)

_libmagma.magma_dcopy.restype = int
_libmagma.magma_dcopy.argtypes = [c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_dcopy(n, dx, incx, dy, incy, queue):
    """
    Vector copy.
    """

    _libmagma.magma_dcopy(n, int(dx), incx, int(dy), incy, queue)

_libmagma.magma_ccopy.restype = int
_libmagma.magma_ccopy.argtypes = [c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_ccopy(n, dx, incx, dy, incy, queue):
    """
    Vector copy.
    """

    _libmagma.magma_ccopy(n, int(dx), incx, int(dy), incy, queue)

_libmagma.magma_zcopy.restype = int
_libmagma.magma_zcopy.argtypes = [c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_zcopy(n, dx, incx, dy, incy, queue):
    """
    Vector copy.
    """

    _libmagma.magma_zcopy(n, int(dx), incx, int(dy), incy, queue)

# SDOT, DDOT, CDOTU, CDOTC, ZDOTU, ZDOTC
_libmagma.magma_sdot.restype = ctypes.c_float
_libmagma.magma_sdot.argtypes = [c_int_type,
                                 ctypes.c_void_p,
                                 c_int_type,
                                 ctypes.c_void_p,
                                 c_int_type,
                                 ctypes.c_void_p]
def magma_sdot(n, dx, incx, dy, incy, queue):
    """
    Vector dot product.
    """

    return _libmagma.magma_sdot(n, int(dx), incx, int(dy), incy, queue)

_libmagma.magma_ddot.restype = ctypes.c_double
_libmagma.magma_ddot.argtypes = [c_int_type,
                                 ctypes.c_void_p,
                                 c_int_type,
                                 ctypes.c_void_p,
                                 c_int_type,
                                 ctypes.c_void_p]
def magma_ddot(n, dx, incx, dy, incy, queue):
    """
    Vector dot product.
    """

    return _libmagma.magma_ddot(n, int(dx), incx, int(dy), incy, queue)

_libmagma.magma_cdotc.restype = cuda.cuFloatComplex
_libmagma.magma_cdotc.argtypes = [c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_cdotc(n, dx, incx, dy, incy, queue):
    """
    Vector dot product.
    """

    return _libmagma.magma_cdotc(n, int(dx), incx, int(dy), incy, queue)

_libmagma.magma_cdotu.restype = cuda.cuFloatComplex
_libmagma.magma_cdotu.argtypes = [c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_cdotu(n, dx, incx, dy, incy, queue):
    """
    Vector dot product.
    """

    return _libmagma.magma_cdotu(n, int(dx), incx, int(dy), incy, queue)

_libmagma.magma_zdotc.restype = cuda.cuDoubleComplex
_libmagma.magma_zdotc.argtypes = [c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_zdotc(n, dx, incx, dy, incy, queue):
    """
    Vector dot product.
    """

    return _libmagma.magma_zdotc(n, int(dx), incx, int(dy), incy, queue)

_libmagma.magma_zdotu.restype = cuda.cuDoubleComplex
_libmagma.magma_zdotu.argtypes = [c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_zdotu(n, dx, incx, dy, incy, queue):
    """
    Vector dot product.
    """

    return _libmagma.magma_zdotu(n, int(dx), incx, int(dy), incy)

# SNRM2, DNRM2, SCNRM2, DZNRM2
_libmagma.magma_snrm2.restype = ctypes.c_float
_libmagma.magma_snrm2.argtypes = [c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_snrm2(n, dx, incx, queue):
    """
    Euclidean norm (2-norm) of vector.
    """

    return _libmagma.magma_snrm2(n, int(dx), incx, queue)

_libmagma.magma_dnrm2.restype = ctypes.c_double
_libmagma.magma_dnrm2.argtypes = [c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_dnrm2(n, dx, incx, queue):
    """
    Euclidean norm (2-norm) of vector.
    """

    return _libmagma.magma_dnrm2(n, int(dx), incx, queue)

_libmagma.magma_scnrm2.restype = ctypes.c_float
_libmagma.magma_scnrm2.argtypes = [c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_scnrm2(n, dx, incx, queue):
    """
    Euclidean norm (2-norm) of vector.
    """

    return _libmagma.magma_scnrm2(n, int(dx), incx, queue)

_libmagma.magma_dznrm2.restype = ctypes.c_double
_libmagma.magma_dznrm2.argtypes = [c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_dznrm2(n, dx, incx, queue):
    """
    Euclidean norm (2-norm) of vector.
    """

    return _libmagma.magma_dznrm2(n, int(dx), incx, queue)

# SROT, DROT, CROT, CSROT, ZROT, ZDROT
_libmagma.magma_srot.argtypes = [c_int_type,
                                 ctypes.c_void_p,
                                 c_int_type,
                                 ctypes.c_void_p,
                                 c_int_type,
                                 ctypes.c_float,
                                 ctypes.c_float,
                                 ctypes.c_void_p]
def magma_srot(n, dx, incx, dy, incy, dc, ds, queue):
    """
    Apply a rotation to vectors.
    """

    _libmagma.magma_srot(n, int(dx), incx, int(dy), incy, dc, ds, queue)

# SROTM, DROTM
_libmagma.magma_srotm.argtypes = [c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p]
def magma_srotm(n, dx, incx, dy, incy, param, queue):
    """
    Apply a real modified Givens rotation.
    """

    _libmagma.magma_srotm(n, int(dx), incx, int(dy), incy, param, queue)

# SROTMG, DROTMG
_libmagma.magma_srotmg.argtypes = [ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_srotmg(d1, d2, x1, y1, param, queue):
    """
    Construct a real modified Givens rotation matrix.
    """

    _libmagma.magma_srotmg(int(d1), int(d2), int(x1), int(y1), param, queue)

# SSCAL, DSCAL, CSCAL, CSSCAL, ZSCAL, ZDSCAL
_libmagma.magma_sscal.argtypes = [c_int_type,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_sscal(n, alpha, dx, incx, queue):
    """
    Scale a vector by a scalar.
    """

    _libmagma.magma_sscal(n, alpha, int(dx), incx, queue)

_libmagma.magma_cscal.argtypes = [c_int_type,
                                  cuda.cuFloatComplex,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_cscal(n, alpha, dx, incx, queue):
    """
    Scale a vector by a scalar.
    """

    _libmagma.magma_cscal(n, alpha, int(dx), incx, queue)

_libmagma.magma_csscal.argtypes = [c_int_type,
                                   ctypes.c_float,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_csscal(n, alpha, dx, incx, queue):
    """
    Scale a vector by a scalar.
    """

    _libmagma.magma_csscal(n, alpha, int(dx), incx, queue)

_libmagma.magma_sscal.argtypes = [c_int_type,
                                  ctypes.c_double,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_dscal(n, alpha, dx, incx, queue):
    """
    Scale a vector by a scalar.
    """

    _libmagma.magma_dscal(n, alpha, int(dx), incx, queue)

_libmagma.magma_zscal.argtypes = [c_int_type,
                                  cuda.cuDoubleComplex,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_zscal(n, alpha, dx, incx, queue):
    """
    Scale a vector by a scalar.
    """

    _libmagma.magma_zscal(n, alpha, int(dx), incx, queue)

_libmagma.magma_zdscal.argtypes = [c_int_type,
                                   ctypes.c_double,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_zdscal(n, alpha, dx, incx, queue):
    """
    Scale a vector by a scalar.
    """

    _libmagma.magma_zdscal(n, alpha, int(dx), incx, queue)

# SSWAP, DSWAP, CSWAP, ZSWAP
_libmagma.magma_sswap.argtypes = [c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_sswap(n, dA, ldda, dB, lddb, queue):
    """
    Swap vectors.
    """

    _libmagma.magma_sswap(n, int(dA), ldda, int(dB), lddb, queue)

# SGEMV, DGEMV, CGEMV, ZGEMV
_libmagma.magma_sgemv.argtypes = [ctypes.c_char,
                                  c_int_type,
                                  c_int_type,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_sgemv(trans, m, n, alpha, dA, ldda, dx, incx, beta,
                dy, incy, queue):
    """
    Matrix-vector product for general matrix.
    """

    _libmagma.magma_sgemv(trans, m, n, alpha, int(dA), ldda, dx, incx,
                          beta, int(dy), incy, queue)

# SGER, DGER, CGERU, CGERC, ZGERU, ZGERC
_libmagma.magma_sger.argtypes = [c_int_type,
                                 c_int_type,
                                 ctypes.c_float,
                                 ctypes.c_void_p,
                                 c_int_type,
                                 ctypes.c_void_p,
                                 c_int_type,
                                 ctypes.c_void_p,
                                 c_int_type,
                                 ctypes.c_void_p]
def magma_sger(m, n, alpha, dx, incx, dy, incy, dA, ldda, queue):
    """
    Rank-1 operation on real general matrix.
    """

    _libmagma.magma_sger(m, n, alpha, int(dx), incx, int(dy), incy,
                         int(dA), ldda, queue)

# SSYMV, DSYMV, CSYMV, ZSYMV
_libmagma.magma_ssymv.argtypes = [ctypes.c_char,
                                  c_int_type,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_ssymv(uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy):
    _libmagma.magma_ssymv(uplo, n, alpha, int(dA), ldda, int(dx), incx, beta,
                          int(dy), incy, queue)

# SSYR, DSYR, CSYR, ZSYR
_libmagma.magma_ssyr.argtypes = [ctypes.c_char,
                                 c_int_type,
                                 ctypes.c_float,
                                 ctypes.c_void_p,
                                 c_int_type,
                                 ctypes.c_void_p,
                                 c_int_type,
                                 ctypes.c_void_p]
def magma_ssyr(uplo, n, alpha, dx, incx, dA, ldda, queue):
    _libmagma.magma_ssyr(uplo, n, alpha, int(dx), incx, int(dA), ldda, queue)

# SSYR2, DSYR2, CSYR2, ZSYR2
_libmagma.magma_ssyr2.argtypes = [ctypes.c_char,
                                  c_int_type,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_ssyr2(uplo, n, alpha, dx, incx, dy, incy, dA, ldda, queue):
    _libmagma.magma_ssyr2(uplo, n, alpha, int(dx), incx,
                          int(dy), incy, int(dA), ldda, queue)

# STRMV, DTRMV, CTRMV, ZTRMV
_libmagma.magma_strmv.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_char,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_strmv(uplo, trans, diag, n,
                dA, ldda, dx, incx, queue):
    _libmagma.magma_strmv(uplo, trans, diag, n,
                          int(dA), ldda, int(dx), incx, queue)

# STRSV, DTRSV, CTRSV, ZTRSV
_libmagma.magma_strsv.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_char,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_strsv(uplo, trans, diag, n,
                dA, ldda, dx, incx, queue):
    _libmagma.magma_strsv(uplo, trans, diag, n,
                          int(dA), ldda, int(dx), incx, queue)

# SGEMM, DGEMM, CGEMM, ZGEMM
_libmagma.magma_sgemm.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  c_int_type,
                                  c_int_type,
                                  c_int_type,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_sgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta,
                dC, lddc, queue):
    _libmagma.magma_sgemm(transA, transB, m, n, k, alpha,
                          int(dA), ldda, int(dB), lddb,
                          beta, int(dC), lddc, queue)

_libmagma.magma_zgemm.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  c_int_type,
                                  c_int_type,
                                  c_int_type,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_zgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta,
                dC, lddc, queue):
    _libmagma.magma_zgemm(transA, transB, m, n, k, alpha,
                          int(dA), ldda, int(dB), lddb,
                          beta, int(dC), lddc, queue)

# SSYMM, DSYMM, CSYMM, ZSYMM
_libmagma.magma_ssymm.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  c_int_type,
                                  c_int_type,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_ssymm(side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta,
                dC, lddc, queue):
    _libmagma.magma_ssymm(side, uplo, m, n, alpha,
                          int(dA), ldda, int(dB), lddb,
                          beta, int(dC), lddc, queue)

# SSYRK, DSYRK, CSYRK, ZSYRK
_libmagma.magma_ssyrk.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  c_int_type,
                                  c_int_type,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_ssyrk(uplo, trans, n, k, alpha, dA, ldda, beta,
                dC, lddc, queue):
    _libmagma.magma_ssyrk(uplo, trans, n, k, alpha,
                          int(dA), ldda, beta, int(dC), lddc, queue)

# SSYR2K, DSYR2K, CSYR2K, ZSYR2K
_libmagma.magma_ssyr2k.argtypes = [ctypes.c_char,
                                   ctypes.c_char,
                                   c_int_type,
                                   c_int_type,
                                   ctypes.c_float,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_float,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_ssyr2k(uplo, trans, n, k, alpha, dA, ldda,
                 dB, lddb, beta, dC, lddc, queue):
    _libmagma.magma_ssyr2k(uplo, trans, n, k, alpha,
                           int(dA), ldda, int(dB), lddb,
                           beta, int(dC), lddc, queue)

# STRMM, DTRMM, CTRMM, ZTRMM
_libmagma.magma_strmm.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_char,
                                  c_int_type,
                                  c_int_type,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_strmm(side, uplo, trans, diag, m, n, alpha, dA, ldda,
                dB, lddb, queue):
    _libmagma.magma_strmm(uplo, trans, diag, m, n, alpha,
                          int(dA), ldda, int(dB), lddb, queue)

# STRSM, DTRSM, CTRSM, ZTRSM
_libmagma.magma_strsm.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_char,
                                  c_int_type,
                                  c_int_type,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_strsm(side, uplo, trans, diag, m, n, alpha, dA, ldda,
                dB, lddb, queue):
    _libmagma.magma_strsm(uplo, trans, diag, m, n, alpha,
                          int(dA), ldda, int(dB), lddb, queue)


# Auxiliary routines:
_libmagma.magma_vec_const.restype = int
_libmagma.magma_vec_const.argtypes = [ctypes.c_char]
def magma_vec_const(job):
    return _libmagma.magma_vec_const(job)

_libmagma.magma_get_spotrf_nb.restype = int
_libmagma.magma_get_spotrf_nb.argtypes = [c_int_type]
def magma_get_spotrf_nb(m):
    return _libmagma.magma_get_spotrf_nb(m)

_libmagma.magma_get_sgetrf_nb.restype = int
_libmagma.magma_get_sgetrf_nb.argtypes = [c_int_type]
def magma_get_sgetrf_nb(m):
    return _libmagma.magma_get_sgetrf_nb(m)

_libmagma.magma_get_sgetri_nb.restype = int
_libmagma.magma_get_sgetri_nb.argtypes = [c_int_type]
def magma_get_sgetri_nb(m):
    return _libmagma.magma_get_sgetri_nb(m)

_libmagma.magma_get_sgeqp3_nb.restype = int
_libmagma.magma_get_sgeqp3_nb.argtypes = [c_int_type]
def magma_get_sgeqp3_nb(m):
    return _libmagma.magma_get_sgeqp3_nb(m)

_libmagma.magma_get_sgeqrf_nb.restype = int
_libmagma.magma_get_sgeqrf_nb.argtypes = [c_int_type, c_int_type]
def magma_get_sgeqrf_nb(m, n):
    return _libmagma.magma_get_sgeqrf_nb(m, n)

_libmagma.magma_get_dgeqrf_nb.restype = int
_libmagma.magma_get_dgeqrf_nb.argtypes = [c_int_type, c_int_type]
def magma_get_dgeqrf_nb(m, n):
    return _libmagma.magma_get_dgeqrf_nb(m, n)

_libmagma.magma_get_cgeqrf_nb.restype = int
_libmagma.magma_get_cgeqrf_nb.argtypes = [c_int_type, c_int_type]
def magma_get_cgeqrf_nb(m, n):
    return _libmagma.magma_get_cgeqrf_nb(m, n)

_libmagma.magma_get_zgeqrf_nb.restype = int
_libmagma.magma_get_zgeqrf_nb.argtypes = [c_int_type, c_int_type]
def magma_get_zgeqrf_nb(m, n):
    return _libmagma.magma_get_zgeqrf_nb(m, n)

_libmagma.magma_get_sgeqlf_nb.restype = int
_libmagma.magma_get_sgeqlf_nb.argtypes = [c_int_type]
def magma_get_sgeqlf_nb(m):
    return _libmagma.magma_get_sgeqlf_nb(m)

_libmagma.magma_get_sgehrd_nb.restype = int
_libmagma.magma_get_sgehrd_nb.argtypes = [c_int_type]
def magma_get_sgehrd_nb(m):
    return _libmagma.magma_get_sgehrd_nb(m)

_libmagma.magma_get_ssytrd_nb.restype = int
_libmagma.magma_get_ssytrd_nb.argtypes = [c_int_type]
def magma_get_ssytrd_nb(m):
    return _libmagma.magma_get_ssytrd_nb(m)

_libmagma.magma_get_sgelqf_nb.restype = int
_libmagma.magma_get_sgelqf_nb.argtypes = [c_int_type]
def magma_get_sgelqf_nb(m):
    return _libmagma.magma_get_sgelqf_nb(m)

_libmagma.magma_get_sgebrd_nb.restype = int
_libmagma.magma_get_sgebrd_nb.argtypes = [c_int_type]
def magma_get_sgebrd_nb(m):
    return _libmagma.magma_get_sgebrd_nb(m)

_libmagma.magma_get_ssygst_nb.restype = int
_libmagma.magma_get_ssygst_nb.argtypes = [c_int_type]
def magma_get_ssygst_nb(m):
    return _libmagma.magma_get_ssgyst_nb(m)

_libmagma.magma_get_sbulge_nb.restype = int
_libmagma.magma_get_sbulge_nb.argtypes = [c_int_type]
def magma_get_sbulge_nb(m):
    return _libmagma.magma_get_sbulge_nb(m)

_libmagma.magma_get_dsytrd_nb.restype = int
_libmagma.magma_get_dsytrd_nb.argtypes = [c_int_type]
def magma_get_dsytrd_nb(m):
    return _libmagma.magma_get_dsytrd_nb(m)

_libmagma.magma_queue_create_internal.restype = int
_libmagma.magma_queue_create_internal.argtypes = [c_int_type,
                                                  ctypes.c_void_p,
                                                  ctypes.c_char_p,
                                                  ctypes.c_char_p,
                                                  c_int_type]
def magma_queue_create(device):
    queue_ptr = ctypes.c_void_p()
    status = _libmagma.magma_queue_create_internal(device, ctypes.byref(queue_ptr), '', '', 0)
    magmaCheckStatus(status)
    return queue_ptr

_libmagma.magma_queue_destroy_internal.restype = int
_libmagma.magma_queue_destroy_internal.argtypes = [ctypes.c_void_p,
                                                   ctypes.c_char_p,
                                                   ctypes.c_char_p,
                                                   c_int_type]
def magma_queue_destroy(queue_ptr):
    status = _libmagma.magma_queue_destroy_internal(queue_ptr, '', '', 0)
    magmaCheckStatus(status)

_libmagma.magma_queue_sync_internal.restype = int
_libmagma.magma_queue_sync_internal.argtypes = [ctypes.c_void_p,
                                                ctypes.c_char_p,
                                                ctypes.c_char_p,
                                                c_int_type]
def magma_queue_sync(queue_ptr):
    status = _libmagma.magma_queue_sync_internal(queue_ptr, '', '', 0)
    magmaCheckStatus(status)

# Buffer size algorithms
def _magma_gesvd_buffersize(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                            func, dtype):
    work = np.zeros(1, dtype)
    func(jobu, jobvt, m, n,
         int(a), lda, int(s), int(u), ldu,
         int(vt), ldvt, int(work.ctypes.data), -1)
    return int(work[0])

def magma_sgesvd_buffersize(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt):
    return _magma_gesvd_buffersize(jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                                   ldvt, magma_sgesvd, np.float32)

def magma_dgesvd_buffersize(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt):
    return _magma_gesvd_buffersize(jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                                   ldvt, magma_dgesvd, np.float64)

def magma_cgesvd_buffersize(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt):
    return _magma_gesvd_buffersize(jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                                   ldvt, magma_cgesvd, np.float32)

def magma_zgesvd_buffersize(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt):
    return _magma_gesvd_buffersize(jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                                   ldvt, magma_zgesvd, np.float64)

def _magma_gels_buffersize(trans, m, n, nrhs, a, lda, b, ldb, func, dtype):
    work = np.zeros(1, dtype)
    func(trans, m, n, nrhs,
         int(a), lda, int(b), ldb, 
         int(work.ctypes.data), -1)
    return int(work[0])

def magma_sgels_buffersize(trans, m, n, nrhs, a, lda, b, ldb):
    return _magma_gels_buffersize(trans, m, n, nrhs, a, lda, b, ldb,
                                  magma_sgels, np.float32)

def magma_dgels_buffersize(trans, m, n, nrhs, a, lda, b, ldb):
    return _magma_gels_buffersize(trans, m, n, nrhs, a, lda, b, ldb,
                                  magma_dgels, np.float64)

def magma_cgels_buffersize(trans, m, n, nrhs, a, lda, b, ldb):
    return _magma_gels_buffersize(trans, m, n, nrhs, a, lda, b, ldb,
                                  magma_cgels, np.float32)

def magma_zgels_buffersize(trans, m, n, nrhs, a, lda, b, ldb):
    return _magma_gels_buffersize(trans, m, n, nrhs, a, lda, b, ldb,
                                  magma_zgels, np.float64)

# LAPACK routines

# SGEBRD, DGEBRD, CGEBRD, ZGEBRD
_libmagma.magma_sgebrd.restype = int
_libmagma.magma_sgebrd.argtypes = [c_int_type,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_sgebrd(m, n, A, lda, d, e, tauq, taup, work, lwork):
    """
    Reduce matrix to bidiagonal form.
    """

    info = c_int_type()
    status = _libmagma.magma_sgebrd.argtypes(m, n, int(A), lda,
                                             int(d), int(e),
                                             int(tauq), int(taup),
                                             int(work), int(lwork),
                                             ctypes.byref(info))
    magmaCheckStatus(status)

# SGEHRD2, DGEHRD2, CGEHRD2, ZGEHRD2
_libmagma.magma_sgehrd2.restype = int
_libmagma.magma_sgehrd2.argtypes = [c_int_type,
                                    c_int_type,
                                    c_int_type,
                                    ctypes.c_void_p,
                                    c_int_type,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p,
                                    c_int_type,
                                    ctypes.c_void_p]
def magma_sgehrd2(n, ilo, ihi, A, lda, tau,
                  work, lwork):
    """
    Reduce matrix to upper Hessenberg form.
    """

    info = c_int_type()
    status = _libmagma.magma_sgehrd2(n, ilo, ihi, int(A), lda,
                                     int(tau), int(work),
                                     lwork, ctypes.byref(info))
    magmaCheckStatus(status)

# SGEHRD, DGEHRD, CGEHRD, ZGEHRD
_libmagma.magma_sgehrd.restype = int
_libmagma.magma_sgehrd.argtypes = [c_int_type,
                                    c_int_type,
                                    c_int_type,
                                    ctypes.c_void_p,
                                    c_int_type,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p,
                                    c_int_type,
                                    ctypes.c_void_p]
def magma_sgehrd(n, ilo, ihi, A, lda, tau,
                 work, lwork, dT):
    """
    Reduce matrix to upper Hessenberg form (fast algorithm).
    """

    info = c_int_type()
    status = _libmagma.magma_sgehrd(n, ilo, ihi, int(A), lda,
                                    int(tau), int(work),
                                    lwork, int(dT), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dgehrd.restype = int
_libmagma.magma_dgehrd.argtypes = _libmagma.magma_sgehrd.argtypes
def magma_dgehrd(n, ilo, ihi, A, lda, tau,
                 work, lwork, dT):
    """
    Reduce matrix to upper Hessenberg form (fast algorithm).
    """

    info = c_int_type()
    status = _libmagma.magma_dgehrd(n, ilo, ihi, int(A), lda,
                                    int(tau), int(work),
                                    lwork, int(dT), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cgehrd.restype = int
_libmagma.magma_cgehrd.argtypes = _libmagma.magma_sgehrd.argtypes
def magma_cgehrd(n, ilo, ihi, A, lda, tau,
                 work, lwork, dT):
    """
    Reduce matrix to upper Hessenberg form (fast algorithm).
    """

    info = c_int_type()
    status = _libmagma.magma_cgehrd(n, ilo, ihi, int(A), lda,
                                    int(tau), int(work),
                                    lwork, int(dT), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zgehrd.restype = int
_libmagma.magma_zgehrd.argtypes = _libmagma.magma_sgehrd.argtypes
def magma_zgehrd(n, ilo, ihi, A, lda, tau,
                 work, lwork, dT):
    """
    Reduce matrix to upper Hessenberg form (fast algorithm).
    """

    info = c_int_type()
    status = _libmagma.magma_zgehrd(n, ilo, ihi, int(A), lda,
                                    int(tau), int(work),
                                    lwork, int(dT), ctypes.byref(info))
    magmaCheckStatus(status)

# SGEHRD_M, DGEHRD_M, CGEHRD_M, ZGEHRD_M
_libmagma.magma_sgehrd_m.restype = int
_libmagma.magma_sgehrd_m.argtypes = _libmagma.magma_sgehrd.argtypes
def magma_sgehrd_m(n, ilo, ihi, A, lda, tau,
                   work, lwork, dT):
    """
    Reduce matrix to upper Hessenberg form (fast algorithm).
    """

    info = c_int_type()
    status = _libmagma.magma_sgehrd_m(n, ilo, ihi, int(A), lda,
                                      int(tau), int(work),
                                      lwork, int(dT), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dgehrd_m.restype = int
_libmagma.magma_dgehrd_m.argtypes = _libmagma.magma_sgehrd.argtypes
def magma_dgehrd_m(n, ilo, ihi, A, lda, tau,
                   work, lwork, dT):
    """
    Reduce matrix to upper Hessenberg form (fast algorithm).
    """

    info = c_int_type()
    status = _libmagma.magma_dgehrd_m(n, ilo, ihi, int(A), lda,
                                      int(tau), int(work),
                                      lwork, int(dT), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cgehrd_m.restype = int
_libmagma.magma_cgehrd_m.argtypes = _libmagma.magma_sgehrd.argtypes
def magma_cgehrd_m(n, ilo, ihi, A, lda, tau,
                   work, lwork, dT):
    """
    Reduce matrix to upper Hessenberg form (fast algorithm).
    """

    info = c_int_type()
    status = _libmagma.magma_cgehrd_m(n, ilo, ihi, int(A), lda,
                                      int(tau), int(work),
                                      lwork, int(dT), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zgehrd_m.restype = int
_libmagma.magma_zgehrd_m.argtypes = _libmagma.magma_sgehrd.argtypes
def magma_zgehrd_m(n, ilo, ihi, A, lda, tau,
                   work, lwork, dT):
    """
    Reduce matrix to upper Hessenberg form (fast algorithm).
    """

    info = c_int_type()
    status = _libmagma.magma_zgehrd_m(n, ilo, ihi, int(A), lda,
                                      int(tau), int(work),
                                      lwork, int(dT), ctypes.byref(info))
    magmaCheckStatus(status)

# SORGHR_M, DORGHR_M, CUNGHR_M, ZUNGHR_M
_libmagma.magma_sorghr_m.restype = int
_libmagma.magma_sorghr_m.argtypes = [c_int_type,
                                     c_int_type,
                                     c_int_type,
                                     ctypes.c_void_p,
                                     c_int_type,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     c_int_type,
                                     ctypes.c_void_p]
def magma_sorghr_m(n, ilo, ihi, A, lda, tau, T, nb):
    """
    Generates a REAL orthogonal matrix Q which is defined as the product of
    IHI-ILO elementary reflectors of order N, as returned by <t>GEHRD
    Multi-GPU, data on host
    """
    info = c_int_type()
    status = _libmagma.magma_sorghr_m(n, ilo, ihi, int(A), lda,
                                      int(tau), int(T), nb,
                                      ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dorghr_m.restype = int
_libmagma.magma_dorghr_m.argtypes = _libmagma.magma_sorghr_m.argtypes
def magma_dorghr_m(n, ilo, ihi, A, lda, tau, T, nb):
    """
    Generates a REAL orthogonal matrix Q which is defined as the product of
    IHI-ILO elementary reflectors of order N, as returned by <t>GEHRD
    Multi-GPU, data on host
    """
    info = c_int_type()
    status = _libmagma.magma_dorghr_m(n, ilo, ihi, int(A), lda,
                                      int(tau), int(T), nb,
                                      ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cunghr_m.restype = int
_libmagma.magma_cunghr_m.argtypes = _libmagma.magma_sorghr_m.argtypes
def magma_cunghr_m(n, ilo, ihi, A, lda, tau, T, nb):
    """
    Generates a REAL orthogonal matrix Q which is defined as the product of
    IHI-ILO elementary reflectors of order N, as returned by <t>GEHRD
    Multi-GPU, data on host
    """
    info = c_int_type()
    status = _libmagma.magma_cunghr_m(n, ilo, ihi, int(A), lda,
                                      int(tau), int(T), nb,
                                      ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zunghr_m.restype = int
_libmagma.magma_zunghr_m.argtypes = _libmagma.magma_sorghr_m.argtypes
def magma_zunghr_m(n, ilo, ihi, A, lda, tau, T, nb):
    """
    Generates a REAL orthogonal matrix Q which is defined as the product of
    IHI-ILO elementary reflectors of order N, as returned by <t>GEHRD
    Multi-GPU, data on host
    """
    info = c_int_type()
    status = _libmagma.magma_zunghr_m(n, ilo, ihi, int(A), lda,
                                      int(tau), int(T), nb,
                                      ctypes.byref(info))
    magmaCheckStatus(status)

# SGELQF, DGELQF, CGELQF, ZGELQF
_libmagma.magma_sgelqf.restype = int
_libmagma.magma_sgelqf.argtypes = [c_int_type,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_sgelqf(m, n, A, lda, tau, work, lwork):

    """
    LQ factorization.
    """

    info = c_int_type()
    status = _libmagma.magma_sgelqf(m, n, int(A), lda,
                                    int(tau), int(work),
                                    lwork, ctypes.byref(info))
    magmaCheckStatus(status)

# SGEQRF, DGEQRF, CGEQRF, ZGEQRF
_libmagma.magma_sgeqrf.restype = int
_libmagma.magma_sgeqrf.argtypes = [c_int_type,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_sgeqrf(m, n, A, lda, tau, work, lwork):
    """
    QR factorization.
    """

    info = c_int_type()
    status = _libmagma.magma_sgeqrf(m, n, int(A), lda,
                                    int(tau), int(work),
                                    lwork, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dgeqrf.restype = int
_libmagma.magma_dgeqrf.argtypes = [c_int_type,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_dgeqrf(m, n, A, lda, tau, work, lwork):
    """
    QR factorization.
    """

    info = c_int_type()
    status = _libmagma.magma_dgeqrf(m, n, int(A), lda,
                                    int(tau), int(work),
                                    lwork, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cgeqrf.restype = int
_libmagma.magma_cgeqrf.argtypes = [c_int_type,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_cgeqrf(m, n, A, lda, tau, work, lwork):
    """
    QR factorization.
    """
    
    info = c_int_type()
    status = _libmagma.magma_cgeqrf(m, n, int(A), lda,
                                    int(tau), int(work),
                                    lwork, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zgeqrf.restype = int
_libmagma.magma_zgeqrf.argtypes = [c_int_type,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_zgeqrf(m, n, A, lda, tau, work, lwork):  
    """
    QR factorization.
    """
    
    info = c_int_type()
    status = _libmagma.magma_zgeqrf(m, n, int(A), lda,
                                    int(tau), int(work),
                                    lwork, ctypes.byref(info))
    magmaCheckStatus(status)

# SGEQRF, DGEQRF, CGEQRF, ZGEQRF (ooc)
_libmagma.magma_sgeqrf_ooc.restype = int
_libmagma.magma_sgeqrf_ooc.argtypes = [c_int_type,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p]
def magma_sgeqrf_ooc(m, n, A, lda, tau, work, lwork):
    """
    QR factorization (ooc).
    """

    info = c_int_type()
    status = _libmagma.magma_sgeqrf_ooc(m, n, int(A), lda,
                                        int(tau), int(work),
                                        lwork, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dgeqrf_ooc.restype = int
_libmagma.magma_dgeqrf_ooc.argtypes = [c_int_type,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p]
def magma_dgeqrf_ooc(m, n, A, lda, tau, work, lwork):
    """
    QR factorization (ooc).
    """

    info = c_int_type()
    status = _libmagma.magma_dgeqrf_ooc(m, n, int(A), lda,
                                        int(tau), int(work),
                                        lwork, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cgeqrf_ooc.restype = int
_libmagma.magma_cgeqrf_ooc.argtypes = [c_int_type,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p]
def magma_cgeqrf_ooc(m, n, A, lda, tau, work, lwork):
    """
    QR factorization (ooc).
    """

    info = c_int_type()
    status = _libmagma.magma_cgeqrf_ooc(m, n, int(A), lda,
                                        int(tau), int(work),
                                        lwork, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zgeqrf_ooc.restype = int
_libmagma.magma_zgeqrf_ooc.argtypes = [c_int_type,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p]
def magma_zgeqrf_ooc(m, n, A, lda, tau, work, lwork):
    """
    QR factorization (ooc).
    """

    info = c_int_type()
    status = _libmagma.magma_zgeqrf_ooc(m, n, int(A), lda,
                                        int(tau), int(work),
                                        lwork, ctypes.byref(info))
    magmaCheckStatus(status)

# SGEQRF_GPU, DGEQRF_GPU, CGEQRF_GPU, ZGEQRF_GPU
_libmagma.magma_sgeqrf_gpu.restype = int
_libmagma.magma_sgeqrf_gpu.argtypes = [c_int_type,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p]
def magma_sgeqrf_gpu(m, n, A, ldda, tau, dT):
    """
    QR factorization (gpu interface, upper triangular R is inverted).
    """

    info = c_int_type()
    status = _libmagma.magma_sgeqrf_gpu(m, n, int(A), ldda,
                                        int(tau), int(dT),
                                        ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dgeqrf_gpu.restype = int
_libmagma.magma_dgeqrf_gpu.argtypes = [c_int_type,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p]
def magma_dgeqrf_gpu(m, n, A, ldda, tau, dT):
    """
    QR factorization (gpu interface, upper triangular R is inverted).
    """

    info = c_int_type()
    status = _libmagma.magma_dgeqrf_gpu(m, n, int(A), ldda,
                                        int(tau), int(dT),
                                        ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cgeqrf_gpu.restype = int
_libmagma.magma_cgeqrf_gpu.argtypes = [c_int_type,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p]
def magma_cgeqrf_gpu(m, n, A, ldda, tau, dT):
    """
    QR factorization (gpu interface,upper triangular R is inverted).
    """

    info = c_int_type()
    status = _libmagma.magma_cgeqrf_gpu(m, n, int(A), ldda,
                                        int(tau), int(dT),
                                        ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zgeqrf_gpu.restype = int
_libmagma.magma_zgeqrf_gpu.argtypes = [c_int_type,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p]
def magma_zgeqrf_gpu(m, n, A, ldda, tau, dT):
    """
    QR factorization (gpu interface, upper triangular R is inverted).
    """

    info = c_int_type()
    status = _libmagma.magma_zgeqrf_gpu(m, n, int(A), ldda,
                                        int(tau), int(dT),
                                        ctypes.byref(info))
    magmaCheckStatus(status)

# SGEQRF2_GPU, DGEQRF2_GPU, CGEQRF2_GPU, ZGEQRF2_GPU
_libmagma.magma_sgeqrf2_gpu.restype = int
_libmagma.magma_sgeqrf2_gpu.argtypes = [c_int_type,
                                        c_int_type,
                                        ctypes.c_void_p,
                                        c_int_type,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p]
def magma_sgeqrf2_gpu(m, n, A, ldda, tau):
    """
    QR factorization (gpu interface, LAPACK-compliant arguments).
    """
    info = c_int_type()
    status = _libmagma.magma_sgeqrf2_gpu(m, n, int(A), ldda,
                                         int(tau),
                                         ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dgeqrf2_gpu.restype = int
_libmagma.magma_dgeqrf2_gpu.argtypes = [c_int_type,
                                        c_int_type,
                                        ctypes.c_void_p,
                                        c_int_type,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p]
def magma_dgeqrf2_gpu(m, n, A, ldda, tau):
    """
    QR factorization (gpu interface, LAPACK-compliant arguments).
    """
    info = c_int_type()
    status = _libmagma.magma_dgeqrf2_gpu(m, n, int(A), ldda,
                                         int(tau),
                                         ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cgeqrf2_gpu.restype = int
_libmagma.magma_cgeqrf2_gpu.argtypes = [c_int_type,
                                        c_int_type,
                                        ctypes.c_void_p,
                                        c_int_type,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p]
def magma_cgeqrf2_gpu(m, n, A, ldda, tau):
    """
    QR factorization (gpu interface, LAPACK-compliant arguments).
    """
    info = c_int_type()
    status = _libmagma.magma_cgeqrf2_gpu(m, n, int(A), ldda,
                                         int(tau),
                                         ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zgeqrf2_gpu.restype = int
_libmagma.magma_zgeqrf2_gpu.argtypes = [c_int_type,
                                        c_int_type,
                                        ctypes.c_void_p,
                                        c_int_type,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p]
def magma_zgeqrf2_gpu(m, n, A, ldda, tau):
    """
    QR factorization (gpu, LAPACK-compliant arguments).
    """
    info = c_int_type()
    status = _libmagma.magma_zgeqrf2_gpu(m, n, int(A), ldda,
                                         int(tau),
                                         ctypes.byref(info))
    magmaCheckStatus(status)

# SGEQRF3_GPU, DGEQRF3_GPU, CGEQRF3_GPU, ZGEQRF3_GPU
_libmagma.magma_sgeqrf3_gpu.restype = int
_libmagma.magma_sgeqrf3_gpu.argtypes = [c_int_type,
                                        c_int_type,
                                        ctypes.c_void_p,
                                        c_int_type,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p]
def magma_sgeqrf3_gpu(m, n, A, ldda, tau, dT):
    """
    QR factorization (gpu interface).
    """
    info = c_int_type()
    status = _libmagma.magma_sgeqrf3_gpu(m, n, int(A), ldda,
                                         int(tau), int(dT),
                                         ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dgeqrf3_gpu.restype = int
_libmagma.magma_dgeqrf3_gpu.argtypes = [c_int_type,
                                        c_int_type,
                                        ctypes.c_void_p,
                                        c_int_type,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p]
def magma_dgeqrf3_gpu(m, n, A, ldda, tau, dT):
    """
    QR factorization (gpu interface).
    """
    info = c_int_type()
    status = _libmagma.magma_dgeqrf3_gpu(m, n, int(A), ldda,
                                         int(tau), int(dT),
                                         ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cgeqrf3_gpu.restype = int
_libmagma.magma_cgeqrf3_gpu.argtypes = [c_int_type,
                                        c_int_type,
                                        ctypes.c_void_p,
                                        c_int_type,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p]
def magma_cgeqrf3_gpu(m, n, A, ldda, tau, dT):
    """
    QR factorization (gpu interface).
    """
    info = c_int_type()
    status = _libmagma.magma_cgeqrf3_gpu(m, n, int(A), ldda,
                                         int(tau), int(dT),
                                         ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zgeqrf3_gpu.restype = int
_libmagma.magma_zgeqrf3_gpu.argtypes = [c_int_type,
                                        c_int_type,
                                        ctypes.c_void_p,
                                        c_int_type,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p]
def magma_zgeqrf3_gpu(m, n, A, ldda, tau, dT):
    """
    QR factorization (gpu interface).
    """
    info = c_int_type()
    status = _libmagma.magma_zgeqrf3_gpu(m, n, int(A), ldda,
                                         int(tau), int(dT),
                                         ctypes.byref(info))
    magmaCheckStatus(status)

# SGEQRF_M, DGEQRF_M, CGEQRF_M, ZGEQRF_M
_libmagma.magma_sgeqrf_m.restype = int
_libmagma.magma_sgeqrf_m.argtypes = [c_int_type,
                                     c_int_type,
                                     c_int_type,
                                     ctypes.c_void_p,
                                     c_int_type,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     c_int_type,
                                     ctypes.c_void_p]
def magma_sgeqrf_m(ngpu, m, n, A, lda, tau, work, lwork):
    """
    QR factorization (multiple gpu,
    GPU memory is allocated in the routine).
    """
    info = c_int_type()
    status = _libmagma.magma_sgeqrf_m(ngpu, m, n, int(A), lda,
                                      int(tau), int(work), lwork,
                                      ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dgeqrf_m.restype = int
_libmagma.magma_dgeqrf_m.argtypes = _libmagma.magma_sgeqrf_m.argtypes
def magma_dgeqrf_m(ngpu, m, n, A, lda, tau, work, lwork):
    """
    QR factorization (multiple gpu,
    GPU memory is allocated in the routine).
    """
    info = c_int_type()
    status = _libmagma.magma_dgeqrf_m(ngpu, m, n, int(A), lda,
                                      int(tau), int(work), lwork,
                                      ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cgeqrf_m.restype = int
_libmagma.magma_cgeqrf_m.argtypes = _libmagma.magma_sgeqrf_m.argtypes
def magma_cgeqrf_m(ngpu, m, n, A, lda, tau, work, lwork):
    """
    QR factorization (multiple gpu,
    GPU memory is allocated in the routine).
    """
    info = c_int_type()
    status = _libmagma.magma_cgeqrf_m(ngpu, m, n, int(A), lda,
                                      int(tau), int(work), lwork,
                                      ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zgeqrf_m.restype = int
_libmagma.magma_zgeqrf_m.argtypes = _libmagma.magma_sgeqrf_m.argtypes
def magma_zgeqrf_m(ngpu, m, n, A, lda, tau, work, lwork):
    """
    QR factorization (multiple gpu,
    GPU memory is allocated in the routine).
    """
    info = c_int_type()
    status = _libmagma.magma_zgeqrf_m(ngpu, m, n, int(A), lda,
                                      int(tau), int(work), lwork,
                                      ctypes.byref(info))
    magmaCheckStatus(status)

# SGEQRF2_MGPU, DGEQRF2_MGPU, CGEQRF2_MGPU, ZGEQRF2_MGPU
_libmagma.magma_sgeqrf2_mgpu.restype = int
_libmagma.magma_sgeqrf2_mgpu.argtypes = [c_int_type,
                                         c_int_type,
                                         c_int_type,
                                         ctypes.c_void_p,
                                         c_int_type,
                                         ctypes.c_void_p,
                                         ctypes.c_void_p]
def magma_sgeqrf2_mgpu(ngpu, m, n, dlA, ldda, tau):
    """
    QR factorization (multiple gpu,
    GPU memory is allocated in the routine).
    """
    info = c_int_type()
    status = _libmagma.magma_sgeqrf2_mgpu(ngpu, m, n, int(dlA),
                                          ldda, int(tau),
                                          ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dgeqrf2_mgpu.restype = int
_libmagma.magma_dgeqrf2_mgpu.argtypes = [c_int_type,
                                         c_int_type,
                                         c_int_type,
                                         ctypes.c_void_p,
                                         c_int_type,
                                         ctypes.c_void_p,
                                         ctypes.c_void_p]
def magma_dgeqrf2_mgpu(ngpu, m, n, dlA, ldda, tau):
    """
    QR factorization (multiple gpu,
    GPU memory is allocated in the routine).
    """
    info = c_int_type()
    status = _libmagma.magma_dgeqrf2_mgpu(ngpu, m, n, int(dlA),
                                          ldda, int(tau),
                                          ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cgeqrf2_mgpu.restype = int
_libmagma.magma_cgeqrf2_mgpu.argtypes = [c_int_type,
                                         c_int_type,
                                         c_int_type,
                                         ctypes.c_void_p,
                                         c_int_type,
                                         ctypes.c_void_p,
                                         ctypes.c_void_p]
def magma_cgeqrf2_mgpu(ngpu, m, n, dlA, ldda, tau):
    """
    QR factorization (multiple gpu,
    GPU memory is allocated in the routine).
    """
    info = c_int_type()
    status = _libmagma.magma_cgeqrf2_mgpu(ngpu, m, n, int(dlA),
                                          ldda, int(tau),
                                          ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zgeqrf2_mgpu.restype = int
_libmagma.magma_zgeqrf2_mgpu.argtypes = [c_int_type,
                                         c_int_type,
                                         c_int_type,
                                         ctypes.c_void_p,
                                         c_int_type,
                                         ctypes.c_void_p,
                                         ctypes.c_void_p]
def magma_zgeqrf2_mgpu(ngpu, m, n, dlA, ldda, tau):
    """
    QR factorization (multiple gpu,
    GPU memory is allocated in the routine).
    """
    info = c_int_type()
    status = _libmagma.magma_zgeqrf2_mgpu(ngpu, m, n, int(dlA),
                                          ldda, int(tau),
                                          ctypes.byref(info))
    magmaCheckStatus(status)

# SORMQR_M, DORMQR_M, CUNMQR_M, ZUNMQR_M
_libmagma.magma_sormqr_m.restype = int
_libmagma.magma_sormqr_m.argtypes = [c_int_type,
                                    c_int_type,
                                    c_int_type,
                                    c_int_type,
                                    c_int_type,
                                    c_int_type,
                                    ctypes.c_void_p,
                                    c_int_type,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p,
                                    c_int_type,
                                    ctypes.c_void_p,
                                    c_int_type,
                                    ctypes.c_void_p]
def magma_sormqr_m(ngpu, side, trans, m, n, k, A, lda,
                   tau, C, ldc, work, lwork):
    """
    Multiply by Q from QR factorization (multiple gpu,
    GPU memory is allocated in the routine).
    """
    info = c_int_type()
    side = _side_conversion[side]
    trans = _trans_conversion[trans]
    status = _libmagma.magma_sormqr_m(ngpu, side, trans, m, n, k,
                                      int(A), lda, int(tau),
                                      int(C), ldc, int(work), lwork,
                                      ctypes.byref(info))

_libmagma.magma_dormqr_m.restype = int
_libmagma.magma_dormqr_m.argtypes = _libmagma.magma_sormqr_m.argtypes
def magma_dormqr_m(ngpu, side, trans, m, n, k, A, lda,
                   tau, C, ldc, work, lwork):
    """
    Multiply by Q from QR factorization (multiple gpu,
    GPU memory is allocated in the routine).
    """
    info = c_int_type()
    side = _side_conversion[side]
    trans = _trans_conversion[trans]
    status = _libmagma.magma_dormqr_m(ngpu, side, trans, m, n, k,
                                      int(A), lda, int(tau),
                                      int(C), ldc, int(work), lwork,
                                      ctypes.byref(info))

_libmagma.magma_cunmqr_m.restype = int
_libmagma.magma_cunmqr_m.argtypes = _libmagma.magma_sormqr_m.argtypes
def magma_cunmqr_m(ngpu, side, trans, m, n, k, A, lda,
                   tau, C, ldc, work, lwork):
    """
    Multiply by Q from QR factorization (multiple gpu,
    GPU memory is allocated in the routine).
    """
    info = c_int_type()
    side = _side_conversion[side]
    trans = _trans_conversion[trans]
    status = _libmagma.magma_cunmqr_m(ngpu, side, trans, m, n, k,
                                      int(A), lda, int(tau),
                                      int(C), ldc, int(work), lwork,
                                      ctypes.byref(info))

_libmagma.magma_zunmqr_m.restype = int
_libmagma.magma_zunmqr_m.argtypes = _libmagma.magma_sormqr_m.argtypes
def magma_zunmqr_m(ngpu, side, trans, m, n, k, A, lda,
                   tau, C, ldc, work, lwork):
    """
    Multiply by Q from QR factorization (multiple gpu,
    GPU memory is allocated in the routine).
    """
    info = c_int_type()
    side = _side_conversion[side]
    trans = _trans_conversion[trans]
    status = _libmagma.magma_zurmqr_m(ngpu, side, trans, m, n, k,
                                      int(A), lda, int(tau),
                                      int(C), ldc, int(work), lwork,
                                      ctypes.byref(info))


# STRSM_M, DTRSM_M, CTRSM_M, ZTRSM_M
_libmagma.magma_strsm_m.restype = int
_libmagma.magma_strsm_m.argtypes = [c_int_type,
                                    c_int_type,
                                    c_int_type,
                                    c_int_type,
                                    c_int_type,
                                    c_int_type,
                                    c_int_type,
                                    ctypes.c_float,
                                    ctypes.c_void_p,
                                    c_int_type,
                                    ctypes.c_void_p,
                                    c_int_type,
                                    ctypes.c_void_p]
def magma_strsm_m(ngpu, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb):
    """
    Solve triangular Linear equations (multiple gpu,
    GPU memory is allocated in the routine).
    """
    info = c_int_type()
    side = _side_conversion[side]
    trans = _trans_conversion[trans]
    uplo = _uplo_conversion[uplo]
    diag = _diag_conversion[diag]
    status = _libmagma.magma_strsm_m(ngpu, side, uplo, trans,
                                     diag, m, n, alpha, int(A),
                                     lda, int(B), ldb,
                                     ctypes.byref(info))

_libmagma.magma_sormqr.restype = int
_libmagma.magma_sormqr.argtypes = [c_int_type,
                                    c_int_type,
                                    c_int_type,
                                    c_int_type,
                                    c_int_type,
                                    ctypes.c_void_p,
                                    c_int_type,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p,
                                    c_int_type,
                                    ctypes.c_void_p,
                                    c_int_type,
                                    ctypes.c_void_p]
def magma_sormqr(side, trans, m, n, k, A, lda,
                   tau, C, ldc, work, lwork):
    """
    Multiply by Q from QR factorization (multiple gpu,
    GPU memory is allocated in the routine).
    """
    info = c_int_type()
    side = _side_conversion[side]
    trans = _trans_conversion[trans]
    status = _libmagma.magma_sormqr(side, trans, m, n, k,
                                      int(A), lda, int(tau),
                                      int(C), ldc, int(work), lwork,
                                      ctypes.byref(info))

# SORGQR, DORGQR, CUNGQR, ZUNGQR
_libmagma.magma_sorgqr.restype = int
_libmagma.magma_sorgqr.argtypes = [c_int_type,
                                   c_int_type,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_sorgqr(m, n, k, A, lda, tau, dT, nb):
    """
    Generate Q from QR factorization.
    """
    info = c_int_type()
    status = _libmagma.magma_sorgqr(m, n, k, int(A), lda,
                                    int(tau), int(dT), nb,
                                    ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dorgqr.restype = int
_libmagma.magma_dorgqr.argtypes = [c_int_type,
                                   c_int_type,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_dorgqr(m, n, k, A, lda, tau, dT, nb):
    """
    Generate Q from QR factorization.
    """
    info = c_int_type()
    status = _libmagma.magma_dorgqr(m, n, k, int(A), lda,
                                    int(tau), int(dT), nb,
                                    ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cungqr.restype = int
_libmagma.magma_cungqr.argtypes = [c_int_type,
                                   c_int_type,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_cungqr(m, n, k, A, lda, tau, dT, nb):
    """
    Generate Q from QR factorization.
    """
    info = c_int_type()
    status = _libmagma.magma_cungqr(m, n, k, int(A), lda,
                                    int(tau), int(dT), nb,
                                    ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zungqr.restype = int
_libmagma.magma_zungqr.argtypes = [c_int_type,
                                   c_int_type,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_zungqr(m, n, k, A, lda, tau, dT, nb):
    """
    Generate Q from QR factorization.
    """
    info = c_int_type()
    status = _libmagma.magma_zungqr(m, n, k, int(A), lda,
                                    int(tau), int(dT), nb,
                                    ctypes.byref(info))
    magmaCheckStatus(status)

# SORGQR2, DORGQR2, CUNGQR2, ZUNGQR2
_libmagma.magma_sorgqr2.restype = int
_libmagma.magma_sorgqr2.argtypes = [c_int_type,
                                    c_int_type,
                                    c_int_type,
                                    ctypes.c_void_p,
                                    c_int_type,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p]
def magma_sorgqr2(m, n, k, A, lda, tau):
    """
    Generate Q from QR factorization.
    (Recompute T matrices on CPU and send them to GPU)
    """
    info = c_int_type()
    status = _libmagma.magma_sorgqr2(m, n, k, int(A), lda,
                                     int(tau), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dorgqr2.restype = int
_libmagma.magma_dorgqr2.argtypes = [c_int_type,
                                    c_int_type,
                                    c_int_type,
                                    ctypes.c_void_p,
                                    c_int_type,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p]
def magma_dorgqr2(m, n, k, A, lda, tau):
    """
    Generate Q from QR factorization.
    (Recompute T matrices on CPU and send them to GPU)
    """
    info = c_int_type()
    status = _libmagma.magma_dorgqr2(m, n, k, int(A), lda,
                                     int(tau), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cungqr2.restype = int
_libmagma.magma_cungqr2.argtypes = [c_int_type,
                                    c_int_type,
                                    c_int_type,
                                    ctypes.c_void_p,
                                    c_int_type,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p]
def magma_cungqr2(m, n, k, A, lda, tau):
    """
    Generate Q from QR factorization.
    (Recompute T matrices on CPU and send them to GPU)
    """
    info = c_int_type()
    status = _libmagma.magma_cungqr2(m, n, k, int(A), lda,
                                     int(tau), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zungqr2.restype = int
_libmagma.magma_zungqr2.argtypes = [c_int_type,
                                    c_int_type,
                                    c_int_type,
                                    ctypes.c_void_p,
                                    c_int_type,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p]
def magma_zungqr2(m, n, k, A, lda, tau):
    """
    Generate Q from QR factorization.
    (Recompute T matrices on CPU and send them to GPU)
    """
    info = c_int_type()
    status = _libmagma.magma_zungqr2(m, n, k, int(A), lda,
                                     int(tau), ctypes.byref(info))
    magmaCheckStatus(status)

# SORGQR_GPU, DORGQR_GPU, CUNGQR_GPU, ZUNGQR_GPU
_libmagma.magma_sorgqr_gpu.restype = int
_libmagma.magma_sorgqr_gpu.argtypes = [c_int_type,
                                       c_int_type,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p]
def magma_sorgqr_gpu(m, n, k, A, ldda, tau, dT, nb):
    """
    Generate Q from QR factorization (GPU interface).
    """
    info = c_int_type()
    status = _libmagma.magma_sorgqr_gpu(m, n, k, int(A), ldda,
                                        int(tau), int(dT), nb,
                                        ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dorgqr_gpu.restype = int
_libmagma.magma_dorgqr_gpu.argtypes = [c_int_type,
                                       c_int_type,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p]
def magma_dorgqr_gpu(m, n, k, A, ldda, tau, dT, nb):
    """
    Generate Q from QR factorization (GPU interface).
    """
    info = c_int_type()
    status = _libmagma.magma_dorgqr_gpu(m, n, k, int(A), ldda,
                                        int(tau), int(dT), nb,
                                        ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cungqr_gpu.restype = int
_libmagma.magma_cungqr_gpu.argtypes = [c_int_type,
                                       c_int_type,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p]
def magma_cungqr_gpu(m, n, k, A, ldda, tau, dT, nb):
    """
    Generate Q from QR factorization (GPU interface).
    """
    info = c_int_type()
    status = _libmagma.magma_cungqr_gpu(m, n, k, int(A), ldda,
                                        int(tau), int(dT), nb,
                                        ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zungqr_gpu.restype = int
_libmagma.magma_zungqr_gpu.argtypes = [c_int_type,
                                       c_int_type,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p]
def magma_zungqr_gpu(m, n, k, A, ldda, tau, dT, nb):
    """
    Generate Q from QR factorization (GPU interface).
    """
    info = c_int_type()
    status = _libmagma.magma_zungqr_gpu(m, n, k, int(A), ldda,
                                        int(tau), int(dT), nb,
                                        ctypes.byref(info))
    magmaCheckStatus(status)

# SORGQR_2STAGE_GPU, DORGQR_2STAGE_GPU
# CUNGQR_2STAGE_GPU, ZUNGQR_2STAGE_GPU
_libmagma.magma_sorgqr_2stage_gpu.restype = int
_libmagma.magma_sorgqr_2stage_gpu.argtypes = [c_int_type,
                                              c_int_type,
                                              c_int_type,
                                              ctypes.c_void_p,
                                              c_int_type,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                              c_int_type,
                                              ctypes.c_void_p]
def magma_sorgqr_2stage_gpu(m, n, k, A, ldda, tau, dT, nb):
    """
    Generate Q from QR factorization (GPU interface).
    """
    info = c_int_type()
    status = _libmagma.magma_sorgqr_2stage_gpu(m, n, k, int(A), ldda,
                                               int(tau), int(dT), nb,
                                               ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dorgqr_2stage_gpu.restype = int
_libmagma.magma_dorgqr_2stage_gpu.argtypes = [c_int_type,
                                              c_int_type,
                                              c_int_type,
                                              ctypes.c_void_p,
                                              c_int_type,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                              c_int_type,
                                              ctypes.c_void_p]
def magma_dorgqr_2stage_gpu(m, n, k, A, ldda, tau, dT, nb):
    """
    Generate Q from QR factorization (GPU interface).
    """
    info = c_int_type()
    status = _libmagma.magma_dorgqr_2stage_gpu(m, n, k, int(A), ldda,
                                               int(tau), int(dT), nb,
                                               ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cungqr_2stage_gpu.restype = int
_libmagma.magma_cungqr_2stage_gpu.argtypes = [c_int_type,
                                              c_int_type,
                                              c_int_type,
                                              ctypes.c_void_p,
                                              c_int_type,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                              c_int_type,
                                              ctypes.c_void_p]
def magma_cungqr_2stage_gpu(m, n, k, A, ldda, tau, dT, nb):
    """
    Generate Q from QR factorization (GPU interface).
    """
    info = c_int_type()
    status = _libmagma.magma_cungqr_2stage_gpu(m, n, k, int(A), ldda,
                                               int(tau), int(dT), nb,
                                               ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zungqr_2stage_gpu.restype = int
_libmagma.magma_zungqr_2stage_gpu.argtypes = [c_int_type,
                                              c_int_type,
                                              c_int_type,
                                              ctypes.c_void_p,
                                              c_int_type,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                             c_int_type,
                                              ctypes.c_void_p]
def magma_zungqr_2stage_gpu(m, n, k, A, ldda, tau, dT, nb):
    """
    Generate Q from QR factorization (GPU interface).
    """
    info = c_int_type()
    status = _libmagma.magma_zungqr_2stage_gpu(m, n, k, int(A), ldda,
                                               int(tau), int(dT), nb,
                                               ctypes.byref(info))
    magmaCheckStatus(status)

# SORGQR_M, DORGQR_M, CUNGQR_M, ZUNGQR_M
_libmagma.magma_sorgqr_m.restype = int
_libmagma.magma_sorgqr_m.argtypes = [c_int_type,
                                     c_int_type,
                                     c_int_type,
                                     ctypes.c_void_p,
                                     c_int_type,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     c_int_type,
                                     ctypes.c_void_p]
def magma_sorgqr_m(m, n, k, A, lda, tau, dT, nb):
    """
    Generate Q from QR factorization (multi-GPU).
    """
    info = c_int_type()
    status = _libmagma.magma_sorgqr_m(m, n, k, int(A), lda,
                                      int(tau), int(dT), nb,
                                      ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dorgqr_m.restype = int
_libmagma.magma_dorgqr_m.argtypes = _libmagma.magma_sorgqr_m.argtypes
def magma_dorgqr_m(m, n, k, A, lda, tau, dT, nb):
    """
    Generate Q from QR factorization (multi-GPU).
    """
    info = c_int_type()
    status = _libmagma.magma_dorgqr_m(m, n, k, int(A), lda,
                                      int(tau), int(dT), nb,
                                      ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cungqr_m.restype = int
_libmagma.magma_cungqr_m.argtypes = _libmagma.magma_sorgqr_m.argtypes
def magma_cungqr_m(m, n, k, A, lda, tau, dT, nb):
    """
    Generate Q from QR factorization.
    """
    info = c_int_type()
    status = _libmagma.magma_cungqr_m(m, n, k, int(A), lda,
                                      int(tau), int(dT), nb,
                                      ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zungqr_m.restype = int
_libmagma.magma_zungqr_m.argtypes = _libmagma.magma_sorgqr_m.argtypes
def magma_zungqr_m(m, n, k, A, lda, tau, dT, nb):
    """
    Generate Q from QR factorization.
    """
    info = c_int_type()
    status = _libmagma.magma_zungqr_m(m, n, k, int(A), lda,
                                      int(tau), int(dT), nb,
                                      ctypes.byref(info))
    magmaCheckStatus(status)

# SGESV, DGESV, CGESV, ZGESV
_libmagma.magma_sgesv.restype = int
_libmagma.magma_sgesv.argtypes = [c_int_type,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_sgesv(n, nhrs, A, lda, ipiv, B, ldb):
    """
    Solve system of linear equations.
    """

    info = c_int_type()
    status = _libmagma.magma_sgesv(n, nhrs, int(A), lda,
                                   int(ipiv), int(B),
                                   ldb, ctypes.byref(info))
    magmaCheckStatus(status)

# SGETRF, DGETRF, CGETRF, ZGETRF
_libmagma.magma_sgetrf.restype = int
_libmagma.magma_sgetrf.argtypes = [c_int_type,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_sgetrf(m, n, A, lda, ipiv):
    """
    LU factorization.
    """

    info = c_int_type()
    status = _libmagma.magma_sgetrf(m, n, int(A), lda,
                                    int(ipiv), ctypes.byref(info))
    magmaCheckStatus(status)

# SGETRF_M, DGETRF_M, CGETRF_M, ZGETRF_M
_libmagma.magma_sgetrf_m.restype = int
_libmagma.magma_sgetrf_m.argtypes = [c_int_type,
                                     c_int_type,
                                     c_int_type,
                                     ctypes.c_void_p,
                                     c_int_type,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p]
def magma_sgetrf_m(ngpu,m, n, A, lda, ipiv):
    """
    LU factorization. Multi-gpu, data on host.
    """

    info = c_int_type()
    status = _libmagma.magma_sgetrf_m(ngpu,m, n, int(A), lda,
                                      int(ipiv), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dgetrf_m.restype = int
_libmagma.magma_dgetrf_m.argtypes = _libmagma.magma_sgetrf_m.argtypes
def magma_dgetrf_m(ngpu,m, n, A, lda, ipiv):
    """
    LU factorization. Multi-gpu, data on host.
    """

    info = c_int_type()
    status = _libmagma.magma_dgetrf_m(ngpu,m, n, int(A), lda,
                                      int(ipiv), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cgetrf_m.restype = int
_libmagma.magma_cgetrf_m.argtypes = _libmagma.magma_sgetrf_m.argtypes
def magma_cgetrf_m(ngpu,m, n, A, lda, ipiv):
    """
    LU factorization. Multi-gpu, data on host.
    """

    info = c_int_type()
    status = _libmagma.magma_cgetrf_m(ngpu,m, n, int(A), lda,
                                      int(ipiv), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zgetrf_m.restype = int
_libmagma.magma_zgetrf_m.argtypes = _libmagma.magma_sgetrf_m.argtypes
def magma_zgetrf_m(ngpu,m, n, A, lda, ipiv):
    """
    LU factorization. Multi-gpu, data on host.
    """

    info = c_int_type()
    status = _libmagma.magma_zgetrf_m(ngpu,m, n, int(A), lda,
                                      int(ipiv), ctypes.byref(info))
    magmaCheckStatus(status)

## SGETRF2, DGETRF2, CGETRF2, ZGETRF2
#_libmagma.magma_sgetrf2.restype = int
#_libmagma.magma_sgetrf2.argtypes = [c_int_type,
#                                    c_int_type,
#                                    ctypes.c_void_p,
#                                    c_int_type,
#                                    ctypes.c_void_p,
#                                    ctypes.c_void_p]
#def magma_sgetrf2(m, n, A, lda, ipiv):
#
#    """
#    LU factorization (multi-GPU).
#    """
#
#    info = c_int_type()
#    status = _libmagma.magma_sgetrf2(m, n, int(A), lda,
#                                    int(ipiv), ctypes.byref(info))
#    magmaCheckStatus(status)

# SGEEV, DGEEV, CGEEV, ZGEEV
_libmagma.magma_sgeev.restype = int
_libmagma.magma_sgeev.argtypes = [c_int_type,
                                  c_int_type,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type]
def magma_sgeev(jobvl, jobvr, n, a, lda, wr, wi,
                vl, ldvl, vr, ldvr, work, lwork):
    """
    Compute eigenvalues and eigenvectors.
    """
    jobvl = _vec_conversion[jobvl]
    jobvr = _vec_conversion[jobvr]
    info = c_int_type()
    status = _libmagma.magma_sgeev(jobvl, jobvr, n, int(a), lda, int(wr), int(wi),
                                   int(vl), ldvl, int(vr), ldvr,
                                   int(work), lwork, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dgeev.restype = int
_libmagma.magma_dgeev.argtypes = _libmagma.magma_sgeev.argtypes
def magma_dgeev(jobvl, jobvr, n, a, lda, wr, wi,
                vl, ldvl, vr, ldvr, work, lwork):
    """
    Compute eigenvalues and eigenvectors.
    """
    jobvl = _vec_conversion[jobvl]
    jobvr = _vec_conversion[jobvr]
    info = c_int_type()
    status = _libmagma.magma_dgeev(jobvl, jobvr, n, int(a), lda, int(wr), int(wi),
                                   int(vl), ldvl, int(vr), ldvr,
                                   int(work), lwork, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cgeev.restype = int
_libmagma.magma_cgeev.argtypes = [c_int_type,
                                  c_int_type,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p]
def magma_cgeev(jobvl, jobvr, n, a, lda,
                w, vl, ldvl, vr, ldvr, work, lwork, rwork):
    """
    Compute eigenvalues and eigenvectors.
    """

    jobvl = _vec_conversion[jobvl]
    jobvr = _vec_conversion[jobvr]
    info = c_int_type()
    status = _libmagma.magma_cgeev(jobvl, jobvr, n, int(a), lda,
                                   int(w), int(vl), ldvl, int(vr), ldvr,
                                   int(work), lwork, int(rwork), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zgeev.restype = int
_libmagma.magma_zgeev.argtypes = _libmagma.magma_cgeev.argtypes
def magma_zgeev(jobvl, jobvr, n, a, lda,
                w, vl, ldvl, vr, ldvr, work, lwork, rwork):
    """
    Compute eigenvalues and eigenvectors.
    """

    jobvl = _vec_conversion[jobvl]
    jobvr = _vec_conversion[jobvr]
    info = c_int_type()
    status = _libmagma.magma_zgeev(jobvl, jobvr, n, int(a), lda,
                                   int(w), int(vl), ldvl, int(vr), ldvr,
                                   int(work), lwork, int(rwork), ctypes.byref(info))
    magmaCheckStatus(status)


# SGEEV_M, DGEEV_M, CGEEV_M, ZGEEV_M
_libmagma.magma_sgeev_m.restype = int
_libmagma.magma_sgeev_m.argtypes = _libmagma.magma_sgeev.argtypes
def magma_sgeev_m(jobvl, jobvr, n, a, lda,
                w, vl, ldvl, vr, ldvr, work, lwork, rwork):
    """
    Compute eigenvalues and eigenvectors.
    Multi-GPU, data on host
    """

    jobvl = _vec_conversion[jobvl]
    jobvr = _vec_conversion[jobvr]
    info = c_int_type()
    status = _libmagma.magma_sgeev_m(jobvl, jobvr, n, int(a), lda,
                                     int(w), int(vl), ldvl, int(vr), ldvr,
                                     int(work), lwork, int(rwork), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dgeev_m.restype = int
_libmagma.magma_dgeev_m.argtypes = _libmagma.magma_sgeev.argtypes
def magma_dgeev_m(jobvl, jobvr, n, a, lda,
                w, vl, ldvl, vr, ldvr, work, lwork, rwork):
    """
    Compute eigenvalues and eigenvectors.
    """

    jobvl = _vec_conversion[jobvl]
    jobvr = _vec_conversion[jobvr]
    info = c_int_type()
    status = _libmagma.magma_dgeev_m(jobvl, jobvr, n, int(a), lda,
                                     int(w), int(vl), ldvl, int(vr), ldvr,
                                     int(work), lwork, int(rwork), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cgeev_m.restype = int
_libmagma.magma_cgeev_m.argtypes = _libmagma.magma_sgeev.argtypes
def magma_cgeev_m(jobvl, jobvr, n, a, lda,
                  w, vl, ldvl, vr, ldvr, work, lwork, rwork):
    """
    Compute eigenvalues and eigenvectors.
    """

    jobvl = _vec_conversion[jobvl]
    jobvr = _vec_conversion[jobvr]
    info = c_int_type()
    status = _libmagma.magma_cgeev_m(jobvl, jobvr, n, int(a), lda,
                                     int(w), int(vl), ldvl, int(vr), ldvr,
                                     int(work), lwork, int(rwork), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zgeev_m.restype = int
_libmagma.magma_zgeev_m.argtypes = _libmagma.magma_sgeev.argtypes
def magma_zgeev_m(jobvl, jobvr, n, a, lda,
                  w, vl, ldvl, vr, ldvr, work, lwork, rwork):
    """
    Compute eigenvalues and eigenvectors.
    """

    jobvl = _vec_conversion[jobvl]
    jobvr = _vec_conversion[jobvr]
    info = c_int_type()
    status = _libmagma.magma_zgeev_m(jobvl, jobvr, n, int(a), lda,
                                     int(w), int(vl), ldvl, int(vr), ldvr,
                                     int(work), lwork, int(rwork), ctypes.byref(info))
    magmaCheckStatus(status)

# SGESVD, DGESVD, CGESVD, ZGESVD
_libmagma.magma_sgesvd.restype = int
_libmagma.magma_sgesvd.argtypes = [c_int_type,
                                   c_int_type,
                                   c_int_type,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork):
    """
    SVD decomposition.
    """

    jobu = _vec_conversion[jobu]
    jobvt = _vec_conversion[jobvt]
    info = c_int_type()
    status = _libmagma.magma_sgesvd(jobu, jobvt, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dgesvd.restype = int
_libmagma.magma_dgesvd.argtypes = [c_int_type,
                                   c_int_type,
                                   c_int_type,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork):
    """
    SVD decomposition.
    """

    jobu = _vec_conversion[jobu]
    jobvt = _vec_conversion[jobvt]
    info = c_int_type()
    status = _libmagma.magma_dgesvd(jobu, jobvt, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cgesvd.restype = int
_libmagma.magma_cgesvd.argtypes = [c_int_type,
                                   c_int_type,
                                   c_int_type,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p]
def magma_cgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 rwork):
    """
    SVD decomposition.
    """

    jobu = _vec_conversion[jobu]
    jobvt = _vec_conversion[jobvt]
    info = c_int_type()
    status = _libmagma.magma_cgesvd(jobu, jobvt, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(rwork), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zgesvd.restype = int
_libmagma.magma_zgesvd.argtypes = [c_int_type,
                                   c_int_type,
                                   c_int_type,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_zgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 rwork):
    """
    SVD decomposition.
    """

    jobu = _vec_conversion[jobu]
    jobvt = _vec_conversion[jobvt]
    c_int_type()
    status = _libmagma.magma_zgesvd(jobu, jobvt, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(rwork), ctypes.byref(info))
    magmaCheckStatus(status)

# SGESDD, DGESDD, CGESDD, ZGESDD
_libmagma.magma_sgesdd.restype = int
_libmagma.magma_sgesdd.argtypes = [c_int_type,
                                   c_int_type,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_sgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 iwork):
    """
    SDD decomposition.
    """

    jobz = _vec_conversion[jobz]
    info = c_int_type()
    status = _libmagma.magma_sgesdd(jobz, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(iwork), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dgesdd.restype = int
_libmagma.magma_dgesdd.argtypes = [c_int_type,
                                   c_int_type,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 iwork):
    """
    SDD decomposition.
    """

    jobz = _vec_conversion[jobz]
    info = c_int_type()
    status = _libmagma.magma_dgesdd(jobz, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(iwork), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cgesdd.restype = int
_libmagma.magma_cgesdd.argtypes = [c_int_type,
                                   c_int_type,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_cgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 rwork, iwork):
    """
    SDD decomposition.
    """

    jobz = _vec_conversion[jobz]
    info = c_int_type()
    status = _libmagma.magma_cgesdd(jobz, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(rwork), int(iwork), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zgesdd.restype = int
_libmagma.magma_zgesdd.argtypes = [c_int_type,
                                   c_int_type,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   c_int_type,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_zgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 rwork, iwork):
    """
    SDD decomposition.
    """

    jobz = _vec_conversion[jobz]
    info = c_int_type()
    status = _libmagma.magma_zgesdd(jobz, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(rwork), int(iwork), ctypes.byref(info))
    magmaCheckStatus(status)

# SPOSV, DPOSV, CPOSV, ZPOSV
_libmagma.magma_sposv_gpu.restype = int
_libmagma.magma_sposv_gpu.argtypes = [c_int_type,
                                      c_int_type,
                                      c_int_type,
                                      ctypes.c_void_p,
                                      c_int_type,
                                      ctypes.c_void_p,
                                      c_int_type,
                                      ctypes.c_void_p]
def magma_sposv_gpu(uplo, n, nhrs, a_gpu, lda, b_gpu, ldb):
    """
    Solve linear system with positive semidefinite coefficient matrix.
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
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
    info = c_int_type()
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
    info = c_int_type()
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
    info = c_int_type()
    status = _libmagma.magma_zposv_gpu(uplo, n, nhrs, int(a_gpu), lda,
                                       int(b_gpu), ldb, ctypes.byref(info))
    magmaCheckStatus(status)

# SGESV, DGESV, CGESV, ZGESV
_libmagma.magma_sgesv_gpu.restype = int
_libmagma.magma_sgesv_gpu.argtypes = [c_int_type,
                                      c_int_type,
                                      ctypes.c_void_p,
                                      c_int_type,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      c_int_type,
                                      ctypes.c_void_p]
def magma_sgesv_gpu(n, nhrs, A, lda, ipiv, B, ldb):
    """
    Solve system of linear equations.
    """

    info = c_int_type()
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

    info = c_int_type()
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
    info = c_int_type()
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

    info = c_int_type()
    status = _libmagma.magma_zgesv_gpu(n, nhrs, int(A), lda,
                                   int(ipiv), int(B),
                                   ldb, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_sgesv_nopiv_gpu.restype = int
_libmagma.magma_sgesv_nopiv_gpu.argtypes = [c_int_type,
                                            c_int_type,
                                            ctypes.c_void_p,
                                            c_int_type,
                                            ctypes.c_void_p,
                                            c_int_type,
                                            ctypes.c_void_p]
def magma_sgesv_nopiv_gpu(n, nhrs, A, lda, B, ldb):
    """
    Solve system of linear equations.
    """
    info = c_int_type()
    status = _libmagma.magma_sgesv_nopiv_gpu(n, nhrs, int(A), lda,
                                             int(B), ldb, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dgesv_nopiv_gpu.restype = int
_libmagma.magma_dgesv_nopiv_gpu.argtypes = _libmagma.magma_sgesv_nopiv_gpu.argtypes
def magma_dgesv_nopiv_gpu(n, nhrs, A, lda, B, ldb):
    """
    Solve system of linear equations.
    """

    info = c_int_type()
    status = _libmagma.magma_dgesv_nopiv_gpu(n, nhrs, int(A), lda,
                                             int(B), ldb, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cgesv_nopiv_gpu.restype = int
_libmagma.magma_cgesv_nopiv_gpu.argtypes = _libmagma.magma_sgesv_nopiv_gpu.argtypes
def magma_cgesv_nopiv_gpu(n, nhrs, A, lda, B, ldb):
    """
    Solve system of linear equations.
    """

    info = c_int_type()
    status = _libmagma.magma_cgesv_nopiv_gpu(n, nhrs, int(A), lda,
                                             int(B), ldb, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zgesv_nopiv_gpu.restype = int
_libmagma.magma_zgesv_nopiv_gpu.argtypes = _libmagma.magma_sgesv_nopiv_gpu.argtypes
def magma_zgesv_nopiv_gpu(n, nhrs, A, lda, B, ldb):
    """
    Solve system of linear equations.
    """

    info = c_int_type()
    status = _libmagma.magma_zgesv_nopiv_gpu(n, nhrs, int(A), lda,
                                             int(B), ldb, ctypes.byref(info))
    magmaCheckStatus(status)

# SPOTRF, DPOTRF, CPOTRF, ZPOTRF
_libmagma.magma_spotrf_gpu.restype = int
_libmagma.magma_spotrf_gpu.argtypes = [c_int_type,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_spotrf_gpu(uplo, n, A, lda):
    """
    Cholesky factorization of positive symmetric matrix.
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libmagma.magma_spotrf_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dpotrf_gpu.restype = int
_libmagma.magma_dpotrf_gpu.argtypes = _libmagma.magma_spotrf_gpu.argtypes
def magma_dpotrf_gpu(uplo, n, A, lda):
    """
    Cholesky factorization of positive symmetric matrix.
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libmagma.magma_dpotrf_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_cpotrf_gpu.restype = int
_libmagma.magma_cpotrf_gpu.argtypes = _libmagma.magma_spotrf_gpu.argtypes
def magma_cpotrf_gpu(uplo, n, A, lda):
    """
    Cholesky factorization of positive symmetric matrix.
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libmagma.magma_cpotrf_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zpotrf_gpu.restype = int
_libmagma.magma_zpotrf_gpu.argtypes = _libmagma.magma_zpotrf_gpu.argtypes
def magma_zpotrf_gpu(uplo, n, A, lda):
    """
    Cholesky factorization of positive symmetric matrix.
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libmagma.magma_zpotrf_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_spotrf_m.restype = int
_libmagma.magma_spotrf_m.argtypes = [c_int_type,
                                     c_int_type,
                                     c_int_type,
                                     ctypes.c_void_p,
                                     c_int_type,
                                     ctypes.c_void_p]
def magma_spotrf_m(ngpu, uplo, n, A, lda):
    """
    Cholesky factorization of positive symmetric matrix.
    Multi-gpu, data on host
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libmagma.magma_spotrf_m(ngu, uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dpotrf_m.restype = int
_libmagma.magma_dpotrf_m.argtypes = _libmagma.magma_spotrf_m.argtypes
def magma_dpotrf_m(ngpu, uplo, n, A, lda):
    """
    Cholesky factorization of positive symmetric matrix.
    Multi-gpu, data on host
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libmagma.magma_dpotrf_m(ngpu, uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_cpotrf_m.restype = int
_libmagma.magma_cpotrf_m.argtypes = _libmagma.magma_spotrf_m.argtypes
def magma_cpotrf_gpu(ngpu, uplo, n, A, lda):
    """
    Cholesky factorization of positive symmetric matrix.
    Multi-gpu, data on host
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libmagma.magma_cpotrf_m(ngpu, uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zpotrf_gpu.restype = int
_libmagma.magma_zpotrf_gpu.argtypes = _libmagma.magma_zpotrf_m.argtypes
def magma_zpotrf_m(ngpu, uplo, n, A, lda):
    """
    Cholesky factorization of positive symmetric matrix.
    Multi-gpu, data on host
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libmagma.magma_zpotrf_m(ngpu, uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)

# SPOTRI, DPOTRI, CPOTRI, ZPOTRI
_libmagma.magma_spotri_gpu.restype = int
_libmagma.magma_spotri_gpu.argtypes = [c_int_type,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_spotri_gpu(uplo, n, A, lda):
    """
    Inverse using the Cholesky factorization of positive symmetric matrix.
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libmagma.magma_spotri_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dpotri_gpu.restype = int
_libmagma.magma_dpotri_gpu.argtypes = _libmagma.magma_spotri_gpu.argtypes
def magma_dpotri_gpu(uplo, n, A, lda):
    """
    Inverse using the Cholesky factorization of positive symmetric matrix.
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libmagma.magma_dpotri_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cpotri_gpu.restype = int
_libmagma.magma_cpotri_gpu.argtypes = _libmagma.magma_spotri_gpu.argtypes
def magma_cpotri_gpu(uplo, n, A, lda):
    """
    Inverse using the Cholesky factorization of positive symmetric matrix.
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libmagma.magma_cpotri_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_zpotri_gpu.restype = int
_libmagma.magma_zpotri_gpu.argtypes = _libmagma.magma_spotri_gpu.argtypes
def magma_zpotri_gpu(uplo, n, A, lda):
    """
    Inverse using the Cholesky factorization of positive symmetric matrix.
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libmagma.magma_zpotri_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)

# SGETRF, DGETRF, CGETRF, ZGETRF
_libmagma.magma_sgetrf_gpu.restype = int
_libmagma.magma_sgetrf_gpu.argtypes = [c_int_type,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p]
def magma_sgetrf_gpu(n, m, A, lda, ipiv):
    """
    LU factorization.
    """

    info = c_int_type()
    status = _libmagma.magma_sgetrf_gpu(n, m, int(A), lda,
                                   int(ipiv), ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dgetrf_gpu.restype = int
_libmagma.magma_dgetrf_gpu.argtypes = _libmagma.magma_sgetrf_gpu.argtypes
def magma_dgetrf_gpu(n, m, A, lda, ipiv):
    """
    LU factorization.
    """

    info = c_int_type()
    status = _libmagma.magma_dgetrf_gpu(n, m, int(A), lda,
                                   int(ipiv), ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_cgetrf_gpu.restype = int
_libmagma.magma_cgetrf_gpu.argtypes = _libmagma.magma_sgetrf_gpu.argtypes
def magma_cgetrf_gpu(n, m, A, lda, ipiv):
    """
    LU factorization.
    """

    info = c_int_type()
    status = _libmagma.magma_cgetrf_gpu(n, m, int(A), lda,
                                   int(ipiv), ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_zgetrf_gpu.restype = int
_libmagma.magma_zgetrf_gpu.argtypes = _libmagma.magma_sgetrf_gpu.argtypes
def magma_zgetrf_gpu(n, m, A, lda, ipiv):
    """
    LU factorization.
    """
    info = c_int_type()
    status = _libmagma.magma_zgetrf_gpu(n, m, int(A), lda,
                                   int(ipiv), ctypes.byref(info))
    magmaCheckStatus(status)

# SGELS, CGELS, DGELS, ZGELS
_libmagma.magma_sgels.restype = int
_libmagma.magma_sgels.argtypes = [c_int_type,
                                  c_int_type,
                                  c_int_type,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]
def magma_sgels(trans, m, n, nrhs, A, lda, B, ldb, hwork, lwork):
    """
    Solve overdetermined least squares problem using QR factorization.
    """
    info = c_int_type()
    trans = _trans_conversion[trans]
    status = _libmagma.magma_sgels(trans, m, n, nrhs, int(A), lda,
                                   int(B), ldb, int(hwork), lwork,
                                   ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cgels.restype = int
_libmagma.magma_cgels.argtypes = _libmagma.magma_sgels.argtypes
def magma_cgels(trans, m, n, nrhs, A, lda, B, ldb, hwork, lwork):
    """
    Solve overdetermined least squares problem using QR factorization.
    """
    info = c_int_type()
    trans = _trans_conversion[trans]
    status = _libmagma.magma_cgels(trans, m, n, nrhs, int(A), lda,
                                   int(B), ldb, int(hwork), lwork,
                                   ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dgels.restype = int
_libmagma.magma_dgels.argtypes = _libmagma.magma_sgels.argtypes
def magma_dgels(trans, m, n, nrhs, A, lda, B, ldb, hwork, lwork):
    """
    Solve overdetermined least squares problem using QR factorization.
    """
    info = c_int_type()
    trans = _trans_conversion[trans]
    status = _libmagma.magma_dgels(trans, m, n, nrhs, int(A), lda,
                                   int(B), ldb, int(hwork), lwork,
                                   ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zgels.restype = int
_libmagma.magma_zgels.argtypes = _libmagma.magma_sgels.argtypes
def magma_zgels(trans, m, n, nrhs, A, lda, B, ldb, hwork, lwork):
    """
    Solve overdetermined least squares problem using QR factorization.
    """
    info = c_int_type()
    trans = _trans_conversion[trans]
    status = _libmagma.magma_zgels(trans, m, n, nrhs, int(A), lda,
                                   int(B), ldb, int(hwork), lwork,
                                   ctypes.byref(info))
    magmaCheckStatus(status)

# SGELS_GPU, CGELS_GPU, DGELS_GPU, ZGELS_GPU
_libmagma.magma_sgels_gpu.restype = int
_libmagma.magma_sgels_gpu.argtypes = [c_int_type,
                                      c_int_type,
                                      c_int_type,
                                      c_int_type,
                                      ctypes.c_void_p,
                                      c_int_type,
                                      ctypes.c_void_p,
                                      c_int_type,
                                      ctypes.c_void_p,
                                      c_int_type,
                                      ctypes.c_void_p]
def magma_sgels_gpu(trans, m, n, nrhs, A, lda, B, ldb, hwork, lwork):
    """
    Solve overdetermined least squares problem using QR factorization.
    """
    info = c_int_type()
    trans = _trans_conversion[trans]
    status = _libmagma.magma_sgels_gpu(trans, m, n, nrhs, int(A), lda,
                                       int(B), ldb, int(hwork), lwork,
                                       ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cgels_gpu.restype = int
_libmagma.magma_cgels_gpu.argtypes = _libmagma.magma_sgels_gpu.argtypes
def magma_cgels_gpu(trans, m, n, nrhs, A, lda, B, ldb, hwork, lwork):
    """
    Solve overdetermined least squares problem using QR factorization.
    """
    info = c_int_type()
    trans = _trans_conversion[trans]
    status = _libmagma.magma_cgels_gpu(trans, m, n, nrhs, int(A), lda,
                                       int(B), ldb, int(hwork), lwork,
                                       ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dgels_gpu.restype = int
_libmagma.magma_dgels_gpu.argtypes = _libmagma.magma_sgels_gpu.argtypes
def magma_dgels_gpu(trans, m, n, nrhs, A, lda, B, ldb, hwork, lwork):
    """
    Solve overdetermined least squares problem using QR factorization.
    """
    info = c_int_type()
    trans = _trans_conversion[trans]
    status = _libmagma.magma_dgels_gpu(trans, m, n, nrhs, int(A), lda,
                                       int(B), ldb, int(hwork), lwork,
                                       ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zgels_gpu.restype = int
_libmagma.magma_zgels_gpu.argtypes = _libmagma.magma_sgels_gpu.argtypes
def magma_zgels_gpu(trans, m, n, nrhs, A, lda, B, ldb, hwork, lwork):
    """
    Solve overdetermined least squares problem using QR factorization.
    """
    info = c_int_type()
    trans = _trans_conversion[trans]
    status = _libmagma.magma_zgels_gpu(trans, m, n, nrhs, int(A), lda,
                                       int(B), ldb, int(hwork), lwork,
                                       ctypes.byref(info))
    magmaCheckStatus(status)

# SSYEVD, DSYEVD
_libmagma.magma_ssyevd_gpu.restype = int
_libmagma.magma_ssyevd_gpu.argtypes = [c_int_type,
                                       c_int_type,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p]
def magma_ssyevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, iwork,
                     liwork):
    """
    Compute eigenvalues of real symmetric matrix.
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libmagma.magma_ssyevd_gpu(jobz, uplo, n, int(dA), ldda,
                                        int(w), int(wA), ldwa, int(work),
                                        lwork, int(iwork), liwork, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_dsyevd_gpu.restype = int
_libmagma.magma_dsyevd_gpu.argtypes = [c_int_type,
                                       c_int_type,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p,
                                       c_int_type,
                                       ctypes.c_void_p]
def magma_dsyevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, iwork,
                     liwork):
    """
    Compute eigenvalues of real symmetric matrix.
    """

    jobz = _vec_conversion[jobz]
    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libmagma.magma_dsyevd_gpu(jobz, uplo, n, int(dA), ldda,
                                        int(w), int(wA), ldwa, int(work),
                                        lwork, int(iwork), liwork, ctypes.byref(info))
    magmaCheckStatus(status)


# SSYEVD_M, DSYEVD_M, CHEEVD_M, ZHEEVD_M
_libmagma.magma_ssyevd_m.restype = int
_libmagma.magma_ssyevd_m.argtypes = [c_int_type,
                                     c_int_type,
                                     c_int_type,
                                     c_int_type,
                                     ctypes.c_void_p,
                                     c_int_type,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     c_int_type,
                                     ctypes.c_void_p,
                                     c_int_type,
                                     ctypes.c_void_p]
def magma_ssyevd_m(ngpu, jobz, uplo, n, A, lda, w, work, lwork, iwork, liwork):
    """
    Compute eigenvalues of real symmetric matrix.
    Multi-GPU, data on host
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libmagma.magma_ssyevd_m(ngpu, jobz, uplo, n, int(A), lda,
                                      int(w), int(work),
                                      lwork, int(iwork), liwork, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dsyevd_m.restype = int
_libmagma.magma_dsyevd_m.argtypes = _libmagma.magma_dsyevd_m.argtypes
def magma_dsyevd_m(ngpu, jobz, uplo, n, A, lda, w, work, lwork, iwork, liwork):
    """
    Compute eigenvalues of real symmetric matrix.
    Multi-GPU, data on host
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libmagma.magma_dsyevd_m(ngpu, jobz, uplo, n, int(A), lda,
                                      int(w), int(work),
                                      lwork, int(iwork), liwork, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cheevd_m.restype = int
_libmagma.magma_cheevd_m.argtypes = [c_int_type,
                                     c_int_type,
                                     c_int_type,
                                     c_int_type,
                                     ctypes.c_void_p,
                                     c_int_type,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     c_int_type,
                                     ctypes.c_void_p,
                                     c_int_type,
                                     ctypes.c_void_p,
                                     c_int_type,
                                     ctypes.c_void_p]
def magma_cheevd_m(ngpu, jobz, uplo, n, A, lda, w, work, lwork,
                   rwork, lrwork, iwork, liwork):
    """
    Compute eigenvalues of complex hermitian matrix.
    Multi-GPU, data on host
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libmagma.magma_cheevd_m(ngpu, jobz, uplo, n, int(A), lda,
                                      int(w), int(work), lwork, int(rwork),
                                      lrwork, int(iwork), liwork, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_zheevd_m.restype = int
_libmagma.magma_zheevd_m.argtypes = _libmagma.magma_cheevd_m.argtypes
def magma_zheevd_m(ngpu, jobz, uplo, n, A, lda, w, work, lwork,
                   rwork, lrwork, iwork, liwork):
    """
    Compute eigenvalues of complex hermitian matrix.
    Multi-GPU, data on host
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libmagma.magma_zheevd_m(ngpu, jobz, uplo, n, int(A), lda,
                                      int(w), int(work), lwork, int(rwork),
                                      lrwork, int(iwork), liwork, ctypes.byref(info))
    magmaCheckStatus(status)


# SSYEVDX_M, DSYEVDX_M, CHEEVDX_M, ZHEEVDX_M
_libmagma.magma_ssyevd_m.restype = int
_libmagma.magma_ssyevd_m.argtypes = [c_int_type,
                                     c_int_type,
                                     c_int_type,
                                     c_int_type,
                                     c_int_type,
                                     ctypes.c_void_p,
                                     c_int_type,
                                     ctypes.c_float,
                                     ctypes.c_float,
                                     c_int_type,
                                     c_int_type,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     c_int_type,
                                     ctypes.c_void_p,
                                     c_int_type,
                                     ctypes.c_void_p]
def magma_ssyevdx_m(ngpu, jobz, rnge, uplo, n, A, lda,
                    vl, vu, il, iu, m,
                    w, work, lwork, iwork, liwork):
    """
    Compute eigenvalues of real symmetric matrix.
    Multi-GPU, data on host
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libmagma.magma_ssyevdx_m(ngpu, jobz, uplo, n, int(A), lda,
                                      int(w), int(work),
                                      lwork, int(iwork), liwork, ctypes.byref(info))
    magmaCheckStatus(status)

# SYMMETRIZE
_libmagma.magmablas_ssymmetrize.restype = int
_libmagma.magmablas_ssymmetrize.argtypes = [c_int_type,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type]
def magmablas_ssymmetrize(uplo, n, A, lda):
    """
    Symmetrize a triangular matrix.
    """

    uplo = _uplo_conversion[uplo]
    status = _libmagma.magmablas_ssymmetrize(uplo, n, int(A), lda)
    magmaCheckStatus(status)


_libmagma.magmablas_dsymmetrize.restype = int
_libmagma.magmablas_dsymmetrize.argtypes = _libmagma.magmablas_ssymmetrize.argtypes
def magmablas_dsymmetrize(uplo, n, A, lda):
    """
    Symmetrize a triangular matrix.
    """

    uplo = _uplo_conversion[uplo]
    status = _libmagma.magmablas_dsymmetrize(uplo, n, int(A), lda)
    magmaCheckStatus(status)


_libmagma.magmablas_csymmetrize.restype = int
_libmagma.magmablas_csymmetrize.argtypes = _libmagma.magmablas_ssymmetrize.argtypes
def magmablas_csymmetrize(uplo, n, A, lda):
    """
    Symmetrize a triangular matrix.
    """

    uplo = _uplo_conversion[uplo]
    status = _libmagma.magmablas_csymmetrize(uplo, n, int(A), lda)
    magmaCheckStatus(status)


_libmagma.magmablas_zsymmetrize.restype = int
_libmagma.magmablas_zsymmetrize.argtypes = _libmagma.magmablas_ssymmetrize.argtypes
def magmablas_zsymmetrize(uplo, n, A, lda):
    """
    Symmetrize a triangular matrix.
    """

    uplo = _uplo_conversion[uplo]
    status = _libmagma.magmablas_zsymmetrize(uplo, n, int(A), lda)
    magmaCheckStatus(status)
