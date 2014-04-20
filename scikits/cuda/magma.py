#!/usr/bin/env python

"""
Python interface to MAGMA toolkit.
"""

import re
import cffi
_ffi = cffi.FFI()

_ffi.cdef("""
typedef struct float2 {
    ...;
} float2;
typedef float2 cuFloatComplex;
typedef float2 cuComplex;

typedef struct double2 {
    ...;
} double2;
typedef double2 cuDoubleComplex;

typedef struct CUstream_st *cudaStream_t;
typedef struct CUevent_st *cudaEvent_t;

typedef int magma_int_t;
typedef int magma_err_t;

typedef cudaStream_t   magma_queue_t;
typedef cudaEvent_t    magma_event_t;
typedef int            magma_device_t;

typedef cuDoubleComplex magmaDoubleComplex;
typedef cuFloatComplex  magmaFloatComplex;

typedef void               *magma_ptr;
typedef magma_int_t        *magmaInt_ptr;
typedef float              *magmaFloat_ptr;
typedef double             *magmaDouble_ptr;
typedef magmaFloatComplex  *magmaFloatComplex_ptr;
typedef magmaDoubleComplex *magmaDoubleComplex_ptr;

typedef void               const *magma_const_ptr;
typedef magma_int_t        const *magmaInt_const_ptr;
typedef float              const *magmaFloat_const_ptr;
typedef double             const *magmaDouble_const_ptr;
typedef magmaFloatComplex  const *magmaFloatComplex_const_ptr;
typedef magmaDoubleComplex const *magmaDoubleComplex_const_ptr;

#define MAGMA_SUCCESS               ... // 0
#define MAGMA_ERR                  ... // -100
#define MAGMA_ERR_NOT_INITIALIZED  ... // -101
#define MAGMA_ERR_REINITIALIZED    ... // -102
#define MAGMA_ERR_NOT_SUPPORTED    ... // -103
#define MAGMA_ERR_ILLEGAL_VALUE    ... // -104
#define MAGMA_ERR_NOT_FOUND        ... // -105
#define MAGMA_ERR_ALLOCATION       ... // -106
#define MAGMA_ERR_INTERNAL_LIMIT   ... // -107
#define MAGMA_ERR_UNALLOCATED      ... // -108
#define MAGMA_ERR_FILESYSTEM       ... // -109
#define MAGMA_ERR_UNEXPECTED       ... // -110
#define MAGMA_ERR_SEQUENCE_FLUSHED ... // -111
#define MAGMA_ERR_HOST_ALLOC       ... // -112
#define MAGMA_ERR_DEVICE_ALLOC     ... // -113
#define MAGMA_ERR_CUDASTREAM       ... // -114
#define MAGMA_ERR_INVALID_PTR      ... // -115
#define MAGMA_ERR_UNKNOWN          ... // -116
// #define MAGMA_ERR_NOT_IMPLEMENTED  ... // -117

const char* magma_strerror(magma_err_t error);
magma_err_t magma_init(void);
magma_err_t magma_finalize(void);
void magma_version(int* major, int* minor, int* micro);

magma_int_t magma_getdevice_arch();
void magma_getdevices(
    magma_device_t* devices,
    magma_int_t     size,
    magma_int_t*    numPtr);
void magma_getdevice(magma_device_t* dev);
void magma_setdevice(magma_device_t dev);
void magma_device_sync();

// Level 1 BLAS - single precision real
// in cublas_v2, result returned through output argument
magma_int_t magma_isamax(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
magma_int_t magma_isamin(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
float magma_sasum(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx );

void magma_saxpy(
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_ptr       dy, magma_int_t incy );

void magma_scopy(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_ptr       dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
float
magma_sdot(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
float magma_snrm2(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx );

void magma_srot(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    float dc, float ds );

void magma_srotm(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    magmaFloat_const_ptr param );

void magma_srotmg(
    magmaFloat_ptr d1, magmaFloat_ptr       d2,
    magmaFloat_ptr x1, magmaFloat_const_ptr y1,
    magmaFloat_ptr param );

void magma_sscal(
    magma_int_t n,
    float alpha,
    magmaFloat_ptr dx, magma_int_t incx );

void magma_sswap(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy );

// Level 1 BLAS - single precision complex 
// in cublas_v2, result returned through output argument
magma_int_t magma_icamax(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
magma_int_t magma_icamin(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
float magma_scasum(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx );

void magma_caxpy(
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dy, magma_int_t incy );

void magma_ccopy(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
magmaFloatComplex
magma_cdotc(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy );

magmaFloatComplex
magma_cdotu(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
float magma_scnrm2(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx );

void magma_crot(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy,
    float dc, magmaFloatComplex ds );

void magma_csrot(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy,
    float dc, float ds );

void magma_cscal(
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dx, magma_int_t incx );

void magma_csscal(
    magma_int_t n,
    float alpha,
    magmaFloatComplex_ptr dx, magma_int_t incx );

void magma_cswap(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy );

// Level 1 BLAS - double precision real
// in cublas_v2, result returned through output argument
magma_int_t magma_idamax(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
magma_int_t magma_idamin(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
double magma_dasum(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx );

void magma_daxpy(
    magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_ptr       dy, magma_int_t incy );

void magma_dcopy(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_ptr       dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
double
magma_ddot(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
double magma_dnrm2(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx );

void magma_drot(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy,
    double dc, double ds );

void magma_drotm(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy,
    magmaDouble_const_ptr param );

void magma_drotmg(
    magmaDouble_ptr d1, magmaDouble_ptr       d2,
    magmaDouble_ptr x1, magmaDouble_const_ptr y1,
    magmaDouble_ptr param );

void magma_dscal(
    magma_int_t n,
    double alpha,
    magmaDouble_ptr dx, magma_int_t incx );

void magma_dswap(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy );

// Level 1 BLAS - double precision complex
// in cublas_v2, result returned through output argument
magma_int_t magma_izamax(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
magma_int_t magma_izamin(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
double magma_dzasum(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx );

void magma_zaxpy(
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr       dy, magma_int_t incy );

void magma_zcopy(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr       dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
magmaDoubleComplex
magma_zdotc(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
magmaDoubleComplex
magma_zdotu(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
double magma_dznrm2(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx );

void magma_zrot(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dy, magma_int_t incy,
    double dc, magmaDoubleComplex ds );

void magma_zdrot(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dy, magma_int_t incy,
    double dc, double ds );

void magma_zscal(
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dx, magma_int_t incx );

void magma_zdscal(
    magma_int_t n,
    double alpha,
    magmaDoubleComplex_ptr dx, magma_int_t incx );

void magma_zswap(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dy, magma_int_t incy );

""")

# HAVE_CUBLAS must be set to enable compilation of cffi's shared library:
_ffi_lib = _ffi.verify("""
#include <magma.h>
""", 
include_dirs=['/usr/local/magma/include/'], 
     library_dirs=['/usr/local/magma/lib/'], 
     extra_compile_args=['-DHAVE_CUBLAS=1'], 
     libraries=['magma'])

def magma_strerror(error):
    """
    Return string corresponding to specified MAGMA error code.
    """
    
    return _ffi_lib.magma_strerror(error)

class MAGMA_ERR(Exception):
    try:
        __doc__ = magma_strerror(-100)
    except:
        pass
    pass

MAGMA_EXCEPTIONS = {-100: MAGMA_ERR}
for k, v in _ffi_lib.__dict__.iteritems():
    if re.match('MAGMA_ERR_.*', k) and k != 'MAGMA_ERR':
        MAGMA_EXCEPTIONS[v] = vars()[k] = type(k, (MAGMA_ERR,), 
                                               {'__doc__': magma_strerror(v)})

def magmaCheckStatus(status):
    """
    Raise an exception corresponding to the specified MAGMA status code.
    """

    if status != 0:
        try:
            raise MAGMA_EXCEPTIONS[status]
        except KeyError:
            raise MAGMA_ERR

# Utility functions:
def magma_init():
    """
    Initialize MAGMA.
    """

    _ffi_lib.magma_init()

def magma_finalize():
    """
    Finalize MAGMA.
    """

    _ffi_lib.magma_finalize()

def magma_version():
    """
    Get MAGMA version.
    """

    major = _ffi.new('int *')
    minor = _ffi.new('int *')
    micro = _ffi.new('int *')

    _ffi_lib.magma_version(major, minor, micro)
    return major[0], minor[0], micro[0]

def magma_getdevice_arch():
    """
    Get device architecture.
    """

    return _ffi_lib.magma_getdevice_arch()

def magma_getdevice():
    """
    Get current device used by MAGMA.
    """

    dev = _ffi.new('int *')
    _ffi_lib.magma_getdevice(dev)
    return dev[0]

def magma_setdevice(dev):
    """
    Get current device used by MAGMA.
    """

    _ffi_lib.magma_setdevice(dev)

def magma_device_sync():
    """
    Synchronize device used by MAGMA.
    """

    _ffi_lib.magma_device_sync()

# BLAS routines
for (func_type, array_type) in {'s': 'magmaFloat_const_ptr',
                                'd': 'magmaDouble_const_ptr',
                                'c': 'magmaFloatComplex_const_ptr',
                                'z': 'magmaDoubleComplex_const_ptr'}.iteritems():
                                  
    # ISAMAX, IDAMAX, ICAMAX, IZAMAX
    func_name = 'magma_i%samax' % func_type
    code = """
    def {func_name}(n, dx, incx):    
        dx_ptr = _ffi.cast('{array_type}', dx.ptr)
        return getattr(_ffi_lib, '{func_name}')(n, dx_ptr, incx)
    """.format(func_name=func_name, array_type=array_type).strip()
    exec code in globals()
    globals()[func_name].__doc__ = \
    """
    Index of maximum magnitude element.
    """

    # ISAMIN, IDAMIN, ICAMIN, IZAMIN
    func_name = 'magma_i%samin' % func_type
    code = """
    def {func_name}(n, dx, incx):    
        dx_ptr = _ffi.cast('{array_type}', dx.ptr)
        return getattr(_ffi_lib, '{func_name}')(n, dx_ptr, incx)
    """.format(func_name=func_name, array_type=array_type).strip()
    exec code in vars()
    globals()[func_name].__doc__ = \
    """
    Index of minimum magnitude element.
    """

for (func_type, array_type) in {'s': 'magmaFloat_const_ptr',
                                'd': 'magmaDouble_const_ptr',
                                'sc': 'magmaFloatComplex_const_ptr',
                                'dz': 'magmaDoubleComplex_const_ptr'}.iteritems():

    # SASUM, DASUM, SCASUM, DZASUM
    func_name = 'magma_%sasum' % func_type
    code = """
    def {func_name}(n, dx, incx):           
        dx_ptr = _ffi.cast('{array_type}', dx.ptr)         
        return getattr(_ffi_lib, '{func_name}')(n, dx_ptr, incx)
    """.format(func_name=func_name, array_type=array_type).strip()
    exec code in vars()
    globals()[func_name].__doc__ = \
    """
    Sum of absolute values of vector.       
    """

# # SAXPY, DAXPY, CAXPY, ZAXPY
# def magma_saxpy(n, alpha, dx, incx, dy, incy):
#     """
#     Vector addition.
#     """

#     _ffi_lib.magma_saxpy(n, alpha, int(dx), incx, int(dy), incy)

# _ffi_lib.magma_daxpy.restype = int
# _ffi_lib.magma_daxpy.argtypes = [ctypes.c_int,
#                                   ctypes.c_double,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_daxpy(n, alpha, dx, incx, dy, incy):
#     """
#     Vector addition.
#     """

#     _ffi_lib.magma_daxpy(n, alpha, int(dx), incx, int(dy), incy)

# _ffi_lib.magma_caxpy.restype = int
# _ffi_lib.magma_caxpy.argtypes = [ctypes.c_int,
#                                   cuda.cuFloatComplex,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_caxpy(n, alpha, dx, incx, dy, incy):
#     """
#     Vector addition.
#     """

#     _ffi_lib.magma_caxpy(n, ctypes.byref(cuda.cuFloatComplex(alpha.real,
#                                                               alpha.imag)), 
#                           int(dx), incx, int(dy), incy)

# _ffi_lib.magma_zaxpy.restype = int
# _ffi_lib.magma_zaxpy.argtypes = [ctypes.c_int,
#                                   cuda.cuDoubleComplex,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_zaxpy(n, alpha, dx, incx, dy, incy):
#     """
#     Vector addition.
#     """

#     _ffi_lib.magma_zaxpy(n, ctypes.byref(cuda.cuDoubleComplex(alpha.real,
#                                                                alpha.imag)), 
#                           int(dx), incx, int(dy), incy)

# # SCOPY, DCOPY, CCOPY, ZCOPY
# _ffi_lib.magma_scopy.restype = int
# _ffi_lib.magma_scopy.argtypes = [ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_scopy(n, dx, incx, dy, incy):
#     """
#     Vector copy.
#     """

#     _ffi_lib.magma_scopy(n, int(dx), incx, int(dy), incy)

# _ffi_lib.magma_dcopy.restype = int
# _ffi_lib.magma_dcopy.argtypes = [ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_dcopy(n, dx, incx, dy, incy):
#     """
#     Vector copy.
#     """

#     _ffi_lib.magma_dcopy(n, int(dx), incx, int(dy), incy)

# _ffi_lib.magma_ccopy.restype = int
# _ffi_lib.magma_ccopy.argtypes = [ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_ccopy(n, dx, incx, dy, incy):
#     """
#     Vector copy.
#     """

#     _ffi_lib.magma_ccopy(n, int(dx), incx, int(dy), incy)

# _ffi_lib.magma_zcopy.restype = int
# _ffi_lib.magma_zcopy.argtypes = [ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_zcopy(n, dx, incx, dy, incy):
#     """
#     Vector copy.
#     """

#     _ffi_lib.magma_zcopy(n, int(dx), incx, int(dy), incy)

# # SDOT, DDOT, CDOTU, CDOTC, ZDOTU, ZDOTC
# _ffi_lib.magma_sdot.restype = ctypes.c_float
# _ffi_lib.magma_sdot.argtypes = [ctypes.c_int,
#                                  ctypes.c_void_p,
#                                  ctypes.c_int,
#                                  ctypes.c_void_p,
#                                  ctypes.c_int]
# def magma_sdot(n, dx, incx, dy, incy):
#     """
#     Vector dot product.
#     """

#     return _ffi_lib.magma_sdot(n, int(dx), incx, int(dy), incy)

# _ffi_lib.magma_ddot.restype = ctypes.c_double
# _ffi_lib.magma_ddot.argtypes = [ctypes.c_int,
#                                  ctypes.c_void_p,
#                                  ctypes.c_int,
#                                  ctypes.c_void_p,
#                                  ctypes.c_int]
# def magma_ddot(n, dx, incx, dy, incy):
#     """
#     Vector dot product.
#     """

#     return _ffi_lib.magma_ddot(n, int(dx), incx, int(dy), incy)

# _ffi_lib.magma_cdotc.restype = cuda.cuFloatComplex
# _ffi_lib.magma_cdotc.argtypes = [ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_cdotc(n, dx, incx, dy, incy):
#     """
#     Vector dot product.
#     """

#     return _ffi_lib.magma_cdotc(n, int(dx), incx, int(dy), incy)

# _ffi_lib.magma_cdotu.restype = cuda.cuFloatComplex
# _ffi_lib.magma_cdotu.argtypes = [ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_cdotu(n, dx, incx, dy, incy):
#     """
#     Vector dot product.
#     """

#     return _ffi_lib.magma_cdotu(n, int(dx), incx, int(dy), incy)

# _ffi_lib.magma_zdotc.restype = cuda.cuDoubleComplex
# _ffi_lib.magma_zdotc.argtypes = [ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_zdotc(n, dx, incx, dy, incy):
#     """
#     Vector dot product.
#     """

#     return _ffi_lib.magma_zdotc(n, int(dx), incx, int(dy), incy)

# _ffi_lib.magma_zdotu.restype = cuda.cuDoubleComplex
# _ffi_lib.magma_zdotu.argtypes = [ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_zdotu(n, dx, incx, dy, incy):
#     """
#     Vector dot product.
#     """

#     return _ffi_lib.magma_zdotu(n, int(dx), incx, int(dy), incy)

# # SNRM2, DNRM2, SCNRM2, DZNRM2
# _ffi_lib.magma_snrm2.restype = ctypes.c_float
# _ffi_lib.magma_snrm2.argtypes = [ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_snrm2(n, dx, incx):
#     """
#     Euclidean norm (2-norm) of vector.
#     """

#     return _ffi_lib.magma_snrm2(n, int(dx), incx)

# _ffi_lib.magma_dnrm2.restype = ctypes.c_double
# _ffi_lib.magma_dnrm2.argtypes = [ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_dnrm2(n, dx, incx):
#     """
#     Euclidean norm (2-norm) of vector.
#     """

#     return _ffi_lib.magma_dnrm2(n, int(dx), incx)

# _ffi_lib.magma_scnrm2.restype = ctypes.c_float
# _ffi_lib.magma_scnrm2.argtypes = [ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_int]
# def magma_scnrm2(n, dx, incx):
#     """
#     Euclidean norm (2-norm) of vector.
#     """

#     return _ffi_lib.magma_scnrm2(n, int(dx), incx)

# _ffi_lib.magma_dznrm2.restype = ctypes.c_double
# _ffi_lib.magma_dznrm2.argtypes = [ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_int]
# def magma_dznrm2(n, dx, incx):
#     """
#     Euclidean norm (2-norm) of vector.
#     """

#     return _ffi_lib.magma_dznrm2(n, int(dx), incx)

# # SROT, DROT, CROT, CSROT, ZROT, ZDROT
# _ffi_lib.magma_srot.argtypes = [ctypes.c_int,
#                                  ctypes.c_void_p,
#                                  ctypes.c_int,
#                                  ctypes.c_void_p,
#                                  ctypes.c_int,
#                                  ctypes.c_float,
#                                  ctypes.c_float]
# def magma_srot(n, dx, incx, dy, incy, dc, ds):
#     """
#     Apply a rotation to vectors.
#     """

#     _ffi_lib.magma_srot(n, int(dx), incx, int(dy), incy, dc, ds)

# # SROTM, DROTM
# _ffi_lib.magma_srotm.argtypes = [ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p]
# def magma_srotm(n, dx, incx, dy, incy, param):
#     """
#     Apply a real modified Givens rotation.
#     """

#     _ffi_lib.magma_srotm(n, int(dx), incx, int(dy), incy, param)

# # SROTMG, DROTMG
# _ffi_lib.magma_srotmg.argtypes = [ctypes.c_void_p,
#                                   ctypes.c_void_p,
#                                   ctypes.c_void_p,
#                                   ctypes.c_void_p,
#                                   ctypes.c_void_p]
# def magma_srotmg(d1, d2, x1, y1, param):
#     """
#     Construct a real modified Givens rotation matrix.
#     """

#     _ffi_lib.magma_srotmg(int(d1), int(d2), int(x1), int(y1), param)

# # SSCAL, DSCAL, CSCAL, CSCAL, CSSCAL, ZSCAL, ZDSCAL
# _ffi_lib.magma_sscal.argtypes = [ctypes.c_int,
#                                   ctypes.c_float,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_sscal(n, alpha, dx, incx):
#     """
#     Scale a vector by a scalar.
#     """

#     _ffi_lib.magma_sscal(n, alpha, int(dx), incx)

# # SSWAP, DSWAP, CSWAP, ZSWAP
# _ffi_lib.magma_sswap.argtypes = [ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_sswap(n, dA, ldda, dB, lddb):
#     """
#     Swap vectors.
#     """

#     _ffi_lib.magma_sswap(n, int(dA), ldda, int(dB), lddb)

# # SGEMV, DGEMV, CGEMV, ZGEMV
# _ffi_lib.magma_sgemv.argtypes = [ctypes.c_char,
#                                   ctypes.c_int,
#                                   ctypes.c_int,
#                                   ctypes.c_float,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_float,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_sgemv(trans, m, n, alpha, dA, ldda, dx, incx, beta,
#                 dy, incy):
#     """
#     Matrix-vector product for general matrix.
#     """

#     _ffi_lib.magma_sgemv(trans, m, n, alpha, int(dA), ldda, dx, incx,
#                           beta, int(dy), incy)

# # SGER, DGER, CGERU, CGERC, ZGERU, ZGERC
# _ffi_lib.magma_sger.argtypes = [ctypes.c_int,
#                                  ctypes.c_int,
#                                  ctypes.c_float,
#                                  ctypes.c_void_p,
#                                  ctypes.c_int,
#                                  ctypes.c_void_p,
#                                  ctypes.c_int,
#                                  ctypes.c_void_p,
#                                  ctypes.c_int]
# def magma_sger(m, n, alpha, dx, incx, dy, incy, dA, ldda):
#     """
#     Rank-1 operation on real general matrix.
#     """

#     _ffi_lib.magma_sger(m, n, alpha, int(dx), incx, int(dy), incy,
#                          int(dA), ldda)

# # SSYMV, DSYMV, CSYMV, ZSYMV
# _ffi_lib.magma_ssymv.argtypes = [ctypes.c_char,
#                                   ctypes.c_int,
#                                   ctypes.c_float,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_float,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_ssymv(uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy):
#     _ffi_lib.magma_ssymv(uplo, n, alpha, int(dA), ldda, int(dx), incx, beta,
#                           int(dy), incy)

# # SSYR, DSYR, CSYR, ZSYR
# _ffi_lib.magma_ssyr.argtypes = [ctypes.c_char,
#                                  ctypes.c_int,
#                                  ctypes.c_float,
#                                  ctypes.c_void_p,
#                                  ctypes.c_int,
#                                  ctypes.c_void_p,
#                                  ctypes.c_int]
# def magma_ssyr(uplo, n, alpha, dx, incx, dA, ldda):
#     _ffi_lib.magma_ssyr(uplo, n, alpha, int(dx), incx, int(dA), ldda)

# # SSYR2, DSYR2, CSYR2, ZSYR2
# _ffi_lib.magma_ssyr2.argtypes = [ctypes.c_char,
#                                  ctypes.c_int,
#                                  ctypes.c_float,
#                                  ctypes.c_void_p,
#                                  ctypes.c_int,
#                                  ctypes.c_void_p,
#                                  ctypes.c_int,
#                                  ctypes.c_void_p,
#                                  ctypes.c_int]
# def magma_ssyr2(uplo, n, alpha, dx, incx, dy, incy, dA, ldda):
#     _ffi_lib.magma_ssyr2(uplo, n, alpha, int(dx), incx, 
#                           int(dy), incy, int(dA), ldda)

# # STRMV, DTRMV, CTRMV, ZTRMV
# _ffi_lib.magma_strmv.argtypes = [ctypes.c_char,
#                                   ctypes.c_char,
#                                   ctypes.c_char,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_strmv(uplo, trans, diag, n,
#                 dA, ldda, dx, incx):
#     _ffi_lib.magma_strmv(uplo, trans, diag, n,
#                           int(dA), ldda, int(dx), incx)                          

# # STRSV, DTRSV, CTRSV, ZTRSV
# _ffi_lib.magma_strsv.argtypes = [ctypes.c_char,
#                                   ctypes.c_char,
#                                   ctypes.c_char,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_strsv(uplo, trans, diag, n,
#                 dA, ldda, dx, incx):
#     _ffi_lib.magma_strsv(uplo, trans, diag, n,
#                           int(dA), ldda, int(dx), incx)                          

# # SGEMM, DGEMM, CGEMM, ZGEMM
# _ffi_lib.magma_sgemm.argtypes = [ctypes.c_char,
#                                   ctypes.c_char,
#                                   ctypes.c_int,
#                                   ctypes.c_int,
#                                   ctypes.c_int,
#                                   ctypes.c_float,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_float,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_sgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta,
#                 dC, lddc):
#     _ffi_lib.magma_sgemm(transA, transB, m, n, k, alpha, 
#                           int(dA), ldda, int(dB), lddb,
#                           beta, int(dC), lddc)

# # SSYMM, DSYMM, CSYMM, ZSYMM
# _ffi_lib.magma_ssymm.argtypes = [ctypes.c_char,
#                                   ctypes.c_char,
#                                   ctypes.c_int,
#                                   ctypes.c_int,
#                                   ctypes.c_float,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_float,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_ssymm(side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta,
#                 dC, lddc):
#     _ffi_lib.magma_ssymm(side, uplo, m, n, alpha, 
#                           int(dA), ldda, int(dB), lddb,
#                           beta, int(dC), lddc)

# # SSYRK, DSYRK, CSYRK, ZSYRK
# _ffi_lib.magma_ssyrk.argtypes = [ctypes.c_char,
#                                   ctypes.c_char,
#                                   ctypes.c_int,
#                                   ctypes.c_int,
#                                   ctypes.c_float,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_float,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_ssyrk(uplo, trans, n, k, alpha, dA, ldda, beta,
#                 dC, lddc):
#     _ffi_lib.magma_ssyrk(uplo, trans, n, k, alpha, 
#                           int(dA), ldda, beta, int(dC), lddc)

# # SSYR2K, DSYR2K, CSYR2K, ZSYR2K
# _ffi_lib.magma_ssyr2k.argtypes = [ctypes.c_char,
#                                    ctypes.c_char,
#                                    ctypes.c_int,
#                                    ctypes.c_int,
#                                    ctypes.c_float,
#                                    ctypes.c_void_p,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_int,
#                                    ctypes.c_float,
#                                    ctypes.c_void_p,
#                                    ctypes.c_int]
# def magma_ssyr2k(uplo, trans, n, k, alpha, dA, ldda, 
#                  dB, lddb, beta, dC, lddc):                
#     _ffi_lib.magma_ssyr2k(uplo, trans, n, k, alpha, 
#                            int(dA), ldda, int(dB), lddb, 
#                            beta, int(dC), lddc)

# # STRMM, DTRMM, CTRMM, ZTRMM
# _ffi_lib.magma_strmm.argtypes = [ctypes.c_char,
#                                   ctypes.c_char,
#                                   ctypes.c_char,
#                                   ctypes.c_char,
#                                   ctypes.c_int,
#                                   ctypes.c_int,
#                                   ctypes.c_float,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_strmm(side, uplo, trans, diag, m, n, alpha, dA, ldda, 
#                 dB, lddb):                
#     _ffi_lib.magma_strmm(uplo, trans, diag, m, n, alpha, 
#                           int(dA), ldda, int(dB), lddb)

# # STRSM, DTRSM, CTRSM, ZTRSM
# _ffi_lib.magma_strsm.argtypes = [ctypes.c_char,
#                                   ctypes.c_char,
#                                   ctypes.c_char,
#                                   ctypes.c_char,
#                                   ctypes.c_int,
#                                   ctypes.c_int,
#                                   ctypes.c_float,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int]
# def magma_strsm(side, uplo, trans, diag, m, n, alpha, dA, ldda, 
#                 dB, lddb):                
#     _ffi_lib.magma_strsm(uplo, trans, diag, m, n, alpha, 
#                           int(dA), ldda, int(dB), lddb)


# # Auxiliary routines:
# _ffi_lib.magma_get_spotrf_nb.restype = int
# _ffi_lib.magma_get_spotrf_nb.argtypes = [ctypes.c_int]
# def magma_get_spotrf_nb(m):
#     return _ffi_lib.magma_get_spotrf_nb(m)

# _ffi_lib.magma_get_sgetrf_nb.restype = int
# _ffi_lib.magma_get_sgetrf_nb.argtypes = [ctypes.c_int]
# def magma_get_sgetrf_nb(m):
#     return _ffi_lib.magma_get_sgetrf_nb(m)

# _ffi_lib.magma_get_sgetri_nb.restype = int
# _ffi_lib.magma_get_sgetri_nb.argtypes = [ctypes.c_int]
# def magma_get_sgetri_nb(m):
#     return _ffi_lib.magma_get_sgetri_nb(m)

# _ffi_lib.magma_get_sgeqp3_nb.restype = int
# _ffi_lib.magma_get_sgeqp3_nb.argtypes = [ctypes.c_int]
# def magma_get_sgeqp3_nb(m):
#     return _ffi_lib.magma_get_sgeqp3_nb(m)

# _ffi_lib.magma_get_sgeqrf_nb.restype = int
# _ffi_lib.magma_get_sgeqrf_nb.argtypes = [ctypes.c_int]
# def magma_get_sgeqrf_nb(m):
#     return _ffi_lib.magma_get_sgeqrf_nb(m)

# _ffi_lib.magma_get_sgeqlf_nb.restype = int
# _ffi_lib.magma_get_sgeqlf_nb.argtypes = [ctypes.c_int]
# def magma_get_sgeqlf_nb(m):
#     return _ffi_lib.magma_get_sgeqlf_nb(m)

# _ffi_lib.magma_get_sgehrd_nb.restype = int
# _ffi_lib.magma_get_sgehrd_nb.argtypes = [ctypes.c_int]
# def magma_get_sgehrd_nb(m):
#     return _ffi_lib.magma_get_sgehrd_nb(m)

# _ffi_lib.magma_get_ssytrd_nb.restype = int
# _ffi_lib.magma_get_ssytrd_nb.argtypes = [ctypes.c_int]
# def magma_get_ssytrd_nb(m):
#     return _ffi_lib.magma_get_ssytrd_nb(m)

# _ffi_lib.magma_get_sgelqf_nb.restype = int
# _ffi_lib.magma_get_sgelqf_nb.argtypes = [ctypes.c_int]
# def magma_get_sgelqf_nb(m):
#     return _ffi_lib.magma_get_sgelqf_nb(m)

# _ffi_lib.magma_get_sgebrd_nb.restype = int
# _ffi_lib.magma_get_sgebrd_nb.argtypes = [ctypes.c_int]
# def magma_get_sgebrd_nb(m):
#     return _ffi_lib.magma_get_sgebrd_nb(m)

# _ffi_lib.magma_get_ssygst_nb.restype = int
# _ffi_lib.magma_get_ssygst_nb.argtypes = [ctypes.c_int]
# def magma_get_ssygst_nb(m):
#     return _ffi_lib.magma_get_ssgyst_nb(m)

# _ffi_lib.magma_get_sgesvd_nb.restype = int
# _ffi_lib.magma_get_sgesvd_nb.argtypes = [ctypes.c_int]
# def magma_get_sgesvd_nb(m):
#     return _ffi_lib.magma_get_sgesvd_nb(m)

# _ffi_lib.magma_get_ssygst_nb_m.restype = int
# _ffi_lib.magma_get_ssygst_nb_m.argtypes = [ctypes.c_int]
# def magma_get_ssygst_nb_m(m):
#     return _ffi_lib.magma_get_ssgyst_nb_m(m)

# _ffi_lib.magma_get_sbulge_nb.restype = int
# _ffi_lib.magma_get_sbulge_nb.argtypes = [ctypes.c_int]
# def magma_get_sbulge_nb(m):
#     return _ffi_lib.magma_get_sbulge_nb(m)

# _ffi_lib.magma_get_sbulge_nb_mgpu.restype = int
# _ffi_lib.magma_get_sbulge_nb_mgpu.argtypes = [ctypes.c_int]
# def magma_get_sbulge_nb_mgpu(m):
#     return _ffi_lib.magma_get_sbulge_nb_mgpu(m)

# # LAPACK routines

# # SGEBRD, DGEBRD, CGEBRD, ZGEBRD
# _ffi_lib.magma_sgebrd.restype = int
# _ffi_lib.magma_sgebrd.argtypes = [ctypes.c_int,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_void_p,
#                                    ctypes.c_void_p,
#                                    ctypes.c_void_p,
#                                    ctypes.c_void_p,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p]
# def magma_sgebrd(m, n, A, lda, d, e, tauq, taup, work, lwork, info):
#     """
#     Reduce matrix to bidiagonal form.
#     """

#     status = _ffi_lib.magma_sgebrd.argtypes(m, n, int(A), lda,
#                                              int(d), int(e),
#                                              int(tauq), int(taup),
#                                              int(work), int(lwork),
#                                              int(info))
#     magmaCheckStatus(status)

# # SGEHRD2, DGEHRD2, CGEHRD2, ZGEHRD2
# _ffi_lib.magma_sgehrd2.restype = int
# _ffi_lib.magma_sgehrd2.argtypes = [ctypes.c_int,
#                                     ctypes.c_int,
#                                     ctypes.c_int,
#                                     ctypes.c_void_p,
#                                     ctypes.c_int,
#                                     ctypes.c_void_p,
#                                     ctypes.c_void_p,
#                                     ctypes.c_int,
#                                     ctypes.c_void_p]
# def magma_sgehrd2(n, ilo, ihi, A, lda, tau,
#                   work, lwork, info):
#     """
#     Reduce matrix to upper Hessenberg form.
#     """
    
#     status = _ffi_lib.magma_sgehrd2(n, ilo, ihi, int(A), lda,
#                                      int(tau), int(work), 
#                                      lwork, int(info))
#     magmaCheckStatus(status)

# # SGEHRD, DGEHRD, CGEHRD, ZGEHRD
# _ffi_lib.magma_sgehrd.restype = int
# _ffi_lib.magma_sgehrd.argtypes = [ctypes.c_int,
#                                     ctypes.c_int,
#                                     ctypes.c_int,
#                                     ctypes.c_void_p,
#                                     ctypes.c_int,
#                                     ctypes.c_void_p,
#                                     ctypes.c_void_p,
#                                     ctypes.c_int,
#                                     ctypes.c_void_p]
# def magma_sgehrd(n, ilo, ihi, A, lda, tau,
#                  work, lwork, dT, info):
#     """
#     Reduce matrix to upper Hessenberg form (fast algorithm).
#     """
    
#     status = _ffi_lib.magma_sgehrd(n, ilo, ihi, int(A), lda,
#                                     int(tau), int(work), 
#                                     lwork, int(dT), int(info))
#     magmaCheckStatus(status)

# # SGELQF, DGELQF, CGELQF, ZGELQF
# _ffi_lib.magma_sgelqf.restype = int
# _ffi_lib.magma_sgelqf.argtypes = [ctypes.c_int,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_void_p,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p]
# def magma_sgelqf(m, n, A, lda, tau, work, lwork, info):
                 
#     """
#     LQ factorization.
#     """
    
#     status = _ffi_lib.magma_sgelqf(m, n, int(A), lda,
#                                     int(tau), int(work), 
#                                     lwork, int(info))
#     magmaCheckStatus(status)

# # SGEQRF, DGEQRF, CGEQRF, ZGEQRF
# _ffi_lib.magma_sgeqrf.restype = int
# _ffi_lib.magma_sgeqrf.argtypes = [ctypes.c_int,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_void_p,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p]
# def magma_sgeqrf(m, n, A, lda, tau, work, lwork, info):
                 
#     """
#     QR factorization.
#     """
    
#     status = _ffi_lib.magma_sgeqrf(m, n, int(A), lda,
#                                     int(tau), int(work), 
#                                     lwork, int(info))
#     magmaCheckStatus(status)

# # SGEQRF4, DGEQRF4, CGEQRF4, ZGEQRF4
# _ffi_lib.magma_sgeqrf4.restype = int
# _ffi_lib.magma_sgeqrf4.argtypes = [ctypes.c_int,
#                                     ctypes.c_int,
#                                     ctypes.c_int,
#                                     ctypes.c_void_p,
#                                     ctypes.c_int,
#                                     ctypes.c_void_p,
#                                     ctypes.c_void_p,
#                                     ctypes.c_int,
#                                     ctypes.c_void_p]
# def magma_sgeqrf4(num_gpus, m, n, a, lda, tau, work, lwork, info):
                 
#     """

#     """
    
#     status = _ffi_lib.magma_sgeqrf4(num_gpus, m, n, int(a), lda,
#                                     int(tau), int(work), 
#                                     lwork, int(info))
#     magmaCheckStatus(status)

# # SGEQRF, DGEQRF, CGEQRF, ZGEQRF (ooc)
# _ffi_lib.magma_sgeqrf_ooc.restype = int
# _ffi_lib.magma_sgeqrf_ooc.argtypes = [ctypes.c_int,
#                                        ctypes.c_int,
#                                        ctypes.c_void_p,
#                                        ctypes.c_int,
#                                        ctypes.c_void_p,
#                                        ctypes.c_void_p,
#                                        ctypes.c_int,
#                                        ctypes.c_void_p]
# def magma_sgeqrf_ooc(m, n, A, lda, tau, work, lwork, info):
                 
#     """
#     QR factorization (ooc).
#     """
    
#     status = _ffi_lib.magma_sgeqrf_ooc(m, n, int(A), lda,
#                                         int(tau), int(work), 
#                                         lwork, int(info))
#     magmaCheckStatus(status)

# # SGESV, DGESV, CGESV, ZGESV
# _ffi_lib.magma_sgesv.restype = int
# _ffi_lib.magma_sgesv.argtypes = [ctypes.c_int,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p]
# def magma_sgesv(n, nhrs, A, lda, ipiv, B, ldb, info):
                 
#     """
#     Solve system of linear equations.
#     """
    
#     status = _ffi_lib.magma_sgesv(n, nhrs, int(A), lda,
#                                    int(ipiv), int(B), 
#                                    ldb, int(info))
#     magmaCheckStatus(status)

# # SGETRF, DGETRF, CGETRF, ZGETRF
# _ffi_lib.magma_sgetrf.restype = int
# _ffi_lib.magma_sgetrf.argtypes = [ctypes.c_int,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_void_p]
# def magma_sgetrf(m, n, A, lda, ipiv, info):
                 
#     """
#     LU factorization.
#     """
    
#     status = _ffi_lib.magma_sgetrf(m, n, int(A), lda,
#                                     int(ipiv), int(info))   
#     magmaCheckStatus(status)

# # SGETRF2, DGETRF2, CGETRF2, ZGETRF2
# _ffi_lib.magma_sgetrf2.restype = int
# _ffi_lib.magma_sgetrf2.argtypes = [ctypes.c_int,
#                                     ctypes.c_int,
#                                     ctypes.c_void_p,
#                                     ctypes.c_int,
#                                     ctypes.c_void_p,
#                                     ctypes.c_void_p]
# def magma_sgetrf2(m, n, A, lda, ipiv, info):
                 
#     """
#     LU factorization (multi-GPU).
#     """
    
#     status = _ffi_lib.magma_sgetrf2(m, n, int(A), lda,
#                                     int(ipiv), int(info))
#     magmaCheckStatus(status)

# # SGEEV, DGEEV, CGEEV, ZGEEV
# _ffi_lib.magma_sgeev.restype = int
# _ffi_lib.magma_sgeev.argtypes = [ctypes.c_char,
#                                   ctypes.c_char,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_int,
#                                   ctypes.c_void_p,
#                                   ctypes.c_void_p]
# def magma_sgeev(jobvl, jobvr, n, a, lda,
#                 w, vl, ldvl, vr, ldvr, work, lwork, rwork, info):
                 
#     """
#     Compute eigenvalues and eigenvectors.
#     """

#     status = _ffi_lib.magma_sgeev(jobvl, jobvr, n, int(a), lda,
#                                    int(w), int(vl), ldvl, int(vr), ldvr, 
#                                    int(work), lwork, int(rwork), int(info))
#     magmaCheckStatus(status)

# # SGESVD, DGESVD, CGESVD, ZGESVD
# _ffi_lib.magma_sgesvd.restype = int
# _ffi_lib.magma_sgesvd.argtypes = [ctypes.c_char,
#                                    ctypes.c_char,
#                                    ctypes.c_int,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_void_p,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_void_p]
# def magma_sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
#                  rwork, info):
#     """
#     SVD decomposition.
#     """

#     status = _ffi_lib.magma_sgesvd(jobu, jobvt, m, n, 
#                                     int(a), lda, int(s), int(u), ldu,
#                                     int(vt), ldvt, int(work), lwork, 
#                                     int(rwork), int(info))
#     magmaCheckStatus(status)
