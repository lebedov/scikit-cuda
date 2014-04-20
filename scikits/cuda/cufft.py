#!/usr/bin/env python

"""
Python interface to CUFFT functions.

Note: this module does not explicitly depend on PyCUDA.
"""

import re
import cffi
_ffi = cffi.FFI()

# General CUFFT error:
class cufftError(Exception):
    """CUFFT error"""
    pass

# Exceptions corresponding to different CUFFT errors:
class cufftInvalidPlan(cufftError):
    """CUFFT was passed an invalid plan handle."""
    pass

class cufftAllocFailed(cufftError):
    """CUFFT failed to allocate GPU memory."""
    pass

class cufftInvalidType(cufftError):
    """The user requested an unsupported type."""
    pass

class cufftInvalidValue(cufftError):
    """The user specified a bad memory pointer."""
    pass

class cufftInternalError(cufftError):
    """Internal driver error."""
    pass

class cufftExecFailed(cufftError):
    """CUFFT failed to execute an FFT on the GPU."""
    pass

class cufftSetupFailed(cufftError):
    """The CUFFT library failed to initialize."""
    pass

class cufftInvalidSize(cufftError):
    """The user specified an unsupported FFT size."""
    pass

class cufftUnalignedData(cufftError):
    """Input or output does not satisfy texture alignment requirements."""
    pass

_ffi.cdef("""
typedef struct CUstream_st *cudaStream_t;

typedef struct float2 {
    ...;
} float2;
typedef float2 cuFloatComplex;
typedef float2 cuComplex;

typedef struct double2 {
    ...;
} double2;
typedef double2 cuDoubleComplex;

typedef float cufftReal;
typedef double cufftDoubleReal;

typedef cuComplex cufftComplex;
typedef cuDoubleComplex cufftDoubleComplex;

// Result status codes:
typedef enum cufftResult_t {
  CUFFT_SUCCESS        = 0x0,
  CUFFT_INVALID_PLAN   = 0x1,
  CUFFT_ALLOC_FAILED   = 0x2,
  CUFFT_INVALID_TYPE   = 0x3,
  CUFFT_INVALID_VALUE  = 0x4,
  CUFFT_INTERNAL_ERROR = 0x5,
  CUFFT_EXEC_FAILED    = 0x6,
  CUFFT_SETUP_FAILED   = 0x7,
  CUFFT_INVALID_SIZE   = 0x8,
  CUFFT_UNALIGNED_DATA = 0x9
} cufftResult;

// Data transformation types:
typedef enum cufftType_t {
  CUFFT_R2C = 0x2a,     // Real to Complex (interleaved)
  CUFFT_C2R = 0x2c,     // Complex (interleaved) to Real
  CUFFT_C2C = 0x29,     // Complex to Complex, interleaved
  CUFFT_D2Z = 0x6a,     // Double to Double-Complex
  CUFFT_Z2D = 0x6c,     // Double-Complex to Double
  CUFFT_Z2Z = 0x69      // Double-Complex to Double-Complex
} cufftType;

// FFT direction:
#define CUFFT_FORWARD ... // -1, Forward FFT
#define CUFFT_INVERSE ... // 1, Inverse FFT

typedef enum cufftCompatibility_t {
    CUFFT_COMPATIBILITY_NATIVE          = 0x00, 
    CUFFT_COMPATIBILITY_FFTW_PADDING    = 0x01,    // The default value
    CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC = 0x02,
    CUFFT_COMPATIBILITY_FFTW_ALL        = 0x03
} cufftCompatibility;

typedef int cufftHandle;

cufftResult cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch);
cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type);
cufftResult cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType type);
cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n,
                          int *inembed, int istride, int idist,
                          int *onembed, int ostride, int odist,
                          cufftType type, int batch);
                                                                    
cufftResult cufftExecC2C(cufftHandle plan, cufftComplex *idata, 
                         cufftComplex *odata, int direction);                         
cufftResult cufftExecR2C(cufftHandle plan, 
                         cufftReal *idata,
                         cufftComplex *odata);
cufftResult cufftExecC2R(cufftHandle plan, 
                         cufftComplex *idata,
                         cufftReal *odata);
cufftResult cufftExecZ2Z(cufftHandle plan, 
                         cufftDoubleComplex *idata,
                         cufftDoubleComplex *odata,
                         int direction);
cufftResult cufftExecD2Z(cufftHandle plan, 
                         cufftDoubleReal *idata,
                         cufftDoubleComplex *odata);
cufftResult cufftExecZ2D(cufftHandle plan, 
                         cufftDoubleComplex *idata,
                         cufftDoubleReal *odata);

cufftResult cufftDestroy(cufftHandle plan);
cufftResult cufftSetStream(cufftHandle plan,
                           cudaStream_t stream);
cufftResult cufftSetCompatibilityMode(cufftHandle plan,
                                      cufftCompatibility mode);
cufftResult cufftGetVersion(int *version);
""")

_ffi_lib = _ffi.verify("""
#include <cuda.h>
#include <cufft.h>
""", libraries=['cufft'], library_dirs=['/usr/local/cuda/lib64/'], include_dirs=['/usr/local/cuda/include/'])

# Import all CUFFT* definitions directly into module namespace:
for k, v in _ffi_lib.__dict__.iteritems():
    if re.match('CUFFT.*', k):
        vars()[k] = v

cufftExceptions = {
    CUFFT_INVALID_PLAN: cufftInvalidPlan,
    CUFFT_ALLOC_FAILED: cufftAllocFailed,
    CUFFT_INVALID_TYPE: cufftInvalidType,
    CUFFT_INTERNAL_ERROR: cufftInternalError,
    CUFFT_EXEC_FAILED: cufftExecFailed,
    CUFFT_SETUP_FAILED: cufftSetupFailed,
    CUFFT_INVALID_SIZE: cufftInvalidSize,
    CUFFT_UNALIGNED_DATA: cufftUnalignedData
}

def cufftCheckStatus(status):
    """
    Raise an exception if the specified CUFFT status is an error.

    Parameters
    ----------
    status : int
        CUFFT status.
    """

    if status != 0:
        try:
            raise cufftExceptions[status]
        except KeyError:
            raise cufftError

def cufftPlan1d(nx, fft_type, batch):
    """
    Create 1D FFT plan configuration.

    Parameters
    ----------
    nx : int
        Transform size.
    fft_type : int
        Transform type.
    batch : int
        Transform batch size.

    Returns
    -------
    plan : int
        CUFFT plan handle.
    """

    plan = _ffi.new('cufftHandle *')
    status = _ffi_lib.cufftPlan1d(plan, nx, fft_type, batch)
    cufftCheckStatus(status)
    return plan[0]

def cufftPlan2d(nx, ny, fft_type):
    """
    Create 2D FFT plan configuration.

    Parameters
    ----------
    nx : int
        Transform size in x direction.
    ny : int
        Transform size in y direction.
    fft_type : int
        Transform type.

    Returns
    -------
    plan : int
        CUFFT plan handle.
    """

    plan = _ffi.new('cufftHandle *')
    status = _ffi_lib.cufftPlan2d(ctypes.byref(plan), nx, ny,
                                  fft_type)
    cufftCheckStatus(status)
    return plan

def cufftPlan3d(nx, ny, nz, fft_type):
    """
    Create 3D FFT plan configuration.

    Parameters
    ----------
    nx : int
        Transform size in x direction.
    ny : int
        Transform size in y direction.
    nz : int
        Transform size in z direction.
    fft_type : int
        Transform type.

    Returns
    -------
    plan : int
        CUFFT plan handle.
    """

    plan = _ffi.new('cufftHandle *')
    status = _ffi_lib.cufftPlan3d(plan, nx, ny, nz,
                                  fft_type)
    cufftCheckStatus(status)
    return plan[0]

def cufftPlanMany(rank, n, 
                  inembed, istride, idist, 
                  onembed, ostride, odist, fft_type, batch):
    """
    Create batched FFT plan configuration.
    
    Parameters
    ----------
    rank : int
        Dimensionality of transform.
    n : int
        Array of size `rank` describing the size of each dimension.
    inembed : int
        Pointer of size `rank` that indicates the storage dimensions 
        of the input data in memory.
    istride : int
        Distance between two successive input elements in the innermost
        dimension.
    idist : int
        Distance between the first element of two consecutive sequences in a
        batch of input data.
    onembed : int
        Pointer of size `rank` that indicates the storage dimensions 
        of the output data in memory.
    ostride : int
        Distance between two successive output elements in the innermost
        dimension.
    odist : int
        Distance between the first element of two consecutive sequences in a
        batch of output data.
    type : int
        Transform type.
    batch : int
        Transform batch size.

    Returns
    -------
    plan : int
        CUFFT plan handle.
    """

    plan = _ffi.new('cufftHandle *')
    if inembed is None:
        inembed = 0
    if onembed is None:
        onembed = 0
    status = _ffi_lib.cufftPlanMany(plan, rank, 
                                    _ffi.cast('int *', n),
                                    _ffi.cast('int *', inembed), istride, idist, 
                                    _ffi.cast('int *', onembed), ostride, odist, 
                                    fft_type, batch)
    cufftCheckStatus(status)
    return plan[0]

def cufftExecC2C(plan, idata, odata, direction):
    """
    Execute single precision complex-to-complex transform plan.

    Parameters
    ----------
    plan : int
        CUFFT plan handle.
    idata : int
        Pointer to complex input data.
    odata : int
        Pointer to complex output data.
    """
    
    status = _ffi_lib.cufftExecC2C(plan, 
                                   _ffi.cast('cufftComplex *', idata), 
                                   _ffi.cast('cufftComplex *', odata),
                                   direction)
    cufftCheckStatus(status)

def cufftExecR2C(plan, idata, odata):
    """
    Execute single precision real-to-complex forward transform plan.

    Parameters
    ----------
    plan : int
        CUFFT plan handle.
    idata : int
        Pointer to real input data.
    odata : int
        Pointer to complex output data.
    """
    
    status = _ffi_lib.cufftExecR2C(plan, 
                                   _ffi.cast('cufftReal *', idata), 
                                   _ffi.cast('cufftComplex *', odata))
    cufftCheckStatus(status)

def cufftExecC2R(plan, idata, odata):
    """
    Execute single precision complex-to-real reverse transform plan.

    Parameters
    ----------
    plan : int
        CUFFT plan handle.
    idata : int
        Pointer to complex input data.
    odata : int
        Pointer to real output data.
    """
    
    status = _ffi_lib.cufftExecC2R(plan, 
                                   _ffi.cast('cufftComplex *', idata), 
                                   _ffi.cast('cufftReal *', odata))
    cufftCheckStatus(status)

def cufftExecZ2Z(plan, idata, odata, direction):
    """
    Execute double precision complex-to-complex transform plan.

    Parameters
    ----------
    plan : int
        CUFFT plan handle.
    idata : int
        Pointer to complex input data.
    odata : int
        Pointer to complex output data.
    """
    
    status = _ffi_lib.cufftExecZ2Z(plan, 
                                   _ffi.cast('cufftDoubleComplex *', idata), 
                                   _ffi.cast('cufftDoubleComplex *', odata),
                                   direction)
    cufftCheckStatus(status)

def cufftExecD2Z(plan, idata, odata):
    """Execute double precision real-to-complex forward transform plan.

    Parameters
    ----------
    plan : int
        CUFFT plan handle.
    idata : int
        Pointer to real input data.
    odata : int
        Pointer to complex output data.
    """
    
    status = _ffi_lib.cufftExecD2Z(plan, 
                                   _ffi.cast('cufftDoubleReal *', idata), 
                                   _ffi.cast('cufftDoubleComplex *', odata))
    cufftCheckStatus(status)

def cufftExecZ2D(plan, idata, odata):
    """
    Execute double precision complex-to-real reverse transform plan.

    Parameters
    ----------
    plan : int
        CUFFT plan handle.
    idata : int
        Pointer to complex input data.
    odata : int
        Pointer to real output data.
    """
    
    status = _ffi_lib.cufftExecZ2D(plan, 
                                   _ffi.cast('cufftDoubleComplex *', idata), 
                                   _ffi.cast('cufftDoubleReal *', odata))
    cufftCheckStatus(status)

def cufftDestroy(plan):
    """
    Destroy FFT plan.

    Parameters
    ----------
    plan : int
        CUFFT plan handle.
    """
    
    status = _ffi_lib.cufftDestroy(plan)
    cufftCheckStatus(status)

def cufftSetStream(plan, stream):
    """
    Associate a CUDA stream with a CUFFT plan.

    Parameters
    ----------
    plan : int
        CUFFT plan handle.
    stream : int
        CUDA stream identifier.
    """

    status = _ffi_lib.cufftSetStream(plan, mode)
    cufftCheckStatus(status)

def cufftSetCompatibilityMode(plan, mode):
    """
    Set FFTW compatibility mode.

    Parameters
    ----------
    plan : int
        CUFFT plan handle
    mode : int
        FFTW compatibility mode.    
    """

    status = _ffi_lib.cufftSetCompatibilityMode(plan, mode)
    cufftCheckStatus(status)

def cufftGetVersion():
    """
    Get CUFFT version.

    Returns
    -------
    version : int
        CUFFT version.
    """

    version = ffi.new('int *');    
    status = _ffi_lib.cufftGetVersion(version)
    cufftCheckStatus(status)
    return version[0]
