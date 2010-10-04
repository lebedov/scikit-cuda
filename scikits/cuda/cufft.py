#!/usr/bin/env python

"""
Python interface to CUFFT functions.

Note: this module does not explicitly depend on PyCUDA.
"""

import sys
import ctypes

if sys.platform == 'linux2':
    _libcufft_libname = 'libcufft.so'
elif sys.platform == 'darwin':
    _libcufft_libname = 'libcufft.dylib'
else:
    raise RuntimeError('unsupported platform')

# Print understandable error message when library cannot be found:
try:
    _libcufft = ctypes.cdll.LoadLibrary(_libcufft_libname)
except OSError:
    raise OSError('%s not found' % _libcufft_libname)
    
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

cufftExceptions = {
    0x1: cufftInvalidPlan,
    0x2: cufftAllocFailed,
    0x3: cufftInvalidType,
    0x4: cufftInvalidValue,
    0x5: cufftInternalError,
    0x6: cufftExecFailed,
    0x7: cufftSetupFailed,
    0x8: cufftInvalidSize,
    }

def cufftCheckStatus(status):
    """Raise an exception if the specified CUBLAS status is an
    error."""

    if status != 0:
        try:
            raise cufftExceptions[status]
        except KeyError:
            raise cufftError
        

# Data transformation types:
CUFFT_R2C = 0x2a
CUFFT_C2R = 0x2c
CUFFT_C2C = 0x29
CUFFT_D2Z = 0x6a
CUFFT_Z2D = 0x6c
CUFFT_Z2Z = 0x69

# Transformation directions
CUFFT_FORWARD = -1
CUFFT_INVERSE = 1

# FFT functions implemented by CUFFT:
_libcufft.cufftPlan1d.restype = int
_libcufft.cufftPlan1d.argtypes = [ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_int]
def cufftPlan1d(nx, fft_type, batch):
    """Create 1D FFT plan configuration."""

    plan = ctypes.c_void_p()
    status = _libcufft.cufftPlan1d(ctypes.byref(plan), nx, fft_type, batch)
    cufftCheckStatus(status)
    return plan

_libcufft.cufftPlan2d.restype = int
_libcufft.cufftPlan2d.argtypes = [ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_int]
def cufftPlan2d(nx, ny, fft_type):
    """Create 2D FFT plan configuration."""

    plan = ctypes.c_void_p()
    status = _libcufft.cufftPlan2d(ctypes.byref(plan), nx, ny,
                                   fft_type)
    cufftCheckStatus(status)
    return plan

_libcufft.cufftPlan3d.restype = int
_libcufft.cufftPlan3d.argtypes = [ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_int]
def cufftPlan3d(nx, ny, nz, fft_type):
    """Create 3D FFT plan configuration."""

    plan = ctypes.c_void_p()
    status = _libcufft.cufftPlan3d(ctypes.byref(plan), nx, ny, nz,
                                   fft_type)
    cufftCheckStatus(status)
    return plan

_libcufft.cufftDestroy.restype = int
_libcufft.cufftDestroy.argtypes = [ctypes.c_void_p]
def cufftDestroy(plan):
    """Destroy FFT plan."""
    
    status = _libcufft.cufftDestroy(plan)
    cufftCheckStatus(status)

_libcufft.cufftExecC2C.restype = int
_libcufft.cufftExecC2C.argtypes = [ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cufftExecC2C(plan, idata, odata, direction):
    """Execute single precision complex-to-complex transform plan as
    specified by `direction`."""
    
    status = _libcufft.cufftExecC2C(plan, idata, odata,
                                    direction)
    cufftCheckStatus(status)

_libcufft.cufftExecR2C.restype = int
_libcufft.cufftExecR2C.argtypes = [ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def cufftExecR2C(plan, idata, odata):
    """Execute single precision real-to-complex forward transform plan."""
    
    status = _libcufft.cufftExecR2C(plan, idata, odata)
    cufftCheckStatus(status)

_libcufft.cufftExecC2R.restype = int
_libcufft.cufftExecC2R.argtypes = [ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def cufftExecC2R(plan, idata, odata):
    """Execute single precision complex-to-real reverse transform plan."""
    
    status = _libcufft.cufftExecC2R(plan, idata, odata)
    cufftCheckStatus(status)

_libcufft.cufftExecZ2Z.restype = int
_libcufft.cufftExecZ2Z.argtypes = [ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cufftExecZ2Z(plan, idata, odata, direction):
    """Execute double precision complex-to-complex transform plan as
    specified by `direction`."""
    
    status = _libcufft.cufftExecZ2Z(plan, idata, odata,
                                    direction)
    cufftCheckStatus(status)

_libcufft.cufftExecD2Z.restype = int
_libcufft.cufftExecD2Z.argtypes = [ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def cufftExecD2Z(plan, idata, odata):
    """Execute double precision real-to-complex forward transform plan."""
    
    status = _libcufft.cufftExecD2Z(plan, idata, odata)
    cufftCheckStatus(status)

_libcufft.cufftExecZ2D.restype = int
_libcufft.cufftExecZ2D.argtypes = [ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def cufftExecZ2D(plan, idata, odata):
    """Execute double precision complex-to-real transform plan."""
    
    status = _libcufft.cufftExecZ2D(plan, idata, odata)
    cufftCheckStatus(status)

