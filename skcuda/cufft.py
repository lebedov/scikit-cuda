#!/usr/bin/env python

"""
Python interface to CUFFT functions.

Note: this module does not explicitly depend on PyCUDA.
"""

import ctypes
import operator
import re
import sys

# Load library:
_linux_version_list = [10.1, 10.0, 9.2, 9.1, 9.0, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.0]
_win32_version_list = [10, 100, 92, 91, 90, 80, 75, 70, 65, 60, 55, 50, 40]
if 'linux' in sys.platform:
    _libcufft_libname_list = ['libcufft.so'] + \
                             ['libcufft.so.%s' % v for v in _linux_version_list]
elif sys.platform == 'darwin':
    _libcufft_libname_list = ['libcufft.dylib']
elif sys.platform == 'win32':
    if sys.maxsize > 2**32:
        _libcufft_libname_list = ['cufft.dll'] + \
            ['cufft64_%s.dll' % v for v in _win32_version_list]
    else:
        _libcufft_libname_list = ['cufft.dll'] + \
            ['cufft32_%s.dll' % v for v in _win32_version_list]

else:
    raise RuntimeError('unsupported platform')

# Print understandable error message when library cannot be found:
_libcufft = None
for _libcufft_libname in _libcufft_libname_list:
    try:
        if sys.platform == 'win32':
            _libcufft = ctypes.windll.LoadLibrary(_libcufft_libname)
        else:
            _libcufft = ctypes.cdll.LoadLibrary(_libcufft_libname)
    except OSError:
        pass
    else:
        break
if _libcufft == None:
    raise OSError('cufft library not found')

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

cufftExceptions = {
    0x1: cufftInvalidPlan,
    0x2: cufftAllocFailed,
    0x3: cufftInvalidType,
    0x4: cufftInvalidValue,
    0x5: cufftInternalError,
    0x6: cufftExecFailed,
    0x7: cufftSetupFailed,
    0x8: cufftInvalidSize,
    0x9: cufftUnalignedData
    }

class _types:
    """Some alias types."""
    plan = ctypes.c_int
    stream = ctypes.c_void_p
    worksize = ctypes.c_size_t

def cufftCheckStatus(status):
    """Raise an exception if the specified CUBLAS status is an error."""

    if status != 0:
        try:
            e = cufftExceptions[status]
        except KeyError:
            raise cufftError
        else:
            raise e

_libcufft.cufftGetVersion.restype = int
_libcufft.cufftGetVersion.argtypes = [ctypes.c_void_p]

def cufftGetVersion():
    """
    Get CUFFT version.
    """

    version = ctypes.c_int()
    result = _libcufft.cufftGetVersion(ctypes.byref(version))
    cufftCheckStatus(result)
    return version.value

_cufft_version = int(cufftGetVersion())

class _cufft_version_req(object):
    """
    Decorator to replace function with a placeholder that raises an exception
    if a specified condition on the installed CUFFT version `v` is not satisfied.
    """

    def __init__(self, v, op):
        self.op_str = op
        if op == '>':
            self.op = operator.gt
        elif op == '>=':
            self.op = operator.ge
        elif op == '==':
            self.op = operator.eq
        elif op == '<':
            self.op = operator.lt
        elif op == '<=':
            self.op = operator.le
        else:
            raise ValueError('unrecognized comparison operator')
        self.vs = str(v)
        if isinstance(v, int):
            self.vi = str(v)
            if len(self.vi) != 4:
                raise ValueError('integer version number must be 4 digits')
        else:
            major, minor = re.search(r'(\d+)\.(\d+)', self.vs).groups()
            self.vi = major.ljust(len(major)+1, '0')+minor.ljust(2, '0')

    def __call__(self,f):
        def f_new(*args,**kwargs):
            raise NotImplementedError('CUFFT '+self.op_str+' '+self.vs+' required')
        f_new.__doc__ = f.__doc__

        if self.op(_cufft_version, int(self.vi)):
            return f
        else:
            return f_new

# Data transformation types:
CUFFT_R2C = 0x2a
CUFFT_C2R = 0x2c
CUFFT_C2C = 0x29
CUFFT_D2Z = 0x6a
CUFFT_Z2D = 0x6c
CUFFT_Z2Z = 0x69

# Transformation directions:
CUFFT_FORWARD = -1
CUFFT_INVERSE = 1

# FFTW compatibility modes:
CUFFT_COMPATIBILITY_NATIVE = 0x00
CUFFT_COMPATIBILITY_FFTW_PADDING = 0x01
CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC = 0x02
CUFFT_COMPATIBILITY_FFTW_ALL = 0x03

# FFT functions implemented by CUFFT:
_libcufft.cufftPlan1d.restype = int
_libcufft.cufftPlan1d.argtypes = [ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_int]
def cufftPlan1d(nx, fft_type, batch):
    """
    Create 1D FFT plan configuration.

    References
    ----------
    `cufftPlan1d <http://docs.nvidia.com/cuda/cufft/#function-cufftplan1d>`_
    """

    plan = _types.plan()
    status = _libcufft.cufftPlan1d(ctypes.byref(plan), nx, fft_type, batch)
    cufftCheckStatus(status)
    return plan

_libcufft.cufftPlan2d.restype = int
_libcufft.cufftPlan2d.argtypes = [ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_int]
def cufftPlan2d(nx, ny, fft_type):
    """
    Create 2D FFT plan configuration.

    References
    ----------
    `cufftPlan2d <http://docs.nvidia.com/cuda/cufft/#function-cufftplan2d>`_
    """

    plan = _types.plan()
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
    """
    Create 3D FFT plan configuration.

    References
    ----------
    `cufftPlan3d <http://docs.nvidia.com/cuda/cufft/#function-cufftplan3d>`_
    """

    plan = _types.plan()
    status = _libcufft.cufftPlan3d(ctypes.byref(plan), nx, ny, nz,
                                   fft_type)
    cufftCheckStatus(status)
    return plan

_libcufft.cufftPlanMany.restype = int
_libcufft.cufftPlanMany.argtypes = [ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int]
def cufftPlanMany(rank, n,
                  inembed, istride, idist,
                  onembed, ostride, odist, fft_type, batch):
    """
    Create batched FFT plan configuration.

    References
    ----------
    `cufftPlanMany <http://docs.nvidia.com/cuda/cufft/#function-cufftplanmany>`_
    """

    plan = _types.plan()
    status = _libcufft.cufftPlanMany(ctypes.byref(plan), rank, n,
                                     inembed, istride, idist,
                                     onembed, ostride, odist,
                                     fft_type, batch)
    cufftCheckStatus(status)
    return plan

_libcufft.cufftDestroy.restype = int
_libcufft.cufftDestroy.argtypes = [_types.plan]
def cufftDestroy(plan):
    """
    Destroy FFT plan.

    References
    ----------
    `cufftDestroy <http://docs.nvidia.com/cuda/cufft/#function-cufftdestroy>`_
    """

    status = _libcufft.cufftDestroy(plan)
    cufftCheckStatus(status)

if _cufft_version <= 9010:
    _libcufft.cufftSetCompatibilityMode.restype = int
    _libcufft.cufftSetCompatibilityMode.argtypes = [_types.plan,
                                                    ctypes.c_int]
@_cufft_version_req(9.1, '<=')
def cufftSetCompatibilityMode(plan, mode):
    """
    Set FFTW compatibility mode.

    References
    ----------
    `cufftSetCompatibilityMode <http://docs.nvidia.com/cuda/cufft/#function-cufftsetcompatibilitymode>`_
    """

    status = _libcufft.cufftSetCompatibilityMode(plan, mode)
    cufftCheckStatus(status)

_libcufft.cufftExecC2C.restype = int
_libcufft.cufftExecC2C.argtypes = [_types.plan,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cufftExecC2C(plan, idata, odata, direction):
    """Execute single precision complex-to-complex transform plan as
    specified by `direction`.

    References
    ----------
    `cufftExecC2C <http://docs.nvidia.com/cuda/cufft/#function-cufftexecc2c-cufftexecz2z>`_
    """

    status = _libcufft.cufftExecC2C(plan, idata, odata,
                                    direction)
    cufftCheckStatus(status)

_libcufft.cufftExecR2C.restype = int
_libcufft.cufftExecR2C.argtypes = [_types.plan,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def cufftExecR2C(plan, idata, odata):
    """
    Execute single precision real-to-complex forward transform plan.

    References
    ----------
    `cufftExecR2C <http://docs.nvidia.com/cuda/cufft/#function-cufftexecr2c-cufftexecd2z>`_
    """

    status = _libcufft.cufftExecR2C(plan, idata, odata)
    cufftCheckStatus(status)

_libcufft.cufftExecC2R.restype = int
_libcufft.cufftExecC2R.argtypes = [_types.plan,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def cufftExecC2R(plan, idata, odata):
    """
    Execute single precision complex-to-real reverse transform plan.

    References
    ----------
    `cufftExecC2R <http://docs.nvidia.com/cuda/cufft/#function-cufftexecc2r-cufftexecz2d>`_
    """

    status = _libcufft.cufftExecC2R(plan, idata, odata)
    cufftCheckStatus(status)

_libcufft.cufftExecZ2Z.restype = int
_libcufft.cufftExecZ2Z.argtypes = [_types.plan,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cufftExecZ2Z(plan, idata, odata, direction):
    """
    Execute double precision complex-to-complex transform plan as
    specified by `direction`.

    References
    ----------
    `cufftExecZ2Z <http://docs.nvidia.com/cuda/cufft/#function-cufftexecc2c-cufftexecz2z>`_
"""

    status = _libcufft.cufftExecZ2Z(plan, idata, odata,
                                    direction)
    cufftCheckStatus(status)

_libcufft.cufftExecD2Z.restype = int
_libcufft.cufftExecD2Z.argtypes = [_types.plan,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def cufftExecD2Z(plan, idata, odata):
    """
    Execute double precision real-to-complex forward transform plan.

    References
    ----------
    `cufftExecD2Z <http://docs.nvidia.com/cuda/cufft/#function-cufftexecr2c-cufftexecd2z>`_
    """

    status = _libcufft.cufftExecD2Z(plan, idata, odata)
    cufftCheckStatus(status)

_libcufft.cufftExecZ2D.restype = int
_libcufft.cufftExecZ2D.argtypes = [_types.plan,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def cufftExecZ2D(plan, idata, odata):
    """
    Execute double precision complex-to-real transform plan.

    References
    ----------
    `cufftExecZ2D <http://docs.nvidia.com/cuda/cufft/#function-cufftexecc2r-cufftexecz2d>`_
    """

    status = _libcufft.cufftExecZ2D(plan, idata, odata)
    cufftCheckStatus(status)

_libcufft.cufftSetStream.restype = int
_libcufft.cufftSetStream.argtypes = [_types.plan,
                                     _types.stream]
def cufftSetStream(plan, stream):
    """
    Associate a CUDA stream with a CUFFT plan.

    References
    ----------
    `cufftSetStream <http://docs.nvidia.com/cuda/cufft/#function-cufftsetstream>`_
    """

    status = _libcufft.cufftSetStream(plan, stream)
    cufftCheckStatus(status)

_libcufft.cufftEstimate1d.restype = int
_libcufft.cufftEstimate1d.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cufftEstimate1d(nx, fft_type, batch=1):
    """
    Return estimated work area for 1D FFT.

    References
    ----------
    `cufftEstimate1d <http://docs.nvidia.com/cuda/cufft/#function-cufftestimate1d>`_
    """

    worksize = _types.worksize()
    status = _libcufft.cufftEstimate1d(nx, fft_type, batch,
                                       ctypes.byref(worksize))
    cufftCheckStatus(status)
    return worksize.value

_libcufft.cufftEstimate2d.restype = int
_libcufft.cufftEstimate2d.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cufftEstimate2d(nx, ny, fft_type):
    """
    Return estimated work area for 2D FFT.

    References
    ----------
    `cufftEstimate2d <http://docs.nvidia.com/cuda/cufft/#function-cufftestimate2d>`_
    """
    worksize = _types.worksize()
    status = _libcufft.cufftEstimate2d(nx, ny, fft_type,
                                       ctypes.byref(worksize))
    cufftCheckStatus(status)
    return worksize.value

_libcufft.cufftEstimate3d.restype = int
_libcufft.cufftEstimate3d.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cufftEstimate3d(nx, ny, nz, fft_type):
    """
    Return estimated work area for 3D FFT.

    References
    ----------
    `cufftEstimate3d <http://docs.nvidia.com/cuda/cufft/#function-cufftestimate3d>`_
    """
    worksize = _types.worksize()
    status = _libcufft.cufftEstimate3d(nx, ny, nz, fft_type,
                                       ctypes.byref(worksize))
    cufftCheckStatus(status)
    return worksize.value

_libcufft.cufftEstimateMany.restype = int
_libcufft.cufftEstimateMany.argtypes = [ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_void_p]
def cufftEstimateMany(rank, n,
                      inembed, istride, idist,
                      onembed, ostride, odist, fft_type, batch):
    """
    Return estimated work area for batched FFT.

    References
    ----------
    `cufftEstimateMany <http://docs.nvidia.com/cuda/cufft/#function-cufftestimatemany>`_
    """

    worksize = _types.worksize()
    status = _libcufft.cufftEstimateMany(rank, n,
                                         inembed, istride, idist,
                                         onembed, ostride, odist,
                                         fft_type, batch,
                                         ctypes.byref(worksize))
    cufftCheckStatus(status)
    return worksize.value

_libcufft.cufftGetSize1d.restype = int
_libcufft.cufftGetSize1d.argtypes = [_types.plan,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p]
def cufftGetSize1d(plan, nx, fft_type, batch=1):
    """
    Return more accurate estimate of work area size required for 1D FFT,
    taking into account any plan settings that may have been made.

    References
    ----------
    `cufftGetSize1d <http://docs.nvidia.com/cuda/cufft/#function-cufftgetsize1d>`_
    """

    worksize = _types.worksize()
    status = _libcufft.cufftGetSize1d(plan, nx, fft_type, batch,
                                      ctypes.byref(worksize))
    cufftCheckStatus(status)
    return worksize.value

_libcufft.cufftGetSize2d.restype = int
_libcufft.cufftGetSize2d.argtypes = [_types.plan,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p]
def cufftGetSize2d(plan, nx, ny, fft_type):
    """
    Return more accurate estimate of work area size required for 2D FFT,
    taking into account any plan settings that may have been made.

    References
    ----------
    `cufftGetSize2d <http://docs.nvidia.com/cuda/cufft/#function-cufftgetsize2d>`_
    """
    worksize = _types.worksize()
    status = _libcufft.cufftGetSize2d(plan, nx, ny, fft_type,
                                      ctypes.byref(worksize))
    cufftCheckStatus(status)
    return worksize.value

_libcufft.cufftGetSize3d.restype = int
_libcufft.cufftGetSize3d.argtypes = [_types.plan,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p]
def cufftGetSize3d(plan, nx, ny, nz, fft_type):
    """
    Return more accurate estimate of work area size required for 3D FFT,
    taking into account any plan settings that may have been made.

    References
    ----------
    `cufftGetSize3d <http://docs.nvidia.com/cuda/cufft/#function-cufftgetsize3d>`_
    """
    worksize = _types.worksize()
    status = _libcufft.cufftGetSize3d(plan, nx, ny, nz, fft_type,
                                      ctypes.byref(worksize))
    cufftCheckStatus(status)
    return worksize.value

_libcufft.cufftGetSizeMany.restype = int
_libcufft.cufftGetSizeMany.argtypes = [_types.plan,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def cufftGetSizeMany(plan, rank, n,
                     inembed, istride, idist,
                     onembed, ostride, odist, fft_type, batch):
    """
    Return more accurate estimate of work area size required for batched FFT,
    taking into account any plan settings that may have been made.

    References
    ----------
    `cufftGetSizeMany <http://docs.nvidia.com/cuda/cufft/#function-cufftgetsizemany>`_
    """

    worksize = _types.worksize()
    status = _libcufft.cufftGetSizeMany(plan, rank, n,
                                        inembed, istride, idist,
                                        onembed, ostride, odist,
                                        fft_type, batch,
                                        ctypes.byref(worksize))
    cufftCheckStatus(status)
    return worksize.value

_libcufft.cufftGetSize.restype = int
_libcufft.cufftGetSize.argtypes = [_types.plan,
                                   ctypes.c_void_p]
def cufftGetSize(plan):
    """
    Return actual size of work area for FFT described in plan.

    References
    ----------
    `cufftGetSize <http://docs.nvidia.com/cuda/cufft/#function-cufftgetsize>`_
    """
    worksize = _types.worksize()
    status = _libcufft.cufftGetSize(plan, ctypes.byref(worksize))
    cufftCheckStatus(status)
    return worksize.value

_libcufft.cufftCreate.restype = int
_libcufft.cufftCreate.argtypes = [ctypes.c_void_p]
def cufftCreate():
    """
    Create FFT plan handle.

    References
    ----------
    `cufftCreate <http://docs.nvidia.com/cuda/cufft/#function-cufftcreate>`_
    """
    plan = _types.plan()
    status = _libcufft.cufftCreate(ctypes.byref(plan))
    cufftCheckStatus(status)
    return plan

_libcufft.cufftMakePlan1d.restype = int
_libcufft.cufftMakePlan1d.argtypes = [_types.plan,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cufftMakePlan1d(plan, nx, fft_type, batch):
    """
    Create 1D FFT plan configuration.

    References
    ----------
    `cufftMakePlan1d <http://docs.nvidia.com/cuda/cufft/#function-cufftmakeplan1d>`_
    """
    worksize = _types.worksize()
    status = _libcufft.cufftMakePlan1d(plan, nx, fft_type, batch,
                                       ctypes.byref(worksize))
    cufftCheckStatus(status)
    return worksize.value

_libcufft.cufftMakePlan2d.restype = int
_libcufft.cufftMakePlan2d.argtypes = [_types.plan,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cufftMakePlan2d(plan, nx, ny, fft_type):
    """
    Create 2D FFT plan configuration.

    References
    ----------
    `cufftMakePlan2d <http://docs.nvidia.com/cuda/cufft/#function-cufftmakeplan2d>`_
    """
    worksize = _types.worksize()
    status = _libcufft.cufftMakePlan2d(plan, nx, ny, fft_type,
                                       ctypes.byref(worksize))
    cufftCheckStatus(status)
    return worksize.value

_libcufft.cufftMakePlan3d.restype = int
_libcufft.cufftMakePlan3d.argtypes = [_types.plan,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cufftMakePlan3d(plan, nx, ny, nz, fft_type):
    """
    Create 3D FFT plan configuration.

    References
    ----------
    `cufftMakePlan3d <http://docs.nvidia.com/cuda/cufft/#function-cufftmakeplan3d>`_
    """
    worksize = _types.worksize()
    status = _libcufft.cufftMakePlan3d(plan, nx, ny, nz, fft_type,
                                       ctypes.byref(worksize))
    cufftCheckStatus(status)
    return worksize.value

_libcufft.cufftMakePlanMany.restype = int
_libcufft.cufftMakePlanMany.argtypes = [_types.plan,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_void_p]
def cufftMakePlanMany(plan, rank, n,
                      inembed, istride, idist,
                      onembed, ostride, odist, fft_type, batch):
    """
    Create batched FFT plan configuration.

    References
    ----------
    `cufftMakePlanMany <http://docs.nvidia.com/cuda/cufft/#function-cufftmakeplanmany>`_
    """

    worksize = _types.worksize()
    status = _libcufft.cufftMakePlanMany(plan, rank, n,
                                         inembed, istride, idist,
                                         onembed, ostride, odist,
                                         fft_type, batch,
                                         ctypes.byref(worksize))
    cufftCheckStatus(status)
    return worksize.value

_libcufft.cufftSetAutoAllocation.restype = int
_libcufft.cufftSetAutoAllocation.argtypes = [_types.plan,
                                             ctypes.c_int]
def cufftSetAutoAllocation(plan, auto_allocate):
    """
    Indicate whether the caller intends to allocate and manage work areas for
    plans that have been generated.

    References
    ----------
    `cufftSetAutoAllocation <http://docs.nvidia.com/cuda/cufft/#function-cufftsetautoallocation>`_
    """
    status = _libcufft.cufftSetAutoAllocation(plan, auto_allocate)
    cufftCheckStatus(status)

_libcufft.cufftSetWorkArea.restype = int
_libcufft.cufftSetWorkArea.argtypes = [_types.plan,
                                       ctypes.c_void_p]
def cufftSetWorkArea(plan, work_area):
    """
    Override the work area pointer associated with a plan.

    References
    ----------
    `cufftSetworkArea <http://docs.nvidia.com/cuda/cufft/#function-cufftsetworkarea>`_
    """
    status = _libcufft.cufftSetWorkArea(plan, work_area)
    cufftCheckStatus(status)
