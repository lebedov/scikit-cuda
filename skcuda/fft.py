#!/usr/bin/env python

"""
PyCUDA-based FFT functions.
"""

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.elementwise as el
import pycuda.tools as tools
import numpy as np

from . import cufft
from .cufft import CUFFT_COMPATIBILITY_NATIVE, \
     CUFFT_COMPATIBILITY_FFTW_PADDING, \
     CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC, \
     CUFFT_COMPATIBILITY_FFTW_ALL

from . import misc

class Plan:
    """
    CUFFT plan class.

    This class represents an FFT plan for CUFFT.

    Parameters
    ----------
    shape : tuple of ints
        Transform shape. May contain more than 3 elements.
    in_dtype : { numpy.float32, numpy.float64, numpy.complex64, numpy.complex128 }
        Type of input data.
    out_dtype : { numpy.float32, numpy.float64, numpy.complex64, numpy.complex128 }
        Type of output data.
    batch : int
        Number of FFTs to configure in parallel (default is 1).
    stream : pycuda.driver.Stream
        Stream with which to associate the plan. If no stream is specified,
        the default stream is used.
    mode : int
        FFTW compatibility mode.
    inembed : numpy.array with dtype=numpy.int32
        number of elements in each dimension of the input array
    istride : int
        distance between two successive input elements in the least significant
        (innermost) dimension
    idist : int
        distance between the first element of two consective batches in the
        input data
    onembed : numpy.array with dtype=numpy.int32
        number of elements in each dimension of the output array
    ostride : int
        distance between two successive output elements in the least significant
        (innermost) dimension
    odist : int
        distance between the first element of two consective batches in the
        output data
    auto_allocate : bool
        indicates whether the caller intends to allocate and manage the work area
    """

    def __init__(self, shape, in_dtype, out_dtype, batch=1, stream=None,
                 mode=0x01, inembed=None, istride=1, idist=0, onembed=None,
                 ostride=1, odist=0, auto_allocate=True):

        if np.isscalar(shape):
            self.shape = (shape, )
        else:
            self.shape = shape

        self.in_dtype = in_dtype
        self.out_dtype = out_dtype

        if batch <= 0:
            raise ValueError('batch size must be greater than 0')
        self.batch = batch

        # Determine type of transformation:
        if in_dtype == np.float32 and out_dtype == np.complex64:
            self.fft_type = cufft.CUFFT_R2C
            self.fft_func = cufft.cufftExecR2C
        elif in_dtype == np.complex64 and out_dtype == np.float32:
            self.fft_type = cufft.CUFFT_C2R
            self.fft_func = cufft.cufftExecC2R
        elif in_dtype == np.complex64 and out_dtype == np.complex64:
            self.fft_type = cufft.CUFFT_C2C
            self.fft_func = cufft.cufftExecC2C
        elif in_dtype == np.float64 and out_dtype == np.complex128:
            self.fft_type = cufft.CUFFT_D2Z
            self.fft_func = cufft.cufftExecD2Z
        elif in_dtype == np.complex128 and out_dtype == np.float64:
            self.fft_type = cufft.CUFFT_Z2D
            self.fft_func = cufft.cufftExecZ2D
        elif in_dtype == np.complex128 and out_dtype == np.complex128:
            self.fft_type = cufft.CUFFT_Z2Z
            self.fft_func = cufft.cufftExecZ2Z
        else:
            raise ValueError('unsupported input/output type combination')

        # Check for double precision support:
        capability = misc.get_compute_capability(misc.get_current_device())
        if capability < 1.3 and \
           (misc.isdoubletype(in_dtype) or misc.isdoubletype(out_dtype)):
            raise RuntimeError('double precision requires compute capability '
                               '>= 1.3 (you have %g)' % capability)

        if inembed is not None:
            inembed = inembed.ctypes.data
        if onembed is not None:
            onembed = onembed.ctypes.data

        # Set up plan:
        if len(self.shape) <= 0:
            raise ValueError('invalid transform size')
        n = np.asarray(self.shape, np.int32)
        self.handle = cufft.cufftCreate()
        # Set FFTW compatibility mode:
        cufft.cufftSetCompatibilityMode(self.handle, mode)
        # Set auto-allocate mode
        cufft.cufftSetAutoAllocation(self.handle, auto_allocate)
        self.worksize = cufft.cufftMakePlanMany(
            self.handle, len(self.shape), n.ctypes.data, inembed, istride, idist,
            onembed, ostride, odist, self.fft_type, self.batch)

        # Associate stream with plan:
        if stream != None:
            cufft.cufftSetStream(self.handle, stream.handle)

    def set_work_area(self, work_area):
        """
        Associate a caller-managed work area with the plan.

        Parameters
        ----------
        work_area : pycuda.gpuarray.GPUArray
        """
        cufft.cufftSetWorkArea(self.handle, int(work_area.gpudata))

    def __del__(self):

        # Don't complain if handle destruction fails because the plan
        # may have already been cleaned up:
        try:
            cufft.cufftDestroy(self.handle)
        except:
            pass

def _scale_inplace(a, x_gpu):
    """
    Scale an array by a specified value in-place.
    """

    # Cache the kernel to avoid invoking the compiler if the
    # specified scale factor and array type have already been encountered:
    try:
        func = _scale_inplace.cache[(a, x_gpu.dtype)]
    except KeyError:
        ctype = tools.dtype_to_ctype(x_gpu.dtype)
        func = el.ElementwiseKernel(
            "{ctype} a, {ctype} *x".format(ctype=ctype),
            "x[i] /= a")
        _scale_inplace.cache[(a, x_gpu.dtype)] = func
    func(x_gpu.dtype.type(a), x_gpu)
_scale_inplace.cache = {}

def _fft(x_gpu, y_gpu, plan, direction, scale=None):
    """
    Fast Fourier Transform.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Input array.
    y_gpu : pycuda.gpuarray.GPUArray
        Output array.
    plan : Plan
        FFT plan.
    direction : { cufft.CUFFT_FORWARD, cufft.CUFFT_INVERSE }
        Transform direction. Only affects in-place transforms.

    Optional Parameters
    -------------------
    scale : int or float
        Scale the values in the output array by dividing them by this value.

    Notes
    -----
    This function should not be called directly.

    """

    if (x_gpu.gpudata == y_gpu.gpudata) and \
           plan.fft_type not in [cufft.CUFFT_C2C, cufft.CUFFT_Z2Z]:
        raise ValueError('can only compute in-place transform of complex data')

    if direction == cufft.CUFFT_FORWARD and \
           plan.in_dtype in np.sctypes['complex'] and \
           plan.out_dtype in np.sctypes['float']:
        raise ValueError('cannot compute forward complex -> real transform')

    if direction == cufft.CUFFT_INVERSE and \
           plan.in_dtype in np.sctypes['float'] and \
           plan.out_dtype in np.sctypes['complex']:
        raise ValueError('cannot compute inverse real -> complex transform')

    if plan.fft_type in [cufft.CUFFT_C2C, cufft.CUFFT_Z2Z]:
        plan.fft_func(plan.handle, int(x_gpu.gpudata), int(y_gpu.gpudata),
                      direction)
    else:
        plan.fft_func(plan.handle, int(x_gpu.gpudata),
                      int(y_gpu.gpudata))

    # Scale the result by dividing it by the number of elements:
    if scale != None:
        _scale_inplace(scale, y_gpu)

def fft(x_gpu, y_gpu, plan, scale=False):
    """
    Fast Fourier Transform.

    Compute the FFT of some data in device memory using the
    specified plan.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Input array.
    y_gpu : pycuda.gpuarray.GPUArray
        FFT of input array.
    plan : Plan
        FFT plan.
    scale : bool, optional
        If True, scale the computed FFT by the number of elements in
        the input array.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> N = 128
    >>> x = np.asarray(np.random.rand(N), np.float32)
    >>> xf = np.fft.fft(x)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> xf_gpu = gpuarray.empty(N/2+1, np.complex64)
    >>> plan = Plan(x.shape, np.float32, np.complex64)
    >>> fft(x_gpu, xf_gpu, plan)
    >>> np.allclose(xf[0:N/2+1], xf_gpu.get(), atol=1e-6)
    True

    Returns
    -------
    y_gpu : pycuda.gpuarray.GPUArray
        Computed FFT.

    Notes
    -----
    For real to complex transformations, this function computes
    N/2+1 non-redundant coefficients of a length-N input signal.

    """

    if scale == True:
        return _fft(x_gpu, y_gpu, plan, cufft.CUFFT_FORWARD, x_gpu.size/plan.batch)
    else:
        return _fft(x_gpu, y_gpu, plan, cufft.CUFFT_FORWARD)

def ifft(x_gpu, y_gpu, plan, scale=False):
    """
    Inverse Fast Fourier Transform.

    Compute the inverse FFT of some data in device memory using the
    specified plan.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Input array.
    y_gpu : pycuda.gpuarray.GPUArray
        Inverse FFT of input array.
    plan : Plan
        FFT plan.
    scale : bool, optional
        If True, scale the computed inverse FFT by the number of
        elements in the output array.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> N = 128
    >>> x = np.asarray(np.random.rand(N), np.float32)
    >>> xf = np.asarray(np.fft.fft(x), np.complex64)
    >>> xf_gpu = gpuarray.to_gpu(xf[0:N/2+1])
    >>> x_gpu = gpuarray.empty(N, np.float32)
    >>> plan = Plan(N, np.complex64, np.float32)
    >>> ifft(xf_gpu, x_gpu, plan, True)
    >>> np.allclose(x, x_gpu.get(), atol=1e-6)
    True

    Notes
    -----
    For complex to real transformations, this function assumes the
    input contains N/2+1 non-redundant FFT coefficents of a signal of
    length N.

    """

    if scale == True:
        return _fft(x_gpu, y_gpu, plan, cufft.CUFFT_INVERSE, y_gpu.size/plan.batch)
    else:
        return _fft(x_gpu, y_gpu, plan, cufft.CUFFT_INVERSE)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
