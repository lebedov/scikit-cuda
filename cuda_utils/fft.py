#!/usr/bin/env python

"""
PyCUDA-based FFT functions.
"""

import pycuda.gpuarray as gpuarray
import numpy as np

import cufft

class Plan:
    """
    CUFFT plan class.
    
    This class represents an FFT plan for CUFFT.

    Parameters
    ----------
    shape : tuple of ints
        Shape of data to transform. Must contain no more than 3
        elements.
    dtype : { numpy.float32, numpy.float64, numpy.complex64, numpy.complex128 }
        Type of data to transform.
        
    """
    
    def __init__(self, shape, in_dtype, out_dtype):

        if np.isscalar(shape):
            self.shape = (shape, )
        else:            
            self.shape = shape

        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        
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

        # Set up plan:
        if len(self.shape) == 1:
            self.handle = cufft.cufftPlan1d(self.shape[0],
                                            self.fft_type, 1)
        elif len(self.shape) == 2:
            self.handle = cufft.cufftPlan2d(self.shape[0], self.shape[1],
                                            self.fft_type)
        elif len(self.shape) == 3:
            self.handle = cufft.cufftPlan3d(self.shape[0], self.shape[1],
                                            self.shape[2], self.fft_type)
        else:
            raise ValueError('unsupported data shape')
                                            
    def __del__(self):
        cufft.cufftDestroy(self.handle)
          
def _fft(x_gpu, p, direction, scale=False, inplace=False):
    """
    Fast Fourier Transform.

    Notes
    -----
    This function should not be called directly.
    
    """

    if inplace:
        if p.fft_type not in [cufft.CUFFT_C2C, cufft.CUFFT_Z2Z]:
            raise ValueError('can only perform inplace transformation of complex data')

        p.fft_func(p.handle, int(x_gpu.gpudata), int(x_gpu.gpudata),
                   direction)

        # Scale the result by dividing it by the number of elements:
        if scale:
            x_gpu.gpudata = (x_gpu/np.prod(x_gpu.shape)).gpudata
            
        # Don't return any value when inplace == True
        
    else:

        if direction == cufft.CUFFT_FORWARD and \
               p.in_dtype in [np.complex64, np.complex128] and \
               p.out_dtype in [np.float64, np.float128]:
            raise ValueError('cannot perform forward complex -> real transformation')

        if direction == cufft.CUFFT_INVERSE and \
               p.in_dtype in [np.float64, np.float128] and \
               p.out_dtype in [np.complex64, np.complex128]:
            raise ValueError('cannot perform inverse real -> complex transformation')

        # Create new GPUArray as output:
        y_gpu = gpuarray.empty(x_gpu.shape, p.out_dtype)

        if p.fft_type in [cufft.CUFFT_C2C, cufft.CUFFT_Z2Z]:
            p.fft_func(p.handle, int(x_gpu.gpudata), int(y_gpu.gpudata),
                       direction)
        else:
            p.fft_func(p.handle, int(x_gpu.gpudata),
                       int(y_gpu.gpudata))
                       
        # Scale the result by dividing it by the number of elements:
        if scale:
            y_gpu.gpudata = (y_gpu/np.prod(x_gpu.shape)).gpudata

        return y_gpu

def fft(x_gpu, p, scale=False, inplace=False):
    """
    Fast Fourier Transform.

    Compute the FFT of some data in device memory using the
    specified plan.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Input array.
    p : Plan
        FFT plan.
    scale : bool, optional
        If True, scale the computed FFT by the
        length of the input signal.        
    inplace : bool, optional
        If True, replace the contents of the input array with the
        computed FFT and don't return anything.

    Returns
    -------
    y_gpu : pycuda.gpuarray.GPUArray
        Computed FFT.

    """

    return _fft(x_gpu, p, cufft.CUFFT_FORWARD, scale, inplace)

def ifft(x_gpu, p, scale=False, inplace=False):
    """
    Inverse Fast Fourier Transform.

    Compute the inverse FFT of some data in device memory using the
    specified plan.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Input array.
    p : Plan
        FFT plan.
    scale : bool, optional
        If True, scale the computed inverse FFT by the
        length of the input signal.        
    inplace : bool, optional
        If True, replace the contents of the input array with the
        computed inverse FFT and don't return anything.

    Returns
    -------
    y_gpu : pycuda.gpuarray.GPUArray
        Computed inverse FFT.

    """
    
    return _fft(x_gpu, p, cufft.CUFFT_INVERSE, scale, inplace)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
