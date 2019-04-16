#!/usr/bin/env python

"""
PyCUDA-based special functions.
"""

import os
import pycuda.gpuarray as gpuarray
import pycuda.elementwise as elementwise
from pycuda.tools import context_dependent_memoize
import numpy as np

from . import misc
from .misc import init

# Get installation location of C headers:
from . import install_headers

@context_dependent_memoize
def _get_sici_kernel(dtype):
    if dtype == np.float32:
        args = 'float *x, float *si, float *ci'
        op = 'sicif(x[i], &si[i], &ci[i])'
    elif dtype == np.float64:
        args = 'double *x, double *si, double *ci'
        op = 'sici(x[i], &si[i], &ci[i])'
    else:
        raise ValueError('unsupported type')

    return elementwise.ElementwiseKernel(args, op,
                                 options=["-I", install_headers],
                                 preamble='#include "cuSpecialFuncs.h"')

def sici(x_gpu):
    """
    Sine/Cosine integral.

    Computes the sine and cosine integral of every element in the
    input matrix.

    Parameters
    ----------
    x_gpu : GPUArray
        Input matrix of shape `(m, n)`.

    Returns
    -------
    (si_gpu, ci_gpu) : tuple of GPUArrays
        Tuple of GPUarrays containing the sine integrals and cosine
        integrals of the entries of `x_gpu`.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import scipy.special
    >>> import special
    >>> x = np.array([[1, 2], [3, 4]], np.float32)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> (si_gpu, ci_gpu) = sici(x_gpu)
    >>> (si, ci) = scipy.special.sici(x)
    >>> np.allclose(si, si_gpu.get())
    True
    >>> np.allclose(ci, ci_gpu.get())
    True
    """

    si_gpu = gpuarray.empty_like(x_gpu)
    ci_gpu = gpuarray.empty_like(x_gpu)
    func = _get_sici_kernel(x_gpu.dtype)
    func(x_gpu, si_gpu, ci_gpu)

    return (si_gpu, ci_gpu)

@context_dependent_memoize
def _get_exp1_kernel(dtype):
    if dtype == np.complex64:
        args = 'pycuda::complex<float> *z, pycuda::complex<float> *e'
    elif dtype == np.complex128:
        args = 'pycuda::complex<double> *z, pycuda::complex<double> *e'
    else:
        raise ValueError('unsupported type')
    op = 'e[i] = exp1(z[i])'

    return elementwise.ElementwiseKernel(args, op,
                                 options=["-I", install_headers],
                                 preamble='#include "cuSpecialFuncs.h"')

def exp1(z_gpu):
    """
    Exponential integral with `n = 1` of complex arguments.

    Parameters
    ----------
    z_gpu : GPUArray
        Input matrix of shape `(m, n)`.

    Returns
    -------
    e_gpu : GPUArray
        GPUarrays containing the exponential integrals of
        the entries of `z_gpu`.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import scipy.special
    >>> import special
    >>> z = np.asarray(np.random.rand(4, 4)+1j*np.random.rand(4, 4), np.complex64)
    >>> z_gpu = gpuarray.to_gpu(z)
    >>> e_gpu = exp1(z_gpu)
    >>> e_sp = scipy.special.exp1(z)
    >>> np.allclose(e_sp, e_gpu.get())
    True
    """

    e_gpu = gpuarray.empty_like(z_gpu)
    func = _get_exp1_kernel(z_gpu.dtype)
    func(z_gpu, e_gpu)

    return e_gpu
exp1.cache = {}

@context_dependent_memoize
def _get_expi_kernel(dtype):
    if dtype == np.complex64:
        args = 'pycuda::complex<float> *z, pycuda::complex<float> *e'
    elif dtype == np.complex128:
        args = 'pycuda::complex<double> *z, pycuda::complex<double> *e'
    else:
        raise ValueError('unsupported type')
    op = 'e[i] = expi(z[i])'

    return elementwise.ElementwiseKernel(args, op,
                                 options=["-I", install_headers],
                                 preamble='#include "cuSpecialFuncs.h"')

def expi(z_gpu):
    """
    Exponential integral of complex arguments.

    Parameters
    ----------
    z_gpu : GPUArray
        Input matrix of shape `(m, n)`.

    Returns
    -------
    e_gpu : GPUArray
        GPUarrays containing the exponential integrals of
        the entries of `z_gpu`.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import scipy.special
    >>> import special
    >>> z = np.asarray(np.random.rand(4, 4)+1j*np.random.rand(4, 4), np.complex64)
    >>> z_gpu = gpuarray.to_gpu(z)
    >>> e_gpu = expi(z_gpu)
    >>> e_sp = scipy.special.expi(z)
    >>> np.allclose(e_sp, e_gpu.get())
    True
    """

    e_gpu = gpuarray.empty_like(z_gpu)
    func = _get_expi_kernel(z_gpu.dtype)
    func(z_gpu, e_gpu)

    return e_gpu

if __name__ == "__main__":
    import doctest
    doctest.testmod()

