#!/usr/bin/env python

"""
PyCUDA-based special functions.
"""

import os
from string import Template
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np

from misc import get_dev_attrs, select_block_grid_sizes, init, get_current_device

# Get installation location of C headers:
from . import install_headers

# Adapted from Cephes library:
sici_template = Template("""
#include "cuSpecialFuncs.h"

#if ${use_double}
#define FLOAT double
#define SICI(x, si, ci) sici(x, si, ci)
#else
#define FLOAT float
#define SICI(x, si, ci) sicif(x, si, ci)
#endif

__global__ void sici_array(FLOAT *x, FLOAT *si,
                           FLOAT *ci, unsigned int N) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;
    FLOAT si_temp, ci_temp;

    if (idx < N) {         
        SICI(x[idx], &si_temp, &ci_temp);
        si[idx] = si_temp;
        ci[idx] = ci_temp;
    }
}
""")

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
    >>> (si_gpu, ci_gpu) = sici(x_gpu, pycuda.autoinit.device)
    >>> (si, ci) = scipy.special.sici(x)
    >>> np.allclose(si, si_gpu.get())
    True
    >>> np.allclose(ci, ci_gpu.get())
    True
    
    """

    if x_gpu.dtype == np.float32:
        use_double = 0
    elif x_gpu.dtype == np.float64:
        use_double = 1
    else:
        raise ValueError('unsupported type')
    
    # Get block/grid sizes:
    dev = get_current_device()
    block_dim, grid_dim = select_block_grid_sizes(dev, x_gpu.shape)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    sici_mod = \
             SourceModule(sici_template.substitute(use_double=use_double),
                          cache_dir=cache_dir,
                          options=["-I", install_headers])
    sici_func = sici_mod.get_function("sici_array")

    si_gpu = gpuarray.empty_like(x_gpu)
    ci_gpu = gpuarray.empty_like(x_gpu)
    sici_func(x_gpu, si_gpu, ci_gpu,
              np.uint32(x_gpu.size),
              block=block_dim,
              grid=grid_dim)
    return (si_gpu, ci_gpu)

ei_template = Template("""
#include <pycuda/pycuda-complex.hpp>
#include "cuSpecialFuncs.h"

#if ${use_double}
#define FLOAT double
#define COMPLEX pycuda::complex<double>
#define E1Z(z) e1z(z)
#define EIXZ(z) eixz(z)
#else
#define FLOAT float
#define COMPLEX pycuda::complex<float>
#define E1Z(z) e1zf(z)
#define EIXZ(z) eixzf(z)
#endif

__global__ void e1z_array(COMPLEX *z, COMPLEX *e,
                          unsigned int N) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < N) 
        e[idx] = E1Z(z[idx]);
}

__global__ void eixz_array(COMPLEX *z, COMPLEX *e,
                           unsigned int N) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < N) 
        e[idx] = EIXZ(z[idx]);
}
""")

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
    >>> e_gpu = exp1(z_gpu, pycuda.autoinit.device)
    >>> e_sp = scipy.special.exp1(z)
    >>> np.allclose(e_sp, e_gpu.get())
    True

    """

    if z_gpu.dtype == np.complex64:
        use_double = 0
    elif z_gpu.dtype == np.complex128:
        use_double = 1
    else:
        raise ValueError('unsupported type')

    
    # Get block/grid sizes; the number of threads per block is limited
    # to 256 because the kernel defined above uses too many
    # registers to be invoked more threads per block:
    dev = get_current_device()
    max_threads_per_block = 256
    block_dim, grid_dim = \
               select_block_grid_sizes(dev, z_gpu.shape, max_threads_per_block)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    ei_mod = \
             SourceModule(ei_template.substitute(use_double=use_double),
                          cache_dir=cache_dir,
                          options=["-I", install_headers])
    e1z_func = ei_mod.get_function("e1z_array")

    e_gpu = gpuarray.empty_like(z_gpu)
    e1z_func(z_gpu, e_gpu,
              np.uint32(z_gpu.size),
              block=block_dim,
              grid=grid_dim)
    return e_gpu

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
    >>> e_gpu = expi(z_gpu, pycuda.autoinit.device)
    >>> e_sp = scipy.special.expi(z)
    >>> np.allclose(e_sp, e_gpu.get())
    True

    """

    if z_gpu.dtype == np.complex64:
        use_double = 0
    elif z_gpu.dtype == np.complex128:
        use_double = 1
    else:
        raise ValueError('unsupported type')
   
    # Get block/grid sizes; the number of threads per block is limited
    # to 128 because the kernel defined above uses too many
    # registers to be invoked more threads per block:
    dev = get_current_device()
    max_threads_per_block = 128
    block_dim, grid_dim = \
               select_block_grid_sizes(dev, z_gpu.shape, max_threads_per_block)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    ei_mod = \
             SourceModule(ei_template.substitute(use_double=use_double),
                          cache_dir=cache_dir,
                          options=["-I", install_headers])
    eixz_func = ei_mod.get_function("eixz_array")

    e_gpu = gpuarray.empty_like(z_gpu)
    eixz_func(z_gpu, e_gpu,
              np.uint32(z_gpu.size),
              block=block_dim,
              grid=grid_dim)
    return e_gpu
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()

