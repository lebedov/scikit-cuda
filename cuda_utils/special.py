#!/usr/bin/env python

"""
PyCUDA-based special functions.
"""

import os
from string import Template
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np

from cuda_utils.misc import get_dev_attrs, select_block_grid_sizes

# Get installation location of C headers:
from cuda_utils import install_headers

# Adapted from Cephes library:
sici_mod_template = Template("""
#include "cuSpecialFuncs.h"

#define USE_DOUBLE ${use_double}
#if USE_DOUBLE == 0
#define FLOAT float
#define SICI(x, si, ci) sicif(x, si, ci)
#else
#define FLOAT double
#define SICI(x, si, ci) sici(x, si, ci)
#endif

__global__ void sici_array(FLOAT *x, FLOAT *si,
                           FLOAT *ci, unsigned int N) {
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;
    FLOAT si_temp, ci_temp;

    if (idx < N) {         
        SICI(x[idx], &si_temp, &ci_temp);
        si[idx] = si_temp;
        ci[idx] = ci_temp;
    }
}
""")

def sici(x_gpu, dev):
    """
    Sine/Cosine integral.

    Computes the sine and cosine integral of every element in the
    input matrix.

    Parameters
    ----------
    x_gpu : GPUArray
        Input matrix of shape `(m, n)`.
    dev : pycuda.driver.Device
        Device object to be used.
        
    Returns
    -------
    (si_gpu, ci_gpu) : tuple of GPUArrays
        Tuple of GPUarrays containing the sine integrals and cosine
        integrals of the entries of `x_gpu`.
        
    Example
    -------
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
    max_threads_per_block, max_block_dim, max_grid_dim = get_dev_attrs(dev)
    block_dim, grid_dim = select_block_grid_sizes(dev, x_gpu.shape)
    max_blocks_per_grid = max(max_grid_dim)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    sici_mod = \
             SourceModule(sici_mod_template.substitute(use_double=use_double,
                          max_threads_per_block=str(max_threads_per_block),
                          max_blocks_per_grid=str(max_blocks_per_grid)),
                          cache_dir=cache_dir,
                          options=["-I", install_headers])
    sici_func = sici_mod.get_function("sici_array")

    si_gpu = gpuarray.empty_like(x_gpu)
    ci_gpu = gpuarray.empty_like(x_gpu)
    sici_func(x_gpu.gpudata, si_gpu.gpudata, ci_gpu.gpudata,
              np.uint32(x_gpu.size),
              block=block_dim,
              grid=grid_dim)
    return (si_gpu, ci_gpu)

# Adapted from specfun.f in scipy:
e1z_mod_template = Template("""
#include <cuComplex.h>
#include "cuComplexFuncs.h"

#define PI 3.1415926535897931
#define EL 0.5772156649015328

#define USE_DOUBLE ${use_double}
#if USE_DOUBLE == 0
#define FLOAT float
#define COMPLEX cuFloatComplex
#define CREAL(z) cuCrealf(z)
#define CIMAG(z) cuCimagf(z)
#define CABS(z) cuCabsf(z)
#define MAKE_COMPLEX(x, y) make_cuFloatComplex(x, y)
#define POW(x, y) powf(x, y)
#define CMUL(x, y) cuCmulf(x, y)
#define CDIV(x, y) cuCdivf(x, y)
#define CADD(x, y) cuCaddf(x, y)
#define CSUB(x, y) cuCsubf(x, y)
#define CLOG(z) cuClogf(z)
#define CEXP(z) cuCexpf(z)
#define CNEG(z) make_cuFloatComplex(-z.x, -z.y)
#else
#define FLOAT double
#define COMPLEX cuDoubleComplex
#define CREAL(z) cuCreal(z)
#define CIMAG(z) cuCimag(z)
#define CABS(z) cuCabs(z)
#define MAKE_COMPLEX(x, y) make_cuDoubleComplex(x, y)
#define POW(x, y) pow(x, y)
#define CMUL(x, y) cuCmul(x, y)
#define CDIV(x, y) cuCdiv(x, y)
#define CADD(x, y) cuCadd(x, y)
#define CSUB(x, y) cuCsub(x, y)
#define CLOG(z) cuClog(z)
#define CEXP(z) cuCexp(z)
#define CNEG(z) make_cuDoubleComplex(-z.x, -z.y)
#endif

__device__ COMPLEX _e1z(COMPLEX z) {
    FLOAT x = CREAL(z);
    FLOAT a0 = CABS(z);
    COMPLEX ce1, cr, ct0, kc, ct;
    
    if (a0 == 0.0)
        ce1 = MAKE_COMPLEX(1.0e300, 0.0);
    else if ((a0 < 10.0) || (x < 0.0 && a0 < 20.0)) {
        ce1 = MAKE_COMPLEX(1.0, 0.0);
        cr = MAKE_COMPLEX(1.0, 0.0);
        for (unsigned int k = 1; k <= 150; k++) {
            cr = CDIV(CNEG(CMUL(CMUL(cr, MAKE_COMPLEX(k, 0.0)), z)),
                     MAKE_COMPLEX(POW(k+1.0, 2.0), 0.0));
            ce1 = CADD(ce1, cr);
            if (CABS(cr) <= CABS(ce1)*1.0e-15)
                break;
        }
        ce1 = CADD(CSUB(MAKE_COMPLEX(-EL, 0.0), CLOG(z)),
                   CMUL(z, ce1));                   
    } else {
        ct0 = MAKE_COMPLEX(0.0, 0.0);
        for (unsigned int k = 120; k >= 1; k--) {
            kc = MAKE_COMPLEX(k, 0.0);
            ct0 = CDIV(kc, (CADD(MAKE_COMPLEX(1.0, 0.0), CDIV(kc, CADD(z, ct0)))));
        }
        ct = CDIV(MAKE_COMPLEX(1.0, 0.0), (CADD(z, ct0)));
        ce1 = CMUL(CEXP(CNEG(z)), ct);
        if (x <= 0.0 && CIMAG(z) == 0.0)
            ce1 = CSUB(ce1, MAKE_COMPLEX(0.0, -PI));
    }
    return ce1;
}

__global__ void e1z(COMPLEX *z, COMPLEX *e,
                    unsigned int N) {
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;

    if (idx < N) 
        e[idx] = _e1z(z[idx]);
}

""")

def e1z(z_gpu, dev):
    """
    Exponential integral with `n = 1` of complex arguments.

    Parameters
    ----------
    x_gpu : GPUArray
        Input matrix of shape `(m, n)`.
    dev : pycuda.driver.Device
        Device object to be used.
        
    Returns
    -------
    e_gpu : GPUArray
        GPUarrays containing the exponential integrals of
        the entries of `z_gpu`.

    Example
    -------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import scipy.special
    >>> import special
    >>> z = np.asarray(np.random.rand(4, 4)+1j*np.random.rand(4, 4), np.complex64)
    >>> z_gpu = gpuarray.to_gpu(z)
    >>> e_gpu = e1z(z_gpu, pycuda.autoinit.device)
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

    # Get block/grid sizes:
    max_threads_per_block, max_block_dim, max_grid_dim = get_dev_attrs(dev)
    block_dim, grid_dim = select_block_grid_sizes(dev, z_gpu.shape)
    max_blocks_per_grid = max(max_grid_dim)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    e1z_mod = \
             SourceModule(e1z_mod_template.substitute(use_double=use_double,
                          max_threads_per_block=str(max_threads_per_block),
                          max_blocks_per_grid=str(max_blocks_per_grid)),
                          cache_dir=cache_dir,
                          options=["-I", install_headers])
    e1z_func = e1z_mod.get_function("e1z")

    e_gpu = gpuarray.empty_like(z_gpu)
    e1z_func(z_gpu.gpudata, e_gpu.gpudata,
              np.uint32(z_gpu.size),
              block=block_dim,
              grid=grid_dim)
    return e_gpu

    
if __name__ == "__main__":
    import doctest
    doctest.testmod()

