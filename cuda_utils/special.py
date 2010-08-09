#!/usr/bin/env python

"""
PyCUDA-based special functions.
"""

import os
from string import Template
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np

# Get installation location of C headers:
from __info__ import install_headers

# Adapted from Cephes library:
sici_mod_template = Template("""
#include <cuda_utils/cuSpecialFuncs.h>

#define USE_DOUBLE ${use_double}
#if USE_DOUBLE == 0
#define FLOAT float
#define SICI(x, si, ci) sicif(x, si, ci)
#else
#define FLOAT double
#define SICI(x, si, ci) sici(x, si, ci)
#endif

__global__ void sici_array(FLOAT *x, FLOAT *si,
                     FLOAT *ci, int width, int height) {
    int xIndex = blockIdx.x * ${tile_dim} + threadIdx.x;
    int yIndex = blockIdx.y * ${tile_dim} + threadIdx.y;

    int index = xIndex + width*yIndex;
    FLOAT si_temp, ci_temp;
    for (int i=0; i<${tile_dim}; i+=${block_rows}) {
         
        SICI(x[index+i*width], &si_temp, &ci_temp);
        si[index+i*width] = si_temp;
        ci[index+i*width] = ci_temp;
    }
}
""")

def sici(x_gpu, tile_dim, block_rows):
    """
    Sine/Cosine integral.

    Computes the sine and cosine integral of every element in the
    input matrix.

    Parameters
    ----------
    x_gpu : GPUArray
        Input matrix of shape `(m, n)`.
    tile_dim : int
        Each block of threads processes `tile_dim x tile_dim` elements.
    block_rows : int
        Each thread processes `tile_dim/block_rows` elements;
        `block_rows` must therefore divide `tile_dim`.

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
    >>> (si_gpu, ci_gpu) = sici(x_gpu, 2, 1)
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

    sici_mod = \
             SourceModule(sici_mod_template.substitute(tile_dim=str(tile_dim),
                                                       block_rows=str(block_rows),
                                                       use_double=use_double),
#                          cache_dir=False, # only use when debugging
                              options=["-I", install_headers])
    sici_func = sici_mod.get_function("sici_array")

    si_gpu = gpuarray.empty_like(x_gpu)
    ci_gpu = gpuarray.empty_like(x_gpu)
    sici_func(x_gpu.gpudata, si_gpu.gpudata, ci_gpu.gpudata,
              np.uint32(x_gpu.shape[0]), np.uint32(x_gpu.shape[1]),
              block=(tile_dim, block_rows, 1),
              grid=(x_gpu.shape[0]/tile_dim, x_gpu.shape[1]/tile_dim))
    return (si_gpu, ci_gpu)

# Adapted from specfun.f in scipy:
e1z_mod_template = Template("""
#include <cuComplex.h>
#include <cuda_utils/cuComplexFuncs.h>

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
                    int width, int height) {
    int xIndex = blockIdx.x * ${tile_dim} + threadIdx.x;
    int yIndex = blockIdx.y * ${tile_dim} + threadIdx.y;

    int index = xIndex + width*yIndex;
    for (int i=0; i<${tile_dim}; i+=${block_rows}) {         
        e[index+i*width] = _e1z(z[index+i*width]);
    }
}

""")

def e1z(z_gpu, tile_dim, block_rows):
    """
    Exponential integral with `n = 1` of complex arguments.

    Example
    -------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import scipy.special
    >>> import special
    >>> z = np.asarray(np.random.rand(4, 4)+1j*np.random.rand(4, 4), np.complex64)
    >>> z_gpu = gpuarray.to_gpu(z)
    >>> e_gpu = e1z(z_gpu, 2, 1)
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

    e1z_mod = \
             SourceModule(e1z_mod_template.substitute(tile_dim=str(tile_dim),
                                                      block_rows=str(block_rows),
                                                      use_double=use_double),
                          options=["-I", install_headers])
    e1z_func = e1z_mod.get_function("e1z")

    e_gpu = gpuarray.empty_like(z_gpu)
    e1z_func(z_gpu.gpudata, e_gpu.gpudata,
              np.uint32(z_gpu.shape[0]), np.uint32(z_gpu.shape[1]),
              block=(tile_dim, block_rows, 1),
              grid=(z_gpu.shape[0]/tile_dim, z_gpu.shape[1]/tile_dim))
    return e_gpu

    
if __name__ == "__main__":
    import doctest
    doctest.testmod()

