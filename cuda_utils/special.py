#!/usr/bin/env python

"""
PyCUDA-based special functions.
"""

import os
from string import Template
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

import numpy as np

# Adapted from Cephes library:
sici_mod_template = Template("""
#define USE_DOUBLE ${use_double}
#if USE_DOUBLE == 0
#define MAXNUM 3.4028234663852885981170418348451692544E38
#define FLOAT float
#define SIN(x) sinf(x)
#define COS(x) cosf(x)
#define LOG(x) logf(x)
#else
#define MAXNUM 1.7976931348623157081452742373170435679E308
#define FLOAT double
#define SIN(x) sin(x)
#define COS(x) cos(x)
#define LOG(x) log(x)
#endif

#define PIO2 1.57079632679489661923

__device__ FLOAT polevl(FLOAT x, FLOAT *coef, int N) {
    FLOAT ans;
    FLOAT *p;
    int i;

    p = coef;
    ans = *p++;
    i = N;

    do
        ans = ans*x + *p++;
    while (--i);

    return (ans);
}

__device__ FLOAT p1evl(FLOAT x, FLOAT *coef, int N) {
    FLOAT ans;
    FLOAT *p;
    int i;

    p = coef;
    ans = x + *p++;
    i = N-1;

    do
        ans = ans*x + *p++;
    while (--i);

    return (ans);
}

#define EUL 0.57721566490153286061

__constant__ FLOAT SN[] = {
    -8.39167827910303881427E-11,
    4.62591714427012837309E-8,
    -9.75759303843632795789E-6,
    9.76945438170435310816E-4,
    -4.13470316229406538752E-2,
    1.00000000000000000302E0,
};
__constant__ FLOAT SD[] = {
    2.03269266195951942049E-12,
    1.27997891179943299903E-9,
    4.41827842801218905784E-7,
    9.96412122043875552487E-5,
    1.42085239326149893930E-2,
    9.99999999999999996984E-1,
};
__constant__ FLOAT CN[] = {
    2.02524002389102268789E-11,
    -1.35249504915790756375E-8,
    3.59325051419993077021E-6,
    -4.74007206873407909465E-4,
    2.89159652607555242092E-2,
    -1.00000000000000000080E0,
};
__constant__ FLOAT CD[] = {
    4.07746040061880559506E-12,
    3.06780997581887812692E-9,
    1.23210355685883423679E-6,
    3.17442024775032769882E-4,
    5.10028056236446052392E-2,
    4.00000000000000000080E0,
};
__constant__ FLOAT FN4[] = {
    4.23612862892216586994E0,
    5.45937717161812843388E0,
    1.62083287701538329132E0,
    1.67006611831323023771E-1,
    6.81020132472518137426E-3,
    1.08936580650328664411E-4,
    5.48900223421373614008E-7,
};
__constant__ FLOAT FD4[] = {
    /*  1.00000000000000000000E0,*/
    8.16496634205391016773E0,
    7.30828822505564552187E0,
    1.86792257950184183883E0,
    1.78792052963149907262E-1,
    7.01710668322789753610E-3,
    1.10034357153915731354E-4,
    5.48900252756255700982E-7,
};
__constant__ FLOAT FN8[] = {
    4.55880873470465315206E-1,
    7.13715274100146711374E-1,
    1.60300158222319456320E-1,
    1.16064229408124407915E-2,
    3.49556442447859055605E-4,
    4.86215430826454749482E-6,
    3.20092790091004902806E-8,
    9.41779576128512936592E-11,
    9.70507110881952024631E-14,
};
__constant__ FLOAT FD8[] = {
    /*  1.00000000000000000000E0,*/
    9.17463611873684053703E-1,
    1.78685545332074536321E-1,
    1.22253594771971293032E-2,
    3.58696481881851580297E-4,
    4.92435064317881464393E-6,
    3.21956939101046018377E-8,
    9.43720590350276732376E-11,
    9.70507110881952025725E-14,
};
                                  
__constant__ FLOAT GN4[] = {
    8.71001698973114191777E-2,
    6.11379109952219284151E-1,
    3.97180296392337498885E-1,
    7.48527737628469092119E-2,
    5.38868681462177273157E-3,
    1.61999794598934024525E-4,
    1.97963874140963632189E-6,
    7.82579040744090311069E-9,
};
__constant__ FLOAT GD4[] = {
    /*  1.00000000000000000000E0,*/
    1.64402202413355338886E0,
    6.66296701268987968381E-1,
    9.88771761277688796203E-2,
    6.22396345441768420760E-3,
    1.73221081474177119497E-4,
    2.02659182086343991969E-6,
    7.82579218933534490868E-9,
};
__constant__ FLOAT GN8[] = {
    6.97359953443276214934E-1,
    3.30410979305632063225E-1,
    3.84878767649974295920E-2,
    1.71718239052347903558E-3,
    3.48941165502279436777E-5,
    3.47131167084116673800E-7,
    1.70404452782044526189E-9,
    3.85945925430276600453E-12,
    3.14040098946363334640E-15,
};
__constant__ FLOAT GD8[] = {
    /*  1.00000000000000000000E0,*/
    1.68548898811011640017E0,
    4.87852258695304967486E-1,
    4.67913194259625806320E-2,
    1.90284426674399523638E-3,
    3.68475504442561108162E-5,
    3.57043223443740838771E-7,
    1.72693748966316146736E-9,
    3.87830166023954706752E-12,
    3.14040098946363335242E-15,
};

__device__ void _sici(FLOAT x, FLOAT *si, FLOAT *ci) {
    FLOAT z, c, s, f, g;
    short sign;
    
    if (x < 0.0) {
        sign = -1;
        x = -x;
    } else 
        sign = 0;

    if (x == 0.0) {
        *si = 0;
        *ci = -MAXNUM;
        return;
    }

    if (x > 1.0e9) {
        *si = PIO2 - COS(x)/x;
        *ci = SIN(x)/x;
    }

    if (x > 4.0) 
        goto asympt;

    z = x*x;
    s = x*polevl(z, SN, 5)/polevl(z, SD, 5);
    c = z*polevl(z, CN, 5)/polevl(z, CD, 5);

    if (sign)
        s = -s;

    *si = s;
    *ci = EUL + LOG(x) + c;
    return;

asympt:
    s = SIN(x);
    c = COS(x);
    z = 1.0/(x*x);

    if (x < 8.0) {
        f = polevl(z, FN4, 6)/(x*p1evl(z, FD4, 7));
        g = z*polevl(z, GN4, 7)/p1evl(z, GD4, 7);
    } else {
        f = polevl(z, FN8, 8)/(x*p1evl(z, FD8, 8));
        g = z*polevl(z, GN8, 8)/p1evl(z, GD8, 9);
    }
    *si = PIO2 - f*c - g*s;
    if (sign)
        *si = -(*si);
    *ci = f*s - g*c;
    return;
}

__global__ void sici(FLOAT *x, FLOAT *si,
                     FLOAT *ci, int width, int height) {
    int xIndex = blockIdx.x * ${tile_dim} + threadIdx.x;
    int yIndex = blockIdx.y * ${tile_dim} + threadIdx.y;

    int index = xIndex + width*yIndex;
    FLOAT si_temp, ci_temp;
    for (int i=0; i<${tile_dim}; i+=${block_rows}) {
         
        _sici(x[index+i*width], &si_temp, &ci_temp);
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
                                                       use_double=use_double))
    sici_func = sici_mod.get_function("sici")

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
                          options=["-I", os.getenv('CPATH')])
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

