#!/usr/bin/env python

"""
PyCUDA-based integration functions.
"""

from string import Template
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np

import ctypes
import cublas

from misc import get_dev_attrs, select_block_grid_sizes, init

gen_trapz_mult_template = Template("""
#include <pycuda/pycuda-complex.hpp>

#define USE_DOUBLE ${use_double}
#define USE_COMPLEX ${use_complex}
#if USE_DOUBLE == 1
#if USE_COMPLEX == 1
#define TYPE pycuda::complex<double>
#else
#define TYPE double
#endif
#else
#if USE_COMPLEX == 1
#define TYPE pycuda::complex<float>
#else
#define TYPE float
#endif
#endif

__global__ void gen_trapz_mult(TYPE *mult, unsigned int N) {
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;

    if (idx < N) {
        if ((idx == 0) || (idx == N-1)) {                      
            mult[idx] = TYPE(0.5);
        } else {
            mult[idx] = TYPE(1.0);
        }
    }
}
""")

def gen_trapz_mult(N, mult_type, dev):
    """
    Generate multiplication array for 1D trapezoidal integration.

    Generates an array whose dot product with some array of equal
    length is equivalent to the definite integral of the latter
    computed using trapezoidal integration.

    Parameters
    ----------
    N : int
        Length of array.
    mult_type : float type
        Floating point type to use when generating the array.
    dev : pycuda.driver.Device
        Device object to be used.

    Returns
    -------
    result : pycuda.gpuarray.GPUArray
        Generated array.

    """
    
    if mult_type not in [np.float32, np.float64, np.complex64,
                         np.complex128]:
        raise ValueError('unrecognized type')

    use_double = int(mult_type in [np.float64, np.complex128])
    use_complex = int(mult_type in [np.complex64, np.complex128])

    # Allocate output matrix:
    mult_gpu = gpuarray.empty(N, mult_type)

    # Get block/grid sizes:
    max_threads_per_block, max_block_dim, max_grid_dim = get_dev_attrs(dev)
    block_dim, grid_dim = select_block_grid_sizes(dev, N)
    max_blocks_per_grid = max(max_grid_dim)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    gen_trapz_mult_mod = \
                       SourceModule(gen_trapz_mult_template.substitute(use_double=use_double,
                                                                       use_complex=use_complex,
                                    max_threads_per_block=max_threads_per_block,
                                    max_blocks_per_grid=max_blocks_per_grid),
                                    cache_dir=cache_dir)

    gen_trapz_mult = gen_trapz_mult_mod.get_function("gen_trapz_mult")    
    gen_trapz_mult(mult_gpu, np.uint32(N),
                   block=block_dim,
                   grid=grid_dim)
    
    return mult_gpu

def trapz(x_gpu, dev, dx=1.0):
    """
    1D trapezoidal integration.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Input array to integrate.
    dx : scalar
        Spacing.

    Returns
    -------
    result : float
        Definite integral as approximated by the trapezoidal rule.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray
    >>> import numpy as np
    >>> import integrate
    >>> integrate.init()
    >>> x = np.asarray(np.random.rand(10), np.float32)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> z = integrate.trapz(x_gpu, pycuda.autoinit.device)
    >>> np.allclose(np.trapz(x), z)
    True
    
    """
    
    if len(x_gpu.shape) > 1:
        raise ValueError('input array must be 1D')
    if np.iscomplex(dx):
        raise ValueError('dx must be real')

    float_type = x_gpu.dtype.type
    if float_type == np.complex64:
        cublas_func = cublas._libcublas.cublasCdotu        
    elif float_type == np.float32:
        cublas_func = cublas._libcublas.cublasSdot
    elif float_type == np.complex128:
        cublas_func = cublas._libcublas.cublasZdotu
    elif float_type == np.float64:
        cublas_func = cublas._libcublas.cublasDdot
    else:
        raise ValueError('unsupported input type')

    trapz_mult_gpu = gen_trapz_mult(x_gpu.size, float_type, dev)
    result = cublas_func(x_gpu.size, int(x_gpu.gpudata), 1,
                         int(trapz_mult_gpu.gpudata), 1)

    dx = float_type(dx)
    if float_type == np.complex64:
        return (np.float32(result.x)+1j*np.float32(result.y))*dx
    elif float_type == np.complex128:
        return (np.float64(result.x)+1j*np.float64(result.y))*dx
    elif float_type == np.float32:
        return np.float32(result)*dx
    else:
        return np.float64(result)*dx

gen_trapz2d_mult_template = Template("""
#include <pycuda/pycuda-complex.hpp>

#define USE_DOUBLE ${use_double}
#define USE_COMPLEX ${use_complex}
#if USE_DOUBLE == 1
#if USE_COMPLEX == 1
#define TYPE pycuda::complex<double>
#else
#define TYPE double
#endif
#else
#if USE_COMPLEX == 1
#define TYPE pycuda::complex<float>
#else
#define TYPE float
#endif
#endif

// Ny: number of rows
// Nx: number of columns
__global__ void gen_trapz2d_mult(TYPE *mult,
                                 unsigned int Ny, unsigned int Nx) {
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;

    if (idx < Nx*Ny) {
        if (idx == 0 || idx == Nx-1 || idx == Nx*(Ny-1) || idx == Nx*Ny-1)
            mult[idx] = TYPE(0.25);
        else if ((idx > 0 && idx < Nx-1) || (idx % Nx == 0) ||
                (((idx + 1) % Nx) == 0) || (idx > Nx*(Ny-1) && idx < Nx*Ny-1))
            mult[idx] = TYPE(0.5);
        else 
            mult[idx] = TYPE(1.0);
    }
}
""")

def gen_trapz2d_mult(mat_shape, mult_type, dev):
    """
    Generate multiplication matrix for 2D trapezoidal integration.

    Generates a matrix whose dot product with some other matrix of
    equal length (when flattened) is equivalent to the definite double
    integral of the latter computed using trapezoidal integration.

    Parameters
    ----------
    mat_shape : tuple
        Shape of matrix.
    mult_type : float type
        Floating point type to use when generating the array.
    dev : pycuda.driver.Device
        Device object to be used.

    Returns
    -------
    result : pycuda.gpuarray.GPUArray
        Generated matrix.

    """

    if mult_type not in [np.float32, np.float64, np.complex64,
                         np.complex128]:
        raise ValueError('unrecognized type')

    use_double = int(mult_type in [np.float64, np.complex128])
    use_complex = int(mult_type in [np.complex64, np.complex128])

    # Allocate output matrix:
    Ny, Nx = mat_shape
    mult_gpu = gpuarray.empty(mat_shape, mult_type)

    # Get block/grid sizes:
    max_threads_per_block, max_block_dim, max_grid_dim = get_dev_attrs(dev)
    block_dim, grid_dim = select_block_grid_sizes(dev, mat_shape)
    max_blocks_per_grid = max(max_grid_dim)
    
    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    gen_trapz2d_mult_mod = \
                         SourceModule(gen_trapz2d_mult_template.substitute(use_double=use_double,
                                                                           use_complex=use_complex,
                                    max_threads_per_block=max_threads_per_block,
                                    max_blocks_per_grid=max_blocks_per_grid),
                                    cache_dir=cache_dir)

    gen_trapz2d_mult = gen_trapz2d_mult_mod.get_function("gen_trapz2d_mult")    
    gen_trapz2d_mult(mult_gpu, np.uint32(Ny), np.uint32(Nx),
                     block=block_dim,
                     grid=grid_dim)
    
    return mult_gpu

def trapz2d(x_gpu, dev, dx=1.0, dy=1.0):
    """
    2D trapezoidal integration.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Input matrix to integrate.
    dx : float
        X-axis spacing.
    dy : float
        Y-axis spacing
        
    Returns
    -------
    result : float
        Definite double integral as approximated by the trapezoidal rule.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray
    >>> import numpy as np
    >>> import integrate
    >>> integrate.init()
    >>> x = np.asarray(np.random.rand(10, 10), np.float32)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> z = integrate.trapz2d(x_gpu, pycuda.autoinit.device)
    >>> np.allclose(np.trapz(np.trapz(x)), z)
    True

    """
    
    if len(x_gpu.shape) != 2:
        raise ValueError('input array must be 2D')
    if np.iscomplex(dx) or np.iscomplex(dy):
        raise ValueError('dx and dy must be real')

    float_type = x_gpu.dtype.type
    if float_type == np.complex64:
        cublas_func = cublas._libcublas.cublasCdotu        
    elif float_type == np.float32:
        cublas_func = cublas._libcublas.cublasSdot
    elif float_type == np.complex128:
        cublas_func = cublas._libcublas.cublasZdotu
    elif float_type == np.float64:
        cublas_func = cublas._libcublas.cublasDdot
    else:
        raise ValueError('unsupported input type')
                                            
    trapz_mult_gpu = gen_trapz2d_mult(x_gpu.shape, float_type, dev)
    result = cublas_func(x_gpu.size, int(x_gpu.gpudata), 1,
                        int(trapz_mult_gpu.gpudata), 1)

    dxdy = float_type(dx)*float_type(dy)
    if float_type == np.complex64:
        return (np.float32(result.x)+1j*np.float32(result.y))*dxdy
    elif float_type == np.complex128:
        return (np.float64(result.x)+1j*np.float64(result.y))*dxdy
    elif float_type == np.float32:
        return np.float32(result)*dxdy
    else:
        return np.float64(result)*dxdy
