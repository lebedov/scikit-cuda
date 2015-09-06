#!/usr/bin/env python

"""
PyCUDA-based integration functions.
"""

from string import Template
from pycuda.tools import context_dependent_memoize
from pycuda.compiler import SourceModule
import pycuda.elementwise as elementwise
import pycuda.gpuarray as gpuarray
import pycuda.tools as tools
import numpy as np

import cublas
import misc

from misc import init

def gen_trapz_mult(N, dtype):
    """
    Generate multiplication array for 1D trapezoidal integration.

    Generates an array whose dot product with some array of equal
    length is equivalent to the definite integral of the latter
    computed using trapezoidal integration.

    Parameters
    ----------
    N : int
        Length of array.
    dtype : float type
        Floating point type to use when generating the array.

    Returns
    -------
    result : pycuda.gpuarray.GPUArray
        Generated array.
    """

    if dtype not in [np.float32, np.float64, np.complex64,
                     np.complex128]:
        raise ValueError('unrecognized type')

    ctype = tools.dtype_to_ctype(dtype)
    func = elementwise.ElementwiseKernel("{ctype} *x".format(ctype=ctype),
                                         "x[i] = ((i == 0) || (i == {M})) ? 0.5 : 1".format(M=N-1))
    x_gpu = gpuarray.empty(N, dtype)
    func(x_gpu)
    return x_gpu

def trapz(x_gpu, dx=1.0, handle=None):
    """
    1D trapezoidal integration.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Input array to integrate.
    dx : scalar
        Spacing.
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.

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
    >>> z = integrate.trapz(x_gpu)
    >>> np.allclose(np.trapz(x), z)
    True
    """

    if handle is None:
        handle = misc._global_cublas_handle

    if len(x_gpu.shape) > 1:
        raise ValueError('input array must be 1D')
    if np.iscomplex(dx):
        raise ValueError('dx must be real')

    float_type = x_gpu.dtype.type
    if float_type == np.complex64:
        cublas_func = cublas.cublasCdotu
    elif float_type == np.float32:
        cublas_func = cublas.cublasSdot
    elif float_type == np.complex128:
        cublas_func = cublas.cublasZdotu
    elif float_type == np.float64:
        cublas_func = cublas.cublasDdot
    else:
        raise ValueError('unsupported input type')

    trapz_mult_gpu = gen_trapz_mult(x_gpu.size, float_type)
    result = cublas_func(handle, x_gpu.size, x_gpu.gpudata, 1,
                         trapz_mult_gpu.gpudata, 1)

    return float_type(dx)*result


@context_dependent_memoize
def _get_trapz2d_mult_kernel(use_double, use_complex):
    template = Template("""
    #include <pycuda-complex.hpp>

    #if ${use_double}
    #if ${use_complex}
    #define TYPE pycuda::complex<double>
    #else
    #define TYPE double
    #endif
    #else
    #if ${use_complex}
    #define TYPE pycuda::complex<float>
    #else
    #define TYPE float
    #endif
    #endif

    // Ny: number of rows
    // Nx: number of columns
    __global__ void gen_trapz2d_mult(TYPE *mult,
                                     unsigned int Ny, unsigned int Nx) {
        unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                           blockIdx.x*blockDim.x+threadIdx.x;

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

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    tmpl = template.substitute(use_double=use_double, use_complex=use_complex)
    cache_dir=None
    mod = SourceModule(tmpl, cache_dir=cache_dir)
    return mod.get_function("gen_trapz2d_mult")

def gen_trapz2d_mult(mat_shape, dtype):
    """
    Generate multiplication matrix for 2D trapezoidal integration.

    Generates a matrix whose dot product with some other matrix of
    equal length (when flattened) is equivalent to the definite double
    integral of the latter computed using trapezoidal integration.

    Parameters
    ----------
    mat_shape : tuple
        Shape of matrix.
    dtype : float type
        Floating point type to use when generating the array.

    Returns
    -------
    result : pycuda.gpuarray.GPUArray
        Generated matrix.
    """

    if dtype not in [np.float32, np.float64, np.complex64,
                         np.complex128]:
        raise ValueError('unrecognized type')

    use_double = int(dtype in [np.float64, np.complex128])
    use_complex = int(dtype in [np.complex64, np.complex128])

    # Allocate output matrix:
    Ny, Nx = mat_shape
    mult_gpu = gpuarray.empty(mat_shape, dtype)

    # Get block/grid sizes:
    dev = misc.get_current_device()
    block_dim, grid_dim = misc.select_block_grid_sizes(dev, mat_shape)
    gen_trapz2d_mult = _get_trapz2d_mult_kernel(use_double, use_complex)
    gen_trapz2d_mult(mult_gpu, np.uint32(Ny), np.uint32(Nx),
                     block=block_dim,
                     grid=grid_dim)

    return mult_gpu

def trapz2d(x_gpu, dx=1.0, dy=1.0, handle=None):
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
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.

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
    >>> z = integrate.trapz2d(x_gpu)
    >>> np.allclose(np.trapz(np.trapz(x)), z)
    True
    """

    if handle is None:
        handle = misc._global_cublas_handle

    if len(x_gpu.shape) != 2:
        raise ValueError('input array must be 2D')
    if np.iscomplex(dx) or np.iscomplex(dy):
        raise ValueError('dx and dy must be real')

    float_type = x_gpu.dtype.type
    if float_type == np.complex64:
        cublas_func = cublas.cublasCdotu
    elif float_type == np.float32:
        cublas_func = cublas.cublasSdot
    elif float_type == np.complex128:
        cublas_func = cublas.cublasZdotu
    elif float_type == np.float64:
        cublas_func = cublas.cublasDdot
    else:
        raise ValueError('unsupported input type')

    trapz_mult_gpu = gen_trapz2d_mult(x_gpu.shape, float_type)
    result = cublas_func(handle, x_gpu.size, x_gpu.gpudata, 1,
                         trapz_mult_gpu.gpudata, 1)

    return float_type(dx)*float_type(dy)*result

if __name__ == "__main__":
    import doctest
    doctest.testmod()
