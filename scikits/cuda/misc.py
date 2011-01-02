#!/usr/bin/env python

"""
General PyCUDA utility functions.
"""

import string
from string import Template
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np

def get_compute_capability(dev):
    """
    Get the compute capability of the specified device.

    Retrieve the compute capability of the specified CUDA device and
    return it as a floating point value.

    Parameters
    ----------
    d : pycuda.driver.Device
        Device object to examine.

    Returns
    -------
    c : float
        Compute capability.

    """

    return np.float(string.join([str(i) for i in
                                 dev.compute_capability()], '.'))

def get_dev_attrs(dev):
    """
    Get select CUDA device attributes.

    Retrieve select attributes of the specified CUDA device that
    relate to maximum thread block and grid sizes.
    
    Parameters
    ----------
    d : pycuda.driver.Device
        Device object to examine.

    Returns
    -------
    attrs : list
        List containing [MAX_THREADS_PER_BLOCK,
        (MAX_BLOCK_DIM_X, MAX_BLOCK_DIM_Y, MAX_BLOCK_DIM_Z),
        (MAX_GRID_DIM_X, MAX_GRID_DIM_Y)]
        
    """
    
    attrs = dev.get_attributes()
    return [attrs[drv.device_attribute.MAX_THREADS_PER_BLOCK],
            (attrs[drv.device_attribute.MAX_BLOCK_DIM_X],
             attrs[drv.device_attribute.MAX_BLOCK_DIM_Y],
             attrs[drv.device_attribute.MAX_BLOCK_DIM_Z]),
            (attrs[drv.device_attribute.MAX_GRID_DIM_X],
            attrs[drv.device_attribute.MAX_GRID_DIM_Y])]


def select_block_grid_sizes(dev, data_shape, threads_per_block=None):
    """
    Determine CUDA block and grid dimensions given device constraints.

    Determine the CUDA block and grid dimensions allowed by a GPU
    device that are sufficient for processing every element of an
    array in a separate thread.
    
    Parameters
    ----------
    d : pycuda.driver.Device
        Device object to be used.
    data_shape : tuple
        Shape of input data array. Must be of length 2.
    threads_per_block : int, optional
        Number of threads to execute in each block. If this is None,
        the maximum number of threads per block allowed by device `d`
        is used.
        
    Returns
    -------
    block_dim : tuple
        X, Y, and Z dimensions of minimal required thread block.
    grid_dim : tuple
        X and Y dimensions of minimal required block grid.

    Notes
    -----
    Using the scheme in this function, all of the threads in the grid can be enumerated
    as `i = blockIdx.y*max_threads_per_block*max_blocks_per_grid+
    blockIdx.x*max_threads_per_block+threadIdx.x`.
    The indices of the element `data[ix, iy]` where `data.shape == [r, c]`
    can be computed as `ix = int(floor(i/c))` and `iy = i % c`.

    It is advisable that the number of threads per block be a multiple
    of the warp size to fully utilize a device's computing resources.
    
    """

    # Sanity checks:
    if np.isscalar(data_shape):
        data_shape = (data_shape,)
    if len(data_shape) > 2:
        raise ValueError('data arrays of dimension > 2 not yet supported')

    # Number of elements to process; we need to cast the result of
    # np.prod to a Python int to prevent PyCUDA's kernel execution
    # framework from getting confused when
    N = int(np.prod(data_shape))

    # Get device constraints:
    max_threads_per_block, max_block_dim, max_grid_dim = get_dev_attrs(dev)

    if threads_per_block != None:
        max_threads_per_block = threads_per_block
        
    # Assume that the maximum number of threads per block is no larger
    # than the maximum X and Y dimension of a thread block:
    assert max_threads_per_block <= max_block_dim[0]
    assert max_threads_per_block <= max_block_dim[1]

    # Assume that the maximum X and Y dimensions of a grid are the
    # same:
    max_blocks_per_grid = max(max_grid_dim)
    assert max_blocks_per_grid == max_grid_dim[0]
    assert max_blocks_per_grid == max_grid_dim[1]

    # Actual number of thread blocks needed:
    blocks_needed = N/max_threads_per_block+1
    
    if blocks_needed*max_threads_per_block < max_threads_per_block*max_blocks_per_grid:
        grid_x = blocks_needed
        grid_y = 1
    elif blocks_needed*max_threads_per_block < max_threads_per_block*max_blocks_per_grid**2:
        grid_x = max_blocks_per_grid
        grid_y = blocks_needed/max_blocks_per_grid+1
    else:
        raise ValueError('array size too large')

    return (max_threads_per_block, 1, 1), (grid_x, grid_y)

maxabs_mod_template = Template("""
#include <cuComplex.h>

#define USE_DOUBLE ${use_double}
#define USE_COMPLEX ${use_complex}

#if USE_DOUBLE == 1
#define REAL_TYPE double
#if USE_COMPLEX == 1
#define TYPE cuDoubleComplex
#define ABS(z) cuCabs(z)
#else
#define TYPE double
#define ABS(X) abs(x)
#endif
#else
#define REAL_TYPE float
#if USE_COMPLEX == 1
#define TYPE cuFloatComplex
#define ABS(z) cuCabsf(z)
#else
#define TYPE float
#define ABS(x) fabs(x)
#endif
#endif

// This kernel is only meant to be run in one thread;
// N must contain the length of x:
__global__ void maxabs(TYPE *x, REAL_TYPE *m,
                       unsigned int N) {
    unsigned int idx = threadIdx.x;

    REAL_TYPE result, temp;
    
    if (idx == 0) {
        result = 0;
        for (unsigned int i = 0; i < N; i++) {
           temp = ABS(x[i]);
           if (temp >= result)
               result = temp;
        }
        m[0] = result;
    }
}                             
""")

def maxabs(x_gpu):
    """
    Get maximum absolute value.

    Find maximum absolute value in the specified array.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Input array.

    Returns
    -------
    m_gpu : pycuda.gpuarray.GPUArray
        Length 1 array containing the maximum absolute value in
        `x_gpu`.

    Notes
    -----
    This implementation could be made faster by computing the absolute
    values of the input array in parallel.
    
    Example
    -------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import fft
    >>> x_gpu = gpuarray.to_gpu(np.array([-1, 2, -3], np.float32))
    >>> m_gpu = fft.maxabs(x_gpu)
    >>> np.allclose(m_gpu.get(), 3.0)
    True
    >>> y_gpu = gpuarray.to_gpu(np.array([-1j, 2, -3j], np.complex64))
    >>> m_gpu = fft.maxabs(y_gpu)
    >>> np.allclose(m_gpu.get(), 3.0)
    True
    
    """

    if x_gpu.dtype == np.double:
        use_double = 1
        real_type = np.float64
    else:
        use_double = 0
        real_type = np.float32
    
    if x_gpu.dtype in [np.complex64, np.complex128]:
        use_complex = 1
    else:
        use_complex = 0

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir = None
    maxabs_mod = \
               SourceModule(maxabs_mod_template.substitute(use_double=use_double,
                                                           use_complex=use_complex),
                            cache_dir=cache_dir)

    maxabs = maxabs_mod.get_function("maxabs")
    m_gpu = gpuarray.empty(1, real_type)
    maxabs(x_gpu.gpudata, m_gpu.gpudata, np.uint32(x_gpu.size),
           block=(1, 1, 1), grid=(1, 1))

    return m_gpu

if __name__ == "__main__":
    import doctest
    doctest.testmod()
