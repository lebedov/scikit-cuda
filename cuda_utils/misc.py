#!/usr/bin/env python

"""
General PyCUDA utility functions.
"""

import pycuda.driver as drv
import numpy as np

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


def select_block_grid_sizes(dev, data_shape):
    """
    Determine CUDA block and grid sizes given device constraints.

    Parameters
    ----------
    d : pycuda.driver.Device
        Device object to be used.
    data_shape : tuple
        Shape of input data array. Must be of length 2.

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
    
    # Assume that the maximum number of threads per block is equal to
    # the maximum X and Y dimension of a thread block:
    assert max_threads_per_block == max_block_dim[0]
    assert max_threads_per_block == max_block_dim[1]

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
