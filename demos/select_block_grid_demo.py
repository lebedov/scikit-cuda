#!/usr/bin/env python

"""
Demonstrates how to use automatically selected block/grid sizes
in a PyCUDA kernel.
"""

from string import Template
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np

from scikits.cuda.misc import get_dev_attrs, select_block_grid_sizes

# Device selected by PyCUDA:
dev = pycuda.autoinit.device

# Allocate input and output arrays:
a = np.asarray(np.random.rand(1000, 1000), np.float32)
b = np.empty_like(a)

# Determine device constraints and block/grid sizes:
max_threads_per_block, max_block_dim, max_grid_dim = \
                       get_dev_attrs(dev)
block_dim, grid_dim = select_block_grid_sizes(dev, a.shape)

# Perform element-wise operation on input matrix:
func_mod_template = Template("""
__global__ void func(${float} *a, ${float} *b, unsigned int N) {
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;

    if (idx < N)
        b[idx] = 5*a[idx];
}
""")

func_mod = SourceModule(func_mod_template.substitute(float='float',
                                                     max_threads_per_block=str(max_threads_per_block),
                                                     max_blocks_per_grid=str(max(max_grid_dim))),
                                                     cache_dir=False)


func = func_mod.get_function('func')
exec_time = func(drv.In(a), drv.Out(b), np.uint32(np.prod(a.shape)),
                 block=block_dim, grid=grid_dim,
                 time_kernel=True)

print 'Success status: ', np.allclose(b, 5*a)
print 'exec time = ', exec_time
