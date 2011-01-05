#!/usr/bin/env python

"""
Demonstrates how to access 3D arrays within a PyCUDA kernel in a
numpy-consistent manner.
"""

from string import Template
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np

import scikits.cuda.misc as misc

R = 3
C = 4
S = 5
N = R*C*S

# Define a 3D array:
# x_orig = np.arange(0, N, 1, np.float64)
x_orig = np.asarray(np.random.rand(N), np.float64)
x = x_orig.reshape((R, C, S))

# These functions demonstrate how to convert a linear index into subscripts:
r = lambda i: i/(C*S)
c = lambda i: np.mod(i, C*S)/S
s = lambda i: np.mod(np.mod(i, C*S), S)

# x[ind(i)] should be equivalent to x.flat[i]:
ind = lambda i: (r(i), c(i), s(i))

func_mod_template = Template("""
// Macro for converting subscripts to linear index:
#define INDEX(r, c, s) r*${C}*${S}+c*${S}+s

__global__ void func(double *x, unsigned int N) {
    // Obtain the linear index corresponding to the current thread:
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;

    // Convert the linear index to subscripts:
    unsigned int r = idx/(${C}*${S});
    unsigned int c = (idx%(${C}*${S}))/${S};
    unsigned int s = (idx%(${C}*${S}))%${S};

    // Use the subscripts to access the array:
    if (idx < N) {
        if (c == 0)
           x[INDEX(r,c,s)] = 100;
    }
}
""")

max_threads_per_block, max_block_dim, max_grid_dim = misc.get_dev_attrs(pycuda.autoinit.device)
block_dim, grid_dim = misc.select_block_grid_sizes(pycuda.autoinit.device, x.shape)
max_blocks_per_grid = max(max_grid_dim)

func_mod = \
         SourceModule(func_mod_template.substitute(max_threads_per_block=max_threads_per_block,
                                                   max_blocks_per_grid=max_blocks_per_grid,
                                                   R=R, C=C, S=S))
func = func_mod.get_function('func')
x_gpu = gpuarray.to_gpu(x)
func(x_gpu.gpudata, np.uint32(x_gpu.size),
     block=block_dim,
     grid=grid_dim)
x_np = x.copy()
x_np[:, 0, :] = 100

print 'Success status: ', np.allclose(x_np, x_gpu.get())
