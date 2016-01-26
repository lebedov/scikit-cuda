#!/usr/bin/env python

"""
Demonstrates how to access 3D arrays within a PyCUDA kernel in a
numpy-consistent manner.
"""
from __future__ import print_function

from string import Template
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np

import skcuda.misc as misc

A = 3
B = 4
C = 5
N = A * B * C

# Define a 3D array:
# x_orig = np.arange(0, N, 1, np.float64)
x_orig = np.asarray(np.random.rand(N), np.float64)
x = x_orig.reshape((A, B, C))

# These functions demonstrate how to convert a linear index into subscripts:
a = lambda i: i / (B * C)
b = lambda i: np.mod(i, B * C) / C
c = lambda i: np.mod(np.mod(i, B * C), C)

# Check that x[ind(i)] is equivalent to x.flat[i]:
subscript = lambda i: (a(i), b(i), c(i))
for i in range(x.size):
    assert x.flat[i] == x[subscript(i)]

# Check that x[i,j,k] is equivalent to x.flat[index(i,j,k)]:
index = lambda i, j, k: i * B * C + j * C + k
for i in range(A):
    for j in range(B):
        for k in range(C):
            assert x[i, j, k] == x.flat[index(i, j, k)]

func_mod_template = Template("""
// Macro for converting subscripts to linear index:
#define INDEX(a, b, c) a*${B}*${C}+b*${C}+c

__global__ void func(double *x, unsigned int N) {
    // Obtain the linear index corresponding to the current thread:
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;

    // Convert the linear index to subscripts:
    unsigned int a = idx/(${B}*${C});
    unsigned int b = (idx%(${B}*${C}))/${C};
    unsigned int c = (idx%(${B}*${C}))%${C};

    // Use the subscripts to access the array:
    if (idx < N) {
        if (b == 0)
           x[INDEX(a,b,c)] = 100;
    }
}
""")

max_threads_per_block, max_block_dim, max_grid_dim = misc.get_dev_attrs(pycuda.autoinit.device)
block_dim, grid_dim = misc.select_block_grid_sizes(pycuda.autoinit.device, x.shape)
max_blocks_per_grid = max(max_grid_dim)

func_mod = \
    SourceModule(func_mod_template.substitute(max_threads_per_block=max_threads_per_block,
                                              max_blocks_per_grid=max_blocks_per_grid,
                                              A=A, B=B, C=C))
func = func_mod.get_function('func')
x_gpu = gpuarray.to_gpu(x)
func(x_gpu.gpudata, np.uint32(x_gpu.size),
     block=block_dim,
     grid=grid_dim)
x_np = x.copy()
x_np[:, 0, :] = 100

print('Success status: ', np.allclose(x_np, x_gpu.get()))
