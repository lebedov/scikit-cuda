#!/usr/bin/env python

"""
Demonstrates how to access 4D arrays within a PyCUDA kernel in a
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
D = 6
N = A * B * C * D

# Define a 3D array:
# x_orig = np.arange(0, N, 1, np.float64)
x_orig = np.asarray(np.random.rand(N), np.float64)
x = x_orig.reshape((A, B, C, D))

# These functions demonstrate how to convert a linear index into subscripts:
a = lambda i: i / (B * C * D)
b = lambda i: np.mod(i, B * C * D) / (C * D)
c = lambda i: np.mod(np.mod(i, B * C * D), C * D) / D
d = lambda i: np.mod(np.mod(np.mod(i, B * C * D), C * D), D)

# Check that x[subscript(i)] is equivalent to x.flat[i]:
subscript = lambda i: (a(i), b(i), c(i), d(i))
for i in range(x.size):
    assert x.flat[i] == x[subscript(i)]

# Check that x[i,j,k,l] is equivalent to x.flat[index(i,j,k,l)]:
index = lambda i, j, k, l: i * B * C * D + j * C * D + k * D + l
for i in range(A):
    for j in range(B):
        for k in range(C):
            for l in range(D):
                assert x[i, j, k, l] == x.flat[index(i, j, k, l)]

func_mod_template = Template("""
// Macro for converting subscripts to linear index:
#define INDEX(a, b, c, d) a*${B}*${C}*${D}+b*${C}*${D}+c*${D}+d

__global__ void func(double *x, unsigned int N) {
    // Obtain the linear index corresponding to the current thread:
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;

    // Convert the linear index to subscripts:
    unsigned int a = idx/(${B}*${C}*${D});
    unsigned int b = (idx%(${B}*${C}*${D}))/(${C}*${D});
    unsigned int c = ((idx%(${B}*${C}*${D}))%(${C}*${D}))/${D};
    unsigned int d = ((idx%(${B}*${C}*${D}))%(${C}*${D}))%${D};

    // Use the subscripts to access the array:
    if (idx < N) {
        if (c == 0)
           x[INDEX(a,b,c,d)] = 100;
    }
}
""")

max_threads_per_block, max_block_dim, max_grid_dim = misc.get_dev_attrs(pycuda.autoinit.device)
block_dim, grid_dim = misc.select_block_grid_sizes(pycuda.autoinit.device, x.shape)
max_blocks_per_grid = max(max_grid_dim)

func_mod = \
    SourceModule(func_mod_template.substitute(max_threads_per_block=max_threads_per_block,
                                              max_blocks_per_grid=max_blocks_per_grid,
                                              A=A, B=B, C=C, D=D))
func = func_mod.get_function('func')
x_gpu = gpuarray.to_gpu(x)
func(x_gpu.gpudata, np.uint32(x_gpu.size),
     block=block_dim,
     grid=grid_dim)
x_np = x.copy()
x_np[:, :, 0, :] = 100

print('Success status: ', np.allclose(x_np, x_gpu.get()))
