#!/usr/bin/env python

"""
Demo of how to call low-level MAGMA wrappers to perform SVD decomposition.

Note MAGMA's SVD implementation is a hybrid of CPU/GPU code; the inputs
therefore must be in host memory.
"""

import numpy as np
import skcuda.magma as magma

magma.magma_init()
x = np.asarray([[1.80, 2.88, 2.05, -0.89],
                [5.25, -2.95, -0.95, -3.80], 
                [1.58, -2.69, -2.90, -1.04],
                [-1.11, -0.66, -0.59, 0.80]]).astype(np.float32)
x_orig = x.copy()

# Need to reverse dimensions because MAGMA expects column-major matrices:
n, m = x.shape

# Set up output buffers:
s = np.zeros(min(m, n), np.float32)
u = np.zeros((m, m), np.float32)
vh = np.zeros((n, n), np.float32)

# Set up workspace:
Lwork = magma.magma_sgesvd_buffersize('A', 'A', m, n, x.ctypes.data, m, s.ctypes.data,
                                      u.ctypes.data, m, vh.ctypes.data, n)
workspace = np.zeros(Lwork, np.float32)

# Compute:
status = magma.magma_sgesvd('A', 'A', m, n, x.ctypes.data, m, s.ctypes.data,
                            u.ctypes.data, m, vh.ctypes.data, n,
                            workspace.ctypes.data, Lwork)

# Confirm that solution is correct by ensuring that the original matrix can be
# obtained from the decomposition:
print 'correct solution: ', np.allclose(x_orig, np.dot(vh, np.dot(np.diag(s), u)), 1e-4)
magma.magma_finalize()
