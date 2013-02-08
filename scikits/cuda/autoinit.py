#!/usr/bin/env python

"""
Autoinitialize CUDA tools.
"""

import cublas
try:
    import cula
    _has_cula = True
except (ImportError, OSError):
    _has_cula = False

cublas.cublasInit()
if _has_cula:
    cula.culaInitialize()
