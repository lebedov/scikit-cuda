#!/usr/bin/env python

"""
Autoinitialize CUDA tools.
"""

import cublas
import cula

cublas.cublasInit()
cula.culaInitialize()
