#!/usr/bin/env python
from __future__ import division, print_function

"""
Python interface to cuSPARSE functions.

Note: You may need to set the environment variable CUDA_ROOT to the base of
your CUDA installation.
"""
# import low level cuSPARSE python wrappers and constants

try:
    from ._cusparse_cffi import *
except Exception as e:
    estr = "autogenerattion and import of cuSPARSE wrappers failed\n"
    estr += ("Try setting the CUDA_ROOT environment variable to the base of"
             "your CUDA installation.  The autogeneration script tries to find"
             "the CUSPARSE header at CUDA_ROOT/include/cusparse_v2.h\n")
    raise ImportError(estr)
