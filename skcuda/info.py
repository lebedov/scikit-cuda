#!/usr/bin/env python

"""
scikit-cuda
===========
scikit-cuda provides Python interfaces to many of the functions
in the CUDA device/runtime, CUBLAS, CUFFT, and CUSOLVER
libraries distributed as part of NVIDIA's CUDA Programming Toolkit
[1]_, as well as interfaces to select functions in the free and standard
versions of the CULA Dense Toolkit [2]_. Both low-level wrapper functions
similar to their C counterparts and high-level functions comparable to those in
NumPy and Scipy [3]_ are provided

High-level modules
------------------
- autoinit       Automatic GPU library initialization module.
- fft            Fast Fourier Transform functions.
- integrate      Numerical integration functions.
- linalg         Linear algebra functions.
- rlinalg        Randomized linear algebra functions.
- misc           Miscellaneous support functions.
- special        Special math functions.

Low-level modules
-----------------
- cublas         Function wrappers for the CUBLAS library.
- cufft          Function wrappers for the CUFFT library.
- cuda           Function wrappers for the CUDA device/runtime libraries.
- cula           Function wrappers for the CULA library.
- cusolver       Function wrappers for the CUSOLVER library.
- pcula          Function wrappers for the multi-GPU CULA library.

.. [1] http://www.nvidia.com/cuda
.. [2] http://www.culatools.com/
.. [5] http://www.scipy.org/
"""
