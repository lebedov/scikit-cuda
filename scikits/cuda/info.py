#!/usr/bin/env python

"""
scikits.cuda
============

This SciKit (toolkit for SciPy [1]_) provides Python interfaces to a
subset of the functions in the CUDA, CUDART, CUBLAS, and CUFFT
libraries distributed as part of NVIDIA's CUDA Programming Toolkit
[2]_, as well as interfaces to select functions in the free-of-charge
CULA Toolkit [3]_. In contrast to most existing Python wrappers for
these libraries (many of which only provide a low-level interface to
the actual library functions), this package uses PyCUDA [4]_ to
provide high-level functions comparable to those in the NumPy package
[5]_.


High-level modules
------------------

- autoinit	 Import this module to automatically initialize CUBLAS and CULA.
- fft            Fast Fourier Transform functions.
- integrate	 Numerical integration functions.
- linalg         Linear algebra functions.
- misc           Miscellaneous support functions.
- special        Special math functions.

Low-level modules
-----------------

- cublas         Wrappers for functions in the CUBLAS library.
- cufft          Wrappers for functions in the CUFFT library.
- cuda           Wrappers for functions in the CUDA/CUDART libraries.
- cula           Wrappers for functions in the CULA library.

.. [1] http://www.scipy.org/
.. [2] http://www.nvidia.com/cuda
.. [3] http://www.culatools.com/
.. [4] http://mathema.tician.de/software/pycuda/
.. [5] http://numpy.scipy.org/
.. [6] http://bionet.ee.columbia.edu/
.. [7] http://www.mathcs.emory.edu/~yfan/PARRET

"""
