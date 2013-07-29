.. -*- rst -*-

Change Log
==========

Release 0.043 - (under development)
-----------------------------------
* Improve support for latest NVIDIA GPUs.
* Add more wrappers for CUBLAS 5 functions (enh. by Teodor Moldovan)
* Add support for CULA Dense Free R17 (enh. by Alex Rubinsteyn)
* Memoize elementwise kernel used by ifft scaling (#37).
* Speed up misc.maxabs using reduction and kernel memoization.
* Speed up misc.cumsum using scan and kernel memoization.
* Speed up linalg.conj using elementwise kernel and memoization.
* Add wrappers for experimental multi-GPU CULA routines in CULA Dense R14+.

Release 0.042 - (March 10, 2013)
--------------------------------
* Add complex exponential integral.
* Fix typo in cublasCgbmv.
* Use CUBLAS v2 API, add preliminary support for CUBLAS 5 functions.
* Detect CUBLAS version without initializing the GPU.
* Work around numpy bug #1898.
* Fix issues with pycuda installations done via easy_install/pip. 
* Add support for specifying streams when creating FFT plans.
* Successfully find CULA R13a libraries.
* Raise exceptions when functions in the full release of CULA Dense are invoked
  without the library installed.
* Perform post-fft scaling in-place.
* Fix broken Python 2.6 compatibility (#19).
* Download distribute for package installation if it isn't available.
* Prevent absence of CULA from causing import errors (enh. by Jacob Frelinger)
* FFT batch tests and FFTW mode configuration (enh. by Lars Pastewka)

Release 0.041 - (May 22, 2011)
------------------------------
* Fix bug preventing installation with pip.

Release 0.04 - (May 11, 2011)
-----------------------------
* Fix bug in cutoff_invert kernel.
* Add get_compute_capability function and other goodies to misc module.
* Use pycuda-complex.hpp to improve kernel readability.
* Add integrate module.
* Add unit tests for high-level functions.
* Automatically determine device used by current context.
* Support batched and multidimensional FFT operations.
* Extended dot() function to support implicit transpose/Hermitian.
* Support for in-place computation of singular vectors in svd() function.
* Simplify kernel launch setup.
* More CULA routine wrappers.
* Wrappers for CULA R11 auxiliary routines.

Release 0.03 - (November 22, 2010)
----------------------------------
* Add support for some functions in the premium version of CULA toolkit.
* Add wrappers for all lapack functions in basic CULA toolkit.
* Fix pinv() to properly invert complex matrices.
* Add Hermitian transpose.
* Add tril function.
* Fix missing library detection.
* Include missing CUDA headers in package.

Release 0.02 - (September 21, 2010)
-----------------------------------
* Add documentation.
* Update copyright information.

Release 0.01 - (September 17, 2010)
-----------------------------------
* First public release.

