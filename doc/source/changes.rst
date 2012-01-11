.. -*- rst -*-

Change Log
==========

Release 0.042 - ()
------------------
* Add complex exponential integral.
* Fix typo in cublasCgbmv.
* Correctly load BLAS symbols on MacOSX and with CUDA 4.0.
* Work around numpy bug #1898.
* Fix issues with pycuda installations done via easy_install/pip. 
* Add support for specifying streams when creating FFT plans.
* Successfully find CULA R13a libraries.

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
------------------------------------
* Add documentation.
* Update copyright information.

Release 0.01 - (September 17, 2010)
-----------------------------------
* First public release.

