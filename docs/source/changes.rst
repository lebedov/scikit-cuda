.. -*- rst -*-

Change Log
==========

Release 0.5.4 (Under Development)
---------------------------------
* Add wrappers for matrix-matrix multiplication of batches of matrices (#278).

Release 0.5.3 (May 26, 2019)
----------------------------
* Add support for CUDA 10.*.
* Add MAGMA GELS wrappers (#271).
* Add context-dependent memoization to skcuda.fft and other modules (#273).
* Fix issues finding CUDA libraries on Windows.
  
Release 0.5.2 (November 6, 2018)
--------------------------------
* Prevent exceptions when CULA Dense free is present (#146).
* Fix Python 3 issues with CUSOLVER wrapper functions (#145)
* Add support for using either CUSOLVER or CULA for computing SVD.
* Add support for using either CUSOLVER or CULA for computing determinant.
* Compressed Dynamic Mode Decomposition (enh. by N. Benjamin Erichson).
* Support for CUFFT extensible plan API (enh. by Bruce Merry).
* Wrappers for CUFFT size estimation (enh. by Luke Pfister).
* Wrappers for CUBLAS-XT functions.
* More wrappers for MAGMA functions (enh. by Nikul H. Ukani).
* Python 3 compatibility improvements (enh. by Joseph Martinot-Lagarde).
* Allow specification of order in misc.zeros and misc.ones.
* Preserve strides in misc.zeros_like and misc.ones_like.
* Add support for Cholesky factorization/solving using CUSOLVER (#198).
* Add cholesky() function that zeros out non-factor entries in result (#199).
* Add support for CUDA 8.0 libraries (#171).
* Workaround for libgomp + CUDA 8.0 weirdness (fix by Kevin Flansburg).
* Fix broken matrix-vector dot product (#156).
* Initialize MAGMA before CUSOLVER to prevent internal errors in certain
  CUSOLVER functions.
* Skip CULA-dependent unit tests when CULA isn't present.
* CUSOLVER support for symmetric eigenvalue decomposition (enh. by Bryant Menn).
* CUSOLVER support for matrix inversion, QR decomposition (#198).
* Prevent objdump output from changing due to environment language (fix by 
  Arnaud Bergeron).
* Fix diag() support for column-major 2D array inputs (#219).
* Use absolute path for skcuda header includes (enh. by S. Clarkson).
* Fix QR issues by reverting fix for #131 and raising PyCUDA version requirement 
  (fix by S. Clarkson).
* More batch CUBLAS wrappers (enh. by Li Yong Liu)
* Numerical integration with Simpson's Rule (enh. by Alexander Weyman)
* Make CUSOLVER default backend for functions that can use either CULA or
  CUSOLVER.
* Fix CUDA errors that only occur when unit tests are run en masse with nose or
  setuptools (#257).
* Fix MAGMA eigenvalue decomposition wrappers (#265, fix by Wing-Kit Lee).

Release 0.5.1 - (October 30, 2015)
----------------------------------
* More CUSOLVER wrappers.
* Eigenvalue/eigenvector computation (eng. by N. Benjamin Erichson).
* QR decomposition (enh. by N. Benjamin Erichson).
* Improved Windows 10 compatibility (enh. by N. Benjamin Erichson).
* Function for constructing Vandermonde matrix in GPU memory (enh. by N. Benjamin Erichson).
* Standard and randomized Dynamic Mode Decomposition (enh. by N. Benjamin Erichson).
* Randomized linear algebra routines (enh. by N. Benjamin Erichson).
* Add triu function (enh. by N. Benjamin Erichson).
* Support Bessel correction in computation of variance and standard 
  deviation (#143).
* Fix pip installation issues.

Release 0.5.0 - (July 14, 2015)
-------------------------------
* Rename package to scikit-cuda.
* Reductions sum, mean, var, std, max, min, argmax, argmin accept keepdims option.
* The same reductions now return a GPUArray instead of ndarray if axis=None.
* Switch to PEP 440 version numbering.
* Replace distribute_setup.py with ez_setup.py.
* Improve support for latest NVIDIA GPUs.
* Direct links to online NVIDIA documentation in CUBLAS, CUFFT wrapper
  docstrings.
* Add wrappers for CUSOLVER in CUDA 7.0.
* Add skcuda namespace package that contains all modules in scikits.cuda namespace.
* Add more wrappers for CUBLAS 5 functions (enh. by Teodor Moldovan, Sander
  Dieleman).
* Add support for CULA Dense Free R17 (enh. by Alex Rubinsteyn).
* Memoize elementwise kernel used by ifft scaling (#37).
* Speed up misc.maxabs using reduction and kernel memoization.
* Speed up misc.cumsum using scan and kernel memoization.
* Speed up linalg.conj and misc.diff using elementwise kernel and memoization.
* Speed up special.{sici,exp1,expi} using elementwise kernel and memoization.
* Add wrappers for experimental multi-GPU CULA routines in CULA Dense R14+.
* Use ldconfig to find library paths rather than libdl (#39).
* Fix win32 platform detection.
* Add Cholesky factorization/solve routines (enh. by Steve Taylor).
* Fix Cholesky factorization/solve routines (fix by Thomas Unterthiner).
* Enable dot() function to operate inplace (enh. by Thomas Unterthiner).
* Python 3 compatibility improvements (enh. by Thomas Unterthiner).
* Support for Fortran-order arrays in dot() and cho_solve() (enh. by Thomas Unterthiner)
* CULA-based matrix inversion (enh. by Thomas Unterthiner).
* Add add_diag() function (enh. by Thomas Unterthiner).
* Use cublas*copy in diag() function (enh. by Thomas Unterthiner).
* Improved MacOSX compatibility (enh. by Michael M. Forbes).
* Find CUBLAS version even when it is only accessible via LD_LIBRARY_PATH (enh. by Frédéric Bastien).
* Get both major and minor version numbers from CUBLAS library when determining
  version.
* Handle unset LD_LIBRARY_PATH variable (fix by Jan Schlüter).
* Fix library search on MacOS X (fix by capdevc).
* Fix library search on Windows.
* Add Windows support to CULA wrappers.
* Enable specification of memory pool allocator to linalg functions (enh.  by
  Thomas Unterthiner).
* Improve misc.select_block_grid_sizes() logic to handle different GPU hardware.
* Compute transpose using CUDA 5.0 CUBLAS functions rather than with inefficient naive kernel.
* Use ReadTheDocs theme when building HTML docs locally.
* Support additional cufftPlanMany() parameters when creating FFT plans (enh. by
  Gregory R. Lee).
* Improved Python 3.4 compatibility (enh. by Eric Larson).
* Avoid unnecessary import of cublas when importing fft module (enh. by Eric
  Larson).
* Matrix trace function (enh. by Thomas Unterthiner).
* Functions for computing simple axis-wise stats over matrices (enh. by Thomas
  Unterthiner).
* Matrix add_dot, add_matvec, div_matvec, mult_matvec functions (enh. by Thomas
  Unterthiner).
* Faster dot_diag implementation using CUBLAS matrix-matrix multiplication (enh.
  by Thomas Unterthiner).
* Memoize SourceModule calls to speed up various high-level functions (enh. by
  Thomas Unterthiner).
* Function for computing matrix determinant (enh. by Thomas Unterthiner).
* Function for computing min/max and argmin/argmax along a matrix axis
  (enh. by Thomas Unterthiner).
* Set default value of the parameter 'overwrite' to False in all linalg
  functions.
* Elementwise arithmetic operations with broadcasting up to 2 dimensions
  (enh. David Wei Chiang)

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
