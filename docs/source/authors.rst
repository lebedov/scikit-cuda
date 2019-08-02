.. -*- rst -*-

Authors & Acknowledgments
=========================
This software was written and packaged by `Lev Givon 
<http://www.columbia.edu/~lev/>`_.  Although it
depends upon the excellent `PyCUDA <http://mathema.tician.de/software/pycuda/>`_ 
package by `Andreas Klöckner <http://mathema.tician.de/aboutme/>`_, scikit-cuda 
is developed independently of PyCUDA.

Special thanks are due to the following parties for their contributions:

- `Frédéric Bastien <https://github.com/nouiz>`_ - CUBLAS version detection enhancements.
- `Arnaud Bergeron <https://github.com/abergeron>`_ - Fix to prevent LANG from 
  affecting objdump output.
- `David Wei Chiang <https://github.com/davidweichiang>`_ - Improvements to 
  vectorized functions, bug fixes.
- `Sander Dieleman <https://github.com/benanne>`_ - CUBLAS 5 bindings.
- `Chris Capdevila <https://github.com/capdevc>`_ - MacOS X library search fix.
- `Ben Erichson <https://github.com/Benli11>`_ - QR decomposition, eigenvalue/eigenvector computation, Dynamic 
  Mode Decomposition, randomized linear algebra routines.
- `Ying Wei (Daniel) Fan
  <https://www.linkedin.com/pub/ying-wai-daniel-fan/5b/b8a/57>`_ - Kindly
  permitted reuse of CUBLAS wrapper code in his PARRET Python package.
- `Michael M. Forbes <https://github.com/mforbes>`_ - Improved MacOSX compatibility, bug fixes.
- `Jacob Frelinger <https://github.com/jfrelinger>`_ - Various enhancements.
- Tim Klein - Additional MAGMA wrappers.
- `Joseph Martinot-Lagarde <https://github.com/Nodd>`_ - Python 3 compatibility 
  improvements.
- `Eric Larson <https://github.com/larsoner>`_ - Various enhancements.
- `Gregory R. Lee <https://github.com/grlee77>`_ - Enhanced FFT plan creation.
- `Bryant Menn <https://github.com/bmenn>`_ - CUSOLVER support for symmetric 
  eigenvalue decomposition.
- `Bruce Merry <https://github.com/bmerry>`_ - Support for CUFFT extensible plan 
  API.
- `Teodor Mihai Moldovan <https://github.com/teodor-moldovan>`_ - CUBLAS 5 
  bindings.
- `Lars Pastewka <https://github.com/pastewka>`_ - FFT tests and FFTW compatibility mode configuration.
- `Li Yong Liu <http://laoniu85.github.io>`_ - CUBLAS batch wrappers.
- `Luke Pfister <https://www.linkedin.com/pub/luke-pfister/11/70a/731>`_ - Bug 
  fixes.
- `Michael Rader <https://github.com/mrader1248>`_ - Bug fixes.
- `Nate Merrill <https://github.com/nmerrill67>`_ - PCA module.
- `Alex Rubinsteyn <https://github.com/iskandr>`_ - Support for CULA Dense Free R17.
- `Xing Shi <https://github.com/shixing>`_ - Bug fixes.
- `Steve Taylor <https://github.com/stevertaylor>`_ - Cholesky factorization/solve functions.
- `Rob Turetsky <https://www.linkedin.com/in/robturetsky>`_ - Useful feedback.
- `Thomas Unterthiner <https://github.com/untom>`_ - Additional high-level and wrapper functions.
- `Nikul H. Ukani <https://github.com/nikulukani>`_ - Additional MAGMA wrappers.
- `S. Clarkson <https://github.com/sclarkson>`_ - Bug fixes.
- `Stefan van der Walt <https://github.com/stefanv>`_ - Bug fixes.
- `Feng Wang <https://github.com/cnwangfeng>`_ - Bug reports.
- `Alexander Weyman <https://github.com/AlexanderWeyman>`_ - Simpson's Rule.
- `Evgeniy Zheltonozhskiy <https://github.com/randl>`_ - Complex Hermitian 
  support eigenvalue decomposition.
- `Wing-Kit Lee <https://github.com/wingkitlee>`_ - Fixes for MAGMA eigenvalue 
  decomp wrappers.
- `Yiyin Zhou <https://github.com/yiyin>`_ - Patches, bug reports, and function 
  wrappers 
