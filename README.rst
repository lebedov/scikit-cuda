.. -*- rst -*-

CUDA SciKit
===========

Package Description
-------------------
The CUDA SciKit (toolkit for SciPy [1]_) provides Python interfaces to a
subset of the functions in the CUDA, CUDART, CUBLAS, and CUFFT
libraries distributed as part of NVIDIA's CUDA Programming Toolkit
[2]_, as well as interfaces to select functions in the free and
standard versions of the CULA Dense Toolkit [3]_. In contrast to most
existing Python wrappers for these libraries (many of which only
provide a low-level interface to the actual library functions), this
package uses PyCUDA [4]_ to provide high-level functions comparable to
those in the NumPy package [5]_.

.. image:: https://zenodo.org/badge/6233/lebedov/scikits.cuda.svg
    :target: http://dx.doi.org/10.5281/zenodo.16269
    :alt: 0.5.0b1
.. image:: https://pypip.in/version/scikits.cuda/badge.png
    :target: https://pypi.python.org/pypi/scikits.cuda
    :alt: Latest Version
.. image:: https://pypip.in/d/scikits.cuda/badge.png
    :target: https://pypi.python.org/pypi/scikits.cuda
    :alt: Downloads
.. image:: http://prime4commit.com/projects/102.svg
    :target: http://prime4commit.com/projects/102
    :alt: Support the project

Documentation
-------------
Package documentation is available at
`<http://scikit-cuda.readthedocs.org/>`_.

Development
-----------
The latest source code can be obtained from
`<http://github.com/lebedov/scikits.cuda>`_.

Authors & Acknowledgments
-------------------------
See the included AUTHORS file for more information.

License
-------
This software is licensed under the 
`BSD License <http://www.opensource.org/licenses/bsd-license.php>`_.
See the included LICENSE file for more information.

.. [1] http://www.scipy.org/
.. [2] http://www.nvidia.com/cuda/
.. [3] http://www.culatools.com/dense/
.. [4] http://mathema.tician.de/software/pycuda/
.. [5] http://numpy.scipy.org/
