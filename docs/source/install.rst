.. -*- rst -*-

Installation
============

Quick Installation
------------------
If you have `pip <http://pypi.python.org/pypi/pip>`_ installed, you should be
able to install the latest stable release of ``scikit-cuda`` by running the
following::

   pip install scikit-cuda

All dependencies should be automatically downloaded and installed if they are
not already on your system.

Obtaining the Latest Software
-----------------------------
The latest stable and development versions of ``scikit-cuda`` can be downloaded 
from `GitHub <https://github.com/lebedov/scikit-cuda>`_

Online documentation is available at `<https://scikit-cuda.readthedocs.org>`_

Installation Dependencies
-------------------------
``scikit-cuda`` requires that the following software packages be
installed:

* `Python <http://www.python.org>`_ 2.7 or 3.4.
* `Setuptools <http://pythonhosted.org/setuptools>`_ 0.6c10 or later.
* `Mako <http://www.makotemplates.org/>`_ 1.0.1 or later.
* `NumPy <http://www.numpy.org>`_ 1.2.0 or later.
* `PyCUDA <http://mathema.tician.de/software/pycuda>`_ 2016.1 or later (some
  parts of ``scikit-cuda`` might not work properly with earlier versions).
* `NVIDIA CUDA Toolkit <http://www.nvidia.com/object/cuda_home_new.html>`_ 5.0 
  or later.

Note that both Python and the CUDA Toolkit must be built for the same 
architecture, i.e., Python compiled for a 32-bit architecture will not find the 
libraries provided by a 64-bit CUDA installation. CUDA versions from 7.0 onwards 
are 64-bit.

To run the unit tests, the following packages are also required:

* `nose <http://code.google.com/p/python-nose/>`_ 0.11 or later.
* `SciPy <http://www.scipy.org>`_ 0.14.0 or later.

Some of the linear algebra functionality relies on the CULA toolkit;
as of 2017, it is available to premium tier users of E.M. Photonics' HPC site
`Celerity Tools <http://www.celeritytools.com>`_:

* `CULA <http://www.culatools.com/dense/>`_ R16a or later.

To build the documentation, the following packages are also required:

* `Docutils <http://docutils.sourceforge.net>`_ 0.5 or later.
* `Jinja2 <http://jinja.pocoo.org>`_ 2.2 or later.
* `Pygments <http://pygments.org>`_ 0.8 or later.
* `Sphinx <http://sphinx.pocoo.org>`_ 1.0.1 or later.
* `Sphinx ReadTheDocs Theme
  <https://github.com/snide/sphinx_rtd_theme>`_ 0.1.6 or later.

Platform Support
----------------
The software has been developed and tested on Linux; it should also work on
other Unix-like platforms supported by the above packages. Parts of the package
may work on Windows as well, but remain untested.

Building and Installation
-------------------------
``scikit-cuda`` searches for CUDA libraries in the system library
search path when imported. You may have to modify this path (e.g., by adding the
path to the CUDA libraries to ``/etc/ld.so.conf`` and running ``ldconfig`` as 
root or to the
``LD_LIBRARY_PATH`` environmental variable on Linux, or by adding the CUDA 
library path to the ``DYLD_LIBRARY_PATH`` on MacOSX) if the libraries are
not being found.

To build and install the toolbox, download and unpack the source 
release and run::

   python setup.py install

from within the main directory in the release. To rebuild the
documentation, run::

   python setup.py build_sphinx

Running the Unit Tests
----------------------
To run all of the package unit tests, download and unpack the package source
tarball and run::

   python setup.py test

from within the main directory in the archive. Tests for individual
modules (found in the ``tests/`` subdirectory) can also be run
directly.

Getting Started
---------------
The functions provided by ``scikit-cuda`` are grouped into several submodules in
the ``skcuda`` namespace package. Sample code demonstrating how to use
different parts of the toolbox is located in the ``demos/`` subdirectory of the
source release. Many of the high-level functions also contain doctests that
describe their usage.
