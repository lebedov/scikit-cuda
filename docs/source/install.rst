.. -*- rst -*-

Installation Instructions
=========================

Quick Installation
------------------
If you have `pip <http://pypi.python.org/pypi/pip>`_ installed, you should be
able to install the latest stable release of ``scikits.cuda`` by running the
following::

   pip install scikits.cuda

All dependencies should be automatically downloaded and installed if they are
not already on your system.

Obtaining the Latest Software
-----------------------------

The latest stable and development versions of ``scikits.cuda`` can be downloaded from 
`GitHub <http://github.com/lebedov/scikits.cuda>`_

Online documentation for ``scikits.cuda`` is available at 
`<http://lebedov.github.com/scikits.cuda/>`_

Installation Dependencies
-------------------------

``scikits.cuda`` requires that the following software packages be
installed:

* `Python <http://www.python.org>`_ 2.5 or later.
* `setuptools <http://peak.telecommunity.com/DevCenter/setuptools>`_ 0.6c10 or later.
* `NumPy <http://numpy.scipy.org>`_ 1.2.0 or later.
* `PyCUDA <http://mathema.tician.de/software/pycuda>`_ 0.94.2 or later (some
  parts of ``scikits.cuda`` might not work properly with earlier versions).
* `NIVIDIA CUDA Toolkit <http://www.nvidia.com/object/cuda_home_new.html>`_ 3.0 or later.

To run the unit tests, the following packages are also required:

* `nose <http://code.google.com/p/python-nose/>`_ 0.11 or later.
* `SciPy <http://www.scipy.org>`_ 0.8.0 or later.

Some of the linear algebra functionality relies on the CULA toolkit; the single
precision release of the toolkit is free of charge, but requires registration. 
Depending on the version of CULA installed, some functions may not be available:

* `CULA <http://www.culatools.com/get-cula/>`_ 2.0 or later.

To build the documentation, the following packages are also required:

* `Docutils <http://docutils.sourceforge.net>`_ 0.5 or later.
* `Jinja2 <http://jinja.pocoo.org>`_ 2.2 or later.
* `Pygments <http://pygments.org>`_ 0.8 or later.
* `Sphinx <http://sphinx.pocoo.org>`_ 1.0.1 or later.

The software has been tested on Linux; it should also work on other
platforms supported by the above packages.

Building and Installation
-------------------------

``scikits.cuda`` searches for CUDA libraries in the system library
search path when imported. You may have to modify this path (e.g., by adding the
path to the CUDA libraries to /etc/ld.so.conf or to the
LD_LIBRARY_PATH environmental variable on Linux) if the libraries are
not being found.

To build and install the toolbox, download and unpack the source 
release and run::

   python setup.py install

from within the main directory in the release. To rebuild the
documentation, run::

   python setup.py build_docs

Running the Unit Tests
----------------------
To run all of the package unit tests, download and unpack the package source
tarball and run::

   nosetests

from within the main directory in the archive. Tests for individual
modules (found in the ``tests/`` subdirectory) can also be run
directly.

Getting Started
---------------
Sample code demonstrating how to use different parts of the toolbox is
located in the ``demos/`` subdirectory of the source release. Most of 
the high-level functions also contain doctests that describe their usage.

