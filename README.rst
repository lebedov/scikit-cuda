.. -*- rst -*-

..  image:: https://raw.githubusercontent.com/lebedov/scikit-cuda/master/docs/source/_static/logo.png
   :alt: scikit-cuda

Package Description
-------------------
scikit-cuda provides Python interfaces to many of the functions in the CUDA
device/runtime, CUBLAS, CUFFT, and CUSOLVER libraries distributed as part of
NVIDIA's `CUDA Programming Toolkit <http://www.nvidia.com/cuda/>`_, as well as
interfaces to select functions in the `CULA Dense Toolkit <http://www.culatools.com/dense>`_.
Both low-level wrapper functions similar to their C counterparts and high-level
functions comparable to those in `NumPy and Scipy <http://www.scipy.org>`_ are provided.

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.3229433.svg
    :target: http://dx.doi.org/10.5281/zenodo.3229433
    :alt: 0.5.3
.. image:: https://img.shields.io/pypi/v/scikit-cuda.svg
    :target: https://pypi.python.org/pypi/scikit-cuda
    :alt: Latest Version
.. image:: https://img.shields.io/pypi/dm/scikit-cuda.svg
    :target: https://pypi.python.org/pypi/scikit-cuda
    :alt: Downloads
.. image:: http://prime4commit.com/projects/102.svg
    :target: http://prime4commit.com/projects/102
    :alt: Support the project
.. image:: https://www.openhub.net/p/scikit-cuda/widgets/project_thin_badge?format=gif
    :target: https://www.openhub.net/p/scikit-cuda?ref=Thin+badge
    :alt: Open Hub

Documentation
-------------
Package documentation is available at
`<http://scikit-cuda.readthedocs.org/>`_.  Many of the high-level
functions have examples in their docstrings. More illustrations of how
to use both the wrappers and high-level functions can be found in the
``demos/`` and ``tests/`` subdirectories.

Development
-----------
The latest source code can be obtained from
`<https://github.com/lebedov/scikit-cuda>`_.

When submitting bug reports or questions via the `issue tracker
<https://github.com/lebedov/scikit-cuda/issues>`_, please include the following
information:

- Python version.
- OS platform.
- CUDA and PyCUDA version.
- Version or git revision of scikit-cuda.

Citing
------
If you use scikit-cuda in a scholarly publication, please cite it as follows: ::

    @misc{givon_scikit-cuda_2019,
              author = {Lev E. Givon and
                        Thomas Unterthiner and
                        N. Benjamin Erichson and
                        David Wei Chiang and
                        Eric Larson and
                        Luke Pfister and
                        Sander Dieleman and
                        Gregory R. Lee and
                        Stefan van der Walt and
                        Bryant Menn and
                        Teodor Mihai Moldovan and
                        Fr\'{e}d\'{e}ric Bastien and
                        Xing Shi and
                        Jan Schl\"{u}ter and
                        Brian Thomas and
                        Chris Capdevila and
                        Alex Rubinsteyn and
                        Michael M. Forbes and
                        Jacob Frelinger and
                        Tim Klein and
                        Bruce Merry and
                        Nate Merill and
                        Lars Pastewka and
                        Li Yong Liu and
                        S. Clarkson and
                        Michael Rader and
                        Steve Taylor and
                        Arnaud Bergeron and
                        Nikul H. Ukani and
                        Feng Wang and
                        Wing-Kit Lee and
                        Yiyin Zhou},
        title        = {scikit-cuda 0.5.3: a {Python} interface to {GPU}-powered libraries},
        month        = May,
        year         = 2019,
        doi          = {10.5281/zenodo.3229433},
        url          = {http://dx.doi.org/10.5281/zenodo.3229433},
        note         = {\url{http://dx.doi.org/10.5281/zenodo.3229433}}
    }

Authors & Acknowledgments
-------------------------
See the included `AUTHORS
<https://github.com/lebedov/scikit-cuda/blob/master/docs/source/authors.rst>`_
file for more information.

Note Regarding CULA Availability
--------------------------------
As of 2021, the CULA toolkit by `EM Photonics <https://emphotonics.com/>`_ no longer appears to be available.

Related
-------
Python wrappers for `cuDNN <https://developer.nvidia.com/cudnn>`_ by Hannes
Bretschneider are available `here
<https://github.com/hannes-brt/cudnn-python-wrappers>`_.

`ArrayFire <https://github.com/arrayfire/arrayfire>`_ is a free library containing many GPU-based routines with an `officially supported Python interface <https://github.com/arrayfire/arrayfire-python>`_.

License
-------
This software is licensed under the `BSD License
<http://www.opensource.org/licenses/bsd-license.php>`_.  See the included
`LICENSE
<https://github.com/lebedov/scikit-cuda/blob/master/docs/source/license.rst>`_
file for more information.
