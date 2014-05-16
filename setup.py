#!/usr/bin/env python

import sys, os
from glob import glob

# Install setuptools if it isn't available:
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from distutils.command.install import INSTALL_SCHEMES
from distutils.command.install_headers import install_headers
from setuptools import find_packages
from setuptools import setup

NAME =               'scikits.cuda'
VERSION =            '0.5.0'
AUTHOR =             'Lev Givon'
AUTHOR_EMAIL =       'lev@columbia.edu'
URL =                'http://github.com/lebedov/scikits.cuda/'
DESCRIPTION =        'Python interface to GPU-powered libraries'
LONG_DESCRIPTION =   DESCRIPTION
DOWNLOAD_URL =       URL
LICENSE =            'BSD'
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development']
NAMESPACE_PACKAGES = ['scikits']
PACKAGES =           find_packages()

if __name__ == "__main__":
    if os.path.exists('MANIFEST'): 
        os.remove('MANIFEST')

    # This enables the installation of scikits/__init__.py as a data
    # file:
    for scheme in INSTALL_SCHEMES.values():
        scheme['data'] = scheme['purelib']

    setup(
        name = NAME,
        version = VERSION,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        license = LICENSE,
        classifiers = CLASSIFIERS,
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        url = URL,
        namespace_packages = NAMESPACE_PACKAGES,
        packages = PACKAGES,

        # Force installation of __init__.py in namespace package:
        data_files = [('scikits', ['scikits/__init__.py'])],
        include_package_data = True,
        install_requires = ['numpy',
                            'pycuda >= 0.94.2'],
        extras_require = dict(
            scipy = ['scipy >= 0.8.0']
        ))
