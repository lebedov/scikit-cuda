#!/usr/bin/env python

import sys
import os
from glob import glob
from distutils.command.install import INSTALL_SCHEMES
from distutils.command.install_headers import install_headers
from setuptools import find_packages
from setuptools import setup
from scikits.cuda.version import __version__

NAME =               'scikits.cuda'
VERSION =            __version__
AUTHOR =             'Lev Givon'
AUTHOR_EMAIL =       'lev@columbia.edu'
URL =                'http://github.com/lebedov/scikits.cuda/'
DESCRIPTION =        'Python utilities for CUDA'
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

# Install the C headers in the include/ subdirectory of whatever
# directory in which the scikits.cuda python modules are installed
# rather than wherever they would be installed by default
class custom_install_headers(install_headers):
    def run(self):
        inst_obj = self.distribution.command_obj['install']
        self.install_dir = inst_obj.install_lib + 'scikits/cuda/include'
        install_headers.run(self)

if __name__ == "__main__":
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')
    
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
        headers = glob('scikits/cuda/include/*.h'),
        data_files = [('scikits', ['scikits/__init__.py'])],
        install_requires = ['numpy',
                            'pycuda >= 0.94.2'],
        cmdclass = {'install_headers': custom_install_headers})
