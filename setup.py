#!/usr/bin/env python

import os
import re
import sys

from distutils.command.install_headers import install_headers
from setuptools import find_packages
from setuptools import setup

NAME =               'scikit-cuda'
VERSION =            '0.5.2'
AUTHOR =             'Lev Givon'
AUTHOR_EMAIL =       'lev@columbia.edu'
URL =                'https://github.com/lebedov/scikit-cuda/'
DESCRIPTION =        'Python interface to GPU-powered libraries'
with open('README.rst', 'r') as f:
    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION = re.search('.*(^Package Description.*)', LONG_DESCRIPTION, re.MULTILINE|re.DOTALL).group(1)
DOWNLOAD_URL =       URL
LICENSE =            'BSD'
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development']
NAMESPACE_PACKAGES = ['scikits']
PACKAGES =           find_packages()

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:
    install_requires = ['mako >= 1.0.1',
                        'numpy >= 1.2.0',
                        'pycuda >= 2014.1']
    tests_require = ['nose >= 0.11',
                    'scipy >= 0.14.0'],
    extras_require = dict(scipy = ['scipy >= 0.14.0'],
                          sphinx = ['sphinx >= 1.3'],
                          sphinx_rtd_theme = ['sphinx_rtd_theme >= 0.1.6'])
else:
    install_requires = []
    tests_require = []
    extras_require = {}

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

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
        include_package_data = True,
        install_requires = install_requires,
        tests_require = tests_require,
        extras_require = extras_require,
        test_suite='nose.collector')
