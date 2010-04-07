#!/usr/bin/env python

import sys
import os

from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_python_lib

NAME =               'cula_wrapper'
VERSION =            '0.01'
IS_RELEASED =        True
AUTHOR =             'Lev Givon'
AUTHOR_EMAIL =       'lev@columbia.edu'
URL =                'http://bionet.ee.columbia.edu/code/'
MAINTAINER =         'Lev Givon'
MAINTAINER_EMAIL =   'lev@columbia.edu'
DESCRIPTION =        'Python wrapper for CULA libraries'
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

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name = NAME,
          version = VERSION,
          author = AUTHOR,
          author_email = AUTHOR_EMAIL,
          url = URL,
          maintainer = MAINTAINER,
          maintainer_email = MAINTAINER_EMAIL,
          description = DESCRIPTION,
          license = LICENSE,
          classifiers = CLASSIFIERS,
          py_modules = ['cula'])
