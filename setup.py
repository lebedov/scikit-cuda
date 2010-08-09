#!/usr/bin/env python

import sys
import os
from glob import glob
from distutils.command.build_py import build_py
from distutils.core import setup

NAME =               'cuda_utils'
VERSION =            '0.01'
AUTHOR =             'Lev Givon'
AUTHOR_EMAIL =       'lev@columbia.edu'
URL =                'http://bionet.ee.columbia.edu/code/'
MAINTAINER =         'Lev Givon'
MAINTAINER_EMAIL =   'lev@columbia.edu'
DESCRIPTION =        'Python utilities for CUDA'
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
NAMESPACE_PACKAGE = 'cuda_utils'
PACKAGES =           [NAMESPACE_PACKAGE]

# Overwrite the copy of cuda_utils/__info__.py that will be installed
# with the actual header installation path. This is necessary so that
# PyCUDA can find the headers when executing the kernels in this
# package that use it:
class custom_build_py(build_py):
    def run(self):
        build_py.run(self)
        package_dir = self.get_package_dir(NAMESPACE_PACKAGE)
        inst_obj = self.distribution.command_obj['install']
        filename = os.path.join(self.build_lib, package_dir, '__info__.py')
        f = open(filename, 'w')
        f.write('# Installation location of C headers:\n')
        f.write('install_headers = \"%s\"\n' % inst_obj.install_headers)
        f.close()
        
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
          packages = PACKAGES,
          headers = glob('cuda_utils/*.h'),
          cmdclass={"build_py": custom_build_py})

