from __future__ import absolute_import

import os
import warnings
warnings.warn('The scikits.cuda namespace package is deprecated and will be '
              'removed in the future; please import the skcuda package '
              'instead.', DeprecationWarning, stacklevel=2)

# This import must precede the invocation of extend_path() to work with Python
# 3:
import skcuda

from pkgutil import extend_path
__path__ = extend_path(__path__, 'skcuda')

from .info import __doc__
from .version import __version__

# Needed to ensure correct header location even when modules are import as
# scikits.cuda.something:
install_headers = skcuda.__file__.replace(os.path.basename(skcuda.__file__), '') + 'include'
