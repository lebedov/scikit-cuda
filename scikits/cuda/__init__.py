from __future__ import absolute_import

import warnings
warnings.warn('The scikits.cuda namespace package is deprecated and will be '
              'removed in the future; please import the skcuda package '
              'instead.', DeprecationWarning, stacklevel=2)

from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from .info import __doc__
from .version import __version__

# Installation location of C headers:
import os
install_headers = __file__.replace(os.path.basename(__file__), '') + 'include'
