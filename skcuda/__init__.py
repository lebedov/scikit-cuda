from __future__ import absolute_import

import os
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from .info import __doc__
from .version import __version__

# Location of headers:
install_headers = \
    __file__.replace(os.path.basename(__file__), '') + 'include'
