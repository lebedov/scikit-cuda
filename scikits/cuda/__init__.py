from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from info import __doc__
from version import __version__

# Installation location of C headers:
import os
install_headers = __file__.replace(os.path.basename(__file__), '') + 'include'


