from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from info import __doc__
from version import __version__

# Make the path of the CUDA header installation directory available:
from __info__ import install_headers

