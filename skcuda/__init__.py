from __future__ import absolute_import

try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)

from .info import __doc__
from .version import __version__

# Location of headers:
import os
install_headers = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'include')
