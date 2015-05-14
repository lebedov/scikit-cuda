import os

# This import must precede the invocation of extend_path() to work with Python
# 3:
import scikits.cuda

from pkgutil import extend_path
__path__ = extend_path(__path__, 'scikits.cuda')

from .info import __doc__
from .version import __version__

# Needed to ensure correct header location even when modules are imported as
# skcuda.something:
install_headers = \
    scikits.cuda.__file__.replace(os.path.basename(scikits.cuda.__file__), '') + 'include'
