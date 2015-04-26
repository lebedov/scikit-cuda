import os
from pkgutil import extend_path
__path__ = extend_path(__path__, 'scikits.cuda')

# Needed to ensure correct header location even when modules are imported as
# skcuda.something:
import scikits.cuda
install_headers = \
    scikits.cuda.__file__.replace(os.path.basename(scikits.cuda.__file__), '') + 'include'
