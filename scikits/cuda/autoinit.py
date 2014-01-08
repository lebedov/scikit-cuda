#!/usr/bin/env python

"""
Autoinitialize CUDA tools.
"""

import atexit
import misc

try:
    import cula
    _has_cula = True
except (ImportError, OSError):
    _has_cula = False

misc.init()
if _has_cula:
    cula.culaInitialize()
atexit.register(misc.shutdown)
