# WARNING: This file is overwritten upon package installation! The
# contents below are solely for development purposes to allow one to
# run doctests and demos from within the package directory.

# Installation location of C headers:
import os
install_headers = __file__.replace(os.path.basename(__file__), '')
