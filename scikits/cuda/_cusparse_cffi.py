"""
Autogenerate Python interface to cuSPARSE functions.

"""
from __future__ import absolute_import, print_function

import os
import re
import numpy as np

from os.path import join as pjoin

from ._cusparse_cffi_autogen import (generate_cffi_cdef,
                                     ffi_init_cusparse,
                                     generate_func_descriptions_json,
                                     generate_cusparse_python_wrappers)

base_dir = os.path.dirname(__file__)
cffi_file = pjoin(base_dir, '_cusparse.cffi')
python_wrapper_file = pjoin(base_dir, '_cusparse_python.py')
variable_defs_json = pjoin(base_dir, 'cusparse_variable_descriptions.json')
func_defs_json = pjoin(base_dir, 'cusparse_func_descriptions.json')

if not os.path.exists(cffi_file):
    print("first import of cusparse:  cffi interface file being created."
          "This may take several seconds")
    cffi_cdef = generate_cffi_cdef(cffi_out_file=cffi_file)
else:
    with open(cffi_file, 'r') as f:
        cffi_cdef = f.read()

ffi, ffi_lib = ffi_init_cusparse(cffi_cdef)

if not os.path.exists(python_wrapper_file):
    print("first import of cusparse:  python wrappers being created.")
    if not os.path.exists(func_defs_json):
        generate_func_descriptions_json(ffi_lib, json_file=func_defs_json)
    generate_cusparse_python_wrappers(cffi_cdef,
                                      variable_defs_json=variable_defs_json,
                                      func_defs_json=func_defs_json,
                                      python_wrapper_file=python_wrapper_file)


class CUSPARSE_ERROR(Exception):
    """CUSPARSE error"""
    pass

# Use CUSPARSE_STATUS* definitions to dynamically create corresponding
# exception classes and populate dictionary used to raise appropriate
# exception in response to the corresponding CUSPARSE error code:
CUSPARSE_STATUS_SUCCESS = ffi_lib.CUSPARSE_STATUS_SUCCESS
CUSPARSE_EXCEPTIONS = {-1: CUSPARSE_ERROR}
for k, v in ffi_lib.__dict__.items():
    # Skip CUSPARSE_STATUS_SUCCESS:
    if re.match('CUSPARSE_STATUS.*', k) and v != CUSPARSE_STATUS_SUCCESS:
        CUSPARSE_EXCEPTIONS[v] = vars()[k] = type(k, (CUSPARSE_ERROR,), {})


# Import various other enum values into module namespace:
regex = 'CUSPARSE_(?!STATUS).*'
for k, v in ffi_lib.__dict__.items():
    if re.match(regex, k):
        # print("k={}, v={}".format(k,v))
        vars()[k] = v


def cusparseCheckStatus(status):
    """
    Raise CUSPARSE exception

    Raise an exception corresponding to the specified CUSPARSE error
    code.

    Parameters
    ----------
    status : int
        CUSPARSE error code.

    See Also
    --------
    CUSPARSE_EXCEPTIONS
    """

    if status != 0:
        try:
            raise CUSPARSE_EXCEPTIONS[status]
        except KeyError:
            raise CUSPARSE_ERROR

# execute the python wrapper code
with open(python_wrapper_file) as f:
    code = compile(f.read(), python_wrapper_file, 'exec')
    exec(code)


__all__ = [k for k, v in ffi_lib.__dict__.items()]
__all__.append('CUSPARSE_ERROR')
__all__.append('CUSPARSE_EXCEPTIONS')
__all__.append('cusparseCheckStatus')
__all__.append('ffi')
__all__.append('ffi_lib')
