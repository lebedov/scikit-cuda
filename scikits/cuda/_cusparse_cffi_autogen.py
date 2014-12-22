"""
Support functions for autogenerating Python interface to the cuSPARSE library.

"""

"""
Developed on linux for CUDA v6.5, but should support any version >4 where
cusparse_v2.h can be found in the CUDA_ROOT/include.

Set the environment variable CUDA_ROOT to the base of your CUDA installation

Note from the NVIDIA CUSPARSE release notes:

"The csr2csc() and bsr2bsc() routines contain a bug in the CUDA 6.0 and 6.5
releases. As a consequence, csrsv(), csrsv2(), csrsm(), bsrsv2(), bsrsm2(),
and csrgemm() may produce incorrect results when working with transpose
(CUSPARSE_OPERATION_TRANSPOSE) or conjugate-transpose
(CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE) operations. These routines work
correctly when the non-transpose (CUSPARSE_OPERATION_NON_TRANSPOSE) operation
is selected. The bug has been fixed in the CUDA 7.0 release."
"""

import os
import re
import json
import numpy as np

import cffi

# TODO: improve autodetection of cuda include folder
CUDA_ROOT = os.environ.get('CUDA_ROOT', None) or '/usr/local/cuda'
cuda_include_path = os.path.join(CUDA_ROOT, 'include')
base_dir = os.path.dirname(__file__)


def ffi_init_cusparse(cffi_cdef):
    ffi = cffi.FFI()
    ffi.cdef(cffi_cdef)

    # Get the address in a cdata pointer:
    __verify_scr = """
    #include <cusparse_v2.h>
    #include <driver_types.h>
    """
    ffi_lib = ffi.verify(__verify_scr, libraries=['cusparse'],
                           include_dirs=['/usr/local/cuda/include'],
                           library_dirs=['/usr/local/cuda/lib64/'])
    return ffi, ffi_lib


def generate_cffi_cdef(
        cuda_include_path=cuda_include_path, cffi_out_file=None):
    """ generate the CUSPARSE FFI definition

    Parameters
    ----------
    cuda_include_path : str
        CUDA include path
    cffi_out_file : str, optional
        if provided, write the definition out to a file

    Returns
    -------
    cffi_cdef : str
        function definitions for use with cffi.  e.g. input to FFI.verify()

    """
    v2_header = os.path.join(cuda_include_path, 'cusparse_v2.h')
    if not os.path.exists(v2_header):
        # on old versions there was only cusparse.h
        v2_header = os.path.join(cuda_include_path, 'cusparse.h')
        if not os.path.exists(v2_header):
            raise ValueError("cusparse header file not found in expected "
                             "location.  Try defining CUDA_ROOT")

    with open(v2_header, 'r') as f:
        cusparse_hdr = f.readlines()

    # in newer versions cusparse_v2.h just points to cusparse.h
    for line in cusparse_hdr:
        # if v2 header includes cusparse.h, read that one instead
        if line.startswith('#include "cusparse.h"'):
            v2_header = os.path.join(cuda_include_path, 'cusparse.h')
            with open(v2_header, 'r') as f:
                cusparse_hdr = f.readlines()

    # skip lines leading up to first typedef
    for idx, line in enumerate(cusparse_hdr):
        if line.startswith('typedef'):
            start_line = idx
            break

    # skip closing #if defined logic
    for idx, line in enumerate(cusparse_hdr[start_line:]):
        if line.startswith('#if defined(__cplusplus)') or \
           'Define the following symbols for the new API' in line:
            # second match is to avoid CFFI compilation errror due to the final
            # define statements in v4.1 through v5.5
            end_line = start_line + idx
            break

    # define other data types needed by FFI
    # ... will be filled in from cuComplex.h by the C compiler
    cffi_cdef = """
    typedef struct CUstream_st *cudaStream_t;

    typedef struct float2 {
        ...;
    } float2;
    typedef float2 cuFloatComplex;
    typedef float2 cuComplex;

    typedef struct double2 {
        ...;
    } double2;
    typedef double2 cuDoubleComplex;

    typedef float cufftReal;
    typedef double cufftDoubleReal;

    typedef cuComplex cufftComplex;
    typedef cuDoubleComplex cufftDoubleComplex;

    /* definitions from cusparse header below this point */
    """

    cffi_cdef += ''.join(cusparse_hdr[start_line:end_line])


    """
    don't use the _v2 versions of the function names defined in CUDA v4.1
    through v5.5
    """
    cffi_cdef = cffi_cdef.replace('_v2(', '(')

    if os.name == 'nt':  # Win
        cffi_cdef = cffi_cdef.replace('CUSPARSEAPI', '__stdcall')
    else:  # posix, etc
        cffi_cdef = cffi_cdef.replace('CUSPARSEAPI', '')

    if cffi_out_file is not None:
        # create specified output directory if it doesn't already exist
        out_dir = os.path.dirname(cffi_out_file)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(cffi_out_file, 'w') as f:
            f.write(cffi_cdef)

    return cffi_cdef

# sourcefilename = ffi.verifier.sourcefilename
# modulefilename = ffi.verifier.modulefilename


def reindent(s, numSpaces=4, lstrip=True):
    """add indentation to a multiline string.

    Parameters
    ----------
    s : str
        string to reformat
    numSpaces : str, optional
        number of spaces to indent each line by
    lstrip : bool, optional
        if True, lstrip() prior to adding numSpaces

    Returns
    -------
    s : str
        reformatted str
    """
    s = s.split('\n')
    if lstrip:
        s = [line.lstrip() for line in s]

    for idx, line in enumerate(s):
        if line.strip() == '':
            # don't indent empty lines
            s[idx] = ''
        else:
            s[idx] = (numSpaces * ' ') + line
    s = '\n'.join(s)
    return s


def _find_breakpoint(line, break_pattern=', ', nmax=80):
    """ determine where to break the line """
    locs = [m.start() for m in re.finditer(break_pattern, line)]
    if len(locs) > 0:
        break_loc = locs[np.where(
            np.asarray(locs) < (nmax - len(break_pattern)))[0][-1]]
        break_loc += len(break_pattern)
    else:
        break_loc = None
    return locs, break_loc


def _split_line(line, break_pattern=', ', nmax=80, pad_char='('):
    """ split a line (repeatedly) until length < nmax chars.

    split will occur at last occurence of break_pattern occuring before nmax
    characters

    subsequent lines will be indented until the first occurance of pad_char
    in the initial line

    Parameters
    ----------
    line : str
        line to reformat
    break_pattern : str, optional
        break line only where this pattern occurs
    nmax : int, optional
        max number of characters to allow
    pad_char : str, optional
        auto-indent subsequent lines up to the first occurance of pad_char

    Returns
    -------
    new_line : str
        reformatted line
    """
    if len(line) < nmax:
        return line.rstrip() + '\n'
    locs, break_loc = _find_breakpoint(line,
                                       break_pattern=break_pattern,
                                       nmax=nmax)
    if break_loc is None:
        return line.rstrip() + '\n'
    if pad_char is not None:
        npad = line.find(pad_char) + 1
    else:
        npad = 0

    lines = []
    lines.append(line[:break_loc].rstrip())
    line = ' ' * npad + line[break_loc:]
    while (len(line) > nmax - 1) and (break_loc is not None):
        locs, break_loc = _find_breakpoint(line,
                                           break_pattern=break_pattern,
                                           nmax=nmax)
        lines.append(line[:break_loc].rstrip())
        line = ' ' * npad + line[break_loc:]
    lines.append(line.rstrip())
    return '\n'.join(lines) + '\n'


def _build_func_sig(func_name, arg_dict, return_type):
    """ generate the python wrapper function signature line(s). """

    if 'Create' in func_name:
        # don't pass in any argument to creation functions
        return "def %s():\n" % func_name

    if ('Get' in func_name) and (return_type == 'cusparseStatus_t') and \
        len(arg_dict) == 2:
        basic_getter = True
    else:
        basic_getter = False

    sig = "def %s(" % func_name
    for k, v in arg_dict.iteritems():
        is_ptr = '*' in v
        if is_ptr and basic_getter:
            continue
        sig += k + ", "
    sig = sig[:-2] + "):\n"
    # wrap to 2nd line if too long
    return _split_line(sig, break_pattern=', ', nmax=79)


def _build_doc_str(arg_dict, func_description='', variable_descriptions={}):
    """ generate python wrapper docstring """
    docstr = '"""' + func_description + '\n'
    docstr += 'Parameters\n----------\n'
    for k, v in arg_dict.iteritems():
        docstr += k + " : " + v + "\n"
        if k in variable_descriptions:
            docstr += reindent(variable_descriptions[k],
                               numSpaces=4,
                               lstrip=True)
        else:
            print("no variable description provided for {}".format(k))
    docstr += '"""\n'
    return reindent(docstr, numSpaces=4, lstrip=False)


def _build_body(func_name, arg_dict, return_type):
    """ generate python_wrapper function body

    Note: this code is highly specific to the particulars of the cuSPARSE
    library

    """
    body = ""
    arg_list = ""

    # the following are pointers to scalar outputs
    scalar_ptr_outputs = ['nnzTotalDevHostPtr',
                          'pBufferSizeInBytes',
                          'resultDevHostPtr']

    is_creator = 'cusparseCreate' in func_name
    is_getter = 'cusparseGet' in func_name

    if return_type == 'cusparseStatus_t' and not (is_creator or is_getter):
        is_return = False
    else:
        is_return = True

    # else:
    return_str = ''
    for k, v in arg_dict.iteritems():

        """
        set some flags based on the name/type of the argument
        will use these flags to determine whether and how to call ffi.new or
        ffi.cast on each variable
        """
        is_ptr = '*' in v
        is_cusparse_type = '_t' in v
        is_cusparse_ptr = is_ptr and is_cusparse_type
        is_output_scalar = k in scalar_ptr_outputs
        if k in ['alpha', 'beta']:
            is_scalar = True
        else:
            is_scalar = False
        if is_getter:
            is_gpu_array = False
        else:
            is_gpu_array = is_ptr and (not is_cusparse_ptr) and (not is_scalar)
        if 'Complex' in v:
            is_complex = True
        else:
            is_complex = False

        # convert variable to appropriate type for the FFI
        if is_output_scalar:
            # for scalar outputs make a new pointer
            body += "%s = ffi.cast('%s', %s)\n" % (k, v, k)
        elif is_getter and is_ptr and (return_type == 'cusparseStatus_t'):
            # any pointers in cusparseGet* are new outputs to be created
            body += "%s = ffi.new('%s')\n" % (k, v)
        elif is_gpu_array:
            # pass pointer to GPU array data (use either .ptr or .gpudata)
            body += "%s = ffi.cast('%s', %s.ptr)\n" % (k, v, k)
        elif is_cusparse_ptr:
            if is_creator:
                # generate custom cusparse type
                body += "%s = ffi.new('%s')\n" % (k, v)
            else:
                # cast to the custom cusparse type
                body += "%s = ffi.cast('%s', %s)\n" % (k, v, k)
        elif is_ptr and is_scalar:
            # create new pointer, with value initialized to scalar
            if is_complex:
                # complex case is a bit tricky.  requires ffi.buffer
                body += "%sffi = ffi.new('%s')\n" % (k, v)
                if 'cusparseC' in func_name:
                    body += "ffi.buffer(%sffi)[:] = \
                        np.complex64(%s).tostring()\n" % (k, k)
                elif 'cusparseZ' in func_name:
                    body += "ffi.buffer(%sffi)[:] = \
                        np.complex128(%s).tostring()\n" % (k, k)
            else:
                body += "%s = ffi.new('%s', %s)\n" % (k, v, k)
        elif is_ptr or v == 'cudaStream_t':
            # case non-scalar pointer to appropriate type
            body += "%s = ffi.cast('%s', %s)\n" % (k, v, k)
        else:
            # don't need explicit cast for plain int, float, etc
            pass

        # build the list of arguments to pass to the API
        if is_ptr and is_scalar and is_complex:
            # take into account modified argument name for complex scalars
            arg_list += "%sffi, " % k
        else:
            arg_list += "%s, " % k

    # add the function call and optionally return the result
    last_key = k
    arg_list = arg_list[:-2]  # remove trailing ", "
    if is_getter and return_type != 'cusparseStatus_t':
        body += "return ffi_lib.%s(%s)\n" % (func_name, arg_list)
    else:
        # check cusparseStatus_t state before returning
        call_str = "status = ffi_lib.%s(%s)\n" % (func_name, arg_list)
        body += _split_line(call_str, break_pattern=', ', nmax=76)
        body += "cusparseCheckStatus(status)\n"
        if is_return:
            # len(arg_dict) == 2) is to avoid return for cusparseGetLevelInfo
            if is_creator or (is_getter and (len(arg_dict) == 2)):
                body += "return %s[0]\n" % last_key
            else:
                body += "#TODO: return the appropriate result"
    body += '\n\n'
    return reindent(body, numSpaces=4, lstrip=False)


def _func_str(func_name, arg_dict, return_type,
             variable_descriptions={}, func_description=''):
    """ build a single python wrapper """
    fstr = _build_func_sig(func_name, arg_dict, return_type)
    fstr += _build_doc_str(arg_dict, func_description=func_description,
                          variable_descriptions=variable_descriptions)
    fstr += _build_body(func_name, arg_dict, return_type)
    return fstr


def build_python_func(cdef, variable_descriptions={}, func_descriptions={}):
    """ wrap a single python function corresponding to the given cdef C
    function string.

    Parameters
    ----------
    cdef : str
        single line string containing a C function definition
    variable_descriptions : dict
        dictionary of variable descriptions for the docstring
    func_descriptions : dict
        dictionary of function descriptions for the docstring

    Returns
    -------
    str corresponding to the python_wrapper
    """
    cdef_regex = "(\w*)\s*(\w*)\s*\((.*)\).*"
    p = re.compile(cdef_regex)
    match = p.search(cdef)
    (return_type, func_name, func_args) = match.group(1, 2, 3)
    func_args = func_args.split(', ')

    from collections import OrderedDict
    arg_dict = OrderedDict()
    for arg in func_args:
        substr = arg.split()
        if len(substr) == 2:
            val = substr[0]
        else:
            val = substr[-2]
        key = substr[-1]
        # handle pointer
        if key[0] == '*':
            val += ' *'
            key = key[1:]
        # handle pointer to pointer
        if key[0] == '*':
            val += '*'
            key = key[1:]
        arg_dict[key] = val

    func_description = func_descriptions.get(func_name, '')
    return _func_str(func_name, arg_dict, return_type,
                    variable_descriptions=variable_descriptions,
                    func_description=func_description)


# with open('cusparse_variable_descriptions.json', 'w') as fid:
#    json.dump(variable_descriptions, fid, sort_keys=True, indent=4)

def get_variable_descriptions(var_def_json):
    """ load variable description dictionary from .json file"""
    with open(var_def_json, 'r') as fid:
        variable_descriptions = json.load(fid)
    for k, v in variable_descriptions.iteritems():
        variable_descriptions[k] = _split_line(v, break_pattern=' ', nmax=72,
                                               pad_char=None)
    return variable_descriptions


def get_function_descriptions(func_def_json):
    """ load function description dictionary from .json file"""
    with open(func_def_json, 'r') as fid:
        func_descriptions = json.load(fid)
    for k, v in func_descriptions.iteritems():
        func_descriptions[k] = _split_line(v, break_pattern=' ', nmax=72,
                                           pad_char=None)
    return func_descriptions


def generate_func_descriptions_json(ffi_lib, json_file):
    func_descriptions = {}
    for t in ['S', 'D', 'C', 'Z']:
        func_descriptions['cusparse' + t + 'axpyi'] = 'scalar multiply and add: y = y + alpha * x'
        func_descriptions['cusparse' + t + 'doti'] = 'dot product: result = y.T * x'
        func_descriptions['cusparse' + t + 'dotci'] = 'complex conjugate dot product: result = y.H * x'
        func_descriptions['cusparse' + t + 'gthr'] = 'gather elements of y at locations xInd into data array xVal'
        func_descriptions['cusparse' + t + 'gthrz'] = 'gather elements of y at locations xInd into data array xVal.  Also zeros the gathered elements in y'
        func_descriptions['cusparse' + t + 'roti'] = 'applies Givens rotation matrix to sparse x and dense y'
        func_descriptions['cusparse' + t + 'sctr'] = 'scatters elements of vector x into the vector y (at locations xInd)'
        func_descriptions['cusparse' + t + 'bsrmv'] = 'sparse BSR matrix vector product:  y = alpha * op(A)*x + beta * y'
        func_descriptions['cusparse' + t + 'bsrxmv'] = 'sparse BSRX matrix vector product:  y = alpha * op(A)*x + beta * y'
        func_descriptions['cusparse' + t + 'csrmv'] = 'sparse CSR matrix vector product:  y = alpha * op(A)*x + beta * y'
        func_descriptions['cusparse' + t + 'bsrsv2_bufferSize'] = 'return size of buffer used in bsrsv2'
        func_descriptions['cusparse' + t + 'bsrsv2_analysis'] = 'perform analysis phase of bsrsv2'
        func_descriptions['cusparse' + t + 'bsrsv2_solve'] = 'perform solve phase of bsrsv2'
        func_descriptions['cusparse' + t + 'csrsv2_bufferSize'] = 'return size of buffer used in csrsv2'
        func_descriptions['cusparse' + t + 'csrsv2_analysis'] = 'perform analysis phase of csrsv2'
        func_descriptions['cusparse' + t + 'csrsv2_solve'] = 'perform solve phase of csrsv2'
        func_descriptions['cusparse' + t + 'csrsv_analysis'] = 'perform analysis phase of csrsv'
        func_descriptions['cusparse' + t + 'csrsv_solve'] = 'perform solve phase of csrsv'
        func_descriptions['cusparse' + t + 'hybsv_analysis'] = 'perform analysis phase of hybsv'
        func_descriptions['cusparse' + t + 'hybsv_solve'] = 'perform solve phase of hybsv'
        func_descriptions['cusparse' + t + 'hybmv'] = 'sparse HYB matrix vector product:  y = alpha * op(A)*x + beta * y'
        func_descriptions['cusparse' + t + 'csrmm'] = 'sparse CSR matrix-matrix product:  C = alpha * op(A) * B + beta * C'
        func_descriptions['cusparse' + t + 'bsrmm'] = 'sparse BSR matrix-matrix product:  C = alpha * op(A) * B + beta * C'
        func_descriptions['cusparse' + t + 'csrmm2'] = 'sparse CSR matrix-matrix product type 2:  C = alpha * op(A) * op(B) + beta * C'
        func_descriptions['cusparse' + t + 'csrsm_analysis'] = 'perform analysis phase of csrsm'
        func_descriptions['cusparse' + t + 'csrsm_solve'] = 'perform solve phase of csrsm'
        func_descriptions['cusparse' + t + 'bsrsm2_bufferSize'] = 'return size of buffer used in bsrsm2'
        func_descriptions['cusparse' + t + 'bsrsm2_analysis'] = 'perform analysis phase of bsrsm2'
        func_descriptions['cusparse' + t + 'bsrsm2_solve'] = 'perform solve phase of bsrsm2'
        func_descriptions['cusparse' + t + 'csrgeam'] = 'sparse CSR matrix-matrix operation:  C = alpha * A + beta * B'
        func_descriptions['cusparse' + t + 'csrgemm'] = 'sparse CSR matrix-matrix operation:  C = op(A) * op(B)'
        func_descriptions['cusparse' + t + 'csric0'] = 'CSR incomplete-Cholesky factorization:  op(A) ~= R.T * R'
        func_descriptions['cusparse' + t + 'csric02_bufferSize'] = 'return csric02 (A ~= L * L.H) buffer size'
        func_descriptions['cusparse' + t + 'csric02_analysis'] = 'perform csric02 (A ~= L * L.H) analysis phase'
        func_descriptions['cusparse' + t + 'csric02'] = 'perform csric02 (A ~= L * L.H) solve phase'
        func_descriptions['cusparse' + t + 'csrilu0'] = 'CSR incomplete-LU factorization:  op(A) ~= LU'
        func_descriptions['cusparse' + t + 'csrilu02_numericBoost'] = 'use a boost value to replace a numerical value in incomplete LU factorization'
        func_descriptions['cusparse' + t + 'csrilu02_bufferSize'] = 'return csrilu02 (A ~= LU) buffer size'
        func_descriptions['cusparse' + t + 'csrilu02_analysis'] = 'perform csrilu02 (A ~= LU) analysis phase'
        func_descriptions['cusparse' + t + 'csrilu02'] = 'perform csrilu02 (A ~= LU) solve phase'
        func_descriptions['cusparse' + t + 'bsric02_bufferSize'] = 'return bsric02 (A ~= L * L.H) buffer size'
        func_descriptions['cusparse' + t + 'bsric02_analysis'] = 'perform bsric02 (A ~= L * L.H) analysis phase'
        func_descriptions['cusparse' + t + 'bsric02'] = 'perform bsric02 (A ~= L * L.H) solve phase'
        func_descriptions['cusparse' + t + 'bsrilu02_numericBoost'] = 'use a boost value to replace a numerical value in incomplete LU factorization'
        func_descriptions['cusparse' + t + 'bsrilu02_bufferSize'] = 'return bsrilu02 (A ~= LU) buffer size'
        func_descriptions['cusparse' + t + 'bsrilu02_analysis'] = 'perform bsrilu02 (A ~= LU) analysis phase'
        func_descriptions['cusparse' + t + 'bsrilu02'] = 'perform bsrilu02 (A ~= LU) solve phase'
        func_descriptions['cusparse' + t + 'gtsv'] = 'solve tridiagonal system with multiple right-hand sides: A * Y = alpha * X'
        func_descriptions['cusparse' + t + 'gtsv_nopivot'] = 'solve tridiagonal system with multiple right-hand sides: A * Y = alpha * X'
        func_descriptions['cusparse' + t + 'gtsvStridedBatch'] = 'solve multiple tridiagonal systems for i = 0, ..., batchCount: A_i * y_i = alpha * x_i'
        func_descriptions['cusparse' + t + 'bsr2csr'] = 'convert sparse matrix formats: BSR to CSR'
        func_descriptions['cusparse' + t + 'gebsr2gebsc_bufferSize'] = 'return gebsr2gebsc buffer size'
        func_descriptions['cusparse' + t + 'gebsr2gebsc'] = 'convert sparse matrix formats: GEBSR to GEBSC'
        func_descriptions['cusparse' + t + 'gebsr2gebsr_bufferSize'] = 'return gebsr2gebsr or gebsr2gebsrNnz buffer size'
        func_descriptions['cusparse' + t + 'gebsr2gebsr'] = 'convert sparse matrix formats: GEBSR to GEBSR'
        func_descriptions['cusparse' + t + 'gebsr2csr'] = 'convert sparse matrix formats: GEBSR to CSR'
        func_descriptions['cusparse' + t + 'csr2gebsr_bufferSize'] = 'return csr2gebsr buffer size'
        func_descriptions['cusparse' + t + 'csr2gebsr'] = 'convert sparse matrix formats: CSR to GEBSR'
        func_descriptions['cusparse' + t + 'coo2csr'] = 'convert sparse matrix formats: COO to CSR'
        func_descriptions['cusparse' + t + 'csc2dense'] = 'convert sparse matrix formats: CSC to dense'
        func_descriptions['cusparse' + t + 'csc2hyb'] = 'convert sparse matrix formats: CSC to HYB'
        func_descriptions['cusparse' + t + 'csr2bsr'] = 'convert sparse matrix formats: CSR to BSR'
        func_descriptions['cusparse' + t + 'csr2coo'] = 'convert sparse matrix formats: CSR to COO'
        func_descriptions['cusparse' + t + 'csr2csc'] = 'convert sparse matrix formats: CSR to CSC'
        func_descriptions['cusparse' + t + 'csr2dense'] = 'convert sparse matrix formats: CSR to dense'
        func_descriptions['cusparse' + t + 'csr2hyb'] = 'convert sparse matrix formats: CSR to HYB'
        func_descriptions['cusparse' + t + 'dense2csc'] = 'convert sparse matrix formats: dense to CSC'
        func_descriptions['cusparse' + t + 'dense2csr'] = 'convert sparse matrix formats: dense to CSR'
        func_descriptions['cusparse' + t + 'dense2hyb'] = 'convert sparse matrix formats: dense to HYB'
        func_descriptions['cusparse' + t + 'hyb2csc'] = 'convert sparse matrix formats: HYB to CSC'
        func_descriptions['cusparse' + t + 'hyb2csr'] = 'convert sparse matrix formats: HYB to CSR'
        func_descriptions['cusparse' + t + 'hyb2dense'] = 'convert sparse matrix formats: HYB to dense'
        func_descriptions['cusparse' + t + 'nnz'] = 'compute number of nonzero elements per row or column and the total number of nonzero elements'

    # operations common across all precisions
    func_descriptions['cusparseXcsrgemmNnz'] = 'determine csrRowPtrC and the total number of nonzero elements for gemm'
    func_descriptions['cusparseXcsrgeamNnz'] = 'determine csrRowPtrC and the total number of nonzero elements for geam'
    func_descriptions['cusparseXcsr2gebsrNnz'] = 'determine the number of nonzero block columns per block row for csr2gebsr'
    func_descriptions['cusparseXgebsr2gebsrNnz'] = 'determine the number of nonzero block columns per block row for gebsr2gebsr'
    func_descriptions['cusparseXcsr2bsrNnz'] = 'determine the number of nonzero block columns per block row for csr2bsr'
    func_descriptions['cusparseXbsrsv2_zeroPivot'] = 'return numerical zero location for bsrsv2'
    func_descriptions['cusparseXcsrsv2_zeroPivot'] = 'return numerical zero location for csrsv2'
    func_descriptions['cusparseXbsrsm2_zeroPivot'] = 'return numerical zero location for bsrsm2'
    func_descriptions['cusparseXbsrilu02_zeroPivot'] = 'return numerical zero location for bsrilu02'
    func_descriptions['cusparseXcsrilu02_zeroPivot'] = 'return numerical zero location for csrilu02'
    func_descriptions['cusparseXbsric02_zeroPivot'] = 'return numerical zero location for bsric02'
    func_descriptions['cusparseXcsric02_zeroPivot'] = 'return numerical zero location for csric02'
    func_descriptions['cusparseXbsrsm2_zeroPivot'] = 'return numerical zero location for bsrsm2'

    create_funcs = [cdef for cdef in ffi_lib.__dict__ if 'Create' in cdef]
    for func in create_funcs:
        tmp, obj = func.split('Create')
        if obj:
            func_descriptions[func] = "Create cuSPARSE {} structure.".format(obj)
        else:
            func_descriptions[func] = "Create cuSPARSE context."
    destroy_funcs = [cdef for cdef in ffi_lib.__dict__ if 'Destroy' in cdef]
    for func in destroy_funcs:
        tmp, obj = func.split('Destroy')
        if obj:
            func_descriptions[func] = "Destroy cuSPARSE {} structure.".format(obj)
        else:
            func_descriptions[func] = "Destroy cuSPARSE context."
    get_funcs = [cdef for cdef in ffi_lib.__dict__ if 'Get' in cdef]
    for func in get_funcs:
        tmp, obj = func.split('Get')
        func_descriptions[func] = "Get cuSPARSE {}.".format(obj)

    set_funcs = [cdef for cdef in ffi_lib.__dict__ if 'Set' in cdef]
    for func in set_funcs:
        tmp, obj = func.split('Set')
        func_descriptions[func] = "Set cuSPARSE {}.".format(obj)

    # prune any of the above that aren't in ffi_lib:
    func_descriptions = dict(
        (k, v) for k, v in func_descriptions.iteritems(
            ) if k in ffi_lib.__dict__)

    with open(json_file, 'w') as fid:
        json.dump(func_descriptions, fid, sort_keys=True, indent=4)


def generate_cusparse_python_wrappers(cffi_cdef=None, variable_defs_json='',
                                      func_defs_json='',
                                      python_wrapper_file=None):
    """ generate python wrappers for all functions within cffi_cdef.

    Parameters
    ----------
    cffi_cdef : str
        cffi definition string as generated by `generate_cffi_cdef`
    variable_defs_json : str, optional
        filename of .json file containing dictionary of variable descriptions
    func_defs_json : str, optional
        filename of .json file containing dictionary of function descriptions
    python_wrapper_file : str, optional
        file to output the generated python wrappers to

    Returns
    -------
    python_wrappers : str
        string containing all of the python wrappers

    """
    cffi_cdef_list = cffi_cdef.split('\n')

    # find lines containing a function definition
    func_def_lines = []
    for idx, line in enumerate(cffi_cdef_list):
        if line.startswith('cusparse'):
            func_def_lines.append(idx)

    # reformat each definition into a single line for easier string processing
    n_funcs = len(func_def_lines)
    cdef_list = []
    for i in range(len(func_def_lines)):
        loc1 = func_def_lines[i]
        if i < n_funcs - 1:
            loc2 = func_def_lines[i + 1]
            cdef = ' '.join([l.strip() for l in cffi_cdef_list[loc1:loc2]])
        else:
            cdef = ' '.join([l.strip() for l in cffi_cdef_list[loc1:]])
        # strip any remaining comments after the semicolon
        cdef = cdef[:cdef.find(';') + 1]
        cdef_list.append(cdef)

    # read function and variable definition strings to use when building the
    # the Python doc strings
    if variable_defs_json:
        variable_descriptions = get_function_descriptions(variable_defs_json)
    if func_defs_json:
        func_descriptions = get_function_descriptions(func_defs_json)

    # build the wrappers
    python_wrappers = ''
    for cdef in cdef_list:
        python_wrappers += build_python_func(
            cdef,
            variable_descriptions=variable_descriptions,
            func_descriptions=func_descriptions)

    if python_wrapper_file is not None:
        with open(python_wrapper_file, 'w') as f:
            f.write(python_wrappers)
    return python_wrappers
