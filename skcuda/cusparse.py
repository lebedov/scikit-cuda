#!/usr/bin/env python

"""
Python interface to CUSPARSE functions.

Note: this module does not explicitly depend on PyCUDA.
"""

from __future__ import absolute_import

import ctypes.util
import platform
from string import Template
import sys
import warnings

import numpy as np

from . import cuda

# Load library:
_version_list = [10.1, 10.0, 9.2, 9.1, 9.0, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.0]
_win32_version_list = [10, 10, 100, 92, 91, 90, 80, 75, 70, 65, 60, 55, 50, 40]
if 'linux' in sys.platform:
    _libcusparse_libname_list = ['libcusparse.so'] + \
                                ['libcusparse.so.%s' % v for v in _linux_version_list]
elif sys.platform == 'darwin':
    _libcusparse_libname_list = ['libcusparse.dylib']
elif sys.platform == 'win32':
    if platform.machine().endswith('64'):
        _libcusparse_libname_list = ['cusparse.dll'] + \
            ['cusparse64_%s.dll' % v for v in _win32_version_list]
    else:
        _libcusparse_libname_list = ['cusparse.dll'] + \
            ['cusparse32_%s.dll' % v for v in _win32_version_list]
else:
    raise RuntimeError('unsupported platform')

# Print understandable error message when library cannot be found:
_libcusparse = None
for _libcusparse_libname in _libcusparse_libname_list:
    try:
        if sys.platform == 'win32':
            _libcusparse = ctypes.windll.LoadLibrary(_libcusparse_libname)
        else:
            _libcusparse = ctypes.cdll.LoadLibrary(_libcusparse_libname)
    except OSError:
        pass
    else:
        break
if _libcusparse == None:
    OSError('CUDA sparse library not found')

class cusparseError(Exception):
    """CUSPARSE error"""
    pass

class cusparseStatusNotInitialized(cusparseError):
    """CUSPARSE library not initialized"""
    pass

class cusparseStatusAllocFailed(cusparseError):
    """CUSPARSE resource allocation failed"""
    pass

class cusparseStatusInvalidValue(cusparseError):
    """Unsupported value passed to the function"""
    pass

class cusparseStatusArchMismatch(cusparseError):
    """Function requires a feature absent from the device architecture"""
    pass

class cusparseStatusMappingError(cusparseError):
    """An access to GPU memory space failed"""
    pass

class cusparseStatusExecutionFailed(cusparseError):
    """GPU program failed to execute"""
    pass

class cusparseStatusInternalError(cusparseError):
    """An internal CUSPARSE operation failed"""
    pass

class cusparseStatusMatrixTypeNotSupported(cusparseError):
    """The matrix type is not supported by this function"""
    pass

# TODO: Check if this is complete list of exceptions, and that numbers are correct.
cusparseExceptions = {
    1: cusparseStatusNotInitialized,
    2: cusparseStatusAllocFailed,
    3: cusparseStatusInvalidValue,
    4: cusparseStatusArchMismatch,
    5: cusparseStatusMappingError,
    6: cusparseStatusExecutionFailed,
    7: cusparseStatusInternalError,
    8: cusparseStatusMatrixTypeNotSupported,
    }

# Matrix types:
CUSPARSE_MATRIX_TYPE_GENERAL = 0
CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1
CUSPARSE_MATRIX_TYPE_HERMITIAN = 2
CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3

CUSPARSE_FILL_MODE_LOWER = 0
CUSPARSE_FILL_MODE_UPPER = 1

# Whether or not a matrix' diagonal entries are unity:
CUSPARSE_DIAG_TYPE_NON_UNIT = 0
CUSPARSE_DIAG_TYPE_UNIT = 1

# Matrix index bases:
CUSPARSE_INDEX_BASE_ZERO = 0
CUSPARSE_INDEX_BASE_ONE = 1

# Operation types:
CUSPARSE_OPERATION_NON_TRANSPOSE = 0
CUSPARSE_OPERATION_TRANSPOSE = 1
CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2

# Whether or not to parse elements of a dense matrix row or column-wise.
CUSPARSE_DIRECTION_ROW = 0
CUSPARSE_DIRECTION_COLUMN = 1

# Helper functions:
class cusparseMatDescr(ctypes.Structure):
    _fields_ = [
        ('MatrixType', ctypes.c_int),
        ('FillMode', ctypes.c_int),
        ('DiagType', ctypes.c_int),
        ('IndexBase', ctypes.c_int)
        ]

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
    cusparseExceptions

    """

    if status != 0:
        try:
            raise cusparseExceptions[status]
        except KeyError:
            raise cusparseError

_libcusparse.cusparseCreate.restype = int
_libcusparse.cusparseCreate.argtypes = [ctypes.c_void_p]
def cusparseCreate():
    """
    Initialize CUSPARSE.

    Initializes CUSPARSE and creates a handle to a structure holding
    the CUSPARSE library context.

    Returns
    -------
    handle : int
        CUSPARSE library context.

    """

    handle = ctypes.c_void_p()
    status = _libcusparse.cusparseCreate(ctypes.byref(handle))
    cusparseCheckStatus(status)
    return handle.value

_libcusparse.cusparseDestroy.restype = int
_libcusparse.cusparseDestroy.argtypes = [ctypes.c_void_p]
def cusparseDestroy(handle):
    """
    Release CUSPARSE resources.

    Releases hardware resources used by CUSPARSE

    Parameters
    ----------
    handle : int
        CUSPARSE library context.

    """

    status = _libcusparse.cusparseDestroy(handle)
    cusparseCheckStatus(status)

_libcusparse.cusparseGetVersion.restype = int
_libcusparse.cusparseGetVersion.argtypes = [ctypes.c_int,
                                            ctypes.c_void_p]
def cusparseGetVersion(handle):
    """
    Return CUSPARSE library version.

    Returns the version number of the CUSPARSE library.

    Parameters
    ----------
    handle : int
        CUSPARSE library context.

    Returns
    -------
    version : int
        CUSPARSE library version number.

    """

    version = ctypes.c_int()
    status = _libcusparse.cusparseGetVersion(handle,
                                             ctypes.byref(version))
    cusparseCheckStatus(status)
    return version.value

_libcusparse.cusparseSetStream.restype = int
_libcusparse.cusparseSetStream.argtypes = [ctypes.c_int,
                                                 ctypes.c_int]
def cusparseSetStream(handle, id):
    """
    Sets the CUSPARSE stream in which kernels will run.

    Parameters
    ----------
    handle : int
        CUSPARSE library context.
    id : int
        Stream ID.

    """

    status = _libcusparse.cusparseSetStream(handle, id)
    cusparseCheckStatus(status)

_libcusparse.cusparseCreateMatDescr.restype = int
_libcusparse.cusparseCreateMatDescr.argtypes = [cusparseMatDescr]
def cusparseCreateMatDescr():
    """
    Initialize a sparse matrix descriptor.

    Initializes the `MatrixType` and `IndexBase` fields of the matrix
    descriptor to the default values `CUSPARSE_MATRIX_TYPE_GENERAL`
    and `CUSPARSE_INDEX_BASE_ZERO`.

    Returns
    -------
    desc : cusparseMatDescr
        Matrix descriptor.

    """

    desc = cusparseMatrixDesc()
    status = _libcusparse.cusparseCreateMatDescr(ctypes.byref(desc))
    cusparseCheckStatus(status)
    return desc

_libcusparse.cusparseDestroyMatDescr.restype = int
_libcusparse.cusparseDestroyMatDescr.argtypes = [ctypes.c_int]
def cusparseDestroyMatDescr(desc):
    """
    Releases the memory allocated for the matrix descriptor.

    Parameters
    ----------
    desc : cusparseMatDescr
        Matrix descriptor.

    """

    status = _libcusparse.cusparseDestroyMatDescr(desc)
    cusparseCheckStatus(status)

_libcusparse.cusparseSetMatType.restype = int
_libcusparse.cusparseSetMatType.argtypes = [cusparseMatDescr,
                                            ctypes.c_int]
def cusparseSetMatType(desc, type):
    """
    Sets the matrix type of the specified matrix.

    Parameters
    ----------
    desc : cusparseMatDescr
        Matrix descriptor.
    type : int
        Matrix type.

    """

    status = _libcusparse.cusparseSetMatType(desc, type)
    cusparseCheckStatus(status)

_libcusparse.cusparseGetMatType.restype = int
_libcusparse.cusparseGetMatType.argtypes = [cusparseMatDescr]    
def cusparseGetMatType(desc):
    """
    Gets the matrix type of the specified matrix.

    Parameters
    ----------
    desc : cusparseMatDescr
        Matrix descriptor.

    Returns
    -------
    type : int
        Matrix type.

    """

    return _libcusparse.cusparseGetMatType(desc)

# Format conversion functions:
_libcusparse.cusparseSnnz.restype = int
_libcusparse.cusparseSnnz.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      cusparseMatDescr,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p]
def cusparseSnnz(handle, dirA, m, n, descrA, A, lda, 
                 nnzPerRowColumn, nnzTotalDevHostPtr):
    """
    Compute number of non-zero elements per row, column, or dense matrix.

    Parameters
    ----------
    handle : int
        CUSPARSE library context.
    dirA : int
        Data direction of elements.
    m : int
        Rows in A.
    n : int
        Columns in A.
    descrA : cusparseMatDescr
        Matrix descriptor.
    A : pycuda.gpuarray.GPUArray
        Dense matrix of dimensions (lda, n).
    lda : int
        Leading dimension of A.
    
    Returns
    -------
    nnzPerRowColumn : pycuda.gpuarray.GPUArray
        Array of length m or n containing the number of 
        non-zero elements per row or column, respectively.
    nnzTotalDevHostPtr : pycuda.gpuarray.GPUArray
        Total number of non-zero elements in device or host memory.

    """

    # Unfinished:
    nnzPerRowColumn = gpuarray.empty()
    nnzTotalDevHostPtr = gpuarray.empty()

    status = _libcusparse.cusparseSnnz(handle, dirA, m, n, 
                                       descrA, int(A), lda,
                                       int(nnzPerRowColumn), int(nnzTotalDevHostPtr))
    cusparseCheckStatus(status)
    return nnzPerVector, nnzHost

_libcusparse.cusparseSdense2csr.restype = int
_libcusparse.cusparseSdense2csr.argtypes = [ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            cusparseMatDescr,
                                            ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p]
def cusparseSdense2csr(handle, m, n, descrA, A, lda, 
                       nnzPerRow, csrValA, csrRowPtrA, csrColIndA):
    # Unfinished
    pass

_libcusparse.cusparseSgtsv2StridedBatch_bufferSizeExt.restype = int
_libcusparse.cusparseSgtsv2StridedBatch_bufferSizeExt.argtypes =\
    [ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p
    ]
def cusparseSgtsv2StridedBatch_bufferSizeExt(handle, m, dl, d, du, x, batchCount, batchStride):
    """
    Calculate size of work buffer used by cusparseSgtsv2StridedBatch.

    Parameters
    ----------
    handle : ctypes.c_void_p
        cuSPARSE context
    m : int
        Size of the linear system (must be >= 3)
    dl : ctypes.c_void_p
        Pointer to ${precision} ${real} dense array containing the lower
        diagonal of the tri-diagonal linear system. The lower diagonal dl(i)
        that corresponds to the ith linear system starts at location
        dl+batchStride*i in memory. Also, the first element of each lower
        diagonal must be zero.
    d : ctypes.c_void_p
        Pointer to ${precision} ${real} dense array containing the main
        diagonal of the tri-diagonal linear system. The main diagonal d(i)
        that corresponds to the ith linear system starts at location
        d+batchStride*i in memory.
    du : ctypes.c_void_p
        Pointer to ${precision} ${real} dense array containing the upper
        diagonal of the tri-diagonal linear system. The upper diagonal du(i)
        that corresponds to the ith linear system starts at location
        du+batchStride*i in memory. Also, the last element of each upper
        diagonal must be zero.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} dense array that contains the
        right-hand-side of the tri-diagonal linear system. The
        right-hand-side x(i) that corresponds to the ith linear system
        starts at location x+batchStride*i in memory.
    batchCount : int
        Number of systems to solve.
    batchStride : int
        Stride (number of elements) that separates the vectors of every
        system (must be at least m).

    Returns
    -------
    bufferSizeInBytes : int
        number of bytes of the buffer used in the gtsv2StridedBatch.

    References
    ----------
    `cusparse<t>gtsv2StridedBatch_bufferSizeExt <https://docs.nvidia.com/cuda/cusparse/index.html#gtsv2stridedbatch_bufferSize>`_
    """
    bufferSizeInBytes = ctypes.c_int()
    status = _libcusparse.cusparseSgtsv2StridedBatch_bufferSizeExt(
        handle, m, int(dl), int(d), int(du), int(x), batchCount, batchStride,
        ctypes.byref(bufferSizeInBytes))
    cusparseCheckStatus(status)
    return bufferSizeInBytes.value

_libcusparse.cusparseSgtsv2StridedBatch.restype = int
_libcusparse.cusparseSgtsv2StridedBatch.argtypes =\
    [ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p
    ]
def cusparseSgtsv2StridedBatch(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer):
    """
    Compute the solution of multiple tridiagonal linear systems.
    
    Solves multiple tridiagonal linear systems, for i=0,…,batchCount:
        A(i) ∗ y(i) = x(i)
    The coefficient matrix A of each of these tri-diagonal linear system is
    defined with three vectors corresponding to its lower (dl), main (d), and
    upper (du) matrix diagonals; the right-hand sides are stored in the dense
    matrix X. Notice that solution Y overwrites right-hand-side matrix X on exit.
    The different matrices are assumed to be of the same size and are stored with
    a fixed batchStride in memory.

    The routine does not perform any pivoting and uses a combination of the
    Cyclic Reduction (CR) and the Parallel Cyclic Reduction (PCR) algorithms to
    find the solution. It achieves better performance when m is a power of 2.

    Parameters
    ----------
    handle : ctypes.c_void_p
        cuSPARSE context
    m : int
        Size of the linear system (must be >= 3)
    dl : ctypes.c_void_p
        Pointer to ${precision} ${real} dense array containing the lower
        diagonal of the tri-diagonal linear system. The lower diagonal dl(i)
        that corresponds to the ith linear system starts at location
        dl+batchStride*i in memory. Also, the first element of each lower
        diagonal must be zero.
    d : ctypes.c_void_p
        Pointer to ${precision} ${real} dense array containing the main
        diagonal of the tri-diagonal linear system. The main diagonal d(i)
        that corresponds to the ith linear system starts at location
        d+batchStride*i in memory.
    du : ctypes.c_void_p
        Pointer to ${precision} ${real} dense array containing the upper
        diagonal of the tri-diagonal linear system. The upper diagonal du(i)
        that corresponds to the ith linear system starts at location
        du+batchStride*i in memory. Also, the last element of each upper
        diagonal must be zero.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} dense array that contains the
        right-hand-side of the tri-diagonal linear system. The
        right-hand-side x(i) that corresponds to the ith linear system
        starts at location x+batchStride*i in memory.
    batchCount : int
        Number of systems to solve.
    batchStride : int
        Stride (number of elements) that separates the vectors of every
        system (must be at least m).
    pBuffer: ctypes.c_void_p
        Buffer allocated by the user, the size is return by gtsv2StridedBatch_bufferSizeExt

    Returns
    -------
    bufferSizeInBytes : int
        number of bytes of the buffer used in the gtsv2StridedBatch.

    References
    ----------
    `cusparse<t>gtsv2StridedBatch <https://docs.nvidia.com/cuda/cusparse/index.html#gtsv2stridedbatch>`_
    """
    status = _libcusparse.cusparseSgtsv2StridedBatch(
        handle, m, int(dl), int(d), int(du), int(x), batchCount, batchStride, int(pBuffer))
    cusparseCheckStatus(status)