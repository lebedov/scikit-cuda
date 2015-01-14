#!/usr/bin/env python
from __future__ import division, print_function, absolute_import
import functools
import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv

from . import misc

from .misc import init

try:
    import scipy.sparse
    from scipy.sparse.sputils import isscalarlike
    has_scipy = True
except ImportError:
    has_scipy = False

    # copy of isscalarlike from scipy.sparse.sputils
    def isscalarlike(x):
        """Is x either a scalar, an array scalar, or a 0-dim array?"""
        return np.isscalar(x) or (isdense(x) and x.ndim == 0)

toolkit_version = drv.get_version()

if toolkit_version < (3, 2, 0):
    raise ImportError("cuSPARSE not present prior to v3.2 of the CUDA toolkit")

"""
Python interface to cuSPARSE functions.

Note: You may need to set the environment variable CUDA_ROOT to the base of
your CUDA installation.
"""
# import low level cuSPARSE python wrappers and constants

try:
    from ._cusparse_cffi import *
except Exception as e:
    print(repr(e))
    estr = "autogenerattion and import of cuSPARSE wrappers failed\n"
    estr += ("Try setting the CUDA_ROOT environment variable to the base of "
             "your CUDA installation.  The autogeneration script tries to "
             "find the CUSPARSE header at CUDA_ROOT/include/cusparse_v2.h or "
             "CUDA_ROOT/include/cusparse.h\n")
    raise ImportError(estr)

# define higher level wrappers for common functions
# will check dimensions, autoset some variables and call the appriopriate
# function based on the input dtype

def defineIf(condition):
    def decorator(func):
        @functools.wraps(func)
        def func_wrapper(*args, **kwargs):
            if condition:
                return func(*args, **kwargs)
            else:
                raise NotImplementedError("requested cuSPARSE function not "
                                          "available for your CUDA version")
        return func_wrapper
    return decorator


def copyMatDescr(descr):
    """ create a new copy of Matrix Descriptor, descr """
    descr_copy = cusparseCreateMatDescr()
    cusparseSetMatType(descr_copy, cusparseGetMatType(descr))
    cusparseSetMatIndexBase(descr_copy, cusparseGetMatIndexBase(descr))
    cusparseSetMatDiagType(descr_copy, cusparseGetMatDiagType(descr))
    cusparseSetMatFillMode(descr_copy, cusparseGetMatFillMode(descr))
    return descr_copy


def dense_nnz(descrA, A, handle=None, dirA=CUSPARSE_DIRECTION_ROW, lda=None,
              nnzPerRowCol=None, nnzTotalDevHostPtr=None):
    """ higher level wrapper to cusparse<t>nnz routines """
    if not isinstance(A, pycuda.gpuarray.GPUArray):
        raise ValueError("A must be a pyCUDA gpuarray")
    if len(A.shape) != 2:
        raise ValueError("A must be 2D")
    if lda is None:
        lda = A.shape[0]

    if handle is None:
        handle = misc._global_cusparse_handle

    m, n = A.shape
    assert lda >= m
    dtype = A.dtype

    alloc = misc._global_cusparse_allocator

    if nnzPerRowCol is None:
        if dirA == CUSPARSE_DIRECTION_ROW:
            nnzPerRowCol = gpuarray.zeros((m, ), dtype=np.int32,
                                          allocator=alloc)
        elif dirA == CUSPARSE_DIRECTION_COLUMN:
            nnzPerRowCol = gpuarray.zeros((n, ), dtype=np.int32,
                                          allocator=alloc)
        else:
            raise ValueError("Invalid dirA")
    if nnzTotalDevHostPtr is None:
        nnzTotalDevHostPtr = ffi.new('int *', 0)
    if dtype == np.float32:
        fn = cusparseSnnz
    elif dtype == np.float64:
        fn = cusparseDnnz
    elif dtype == np.complex64:
        fn = cusparseCnnz
    elif dtype == np.complex128:
        fn = cusparseZnnz
    else:
        raise ValueError("unsupported sparse matrix dtype: %s" % dtype)
    fn(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol,
       nnzTotalDevHostPtr)
    return nnzPerRowCol, nnzTotalDevHostPtr[0]


def dense2csr(A, handle=None, descrA=None, lda=None, check_inputs=True):
    """ Convert dense matrix to CSR. """
    if not isinstance(A, pycuda.gpuarray.GPUArray):
        # try moving list or numpy array to GPU
        A = np.asfortranarray(np.atleast_2d(A))
        A = gpuarray.to_gpu(A)
    if check_inputs:
        if not isinstance(A, pycuda.gpuarray.GPUArray):
            raise ValueError("A must be a pyCUDA gpuarray")
        if len(A.shape) != 2:
            raise ValueError("A must be 2D")
        if descrA is not None:
            if cusparseGetMatType(descrA) != CUSPARSE_MATRIX_TYPE_GENERAL:
                raise ValueError("Only general matrix type supported")
        if not A.flags.f_contiguous:
            raise ValueError("Dense matrix A must be in column-major order")

    if lda is None:
        lda = A.shape[0]
    m, n = A.shape
    assert lda >= m
    dtype = A.dtype

    if handle is None:
        handle = misc._global_cusparse_handle

    if descrA is None:
        descrA = cusparseCreateMatDescr()
        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
        cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)

    nnzPerRow, nnz = dense_nnz(
        descrA, A, handle=handle, dirA=CUSPARSE_DIRECTION_ROW, lda=lda)

    alloc = misc._global_cusparse_allocator

    csrRowPtrA = gpuarray.zeros((m+1, ), dtype=np.int32, allocator=alloc)
    csrColIndA = gpuarray.zeros((nnz, ), dtype=np.int32, allocator=alloc)
    csrValA = gpuarray.zeros((nnz, ), dtype=dtype, allocator=alloc)

    if dtype == np.float32:
        fn = cusparseSdense2csr
    elif dtype == np.float64:
        fn = cusparseDdense2csr
    elif dtype == np.complex64:
        fn = cusparseCdense2csr
    elif dtype == np.complex128:
        fn = cusparseZdense2csr
    else:
        raise ValueError("unsupported sparse matrix dtype: %s" % dtype)

    fn(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA,
       csrColIndA)
    return (descrA, csrValA, csrRowPtrA, csrColIndA)


def csr2dense(m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A=None,
              handle=None, lda=None, check_inputs=True):
    """ convert CSR matrix to dense """
    if check_inputs:
        if A is not None:
            if not isinstance(A, pycuda.gpuarray.GPUArray):
                raise ValueError("A must be a pyCUDA gpuarray")
            if len(A.shape) != 2:
                raise ValueError("A must be 2D")
        if cusparseGetMatType(descrA) != CUSPARSE_MATRIX_TYPE_GENERAL:
            raise ValueError("Only general matrix type supported")
        if cusparseGetMatIndexBase(descrA) != CUSPARSE_INDEX_BASE_ZERO:
            raise ValueError("Only base 0 matrix supported")
        for arr in [csrValA, csrRowPtrA, csrColIndA]:
            if not isinstance(arr, pycuda.gpuarray.GPUArray):
                raise ValueError("csr* inputs must be a pyCUDA gpuarrays")
        if (csrRowPtrA.size != m + 1):
            raise ValueError("A: inconsistent size")

    if handle is None:
        handle = misc._global_cusparse_handle

    if lda is None:
        lda = m
    assert lda >= m
    dtype = csrValA.dtype

    alloc = misc._global_cusparse_allocator
    A = gpuarray.zeros((m, n), dtype=dtype, order='F', allocator=alloc)

    if dtype == np.float32:
        fn = cusparseScsr2dense
    elif dtype == np.float64:
        fn = cusparseDcsr2dense
    elif dtype == np.complex64:
        fn = cusparseCcsr2dense
    elif dtype == np.complex128:
        fn = cusparseZcsr2dense
    else:
        raise ValueError("unsupported sparse matrix dtype: %s" % dtype)

    fn(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda)
    return A


def dense2csc(A, handle=None, descrA=None, lda=None, check_inputs=True):
    """ Convert dense matrix to CSC. """
    if not isinstance(A, pycuda.gpuarray.GPUArray):
        # try moving list or numpy array to GPU
        A = np.asfortranarray(np.atleast_2d(A))
        A = gpuarray.to_gpu(A)
    if check_inputs:
        if not isinstance(A, pycuda.gpuarray.GPUArray):
            raise ValueError("A must be a pyCUDA gpuarray")
        if len(A.shape) != 2:
            raise ValueError("A must be 2D")
        if descrA is not None:
            if cusparseGetMatType(descrA) != CUSPARSE_MATRIX_TYPE_GENERAL:
                raise ValueError("Only general matrix type supported")
        if not A.flags.f_contiguous:
            raise ValueError("Dense matrix A must be in column-major order")

    if lda is None:
        lda = A.shape[0]
    m, n = A.shape
    assert lda >= m
    dtype = A.dtype

    if handle is None:
        handle = misc._global_cusparse_handle

    if descrA is None:
        descrA = cusparseCreateMatDescr()
        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
        cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)

    nnzPerCol, nnz = dense_nnz(
        descrA, A, handle=handle, dirA=CUSPARSE_DIRECTION_COLUMN, lda=lda)

    alloc = misc._global_cusparse_allocator
    cscColPtrA = gpuarray.zeros((n+1, ), dtype=np.int32, allocator=alloc)
    cscRowIndA = gpuarray.zeros((nnz, ), dtype=np.int32, allocator=alloc)
    cscValA = gpuarray.zeros((nnz, ), dtype=dtype, allocator=alloc)
    if dtype == np.float32:
        fn = cusparseSdense2csc
    elif dtype == np.float64:
        fn = cusparseDdense2csc
    elif dtype == np.complex64:
        fn = cusparseCdense2csc
    elif dtype == np.complex128:
        fn = cusparseZdense2csc
    else:
        raise ValueError("unsupported sparse matrix dtype: %s" % dtype)

    fn(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA,
       cscColPtrA)
    return (descrA, cscValA, cscColPtrA, cscRowIndA)


def csc2dense(m, n, descrA, cscValA, cscColPtrA, cscRowIndA, A=None,
              handle=None, lda=None, check_inputs=True):
    """ convert CSC matrix to dense """
    if check_inputs:
        if A is not None:
            if not isinstance(A, pycuda.gpuarray.GPUArray):
                raise ValueError("A must be a pyCUDA gpuarray")
            if len(A.shape) != 2:
                raise ValueError("A must be 2D")
        if cusparseGetMatType(descrA) != CUSPARSE_MATRIX_TYPE_GENERAL:
            raise ValueError("Only general matrix type supported")
        if cusparseGetMatIndexBase(descrA) != CUSPARSE_INDEX_BASE_ZERO:
            raise ValueError("Only base 0 matrix supported")
        for arr in [cscValA, cscColPtrA, cscRowIndA]:
            if not isinstance(arr, pycuda.gpuarray.GPUArray):
                raise ValueError("csc* inputs must be a pyCUDA gpuarrays")
        if (cscColPtrA.size != n + 1):
            raise ValueError("A: inconsistent size")

    if handle is None:
        handle = misc._global_cusparse_handle

    if lda is None:
        lda = m
    assert lda >= m
    dtype = cscValA.dtype
    alloc = misc._global_cusparse_allocator
    A = gpuarray.zeros((m, n), dtype=dtype, order='F', allocator=alloc)

    if dtype == np.float32:
        fn = cusparseScsc2dense
    elif dtype == np.float64:
        fn = cusparseDcsc2dense
    elif dtype == np.complex64:
        fn = cusparseCcsc2dense
    elif dtype == np.complex128:
        fn = cusparseZcsc2dense
    else:
        raise ValueError("unsupported sparse matrix dtype: %s" % dtype)

    fn(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda)
    return A


def csr2coo(csrRowPtr, nnz, handle=None, m=None, cooRowInd=None,
            idxBase=CUSPARSE_INDEX_BASE_ZERO, check_inputs=True):
    """ convert CSR to COO """
    if check_inputs:
        if cooRowInd is not None:
            if not isinstance(cooRowInd, pycuda.gpuarray.GPUArray):
                raise ValueError("cooRowInd must be a pyCUDA gpuarray")
        if not isinstance(csrRowPtr, pycuda.gpuarray.GPUArray):
            raise ValueError("csrRowPtr must be a pyCUDA gpuarraya")
    if handle is None:
        handle = misc._global_cusparse_handle
    if m is None:
        m = csrRowPtr.size - 1
    if cooRowInd is None:
        alloc = misc._global_cusparse_allocator
        cooRowInd = gpuarray.zeros((nnz, ), dtype=np.int32, allocator=alloc)
    cusparseXcsr2coo(handle, csrRowPtr, nnz, m, cooRowInd, idxBase)
    return cooRowInd


# define with alternate naming for convenience
def csc2coo(cscColPtr, nnz, handle=None, m=None, cooColInd=None,
            idxBase=CUSPARSE_INDEX_BASE_ZERO, check_inputs=True):
    """ convert CSC to COO """
    # if m is None:
    #     m = cooColPtr.size - 1
    cooColInd = csr2coo(csrRowPtr=cscColPtr, nnz=nnz, handle=handle, m=m,
                        cooRowInd=cooColInd, idxBase=idxBase,
                        check_inputs=check_inputs)
    return cooColInd


def coo2csr(cooRowInd, m, handle=None, nnz=None, csrRowPtr=None,
            idxBase=CUSPARSE_INDEX_BASE_ZERO, check_inputs=True):
    """ convert COO to CSR """
    if check_inputs:
        if csrRowPtr is not None:
            if not isinstance(csrRowPtr, pycuda.gpuarray.GPUArray):
                raise ValueError("csrRowPtr must be a pyCUDA gpuarray")
        if not isinstance(cooRowInd, pycuda.gpuarray.GPUArray):
            raise ValueError("cooRowInd must be a pyCUDA gpuarraya")
    if handle is None:
        handle = misc._global_cusparse_handle
    if nnz is None:
        nnz = cooRowInd.size
    if csrRowPtr is None:
        alloc = misc._global_cusparse_allocator
        csrRowPtr = gpuarray.zeros((m+1, ), dtype=np.int32, allocator=alloc)
    cusparseXcoo2csr(handle, cooRowInd, nnz, m, csrRowPtr, idxBase)
    return csrRowPtr


# define with alternate naming for convenience
def coo2csc(cooColInd, m, handle=None, nnz=None, cscColPtr=None,
            idxBase=CUSPARSE_INDEX_BASE_ZERO, check_inputs=True):
    """ convert COO to CSC"""
    cscColPtr = coo2csr(cooRowInd=cooColInd, m=m, handle=handle, nnz=nnz,
                        csrRowPtr=cscColPtr, idxBase=idxBase,
                        check_inputs=check_inputs)
    return cscColPtr


def csr2csc(m, n, csrVal, csrRowPtr, csrColInd, handle=None, nnz=None,
            cscVal=None, cscColPtr=None, cscRowInd=None, A=None,
            copyValues=CUSPARSE_ACTION_NUMERIC,
            idxBase=CUSPARSE_INDEX_BASE_ZERO, check_inputs=True):
    """ convert CSR to CSC """
    if check_inputs:
        if (cscVal is not None) or (cscColPtr is not None) or \
           (cscRowInd is not None):
            for arr in [cscVal, cscColPtr, csrRowInd]:
                if not isinstance(arr, pycuda.gpuarray.GPUArray):
                    raise ValueError("csc* inputs must all be pyCUDA gpuarrays"
                                     " or None")
        for arr in [csrVal, csrRowPtr, csrColInd]:
            if not isinstance(arr, pycuda.gpuarray.GPUArray):
                raise ValueError("csr* inputs must be a pyCUDA gpuarrays")
        if (csrRowPtr.size != m + 1):
            raise ValueError("A: inconsistent size")
    if handle is None:
        handle = misc._global_cusparse_handle
    dtype = csrVal.dtype
    nnz = csrVal.size
    if cscVal is None:
        alloc = misc._global_cusparse_allocator
        cscVal = gpuarray.zeros((nnz, ), dtype=dtype, allocator=alloc)
        cscColPtr = gpuarray.zeros((n+1, ), dtype=np.int32, allocator=alloc)
        cscRowInd = gpuarray.zeros((nnz, ), dtype=np.int32, allocator=alloc)
    if dtype == np.float32:
        fn = cusparseScsr2csc
    elif dtype == np.float64:
        fn = cusparseDcsr2csc
    elif dtype == np.complex64:
        fn = cusparseCcsr2csc
    elif dtype == np.complex128:
        fn = cusparseZcsr2csc
    else:
        raise ValueError("unsupported sparse matrix dtype: %s" % dtype)
    fn(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd,
       cscColPtr, copyValues, idxBase)
    return (cscVal, cscColPtr, cscRowInd)


# also define csc2csr as a convenience
def csc2csr(m, n, cscVal, cscColPtr, cscRowInd, handle=None, nnz=None,
            csrVal=None, csrRowPtr=None, csrColInd=None,
            copyValues=CUSPARSE_ACTION_NUMERIC,
            idxBase=CUSPARSE_INDEX_BASE_ZERO, check_inputs=True):
    """ convert CSC to CSR """
    csrVal, csrRowPtr, csrColInd = csr2csc(
        m, n, cscVal, cscColPtr, cscRowInd, handle=handle, nnz=nnz,
        cscVal=csrVal, cscColPtr=csrRowPtr, cscRowInd=csrColInd,
        copyValues=copyValues, idxBase=idxBase, check_inputs=check_inputs)
    return csrVal, csrRowPtr, csrColInd


def csrmv(descrA, csrValA, csrRowPtrA, csrColIndA, m, n, x, handle=None,
          nnz=None, transA=CUSPARSE_OPERATION_NON_TRANSPOSE, alpha=1.0,
          beta=0.0, y=None, check_inputs=True):
    """ multiply a sparse matrix A, by dense vector x:
    y = alpha * transA(A)*x + beta*y

    higher level wrapper to cusparse<t>csrmv routines
    """
    if check_inputs:
        if not isinstance(csrValA, pycuda.gpuarray.GPUArray):
            raise ValueError("csrValA must be a pyCUDA gpuarray")
        if not isinstance(csrRowPtrA, pycuda.gpuarray.GPUArray):
            raise ValueError("csrRowPtrA must be a pyCUDA gpuarray")
        if not isinstance(csrColIndA, pycuda.gpuarray.GPUArray):
            raise ValueError("csrColIndA must be a pyCUDA gpuarray")
        if not isinstance(x, pycuda.gpuarray.GPUArray):
            raise ValueError("x must be a pyCUDA gpuarray")

    if handle is None:
        handle = misc._global_cusparse_handle
    if nnz is None:
        nnz = csrValA.size
    dtype = csrValA.dtype
    if y is None:
        alloc = misc._global_cusparse_allocator
        if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
            y = gpuarray.zeros((m, ), dtype=dtype, allocator=alloc)
        else:
            y = gpuarray.zeros((n, ), dtype=dtype, allocator=alloc)
    # perform some basic sanity checks
    if check_inputs:
        if csrValA.size != nnz:
            raise ValueError("length of csrValA array must match nnz")
        if (x.dtype != dtype) or (y.dtype != dtype):
            raise ValueError("incompatible dtypes")
        if csrRowPtrA.size != (m+1):
            raise ValueError("length of csrRowPtrA array must be m+1")
        if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
            if x.size != n:
                raise ValueError("sizes of x, A incompatible")
            if y.size != m:
                raise ValueError("sizes of y, A incompatible")
        else:
            if x.size != m:
                raise ValueError("sizes of x, A incompatible")
            if y.size != n:
                raise ValueError("sizes of y, A incompatible")
    if dtype == np.float32:
        fn = cusparseScsrmv
    elif dtype == np.float64:
        fn = cusparseDcsrmv
    elif dtype == np.complex64:
        fn = cusparseCcsrmv
    elif dtype == np.complex128:
        fn = cusparseZcsrmv
    else:
        raise ValueError("unsupported sparse matrix dtype: %s" % dtype)
    if toolkit_version >= (4, 1, 0):

        fn(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA,
           csrColIndA, x, beta, y)
    else:
        fn(handle, transA, m, n, alpha, descrA, csrValA, csrRowPtrA,
           csrColIndA, x, beta, y)

    return y


def csrmm(m, n, k, descrA, csrValA, csrRowPtrA, csrColIndA, B, handle=None,
          C=None, nnz=None, transA=CUSPARSE_OPERATION_NON_TRANSPOSE, alpha=1.0,
          beta=0.0, ldb=None, ldc=None, check_inputs=True):
    """ multiply a sparse matrix, A, by dense matrix B:
    C = alpha * transA(A) * B + beta * C.

    higher level wrapper to cusparse<t>csrmm routines
    """
    if check_inputs:
        for item in [csrValA, csrRowPtrA, csrColIndA, B]:
            if not isinstance(item, pycuda.gpuarray.GPUArray):
                raise ValueError("csr*, B, must be pyCUDA gpuarrays")
        if C is not None:
            if not isinstance(C, pycuda.gpuarray.GPUArray):
                raise ValueError("C must be a pyCUDA gpuarray or None")
        # dense matrices must be in column-major order
        if not B.flags.f_contiguous:
            raise ValueError("Dense matrix B must be in column-major order")
    if handle is None:
        handle = misc._global_cusparse_handle

    dtype = csrValA.dtype
    if C is None:
        if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
            ldc = m
        else:
            ldc = k
        alloc = misc._global_cusparse_allocator
        C = gpuarray.zeros((ldc, n), dtype=dtype, order='F', allocator=alloc)
    elif not C.flags.f_contiguous:
        raise ValueError("Dense matrix C must be in column-major order")
    if nnz is None:
        nnz = csrValA.size
    if ldb is None:
        ldb = B.shape[0]
    if ldc is None:
        ldc = C.shape[0]

    # perform some basic sanity checks
    if check_inputs:
        if csrValA.size != nnz:
            raise ValueError("length of csrValA array must match nnz")
        if (B.dtype != dtype) or (C.dtype != dtype):
            raise ValueError("A, B, C must share a common dtype")
        if ldb < B.shape[0]:
            raise ValueError("ldb invalid for matrix B")
        if ldc < C.shape[0]:
            raise ValueError("ldc invalid for matrix C")
        if (C.shape[1] != n) or (B.shape[1] != n):
            raise ValueError("bad shape for B or C")
        if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
            if (ldb != k) or (ldc != m):
                raise ValueError("size of A incompatible with B or C")
        else:
            if (ldb != m) or (ldc != k):
                raise ValueError("size of A incompatible with B or C")
        if csrRowPtrA.size != m+1:
            raise ValueError("length of csrRowPtrA invalid")
    if dtype == np.float32:
        fn = cusparseScsrmm
    elif dtype == np.float64:
        fn = cusparseDcsrmm
    elif dtype == np.complex64:
        fn = cusparseCcsrmm
    elif dtype == np.complex128:
        fn = cusparseZcsrmm
    else:
        raise ValueError("unsupported sparse matrix dtype: %s" % dtype)
    if toolkit_version >= (4, 1, 0):
        fn(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA,
           csrColIndA, B, ldb, beta, C, ldc)
    else:
        fn(handle, transA, m, n, k, alpha, descrA, csrValA, csrRowPtrA,
           csrColIndA, B, ldb, beta, C, ldc)
    return C



@defineIf(toolkit_version >= (5, 5, 0))
def csrmm2(m, n, k, descrA, csrValA, csrRowPtrA, csrColIndA, B, handle=None,
           C=None, nnz=None, transA=CUSPARSE_OPERATION_NON_TRANSPOSE,
           transB=CUSPARSE_OPERATION_NON_TRANSPOSE, alpha=1.0, beta=0.0,
           ldb=None, ldc=None, check_inputs=True):
    """ multiply two sparse matrices:  C = transA(A) * transB(B)

    higher level wrapper to cusparse<t>csrmm2 routines.
    """
    if check_inputs:
        for item in [csrValA, csrRowPtrA, csrColIndA, B]:
            if not isinstance(item, pycuda.gpuarray.GPUArray):
                raise ValueError("csr*, B, must be pyCUDA gpuarrays")
        if C is not None:
            if not isinstance(C, pycuda.gpuarray.GPUArray):
                raise ValueError("C must be a pyCUDA gpuarray or None")
        # dense matrices must be in column-major order
        if not B.flags.f_contiguous:
            raise ValueError("Dense matrix B must be column-major order")

        if transB == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
            raise ValueError("Conjugate transpose operation not supported "
                             "for dense matrix B")

        if (transB == CUSPARSE_OPERATION_TRANSPOSE) and \
           (transA != CUSPARSE_OPERATION_NON_TRANSPOSE):
            raise ValueError("if B is transposed, only A non-transpose is "
                             "supported")
    if handle is None:
        handle = misc._global_cusparse_handle

    dtype = csrValA.dtype
    if C is None:
        if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
            ldc = m
        else:
            ldc = k
        alloc = misc._global_cusparse_allocator
        C = gpuarray.zeros((ldc, n), dtype=dtype, order='F',
                           allocator=alloc)
    elif not C.flags.f_contiguous:
        raise ValueError("Dense matrix C must be in column-major order")
    if nnz is None:
        nnz = csrValA.size
    if ldb is None:
        ldb = B.shape[0]
    if ldc is None:
        ldc = C.shape[0]

    # perform some basic sanity checks
    if check_inputs:
        if csrValA.size != nnz:
            raise ValueError("length of csrValA array must match nnz")
        if (B.dtype != dtype) or (C.dtype != dtype):
            raise ValueError("A, B, C must share a common dtype")
        if ldb < B.shape[0]:
            raise ValueError("ldb invalid for matrix B")
        if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
            ldOpA = m  # leading dimension for op(A)
            tdOpA = k  # trailing dimension for op(A)
        else:
            ldOpA = k
            tdOpA = m
        if transB == CUSPARSE_OPERATION_NON_TRANSPOSE:
            if B.shape[1] != n:
                raise ValueError("B, n incompatible")
            if (ldb < tdOpA):
                raise ValueError("size of A incompatible with B")
        else:
            if ldb < n:
                raise ValueError("B, n incompatible")
            if (B.shape[1] != tdOpA):
                raise ValueError("size of A incompatible with B")
        if (C.shape[1] != n):
            raise ValueError("bad shape for C")
        if (ldc != ldOpA):
            raise ValueError("size of A incompatible with C")
        if csrRowPtrA.size != m+1:
            raise ValueError("length of csrRowPtrA invalid")
    if dtype == np.float32:
        fn = cusparseScsrmm2
    elif dtype == np.float64:
        fn = cusparseDcsrmm2
    elif dtype == np.complex64:
        fn = cusparseCcsrmm2
    elif dtype == np.complex128:
        fn = cusparseZcsrmm2
    else:
        raise ValueError("unsupported sparse matrix dtype: %s" % dtype)
    transa = transA
    transb = transB
    try:
        fn(handle, transa, transb, m, n, k, nnz, alpha, descrA, csrValA,
           csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc)
    except CUSPARSE_STATUS_INVALID_VALUE as e:
        print("m={}, n={}, k={}, nnz={}, ldb={}, ldc={}".format(
            m, n, k, nnz, ldb, ldc))
        raise(e)
    return C


@defineIf(toolkit_version >= (5, 0, 0))
def _csrgeamNnz(m, n, descrA, csrRowPtrA, csrColIndA, descrB, csrRowPtrB,
                csrColIndB, handle=None, descrC=None, csrRowPtrC=None,
                nnzA=None, nnzB=None, check_inputs=True):
    """ support routine for csrgeam

    higher level wrapper to cusparseXcsrgeamNnz.
    """
    if check_inputs:
        for array in [csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB]:
            if not isinstance(array, pycuda.gpuarray.GPUArray):
                raise ValueError("all csr* inputs must be a pyCUDA gpuarray")
        if cusparseGetMatType(descrA) != CUSPARSE_MATRIX_TYPE_GENERAL:
            raise ValueError("Only general matrix type supported")
        if cusparseGetMatType(descrB) != CUSPARSE_MATRIX_TYPE_GENERAL:
            raise ValueError("Only general matrix type supported")
        if descrC is not None:
            if not isinstance(csrRowPtrC, pycuda.gpuarray.GPUArray):
                raise ValueError("csrRowPtrC must be a gpuarray or None")
            if cusparseGetMatType(descrC) != CUSPARSE_MATRIX_TYPE_GENERAL:
                raise ValueError("Only general matrix type supported")

    if handle is None:
        handle = misc._global_cusparse_handle
    if nnzA is None:
        nnzA = csrColIndA.size
    if nnzB is None:
        nnzB = csrColIndB.size
    if descrC is None:
        return_descrC = True
        descrC = cusparseCreateMatDescr()
        cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL)
    else:
        return_descrC = False
    if csrRowPtrC is None:
        alloc = misc._global_cusparse_allocator
        csrRowPtrC = gpuarray.zeros((m+1, ), dtype=np.int32,
                                    allocator=alloc)
    nnzTotalDevHostPtr = ffi.new('int *', 0)

    # perform some basic sanity checks
    if check_inputs:
        if csrColIndA.size != nnzA:
            raise ValueError("length of csrValA array must match nnzA")
        if csrColIndB.size != nnzB:
            raise ValueError("length of csrValB array must match nnzB")
        if csrRowPtrA.size != m+1:
            raise ValueError("length of csrRowPtrA array must be m+1")
        if csrRowPtrB.size != m+1:
            raise ValueError("length of csrRowPtrB array must be m+1")

    cusparseXcsrgeamNnz(handle, m, n, descrA, nnzA, csrRowPtrA, csrColIndA,
                        descrB, nnzB, csrRowPtrB, csrColIndB, descrC,
                        csrRowPtrC, nnzTotalDevHostPtr)
    nnzC = nnzTotalDevHostPtr[0]
    if return_descrC:
        return descrC, csrRowPtrC, nnzC
    else:
        return nnzC


@defineIf(toolkit_version >= (5, 0, 0))
def csrgeam(m, n, descrA, csrValA, csrRowPtrA, csrColIndA, descrB, csrValB,
            csrRowPtrB, csrColIndB, handle=None, alpha=1.0, beta=0.0,
            nnzA=None, nnzB=None,  check_inputs=True):
    """ add two sparse matrices:  C = alpha*A + beta*B.
    higher level wrapper to cusparse<t>csrgemm routines.
    """
    if check_inputs:
        for array in [csrValA, csrRowPtrA, csrColIndA, csrValB, csrRowPtrB,
                      csrColIndB]:
            if not isinstance(array, pycuda.gpuarray.GPUArray):
                raise ValueError("all csr* inputs must be a pyCUDA gpuarray")
        if cusparseGetMatType(descrA) != CUSPARSE_MATRIX_TYPE_GENERAL:
            raise ValueError("Only general matrix type supported")
        if cusparseGetMatType(descrB) != CUSPARSE_MATRIX_TYPE_GENERAL:
            raise ValueError("Only general matrix type supported")

    if handle is None:
        handle = misc._global_cusparse_handle
    if nnzA is None:
        nnzA = csrValA.size
    if nnzB is None:
        nnzB = csrValB.size
    dtype = csrValA.dtype

    # perform some basic sanity checks
    if check_inputs:
        if csrValA.size != nnzA:
            raise ValueError("length of csrValA array must match nnzA")
        if csrValB.size != nnzB:
            raise ValueError("length of csrValB array must match nnzB")
        if (dtype != csrValB.dtype):
            raise ValueError("incompatible dtypes")
        if csrRowPtrA.size != m + 1:
            raise ValueError("bad csrRowPtrA size")
        if csrRowPtrB.size != m + 1:
            raise ValueError("bad csrRowPtrB size")

    # allocate output matrix C descr and row pointers
    descrC = cusparseCreateMatDescr()
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL)
    alloc = misc._global_cusparse_allocator
    csrRowPtrC = gpuarray.zeros((m+1, ), dtype=np.int32, allocator=alloc)

    # call csrgemmNnz to determine nnzC and fill in csrRowPtrC
    nnzC = _csrgeamNnz(m, n, descrA, csrRowPtrA, csrColIndA, descrB,
                       csrRowPtrB, csrColIndB, handle=handle, descrC=descrC,
                       csrRowPtrC=csrRowPtrC, nnzA=nnzA, nnzB=nnzB,
                       check_inputs=False)

    # allocated rest of C based on nnzC
    csrValC = gpuarray.zeros((nnzC, ), dtype=dtype, allocator=alloc)
    csrColIndC = gpuarray.zeros((nnzC, ), dtype=np.int32, allocator=alloc)
    if dtype == np.float32:
        fn = cusparseScsrgeam
    elif dtype == np.float64:
        fn = cusparseDcsrgeam
    elif dtype == np.complex64:
        fn = cusparseCcsrgeam
    elif dtype == np.complex128:
        fn = cusparseZcsrgeam
    else:
        raise ValueError("unsupported sparse matrix dtype: %s" % dtype)

    fn(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA,
       beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC,
       csrValC, csrRowPtrC, csrColIndC)
    return (descrC, csrValC, csrRowPtrC, csrColIndC)


@defineIf(toolkit_version >= (5, 0, 0))
def _csrgemmNnz(m, n, k, descrA, csrRowPtrA, csrColIndA, descrB, csrRowPtrB,
                csrColIndB, handle=None, descrC=None, csrRowPtrC=None,
                nnzA=None, nnzB=None, transA=CUSPARSE_OPERATION_NON_TRANSPOSE,
                transB=CUSPARSE_OPERATION_NON_TRANSPOSE, check_inputs=True):
    """ support routine for csrgemm.

    higher level wrapper to cusparseXcsrgemmNnz.
    """
    if check_inputs:
        for array in [csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB]:
            if not isinstance(array, pycuda.gpuarray.GPUArray):
                raise ValueError("all csr* inputs must be a pyCUDA gpuarray")
        if cusparseGetMatType(descrA) != CUSPARSE_MATRIX_TYPE_GENERAL:
            raise ValueError("Only general matrix type supported")
        if cusparseGetMatType(descrB) != CUSPARSE_MATRIX_TYPE_GENERAL:
            raise ValueError("Only general matrix type supported")
        if descrC is not None:
            if not isinstance(csrRowPtrC, pycuda.gpuarray.GPUArray):
                raise ValueError("csrRowPtrC must be a gpuarray or None")
            if cusparseGetMatType(descrC) != CUSPARSE_MATRIX_TYPE_GENERAL:
                raise ValueError("Only general matrix type supported")
    if handle is None:
        handle = misc._global_cusparse_handle
    if nnzA is None:
        nnzA = csrColIndA.size
    if nnzB is None:
        nnzB = csrColIndB.size
    if descrC is None:
        return_descrC = True
        descrC = cusparseCreateMatDescr()
        cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL)
    else:
        return_descrC = False
    if csrRowPtrC is None:
        alloc = misc._global_cusparse_allocator
        csrRowPtrC = gpuarray.zeros((m+1, ), dtype=np.int32,
                                    allocator=alloc)
    nnzTotalDevHostPtr = ffi.new('int *', 0)

    # perform some basic sanity checks
    if check_inputs:
        if csrColIndA.size != nnzA:
            raise ValueError("length of csrValA array must match nnzA")
        if csrColIndB.size != nnzB:
            raise ValueError("length of csrValB array must match nnzB")
        if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
            ptrA_size = m + 1
        else:
            ptrA_size = k + 1
        if transB == CUSPARSE_OPERATION_NON_TRANSPOSE:
            ptrB_size = k + 1
        else:
            ptrB_size = n + 1
        if csrRowPtrA.size != ptrA_size:
            raise ValueError("length of csrRowPtrA array must be m+1")
        if csrRowPtrB.size != ptrB_size:
            raise ValueError("length of csrRowPtrB array must be n+1")

    cusparseXcsrgemmNnz(handle, transA, transB, m, n, k, descrA, nnzA,
                        csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB,
                        csrColIndB, descrC, csrRowPtrC, nnzTotalDevHostPtr)
    nnzC = nnzTotalDevHostPtr[0]
    if return_descrC:
        return descrC, csrRowPtrC, nnzC
    else:
        return nnzC


@defineIf(toolkit_version >= (5, 0, 0))
def csrgemm(m, n, k, descrA, csrValA, csrRowPtrA, csrColIndA, descrB, csrValB,
            csrRowPtrB, csrColIndB, handle=None, nnzA=None, nnzB=None,
            transA=CUSPARSE_OPERATION_NON_TRANSPOSE,
            transB=CUSPARSE_OPERATION_NON_TRANSPOSE, check_inputs=True):
    """ multiply two sparse matrices:  C = transA(A) * transB(B)

    higher level wrapper to cusparse<t>csrgemm routines.

    Note
    ----
    transA(A) is shape m x k.  transB(B) is shape k x n.  C is shape m x n

    if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
        m, k = A.shape
    else:
        k, m = A.shape

    if transB == CUSPARSE_OPERATION_NON_TRANSPOSE:
        k, n = B.shape
    else:
        n, k = B.shape

    """

    if check_inputs:
        for array in [csrValA, csrRowPtrA, csrColIndA, csrValB, csrRowPtrB,
                      csrColIndB]:
            if not isinstance(array, pycuda.gpuarray.GPUArray):
                raise ValueError("all csr* inputs must be a pyCUDA gpuarray")
        if cusparseGetMatType(descrA) != CUSPARSE_MATRIX_TYPE_GENERAL:
            raise ValueError("Only general matrix type supported")
        if cusparseGetMatType(descrB) != CUSPARSE_MATRIX_TYPE_GENERAL:
            raise ValueError("Only general matrix type supported")

    if handle is None:
        handle = misc._global_cusparse_handle
    if nnzA is None:
        nnzA = csrValA.size
    if nnzB is None:
        nnzB = csrValB.size
    dtype = csrValA.dtype

    # perform some basic sanity checks
    if check_inputs:
        if csrValA.size != nnzA:
            raise ValueError("length of csrValA array must match nnzA")
        if csrValB.size != nnzB:
            raise ValueError("length of csrValB array must match nnzB")
        if (dtype != csrValB.dtype):
            raise ValueError("incompatible dtypes")
        if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
            ptrA_size = m + 1
        else:
            ptrA_size = k + 1
        if transB == CUSPARSE_OPERATION_NON_TRANSPOSE:
            ptrB_size = k + 1
        else:
            ptrB_size = n + 1
        if csrRowPtrA.size != ptrA_size:
            raise ValueError("bad csrRowPtrA size")
        if csrRowPtrB.size != ptrB_size:
            raise ValueError("bad csrRowPtrB size")

    # allocate output matrix C descr and row pointers
    descrC = cusparseCreateMatDescr()
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL)
    alloc = misc._global_cusparse_allocator
    csrRowPtrC = gpuarray.zeros((m+1, ), dtype=np.int32, allocator=alloc)

    # call csrgemmNnz to determine nnzC and fill in csrRowPtrC
    nnzC = _csrgemmNnz(m, n, k, descrA, csrRowPtrA, csrColIndA, descrB,
                       csrRowPtrB, csrColIndB, handle=handle, descrC=descrC,
                       csrRowPtrC=csrRowPtrC, nnzA=nnzA, nnzB=nnzB,
                       transA=transA, transB=transB, check_inputs=False)

    # allocated rest of C based on nnzC
    csrValC = gpuarray.zeros((nnzC, ), dtype=dtype, allocator=alloc)
    csrColIndC = gpuarray.zeros((nnzC, ), dtype=np.int32, allocator=alloc)

    if dtype == np.float32:
        fn = cusparseScsrgemm
    elif dtype == np.float64:
        fn = cusparseDcsrgemm
    elif dtype == np.complex64:
        fn = cusparseCcsrgemm
    elif dtype == np.complex128:
        fn = cusparseZcsrgemm
    else:
        raise ValueError("unsupported sparse matrix dtype: %s" % dtype)

    fn(handle, transA, transB, m, n, k, descrA, nnzA, csrValA, csrRowPtrA,
       csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC,
       csrValC, csrRowPtrC, csrColIndC)
    return (descrC, csrValC, csrRowPtrC, csrColIndC)


class CSR(object):
    """ cuSPARSE CSR (compressed sparse row) matrix object """
    def __init__(self, descr, csrVal, csrRowPtr, csrColInd, shape,
                 handle=None):

        if csrRowPtr.size != (shape[0] + 1):
            raise ValueError("size of RowPtr inconsistent with shape")
        if csrVal.size != csrColInd.size:
            raise ValueError("size of csrVal and csrColInd inconsistent")
        if csrColInd.dtype != np.int32:
            raise ValueError("csrColInd must be a 32-bit integer array")
        if csrRowPtr.dtype != np.int32:
            raise ValueError("csrRowPtr must be a 32-bit integer array")

        # if input arrays are on the host, transfer them to the GPU
        self._alloc = misc._global_cusparse_allocator

        if isinstance(csrVal, np.ndarray):
            csrVal = gpuarray.to_gpu(csrVal, allocator=self._alloc)
        if isinstance(csrRowPtr, np.ndarray):
            csrRowPtr = gpuarray.to_gpu(csrRowPtr, allocator=self._alloc)
        if isinstance(csrColInd, np.ndarray):
            csrColInd = gpuarray.to_gpu(csrColInd, allocator=self._alloc)

        if handle is None:
            self.handle = misc._global_cusparse_handle
        else:
            self.handle = handle
        self.descr = descr
        self.Val = csrVal
        self.RowPtr = csrRowPtr
        self.ColInd = csrColInd
        self.dtype = csrVal.dtype
        self.shape = shape

        # also mirror scipy.sparse.csr_matrix property names for convenience
        self.data = csrVal
        self.indices = csrColInd
        self.indptr = csrRowPtr

        # properties
        self.__matrix_type = None
        self.__index_base = None
        self.__diag_type = None
        self.__fill_mode = None

    # alternative constructor from dense ndarray, gpuarray or cuSPARSE matrix
    @classmethod
    def to_CSR(cls, A, handle=None):
        """ convert dense numpy or gpuarray matrices as well as any
        scipy.sparse matrix formats to cuSPARSE CSR.
        """
        alloc = misc._global_cusparse_allocator
        if has_scipy and isinstance(A, scipy.sparse.spmatrix):
            """Convert scipy.sparse CSR, COO, BSR, etc to cuSPARSE CSR"""
            # converting BSR, COO, etc to CSR
            if A.dtype.char not in ['f', 'd', 'F', 'D']:
                raise ValueError("unsupported numpy dtype {}".format(A.dtype))

            if not isinstance(A, scipy.sparse.csr_matrix):
                A = A.tocsr()

            # avoid .astype() calls if possible for speed
            if A.indptr.dtype != np.int32:
                csrRowPtr = gpuarray.to_gpu(A.indptr.astype(np.int32),
                                            allocator=alloc)
            else:
                csrRowPtr = gpuarray.to_gpu(A.indptr, allocator=alloc)

            if A.indices.dtype != np.int32:
                csrColInd = gpuarray.to_gpu(A.indices.astype(np.int32),
                                            allocator=alloc)
            else:
                csrColInd = gpuarray.to_gpu(A.indices, allocator=alloc)

            csrVal = gpuarray.to_gpu(A.data, allocator=alloc)
            descr = cusparseCreateMatDescr()
            cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL)
            cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO)
        else:
            """Take dense numpy array or pyCUDA gpuarray and convert to CSR """
            if not isinstance(A, pycuda.gpuarray.GPUArray):
                A = np.asfortranarray(np.atleast_2d(A))
                A = gpuarray.to_gpu(A, allocator=alloc)
            else:
                # dense matrix must be column-major
                if not A.flags.f_contiguous:
                    # TODO :an change to Fortran ordering be done directly on
                    # the gpuarray without going back to numpy?
                    A = A.get()
                    A = np.asfortranarray(A)
                    A = gpuarray.to_gpu(A, allocator=alloc)

            (descr, csrVal, csrRowPtr, csrColInd) = dense2csr(A, handle=handle)
        return cls(descr, csrVal, csrRowPtr, csrColInd, A.shape)

    @property
    def matrix_type(self):
        """ matrix type """
        if self.__matrix_type is None:
            return cusparseGetMatType(self.descr)
        else:
            return self.__matrix_type

    @matrix_type.setter
    def matrix_type(self, matrix_type):
        """ matrix type """
        self.__matrix_type = cusparseSetMatType(self.descr, matrix_type)

    @property
    def index_base(self):
        """ matrix index base """
        if self.__index_base is None:
            return cusparseGetMatIndexBase(self.descr)
        else:
            return self.__index_base

    @index_base.setter
    def index_base(self, index_base):
        """ matrix index base """
        self.__index_base = cusparseSetMatIndexBase(self.descr, index_base)

    @property
    def diag_type(self):
        """ matrix diag type """
        if self.__diag_type is None:
            return cusparseGetMatDiagType(self.descr)
        else:
            return self.__diag_type

    @diag_type.setter
    def diag_type(self, diag_type):
        """matrix diag type """
        self.__diag_type = cusparseSetMatDiagType(self.descr, diag_type)

    @property
    def fill_mode(self):
        """matrix fill mode """
        if self.__fill_mode is None:
            return cusparseGetMatFillMode(self.descr)
        else:
            return self.__fill_mode

    @fill_mode.setter
    def fill_mode(self, fill_mode):
        """matrix fill mode """
        self.__fill_mode = cusparseSetMatFillMode(self.descr, fill_mode)

    @property
    def nnz(self):
        """ number of non-zeros """
        return self.Val.size

    # mirror the function name from scipy
    def getnnz(self):
        """ return number of non-zeros"""
        return self.nnz

    def tocsr_scipy(self):
        """ return as scipy csr_matrix in host memory """
        from scipy.sparse import csr_matrix
        return csr_matrix((self.data.get(),
                           self.indices.get(),
                           self.indptr.get()),
                          shape=self.shape)

    def todense(self, lda=None, to_cpu=False, handle=None, stream=None,
                autosync=True):
        """ return dense gpuarray if to_cpu=False, numpy ndarray if to_cpu=True
        """
        m, n = self.shape
        if lda is None:
            lda = m
        else:
            assert lda >= m
        if handle is None:
            handle = self.handle
        if stream is not None:
            cusparseSetStream(handle, stream.handle)
        A = csr2dense(m, n, self.descr, self.Val, self.RowPtr,
                      self.ColInd, handle=handle, lda=lda)
        if autosync:
            drv.Context.synchronize()
        if to_cpu:
            return A.get()
        else:
            return A

    def mv(self, x, transA=CUSPARSE_OPERATION_NON_TRANSPOSE, alpha=1.0,
           beta=0.0, y=None, check_inputs=True, to_cpu=False,
           autosync=True, handle=None, stream=None):
        """ multiplication by dense vector x:  y = alpha*transA(A)*x + beta*y.
        """
        m, n = self.shape

        # try moving list or numpy array to GPU
        if not isinstance(x, pycuda.gpuarray.GPUArray):
            x = np.atleast_1d(x)  # .astype(self.dtype)
            x = gpuarray.to_gpu(x, allocator=self._alloc)
        if handle is None:
            handle = self.handle
        if stream is not None:
            cusparseSetStream(handle, stream.handle)
        y = csrmv(self.descr, self.Val, self.RowPtr, self.ColInd, m, n,
                  x, handle=handle, transA=transA, alpha=alpha, beta=beta, y=y,
                  check_inputs=check_inputs)
        if autosync:
            drv.Context.synchronize()
        if to_cpu:
            return y.get()
        else:
            return y

    def mm(self, B, transA=CUSPARSE_OPERATION_NON_TRANSPOSE, alpha=1.0,
           beta=0.0, C=None, ldb=None, ldc=None, check_inputs=True,
           to_cpu=False, autosync=True, handle=None, stream=None):
        """ multiplication by dense matrix B:  C = alpha*transA(A)*B + beta*C.
        """
        m, k = self.shape
        # try moving list or numpy array to GPU
        if not isinstance(B, pycuda.gpuarray.GPUArray):
            B = np.atleast_2d(B)  # .astype(self.dtype)
            B = gpuarray.to_gpu(B, allocator=self._alloc)
        n = B.shape[1]
        if handle is None:
            handle = self.handle
        if stream is not None:
            cusparseSetStream(handle, stream.handle)
        C = csrmm(m=m, n=n, k=k, descrA=self.descr, csrValA=self.Val,
                  csrRowPtrA=self.RowPtr, csrColIndA=self.ColInd,  B=B,
                  handle=handle, C=C, transA=transA, alpha=alpha, beta=beta,
                  ldb=ldb, ldc=ldc, check_inputs=check_inputs)
        if autosync:
            drv.Context.synchronize()
        if to_cpu:
            return C.get()
        else:
            return C

    @defineIf(toolkit_version >= (5, 5, 0))
    def mm2(self, B, transA=CUSPARSE_OPERATION_NON_TRANSPOSE,
            transB=CUSPARSE_OPERATION_NON_TRANSPOSE, alpha=1.0, beta=0.0,
            C=None, ldb=None, ldc=None, check_inputs=True, to_cpu=False,
            autosync=True, handle=None, stream=None):
        """ multiplication by dense matrix B:  C = alpha*transA(A)*B + beta*C.
        version 2
        """
        if toolkit_version < (5, 5, 0):
            raise ImportError("mm2 not implemented prior to CUDA v5.5")
        m, k = self.shape
        # try moving list or numpy array to GPU
        if not isinstance(B, pycuda.gpuarray.GPUArray):
            B = np.atleast_2d(B)  # .astype(self.dtype)
            B = gpuarray.to_gpu(B, allocator=self._alloc)
        if transB == CUSPARSE_OPERATION_NON_TRANSPOSE:
            n = B.shape[1]
        else:
            n = B.shape[0]
        if handle is None:
            handle = self.handle
        if stream is not None:
            cusparseSetStream(handle, stream.handle)
        C = csrmm2(handle=handle, m=m, n=n, k=k, descrA=self.descr,
                   csrValA=self.Val, csrRowPtrA=self.RowPtr,
                   csrColIndA=self.ColInd, B=B, C=C, transA=transA,
                   transB=transB, alpha=alpha, beta=beta, ldb=ldb, ldc=ldc,
                   check_inputs=check_inputs)
        if autosync:
            drv.Context.synchronize()
        if to_cpu:
            return C.get()
        else:
            return C

    @defineIf(toolkit_version >= (5, 0, 0))
    def geam(self, B, alpha=1.0, beta=1.0, check_inputs=True, autosync=True,
             handle=None, stream=None):
        """ addition of sparse matrix B:  C = alpha*A + beta*B """
        if toolkit_version < (5, 0, 0):
            raise ImportError("geam not implemented prior to CUDA v5.0")
        m, n = self.shape
        if not isinstance(B, CSR):
            # try converting B to cuSPARSE CSR
            B = CSR.to_CSR(B, handle=self.handle)
        if self.shape != B.shape:
            raise ValueError("Incompatible shapes")
        if handle is None:
            handle = self.handle
        if stream is not None:
            cusparseSetStream(handle, stream.handle)
        descrC, ValC, RowPtrC, ColIndC = csrgeam(
            handle=handle, m=m, n=n, descrA=self.descr, csrValA=self.Val,
            csrRowPtrA=self.RowPtr, csrColIndA=self.ColInd, descrB=B.descr,
            csrValB=B.Val, csrRowPtrB=B.RowPtr, csrColIndB=B.ColInd,
            alpha=alpha, beta=beta, nnzA=self.nnz, nnzB=B.nnz,
            check_inputs=True)
        C = CSR(descr=descrC, csrVal=ValC, csrRowPtr=RowPtrC,
                csrColInd=ColIndC, shape=self.shape, handle=self.handle)
        if autosync:
            drv.Context.synchronize()
        return C

    @defineIf(toolkit_version >= (5, 0, 0))
    def gemm(self, B, transA=CUSPARSE_OPERATION_NON_TRANSPOSE,
             transB=CUSPARSE_OPERATION_NON_TRANSPOSE, check_inputs=True,
             autosync=True, handle=None, stream=None):
        """ multiplication by sparse matrix B:  C = transA(A) * transB(B) """
        if toolkit_version < (5, 0, 0):
            raise ImportError("gemm not implemented prior to CUDA v5.0")

        if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
            m, k = self.shape
        else:
            k, m = self.shape
        if not isinstance(B, CSR):
            # try converting B to cuSPARSE CSR
            B = CSR.to_CSR(B, handle=self.handle)
        if transB == CUSPARSE_OPERATION_NON_TRANSPOSE:
            n = B.shape[1]
        else:
            n = B.shape[0]
        if handle is None:
            handle = self.handle
        if stream is not None:
            cusparseSetStream(handle, stream.handle)
        descrC, ValC, RowPtrC, ColIndC = csrgemm(
            handle=handle, m=m, n=n, k=k, descrA=self.descr,
            csrValA=self.Val, csrRowPtrA=self.RowPtr, csrColIndA=self.ColInd,
            descrB=B.descr, csrValB=B.Val, csrRowPtrB=B.RowPtr,
            csrColIndB=B.ColInd, nnzA=self.nnz, nnzB=B.nnz, transA=transA,
            transB=transB, check_inputs=True)
        if autosync:
            drv.Context.synchronize()
        C = CSR(descr=descrC, csrVal=ValC, csrRowPtr=RowPtrC,
                csrColInd=ColIndC, shape=(m, n), handle=self.handle)
        return C

    """
    start of: subset of methods in scipy.sparse.compressed._cs_matrix
    """
    @property
    def A(self):
        "The transpose operator."
        return self.todense()

    @property
    def T(self):
        "The transpose operator."
        return self.transpose()

    @property
    def H(self):
        "The adjoint operator."
        return self.getH()

    @property
    def real(self):
        "The transpose operator."
        return self._real()

    @property
    def imag(self):
        "The transpose operator."
        return self._imag()

    @property
    def size(self):
        "The adjoint operator."
        return self.getnnz()

    @property
    def nbytes(self):
        """ approximate object size in bytes (size of data,
            column indices and row pointers only). """
        nbytes = self.data.nbytes + self.indptr.nbytes + self.indices.nbytes
        return nbytes

    def transpose(self):
        m, n = self.shape
        # use csr2csc to perform the transpose
        cscVal, cscColPtr, cscRowInd = csr2csc(
            m, n, self.Val, self.RowPtr, self.ColInd, handle=self.handle,
            nnz=self.nnz)
        drv.Context.synchronize()
        return CSR(copyMatDescr(self.descr), cscVal, cscColPtr,
                   cscRowInd, self.shape, handle=self.handle)

    def getH(self):
        return self.transpose().conj()

    def conjugate(self):
        return self.conj()

    # implement _with_data similar to scipy.sparse.data._data_matrix
    def _with_data(self, data, copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the structure arrays
        (i.e. .indptr and .indices) are copied.
        """
        if copy:
            return self.__class__(copyMatDescr(self.descr), data,
                                  self.RowPtr.copy(), self.ColInd.copy(),
                                  self.shape, handle=self.handle)
        else:
            return self.__class__(self.descr, data,
                                  self.RowPtr, self.ColInd,
                                  self.shape, handle=self.handle)

    """
    end of: subset of methods in scipy.sparse.compressed._cs_matrix
    """

    """
    start of: subset of methods in scipy.sparse.data._data_matrix
    """
    def conj(self):
        return self._with_data(self.data.conj())

    def _real(self):
        return self._with_data(self.data.real)

    def _imag(self):
        return self._with_data(self.data.imag)

    def __abs__(self):
        return self._with_data(abs(self.data))

    def __neg__(self):
        return self._with_data(abs(self.data))

    def __imul__(self, other):  # self *= other
        if isscalarlike(other):
            self.data *= other
            return self
        else:
            return NotImplemented

    def __itruediv__(self, other):  # self /= other
        if isscalarlike(other):
            recip = 1.0 / other
            self.data *= recip
            return self
        else:
            return NotImplemented

    def astype(self, t):
        return self._with_data(self.data.astype(t))

    def copy(self):
        return self._with_data(self.data.copy(), copy=True)

    def _mul_scalar(self, other):
        return self._with_data(self.data * other)

    """
    end of: subset of methods in scipy.sparse.data._data_matrix
    """

    def __del__(self):
        """ cleanup descriptor upon object deletion """
        cusparseDestroyMatDescr(self.descr)
        # don't destroy the handle as other objects may be using it

    def __repr__(self):
        rstr = "CSR matrix:\n"
        rstr += "\tshape = {}\n".format(self.shape)
        rstr += "\tdtype = {}\n".format(self.dtype)
        rstr += "\tMatrixType = {}\n".format(self.matrix_type)
        rstr += "\tIndexBase = {}\n".format(self.index_base)
        rstr += "\tDiagType = {}\n".format(self.diag_type)
        rstr += "\tFillMode = {}\n".format(self.fill_mode)
        rstr += "\tcontext = {}\n\n".format(self.handle)
        rstr += "\tnnz = {}\n".format(self.nnz)
        rstr += "\tRowPtr = {}\n".format(self.RowPtr)
        rstr += "\tVal = {}\n".format(self.Val)
        return rstr
