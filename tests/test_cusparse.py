from __future__ import division

from scikits.cuda.cusparse import *
from scikits.cuda.cusparse import (_csrgeamNnz, _csrgemmNnz)

import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_almost_equal

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv

import scipy.sparse  # TODO: refactor to remove this

cusparse_real_dtypes = [np.float32, np.float64]
cusparse_complex_dtypes = [np.complex64, np.complex128]
cusparse_dtypes = cusparse_real_dtypes + cusparse_complex_dtypes
trans_list = [CUSPARSE_OPERATION_NON_TRANSPOSE,
              CUSPARSE_OPERATION_TRANSPOSE,
              CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE]


def test_context_create_destroy():
    handle = cusparseCreate()
    cusparseDestroy(handle)


def test_get_version():
    handle = cusparseCreate()
    try:
        version = cusparseGetVersion(handle)
        assert type(version) == int
    finally:
        cusparseDestroy(handle)


def test_create_destroy_hyb():
    # wrappers to functions added in CUDA Toolkit v5.5
    if toolkit_version < (4, 1, 0):
        # skip for old CUDA versions
        return
    HybA = cusparseCreateHybMat()
    cusparseDestroyHybMat(HybA)


def test_set_stream():
    handle = cusparseCreate()
    stream = drv.Stream()
    try:
        cusparseSetStream(handle, stream.handle)
    finally:
        cusparseDestroy(handle)


def test_get_set_PointerMode():
    if toolkit_version < (4, 1, 0):
        # skip for old CUDA versions
        return
    handle = cusparseCreate()
    try:
        # test default mode
        mode = cusparseGetPointerMode(handle)
        assert mode == CUSPARSE_POINTER_MODE_HOST

        # test setting/getting new mode
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE)
        mode = cusparseGetPointerMode(handle)
        assert mode == CUSPARSE_POINTER_MODE_DEVICE

        # can't set outside enumerated range
        assert_raises(CUSPARSE_STATUS_INVALID_VALUE, cusparseSetPointerMode,
                      handle, 2)
    finally:
        cusparseDestroy(handle)


def test_matrix_descriptor_create_get_set_destroy():
    # create matrix description
    descrA = cusparseCreateMatDescr()

    try:
        # get default values/set
        assert cusparseGetMatType(descrA) == CUSPARSE_MATRIX_TYPE_GENERAL
        assert cusparseGetMatDiagType(descrA) == CUSPARSE_DIAG_TYPE_NON_UNIT
        assert cusparseGetMatIndexBase(descrA) == CUSPARSE_INDEX_BASE_ZERO
        assert cusparseGetMatFillMode(descrA) == CUSPARSE_FILL_MODE_LOWER

        # test set/get new values
        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_HERMITIAN)
        assert cusparseGetMatType(descrA) == CUSPARSE_MATRIX_TYPE_HERMITIAN
        cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_UNIT)
        assert cusparseGetMatDiagType(descrA) == CUSPARSE_DIAG_TYPE_UNIT
        cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE)
        assert cusparseGetMatIndexBase(descrA) == CUSPARSE_INDEX_BASE_ONE
        cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_UPPER)
        assert cusparseGetMatFillMode(descrA) == CUSPARSE_FILL_MODE_UPPER

        # can't set outside enumerated range
        assert_raises(
            OverflowError, cusparseSetMatType, descrA, -1)
        assert_raises(
            CUSPARSE_STATUS_INVALID_VALUE, cusparseSetMatType, descrA, 100)
        assert_raises(
            CUSPARSE_STATUS_INVALID_VALUE, cusparseSetMatDiagType, descrA, 100)
        assert_raises(
            CUSPARSE_STATUS_INVALID_VALUE, cusparseSetMatIndexBase, descrA,
            100)
        assert_raises(
            CUSPARSE_STATUS_INVALID_VALUE, cusparseSetMatFillMode, descrA, 100)

        # OLD BEHAVIOR:  float input gets cast to int
        # cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL + 0.5)
        # assert cusparseGetMatType(descrA) == CUSPARSE_MATRIX_TYPE_GENERAL
        assert_raises(TypeError, cusparseSetMatType, descrA,
                      CUSPARSE_MATRIX_TYPE_GENERAL + 0.5)
    finally:
        cusparseDestroyMatDescr(descrA)


def test_dense_nnz():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    A = gpuarray.to_gpu(A_cpu)

    handle = cusparseCreate()
    descrA = cusparseCreateMatDescr()
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST)

    # loop over all directions and dtypes
    try:
        cusparse_dtypes = [np.float32, np.float64, np.complex64, np.complex128]
        for dirA in [CUSPARSE_DIRECTION_ROW, CUSPARSE_DIRECTION_COLUMN]:
            for dtype in cusparse_dtypes:
                nnzRowCol, nnzTotal = dense_nnz(
                    handle, descrA, A.astype(dtype), dirA=dirA)
                assert nnzTotal == 5
                if dirA == CUSPARSE_DIRECTION_ROW:
                    assert_equal(nnzRowCol.get(), [3, 0, 1, 1])
                else:
                    assert_equal(nnzRowCol.get(), [1, 2, 2])
    finally:
        cusparseDestroy(handle)
        cusparseDestroyMatDescr(descrA)


def test_dense2csr_csr2dense():
    A = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    m, n = A.shape
    for dtype in cusparse_dtypes:
        A = A.astype(dtype)
        A_csr_scipy = scipy.sparse.csr_matrix(A)
        (handle, descrA, csrValA, csrRowPtrA, csrColIndA) = dense2csr(A)
        try:
            assert_equal(csrValA.get(), A_csr_scipy.data)
            assert_equal(csrRowPtrA.get(), A_csr_scipy.indptr)
            assert_equal(csrColIndA.get(), A_csr_scipy.indices)

            A_dense = csr2dense(handle, m, n, descrA, csrValA, csrRowPtrA,
                                csrColIndA)
            assert_equal(A, A_dense.get())
        finally:
            # release handle, descrA that were generated within dense2csr
            cusparseDestroy(handle)
            cusparseDestroyMatDescr(descrA)


def test_dense2csc_csc2dense():
    # comparison to scipy ColPtr/RowInd currently known to fail
    # is this a bug or a different coordinate convention?
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]], order='F')
    m, n = A_cpu.shape
    for dtype in cusparse_dtypes:
        A_cpu = A_cpu.astype(dtype)
        A_csc_scipy = scipy.sparse.csc_matrix(A_cpu)
        A = gpuarray.to_gpu(A_cpu)
        (handle, descrA, cscValA, cscColPtrA, cscRowIndA) = dense2csc(A)
        try:
            assert_equal(cscValA.get(), A_csc_scipy.data)
            assert_equal(cscColPtrA.get(), A_csc_scipy.indptr)
            assert_equal(cscRowIndA.get(), A_csc_scipy.indices)

            A_dense = csc2dense(handle, m, n, descrA, cscValA, cscColPtrA,
                                cscRowIndA)
            assert_equal(A_cpu, A_dense.get())
        finally:
            # release handle, descrA that were generated within dense2csc
            cusparseDestroy(handle)
            cusparseDestroyMatDescr(descrA)


def test_csr2csc_csc2csr():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]], order='F')
    m, n = A_cpu.shape

    for dtype in cusparse_dtypes:
        A = gpuarray.to_gpu(A_cpu.astype(dtype))
        A_csr_scipy = scipy.sparse.csr_matrix(A_cpu)
        A_csc_scipy = A_csr_scipy.tocsc()
        (handle, descr, csrVal, csrRowPtr, csrColInd) = dense2csr(A)

        try:
            cscVal, cscColPtr, cscRowInd = csr2csc(handle, m, n, csrVal,
                                                   csrRowPtr, csrColInd)
            # verify match to scipy
            assert_equal(cscVal.get(), A_csc_scipy.data)
            assert_equal(cscColPtr.get(), A_csc_scipy.indptr)
            assert_equal(cscRowInd.get(), A_csc_scipy.indices)

            # repeat for inverse operation
            csrVal, csrRowPtr, csrColInd = csc2csr(handle, n, m, cscVal,
                                                   cscColPtr, cscRowInd)
            # verify match to scipy
            assert_equal(csrVal.get(), A_csr_scipy.data)
            assert_equal(csrRowPtr.get(), A_csr_scipy.indptr)
            assert_equal(csrColInd.get(), A_csr_scipy.indices)

        finally:
            cusparseDestroy(handle)
            cusparseDestroyMatDescr(descr)


def test_csr2coo_coo2csr():
    A_cpu = np.asarray([[1, 0, 0], [0, 2, 0], [3, 0, 4], [0, 0, 5]], order='F')
    m, n = A_cpu.shape

    for dtype in cusparse_dtypes:
        A = gpuarray.to_gpu(A_cpu.astype(dtype))
        (handle, descr, csrVal, csrRowPtr, csrColInd) = dense2csr(A)
        try:
            nnz = csrVal.size
            cscVal, cscColPtr, cscRowInd = csr2csc(handle, m, n, csrVal,
                                                   csrRowPtr, csrColInd)

            cooRowInd = csr2coo(handle, csrRowPtr, nnz)

            # couldn't compare to scipy due to different ordering, so check the
            # values directly
            vals = csrVal.get()
            rows = cooRowInd.get()
            cols = csrColInd.get()
            for idx in range(nnz):
                assert A_cpu[rows[idx], cols[idx]] == vals[idx]

            # repeat for inverse operation
            csrRowPtr_v2 = coo2csr(handle, cooRowInd, m)
            assert_equal(csrRowPtr_v2.get(), csrRowPtr.get())

        finally:
            cusparseDestroy(handle)
            cusparseDestroyMatDescr(descr)


def test_csc2coo_coo2csc():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]], order='F')
    m, n = A_cpu.shape

    for dtype in cusparse_dtypes:
        A = gpuarray.to_gpu(A_cpu.astype(dtype))
        (handle, descr, csrVal, csrRowPtr, csrColInd) = dense2csr(A)

        try:
            nnz = csrVal.size

            cscVal, cscColPtr, cscRowInd = csr2csc(handle, m, n, csrVal,
                                                   csrRowPtr, csrColInd)
            cooColInd = csc2coo(handle, cscColPtr, nnz)
            # couldn't compare to scipy due to different ordering, so check the
            # values directly
            vals = csrVal.get()
            rows = cscRowInd.get()
            cols = cooColInd.get()
            for idx in range(nnz):
                assert A_cpu[rows[idx], cols[idx]] == vals[idx]

            # repeat for inverse operation
            cscColPtr_v2 = coo2csc(handle, cooColInd, n)
            assert_equal(cscColPtr_v2.get(), cscColPtr.get())

        finally:
            cusparseDestroy(handle)
            cusparseDestroyMatDescr(descr)


def test_csrmv():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    # A = gpuarray.to_gpu(A_cpu)

    handle = cusparseCreate()
    descrA = cusparseCreateMatDescr()
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST)

    csr_numpy = scipy.sparse.csr_matrix(A_cpu)
    indptr = csr_numpy.indptr
    indices = csr_numpy.indices
    csr_data = csr_numpy.data

    csrRowPtrA = gpuarray.to_gpu(indptr)
    csrColIndA = gpuarray.to_gpu(indices)
    m, n = csr_numpy.shape
    alpha = 2.0
    # loop over all transpose operations and dtypes
    try:
        for transA in trans_list:
            for dtype in cusparse_dtypes:
                if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
                    x = gpuarray.to_gpu(np.ones((n, ), dtype=dtype))
                else:
                    x = gpuarray.to_gpu(np.ones((m, ), dtype=dtype))
                csrValA = gpuarray.to_gpu(csr_data.astype(dtype))

                # test mutliplication without passing in y
                beta = 0.0
                y = csrmv(handle, descrA, csrValA, csrRowPtrA, csrColIndA, m,
                          n, x, transA=transA, alpha=alpha, beta=beta)
                y_cpu = y.get()

                # repeat, but pass in previous y with beta = 1.0
                beta = 1.0
                y = csrmv(handle, descrA, csrValA, csrRowPtrA, csrColIndA, m,
                          n, x, transA=transA, alpha=alpha, beta=beta, y=y)
                y_cpu2 = y.get()
                if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
                    assert_almost_equal(y_cpu, [2., 2., 4., 6.])
                    assert_almost_equal(y_cpu2, 2 * y_cpu)
                else:
                    assert_almost_equal(y_cpu, [4., 2., 8.])
                    assert_almost_equal(y_cpu2, 2 * y_cpu)
    finally:
        cusparseDestroy(handle)
        cusparseDestroyMatDescr(descrA)


def test_csrmm():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])

    handle = cusparseCreate()
    descrA = cusparseCreateMatDescr()
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST)

    csr_numpy = scipy.sparse.csr_matrix(A_cpu)
    indptr = csr_numpy.indptr
    indices = csr_numpy.indices
    csr_data = csr_numpy.data

    csrRowPtrA = gpuarray.to_gpu(indptr)
    csrColIndA = gpuarray.to_gpu(indices)
    n = 5
    alpha = 2.0

    # loop over all transpose operations and dtypes
    try:
        for transA in trans_list:
            for dtype in cusparse_dtypes:
                csrValA = gpuarray.to_gpu(csr_data.astype(dtype))

                m, k = A_cpu.shape
                if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
                    B_cpu = np.ones((k, n), dtype=dtype, order='F')
                    expected_result = alpha * np.dot(A_cpu, B_cpu)
                else:
                    B_cpu = np.ones((m, n), dtype=dtype, order='F')
                    if transA == CUSPARSE_OPERATION_TRANSPOSE:
                        expected_result = alpha * np.dot(A_cpu.T, B_cpu)
                    else:
                        expected_result = alpha * np.dot(np.conj(A_cpu).T,
                                                         B_cpu)
                B = gpuarray.to_gpu(B_cpu)
                # test mutliplication without passing in C
                beta = 0.0
                C = csrmm(handle, m, n, k, descrA, csrValA, csrRowPtrA,
                          csrColIndA, B, transA=transA, alpha=alpha,
                          beta=beta)
                C_cpu = C.get()
                assert_almost_equal(C_cpu, expected_result)

                # repeat, but pass in previous y with beta = 1.0
                beta = 1.0
                C = csrmm(handle, m, n, k, descrA, csrValA, csrRowPtrA,
                          csrColIndA, B, C=C, transA=transA, alpha=alpha,
                          beta=beta)
                C_cpu2 = C.get()
                assert_almost_equal(C_cpu2, 2*expected_result)
    finally:
        cusparseDestroy(handle)
        cusparseDestroyMatDescr(descrA)


def test_csrmm2():
    if toolkit_version < (5, 5, 0):
        # skip for old CUDA versions
        return
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])

    handle = cusparseCreate()
    descrA = cusparseCreateMatDescr()
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST)

    csr_numpy = scipy.sparse.csr_matrix(A_cpu)
    indptr = csr_numpy.indptr
    indices = csr_numpy.indices
    csr_data = csr_numpy.data

    csrRowPtrA = gpuarray.to_gpu(indptr)
    csrColIndA = gpuarray.to_gpu(indices)
    n = 5
    alpha = 2.0
    try:
        for transB in trans_list[:-1]:
            for transA in trans_list:
                for dtype in cusparse_dtypes:
                    csrValA = gpuarray.to_gpu(csr_data.astype(dtype))

                    m, k = A_cpu.shape
                    if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
                        m, k = A_cpu.shape

                        if transB == CUSPARSE_OPERATION_NON_TRANSPOSE:
                            B_cpu = np.ones((k, n), dtype=dtype, order='F')
                            opB = B_cpu
                        else:
                            B_cpu = np.ones((n, k), dtype=dtype, order='F')
                            opB = B_cpu.T
                        opA = A_cpu
                    else:
                        if transB == CUSPARSE_OPERATION_NON_TRANSPOSE:
                            B_cpu = np.ones((m, n), dtype=dtype, order='F')
                            opB = B_cpu
                        else:
                            # cuSPARSE doesn't implement this case
                            continue
                            # B_cpu = np.ones((n, m), dtype=dtype, order='F')
                            # opB = B_cpu.T
                        if transA == CUSPARSE_OPERATION_TRANSPOSE:
                            opA = A_cpu.T
                        else:
                            opA = np.conj(A_cpu).T

                    expected_result = alpha * np.dot(opA, opB)
                    B = gpuarray.to_gpu(B_cpu)

                    # test mutliplication without passing in C
                    beta = 0.0
                    C = csrmm2(handle, m, n, k, descrA, csrValA, csrRowPtrA,
                               csrColIndA, B, transA=transA, transB=transB,
                               alpha=alpha, beta=beta)
                    C_cpu = C.get()
                    assert_almost_equal(C_cpu, expected_result)

                    # repeat, but pass in previous y with beta = 1.0
                    beta = 1.0
                    C = csrmm2(handle, m, n, k, descrA, csrValA, csrRowPtrA,
                               csrColIndA, B, C=C, transA=transA,
                               transB=transB, alpha=alpha, beta=beta)
                    C_cpu2 = C.get()
                    assert_almost_equal(C_cpu2, 2*expected_result)
    finally:
        cusparseDestroy(handle)
        cusparseDestroyMatDescr(descrA)


def test_csrgeamNnz():
    if toolkit_version < (5, 0, 0):
        # skip for old CUDA versions
        return
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    A_cpu = scipy.sparse.csr_matrix(A_cpu)
    B_cpu = np.asarray([[0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]])
    B_cpu = scipy.sparse.csr_matrix(B_cpu)
    C_cpu = A_cpu + B_cpu
    nnz_expected = C_cpu.getnnz()

    handle = cusparseCreate()
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST)

    descrA = cusparseCreateMatDescr()
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)

    descrB = cusparseCreateMatDescr()
    cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO)

    csrRowPtrA = gpuarray.to_gpu(A_cpu.indptr)
    csrColIndA = gpuarray.to_gpu(A_cpu.indices)

    csrRowPtrB = gpuarray.to_gpu(B_cpu.indptr)
    csrColIndB = gpuarray.to_gpu(B_cpu.indices)

    m, n = A_cpu.shape

    # test alternative case where descrC, csrRowPtrC not preallocated
    descrC, csrRowPtrC, nnzC = _csrgeamNnz(
        handle, m, n, descrA, csrRowPtrA, csrColIndA, descrB, csrRowPtrB,
        csrColIndB, check_inputs=True)
    cusparseDestroyMatDescr(descrC)
    assert_equal(nnzC, nnz_expected)

    # now test cases with preallocated matrix descrC & csrrowPtrC
    descrC = cusparseCreateMatDescr()
    try:
        cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL)
        csrRowPtrC = gpuarray.to_gpu(np.zeros((m+1, ), dtype=np.int32))
        nnzC = _csrgeamNnz(handle, m, n, descrA, csrRowPtrA, csrColIndA,
                           descrB, csrRowPtrB, csrColIndB, descrC,
                           csrRowPtrC, nnzA=None, nnzB=None,
                           check_inputs=True)
        assert_equal(nnzC, nnz_expected)
    finally:
        cusparseDestroyMatDescr(descrC)


def test_csrgeam():
    if toolkit_version < (5, 0, 0):
        # skip for old CUDA versions
        return
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    A_cpu = scipy.sparse.csr_matrix(A_cpu)
    B_cpu = np.asarray([[0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]])
    B_cpu = scipy.sparse.csr_matrix(B_cpu)

    handle = cusparseCreate()
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST)

    descrA = cusparseCreateMatDescr()
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)

    descrB = cusparseCreateMatDescr()
    cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO)

    csrRowPtrA = gpuarray.to_gpu(A_cpu.indptr)
    csrColIndA = gpuarray.to_gpu(A_cpu.indices)

    csrRowPtrB = gpuarray.to_gpu(B_cpu.indptr)
    csrColIndB = gpuarray.to_gpu(B_cpu.indices)

    m, n = A_cpu.shape

    alpha = 0.3
    beta = 5.0
    C_cpu = alpha*A_cpu + beta*B_cpu

    try:
        for dtype in cusparse_dtypes:
            csrValA = gpuarray.to_gpu(A_cpu.data.astype(dtype))
            csrValB = gpuarray.to_gpu(B_cpu.data.astype(dtype))
            try:
                descrC, csrValC, csrRowPtrC, csrColIndC = csrgeam(
                    handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA,
                    descrB, csrValB, csrRowPtrB, csrColIndB, alpha=alpha,
                    beta=beta)
                assert_almost_equal(csrValC.get(), C_cpu.data)
            finally:
                cusparseDestroyMatDescr(descrC)
    finally:
        cusparseDestroy(handle)
        cusparseDestroyMatDescr(descrA)
        cusparseDestroyMatDescr(descrB)


def test_csrgemmNnz():
    if toolkit_version < (5, 0, 0):
        # skip for old CUDA versions
        return
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    # A = gpuarray.to_gpu(A_cpu)

    handle = cusparseCreate()
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST)

    descrA = cusparseCreateMatDescr()
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)

    descrB = cusparseCreateMatDescr()
    cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO)

    csr_numpy = scipy.sparse.csr_matrix(A_cpu)
    indptr = csr_numpy.indptr
    indices = csr_numpy.indices

    B_cpu = csr_numpy.T.tocsr()

    csrRowPtrA = gpuarray.to_gpu(indptr)
    csrColIndA = gpuarray.to_gpu(indices)

    csrRowPtrB = gpuarray.to_gpu(B_cpu.indptr)
    csrColIndB = gpuarray.to_gpu(B_cpu.indices)

    m, k = A_cpu.shape
    n = B_cpu.shape[1]
    # test alternative case where descrC, csrRowPtrC not preallocated
    transA = transB = CUSPARSE_OPERATION_NON_TRANSPOSE
    descrC, csrRowPtrC, nnzC = _csrgemmNnz(
        handle, m, n, k, descrA, csrRowPtrA, csrColIndA, descrB, csrRowPtrB,
        csrColIndB, transA=transA, transB=transB, check_inputs=True)
    cusparseDestroyMatDescr(descrC)

    # now test cases with preallocated matrix description
    descrC = cusparseCreateMatDescr()
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL)

    # loop over all transpose operations and dtypes
    try:
        for transA in trans_list:
            transB = transA

            if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
                m, k = A_cpu.shape
            else:
                k, m = A_cpu.shape

            if transB == CUSPARSE_OPERATION_NON_TRANSPOSE:
                kB, n = B_cpu.shape
            else:
                n, kB = B_cpu.shape

            csrRowPtrC = gpuarray.to_gpu(np.zeros((m+1, ), dtype=np.int32))
            nnzC = _csrgemmNnz(handle, m, n, k, descrA, csrRowPtrA, csrColIndA,
                               descrB, csrRowPtrB, csrColIndB, descrC,
                               csrRowPtrC, nnzA=None, nnzB=None, transA=transA,
                               transB=transB, check_inputs=True)
            if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
                assert nnzC == 8
                assert_equal(csrRowPtrC.get(), [0, 2, 3, 6, 8])
            else:
                assert nnzC == 5
                assert_equal(csrRowPtrC.get(), [0, 2, 3, 5])
    finally:
        cusparseDestroy(handle)
        cusparseDestroyMatDescr(descrA)
        cusparseDestroyMatDescr(descrB)
        cusparseDestroyMatDescr(descrC)


def test_csrgemm():
    if toolkit_version < (5, 0, 0):
        # skip for old CUDA versions
        return
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    # A = gpuarray.to_gpu(A_cpu)

    handle = cusparseCreate()
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST)

    descrA = cusparseCreateMatDescr()
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)

    descrB = cusparseCreateMatDescr()
    cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO)

    csr_numpy = scipy.sparse.csr_matrix(A_cpu)
    indptr = csr_numpy.indptr
    indices = csr_numpy.indices
    csr_data = csr_numpy.data

    B_cpu = csr_numpy.T.tocsr()

    csrRowPtrA = gpuarray.to_gpu(indptr)
    csrColIndA = gpuarray.to_gpu(indices)

    csrRowPtrB = gpuarray.to_gpu(B_cpu.indptr)
    csrColIndB = gpuarray.to_gpu(B_cpu.indices)

    m, k = A_cpu.shape

    try:
        for transA in trans_list:
            transB = transA

            if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
                m, k = A_cpu.shape
            else:
                k, m = A_cpu.shape
            if transB == CUSPARSE_OPERATION_NON_TRANSPOSE:
                kB, n = B_cpu.shape
            else:
                n, kB = B_cpu.shape

            for dtype in cusparse_dtypes:
                csrValA = gpuarray.to_gpu(csr_data.astype(dtype))
                csrValB = gpuarray.to_gpu(B_cpu.data.astype(dtype))

                try:
                    descrC, csrValC, csrRowPtrC, csrColIndC = csrgemm(
                        handle, m, n, k, descrA, csrValA, csrRowPtrA,
                        csrColIndA, descrB, csrValB, csrRowPtrB, csrColIndB,
                        transA=transA, transB=transB)
                    if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
                        assert_almost_equal(csrValC.get(),
                                            [1, 1, 1, 1, 2, 3, 3, 9])
                    else:
                        assert_almost_equal(csrValC.get(), [2, 1, 1, 1, 10])
                finally:
                    cusparseDestroyMatDescr(descrC)
    finally:
        cusparseDestroy(handle)
        cusparseDestroyMatDescr(descrA)
        cusparseDestroyMatDescr(descrB)
