from __future__ import division

from scikits.cuda.cusparse import *
from scikits.cuda.cusparse import (_csrgeamNnz, _csrgemmNnz)

from scikits.cuda import cusparse
cusparse.init()

import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_almost_equal

from unittest import skipIf

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

@skipIf(toolkit_version < (4, 1, 0), "HyB format added in CUDA Toolkit v4.1")
def test_create_destroy_hyb():
    HybA = cusparseCreateHybMat()
    cusparseDestroyHybMat(HybA)


@skipIf(toolkit_version < (4, 0, 0), "Analysis Info added in CUDA v4.0")
def test_create_destroy_AnalysisInfo():
    info = cusparseCreateSolveAnalysisInfo()
    cusparseDestroySolveAnalysisInfo(info)


@skipIf(toolkit_version < (6, 0, 0),
        "additional solve info types introduced in CUDA v6.0")
def test_create_destroy_SolveInfo_v6():
    # CSR cases
    info = cusparse.cusparseCreateCsrsv2Info()
    cusparse.cusparseDestroyCsrsv2Info(info)

    info = cusparse.cusparseCreateCsrilu02Info()
    cusparse.cusparseDestroyCsrilu02Info(info)

    info = cusparse.cusparseCreateCsric02Info()
    cusparse.cusparseDestroyCsric02Info(info)

    # BSR cases
    info = cusparse.cusparseCreateBsrsv2Info()
    cusparse.cusparseDestroyBsrsv2Info(info)

    info = cusparse.cusparseCreateBsrilu02Info()
    cusparse.cusparseDestroyBsrilu02Info(info)

    info = cusparse.cusparseCreateBsric02Info()
    cusparse.cusparseDestroyBsric02Info(info)


@skipIf(toolkit_version < (6, 5, 0),
        "additional solve info types introduced in CUDA v6.5")
def test_create_destroy_SolveInfo_v65():
    info = cusparse.cusparseCreateBsrsm2Info()
    cusparse.cusparseDestroyBsrsm2Info(info)

@skipIf(toolkit_version < (7, 0, 0), "ColorInfo introduced in CUDA v7.0")
def test_create_destroy_ColorInfo():
    # CSR cases
    info = cusparse.cusparseCreateColorInfo()
    cusparse.cusparseDestroyColorInfo(info)


def test_set_stream():
    handle = cusparseCreate()
    stream = drv.Stream()
    try:
        cusparseSetStream(handle, stream.handle)
    finally:
        cusparseDestroy(handle)

@skipIf(toolkit_version < (4, 1, 0), "skip for CUDA < v4.1")
def test_get_set_PointerMode():
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

    descrA = cusparseCreateMatDescr()
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)

    # loop over all directions and dtypes
    try:
        cusparse_dtypes = [np.float32, np.float64, np.complex64, np.complex128]
        for dirA in [CUSPARSE_DIRECTION_ROW, CUSPARSE_DIRECTION_COLUMN]:
            for dtype in cusparse_dtypes:
                nnzRowCol, nnzTotal = dense_nnz(
                    descrA, A.astype(dtype), dirA=dirA)
                assert nnzTotal == 5
                if dirA == CUSPARSE_DIRECTION_ROW:
                    assert_equal(nnzRowCol.get(), [3, 0, 1, 1])
                else:
                    assert_equal(nnzRowCol.get(), [1, 2, 2])
    finally:
        cusparseDestroyMatDescr(descrA)


def test_dense2csr_csr2dense():
    A = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    m, n = A.shape
    for dtype in cusparse_dtypes:
        A = A.astype(dtype)
        A_csr_scipy = scipy.sparse.csr_matrix(A)
        (descrA, csrValA, csrRowPtrA, csrColIndA) = dense2csr(A)
        try:
            assert_equal(csrValA.get(), A_csr_scipy.data)
            assert_equal(csrRowPtrA.get(), A_csr_scipy.indptr)
            assert_equal(csrColIndA.get(), A_csr_scipy.indices)

            A_dense = csr2dense(m, n, descrA, csrValA, csrRowPtrA, csrColIndA)
            assert_equal(A, A_dense.get())
        finally:
            # release descrA that was generated within dense2csr
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
        (descrA, cscValA, cscColPtrA, cscRowIndA) = dense2csc(A)
        try:
            assert_equal(cscValA.get(), A_csc_scipy.data)
            assert_equal(cscColPtrA.get(), A_csc_scipy.indptr)
            assert_equal(cscRowIndA.get(), A_csc_scipy.indices)

            A_dense = csc2dense(m, n, descrA, cscValA, cscColPtrA, cscRowIndA)
            assert_equal(A_cpu, A_dense.get())
        finally:
            # release descrA that was generated within dense2csc
            cusparseDestroyMatDescr(descrA)


def test_csr2csc_csc2csr():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]], order='F')
    m, n = A_cpu.shape

    for dtype in cusparse_dtypes:
        A = gpuarray.to_gpu(A_cpu.astype(dtype))
        A_csr_scipy = scipy.sparse.csr_matrix(A_cpu)
        A_csc_scipy = A_csr_scipy.tocsc()
        (descr, csrVal, csrRowPtr, csrColInd) = dense2csr(A)

        try:
            cscVal, cscColPtr, cscRowInd = csr2csc(m, n, csrVal, csrRowPtr,
                                                   csrColInd)
            # verify match to scipy
            assert_equal(cscVal.get(), A_csc_scipy.data)
            assert_equal(cscColPtr.get(), A_csc_scipy.indptr)
            assert_equal(cscRowInd.get(), A_csc_scipy.indices)

            # repeat for inverse operation
            csrVal, csrRowPtr, csrColInd = csc2csr(n, m, cscVal, cscColPtr,
                                                   cscRowInd)
            # verify match to scipy
            assert_equal(csrVal.get(), A_csr_scipy.data)
            assert_equal(csrRowPtr.get(), A_csr_scipy.indptr)
            assert_equal(csrColInd.get(), A_csr_scipy.indices)

        finally:
            cusparseDestroyMatDescr(descr)


def test_csr2coo_coo2csr():
    A_cpu = np.asarray([[1, 0, 0], [0, 2, 0], [3, 0, 4], [0, 0, 5]], order='F')
    m, n = A_cpu.shape

    for dtype in cusparse_dtypes:
        A = gpuarray.to_gpu(A_cpu.astype(dtype))
        (descr, csrVal, csrRowPtr, csrColInd) = dense2csr(A)
        try:
            nnz = csrVal.size
            cscVal, cscColPtr, cscRowInd = csr2csc(m, n, csrVal, csrRowPtr,
                                                   csrColInd)

            cooRowInd = csr2coo(csrRowPtr, nnz)

            # couldn't compare to scipy due to different ordering, so check the
            # values directly
            vals = csrVal.get()
            rows = cooRowInd.get()
            cols = csrColInd.get()
            for idx in range(nnz):
                assert A_cpu[rows[idx], cols[idx]] == vals[idx]

            # repeat for inverse operation
            csrRowPtr_v2 = coo2csr(cooRowInd, m)
            assert_equal(csrRowPtr_v2.get(), csrRowPtr.get())

        finally:
            cusparseDestroyMatDescr(descr)


def test_csc2coo_coo2csc():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]], order='F')
    m, n = A_cpu.shape

    for dtype in cusparse_dtypes:
        A = gpuarray.to_gpu(A_cpu.astype(dtype))
        (descr, csrVal, csrRowPtr, csrColInd) = dense2csr(A)

        try:
            nnz = csrVal.size

            cscVal, cscColPtr, cscRowInd = csr2csc(m, n, csrVal, csrRowPtr,
                                                   csrColInd)
            cooColInd = csc2coo(cscColPtr, nnz)
            # couldn't compare to scipy due to different ordering, so check the
            # values directly
            vals = csrVal.get()
            rows = cscRowInd.get()
            cols = cooColInd.get()
            for idx in range(nnz):
                assert A_cpu[rows[idx], cols[idx]] == vals[idx]

            # repeat for inverse operation
            cscColPtr_v2 = coo2csc(cooColInd, n)
            assert_equal(cscColPtr_v2.get(), cscColPtr.get())

        finally:
            cusparseDestroyMatDescr(descr)


def test_csrmv():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    # A = gpuarray.to_gpu(A_cpu)

    descrA = cusparseCreateMatDescr()
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)

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
                y = csrmv(descrA, csrValA, csrRowPtrA, csrColIndA, m,
                          n, x, transA=transA, alpha=alpha, beta=beta)
                y_cpu = y.get()

                # repeat, but pass in previous y with beta = 1.0
                beta = 1.0
                y = csrmv(descrA, csrValA, csrRowPtrA, csrColIndA, m,
                          n, x, transA=transA, alpha=alpha, beta=beta, y=y)
                y_cpu2 = y.get()
                if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
                    assert_almost_equal(y_cpu, [2., 2., 4., 6.])
                    assert_almost_equal(y_cpu2, 2 * y_cpu)
                else:
                    assert_almost_equal(y_cpu, [4., 2., 8.])
                    assert_almost_equal(y_cpu2, 2 * y_cpu)
    finally:
        cusparseDestroyMatDescr(descrA)


def test_csrmm():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])

    descrA = cusparseCreateMatDescr()
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)

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
                C = csrmm(m, n, k, descrA, csrValA, csrRowPtrA, csrColIndA, B,
                          transA=transA, alpha=alpha, beta=beta)
                C_cpu = C.get()
                assert_almost_equal(C_cpu, expected_result)

                # repeat, but pass in previous y with beta = 1.0
                beta = 1.0
                C = csrmm(m, n, k, descrA, csrValA, csrRowPtrA, csrColIndA, B,
                          C=C, transA=transA, alpha=alpha, beta=beta)
                C_cpu2 = C.get()
                assert_almost_equal(C_cpu2, 2*expected_result)
    finally:
        cusparseDestroyMatDescr(descrA)


@skipIf(toolkit_version < (5, 5, 0), "skip for CUDA < v5.5")
def test_csrmm2():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])

    descrA = cusparseCreateMatDescr()
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)

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
                    C = csrmm2(m, n, k, descrA, csrValA, csrRowPtrA,
                               csrColIndA, B, transA=transA, transB=transB,
                               alpha=alpha, beta=beta)
                    C_cpu = C.get()
                    assert_almost_equal(C_cpu, expected_result)

                    # repeat, but pass in previous y with beta = 1.0
                    beta = 1.0
                    C = csrmm2(m, n, k, descrA, csrValA, csrRowPtrA,
                               csrColIndA, B, C=C, transA=transA,
                               transB=transB, alpha=alpha, beta=beta)
                    C_cpu2 = C.get()
                    assert_almost_equal(C_cpu2, 2*expected_result)
    finally:
        cusparseDestroyMatDescr(descrA)


@skipIf(toolkit_version < (5, 0, 0), "skip for CUDA < v5.0")
def test_csrgeamNnz():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    A_cpu = scipy.sparse.csr_matrix(A_cpu)
    B_cpu = np.asarray([[0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]])
    B_cpu = scipy.sparse.csr_matrix(B_cpu)
    C_cpu = A_cpu + B_cpu
    nnz_expected = C_cpu.getnnz()

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
        m, n, descrA, csrRowPtrA, csrColIndA, descrB, csrRowPtrB, csrColIndB,
        check_inputs=True)
    cusparseDestroyMatDescr(descrC)
    assert_equal(nnzC, nnz_expected)

    # now test cases with preallocated matrix descrC & csrrowPtrC
    descrC = cusparseCreateMatDescr()
    try:
        cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL)
        csrRowPtrC = gpuarray.to_gpu(np.zeros((m+1, ), dtype=np.int32))
        nnzC = _csrgeamNnz(m, n, descrA, csrRowPtrA, csrColIndA, descrB,
                           csrRowPtrB, csrColIndB, descrC=descrC,
                           csrRowPtrC=csrRowPtrC, nnzA=None, nnzB=None,
                           check_inputs=True)
        assert_equal(nnzC, nnz_expected)
    finally:
        cusparseDestroyMatDescr(descrC)


@skipIf(toolkit_version < (5, 0, 0), "skip for CUDA < v5.0")
def test_csrgeam():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    A_cpu = scipy.sparse.csr_matrix(A_cpu)
    B_cpu = np.asarray([[0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]])
    B_cpu = scipy.sparse.csr_matrix(B_cpu)

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
                    m, n, descrA, csrValA, csrRowPtrA, csrColIndA, descrB,
                    csrValB, csrRowPtrB, csrColIndB, alpha=alpha, beta=beta)
                assert_almost_equal(csrValC.get(), C_cpu.data)
            finally:
                cusparseDestroyMatDescr(descrC)
    finally:
        cusparseDestroyMatDescr(descrA)
        cusparseDestroyMatDescr(descrB)


@skipIf(toolkit_version < (5, 0, 0), "skip for CUDA < v5.0")
def test_csrgemmNnz():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    # A = gpuarray.to_gpu(A_cpu)

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
        m, n, k, descrA, csrRowPtrA, csrColIndA, descrB, csrRowPtrB,
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
            nnzC = _csrgemmNnz(m, n, k, descrA, csrRowPtrA, csrColIndA,
                               descrB, csrRowPtrB, csrColIndB, descrC=descrC,
                               csrRowPtrC=csrRowPtrC, nnzA=None, nnzB=None,
                               transA=transA, transB=transB, check_inputs=True)
            if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
                assert nnzC == 8
                assert_equal(csrRowPtrC.get(), [0, 2, 3, 6, 8])
            else:
                assert nnzC == 5
                assert_equal(csrRowPtrC.get(), [0, 2, 3, 5])
    finally:
        cusparseDestroyMatDescr(descrA)
        cusparseDestroyMatDescr(descrB)
        cusparseDestroyMatDescr(descrC)


@skipIf(toolkit_version < (5, 0, 0), "skip for CUDA < v5.0")
def test_csrgemm():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    # A = gpuarray.to_gpu(A_cpu)

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

                descrC, csrValC, csrRowPtrC, csrColIndC = csrgemm(
                    m, n, k, descrA, csrValA, csrRowPtrA,
                    csrColIndA, descrB, csrValB, csrRowPtrB, csrColIndB,
                    transA=transA, transB=transB)
                cusparseDestroyMatDescr(descrC)
                if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
                    assert_almost_equal(csrValC.get(),
                                        [1, 1, 1, 1, 2, 3, 3, 9])
                else:
                    assert_almost_equal(csrValC.get(), [2, 1, 1, 1, 10])


    finally:
        cusparseDestroyMatDescr(descrA)
        cusparseDestroyMatDescr(descrB)


def test_CSR_construction():
    n = 64
    h = cusparseCreate()
    try:
        for dtype in cusparse_dtypes:
            A = 2*np.eye(n)

            A_scipy_csr = scipy.sparse.csr_matrix(A)

            # generate a CSR matrix from a dense gpuarray
            A_CSR = CSR.to_CSR(gpuarray.to_gpu(A), h)
            assert_equal(A_CSR.Val.get(), A_scipy_csr.data)

            # generate a CSR matrix from a dense numpy array
            A_CSR = CSR.to_CSR(A, h)
            assert_equal(A_CSR.Val.get(), A_scipy_csr.data)

            # generate a CSR matrix from a list of lists
            A_CSR = CSR.to_CSR(A.tolist(), h)
            assert_equal(A_CSR.Val.get(), A_scipy_csr.data)

            # generate a csSPARSE CSR matrix from a scipy.sparse CSR matrix
            A_CSR = CSR.to_CSR(A_scipy_csr, h)
            assert_equal(A_CSR.Val.get(), A_scipy_csr.data)

            # generate a csSPARSE CSR matrix from a scipy.sparse BSR matrix
            A_CSR = CSR.to_CSR(scipy.sparse.bsr_matrix(A), h)
            assert_equal(A_CSR.Val.get(), A_scipy_csr.data)

            # generate a csSPARSE CSR matrix from a scipy.sparse COO matrix
            A_CSR = CSR.to_CSR(scipy.sparse.coo_matrix(A), h)
            assert_equal(A_CSR.Val.get(), A_scipy_csr.data)

            # generate a csSPARSE CSR matrix from a scipy.sparse CSC matrix
            A_CSR = CSR.to_CSR(scipy.sparse.csc_matrix(A), h)
            assert_equal(A_CSR.Val.get(), A_scipy_csr.data)

            # generate a csSPARSE CSR matrix from a scipy.sparse DIA matrix
            A_CSR = CSR.to_CSR(scipy.sparse.dia_matrix(A), h)
            assert_equal(A_CSR.Val.get(), A_scipy_csr.data)

            # generate a csSPARSE CSR matrix from a scipy.sparse DOK matrix
            A_CSR = CSR.to_CSR(scipy.sparse.dok_matrix(A), h)
            assert_equal(A_CSR.Val.get(), A_scipy_csr.data)
    finally:
        cusparseDestroy(h)


def test_CSR_properties():
    n = 64
    h = cusparseCreate()
    A = 2*np.eye(n)

    # generate a CSR matrix from a dense numpy array
    A_CSR = CSR.to_CSR(A, h)

    # test shape
    assert A_CSR.shape == A.shape

    # test matrix_type property
    assert A_CSR.matrix_type == CUSPARSE_MATRIX_TYPE_GENERAL
    A_CSR.matrix_type = CUSPARSE_MATRIX_TYPE_SYMMETRIC
    assert A_CSR.matrix_type == CUSPARSE_MATRIX_TYPE_SYMMETRIC
    assert cusparseGetMatType(A_CSR.descr) == CUSPARSE_MATRIX_TYPE_SYMMETRIC

    # test index_base property
    assert A_CSR.index_base == CUSPARSE_INDEX_BASE_ZERO
    A_CSR.index_base = CUSPARSE_INDEX_BASE_ONE
    assert A_CSR.index_base == CUSPARSE_INDEX_BASE_ONE
    assert cusparseGetMatIndexBase(A_CSR.descr) == CUSPARSE_INDEX_BASE_ONE

    # test diag_type property
    assert A_CSR.diag_type == CUSPARSE_DIAG_TYPE_NON_UNIT
    A_CSR.diag_type = CUSPARSE_DIAG_TYPE_UNIT
    assert A_CSR.diag_type == CUSPARSE_DIAG_TYPE_UNIT
    assert cusparseGetMatDiagType(A_CSR.descr) == CUSPARSE_DIAG_TYPE_UNIT

    # test fill_mode property
    assert A_CSR.fill_mode == CUSPARSE_FILL_MODE_LOWER
    A_CSR.fill_mode = CUSPARSE_FILL_MODE_UPPER
    assert A_CSR.fill_mode == CUSPARSE_FILL_MODE_UPPER
    assert cusparseGetMatFillMode(A_CSR.descr) == CUSPARSE_FILL_MODE_UPPER

    # verify setting value outside the valid range raises an exception
    def set_mat_type(A, t):
        A_CSR.matrix_type = t
    assert_raises(CUSPARSE_STATUS_INVALID_VALUE, set_mat_type, A_CSR, 100)

    # test get nnz
    assert A_CSR.nnz == n
    assert A_CSR.getnnz() == n

    # verify that nnz can't be set
    def set_nnz(A, nnz):
        A_CSR.nnz = nnz
    assert_raises(AttributeError, set_nnz, A_CSR, 5)


def test_CSR_properties2():
    n = 64
    h = cusparseCreate()
    try:
        for dtype in cusparse_complex_dtypes:
            A = 2*np.eye(n).astype(dtype)
            # add a couple of complex values
            A[0, 1] = 1j
            A[1, 0] = -1j
            A_CSR = CSR.to_CSR(A, h)
            orig_data = A_CSR.data.get()
            orig_indices = A_CSR.indices.get()
            orig_indptr = A_CSR.indptr.get()

            assert_equal(abs(A_CSR).data.get(), abs(orig_data))
            assert_equal(-A_CSR.data.get(), -orig_data)

            A_CSR *= 2
            assert_almost_equal(A_CSR.data.get(), 2*orig_data)

            # this test requires from __future__ import division in Python 2.x
            A_CSR /= 2  # A_CSR._with_data(A_CSR.data/2)
            assert_almost_equal(A_CSR.data.get(), orig_data)

            assert_equal(A_CSR.conj().data.get(), np.conj(orig_data))
            assert_equal(A_CSR.real.data.get(), orig_data.real)
            assert_equal(A_CSR.imag.data.get(), orig_data.imag)
            assert_equal(A_CSR.A.get(), A)
            assert_equal(A_CSR.T.A.get(), A.T)

            # make sure the above didn't modify the original data
            assert_equal(A_CSR.data.get(), orig_data)
            assert_equal(A_CSR.indices.get(), orig_indices)
            assert_equal(A_CSR.indptr.get(), orig_indptr)

            assert_equal(A_CSR.H.A.get(), A.T.conj())
            assert_equal(A_CSR.size, A_CSR.nnz)
    finally:
        cusparseDestroy(h)


def test_CSR_todense():
    n = 64
    h = cusparseCreate()
    try:
        dtype = np.float32
        A = 2*np.eye(n)

        # generate a CSR matrix from a dense numpy array
        A_CSR = CSR.to_CSR(A.astype(dtype), h)

        # convert cusparseCSR back to a dense matrix
        A_dense = A_CSR.todense(to_cpu=False)
        assert type(A_dense) == gpuarray.GPUArray
        assert_equal(A_dense.get(), A)

        A_dense = A_CSR.todense(to_cpu=True)
        assert type(A_dense) == np.ndarray
        assert_equal(A_dense, A)
    finally:
        cusparseDestroy(h)


def test_CSR_get():
    n = 64
    h = cusparseCreate()
    try:
        dtype = np.float32
        A = 2*np.eye(n)

        # generate a CSR matrix from a dense numpy array
        A_CSR = CSR.to_CSR(A.astype(dtype), h)

        # convert cusparseCSR back to a dense matrix
        csr_scipy = A_CSR.get()
        # assert_equal(A_CSR.data.get(), csr_scipy.data)
        # assert_equal(A_CSR.indices.get(), csr_scipy.indices)
        # assert_equal(A_CSR.indptr.get(), csr_scipy.indptr)

        assert_equal(csr_scipy.indptr, scipy.sparse.csr_matrix(A).indptr)
        assert_equal(csr_scipy.indices, scipy.sparse.csr_matrix(A).indices)
        assert_equal(csr_scipy.data, scipy.sparse.csr_matrix(A).data)
    finally:
        cusparseDestroy(h)


def test_CSR_mv():
    n = 64
    h = cusparseCreate()
    try:
        A = 2*np.eye(n)

        dtype = np.float32
        x_cpu = np.ones((n, ), dtype=dtype)
        for dtype in cusparse_dtypes:

            # generate a CSR matrix from a dense numpy array
            A_CSR = CSR.to_CSR(A.astype(dtype), h)
            x = gpuarray.to_gpu(x_cpu.astype(dtype))

            # test default operation
            y = A_CSR.mv(x)
            assert_almost_equal(y.get(), A.diagonal())
            # transpose will be the same for this matrix
            y = A_CSR.mv(x, transA=CUSPARSE_OPERATION_TRANSPOSE)
            assert_almost_equal(y.get(), A.diagonal())

            alpha = 5.0
            y = A_CSR.mv(x, alpha=alpha)
            assert_almost_equal(y.get(), alpha*A.diagonal())

            # repeat with non-zero beta and initialize with previous y
            beta = 2.5
            y = A_CSR.mv(x, alpha=alpha, beta=beta, y=y)
            assert_almost_equal(y.get(), (alpha+alpha*beta)*A.diagonal())
    finally:
        cusparseDestroy(h)


def test_CSR_mm():
    m = 64
    n = 10
    h = cusparseCreate()
    try:
        A = 2*np.eye(m)

        dtype = np.float32
        B_cpu = np.ones((m, n), dtype=dtype, order='F')
        for dtype in cusparse_dtypes:
            A = A.astype(dtype)

            # generate a CSR matrix from a dense numpy array
            A_CSR = CSR.to_CSR(A, h)
            B = gpuarray.to_gpu(B_cpu.astype(dtype))

            # test default operation
            C = A_CSR.mm(B)
            assert_almost_equal(C.get(), np.dot(A, B_cpu))
            # transpose will be the same for this matrix
            C = A_CSR.mm(B_cpu.astype(dtype), transA=CUSPARSE_OPERATION_TRANSPOSE)
            assert_almost_equal(C.get(), np.dot(A.T, B_cpu))

            alpha = 5.0
            C = A_CSR.mm(B, alpha=alpha)
            assert_almost_equal(C.get(), alpha*np.dot(A, B_cpu))

            # repeat with non-zero beta and initialize with previous C
            beta = 2.5
            C = A_CSR.mm(B, alpha=alpha, beta=beta, C=C)
            assert_almost_equal(C.get(), (alpha+alpha*beta)*np.dot(A, B_cpu))
    finally:
        cusparseDestroy(h)


@skipIf(toolkit_version < (5, 5, 0), "skip for CUDA < v5.5")
def test_CSR_mm2():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    n = 5
    alpha = 2.0
    h = cusparseCreate()
    try:
        for transB in trans_list[:-1]:
            for transA in trans_list:
                for dtype in cusparse_dtypes:
                    m, k = A_cpu.shape
                    if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
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
                        if transA == CUSPARSE_OPERATION_TRANSPOSE:
                            opA = A_cpu.T
                        else:
                            opA = np.conj(A_cpu).T

                    expected_result = alpha * np.dot(opA, opB)
                    B = gpuarray.to_gpu(B_cpu)

                    A_CSR = CSR.to_CSR(A_cpu.astype(dtype), h)

                    # test mutliplication without passing in C
                    beta = 0.0
                    C = A_CSR.mm2(B, transA=transA, transB=transB, alpha=alpha,
                                  beta=beta, to_cpu=False)
                    assert_almost_equal(C.get(), expected_result)

                    # repeat, but pass in previous y with beta = 1.0
                    beta = 1.0
                    C = A_CSR.mm2(B, transA=transA, transB=transB, alpha=alpha,
                                  beta=beta, C=C, to_cpu=True)
                    assert_almost_equal(C, (1.0+beta)*expected_result)

    finally:
        cusparseDestroy(h)


@skipIf(toolkit_version < (5, 0, 0), "skip for CUDA < v5.0")
def test_CSR_geam():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    B_cpu = np.asarray([[0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]])

    h = cusparseCreate()
    alpha = 0.3
    beta = 5.0

    try:
        for dtype in cusparse_dtypes:
            A_CSR = CSR.to_CSR(A_cpu.astype(dtype), h)
            B_CSR = CSR.to_CSR(B_cpu.astype(dtype), h)
            C_CSR = A_CSR.geam(B_CSR, alpha=alpha, beta=beta)

            # compute on CPU with numpy and compare
            C_cpu = alpha*A_cpu + beta*B_cpu
            assert_almost_equal(C_CSR.todense(to_cpu=True), C_cpu)
    finally:
        cusparseDestroy(h)


@skipIf(toolkit_version < (5, 0, 0), "skip for CUDA < v5.0")
def test_CSR_gemm():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])

    h = cusparseCreate()
    B_cpu = A_cpu.T
    try:
        for transA in trans_list:
            transB = transA

            if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
                opA = A_cpu
                opB = B_cpu
            else:
                opA = A_cpu.T
                opB = B_cpu.T

            for dtype in cusparse_dtypes:
                A_CSR = CSR.to_CSR(A_cpu.astype(dtype), h)
                B_CSR = CSR.to_CSR(B_cpu.astype(dtype), h)
                C_CSR = A_CSR.gemm(B_CSR, transA=transA, transB=transB)

                # compute on CPU with numpy and compare
                C_cpu = np.dot(opA, opB)
                assert_almost_equal(C_CSR.todense(to_cpu=True), C_cpu)
    finally:
        cusparseDestroy(h)
