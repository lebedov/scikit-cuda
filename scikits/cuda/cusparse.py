#!/usr/bin/env python

"""
Python interface to CUSPARSE functions.

Note: this module does not explicitly depend on PyCUDA.
"""

import re
import struct
import numpy as np

import cffi

_ffi = cffi.FFI()
_ffi.cdef("""
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

typedef enum{
    CUSPARSE_STATUS_SUCCESS=0,
    CUSPARSE_STATUS_NOT_INITIALIZED=1,
    CUSPARSE_STATUS_ALLOC_FAILED=2,
    CUSPARSE_STATUS_INVALID_VALUE=3,
    CUSPARSE_STATUS_ARCH_MISMATCH=4,
    CUSPARSE_STATUS_MAPPING_ERROR=5,
    CUSPARSE_STATUS_EXECUTION_FAILED=6,
    CUSPARSE_STATUS_INTERNAL_ERROR=7,
    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED=8,
    CUSPARSE_STATUS_ZERO_PIVOT=9
} cusparseStatus_t;

struct cusparseContext;
typedef struct cusparseContext *cusparseHandle_t;

// Actual contents of following opaque structure:
// typedef struct cusparseMatDescr {
//     cusparseMatrixType_t MatrixType;
//     cusparseFillMode_t FillMode;
//     cusparseDiagType_t DiagType;
//     cusparseIndexBase_t IndexBase;
// } cusparseMatDescr_t;

struct cusparseMatDescr;
typedef struct cusparseMatDescr *cusparseMatDescr_t;

struct cusparseSolveAnalysisInfo;
typedef struct cusparseSolveAnalysisInfo *cusparseSolveAnalysisInfo_t;

/* Opaque structure holding the sparse triangular solve information */
struct csrsv2Info;
typedef struct csrsv2Info *csrsv2Info_t;

struct bsrsv2Info;
typedef struct bsrsv2Info *bsrsv2Info_t;

struct bsrsm2Info;
typedef struct bsrsm2Info *bsrsm2Info_t;

/* Opaque structure holding incomplete Cholesky information */
struct csric02Info;
typedef struct csric02Info *csric02Info_t;

struct bsric02Info;
typedef struct bsric02Info *bsric02Info_t;

/* Opaque structure holding incomplete LU information */
struct csrilu02Info;
typedef struct csrilu02Info *csrilu02Info_t;

struct bsrilu02Info;
typedef struct bsrilu02Info *bsrilu02Info_t;

/* Opaque structure holding the hybrid (HYB) storage information */
struct cusparseHybMat;
typedef struct cusparseHybMat *cusparseHybMat_t;

typedef enum { 
    CUSPARSE_POINTER_MODE_HOST = 0,  
    CUSPARSE_POINTER_MODE_DEVICE = 1        
} cusparsePointerMode_t;

typedef enum {
    CUSPARSE_ACTION_SYMBOLIC = 0,
    CUSPARSE_ACTION_NUMERIC = 1
} cusparseAction_t;

typedef enum {
    CUSPARSE_MATRIX_TYPE_GENERAL = 0,
    CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1,
    CUSPARSE_MATRIX_TYPE_HERMITIAN = 2,
    CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3
} cusparseMatrixType_t;

typedef enum {
    CUSPARSE_FILL_MODE_LOWER = 0,
    CUSPARSE_FILL_MODE_UPPER = 1
} cusparseFillMode_t;

typedef enum {
    CUSPARSE_DIAG_TYPE_NON_UNIT = 0,
    CUSPARSE_DIAG_TYPE_UNIT = 1
} cusparseDiagType_t;

typedef enum {
    CUSPARSE_INDEX_BASE_ZERO = 0,
    CUSPARSE_INDEX_BASE_ONE = 1
} cusparseIndexBase_t;

typedef enum {
    CUSPARSE_OPERATION_NON_TRANSPOSE = 0,
    CUSPARSE_OPERATION_TRANSPOSE = 1,
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
} cusparseOperation_t;

typedef enum {
    CUSPARSE_DIRECTION_ROW = 0,
    CUSPARSE_DIRECTION_COLUMN = 1
} cusparseDirection_t;

typedef enum {
    CUSPARSE_HYB_PARTITION_AUTO = 0,
    CUSPARSE_HYB_PARTITION_USER = 1,
    CUSPARSE_HYB_PARTITION_MAX = 2,
} cusparseHybPartition_t;

typedef enum {
    CUSPARSE_SOLVE_POLICY_NO_LEVEL = 0,
    CUSPARSE_SOLVE_POLICY_USE_LEVEL = 1  
} cusparseSolvePolicy_t;

// initialization and management routines
cusparseStatus_t cusparseCreate(cusparseHandle_t *handle);
cusparseStatus_t cusparseDestroy(cusparseHandle_t handle);
cusparseStatus_t cusparseGetVersion(cusparseHandle_t handle, int *version);
cusparseStatus_t cusparseSetStream(cusparseHandle_t handle,
                                   cudaStream_t streamId);

// matrix descriptor routines
cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t *descrA);
cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t descrA);

cusparseStatus_t cusparseSetMatType(cusparseMatDescr_t descrA,
                                    cusparseMatrixType_t type);
cusparseMatrixType_t cusparseGetMatType(const cusparseMatDescr_t descrA);

cusparseStatus_t cusparseSetMatFillMode(cusparseMatDescr_t descrA,
                                        cusparseFillMode_t fillMode);
cusparseFillMode_t cusparseGetMatFillMode(const cusparseMatDescr_t descrA);

cusparseStatus_t cusparseSetMatDiagType(cusparseMatDescr_t descrA,
                                        cusparseDiagType_t diagType);
cusparseDiagType_t cusparseGetMatDiagType(const cusparseMatDescr_t descrA);

cusparseStatus_t cusparseSetMatIndexBase(cusparseMatDescr_t descrA,
                                         cusparseIndexBase_t base);
cusparseIndexBase_t cusparseGetMatIndexBase(const cusparseMatDescr_t descrA);

/* sparse triangular solve */
cusparseStatus_t cusparseCreateSolveAnalysisInfo(cusparseSolveAnalysisInfo_t *info);
cusparseStatus_t cusparseDestroySolveAnalysisInfo(cusparseSolveAnalysisInfo_t info);
cusparseStatus_t cusparseGetLevelInfo(cusparseHandle_t handle, 
                                      cusparseSolveAnalysisInfo_t info,
                                      int *nlevels,
                                      int **levelPtr,
                                      int **levelInd);

cusparseStatus_t cusparseCreateCsrsv2Info(csrsv2Info_t *info);
cusparseStatus_t cusparseDestroyCsrsv2Info(csrsv2Info_t info);

/* incomplete Cholesky */
cusparseStatus_t cusparseCreateCsric02Info(csric02Info_t *info);
cusparseStatus_t cusparseDestroyCsric02Info(csric02Info_t info);

cusparseStatus_t cusparseCreateBsric02Info(bsric02Info_t *info);
cusparseStatus_t cusparseDestroyBsric02Info(bsric02Info_t info);

/* incomplete LU */
cusparseStatus_t cusparseCreateCsrilu02Info(csrilu02Info_t *info);
cusparseStatus_t cusparseDestroyCsrilu02Info(csrilu02Info_t info);

cusparseStatus_t cusparseCreateBsrilu02Info(bsrilu02Info_t *info);
cusparseStatus_t cusparseDestroyBsrilu02Info(bsrilu02Info_t info);

/* BSR triangular solber */
cusparseStatus_t cusparseCreateBsrsv2Info(bsrsv2Info_t *info);
cusparseStatus_t cusparseDestroyBsrsv2Info(bsrsv2Info_t info);

cusparseStatus_t cusparseCreateBsrsm2Info(bsrsm2Info_t *info);
cusparseStatus_t cusparseDestroyBsrsm2Info(bsrsm2Info_t info);

/* hybrid (HYB) format */
cusparseStatus_t cusparseCreateHybMat(cusparseHybMat_t *hybA);
cusparseStatus_t cusparseDestroyHybMat(cusparseHybMat_t hybA);

// level 1 routines
cusparseStatus_t cusparseSaxpyi(cusparseHandle_t handle,
                                int nnz,
                                const float* alpha,
                                const float *xVal,
                                const int *xInd,
                                float *y,
                                cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseDaxpyi(cusparseHandle_t handle,
                                int nnz,
                                const double *alpha,
                                const double *xVal,
                                const int *xInd,
                                double *y,
                                cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseCaxpyi(cusparseHandle_t handle,
                                int nnz,
                                const cuComplex *alpha,
                                const cuComplex *xVal,
                                const int *xInd,
                                cuComplex *y,
                                cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseZaxpyi(cusparseHandle_t handle,
                                int nnz,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *xVal,
                                const int *xInd,
                                cuDoubleComplex *y,
                                cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseSdoti(cusparseHandle_t handle,
                               int nnz,
                               const float *xVal,
                               const int *xInd,
                               const float *y,
                               float *resultHostPtr,
                               cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseDdoti(cusparseHandle_t handle,
                               int nnz,
                               const double *xVal,
                               const int *xInd,
                               const double *y,
                               double *resultHostPtr,
                               cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseCdoti(cusparseHandle_t handle,
                               int nnz,
                               const cuComplex *xVal,
                               const int *xInd,
                               const cuComplex *y,
                               cuComplex *resultHostPtr,
                               cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseZdoti(cusparseHandle_t handle,
                               int nnz,
                               const cuDoubleComplex *xVal,
                               const int *xInd,
                               const cuDoubleComplex *y,
                               cuDoubleComplex *resultHostPtr,
                               cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseCdotci(cusparseHandle_t handle,
                                int nnz,
                                const cuComplex *xVal,
                                const int *xInd,
                                const cuComplex *y,
                                cuComplex *resultHostPtr,
                                cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseZdotci(cusparseHandle_t handle,
                                int nnz,
                                const cuDoubleComplex *xVal,
                                const int *xInd,
                                const cuDoubleComplex *y,
                                cuDoubleComplex *resultHostPtr,
                                cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseSgthr(cusparseHandle_t handle,
                               int nnz,
                               const float *y,
                               float *xVal,
                               const int *xInd,
                               cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseDgthr(cusparseHandle_t handle,
                               int nnz,
                               const double *y,
                               double *xVal,
                               const int *xInd,
                               cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseCgthr(cusparseHandle_t handle,
                               int nnz,
                               const cuComplex *y,
                               cuComplex *xVal,
                               const int *xInd,
                               cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseZgthr(cusparseHandle_t handle,
                               int nnz,
                               const cuDoubleComplex *y,
                               cuDoubleComplex *xVal,
                               const int *xInd,
                               cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseSgthrz(cusparseHandle_t handle,
                                int nnz,
                                float *y,
                                float *xVal,
                                const int *xInd,
                                cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseDgthrz(cusparseHandle_t handle,
                                int nnz,
                                double *y,
                                double *xVal,
                                const int *xInd,
                                cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseCgthrz(cusparseHandle_t handle,
                                int nnz,
                                cuComplex *y,
                                cuComplex *xVal,
                                const int *xInd,
                                cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseZgthrz(cusparseHandle_t handle,
                                int nnz,
                                cuDoubleComplex *y,
                                cuDoubleComplex *xVal,
                                const int *xInd,
                                cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseSsctr(cusparseHandle_t handle,
                               int nnz,
                               const float *xVal,
                               const int *xInd,
                               float *y,
                               cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseDsctr(cusparseHandle_t handle,
                               int nnz,
                               const double *xVal,
                               const int *xInd,
                               double *y,
                               cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseCsctr(cusparseHandle_t handle,
                               int nnz,
                               const cuComplex *xVal,
                               const int *xInd,
                               cuComplex *y,
                               cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseZsctr(cusparseHandle_t handle,
                               int nnz,
                               const cuDoubleComplex *xVal,
                               const int *xInd,
                               cuDoubleComplex *y,
                               cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseSroti(cusparseHandle_t handle,
                               int nnz,
                               float* xVal,
                               const int *xInd,
                               float *y,
                               const float* c,
                               const float* s,
                               cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseDroti(cusparseHandle_t handle,
                               int nnz,
                               double *xVal,
                               const int *xInd,
                               double *y,
                               const double* c,
                               const double* s,
                               cusparseIndexBase_t idxBase);

// sparse format conversion
cusparseStatus_t cusparseSnnz(cusparseHandle_t handle,
                              cusparseDirection_t dirA,
                              int m,
                              int n,
                              const cusparseMatDescr_t  descrA,
                              const float *A,
                              int lda,
                              int *nnzPerRowCol,
                              int *nnzTotalHostPtr);

cusparseStatus_t cusparseDnnz(cusparseHandle_t handle,
                              cusparseDirection_t dirA,
                              int m,
                              int n,
                              const cusparseMatDescr_t  descrA,
                              const double *A,
                              int lda,
                              int *nnzPerRowCol,
                              int *nnzTotalHostPtr);

cusparseStatus_t cusparseCnnz(cusparseHandle_t handle,
                              cusparseDirection_t dirA,
                              int m,
                              int n,
                              const cusparseMatDescr_t  descrA,
                              const cuComplex *A,
                              int lda,
                              int *nnzPerRowCol,
                              int *nnzTotalHostPtr);

cusparseStatus_t cusparseZnnz(cusparseHandle_t handle,
                              cusparseDirection_t dirA,
                              int m,
                              int n,
                              const cusparseMatDescr_t  descrA,
                              const cuDoubleComplex *A,
                              int lda,
                              int *nnzPerRowCol,
                              int *nnzTotalHostPtr);

cusparseStatus_t cusparseSdense2csr(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const cusparseMatDescr_t descrA,
                                    const float *A,
                                    int lda,
                                    const int *nnzPerRow,
                                    float *csrValA,
                                    int *csrRowPtrA,
                                    int *csrColIndA);

cusparseStatus_t cusparseDdense2csr(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const cusparseMatDescr_t descrA,
                                    const double *A,
                                    int lda,
                                    const int *nnzPerRow,
                                    double *csrValA,
                                    int *csrRowPtrA,
                                    int *csrColIndA);

cusparseStatus_t cusparseCdense2csr(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const cusparseMatDescr_t descrA,
                                    const cuComplex *A,
                                    int lda,
                                    const int *nnzPerRow,
                                    cuComplex *csrValA,
                                    int *csrRowPtrA,
                                    int *csrColIndA);

cusparseStatus_t cusparseZdense2csr(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const cusparseMatDescr_t descrA,
                                    const cuDoubleComplex *A,
                                    int lda,
                                    const int *nnzPerRow,
                                    cuDoubleComplex *csrValA,
                                    int *csrRowPtrA,
                                    int *csrColIndA);

cusparseStatus_t cusparseScsr2dense(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const cusparseMatDescr_t descrA,
                                    const float *csrValA,
                                    const int *csrRowPtrA,
                                    const int *csrColIndA,
                                    float *A,
                                    int lda);

cusparseStatus_t cusparseDcsr2dense(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const cusparseMatDescr_t descrA,
                                    const double *csrValA,
                                    const int *csrRowPtrA,
                                    const int *csrColIndA,
                                    double *A,
                                    int lda);

cusparseStatus_t cusparseCcsr2dense(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const cusparseMatDescr_t descrA,
                                    const cuComplex *csrValA,
                                    const int *csrRowPtrA,
                                    const int *csrColIndA,
                                    cuComplex *A,
                                    int lda);

cusparseStatus_t cusparseZcsr2dense(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const cusparseMatDescr_t descrA,
                                    const cuDoubleComplex *csrValA,
                                    const int *csrRowPtrA,
                                    const int *csrColIndA,
                                    cuDoubleComplex *A,
                                    int lda);

cusparseStatus_t cusparseSdense2csc(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const cusparseMatDescr_t descrA,
                                    const float *A,
                                    int lda,
                                    const int *nnzPerCol,
                                    float *cscValA,
                                    int *cscRowIndA,
                                    int *cscColPtrA);

cusparseStatus_t cusparseDdense2csc(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const cusparseMatDescr_t descrA,
                                    const double *A,
                                    int lda,
                                    const int *nnzPerCol,
                                    double *cscValA,
                                    int *cscRowIndA,
                                    int *cscColPtrA);

cusparseStatus_t cusparseCdense2csc(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const cusparseMatDescr_t descrA,
                                    const cuComplex *A,
                                    int lda,
                                    const int *nnzPerCol,
                                    cuComplex *cscValA,
                                    int *cscRowIndA,
                                    int *cscColPtrA);

cusparseStatus_t cusparseZdense2csc(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const cusparseMatDescr_t descrA,
                                    const cuDoubleComplex *A,
                                    int lda,
                                    const int *nnzPerCol,
                                    cuDoubleComplex *cscValA,
                                    int *cscRowIndA,
                                    int *cscColPtrA);

cusparseStatus_t cusparseScsc2dense(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const cusparseMatDescr_t descrA,
                                    const float *cscValA,
                                    const int *cscRowIndA,
                                    const int *cscColPtrA,
                                    float *A,
                                    int lda);

cusparseStatus_t cusparseDcsc2dense(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const cusparseMatDescr_t descrA,
                                    const double *cscValA,
                                    const int *cscRowIndA,
                                    const int *cscColPtrA,
                                    double *A,
                                    int lda);

cusparseStatus_t cusparseCcsc2dense(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const cusparseMatDescr_t descrA,
                                    const cuComplex *cscValA,
                                    const int *cscRowIndA,
                                    const int *cscColPtrA,
                                    cuComplex *A,
                                    int lda);

cusparseStatus_t cusparseZcsc2dense(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const cusparseMatDescr_t descrA,
                                    const cuDoubleComplex *cscValA,
                                    const int *cscRowIndA,
                                    const int *cscColPtrA,
                                    cuDoubleComplex *A,
                                    int lda);

cusparseStatus_t cusparseXcoo2csr(cusparseHandle_t handle,
                                  const int *cooRowInd,
                                  int nnz,
                                  int m,
                                  int *csrRowPtr,
                                  cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseXcsr2coo(cusparseHandle_t handle,
                                  const int *csrRowPtr,
                                  int nnz,
                                  int m,
                                  int *cooRowInd,
                                  cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseScsr2csc(cusparseHandle_t handle,
                                  int m,
                                  int n,
                                  int nnz,
                                  const float  *csrValA,
                                  const int *csrRowPtrA,
                                  const int *csrColIndA,
                                  float *cscValA,
                                  int *cscRowIndA,
                                  int *cscColPtrA,
                                  cusparseAction_t copyValues,
                                  cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseDcsr2csc(cusparseHandle_t handle,
                                  int m,
                                  int n,
                                  int nnz,
                                  const double  *csrValA,
                                  const int *csrRowPtrA,
                                  const int *csrColIndA,
                                  double *cscValA,
                                  int *cscRowIndA,
                                  int *cscColPtrA,
                                  cusparseAction_t copyValues,
                                  cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseCcsr2csc(cusparseHandle_t handle,
                                  int m,
                                  int n,
                                  int nnz,
                                  const cuComplex *csrValA,
                                  const int *csrRowPtrA,
                                  const int *csrColIndA,
                                  cuComplex *cscValA,
                                  int *cscRowIndA,
                                  int *cscColPtrA,
                                  cusparseAction_t copyValues,
                                  cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseZcsr2csc(cusparseHandle_t handle,
                                  int m,
                                  int n,
                                  int nnz,
                                  const cuDoubleComplex *csrValA,
                                  const int *csrRowPtrA,
                                  const int *csrColIndA,
                                  cuDoubleComplex *cscValA,
                                  int *cscRowIndA,
                                  int *cscColPtrA,
                                  cusparseAction_t copyValues,
                                  cusparseIndexBase_t idxBase);

""")

# Get the address in a cdata pointer:
_ptr_to_long = lambda ptr: struct.Struct('L').unpack(_ffi.buffer(ptr))[0]

__verify_scr = """
#include <cusparse_v2.h>
#include <driver_types.h>
"""
_ffi_lib = _ffi.verify(__verify_scr, libraries=['cusparse'],
                       include_dirs=['/usr/local/cuda/include'],
                       library_dirs=['/usr/local/cuda/lib64/'])


class CUSPARSE_ERROR(Exception):
    """CUSPARSE error"""
    pass

# Use CUSPARSE_STATUS* definitions to dynamically create corresponding
# exception classes and populate dictionary used to raise appropriate
# exception in response to the corresponding CUSPARSE error code:
CUSPARSE_EXCEPTIONS = {-1: CUSPARSE_ERROR}
for k, v in _ffi_lib.__dict__.iteritems():

    # Skip CUSPARSE_STATUS_SUCCESS:
    if re.match('CUSPARSE_STATUS.*', k) and v != 0:
        CUSPARSE_EXCEPTIONS[v] = vars()[k] = type(k, (CUSPARSE_ERROR,), {})

# Import various enum values into module namespace:
regex = 'CUSPARSE_(?:MATRIX|FILL|DIAG|INDEX|OPERATION|DIRECTION).*'
for k, v in _ffi_lib.__dict__.iteritems():
    if re.match(regex, k):
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

    handle = _ffi.new('cusparseHandle_t *')
    status = _ffi_lib.cusparseCreate(handle)
    cusparseCheckStatus(status)
    return struct.Struct('L').unpack(_ffi.buffer(handle))[0]


def cusparseDestroy(handle):
    """
    Release CUSPARSE resources.

    Releases hardware resources used by CUSPARSE

    Parameters
    ----------
    handle : int
        CUSPARSE library context.
    """

    handle = _ffi.cast('cusparseHandle_t', handle)
    status = _ffi_lib.cusparseDestroy(handle)
    cusparseCheckStatus(status)


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

    version = _ffi.new('int *')
    handle = _ffi.cast('cusparseHandle_t', handle)
    status = _ffi_lib.cusparseGetVersion(handle, version)

    cusparseCheckStatus(status)
    return version[0]


def cusparseSetStream(handle, stream_id):
    """
    Sets the CUSPARSE stream in which kernels will run.

    Parameters
    ----------
    handle : int
        CUSPARSE library context.
    stream_id : int
        Stream ID.
    """

    handle = _ffi.cast('cusparseHandle_t', handle)
    sid = _ffi.cast('cudaStream_t', stream_id)
    status = _ffi_lib.cusparseSetStream(handle, sid)
    cusparseCheckStatus(status)


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

    desc = _ffi.new('cusparseMatDescr_t *')
    status = _ffi_lib.cusparseCreateMatDescr(desc)
    cusparseCheckStatus(status)
    return desc[0]


def cusparseDestroyMatDescr(desc):
    """
    Releases the memory allocated for the matrix descriptor.

    Parameters
    ----------
    desc : cusparseMatDescr
        Matrix descriptor.
    """

    status = _ffi_lib.cusparseDestroyMatDescr(desc)
    cusparseCheckStatus(status)


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

    status = _ffi_lib.cusparseSetMatType(desc, type)
    cusparseCheckStatus(status)


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

    return _ffi_lib.cusparseGetMatType(desc)


def cusparseSetMatFillMode(desc, mode):
    """
    Sets the fill mode of the specified matrix.

    Parameters
    ----------
    desc : cusparseMatDescr
        Matrix descriptor.
    mode : int
        Fill mode.
    """

    status = _ffi_lib.cusparseSetMatFillMode(desc, mode)
    cusparseCheckStatus(status)


def cusparseGetMatFillMode(desc):
    """
    Gets the fill mode of the specified matrix.

    Parameters
    ----------
    desc : cusparseMatDescr
        Matrix descriptor.

    Returns
    -------
    mode : int
        Fill mode.
    """

    return _ffi_lib.cusparseGetMatFillMode(desc)


def cusparseSetMatDiagType(desc, type):
    """
    Sets the diagonal type of the specified matrix.

    Parameters
    ----------
    desc : cusparseMatDescr
        Matrix descriptor.
    type : int
        Diagonal type.
    """

    status = _ffi_lib.cusparseSetMatDiagType(desc, type)
    cusparseCheckStatus(status)


def cusparseGetMatDiagType(desc):
    """
    Gets the diagonal type of the specified matrix.

    Parameters
    ----------
    desc : cusparseMatDescr
        Matrix descriptor.

    Returns
    -------
    type : int
        Diagonal type.
    """

    return _ffi_lib.cusparseGetMatFillMode(desc)


def cusparseSetMatIndexBase(desc, base):
    """
    Sets the index base of the specified matrix.

    Parameters
    ----------
    desc : cusparseMatDescr
        Matrix descriptor.
    base : int
        Index base.
    """

    status = _ffi_lib.cusparseSetMatIndexBase(desc, base)
    cusparseCheckStatus(status)


def cusparseGetMatIndexBase(desc):
    """
    Gets the index base of the specified matrix.

    Parameters
    ----------
    desc : cusparseMatDescr
        Matrix descriptor.

    Returns
    -------
    base : int
        Index base.
    """

    return _ffi_lib.cusparseGetMatIndexBase(desc)

# Level 1 functions:
def cusparseSaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase):
    handle = _ffi.cast('cusparseHandle_t', handle)
    xVal = _ffi.cast('float *', xVal)
    xInd = _ffi.cast('int *', xInd)
    xInd = _ffi.cast('int *', xInd)
    y = _ffi.cast('float *', y)
    idxBase = _ffi.cast('cusparseIndexBase_t', idxBase)
    status = _ffi_lib.cusparseZaxpyi(handle, nnz, alpha_ffi, xVal,
                                     xInd, y, idxBase)
    cusparseCheckStatus(status)


def cusparseDaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase):
    handle = _ffi.cast('cusparseHandle_t', handle)
    xVal = _ffi.cast('double *', xVal)
    xInd = _ffi.cast('int *', xInd)
    xInd = _ffi.cast('int *', xInd)
    y = _ffi.cast('double *', y)
    idxBase = _ffi.cast('cusparseIndexBase_t', idxBase)
    status = _ffi_lib.cusparseZaxpyi(handle, nnz, alpha_ffi, xVal,
                                     xInd, y, idxBase)
    cusparseCheckStatus(status)


def cusparseCaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase):
    handle = _ffi.cast('cusparseHandle_t', handle)

    # Can't directly cast complex values because of the cuComplex struct's
    # alignment (which can't currently be manually set because cffi doesn't
    # support GCC-specific directives):
    alpha_ffi = _ffi.new('cuComplex *')
    _ffi.buffer(alpha_ffi)[:] = np.complex64(alpha).tostring()
    xVal = _ffi.cast('cuComplex *', xVal)
    xInd = _ffi.cast('int *', xInd)
    xInd = _ffi.cast('int *', xInd)
    y = _ffi.cast('cuComplex *', y)
    idxBase = _ffi.cast('cusparseIndexBase_t', idxBase)
    status = _ffi_lib.cusparseCaxpyi(handle, nnz, alpha_ffi, xVal,
                                     xInd, y, idxBase)
    cusparseCheckStatus(status)


def cusparseZaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase):
    handle = _ffi.cast('cusparseHandle_t', handle)

    # Can't directly cast complex values because of the cuComplex struct's
    # alignment (which can't currently be manually set because cffi doesn't
    # support GCC-specific directives):
    alpha_ffi = _ffi.new('cuDoubleComplex *')
    _ffi.buffer(alpha_ffi)[:] = np.complex128(alpha).tostring()
    xVal = _ffi.cast('cuDoubleComplex *', xVal)
    xInd = _ffi.cast('int *', xInd)
    xInd = _ffi.cast('int *', xInd)
    y = _ffi.cast('cuDoubleComplex *', y)
    idxBase = _ffi.cast('cusparseIndexBase_t', idxBase)
    status = _ffi_lib.cusparseZaxpyi(handle, nnz, alpha_ffi, xVal,
                                     xInd, y, idxBase)
    cusparseCheckStatus(status)

# Format conversion functions:
def cusparseSnnz(handle, dirA, m, n, descrA, A, lda):
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
    handle = _ffi.cast('cusparseHandle_t', handle)
    dirA = _ffi.cast('cusparseDirection_t', dirA)
    print 'descrA: ', descrA
    print 'dirA: ', dirA
    A = _ffi.cast('float *', A)
    print 'A: ', A
    nnzPerRowColumn = _ffi.new('int *')
    nnzTotalDevHostPtr = _ffi.new('int *')

    status = _ffi_lib.cusparseSnnz(handle, dirA, m, n,
                                   descrA, A, lda,
                                   nnzPerRowColumn, nnzTotalDevHostPtr)
    cusparseCheckStatus(status)
    return nnzPerRowColumn, nnzTotalDevHostPtr


def cusparseSdense2csr(handle, m, n, descrA, A, lda,
                       nnzPerRow, csrValA, csrRowPtrA, csrColIndA):
    # Unfinished
    pass


def cusparseScsr2dense(
        handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda):
    """
    Converts sparse matrix in CSR format into dense matrix.

    The sparse matrix is given by csrValA, csrRowPtrA, and csrColIndA and
    has (csrRowPtrA(m) - csrRowPtrA(0)) nonzero entries.

    Note that A is assumed to be in column-major (fortran) order!

    Parameters
    ----------
    handle : int
        CUSPARSE library context.
    m : int
        Rows in A.
    n : int
        Columns in A.
    descrA : cusparseMatDescr
        Matrix descriptor.
    csrValA : pycuda.gpuarray.GPUArray (dtype=np.float32)
        Vector of the nonzero values of a
    csrRowPtrA : pycuda.gpuarray.GPUArray (dtype=np.int)
        integer array of m + 1 elements that contains the start of every row
    csrColIndA : pycuda.gpuarray.GPUArray (dtype=np.int)
                integer array of nnz ( = csrRowPtrA(m) - csrRowPtrA(0) ) column
            indices of the non-zero elements of matrix A .
    A : pycuda.gpuarray.GPUArray.gpudata
        Device pointer to dense matrix of dimensions (lda, n).
    lda : int
        Leading dimension of A.

    Returns
    -------
    A : pycuda.gpuarray.GPUArray
        Dense matrix of dimensions (lda, n).
    """
    handle = _ffi.cast('cusparseHandle_t', handle)
    descrA = _ffi.cast('cusparseMatDescr_t', descrA)

    csrValA = _ffi.cast('float *', csrValA)
    csrRowPtrA = _ffi.cast('int *', csrRowPtrA)
    csrColIndA = _ffi.cast('int *', csrColIndA)
    A = _ffi.cast('float *', A)
    status = _ffi_lib.cusparseScsr2dense(handle, int(m), int(n), descrA,
                                         csrValA, csrRowPtrA, csrColIndA,
                                         A, int(lda))
    cusparseCheckStatus(status)
    return A


# cuSPARSE level 3
def cusparseScsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA,
                   csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc):
    """
    Computes C = alpha * op(A) * B + beta*C

    Assumes A is a CSR matric, and C and B are column-major dense matrices.

    Parameters
    ----------
    handle : int
        CUSPARSE library context.
    transA : cusparseOperation_t
        Determines whether A is transposed or not
    m : int
        Rows in C.
    n : int
        Columns in C.
    k : int
        Columns in A
    nnz : int
        Number of nonzero entries in A
    alpha : float
        factor alpha
    descrA : cusparseMatDescr
        Matrix descriptor.
    csrValA : pycuda.gpuarray.GPUArray (dtype=np.float32)
        Vector of the nonzero values of a
    csrRowPtrA : pycuda.gpuarray.GPUArray (dtype=np.int)
        integer array of m + 1 elements that contains the start of every row
    csrColIndA : pycuda.gpuarray.GPUArray (dtype=np.int)
            integer array of nnz ( = csrRowPtrA(m) - csrRowPtrA(0) ) column
            indices of the non-zero elements of matrix A .
    B : pycuda.gpuarray.GPUArray
        Dense matrix of dimensions (ldb, n).
    ldb : int
        Leading dimension of b.
    alpha : float
        factor beta
    C : pycuda.gpuarray.GPUArray
        Dense matrix of dimensions (m, n).
    ldc : int
        Leading dimension of c.

    Returns
    -------
    C : pycuda.gpuarray.GPUArray
        resulting matrix
    """
    handle = _ffi.cast('cusparseHandle_t', handle)
    transA = _ffi.cast('cusparseOperation_t', transA)
    descrA = _ffi.cast('cusparseMatDescr_t', descrA)

    alpha = _ffi.new("float[1]", [np.float32(alpha)])

    csrValA = _ffi.cast('float *', csrValA)
    csrRowPtrA = _ffi.cast('int *', csrRowPtrA)
    csrColIndA = _ffi.cast('int *', csrColIndA)

    B = _ffi.cast('float *', B)
    C = _ffi.cast('float *', C)
    beta = _ffi.new("float[1]", [np.float32(beta)])

    status = _ffi_lib.cusparseScsrmm(handle, transA, m, n, k, nnz, alpha,
                                     descrA, csrValA, csrRowPtrA, csrColIndA,
                                     B, ldb, beta, C, ldc)
    cusparseCheckStatus(status)
    return C


def cusparseScsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                    csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc):
    handle = _ffi.cast('cusparseHandle_t', handle)
    transA = _ffi.cast('cusparseOperation_t', transA)
    transB = _ffi.cast('cusparseOperation_t', transB)
    descrA = _ffi.cast('cusparseMatDescr_t', descrA)

    alpha = _ffi.new("float[1]", [np.float32(alpha)])

    csrValA = _ffi.cast('float *', csrValA)
    csrRowPtrA = _ffi.cast('int *', csrRowPtrA)
    csrColIndA = _ffi.cast('int *', csrColIndA)

    B = _ffi.cast('float *', B)
    C = _ffi.cast('float *', C)
    beta = _ffi.new("float[1]", [np.float32(beta)])

    status = _ffi_lib.cusparseScsrmm2(handle, transA, transB, m, n, k, nnz,
                                      alpha, descrA, csrValA, csrRowPtrA,
                                      csrColIndA, B, ldb, beta, C, ldc)
    cusparseCheckStatus(status)
    return C
