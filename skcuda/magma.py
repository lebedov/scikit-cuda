#!/usr/bin/env python

"""
Python interface to MAGMA toolkit.
"""

from __future__ import absolute_import, division, print_function

import sys
import inspect
import ctypes
import atexit
import numpy as np
from numpy.ctypeslib import ndpointer

from . import cuda


# Structures, type definitions and constants

real_Double_t        = ctypes.c_double
magma_int_t          = ctypes.c_int
magma_index_t        = ctypes.c_int
p_magma_index_t      = ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")
p_magmaDoubleComplex = ndpointer(np.complex128, flags="C_CONTIGUOUS")
# TODO: Add definition of cusparseSolveAnalysisInfo
cusparseSolveAnalysisInfo_t = ctypes.c_void_p #ctypes.POINTER(cusparseSolveAnalysisInfo)

MAGMA_SUCCESS = 0
HAVE_CUBLAS   = 1


class magma_uplo_t:
    MagmaUpper         = 121
    MagmaLower         = 122
    MagmaFull          = 123  # lascl, laset
    MagmaHessenberg    = 124  # lascl


class magma_storage_t:
    Magma_CSR          = 611
    Magma_ELLPACKT     = 612
    Magma_ELL          = 613
    Magma_DENSE        = 614
    Magma_BCSR         = 615
    Magma_CSC          = 616
    Magma_HYB          = 617
    Magma_COO          = 618
    Magma_ELLRT        = 619
    Magma_SPMVFUNCTION = 620
    Magma_SELLP        = 621
    Magma_ELLD         = 622
    Magma_CSRLIST      = 623
    Magma_CSRD         = 624
    Magma_CSRL         = 627
    Magma_CSRU         = 628
    Magma_CSRCOO       = 629
    Magma_CUCSR        = 630


class magma_solver_type:
    Magma_CG           = 431
    Magma_CGMERGE      = 432
    Magma_GMRES        = 433
    Magma_BICGSTAB     = 434
    Magma_BICGSTABMERGE  = 435
    Magma_BICGSTABMERGE2 = 436
    Magma_JACOBI       = 437
    Magma_GS           = 438
    Magma_ITERREF      = 439
    Magma_BCSRLU       = 440
    Magma_PCG          = 441
    Magma_PGMRES       = 442
    Magma_PBICGSTAB    = 443
    Magma_PASTIX       = 444
    Magma_ILU          = 445
    Magma_ICC          = 446
    Magma_AILU         = 447
    Magma_AICC         = 448
    Magma_BAITER       = 449
    Magma_LOBPCG       = 450
    Magma_NONE         = 451
    Magma_FUNCTION     = 452
    Magma_IDR          = 453
    Magma_PIDR         = 454
    Magma_CGS          = 455
    Magma_PCGS         = 456
    Magma_CGSMERGE     = 457
    Magma_PCGSMERGE    = 458
    Magma_TFQMR        = 459
    Magma_PTFQMR       = 460
    Magma_TFQMRMERGE   = 461
    Magma_PTFQMRMERGE  = 462
    Magma_QMR          = 463
    Magma_PQMR         = 464
    Magma_QMRMERGE     = 465
    Magma_PQMRMERGE    = 466
    Magma_BOMBARD      = 490
    Magma_BOMBARDMERGE = 491
    Magma_PCGMERGE     = 492
    Magma_BAITERO      = 493
    Magma_IDRMERGE     = 494
    Magma_PBICGSTABMERGE = 495
    Magma_AICT         = 496
    Magma_CUSTOMIC     = 497
    Magma_CUSTOMILU    = 498
    Magma_PIDRMERGE    = 499
    Magma_BICG         = 500
    Magma_BICGMERGE    = 501
    Magma_PBICG        = 502
    Magma_PBICGMERGE   = 503
    Magma_LSQR         = 504


class magma_location_t:
    Magma_CPU          = 571
    Magma_DEV          = 572


class magma_ortho_t:
    Magma_CGSO         = 561
    Magma_FUSED_CGSO   = 562
    Magma_MGSO         = 563


class magma_scale_t:
    Magma_NOSCALE      = 511
    Magma_UNITROW      = 512
    Magma_UNITDIAG     = 513


class magma_precision:
    Magma_DCOMPLEX     = 501
    Magma_FCOMPLEX     = 502
    Magma_DOUBLE       = 503
    Magma_FLOAT        = 504


class _uval(ctypes.Union):
    _fields_ = [('val', ctypes.c_void_p),
                ('dval', ctypes.c_void_p)]


class _udiag(ctypes.Union):
    _fields_ = [('diag', ctypes.c_void_p),
                ('ddiag', ctypes.c_void_p)]


class _urow(ctypes.Union):
    _fields_ = [('row', ctypes.c_void_p),
                ('drow', ctypes.c_void_p)]


class _urowidx(ctypes.Union):
    _fields_ = [('rowidx', ctypes.c_void_p),
                ('drowidx', ctypes.c_void_p)]


class _ucol(ctypes.Union):
    _fields_ = [('col', ctypes.c_void_p),
                ('dcol', ctypes.c_void_p)]


class _ulist(ctypes.Union):
    _fields_ = [('list', ctypes.c_void_p),
                ('dlist', ctypes.c_void_p)]


class magma_t_matrix(ctypes.Structure):
    _anonymous_ = ('uval','udiag','urow','urowidx','ucol','ulist')
    _fields_ = [('storage_type', ctypes.c_int),
                ('memory_location', ctypes.c_int),
                ('sym', ctypes.c_int),
                ('diagorder_type', ctypes.c_int),
                ('fill_mode', ctypes.c_int),
                ('num_rows', ctypes.c_int),
                ('num_cols', ctypes.c_int),
                ('nnz', ctypes.c_int),
                ('max_nnz_row', ctypes.c_int),
                ('diameter', ctypes.c_int),
                ('true_nnz', ctypes.c_int),
                ('uval', _uval),
                ('udiag', _udiag),
                ('urow', _urow),
                ('urowidx', _urowidx),
                ('ucol', _ucol),
                ('ulist', _ulist),
                ('blockinfo', ctypes.c_void_p),
                ('blocksize', ctypes.c_int),
                ('numblocks', ctypes.c_int),
                ('alignment', ctypes.c_int),
                ('major', ctypes.c_int),
                ('ld', ctypes.c_int)
                ]

    def __init__(self, storage_type):
        super(magma_t_matrix, self).__init__(storage_type)


# For the python implementation the c_void_p in the structures can point to any data type.
# Thus, it is not necessary to implement the matrix structures for other data types explicitly.
# We introduce magma_t_matrix and magma_t_vector, where the 't' stands for the corresponding
# type which can be z, c, d or s.
magma_z_matrix = magma_t_matrix
magma_c_matrix = magma_t_matrix
magma_d_matrix = magma_t_matrix
magma_s_matrix = magma_t_matrix

magma_z_vector = magma_z_matrix
magma_c_vector = magma_c_matrix
magma_d_vector = magma_d_matrix
magma_s_vector = magma_s_matrix


class magma_z_solver_par(ctypes.Structure):
    _fields_ = [('solver', ctypes.c_int),                               # solver type
                ('version', ctypes.c_int),                              # sometimes there are different versions
                ('atol', ctypes.c_double),                              # absolute residual stopping criterion
                ('rtol', ctypes.c_double),                              # relative residual stopping criterion
                ('maxiter', ctypes.c_int),                              # upper iteration limit
                ('restart', ctypes.c_int),                              # for GMRES
                ('ortho', ctypes.c_int),                                # for GMRES
                ('numiter', ctypes.c_int),                              # feedback: number of needed iterations
                ('spmv_count', ctypes.c_int),                           # feedback: number of needed SpMV - can be different to iteration count
                ('init_res', ctypes.c_double),                          # feedback: initial residual
                ('final_res', ctypes.c_double),                         # feedback: final residual
                ('iter_res', ctypes.c_double),                          # feedback: iteratively computed residual
                ('runtime', real_Double_t),                             # feedback: runtime needed
                ('res_vec', ctypes.POINTER(real_Double_t)),             # feedback: array containing residuals
                ('timing', ctypes.POINTER(real_Double_t)),              # feedback: detailed timing
                ('verbose', ctypes.c_int),                              # print residual ever 'verbose' iterations
                ('num_eigenvalues', ctypes.c_int),                      # number of EV for eigensolvers
                ('ev_length', ctypes.c_int),                            # needed for framework
                ('eigenvalues', ctypes.POINTER(ctypes.c_double)),       # feedback: array containing eigenvalues
                ('eigenvectors', ctypes.POINTER(cuda.cuDoubleComplex)), # feedback: array containing eigenvectors
                ('info', ctypes.c_int)                                  # feedback: did the solver converge etc
                ]


class magma_c_solver_par(ctypes.Structure):
    _fields_ = [('solver', ctypes.c_int),                               # solver type
                ('version', ctypes.c_int),                              # sometimes there are different versions
                ('atol', ctypes.c_float),                               # absolute residual stopping criterion
                ('rtol', ctypes.c_float),                               # relative residual stopping criterion
                ('maxiter', ctypes.c_int),                              # upper iteration limit
                ('restart', ctypes.c_int),                              # for GMRES
                ('ortho', ctypes.c_int),                                # for GMRES
                ('numiter', ctypes.c_int),                              # feedback: number of needed iterations
                ('spmv_count', ctypes.c_int),                           # feedback: number of needed SpMV - can be different to iteration count
                ('init_res', ctypes.c_float),                           # feedback: initial residual
                ('final_res', ctypes.c_float),                          # feedback: final residual
                ('iter_res', ctypes.c_float),                           # feedback: iteratively computed residual
                ('runtime', real_Double_t),                             # feedback: runtime needed
                ('res_vec', ctypes.POINTER(real_Double_t)),             # feedback: array containing residuals
                ('timing', ctypes.POINTER(real_Double_t)),              # feedback: detailed timing
                ('verbose', ctypes.c_int),                              # print residual ever 'verbose' iterations
                ('num_eigenvalues', ctypes.c_int),                      # number of EV for eigensolvers
                ('ev_length', ctypes.c_int),                            # needed for framework
                ('eigenvalues', ctypes.POINTER(ctypes.c_float)),        # feedback: array containing eigenvalues
                ('eigenvectors', ctypes.POINTER(cuda.cuFloatComplex)),  # feedback: array containing eigenvectors
                ('info', ctypes.c_int)                                  # feedback: did the solver converge etc
                ]


class magma_d_solver_par(ctypes.Structure):
    _fields_ = [('solver', ctypes.c_int),                               # solver type
                ('version', ctypes.c_int),                              # sometimes there are different versions
                ('atol', ctypes.c_double),                              # absolute residual stopping criterion
                ('rtol', ctypes.c_double),                              # relative residual stopping criterion
                ('maxiter', ctypes.c_int),                              # upper iteration limit
                ('restart', ctypes.c_int),                              # for GMRES
                ('ortho', ctypes.c_int),                                # for GMRES
                ('numiter', ctypes.c_int),                              # feedback: number of needed iterations
                ('spmv_count', ctypes.c_int),                           # feedback: number of needed SpMV - can be different to iteration count
                ('init_res', ctypes.c_double),                          # feedback: initial residual
                ('final_res', ctypes.c_double),                         # feedback: final residual
                ('iter_res', ctypes.c_double),                          # feedback: iteratively computed residual
                ('runtime', real_Double_t),                             # feedback: runtime needed
                ('res_vec', ctypes.POINTER(real_Double_t)),             # feedback: array containing residuals
                ('timing', ctypes.POINTER(real_Double_t)),              # feedback: detailed timing
                ('verbose', ctypes.c_int),                              # print residual ever 'verbose' iterations
                ('num_eigenvalues', ctypes.c_int),                      # number of EV for eigensolvers
                ('ev_length', ctypes.c_int),                            # needed for framework
                ('eigenvalues', ctypes.POINTER(ctypes.c_double)),       # feedback: array containing eigenvalues
                ('eigenvectors', ctypes.POINTER(cuda.cuDoubleComplex)), # feedback: array containing eigenvectors
                ('info', ctypes.c_int)                                  # feedback: did the solver converge etc
                ]


class magma_s_solver_par(ctypes.Structure):
    _fields_ = [('solver', ctypes.c_int),                               # solver type
                ('version', ctypes.c_int),                              # sometimes there are different versions
                ('atol', ctypes.c_float),                               # absolute residual stopping criterion
                ('rtol', ctypes.c_float),                               # relative residual stopping criterion
                ('maxiter', ctypes.c_int),                              # upper iteration limit
                ('restart', ctypes.c_int),                              # for GMRES
                ('ortho', ctypes.c_int),                                # for GMRES
                ('numiter', ctypes.c_int),                              # feedback: number of needed iterations
                ('spmv_count', ctypes.c_int),                           # feedback: number of needed SpMV - can be different to iteration count
                ('init_res', ctypes.c_float),                           # feedback: initial residual
                ('final_res', ctypes.c_float),                          # feedback: final residual
                ('iter_res', ctypes.c_float),                           # feedback: iteratively computed residual
                ('runtime', real_Double_t),                             # feedback: runtime needed
                ('res_vec', ctypes.POINTER(real_Double_t)),             # feedback: array containing residuals
                ('timing', ctypes.POINTER(real_Double_t)),              # feedback: detailed timing
                ('verbose', ctypes.c_int),                              # print residual ever 'verbose' iterations
                ('num_eigenvalues', ctypes.c_int),                      # number of EV for eigensolvers
                ('ev_length', ctypes.c_int),                            # needed for framework
                ('eigenvalues', ctypes.POINTER(ctypes.c_float)),        # feedback: array containing eigenvalues
                ('eigenvectors', ctypes.POINTER(cuda.cuFloatComplex)),  # feedback: array containing eigenvectors
                ('info', ctypes.c_int)                                  # feedback: did the solver converge etc
                ]


class magma_z_preconditioner(ctypes.Structure):
    _fields_ = [('solver', ctypes.c_int),
                ('levels', ctypes.c_int),
                ('sweeps', ctypes.c_int),
                ('format', ctypes.c_int),
                ('atol', ctypes.c_double),
                ('rtol', ctypes.c_double),
                ('maxiter', ctypes.c_int),
                ('restart', ctypes.c_int),
                ('numiter', ctypes.c_int),
                ('spmv_count', ctypes.c_int),
                ('init_res', ctypes.c_double),
                ('final_res', ctypes.c_double),
                ('runtime', real_Double_t),                 # feedback: preconditioner runtime needed
                ('setuptime', real_Double_t),               # feedback: preconditioner setup time needed
                ('M', magma_z_matrix),
                ('L', magma_z_matrix),
                ('LT', magma_z_matrix),
                ('U', magma_z_matrix),
                ('UT', magma_z_matrix),
                ('LD', magma_z_matrix),
                ('UD', magma_z_matrix),
                ('d', magma_z_matrix),
                ('d2', magma_z_matrix),
                ('work1', magma_z_matrix),
                ('work2', magma_z_matrix),
                ('int_array_1', ctypes.POINTER(ctypes.c_int)),
                ('int_array_2', ctypes.POINTER(ctypes.c_int)),
                ('cuinfo', cusparseSolveAnalysisInfo_t),
                ('cuinfoL', cusparseSolveAnalysisInfo_t),
                ('cuinfoLT', cusparseSolveAnalysisInfo_t),
                ('cuinfoU', cusparseSolveAnalysisInfo_t),
                ('cuinfoUT', cusparseSolveAnalysisInfo_t)
                # TODO:
                #if defined(HAVE_PASTIX)
                #    pastix_data_t*          pastix_data;
                #    magma_int_t*            iparm;
                #    double*                 dparm;
                #endif
                ]


class magma_c_preconditioner(ctypes.Structure):
    _fields_ = [('solver', ctypes.c_int),
                ('levels', ctypes.c_int),
                ('sweeps', ctypes.c_int),
                ('format', ctypes.c_int),
                ('atol', ctypes.c_float),
                ('rtol', ctypes.c_float),
                ('maxiter', ctypes.c_int),
                ('restart', ctypes.c_int),
                ('numiter', ctypes.c_int),
                ('spmv_count', ctypes.c_int),
                ('init_res', ctypes.c_float),
                ('final_res', ctypes.c_float),
                ('runtime', real_Double_t),                 # feedback: preconditioner runtime needed
                ('setuptime', real_Double_t),               # feedback: preconditioner setup time needed
                ('M', magma_c_matrix),
                ('L', magma_c_matrix),
                ('LT', magma_c_matrix),
                ('U', magma_c_matrix),
                ('UT', magma_c_matrix),
                ('LD', magma_c_matrix),
                ('UD', magma_c_matrix),
                ('d', magma_c_matrix),
                ('d2', magma_c_matrix),
                ('work1', magma_c_matrix),
                ('work2', magma_c_matrix),
                ('int_array_1', ctypes.POINTER(ctypes.c_int)),
                ('int_array_2', ctypes.POINTER(ctypes.c_int)),
                ('cuinfo', cusparseSolveAnalysisInfo_t),
                ('cuinfoL', cusparseSolveAnalysisInfo_t),
                ('cuinfoLT', cusparseSolveAnalysisInfo_t),
                ('cuinfoU', cusparseSolveAnalysisInfo_t),
                ('cuinfoUT', cusparseSolveAnalysisInfo_t)
                # TODO:
                #if defined(HAVE_PASTIX)
                #    pastix_data_t*          pastix_data;
                #    magma_int_t*            iparm;
                #    float*                 dparm;
                #endif
                ]

class magma_d_preconditioner(ctypes.Structure):
    _fields_ = [('solver', ctypes.c_int),
                ('levels', ctypes.c_int),
                ('sweeps', ctypes.c_int),
                ('format', ctypes.c_int),
                ('atol', ctypes.c_double),
                ('rtol', ctypes.c_double),
                ('maxiter', ctypes.c_int),
                ('restart', ctypes.c_int),
                ('numiter', ctypes.c_int),
                ('spmv_count', ctypes.c_int),
                ('init_res', ctypes.c_double),
                ('final_res', ctypes.c_double),
                ('runtime', real_Double_t),                 # feedback: preconditioner runtime needed
                ('setuptime', real_Double_t),               # feedback: preconditioner setup time needed
                ('M', magma_d_matrix),
                ('L', magma_d_matrix),
                ('LT', magma_d_matrix),
                ('U', magma_d_matrix),
                ('UT', magma_d_matrix),
                ('LD', magma_d_matrix),
                ('UD', magma_d_matrix),
                ('d', magma_d_matrix),
                ('d2', magma_d_matrix),
                ('work1', magma_d_matrix),
                ('work2', magma_d_matrix),
                ('int_array_1', ctypes.POINTER(ctypes.c_int)),
                ('int_array_2', ctypes.POINTER(ctypes.c_int)),
                ('cuinfo', cusparseSolveAnalysisInfo_t),
                ('cuinfoL', cusparseSolveAnalysisInfo_t),
                ('cuinfoLT', cusparseSolveAnalysisInfo_t),
                ('cuinfoU', cusparseSolveAnalysisInfo_t),
                ('cuinfoUT', cusparseSolveAnalysisInfo_t)
                # TODO:
                #if defined(HAVE_PASTIX)
                #    pastix_data_t*          pastix_data;
                #    magma_int_t*            iparm;
                #    double*                 dparm;
                #endif
                ]

class magma_s_preconditioner(ctypes.Structure):
    _fields_ = [('solver', ctypes.c_int),
                ('levels', ctypes.c_int),
                ('sweeps', ctypes.c_int),
                ('format', ctypes.c_int),
                ('atol', ctypes.c_float),
                ('rtol', ctypes.c_float),
                ('maxiter', ctypes.c_int),
                ('restart', ctypes.c_int),
                ('numiter', ctypes.c_int),
                ('spmv_count', ctypes.c_int),
                ('init_res', ctypes.c_float),
                ('final_res', ctypes.c_float),
                ('runtime', real_Double_t),                 # feedback: preconditioner runtime needed
                ('setuptime', real_Double_t),               # feedback: preconditioner setup time needed
                ('M', magma_s_matrix),
                ('L', magma_s_matrix),
                ('LT', magma_s_matrix),
                ('U', magma_s_matrix),
                ('UT', magma_s_matrix),
                ('LD', magma_s_matrix),
                ('UD', magma_s_matrix),
                ('d', magma_s_matrix),
                ('d2', magma_s_matrix),
                ('work1', magma_s_matrix),
                ('work2', magma_s_matrix),
                ('int_array_1', ctypes.POINTER(ctypes.c_int)),
                ('int_array_2', ctypes.POINTER(ctypes.c_int)),
                ('cuinfo', cusparseSolveAnalysisInfo_t),
                ('cuinfoL', cusparseSolveAnalysisInfo_t),
                ('cuinfoLT', cusparseSolveAnalysisInfo_t),
                ('cuinfoU', cusparseSolveAnalysisInfo_t),
                ('cuinfoUT', cusparseSolveAnalysisInfo_t)
                # TODO:
                #if defined(HAVE_PASTIX)
                #    pastix_data_t*          pastix_data;
                #    magma_int_t*            iparm;
                #    float*                 dparm;
                #endif
                ]


class magma_zopts(ctypes.Structure):
    _fields_ = [('solver_par', magma_z_solver_par),
                ('precond_par', magma_z_preconditioner),
                ('input_format', ctypes.c_int),
                ('blocksize', ctypes.c_int),
                ('alignment', ctypes.c_int),
                ('output_format', ctypes.c_int),
                ('input_location', ctypes.c_int),
                ('output_location', ctypes.c_int),
                ('scaling', ctypes.c_int)
                ]


class magma_copts(ctypes.Structure):
    _fields_ = [('solver_par', magma_c_solver_par),
                ('precond_par', magma_c_preconditioner),
                ('input_format', ctypes.c_int),
                ('blocksize', ctypes.c_int),
                ('alignment', ctypes.c_int),
                ('output_format', ctypes.c_int),
                ('input_location', ctypes.c_int),
                ('output_location', ctypes.c_int),
                ('scaling', ctypes.c_int)
                ]


class magma_dopts(ctypes.Structure):
    _fields_ = [('solver_par', magma_d_solver_par),
                ('precond_par', magma_d_preconditioner),
                ('input_format', ctypes.c_int),
                ('blocksize', ctypes.c_int),
                ('alignment', ctypes.c_int),
                ('output_format', ctypes.c_int),
                ('input_location', ctypes.c_int),
                ('output_location', ctypes.c_int),
                ('scaling', ctypes.c_int)
                ]


class magma_sopts(ctypes.Structure):
    _fields_ = [('solver_par', magma_s_solver_par),
                ('precond_par', magma_s_preconditioner),
                ('input_format', ctypes.c_int),
                ('blocksize', ctypes.c_int),
                ('alignment', ctypes.c_int),
                ('output_format', ctypes.c_int),
                ('input_location', ctypes.c_int),
                ('output_location', ctypes.c_int),
                ('scaling', ctypes.c_int)
                ]


def magma_zopts_default():
    opts                 = magma_zopts()
    opts.solver_par      = magma_z_solver_par_default()
    opts.precond_par     = magma_z_preconditioner_default()
    opts.input_format    = magma_storage_t.Magma_CSR
    opts.blocksize       = 32
    opts.alignment       = 1
    opts.output_format   = magma_storage_t.Magma_CUCSR
    opts.input_location  = magma_location_t.Magma_CPU
    opts.output_location = magma_location_t.Magma_CPU
    opts.scaling         = magma_scale_t.Magma_NOSCALE

    return opts


def magma_copts_default():
    opts                 = magma_copts()
    opts.solver_par      = magma_c_solver_par_default()
    opts.precond_par     = magma_c_preconditioner_default()
    opts.input_format    = magma_storage_t.Magma_CSR
    opts.blocksize       = 32
    opts.alignment       = 1
    opts.output_format   = magma_storage_t.Magma_CUCSR
    opts.input_location  = magma_location_t.Magma_CPU
    opts.output_location = magma_location_t.Magma_CPU
    opts.scaling         = magma_scale_t.Magma_NOSCALE

    return opts


def magma_dopts_default():
    opts                 = magma_dopts()
    opts.solver_par      = magma_d_solver_par_default()
    opts.precond_par     = magma_d_preconditioner_default()
    opts.input_format    = magma_storage_t.Magma_CSR
    opts.blocksize       = 32
    opts.alignment       = 1
    opts.output_format   = magma_storage_t.Magma_CUCSR
    opts.input_location  = magma_location_t.Magma_CPU
    opts.output_location = magma_location_t.Magma_CPU
    opts.scaling         = magma_scale_t.Magma_NOSCALE

    return opts


def magma_sopts_default():
    opts                 = magma_sopts()
    opts.solver_par      = magma_s_solver_par_default()
    opts.precond_par     = magma_s_preconditioner_default()
    opts.input_format    = magma_storage_t.Magma_CSR
    opts.blocksize       = 32
    opts.alignment       = 1
    opts.output_format   = magma_storage_t.Magma_CUCSR
    opts.input_location  = magma_location_t.Magma_CPU
    opts.output_location = magma_location_t.Magma_CPU
    opts.scaling         = magma_scale_t.Magma_NOSCALE

    return opts


def magma_z_solver_par_default():
    solver_par                 = magma_z_solver_par()
    solver_par.solver          = magma_solver_type.Magma_CGMERGE
    solver_par.version         = 0
    solver_par.atol            = 1e-16
    solver_par.rtol            = 1e-10
    solver_par.maxiter         = 1000
    solver_par.restart         = 50
    solver_par.verbose         = 0
    solver_par.num_eigenvalues = 0

    return solver_par


def magma_c_solver_par_default():
    solver_par                 = magma_c_solver_par()
    solver_par.solver          = magma_solver_type.Magma_CGMERGE
    solver_par.version         = 0
    solver_par.atol            = 1e-8
    solver_par.rtol            = 1e-5
    solver_par.maxiter         = 1000
    solver_par.restart         = 50
    solver_par.verbose         = 0
    solver_par.num_eigenvalues = 0

    return solver_par


def magma_d_solver_par_default():
    solver_par                 = magma_d_solver_par()
    solver_par.solver          = magma_solver_type.Magma_CGMERGE
    solver_par.version         = 0
    solver_par.atol            = 1e-16
    solver_par.rtol            = 1e-10
    solver_par.maxiter         = 1000
    solver_par.restart         = 50
    solver_par.verbose         = 0
    solver_par.num_eigenvalues = 0

    return solver_par


def magma_s_solver_par_default():
    solver_par                 = magma_s_solver_par()
    solver_par.solver          = magma_solver_type.Magma_CGMERGE
    solver_par.version         = 0
    solver_par.atol            = 1e-8
    solver_par.rtol            = 1e-5
    solver_par.maxiter         = 1000
    solver_par.restart         = 50
    solver_par.verbose         = 0
    solver_par.num_eigenvalues = 0

    return solver_par


def magma_z_preconditioner_default():
    precond_par         = magma_z_preconditioner()
    precond_par.solver  = magma_solver_type.Magma_NONE
    precond_par.levels  = 0
    precond_par.sweeps  = 5
    precond_par.format  = magma_precision.Magma_DCOMPLEX
    precond_par.atol    = 1e-16
    precond_par.rtol    = 1e-10
    precond_par.maxiter = 100
    precond_par.restart = 10

    return precond_par


def magma_c_preconditioner_default():
    precond_par         = magma_c_preconditioner()
    precond_par.solver  = magma_solver_type.Magma_NONE
    precond_par.levels  = 0
    precond_par.sweeps  = 5
    precond_par.format  = magma_precision.Magma_FCOMPLEX
    precond_par.atol    = 1e-8
    precond_par.rtol    = 1e-5
    precond_par.maxiter = 100
    precond_par.restart = 10

    return precond_par


def magma_d_preconditioner_default():
    precond_par         = magma_d_preconditioner()
    precond_par.solver  = magma_solver_type.Magma_NONE
    precond_par.levels  = 0
    precond_par.sweeps  = 5
    precond_par.format  = magma_precision.Magma_DCOMPLEX
    precond_par.atol    = 1e-16
    precond_par.rtol    = 1e-10
    precond_par.maxiter = 100
    precond_par.restart = 10

    return precond_par


def magma_s_preconditioner_default():
    precond_par         = magma_s_preconditioner()
    precond_par.solver  = magma_solver_type.Magma_NONE
    precond_par.levels  = 0
    precond_par.sweeps  = 5
    precond_par.format  = magma_precision.Magma_FCOMPLEX
    precond_par.atol    = 1e-8
    precond_par.rtol    = 1e-5
    precond_par.maxiter = 100
    precond_par.restart = 10

    return precond_par


class magma_queue(ctypes.Structure):
    _fields_ = [('own__', ctypes.c_int),
                ('device__', ctypes.c_int),
                #if HAVE_CUBLAS
                ('stream__', ctypes.c_void_p), # cudaStream_t
                ('cublas__', ctypes.c_void_p), # cusparseHandle_t
                ('cusparse__', ctypes.c_void_p) # cusparseHandle_t
                ]

    def __int__(self):
        self.stream__.value   = None
        self.cublas__.value   = None
        self.cusparse__.value = None


"""
magma_queue_t is a pointer to a magma_queue structure.
"""
magma_queue_t        = ctypes.POINTER(magma_queue)


# Load MAGMA library:
if 'linux' in sys.platform:
    _libmagma_libname_list = ['libmagma.so']
elif sys.platform == 'darwin':
    _libmagma_libname_list = ['magma.so', 'libmagma.dylib']
elif sys.platform == 'win32':
    _libmagma_libname_list = ['magma.dll']
else:
    raise RuntimeError('unsupported platform')

_load_err = ''
for _lib in _libmagma_libname_list:
    try:
        _libmagma = ctypes.cdll.LoadLibrary(_lib)
    except OSError:
        _load_err += ('' if _load_err == '' else ', ') + _lib
    else:
        _load_err = ''
        break
if _load_err:
    raise OSError('%s not found' % _load_err)

# Exceptions corresponding to various MAGMA errors:
_libmagma.magma_strerror.restype = ctypes.c_char_p
_libmagma.magma_strerror.argtypes = [ctypes.c_int]
# MAGMA below 1.4.0 uses "L" and "U" to select upper/lower triangular
# matrices, MAGMA 1.5+ uses numeric constants. This dict will be filled
# in magma_init() and will convert between the two modes accordingly
_uplo_conversion = {}
def magma_strerror(error):
    """
    Return string corresponding to specified MAGMA error code.
    """

    return _libmagma.magma_strerror(error)


class MagmaError(Exception):
	def __init__(self, status, info=None):
		self._status = status
		self._info = info
		errstr = "%s (Code: %d)" % (magma_strerror(status), status)
		super(MagmaError,self).__init__(errstr)


def magmaCheckStatus(status):
    """
    Raise an exception corresponding to the specified MAGMA status code.
    """

    if status != 0:
        raise MagmaError(status)

# Utility functions:
_libmagma.magma_version.argtypes = [ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p]


def magma_version():
    """
    Get MAGMA version.
    """
    majv = ctypes.c_int()
    minv = ctypes.c_int()
    micv = ctypes.c_int()
    _libmagma.magma_version(ctypes.byref(majv),
        ctypes.byref(minv), ctypes.byref(micv))
    return (majv.value, minv.value, micv.value)


_libmagma.magma_uplo_const.restype = ctypes.c_int
_libmagma.magma_uplo_const.argtypes = [ctypes.c_char]
_libmagma.magma_init.restype = int
def magma_init():
    """
    Initialize MAGMA.
    """

    global _uplo_conversion
    status = _libmagma.magma_init()
    magmaCheckStatus(status)
    v = magma_version()
    if v >= (1, 5, 0):
        _uplo_conversion.update({"L": _libmagma.magma_uplo_const(b"L"),
                                 "l": _libmagma.magma_uplo_const(b"l"),
                                 "U": _libmagma.magma_uplo_const(b"U"),
                                 "u": _libmagma.magma_uplo_const(b"u")})
    else:
       _uplo_conversion.update({"L": "L", "l": "l", "U": "u", "u": "u"})

_libmagma.magma_finalize.restype = int
def magma_finalize():
    """
    Finalize MAGMA.
    """

    status = _libmagma.magma_finalize()
    magmaCheckStatus(status)

_libmagma.magma_getdevice_arch.restype = int
def magma_getdevice_arch():
    """
    Get device architecture.
    """

    return _libmagma.magma_getdevice_arch()

_libmagma.magma_getdevice.argtypes = [ctypes.c_void_p]
def magma_getdevice():
    """
    Get current device used by MAGMA.
    """

    dev = ctypes.c_int()
    _libmagma.magma_getdevice(ctypes.byref(dev))
    return dev.value

_libmagma.magma_setdevice.argtypes = [ctypes.c_int]
def magma_setdevice(dev):
    """
    Get current device used by MAGMA.
    """

    _libmagma.magma_setdevice(dev)

def magma_device_sync():
    """
    Synchronize device used by MAGMA.
    """

    _libmagma.magma_device_sync()

# BLAS routines

# ISAMAX, IDAMAX, ICAMAX, IZAMAX
_libmagma.magma_isamax.restype = int
_libmagma.magma_isamax.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_isamax(n, dx, incx):
    """
    Index of maximum magnitude element.
    """

    return _libmagma.magma_isamax(n, int(dx), incx)

_libmagma.magma_idamax.restype = int
_libmagma.magma_idamax.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_idamax(n, dx, incx):
    """
    Index of maximum magnitude element.
    """

    return _libmagma.magma_idamax(n, int(dx), incx)

_libmagma.magma_icamax.restype = int
_libmagma.magma_icamax.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_icamax(n, dx, incx):
    """
    Index of maximum magnitude element.
    """

    return _libmagma.magma_icamax(n, int(dx), incx)

_libmagma.magma_izamax.restype = int
_libmagma.magma_izamax.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_izamax(n, dx, incx):
    """
    Index of maximum magnitude element.
    """

    return _libmagma.magma_izamax(n, int(dx), incx)

# ISAMIN, IDAMIN, ICAMIN, IZAMIN
_libmagma.magma_isamin.restype = int
_libmagma.magma_isamin.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_isamin(n, dx, incx):
    """
    Index of minimum magnitude element.
    """

    return _libmagma.magma_isamin(n, int(dx), incx)

_libmagma.magma_idamin.restype = int
_libmagma.magma_idamin.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_idamin(n, dx, incx):
    """
    Index of minimum magnitude element.
    """

    return _libmagma.magma_idamin(n, int(dx), incx)

_libmagma.magma_icamin.restype = int
_libmagma.magma_icamin.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_icamin(n, dx, incx):
    """
    Index of minimum magnitude element.
    """

    return _libmagma.magma_icamin(n, int(dx), incx)

_libmagma.magma_izamin.restype = int
_libmagma.magma_izamin.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_izamin(n, dx, incx):
    """
    Index of minimum magnitude element.
    """

    return _libmagma.magma_izamin(n, int(dx), incx)

# SASUM, DASUM, SCASUM, DZASUM
_libmagma.magma_sasum.restype = int
_libmagma.magma_sasum.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_sasum(n, dx, incx):
    """
    Sum of absolute values of vector.
    """

    return _libmagma.magma_sasum(n, int(dx), incx)

_libmagma.magma_dasum.restype = int
_libmagma.magma_dasum.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_dasum(n, dx, incx):
    """
    Sum of absolute values of vector.
    """

    return _libmagma.magma_dasum(n, int(dx), incx)

_libmagma.magma_scasum.restype = int
_libmagma.magma_scasum.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_scasum(n, dx, incx):
    """
    Sum of absolute values of vector.
    """

    return _libmagma.magma_scasum(n, int(dx), incx)

_libmagma.magma_dzasum.restype = int
_libmagma.magma_dzasum.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_dzasum(n, dx, incx):
    """
    Sum of absolute values of vector.
    """

    return _libmagma.magma_dzasum(n, int(dx), incx)

# SAXPY, DAXPY, CAXPY, ZAXPY
_libmagma.magma_saxpy.restype = int
_libmagma.magma_saxpy.argtypes = [ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_saxpy(n, alpha, dx, incx, dy, incy):
    """
    Vector addition.
    """

    _libmagma.magma_saxpy(n, alpha, int(dx), incx, int(dy), incy)

_libmagma.magma_daxpy.restype = int
_libmagma.magma_daxpy.argtypes = [ctypes.c_int,
                                  ctypes.c_double,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_daxpy(n, alpha, dx, incx, dy, incy):
    """
    Vector addition.
    """

    _libmagma.magma_daxpy(n, alpha, int(dx), incx, int(dy), incy)

_libmagma.magma_caxpy.restype = int
_libmagma.magma_caxpy.argtypes = [ctypes.c_int,
                                  cuda.cuFloatComplex,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_caxpy(n, alpha, dx, incx, dy, incy):
    """
    Vector addition.
    """

    _libmagma.magma_caxpy(n, ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                              alpha.imag)),
                          int(dx), incx, int(dy), incy)

_libmagma.magma_zaxpy.restype = int
_libmagma.magma_zaxpy.argtypes = [ctypes.c_int,
                                  cuda.cuDoubleComplex,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_zaxpy(n, alpha, dx, incx, dy, incy):
    """
    Vector addition.
    """

    _libmagma.magma_zaxpy(n, ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                               alpha.imag)),
                          int(dx), incx, int(dy), incy)

# SCOPY, DCOPY, CCOPY, ZCOPY
_libmagma.magma_scopy.restype = int
_libmagma.magma_scopy.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_scopy(n, dx, incx, dy, incy):
    """
    Vector copy.
    """

    _libmagma.magma_scopy(n, int(dx), incx, int(dy), incy)

_libmagma.magma_dcopy.restype = int
_libmagma.magma_dcopy.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_dcopy(n, dx, incx, dy, incy):
    """
    Vector copy.
    """

    _libmagma.magma_dcopy(n, int(dx), incx, int(dy), incy)

_libmagma.magma_ccopy.restype = int
_libmagma.magma_ccopy.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_ccopy(n, dx, incx, dy, incy):
    """
    Vector copy.
    """

    _libmagma.magma_ccopy(n, int(dx), incx, int(dy), incy)

_libmagma.magma_zcopy.restype = int
_libmagma.magma_zcopy.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_zcopy(n, dx, incx, dy, incy):
    """
    Vector copy.
    """

    _libmagma.magma_zcopy(n, int(dx), incx, int(dy), incy)

# SDOT, DDOT, CDOTU, CDOTC, ZDOTU, ZDOTC
_libmagma.magma_sdot.restype = ctypes.c_float
_libmagma.magma_sdot.argtypes = [ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def magma_sdot(n, dx, incx, dy, incy):
    """
    Vector dot product.
    """

    return _libmagma.magma_sdot(n, int(dx), incx, int(dy), incy)

_libmagma.magma_ddot.restype = ctypes.c_double
_libmagma.magma_ddot.argtypes = [ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def magma_ddot(n, dx, incx, dy, incy):
    """
    Vector dot product.
    """

    return _libmagma.magma_ddot(n, int(dx), incx, int(dy), incy)

_libmagma.magma_cdotc.restype = cuda.cuFloatComplex
_libmagma.magma_cdotc.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_cdotc(n, dx, incx, dy, incy):
    """
    Vector dot product.
    """

    return _libmagma.magma_cdotc(n, int(dx), incx, int(dy), incy)

_libmagma.magma_cdotu.restype = cuda.cuFloatComplex
_libmagma.magma_cdotu.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_cdotu(n, dx, incx, dy, incy):
    """
    Vector dot product.
    """

    return _libmagma.magma_cdotu(n, int(dx), incx, int(dy), incy)

_libmagma.magma_zdotc.restype = cuda.cuDoubleComplex
_libmagma.magma_zdotc.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_zdotc(n, dx, incx, dy, incy):
    """
    Vector dot product.
    """

    return _libmagma.magma_zdotc(n, int(dx), incx, int(dy), incy)

_libmagma.magma_zdotu.restype = cuda.cuDoubleComplex
_libmagma.magma_zdotu.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_zdotu(n, dx, incx, dy, incy):
    """
    Vector dot product.
    """

    return _libmagma.magma_zdotu(n, int(dx), incx, int(dy), incy)

# SNRM2, DNRM2, SCNRM2, DZNRM2
_libmagma.magma_snrm2.restype = ctypes.c_float
_libmagma.magma_snrm2.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_snrm2(n, dx, incx):
    """
    Euclidean norm (2-norm) of vector.
    """

    return _libmagma.magma_snrm2(n, int(dx), incx)

_libmagma.magma_dnrm2.restype = ctypes.c_double
_libmagma.magma_dnrm2.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_dnrm2(n, dx, incx):
    """
    Euclidean norm (2-norm) of vector.
    """

    return _libmagma.magma_dnrm2(n, int(dx), incx)

_libmagma.magma_scnrm2.restype = ctypes.c_float
_libmagma.magma_scnrm2.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_scnrm2(n, dx, incx):
    """
    Euclidean norm (2-norm) of vector.
    """

    return _libmagma.magma_scnrm2(n, int(dx), incx)

_libmagma.magma_dznrm2.restype = ctypes.c_double
_libmagma.magma_dznrm2.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_dznrm2(n, dx, incx):
    """
    Euclidean norm (2-norm) of vector.
    """

    return _libmagma.magma_dznrm2(n, int(dx), incx)

# SROT, DROT, CROT, CSROT, ZROT, ZDROT
_libmagma.magma_srot.argtypes = [ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_float,
                                 ctypes.c_float]
def magma_srot(n, dx, incx, dy, incy, dc, ds):
    """
    Apply a rotation to vectors.
    """

    _libmagma.magma_srot(n, int(dx), incx, int(dy), incy, dc, ds)

# SROTM, DROTM
_libmagma.magma_srotm.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p]
def magma_srotm(n, dx, incx, dy, incy, param):
    """
    Apply a real modified Givens rotation.
    """

    _libmagma.magma_srotm(n, int(dx), incx, int(dy), incy, param)

# SROTMG, DROTMG
_libmagma.magma_srotmg.argtypes = [ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_srotmg(d1, d2, x1, y1, param):
    """
    Construct a real modified Givens rotation matrix.
    """

    _libmagma.magma_srotmg(int(d1), int(d2), int(x1), int(y1), param)

# SSCAL, DSCAL, CSCAL, CSCAL, CSSCAL, ZSCAL, ZDSCAL
_libmagma.magma_sscal.argtypes = [ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_sscal(n, alpha, dx, incx):
    """
    Scale a vector by a scalar.
    """

    _libmagma.magma_sscal(n, alpha, int(dx), incx)

# SSWAP, DSWAP, CSWAP, ZSWAP
_libmagma.magma_sswap.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_sswap(n, dA, ldda, dB, lddb):
    """
    Swap vectors.
    """

    _libmagma.magma_sswap(n, int(dA), ldda, int(dB), lddb)

# SGEMV, DGEMV, CGEMV, ZGEMV
_libmagma.magma_sgemv.argtypes = [ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_sgemv(trans, m, n, alpha, dA, ldda, dx, incx, beta,
                dy, incy):
    """
    Matrix-vector product for general matrix.
    """

    _libmagma.magma_sgemv(trans, m, n, alpha, int(dA), ldda, dx, incx,
                          beta, int(dy), incy)

# SGER, DGER, CGERU, CGERC, ZGERU, ZGERC
_libmagma.magma_sger.argtypes = [ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.c_float,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def magma_sger(m, n, alpha, dx, incx, dy, incy, dA, ldda):
    """
    Rank-1 operation on real general matrix.
    """

    _libmagma.magma_sger(m, n, alpha, int(dx), incx, int(dy), incy,
                         int(dA), ldda)

# SSYMV, DSYMV, CSYMV, ZSYMV
_libmagma.magma_ssymv.argtypes = [ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_ssymv(uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy):
    _libmagma.magma_ssymv(uplo, n, alpha, int(dA), ldda, int(dx), incx, beta,
                          int(dy), incy)

# SSYR, DSYR, CSYR, ZSYR
_libmagma.magma_ssyr.argtypes = [ctypes.c_char,
                                 ctypes.c_int,
                                 ctypes.c_float,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def magma_ssyr(uplo, n, alpha, dx, incx, dA, ldda):
    _libmagma.magma_ssyr(uplo, n, alpha, int(dx), incx, int(dA), ldda)

# SSYR2, DSYR2, CSYR2, ZSYR2
_libmagma.magma_ssyr2.argtypes = [ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_ssyr2(uplo, n, alpha, dx, incx, dy, incy, dA, ldda):
    _libmagma.magma_ssyr2(uplo, n, alpha, int(dx), incx,
                          int(dy), incy, int(dA), ldda)

# STRMV, DTRMV, CTRMV, ZTRMV
_libmagma.magma_strmv.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_strmv(uplo, trans, diag, n,
                dA, ldda, dx, incx):
    _libmagma.magma_strmv(uplo, trans, diag, n,
                          int(dA), ldda, int(dx), incx)

# STRSV, DTRSV, CTRSV, ZTRSV
_libmagma.magma_strsv.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_strsv(uplo, trans, diag, n,
                dA, ldda, dx, incx):
    _libmagma.magma_strsv(uplo, trans, diag, n,
                          int(dA), ldda, int(dx), incx)

# SGEMM, DGEMM, CGEMM, ZGEMM
_libmagma.magma_sgemm.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_sgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta,
                dC, lddc):
    _libmagma.magma_sgemm(transA, transB, m, n, k, alpha,
                          int(dA), ldda, int(dB), lddb,
                          beta, int(dC), lddc)

_libmagma.magma_zgemm.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_zgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta,
                dC, lddc):
    _libmagma.magma_zgemm(transA, transB, m, n, k, alpha,
                          int(dA), ldda, int(dB), lddb,
                          beta, int(dC), lddc)


# SSYMM, DSYMM, CSYMM, ZSYMM
_libmagma.magma_ssymm.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_ssymm(side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta,
                dC, lddc):
    _libmagma.magma_ssymm(side, uplo, m, n, alpha,
                          int(dA), ldda, int(dB), lddb,
                          beta, int(dC), lddc)

# SSYRK, DSYRK, CSYRK, ZSYRK
_libmagma.magma_ssyrk.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_ssyrk(uplo, trans, n, k, alpha, dA, ldda, beta,
                dC, lddc):
    _libmagma.magma_ssyrk(uplo, trans, n, k, alpha,
                          int(dA), ldda, beta, int(dC), lddc)

# SSYR2K, DSYR2K, CSYR2K, ZSYR2K
_libmagma.magma_ssyr2k.argtypes = [ctypes.c_char,
                                   ctypes.c_char,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_float,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_float,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def magma_ssyr2k(uplo, trans, n, k, alpha, dA, ldda,
                 dB, lddb, beta, dC, lddc):
    _libmagma.magma_ssyr2k(uplo, trans, n, k, alpha,
                           int(dA), ldda, int(dB), lddb,
                           beta, int(dC), lddc)

# STRMM, DTRMM, CTRMM, ZTRMM
_libmagma.magma_strmm.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_strmm(side, uplo, trans, diag, m, n, alpha, dA, ldda,
                dB, lddb):
    _libmagma.magma_strmm(uplo, trans, diag, m, n, alpha,
                          int(dA), ldda, int(dB), lddb)

# STRSM, DTRSM, CTRSM, ZTRSM
_libmagma.magma_strsm.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magma_strsm(side, uplo, trans, diag, m, n, alpha, dA, ldda,
                dB, lddb):
    _libmagma.magma_strsm(uplo, trans, diag, m, n, alpha,
                          int(dA), ldda, int(dB), lddb)


# Auxiliary routines:
_libmagma.magma_vec_const.restype = int
_libmagma.magma_vec_const.argtypes = [ctypes.c_char]
def magma_vec_const(job):
    return _libmagma.magma_vec_const(job)

_libmagma.magma_get_spotrf_nb.restype = int
_libmagma.magma_get_spotrf_nb.argtypes = [ctypes.c_int]
def magma_get_spotrf_nb(m):
    return _libmagma.magma_get_spotrf_nb(m)

_libmagma.magma_get_sgetrf_nb.restype = int
_libmagma.magma_get_sgetrf_nb.argtypes = [ctypes.c_int]
def magma_get_sgetrf_nb(m):
    return _libmagma.magma_get_sgetrf_nb(m)

_libmagma.magma_get_sgetri_nb.restype = int
_libmagma.magma_get_sgetri_nb.argtypes = [ctypes.c_int]
def magma_get_sgetri_nb(m):
    return _libmagma.magma_get_sgetri_nb(m)

_libmagma.magma_get_sgeqp3_nb.restype = int
_libmagma.magma_get_sgeqp3_nb.argtypes = [ctypes.c_int]
def magma_get_sgeqp3_nb(m):
    return _libmagma.magma_get_sgeqp3_nb(m)

_libmagma.magma_get_sgeqrf_nb.restype = int
_libmagma.magma_get_sgeqrf_nb.argtypes = [ctypes.c_int]
def magma_get_sgeqrf_nb(m):
    return _libmagma.magma_get_sgeqrf_nb(m)

_libmagma.magma_get_sgeqlf_nb.restype = int
_libmagma.magma_get_sgeqlf_nb.argtypes = [ctypes.c_int]
def magma_get_sgeqlf_nb(m):
    return _libmagma.magma_get_sgeqlf_nb(m)

_libmagma.magma_get_sgehrd_nb.restype = int
_libmagma.magma_get_sgehrd_nb.argtypes = [ctypes.c_int]
def magma_get_sgehrd_nb(m):
    return _libmagma.magma_get_sgehrd_nb(m)

_libmagma.magma_get_ssytrd_nb.restype = int
_libmagma.magma_get_ssytrd_nb.argtypes = [ctypes.c_int]
def magma_get_ssytrd_nb(m):
    return _libmagma.magma_get_ssytrd_nb(m)

_libmagma.magma_get_sgelqf_nb.restype = int
_libmagma.magma_get_sgelqf_nb.argtypes = [ctypes.c_int]
def magma_get_sgelqf_nb(m):
    return _libmagma.magma_get_sgelqf_nb(m)

_libmagma.magma_get_sgebrd_nb.restype = int
_libmagma.magma_get_sgebrd_nb.argtypes = [ctypes.c_int]
def magma_get_sgebrd_nb(m):
    return _libmagma.magma_get_sgebrd_nb(m)

_libmagma.magma_get_ssygst_nb.restype = int
_libmagma.magma_get_ssygst_nb.argtypes = [ctypes.c_int]
def magma_get_ssygst_nb(m):
    return _libmagma.magma_get_ssgyst_nb(m)

_libmagma.magma_get_sgesvd_nb.restype = int
_libmagma.magma_get_sgesvd_nb.argtypes = [ctypes.c_int]
def magma_get_sgesvd_nb(m):
    return _libmagma.magma_get_sgesvd_nb(m)

_libmagma.magma_get_dgesvd_nb.restype = int
_libmagma.magma_get_dgesvd_nb.argtypes = [ctypes.c_int]
def magma_get_dgesvd_nb(m):
    return _libmagma.magma_get_dgesvd_nb(m)

_libmagma.magma_get_cgesvd_nb.restype = int
_libmagma.magma_get_cgesvd_nb.argtypes = [ctypes.c_int]
def magma_get_cgesvd_nb(m):
    return _libmagma.magma_get_cgesvd_nb(m)

_libmagma.magma_get_zgesvd_nb.restype = int
_libmagma.magma_get_zgesvd_nb.argtypes = [ctypes.c_int]
def magma_get_zgesvd_nb(m):
    return _libmagma.magma_get_zgesvd_nb(m)

_libmagma.magma_get_ssygst_nb_m.restype = int
_libmagma.magma_get_ssygst_nb_m.argtypes = [ctypes.c_int]
def magma_get_ssygst_nb_m(m):
    return _libmagma.magma_get_ssgyst_nb_m(m)

_libmagma.magma_get_sbulge_nb.restype = int
_libmagma.magma_get_sbulge_nb.argtypes = [ctypes.c_int]
def magma_get_sbulge_nb(m):
    return _libmagma.magma_get_sbulge_nb(m)

_libmagma.magma_get_dsytrd_nb.restype = int
_libmagma.magma_get_dsytrd_nb.argtypes = [ctypes.c_int]
def magma_get_dsytrd_nb(m):
    return _libmagma.magma_get_dsytrd_nb(m)

# LAPACK routines

# SGEBRD, DGEBRD, CGEBRD, ZGEBRD
_libmagma.magma_sgebrd.restype = int
_libmagma.magma_sgebrd.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p]
def magma_sgebrd(m, n, A, lda, d, e, tauq, taup, work, lwork, info):
    """
    Reduce matrix to bidiagonal form.
    """

    status = _libmagma.magma_sgebrd.argtypes(m, n, int(A), lda,
                                             int(d), int(e),
                                             int(tauq), int(taup),
                                             int(work), int(lwork),
                                             int(info))
    magmaCheckStatus(status)

# SGEHRD2, DGEHRD2, CGEHRD2, ZGEHRD2
_libmagma.magma_sgehrd2.restype = int
_libmagma.magma_sgehrd2.argtypes = [ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p]
def magma_sgehrd2(n, ilo, ihi, A, lda, tau,
                  work, lwork, info):
    """
    Reduce matrix to upper Hessenberg form.
    """

    status = _libmagma.magma_sgehrd2(n, ilo, ihi, int(A), lda,
                                     int(tau), int(work),
                                     lwork, int(info))
    magmaCheckStatus(status)

# SGEHRD, DGEHRD, CGEHRD, ZGEHRD
_libmagma.magma_sgehrd.restype = int
_libmagma.magma_sgehrd.argtypes = [ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p]
def magma_sgehrd(n, ilo, ihi, A, lda, tau,
                 work, lwork, dT, info):
    """
    Reduce matrix to upper Hessenberg form (fast algorithm).
    """

    status = _libmagma.magma_sgehrd(n, ilo, ihi, int(A), lda,
                                    int(tau), int(work),
                                    lwork, int(dT), int(info))
    magmaCheckStatus(status)

# SGELQF, DGELQF, CGELQF, ZGELQF
_libmagma.magma_sgelqf.restype = int
_libmagma.magma_sgelqf.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p]
def magma_sgelqf(m, n, A, lda, tau, work, lwork, info):

    """
    LQ factorization.
    """

    status = _libmagma.magma_sgelqf(m, n, int(A), lda,
                                    int(tau), int(work),
                                    lwork, int(info))
    magmaCheckStatus(status)

# SGEQRF, DGEQRF, CGEQRF, ZGEQRF
_libmagma.magma_sgeqrf.restype = int
_libmagma.magma_sgeqrf.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p]
def magma_sgeqrf(m, n, A, lda, tau, work, lwork, info):

    """
    QR factorization.
    """

    status = _libmagma.magma_sgeqrf(m, n, int(A), lda,
                                    int(tau), int(work),
                                    lwork, int(info))
    magmaCheckStatus(status)

# SGEQRF, DGEQRF, CGEQRF, ZGEQRF (ooc)
_libmagma.magma_sgeqrf_ooc.restype = int
_libmagma.magma_sgeqrf_ooc.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def magma_sgeqrf_ooc(m, n, A, lda, tau, work, lwork, info):

    """
    QR factorization (ooc).
    """

    status = _libmagma.magma_sgeqrf_ooc(m, n, int(A), lda,
                                        int(tau), int(work),
                                        lwork, int(info))
    magmaCheckStatus(status)

# SGESV, DGESV, CGESV, ZGESV
_libmagma.magma_sgesv.restype = int
_libmagma.magma_sgesv.argtypes = [ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p]
def magma_sgesv(n, nhrs, A, lda, ipiv, B, ldb, info):

    """
    Solve system of linear equations.
    """

    status = _libmagma.magma_sgesv(n, nhrs, int(A), lda,
                                   int(ipiv), int(B),
                                   ldb, int(info))
    magmaCheckStatus(status)

# SGETRF, DGETRF, CGETRF, ZGETRF
_libmagma.magma_sgetrf.restype = int
_libmagma.magma_sgetrf.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_sgetrf(m, n, A, lda, ipiv, info):

    """
    LU factorization.
    """

    status = _libmagma.magma_sgetrf(m, n, int(A), lda,
                                    int(ipiv), int(info))
    magmaCheckStatus(status)

## SGETRF2, DGETRF2, CGETRF2, ZGETRF2
#_libmagma.magma_sgetrf2.restype = int
#_libmagma.magma_sgetrf2.argtypes = [ctypes.c_int,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_int,
#                                    ctypes.c_void_p,
#                                    ctypes.c_void_p]
#def magma_sgetrf2(m, n, A, lda, ipiv, info):
#
#    """
#    LU factorization (multi-GPU).
#    """
#
#    status = _libmagma.magma_sgetrf2(m, n, int(A), lda,
#                                    int(ipiv), int(info))
#    magmaCheckStatus(status)

# SGEEV, DGEEV, CGEEV, ZGEEV
_libmagma.magma_sgeev.restype = int
_libmagma.magma_sgeev.argtypes = [ctypes.c_char,
                                  ctypes.c_char,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p]
def magma_sgeev(jobvl, jobvr, n, a, lda,
                w, vl, ldvl, vr, ldvr, work, lwork, rwork, info):

    """
    Compute eigenvalues and eigenvectors.
    """

    status = _libmagma.magma_sgeev(jobvl, jobvr, n, int(a), lda,
                                   int(w), int(vl), ldvl, int(vr), ldvr,
                                   int(work), lwork, int(rwork), int(info))
    magmaCheckStatus(status)

# SGESVD, DGESVD, CGESVD, ZGESVD
_libmagma.magma_sgesvd.restype = int
_libmagma.magma_sgesvd.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p]
def magma_sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 info):
    """
    SVD decomposition.
    """

    status = _libmagma.magma_sgesvd(jobu, jobvt, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(info))
    magmaCheckStatus(status)

_libmagma.magma_dgesvd.restype = int
_libmagma.magma_dgesvd.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p]
def magma_dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 info):
    """
    SVD decomposition.
    """

    status = _libmagma.magma_dgesvd(jobu, jobvt, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(info))
    magmaCheckStatus(status)

_libmagma.magma_cgesvd.restype = int
_libmagma.magma_cgesvd.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p]
def magma_cgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 rwork, info):
    """
    SVD decomposition.
    """

    status = _libmagma.magma_cgesvd(jobu, jobvt, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(rwork),
                                    int(info))
    magmaCheckStatus(status)

_libmagma.magma_zgesvd.restype = int
_libmagma.magma_zgesvd.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_zgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 rwork, info):
    """
    SVD decomposition.
    """

    status = _libmagma.magma_zgesvd(jobu, jobvt, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(rwork), int(info))
    magmaCheckStatus(status)

# SGESDD, DGESDD, CGESDD, ZGESDD
_libmagma.magma_sgesdd.restype = int
_libmagma.magma_sgesdd.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_sgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 iwork, info):
    """
    SDD decomposition.
    """

    status = _libmagma.magma_sgesdd(jobz, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(iwork), int(info))
    magmaCheckStatus(status)

_libmagma.magma_dgesdd.restype = int
_libmagma.magma_dgesdd.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 iwork, info):
    """
    SDD decomposition.
    """

    status = _libmagma.magma_dgesdd(jobz, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(iwork), int(info))
    magmaCheckStatus(status)

_libmagma.magma_cgesdd.restype = int
_libmagma.magma_cgesdd.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_cgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 rwork, iwork, info):
    """
    SDD decomposition.
    """

    status = _libmagma.magma_cgesdd(jobz, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(rwork), int(iwork), int(info))
    magmaCheckStatus(status)

_libmagma.magma_zgesdd.restype = int
_libmagma.magma_zgesdd.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def magma_zgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                 rwork, iwork, info):
    """
    SDD decomposition.
    """

    status = _libmagma.magma_zgesdd(jobz, m, n,
                                    int(a), lda, int(s), int(u), ldu,
                                    int(vt), ldvt, int(work), lwork,
                                    int(rwork), int(iwork), int(info))
    magmaCheckStatus(status)



# SPOSV, DPOSV, CPOSV, ZPOSV
_libmagma.magma_sposv_gpu.restype = int
_libmagma.magma_sposv_gpu.argtypes = [ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p]
def magma_sposv_gpu(uplo, n, nhrs, a_gpu, lda, b_gpu, ldb):
    """
    Solve linear system with positive semidefinite coefficient matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_sposv_gpu(uplo, n, nhrs, int(a_gpu), lda,
                                       int(b_gpu), ldb, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dposv_gpu.restype = int
_libmagma.magma_dposv_gpu.argtypes = _libmagma.magma_sposv_gpu.argtypes
def magma_dposv_gpu(uplo, n, nhrs, a_gpu, lda, b_gpu, ldb):
    """
    Solve linear system with positive semidefinite coefficient matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_dposv_gpu(uplo, n, nhrs, int(a_gpu), lda,
                                       int(b_gpu), ldb, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_cposv_gpu.restype = int
_libmagma.magma_cposv_gpu.argtypes = _libmagma.magma_sposv_gpu.argtypes
def magma_cposv_gpu(uplo, n, nhrs, a_gpu, lda, b_gpu, ldb):
    """
    Solve linear system with positive semidefinite coefficient matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_cposv_gpu(uplo, n, nhrs, int(a_gpu), lda,
                                       int(b_gpu), ldb, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_zposv_gpu.restype = int
_libmagma.magma_zposv_gpu.argtypes = _libmagma.magma_sposv_gpu.argtypes
def magma_zposv_gpu(uplo, n, nhrs, a_gpu, lda, b_gpu, ldb):
    """
    Solve linear system with positive semidefinite coefficient matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_zposv_gpu(uplo, n, nhrs, int(a_gpu), lda,
                                       int(b_gpu), ldb, ctypes.byref(info))
    magmaCheckStatus(status)


# SGESV, DGESV, CGESV, ZGESV
_libmagma.magma_sgesv_gpu.restype = int
_libmagma.magma_sgesv_gpu.argtypes = [ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p]
def magma_sgesv_gpu(n, nhrs, A, lda, ipiv, B, ldb):
    """
    Solve system of linear equations.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_sgesv_gpu(n, nhrs, int(A), lda,
                                   int(ipiv), int(B),
                                   ldb, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_dgesv_gpu.restype = int
_libmagma.magma_dgesv_gpu.argtypes = _libmagma.magma_sgesv_gpu.argtypes
def magma_dgesv_gpu(n, nhrs, A, lda, ipiv, B, ldb):
    """
    Solve system of linear equations.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_dgesv_gpu(n, nhrs, int(A), lda,
                                   int(ipiv), int(B),
                                   ldb, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_cgesv_gpu.restype = int
_libmagma.magma_cgesv_gpu.argtypes = _libmagma.magma_sgesv_gpu.argtypes
def magma_cgesv_gpu(n, nhrs, A, lda, ipiv, B, ldb):
    """
    Solve system of linear equations.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_cgesv_gpu(n, nhrs, int(A), lda,
                                   int(ipiv), int(B),
                                   ldb, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_zgesv_gpu.restype = int
_libmagma.magma_zgesv_gpu.argtypes = _libmagma.magma_sgesv_gpu.argtypes
def magma_zgesv_gpu(n, nhrs, A, lda, ipiv, B, ldb):
    """
    Solve system of linear equations.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_zgesv_gpu(n, nhrs, int(A), lda,
                                   int(ipiv), int(B),
                                   ldb, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_sgesv_nopiv_gpu.restype = int
_libmagma.magma_sgesv_nopiv_gpu.argtypes = [ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_void_p]
def magma_sgesv_nopiv_gpu(n, nhrs, A, lda, B, ldb):
    """
    Solve system of linear equations.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_sgesv_nopiv_gpu(n, nhrs, int(A), lda,
                                             int(B), ldb, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dgesv_nopiv_gpu.restype = int
_libmagma.magma_dgesv_nopiv_gpu.argtypes = _libmagma.magma_sgesv_nopiv_gpu.argtypes
def magma_dgesv_nopiv_gpu(n, nhrs, A, lda, B, ldb):
    """
    Solve system of linear equations.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_dgesv_nopiv_gpu(n, nhrs, int(A), lda,
                                             int(B), ldb, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_cgesv_nopiv_gpu.restype = int
_libmagma.magma_cgesv_nopiv_gpu.argtypes = _libmagma.magma_sgesv_nopiv_gpu.argtypes
def magma_cgesv_nopiv_gpu(n, nhrs, A, lda, B, ldb):
    """
    Solve system of linear equations.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_cgesv_nopiv_gpu(n, nhrs, int(A), lda,
                                             int(B), ldb, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_zgesv_nopiv_gpu.restype = int
_libmagma.magma_zgesv_nopiv_gpu.argtypes = _libmagma.magma_sgesv_nopiv_gpu.argtypes
def magma_zgesv_nopiv_gpu(n, nhrs, A, lda, B, ldb):
    """
    Solve system of linear equations.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_zgesv_nopiv_gpu(n, nhrs, int(A), lda,
                                             int(B), ldb, ctypes.byref(info))
    magmaCheckStatus(status)

# SPOTRF, DPOTRF, CPOTRF, ZPOTRF
_libmagma.magma_spotrf_gpu.restype = int
_libmagma.magma_spotrf_gpu.argtypes = [ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p]
def magma_spotrf_gpu(uplo, n, A, lda):
    """
    Cholesky factorization of positive symmetric matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_spotrf_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_dpotrf_gpu.restype = int
_libmagma.magma_dpotrf_gpu.argtypes = _libmagma.magma_spotrf_gpu.argtypes
def magma_dpotrf_gpu(uplo, n, A, lda):
    """
    Cholesky factorization of positive symmetric matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_dpotrf_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_cpotrf_gpu.restype = int
_libmagma.magma_cpotrf_gpu.argtypes = _libmagma.magma_spotrf_gpu.argtypes
def magma_cpotrf_gpu(uplo, n, A, lda):
    """
    Cholesky factorization of positive symmetric matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_cpotrf_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_zpotrf_gpu.restype = int
_libmagma.magma_zpotrf_gpu.argtypes = _libmagma.magma_zpotrf_gpu.argtypes
def magma_zpotrf_gpu(uplo, n, A, lda):
    """
    Cholesky factorization of positive symmetric matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_zpotrf_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)


# SPOTRI, DPOTRI, CPOTRI, ZPOTRI
_libmagma.magma_spotri_gpu.restype = int
_libmagma.magma_spotri_gpu.argtypes = [ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p]
def magma_spotri_gpu(uplo, n, A, lda):
    """
    Inverse using the Cholesky factorization of positive symmetric matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_spotri_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_dpotri_gpu.restype = int
_libmagma.magma_dpotri_gpu.argtypes = _libmagma.magma_spotri_gpu.argtypes
def magma_dpotri_gpu(uplo, n, A, lda):
    """
    Inverse using the Cholesky factorization of positive symmetric matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_dpotri_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_cpotri_gpu.restype = int
_libmagma.magma_cpotri_gpu.argtypes = _libmagma.magma_spotri_gpu.argtypes
def magma_cpotri_gpu(uplo, n, A, lda):
    """
    Inverse using the Cholesky factorization of positive symmetric matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_cpotri_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_zpotri_gpu.restype = int
_libmagma.magma_zpotri_gpu.argtypes = _libmagma.magma_spotri_gpu.argtypes
def magma_zpotri_gpu(uplo, n, A, lda):
    """
    Inverse using the Cholesky factorization of positive symmetric matrix.
    """
    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_zpotri_gpu(uplo, n, int(A), lda, ctypes.byref(info))
    magmaCheckStatus(status)


# SGETRF, DGETRF, CGETRF, ZGETRF
_libmagma.magma_sgetrf_gpu.restype = int
_libmagma.magma_sgetrf_gpu.argtypes = [ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p]
def magma_sgetrf_gpu(n, m, A, lda, ipiv):
    """
    LU factorization.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_sgetrf_gpu(n, m, int(A), lda,
                                   int(ipiv), ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_dgetrf_gpu.restype = int
_libmagma.magma_dgetrf_gpu.argtypes = _libmagma.magma_sgetrf_gpu.argtypes
def magma_dgetrf_gpu(n, m, A, lda, ipiv):
    """
    LU factorization.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_dgetrf_gpu(n, m, int(A), lda,
                                   int(ipiv), ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_cgetrf_gpu.restype = int
_libmagma.magma_cgetrf_gpu.argtypes = _libmagma.magma_sgetrf_gpu.argtypes
def magma_cgetrf_gpu(n, m, A, lda, ipiv):
    """
    LU factorization.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_cgetrf_gpu(n, m, int(A), lda,
                                   int(ipiv), ctypes.byref(info))
    magmaCheckStatus(status)


_libmagma.magma_zgetrf_gpu.restype = int
_libmagma.magma_zgetrf_gpu.argtypes = _libmagma.magma_sgetrf_gpu.argtypes
def magma_zgetrf_gpu(n, m, A, lda, ipiv):
    """
    LU factorization.
    """
    info = ctypes.c_int()
    status = _libmagma.magma_zgetrf_gpu(n, m, int(A), lda,
                                   int(ipiv), ctypes.byref(info))
    magmaCheckStatus(status)

# SSYEVD, DSYEVD
_libmagma.magma_ssyevd_gpu.restype = int
_libmagma.magma_ssyevd_gpu.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def magma_ssyevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, iwork,
                     liwork):
    """
    Compute eigenvalues of real symmetric matrix.
    """

    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_ssyevd_gpu(jobz, uplo, n, int(dA), ldda,
                                        int(w), int(wA), ldwa, int(work),
                                        lwork, int(iwork), liwork, ctypes.byref(info))
    magmaCheckStatus(status)

_libmagma.magma_dsyevd_gpu.restype = int
_libmagma.magma_dsyevd_gpu.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def magma_dsyevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, iwork,
                     liwork):
    """
    Compute eigenvalues of real symmetric matrix.
    """

    uplo = _uplo_conversion[uplo]
    info = ctypes.c_int()
    status = _libmagma.magma_dsyevd_gpu(jobz, uplo, n, int(dA), ldda,
                                        int(w), int(wA), ldwa, int(work),
                                        lwork, int(iwork), liwork, ctypes.byref(info))
    magmaCheckStatus(status)

# SYMMETRIZE
_libmagma.magmablas_ssymmetrize.restype = int
_libmagma.magmablas_ssymmetrize.argtypes = [ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def magmablas_ssymmetrize(uplo, n, A, lda):
    """
    Symmetrize a triangular matrix.
    """
    uplo = _uplo_conversion[uplo]
    status = _libmagma.magmablas_ssymmetrize(uplo, n, int(A), lda)
    magmaCheckStatus(status)


_libmagma.magmablas_dsymmetrize.restype = int
_libmagma.magmablas_dsymmetrize.argtypes = _libmagma.magmablas_ssymmetrize.argtypes
def magmablas_dsymmetrize(uplo, n, A, lda):
    """
    Symmetrize a triangular matrix.
    """
    uplo = _uplo_conversion[uplo]
    status = _libmagma.magmablas_dsymmetrize(uplo, n, int(A), lda)
    magmaCheckStatus(status)


_libmagma.magmablas_csymmetrize.restype = int
_libmagma.magmablas_csymmetrize.argtypes = _libmagma.magmablas_ssymmetrize.argtypes
def magmablas_csymmetrize(uplo, n, A, lda):
    """
    Symmetrize a triangular matrix.
    """
    uplo = _uplo_conversion[uplo]
    status = _libmagma.magmablas_csymmetrize(uplo, n, int(A), lda)
    magmaCheckStatus(status)


_libmagma.magmablas_zsymmetrize.restype = int
_libmagma.magmablas_zsymmetrize.argtypes = _libmagma.magmablas_ssymmetrize.argtypes
def magmablas_zsymmetrize(uplo, n, A, lda):
    """
    Symmetrize a triangular matrix.
    """
    uplo = _uplo_conversion[uplo]
    status = _libmagma.magmablas_zsymmetrize(uplo, n, int(A), lda)
    magmaCheckStatus(status)


_libmagma_sparse.magma_zmfree.restype = int
_libmagma_sparse.magma_zmfree.argtypes = [ctypes.POINTER(magma_z_matrix), magma_queue_t]
def magma_zmfree(A, queue):
    status = _libmagma_sparse.magma_zmfree(ctypes.byref(A), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_cmfree.restype = int
_libmagma_sparse.magma_cmfree.argtypes = [ctypes.POINTER(magma_c_matrix), magma_queue_t]
def magma_cmfree(A, queue):
    status = _libmagma_sparse.magma_cmfree(ctypes.byref(A), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_dmfree.restype = int
_libmagma_sparse.magma_dmfree.argtypes = [ctypes.POINTER(magma_d_matrix), magma_queue_t]
def magma_dmfree(A, queue):
    status = _libmagma_sparse.magma_dmfree(ctypes.byref(A), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_smfree.restype = int
_libmagma_sparse.magma_smfree.argtypes = [ctypes.POINTER(magma_s_matrix), magma_queue_t]
def magma_smfree(A, queue):
    status = _libmagma_sparse.magma_smfree(ctypes.byref(A), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_zcsrset.restype  = int
_libmagma_sparse.magma_zcsrset.argtypes = [ctypes.c_int,
                                           ctypes.c_int,
                                           ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                           ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                           ndpointer(np.complex128, flags="C_CONTIGUOUS"),
                                           ctypes.POINTER(magma_z_matrix),
                                           magma_queue_t]
def magma_zcsrset(m, n, row, col, val, A, queue):
    #TODO: check that dtype is np.int32. np.ndarray(row, dtype=np.int32)
    status = _libmagma_sparse.magma_zcsrset(m, n, row, col, val, ctypes.byref(A), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_ccsrset.restype  = int
_libmagma_sparse.magma_ccsrset.argtypes = [ctypes.c_int,
                                           ctypes.c_int,
                                           ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                           ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                           ndpointer(np.complex64, flags="C_CONTIGUOUS"),
                                           ctypes.POINTER(magma_c_matrix),
                                           magma_queue_t]
def magma_ccsrset(m, n, row, col, val, A, queue):
    #TODO: check that dtype is np.int32. np.ndarray(row, dtype=np.int32)
    status = _libmagma_sparse.magma_ccsrset(m, n, row, col, val, ctypes.byref(A), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_dcsrset.restype  = int
_libmagma_sparse.magma_dcsrset.argtypes = [ctypes.c_int,
                                           ctypes.c_int,
                                           ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                           ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                           ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                           ctypes.POINTER(magma_d_matrix),
                                           magma_queue_t]
def magma_dcsrset(m, n, row, col, val, A, queue):
    #TODO: check that dtype is np.int32. np.ndarray(row, dtype=np.int32)
    status = _libmagma_sparse.magma_dcsrset(m, n, row, col, val, ctypes.byref(A), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_scsrset.restype  = int
_libmagma_sparse.magma_scsrset.argtypes = [ctypes.c_int,
                                           ctypes.c_int,
                                           ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                           ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                           ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                           ctypes.POINTER(magma_s_matrix),
                                           magma_queue_t]
def magma_scsrset(m, n, row, col, val, A, queue):
    #TODO: check that dtype is np.int32. np.ndarray(row, dtype=np.int32)
    status = _libmagma_sparse.magma_scsrset(m, n, row, col, val, ctypes.byref(A), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_zcsrget.restype  = int
_libmagma_sparse.magma_zcsrget.argtypes = [magma_d_matrix,
                                           ctypes.POINTER(magma_int_t),
                                           ctypes.POINTER(magma_int_t),
                                           ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                                           ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                                           #ctypes.POINTER(ndpointer(np.complex128, flags="C_CONTIGUOUS")),
                                           ctypes.POINTER(ctypes.c_void_p),
                                           magma_queue_t]
def magma_zcsrget(A, queue):
    m     = ctypes.c_int()
    n     = ctypes.c_int()
    val_p = ctypes.c_void_p()
    #val_p = ctypes.POINTER(ndpointer(np.complex128, flags="C_CONTIGUOUS"))()
    row_p = ctypes.POINTER(ctypes.c_int)()
    col_p = ctypes.POINTER(ctypes.c_int)()

    status = _libmagma_sparse.magma_zcsrget(A, ctypes.byref(m), ctypes.byref(n), ctypes.byref(row_p), ctypes.byref(col_p), ctypes.byref(val_p), queue)
    magmaCheckStatus(status)

    val_p = ctypes.cast(val_p, ndpointer(np.complex128, ndim=1, shape = (A.nnz,), flags="C_CONTIGUOUS"))

    val = np.ctypeslib.as_array(val_p, shape = (A.nnz,))
    row = np.ctypeslib.as_array(row_p, shape = (A.nnz,))
    col = np.ctypeslib.as_array(col_p, shape = (A.nnz,))

    return m.value, n.value, row, col, val


_libmagma_sparse.magma_ccsrget.restype  = int
_libmagma_sparse.magma_ccsrget.argtypes = [magma_c_matrix,
                                           ctypes.POINTER(magma_int_t),
                                           ctypes.POINTER(magma_int_t),
                                           ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                                           ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                                           ctypes.POINTER(ctypes.c_void_p),
                                           magma_queue_t]
def magma_ccsrget(A, queue):
    m     = ctypes.c_int()
    n     = ctypes.c_int()
    val_p = ctypes.c_void_p()
    row_p = ctypes.POINTER(ctypes.c_int)()
    col_p = ctypes.POINTER(ctypes.c_int)()

    status = _libmagma_sparse.magma_ccsrget(A, ctypes.byref(m), ctypes.byref(n), ctypes.byref(row_p), ctypes.byref(col_p), ctypes.byref(val_p), queue)
    magmaCheckStatus(status)

    val_p = ctypes.cast(val_p, ndpointer(np.complex64, ndim=1, shape = (A.nnz,), flags="C_CONTIGUOUS"))

    val = np.ctypeslib.as_array(val_p, shape = (A.nnz,))
    row = np.ctypeslib.as_array(row_p, shape = (A.nnz,))
    col = np.ctypeslib.as_array(col_p, shape = (A.nnz,))

    return m.value, n.value, row, col, val


_libmagma_sparse.magma_dcsrget.restype  = int
_libmagma_sparse.magma_dcsrget.argtypes = [magma_d_matrix,
                                           ctypes.POINTER(magma_int_t),
                                           ctypes.POINTER(magma_int_t),
                                           ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                                           ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                                           ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                                           magma_queue_t]
def magma_dcsrget(A, queue):
    m     = ctypes.c_int()
    n     = ctypes.c_int()
    val_p = ctypes.POINTER(ctypes.c_double)()
    row_p = ctypes.POINTER(ctypes.c_int)()
    col_p = ctypes.POINTER(ctypes.c_int)()

    status = _libmagma_sparse.magma_dcsrget(A, ctypes.byref(m), ctypes.byref(n), ctypes.byref(row_p), ctypes.byref(col_p), ctypes.byref(val_p), queue)
    magmaCheckStatus(status)

    val = np.ctypeslib.as_array(val_p, shape = (A.nnz,))
    row = np.ctypeslib.as_array(row_p, shape = (A.nnz,))
    col = np.ctypeslib.as_array(col_p, shape = (A.nnz,))

    return m.value, n.value, row, col, val


_libmagma_sparse.magma_scsrget.restype  = int
_libmagma_sparse.magma_scsrget.argtypes = [magma_s_matrix,
                                           ctypes.POINTER(magma_int_t),
                                           ctypes.POINTER(magma_int_t),
                                           ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                                           ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                                           ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                                           magma_queue_t]
def magma_scsrget(A, queue):
    m     = ctypes.c_int()
    n     = ctypes.c_int()
    val_p = ctypes.POINTER(ctypes.c_float)()
    row_p = ctypes.POINTER(ctypes.c_int)()
    col_p = ctypes.POINTER(ctypes.c_int)()

    status = _libmagma_sparse.magma_scsrget(A, ctypes.byref(m), ctypes.byref(n), ctypes.byref(row_p), ctypes.byref(col_p), ctypes.byref(val_p), queue)
    magmaCheckStatus(status)

    val = np.ctypeslib.as_array(val_p, shape = (A.nnz,))
    row = np.ctypeslib.as_array(row_p, shape = (A.nnz,))
    col = np.ctypeslib.as_array(col_p, shape = (A.nnz,))

    return m.value, n.value, row, col, val


_libmagma_sparse.magma_zvset.restype = int
_libmagma_sparse.magma_zvset.argtypes = [ctypes.c_int,
                                         ctypes.c_int,
                                         ndpointer(np.complex128, flags="C_CONTIGUOUS"),
                                         ctypes.POINTER(magma_z_matrix),
                                         magma_queue_t]
def magma_zvset(m, n, val, v, queue):
    status = _libmagma_sparse.magma_zvset(m, n, val, ctypes.byref(v), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_cvset.restype = int
_libmagma_sparse.magma_cvset.argtypes = [ctypes.c_int,
                                         ctypes.c_int,
                                         ndpointer(np.complex64, flags="C_CONTIGUOUS"),
                                         ctypes.POINTER(magma_c_matrix),
                                         magma_queue_t]
def magma_cvset(m, n, val, v, queue):
    status = _libmagma_sparse.magma_cvset(m, n, val, ctypes.byref(v), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_dvset.restype = int
_libmagma_sparse.magma_dvset.argtypes = [ctypes.c_int,
                                         ctypes.c_int,
                                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                         ctypes.POINTER(magma_d_matrix),
                                         magma_queue_t]
def magma_dvset(m, n, val, v, queue):
    status = _libmagma_sparse.magma_dvset(m, n, val, ctypes.byref(v), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_svset.restype = int
_libmagma_sparse.magma_svset.argtypes = [ctypes.c_int,
                                         ctypes.c_int,
                                         ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                         ctypes.POINTER(magma_s_matrix),
                                         magma_queue_t]
def magma_svset(m, n, val, v, queue):
    status = _libmagma_sparse.magma_svset(m, n, val, ctypes.byref(v), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_zvget.restype  = int
_libmagma_sparse.magma_zvget.argtypes = [magma_z_matrix,
                                         ctypes.POINTER(magma_int_t),
                                         ctypes.POINTER(magma_int_t),
                                         ctypes.POINTER(ctypes.c_void_p),
                                         magma_queue_t]
def magma_zvget(v, queue):
    m     = ctypes.c_int()
    n     = ctypes.c_int()
    val_p = ctypes.c_void_p()

    status = _libmagma_sparse.magma_zvget(v, ctypes.byref(m), ctypes.byref(n), ctypes.byref(val_p), queue)
    magmaCheckStatus(status)

    val_p = ctypes.cast(val_p, ndpointer(np.complex128, ndim=1, shape = (v.nnz,), flags="C_CONTIGUOUS"))
    val = np.ctypeslib.as_array(val_p, shape = (v.nnz,))

    return m.value, n.value, val


_libmagma_sparse.magma_cvget.restype  = int
_libmagma_sparse.magma_cvget.argtypes = [magma_c_matrix,
                                         ctypes.POINTER(magma_int_t),
                                         ctypes.POINTER(magma_int_t),
                                         ctypes.POINTER(ctypes.c_void_p),
                                         magma_queue_t]
def magma_cvget(v, queue):
    m     = ctypes.c_int()
    n     = ctypes.c_int()
    val_p = ctypes.c_void_p()

    status = _libmagma_sparse.magma_cvget(v, ctypes.byref(m), ctypes.byref(n), ctypes.byref(val_p), queue)
    magmaCheckStatus(status)

    val_p = ctypes.cast(val_p, ndpointer(np.complex64, ndim=1, shape = (v.nnz,), flags="C_CONTIGUOUS"))
    val = np.ctypeslib.as_array(val_p, shape = (v.nnz,))

    return m.value, n.value, val


_libmagma_sparse.magma_dvget.restype  = int
_libmagma_sparse.magma_dvget.argtypes = [magma_d_matrix,
                                         ctypes.POINTER(magma_int_t),
                                         ctypes.POINTER(magma_int_t),
                                         ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                                         magma_queue_t]
def magma_dvget(v, queue):
    m     = ctypes.c_int()
    n     = ctypes.c_int()
    val_p = ctypes.POINTER(ctypes.c_double)()

    status = _libmagma_sparse.magma_dvget(v, ctypes.byref(m), ctypes.byref(n), ctypes.byref(val_p), queue)
    magmaCheckStatus(status)

    val = np.ctypeslib.as_array(val_p, shape = (v.nnz,))

    return m.value, n.value, val


_libmagma_sparse.magma_svget.restype  = int
_libmagma_sparse.magma_svget.argtypes = [magma_s_matrix,
                                         ctypes.POINTER(magma_int_t),
                                         ctypes.POINTER(magma_int_t),
                                         ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                                         magma_queue_t]
def magma_svget(v, queue):
    m     = ctypes.c_int()
    n     = ctypes.c_int()
    val_p = ctypes.POINTER(ctypes.c_float)()

    status = _libmagma_sparse.magma_svget(v, ctypes.byref(m), ctypes.byref(n), ctypes.byref(val_p), queue)
    magmaCheckStatus(status)

    val = np.ctypeslib.as_array(val_p, shape = (v.nnz,))

    return m.value, n.value, val


_libmagma_sparse.magma_zvinit.restype  = int
_libmagma_sparse.magma_zvinit.argtypes = [ctypes.POINTER(magma_z_matrix),
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          cuda.cuDoubleComplex,
                                          magma_queue_t]
def magma_zvinit(x, memory_location, num_rows, num_cols, values, queue):
    status = _libmagma_sparse.magma_zvinit(ctypes.byref(x), memory_location, num_rows, num_cols, values, queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_cvinit.restype  = int
_libmagma_sparse.magma_cvinit.argtypes = [ctypes.POINTER(magma_c_matrix),
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          cuda.cuFloatComplex,
                                          magma_queue_t]
def magma_cvinit(x, memory_location, num_rows, num_cols, values, queue):
    status = _libmagma_sparse.magma_cvinit(ctypes.byref(x), memory_location, num_rows, num_cols, values, queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_dvinit.restype  = int
_libmagma_sparse.magma_dvinit.argtypes = [ctypes.POINTER(magma_d_matrix),
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_double,
                                          magma_queue_t]
def magma_dvinit(x, memory_location, num_rows, num_cols, values, queue):
    status = _libmagma_sparse.magma_dvinit(ctypes.byref(x), memory_location, num_rows, num_cols, values, queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_svinit.restype  = int
_libmagma_sparse.magma_svinit.argtypes = [ctypes.POINTER(magma_s_matrix),
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_float,
                                          magma_queue_t]
def magma_svinit(x, memory_location, num_rows, num_cols, values, queue):
    status = _libmagma_sparse.magma_svinit(ctypes.byref(x), memory_location, num_rows, num_cols, values, queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_zmtransfer.restype = int
_libmagma_sparse.magma_zmtransfer.argtypes = [magma_z_matrix,
                                              ctypes.POINTER(magma_z_matrix),
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              magma_queue_t]
def magma_zmtransfer(A, B, src, dst, queue):
    status = _libmagma_sparse.magma_zmtransfer(A, ctypes.byref(B), src, dst, queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_cmtransfer.restype = int
_libmagma_sparse.magma_cmtransfer.argtypes = [magma_c_matrix,
                                              ctypes.POINTER(magma_c_matrix),
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              magma_queue_t]
def magma_cmtransfer(A, B, src, dst, queue):
    status = _libmagma_sparse.magma_cmtransfer(A, ctypes.byref(B), src, dst, queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_dmtransfer.restype = int
_libmagma_sparse.magma_dmtransfer.argtypes = [magma_d_matrix,
                                              ctypes.POINTER(magma_d_matrix),
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              magma_queue_t]
def magma_dmtransfer(A, B, src, dst, queue):
    status = _libmagma_sparse.magma_dmtransfer(A, ctypes.byref(B), src, dst, queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_smtransfer.restype = int
_libmagma_sparse.magma_smtransfer.argtypes = [magma_s_matrix,
                                              ctypes.POINTER(magma_s_matrix),
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              magma_queue_t]
def magma_smtransfer(A, B, src, dst, queue):
    status = _libmagma_sparse.magma_smtransfer(A, ctypes.byref(B), src, dst, queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_zmscale.restype = int
_libmagma_sparse.magma_zmscale.argtypes = [ctypes.POINTER(magma_z_matrix),
                                           ctypes.c_int,
                                           magma_queue_t]
def magma_zmscale(A, scaling, queue):
    _libmagma_sparse.magma_zmscale(ctypes.byref(A), scaling, queue)


_libmagma_sparse.magma_cmscale.restype = int
_libmagma_sparse.magma_cmscale.argtypes = [ctypes.POINTER(magma_c_matrix),
                                           ctypes.c_int,
                                           magma_queue_t]
def magma_cmscale(A, scaling, queue):
    _libmagma_sparse.magma_cmscale(ctypes.byref(A), scaling, queue)


_libmagma_sparse.magma_dmscale.restype = int
_libmagma_sparse.magma_dmscale.argtypes = [ctypes.POINTER(magma_d_matrix),
                                           ctypes.c_int,
                                           magma_queue_t]
def magma_dmscale(A, scaling, queue):
    _libmagma_sparse.magma_dmscale(ctypes.byref(A), scaling, queue)


_libmagma_sparse.magma_smscale.restype = int
_libmagma_sparse.magma_smscale.argtypes = [ctypes.POINTER(magma_s_matrix),
                                           ctypes.c_int,
                                           magma_queue_t]
def magma_smscale(A, scaling, queue):
    _libmagma_sparse.magma_smscale(ctypes.byref(A), scaling, queue)


_libmagma_sparse.magma_zmconvert.restype = int
_libmagma_sparse.magma_zmconvert.argtypes = [magma_z_matrix,
                                               ctypes.POINTER(magma_z_matrix),
                                               ctypes.c_int,
                                               ctypes.c_int,
                                               magma_queue_t]
def magma_zmconvert(A, B, old_format, new_format, queue):
    status = _libmagma_sparse.magma_zmconvert(A, ctypes.byref(B), old_format, new_format, queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_cmconvert.restype = int
_libmagma_sparse.magma_cmconvert.argtypes = [magma_c_matrix,
                                             ctypes.POINTER(magma_c_matrix),
                                             ctypes.c_int,
                                             ctypes.c_int,
                                             magma_queue_t]
def magma_cmconvert(A, B, old_format, new_format, queue):
    status = _libmagma_sparse.magma_cmconvert(A, ctypes.byref(B), old_format, new_format, queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_dmconvert.restype = int
_libmagma_sparse.magma_dmconvert.argtypes = [magma_d_matrix,
                                              ctypes.POINTER(magma_d_matrix),
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              magma_queue_t]
def magma_dmconvert(A, B, old_format, new_format, queue):
    status = _libmagma_sparse.magma_dmconvert(A, ctypes.byref(B), old_format, new_format, queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_smconvert.restype = int
_libmagma_sparse.magma_smconvert.argtypes = [magma_s_matrix,
                                             ctypes.POINTER(magma_s_matrix),
                                             ctypes.c_int,
                                             ctypes.c_int,
                                             magma_queue_t]
def magma_smconvert(A, B, old_format, new_format, queue):
    status = _libmagma_sparse.magma_smconvert(A, ctypes.byref(B), old_format, new_format, queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_zm_5stencil.restype  = int
_libmagma_sparse.magma_zm_5stencil.argtypes = [ctypes.c_int,
                                               ctypes.POINTER(magma_z_matrix),
                                               magma_queue_t]
def magma_zm_5stencil(laplace_size, A, queue):
    status = _libmagma_sparse.magma_zm_5stencil(laplace_size, ctypes.byref(A), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_cm_5stencil.restype  = int
_libmagma_sparse.magma_cm_5stencil.argtypes = [ctypes.c_int,
                                               ctypes.POINTER(magma_c_matrix),
                                               magma_queue_t]
def magma_cm_5stencil(laplace_size, A, queue):
    status = _libmagma_sparse.magma_cm_5stencil(laplace_size, ctypes.byref(A), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_dm_5stencil.restype  = int
_libmagma_sparse.magma_dm_5stencil.argtypes = [ctypes.c_int,
                                               ctypes.POINTER(magma_d_matrix),
                                               magma_queue_t]
def magma_dm_5stencil(laplace_size, A, queue):
    status = _libmagma_sparse.magma_dm_5stencil(laplace_size, ctypes.byref(A), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_sm_5stencil.restype  = int
_libmagma_sparse.magma_sm_5stencil.argtypes = [ctypes.c_int,
                                               ctypes.POINTER(magma_s_matrix),
                                               magma_queue_t]
def magma_sm_5stencil(laplace_size, A, queue):
    status = _libmagma_sparse.magma_sm_5stencil(laplace_size, ctypes.byref(A), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_z_spmv.restype  = int
_libmagma_sparse.magma_z_spmv.argtypes = [cuda.cuDoubleComplex,
                                          magma_z_matrix,
                                          magma_z_matrix,
                                          cuda.cuDoubleComplex,
                                          magma_z_matrix,
                                          magma_queue_t]
def magma_z_spmv(alpha, A, x, beta, y, queue):
    status = _libmagma_sparse.magma_z_spmv(alpha, A, x, beta, y, queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_c_spmv.restype  = int
_libmagma_sparse.magma_c_spmv.argtypes = [cuda.cuFloatComplex,
                                          magma_c_matrix,
                                          magma_c_matrix,
                                          cuda.cuFloatComplex,
                                          magma_c_matrix,
                                          magma_queue_t]
def magma_c_spmv(alpha, A, x, beta, y, queue):
    status = _libmagma_sparse.magma_c_spmv(alpha, A, x, beta, y, queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_d_spmv.restype  = int
_libmagma_sparse.magma_d_spmv.argtypes = [ctypes.c_double,
                                          magma_d_matrix,
                                          magma_d_matrix,
                                          ctypes.c_double,
                                          magma_d_matrix,
                                          magma_queue_t]
def magma_d_spmv(alpha, A, x, beta, y, queue):
    status = _libmagma_sparse.magma_d_spmv(alpha, A, x, beta, y, queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_s_spmv.restype  = int
_libmagma_sparse.magma_s_spmv.argtypes = [ctypes.c_float,
                                          magma_s_matrix,
                                          magma_s_matrix,
                                          ctypes.c_float,
                                          magma_s_matrix,
                                          magma_queue_t]
def magma_s_spmv(alpha, A, x, beta, y, queue):
    status = _libmagma_sparse.magma_s_spmv(alpha, A, x, beta, y, queue)
    magmaCheckStatus(status)


_libmagma.magma_queue_create_v2_internal.restype = int
_libmagma.magma_queue_create_v2_internal.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.POINTER(magma_queue)), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
def magma_queue_create(device, queue):
    status = _libmagma.magma_queue_create_v2_internal(device, ctypes.byref(queue), 'magma_queue_create', __file__, inspect.currentframe().f_back.f_lineno)
    magmaCheckStatus(status)


_libmagma.magma_queue_destroy_internal.restype = int
_libmagma.magma_queue_destroy_internal.argtypes = [ctypes.POINTER(magma_queue), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
def magma_queue_destroy(queue):
    status = _libmagma.magma_queue_destroy_internal(queue, 'magma_queue_destroy', __file__, inspect.currentframe().f_back.f_lineno)
    magmaCheckStatus(status)


_libmagma.magma_queue_get_device.restype = int
_libmagma.magma_queue_get_device.argtypes = [ctypes.POINTER(magma_queue)]
def magma_queue_get_device(queue):
    return _libmagma.magma_queue_get_device(queue)


_libmagma_sparse.magma_zsolverinfo_init.restype = int
_libmagma_sparse.magma_zsolverinfo_init.argtypes = [ctypes.POINTER(magma_z_solver_par),
                                                    ctypes.POINTER(magma_z_preconditioner),
                                                    magma_queue_t]
def magma_zsolverinfo_init(solver_par, precond, queue):
    status = _libmagma_sparse.magma_zsolverinfo_init(ctypes.byref(solver_par), ctypes.byref(precond), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_csolverinfo_init.restype = int
_libmagma_sparse.magma_csolverinfo_init.argtypes = [ctypes.POINTER(magma_c_solver_par),
                                                    ctypes.POINTER(magma_c_preconditioner),
                                                    magma_queue_t]
def magma_csolverinfo_init(solver_par, precond, queue):
    status = _libmagma_sparse.magma_csolverinfo_init(ctypes.byref(solver_par), ctypes.byref(precond), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_dsolverinfo_init.restype = int
_libmagma_sparse.magma_dsolverinfo_init.argtypes = [ctypes.POINTER(magma_d_solver_par),
                                                    ctypes.POINTER(magma_d_preconditioner),
                                                    magma_queue_t]
def magma_dsolverinfo_init(solver_par, precond, queue):
    status = _libmagma_sparse.magma_dsolverinfo_init(ctypes.byref(solver_par), ctypes.byref(precond), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_ssolverinfo_init.restype = int
_libmagma_sparse.magma_ssolverinfo_init.argtypes = [ctypes.POINTER(magma_s_solver_par),
                                                    ctypes.POINTER(magma_s_preconditioner),
                                                    magma_queue_t]
def magma_ssolverinfo_init(solver_par, precond, queue):
    status = _libmagma_sparse.magma_ssolverinfo_init(ctypes.byref(solver_par), ctypes.byref(precond), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_zeigensolverinfo_init.restype = int
_libmagma_sparse.magma_zeigensolverinfo_init.argtypes = [ctypes.POINTER(magma_z_solver_par),
                                                         magma_queue_t]
def magma_zeigensolverinfo_init(solver_par, queue):
    status = _libmagma_sparse.magma_zeigensolverinfo_init(ctypes.byref(solver_par), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_ceigensolverinfo_init.restype = int
_libmagma_sparse.magma_ceigensolverinfo_init.argtypes = [ctypes.POINTER(magma_c_solver_par),
                                                         magma_queue_t]
def magma_ceigensolverinfo_init(solver_par, queue):
    status = _libmagma_sparse.magma_ceigensolverinfo_init(ctypes.byref(solver_par), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_deigensolverinfo_init.restype = int
_libmagma_sparse.magma_deigensolverinfo_init.argtypes = [ctypes.POINTER(magma_d_solver_par),
                                                         magma_queue_t]
def magma_deigensolverinfo_init(solver_par, queue):
    status = _libmagma_sparse.magma_deigensolverinfo_init(ctypes.byref(solver_par), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_seigensolverinfo_init.restype = int
_libmagma_sparse.magma_seigensolverinfo_init.argtypes = [ctypes.POINTER(magma_s_solver_par),
                                                         magma_queue_t]
def magma_seigensolverinfo_init(solver_par, queue):
    status = _libmagma_sparse.magma_seigensolverinfo_init(ctypes.byref(solver_par), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_zsolverinfo.restype = int
_libmagma_sparse.magma_zsolverinfo.argtypes = [ctypes.POINTER(magma_z_solver_par),
                                               ctypes.POINTER(magma_z_preconditioner),
                                               magma_queue_t]
def magma_zsolverinfo(solver_par, precond_par, queue):
    status = _libmagma_sparse.magma_zsolverinfo(ctypes.byref(solver_par), ctypes.byref(precond_par), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_csolverinfo.restype = int
_libmagma_sparse.magma_csolverinfo.argtypes = [ctypes.POINTER(magma_c_solver_par),
                                               ctypes.POINTER(magma_c_preconditioner),
                                               magma_queue_t]
def magma_csolverinfo(solver_par, precond_par, queue):
    status = _libmagma_sparse.magma_csolverinfo(ctypes.byref(solver_par), ctypes.byref(precond_par), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_dsolverinfo.restype = int
_libmagma_sparse.magma_dsolverinfo.argtypes = [ctypes.POINTER(magma_d_solver_par),
                                               ctypes.POINTER(magma_d_preconditioner),
                                               magma_queue_t]
def magma_dsolverinfo(solver_par, precond_par, queue):
    status = _libmagma_sparse.magma_dsolverinfo(ctypes.byref(solver_par), ctypes.byref(precond_par), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_ssolverinfo.restype = int
_libmagma_sparse.magma_ssolverinfo.argtypes = [ctypes.POINTER(magma_s_solver_par),
                                               ctypes.POINTER(magma_s_preconditioner),
                                               magma_queue_t]
def magma_ssolverinfo(solver_par, precond_par, queue):
    status = _libmagma_sparse.magma_ssolverinfo(ctypes.byref(solver_par), ctypes.byref(precond_par), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_zsolverinfo_free.restype = int
_libmagma_sparse.magma_zsolverinfo_free.argtypes = [ctypes.POINTER(magma_z_solver_par),
                                                    ctypes.POINTER(magma_z_preconditioner),
                                                    magma_queue_t]
def magma_zsolverinfo_free(solver_par, precond, queue):
    status = _libmagma_sparse.magma_zsolverinfo_free(ctypes.byref(solver_par), ctypes.byref(precond), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_csolverinfo_free.restype = int
_libmagma_sparse.magma_csolverinfo_free.argtypes = [ctypes.POINTER(magma_c_solver_par),
                                                    ctypes.POINTER(magma_c_preconditioner),
                                                    magma_queue_t]
def magma_csolverinfo_free(solver_par, precond, queue):
    status = _libmagma_sparse.magma_csolverinfo_free(ctypes.byref(solver_par), ctypes.byref(precond), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_dsolverinfo_free.restype = int
_libmagma_sparse.magma_dsolverinfo_free.argtypes = [ctypes.POINTER(magma_d_solver_par),
                                                    ctypes.POINTER(magma_d_preconditioner),
                                                    magma_queue_t]
def magma_dsolverinfo_free(solver_par, precond, queue):
    status = _libmagma_sparse.magma_dsolverinfo_free(ctypes.byref(solver_par), ctypes.byref(precond), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_ssolverinfo_free.restype = int
_libmagma_sparse.magma_ssolverinfo_free.argtypes = [ctypes.POINTER(magma_s_solver_par),
                                                    ctypes.POINTER(magma_s_preconditioner),
                                                    magma_queue_t]
def magma_ssolverinfo_free(solver_par, precond, queue):
    status = _libmagma_sparse.magma_ssolverinfo_free(ctypes.byref(solver_par), ctypes.byref(precond), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_z_solver.restype = int
_libmagma_sparse.magma_z_solver.argtypes = [magma_z_matrix,
                                            magma_z_matrix,
                                            ctypes.POINTER(magma_z_matrix),
                                            ctypes.POINTER(magma_zopts),
                                            magma_queue_t]
def magma_z_solver(A, b, x, opts, queue):
    return _libmagma_sparse.magma_z_solver(A, b, ctypes.byref(x), ctypes.byref(opts), queue)


_libmagma_sparse.magma_c_solver.restype = int
_libmagma_sparse.magma_c_solver.argtypes = [magma_c_matrix,
                                            magma_c_matrix,
                                            ctypes.POINTER(magma_c_matrix),
                                            ctypes.POINTER(magma_copts),
                                            magma_queue_t]
def magma_c_solver(A, b, x, opts, queue):
    return _libmagma_sparse.magma_c_solver(A, b, ctypes.byref(x), ctypes.byref(opts), queue)


_libmagma_sparse.magma_d_solver.restype = int
_libmagma_sparse.magma_d_solver.argtypes = [magma_d_matrix,
                                            magma_d_matrix,
                                            ctypes.POINTER(magma_d_matrix),
                                            ctypes.POINTER(magma_dopts),
                                            magma_queue_t]
def magma_d_solver(A, b, x, opts, queue):
    return _libmagma_sparse.magma_d_solver(A, b, ctypes.byref(x), ctypes.byref(opts), queue)


_libmagma_sparse.magma_s_solver.restype = int
_libmagma_sparse.magma_s_solver.argtypes = [magma_s_matrix,
                                            magma_s_matrix,
                                            ctypes.POINTER(magma_s_matrix),
                                            ctypes.POINTER(magma_sopts),
                                            magma_queue_t]
def magma_s_solver(A, b, x, opts, queue):
    return _libmagma_sparse.magma_s_solver(A, b, ctypes.byref(x), ctypes.byref(opts), queue)


_libmagma_sparse.magma_z_precondsetup.restype  = int
_libmagma_sparse.magma_z_precondsetup.argtypes = [magma_z_matrix,
                                                  magma_z_matrix,
                                                  ctypes.POINTER(magma_z_solver_par),
                                                  ctypes.POINTER(magma_z_preconditioner),
                                                  magma_queue_t]
def magma_z_precondsetup(A, b, solver, precond, queue):
    status = _libmagma_sparse.magma_z_precondsetup(A, b, ctypes.byref(solver), ctypes.byref(precond), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_c_precondsetup.restype  = int
_libmagma_sparse.magma_c_precondsetup.argtypes = [magma_c_matrix,
                                                  magma_c_matrix,
                                                  ctypes.POINTER(magma_c_solver_par),
                                                  ctypes.POINTER(magma_c_preconditioner),
                                                  magma_queue_t]
def magma_c_precondsetup(A, b, solver, precond, queue):
    status = _libmagma_sparse.magma_c_precondsetup(A, b, ctypes.byref(solver), ctypes.byref(precond), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_d_precondsetup.restype  = int
_libmagma_sparse.magma_d_precondsetup.argtypes = [magma_d_matrix,
                                                  magma_d_matrix,
                                                  ctypes.POINTER(magma_d_solver_par),
                                                  ctypes.POINTER(magma_d_preconditioner),
                                                  magma_queue_t]
def magma_d_precondsetup(A, b, solver, precond, queue):
    status = _libmagma_sparse.magma_d_precondsetup(A, b, ctypes.byref(solver), ctypes.byref(precond), queue)
    magmaCheckStatus(status)


_libmagma_sparse.magma_s_precondsetup.restype  = int
_libmagma_sparse.magma_s_precondsetup.argtypes = [magma_s_matrix,
                                                  magma_s_matrix,
                                                  ctypes.POINTER(magma_s_solver_par),
                                                  ctypes.POINTER(magma_s_preconditioner),
                                                  magma_queue_t]
def magma_s_precondsetup(A, b, solver, precond, queue):
    status = _libmagma_sparse.magma_s_precondsetup(A, b, ctypes.byref(solver), ctypes.byref(precond), queue)
    magmaCheckStatus(status)


def free(A):
    if isinstance(A, magma_vector):
        cuda.cudaFree(A.pointer)
    elif isinstance(A, magma_matrix):
        if A.dtype == np.complex128:
            magma_zmfree(A.magma_t_matrix, A.queue)
        elif A.dtype == np.complex64:
            magma_cmfree(A.magma_t_matrix, A.queue)
        elif A.dtype == np.float64:
            magma_dmfree(A.magma_t_matrix, A.queue)
        elif A.dtype == np.float32:
            magma_smfree(A.magma_t_matrix, A.queue)
        else:
            raise ValueError('free: Type not supported.')


def magma_free(A):
    free(A)


def magma_csrset(m, n, indptr, indices, val, A):
    """
    A is a magma_matrix
    m is int
    n is int
    val is ndarray of type dtype
    indptr is ndarray of type int32
    indices is ndarray of type int32
    """

    if val.dtype == np.complex128:
        magma_zcsrset(m, n, indptr, indices, val, A.magma_t_matrix, A.queue)
    elif val.dtype == np.complex64:
        magma_ccsrset(m, n, indptr, indices, val, A.magma_t_matrix, A.queue)
    elif val.dtype == np.float64:
        magma_dcsrset(m, n, indptr, indices, val, A.magma_t_matrix, A.queue)
    elif val.dtype == np.float32:
        magma_scsrset(m, n, indptr, indices, val, A.magma_t_matrix, A.queue)
    else:
        raise ValueError('magma_csrset: Type not supported.')


# TODO:
#def magma_csrget(A, queue):


def magma_solver(A, b, x, opts):
    """
    A is magma_matrix
    b is magma_matrix
    x is magma_matrix
    opts is magma_xopts with x = z, c, d, s
    queue is of type magma_queue_t
    """

    if A.dtype == np.complex128:
        return magma_z_solver(A.magma_t_matrix, b.magma_t_matrix, x.magma_t_matrix, opts, A.queue)
    elif A.dtype == np.complex64:
        return magma_c_solver(A.magma_t_matrix, b.magma_t_matrix, x.magma_t_matrix, opts, A.queue)
    elif A.dtype == np.float64:
        return magma_d_solver(A.magma_t_matrix, b.magma_t_matrix, x.magma_t_matrix, opts, A.queue)
    elif A.dtype == np.float32:
        return magma_s_solver(A.magma_t_matrix, b.magma_t_matrix, x.magma_t_matrix, opts, A.queue)
    else:
        raise ValueError('magma_solver: Type not supported.')


def magma_precondsetup(A, b, solver, precond):
    """
    A is magma_matrix
    b is magma_matrix
    solver is magma_x_solver_par with x = z, c, d, s
    precond is magma_x_preconditioner with x = z, c, d, s
    queue is of type magma_queue_t
    """

    if A.dtype == np.complex128:
        magma_z_precondsetup(A.magma_t_matrix, b.magma_t_matrix, solver, precond, A.queue)
    elif A.dtype == np.complex64:
        magma_c_precondsetup(A.magma_t_matrix, b.magma_t_matrix, solver, precond, A.queue)
    elif A.dtype == np.float64:
        magma_d_precondsetup(A.magma_t_matrix, b.magma_t_matrix, solver, precond, A.queue)
    elif A.dtype == np.float32:
        magma_s_precondsetup(A.magma_t_matrix, b.magma_t_matrix, solver, precond, A.queue)
    else:
        raise ValueError('magma_precondsetup: Type not supported.')


def magma_solverinfo(solver_par, precond_par, queue):
    """
    solver_par is magma_x_solver_par with x = z, c, d, s
    precond is magma_x_preconditioner with x = z, c, d, s
    queue is of type magma_queue_t
    """

    if isinstance(solver_par, magma_z_solver_par):
        magma_zsolverinfo(solver_par, precond_par, queue)
    elif isinstance(solver_par, magma_c_solver_par):
        magma_csolverinfo(solver_par, precond_par, queue)
    elif isinstance(solver_par, magma_d_solver_par):
        magma_dsolverinfo(solver_par, precond_par, queue)
    elif isinstance(solver_par, magma_s_solver_par):
        magma_ssolverinfo(solver_par, precond_par, queue)
    else:
        raise ValueError('magma_solverinfo: Type not supported.')


def magma_solverinfo_init(solver_par, precond, queue):
    """
    solver_par is magma_x_solver_par with x = z, c, d, s
    precond is magma_x_preconditioner with x = z, c, d, s
    queue is of type magma_queue_t
    """

    if isinstance(solver_par, magma_z_solver_par):
        magma_zsolverinfo_init(solver_par, precond, queue)
    elif isinstance(solver_par, magma_c_solver_par):
        magma_csolverinfo_init(solver_par, precond, queue)
    elif isinstance(solver_par, magma_d_solver_par):
        magma_dsolverinfo_init(solver_par, precond, queue)
    elif isinstance(solver_par, magma_s_solver_par):
        magma_ssolverinfo_init(solver_par, precond, queue)
    else:
        raise ValueError('magma_solverinfo_init: Type not supported.')


def magma_solverinfo_free(solver_par, precond, queue):
    """
    solver_par is magma_x_solver_par with x = z, c, d, s
    precond is magma_x_preconditioner with x = z, c, d, s
    queue is of type magma_queue_t
    """

    if isinstance(solver_par, magma_z_solver_par):
        magma_zsolverinfo_free(solver_par, precond, queue)
    elif isinstance(solver_par, magma_c_solver_par):
        magma_csolverinfo_free(solver_par, precond, queue)
    elif isinstance(solver_par, magma_d_solver_par):
        magma_dsolverinfo_free(solver_par, precond, queue)
    elif isinstance(solver_par, magma_s_solver_par):
        magma_ssolverinfo_free(solver_par, precond, queue)
    else:
        raise ValueError('magma_solverinfo_free: Type not supported.')


def magma_opts_default(dtype = np.complex128):
    if dtype == np.complex128:
        return magma_zopts_default()
    elif dtype == np.complex64:
        return magma_copts_default()
    elif dtype == np.float64:
        return magma_dopts_default()
    elif dtype == np.float32:
        return magma_sopts_default()
    else:
        raise ValueError('magma_opts_default: Type not supported.')


def magma_mconvert(A, B, old_format, new_format):
    """
    A is a magma_matrix
    B is a magma_matrix
    old_format is a int
    new_format is a int
    queue is of type magma_queue_t
    """

    if A.dtype == np.complex128:
        magma_zmconvert(A.magma_t_matrix, B.magma_t_matrix, old_format, new_format, A.queue)
    elif A.dtype == np.complex64:
        magma_cmconvert(A.magma_t_matrix, B.magma_t_matrix, old_format, new_format, A.queue)
    elif A.dtype == np.float64:
        magma_dmconvert(A.magma_t_matrix, B.magma_t_matrix, old_format, new_format, A.queue)
    elif A.dtype == np.float32:
        magma_smconvert(A.magma_t_matrix, B.magma_t_matrix, old_format, new_format, A.queue)
    else:
        raise ValueError('magma_mconvert: Type not supported.')


def magma_mscale(A, scaling):
    """
    A is magma_matrix
    queue is of type magma_queue_t
    """

    if A.dtype == np.complex128:
        magma_zmscale(A.magma_t_matrix, scaling, A.queue)
    elif A.dtype == np.complex64:
        magma_cmscale(A.magma_t_matrix, scaling, A.queue)
    elif A.dtype == np.float64:
        magma_dmscale(A.magma_t_matrix, scaling, A.queue)
    elif A.dtype == np.float32:
        magma_smscale(A.magma_t_matrix, scaling, A.queue)
    else:
        raise ValueError('magma_mscale: Type not supported.')


def magma_mtransfer(A, B, src, dst):
    """
    A is magma_matrix
    B is magma_matrix
    src is from magma_location_t
    dst is from magma_location_t
    queue is of type magma_queue_t
    """

    if A.dtype == np.complex128:
        magma_zmtransfer(A.magma_t_matrix, B.magma_t_matrix, src, dst, A.queue)
    elif A.dtype == np.complex64:
        magma_cmtransfer(A.magma_t_matrix, B.magma_t_matrix, src, dst, A.queue)
    elif A.dtype == np.float64:
        magma_dmtransfer(A.magma_t_matrix, B.magma_t_matrix, src, dst, A.queue)
    elif A.dtype == np.float32:
        magma_smtransfer(A.magma_t_matrix, B.magma_t_matrix, src, dst, A.queue)
    else:
        raise ValueError('magma_mtransfer: Type not supported.')


def magma_spmv(alpha, A, x, beta, y):
    """
    A is magma_matrix,
    x is magma_matrix,
    y is magma_matrix,
    alpha is np.complex128 or np.float64,
    beta is np.complex128 or np.float64,
    queue is of type magma_queue_t.
    """

    if A.dtype == np.complex128:
        alpha = np.complex128(alpha)
        alpha_magma = cuda.cuDoubleComplex()
        alpha_magma.x = alpha.real
        alpha_magma.y = alpha.imag

        beta = np.complex128(beta)
        beta_magma = cuda.cuDoubleComplex()
        beta_magma.x = beta.real
        beta_magma.y = beta.imag

        magma_z_spmv(alpha_magma, A.magma_t_matrix, x.magma_t_matrix, beta_magma, y.magma_t_matrix, A.queue)
    elif A.dtype == np.complex64:
        alpha = np.complex64(alpha)
        alpha_magma = cuda.cuFloatComplex()
        alpha_magma.x = alpha.real
        alpha_magma.y = alpha.imag

        beta = np.complex64(beta)
        beta_magma = cuda.cuFloatComplex()
        beta_magma.x = beta.real
        beta_magma.y = beta.imag

        magma_c_spmv(alpha_magma, A.magma_t_matrix, x.magma_t_matrix, beta_magma, y.magma_t_matrix, A.queue)
    elif A.dtype == np.float64:
        magma_d_spmv(alpha, A.magma_t_matrix, x.magma_t_matrix, beta, y.magma_t_matrix, A.queue)
    elif A.dtype == np.float32:
        magma_s_spmv(alpha, A.magma_t_matrix, x.magma_t_matrix, beta, y.magma_t_matrix, A.queue)
    else:
        raise ValueError('magma_spmv: Type not supported.')


def magma_vget(v):
    """
    v is of type magma_matrix
    queue is of type magma_queue_t
    """

    if v.dtype == np.complex128:
        return magma_zvget(v.magma_t_matrix, v.queue)
    elif v.dtype == np.complex64:
        return magma_cvget(v.magma_t_matrix, v.queue)
    elif v.dtype == np.float64:
        return magma_dvget(v.magma_t_matrix, v.queue)
    elif v.dtype == np.float32:
        return magma_svget(v.magma_t_matrix, v.queue)
    else:
        raise ValueError('magma_vget: Type not supported.')


def magma_vcopy(n, x, incx, y, incy):
    """
    n is the array length asint
    x is a magma_matrix
    incx is an int
    y is a magma_matrix
    incy is an int
    """

    if x.dtype == np.complex128:
        magma_zcopy( n, x.magma_t_matrix.dval, incx, y.magma_t_matrix.dval, incy)
    elif x.dtype == np.complex64:
        magma_ccopy( n, x.magma_t_matrix.dval, incx, y.magma_t_matrix.dval, incy)
    elif x.dtype == np.float64:
        magma_dcopy( n, x.magma_t_matrix.dval, incx, y.magma_t_matrix.dval, incy)
    elif x.dtype == np.float32:
        magma_scopy( n, x.magma_t_matrix.dval, incx, y.magma_t_matrix.dval, incy)
    else:
        raise ValueError('magma_vcopy: Type not supported.')


def magma_vinit(x, memory_location, num_rows, num_cols, values):
    """
    x is a magma_matrix
    memory_location is int
    num_rows is int
    num_cols is int
    values is np.complex128 or np.float64
    queue is of type magma_queue_t
    """

    if x.dtype == np.complex128:
        values_magma = cuda.cuDoubleComplex()
        values_magma.x = values.real
        values_magma.y = values.imag

        magma_zvinit(x.magma_t_matrix, memory_location, num_rows, num_cols, values_magma, x.queue)
    elif x.dtype == np.complex64:
        values_magma = cuda.cuFloatComplex()
        values_magma.x = values.real
        values_magma.y = values.imag

        magma_cvinit(x.magma_t_matrix, memory_location, num_rows, num_cols, values_magma, x.queue)
    elif x.dtype == np.float64:
        magma_dvinit(x.magma_t_matrix, memory_location, num_rows, num_cols, values, x.queue)
    elif x.dtype == np.float32:
        magma_svinit(x.magma_t_matrix, memory_location, num_rows, num_cols, values, x.queue)
    else:
        raise ValueError('magma_vinit: Type not supported.')


class magma_matrix:
    def __init__(self,
                 A = None,
                 queue = None,
                 memory_location = magma_location_t.Magma_DEV,
                 storage = magma_storage_t.Magma_CSR,
                 dtype = np.complex128,
                 blocksize = 32,
                 alignment = 1):

        """
        A is a scipy sparse matrix or a ndarray.
        If A is a ndarray, then a (m,1) sparse matrix is created from the array.
        """

        self.memory_location = memory_location
        self.magma_t_matrix  = magma_t_matrix(magma_storage_t.Magma_CSR)
        self.storage         = storage
        self.dtype           = dtype
        self.queue           = queue
        self.blocksize       = blocksize
        self.alignment       = alignment

        if A is not None:
            self.dtype = A.dtype

            A_CPU  = magma_t_matrix(magma_storage_t.Magma_CSR)
            A_CPU.memory_location = magma_location_t.Magma_CPU

            if isinstance(A, np.ndarray):
                if self.dtype == np.complex128:
                    magma_zvset(A.size, 1, A, A_CPU, queue)
                    magma_zmtransfer(A_CPU, self.magma_t_matrix, magma_location_t.Magma_CPU, memory_location, self.queue)
                elif self.dtype == np.complex64:
                    magma_cvset(A.size, 1, A, A_CPU, queue)
                    magma_cmtransfer(A_CPU, self.magma_t_matrix, magma_location_t.Magma_CPU, memory_location, self.queue)
                elif self.dtype == np.float64:
                    magma_dvset(A.size, 1, A, A_CPU, queue)
                    magma_dmtransfer(A_CPU, self.magma_t_matrix, magma_location_t.Magma_CPU, memory_location, self.queue)
                elif self.dtype == np.float32:
                    magma_svset(A.size, 1, A, A_CPU, queue)
                    magma_smtransfer(A_CPU, self.magma_t_matrix, magma_location_t.Magma_CPU, memory_location, self.queue)
                else:
                    raise ValueError('__init__: Type not supported.')

            else:
                A_CONV = magma_t_matrix(self.storage)
                A_CONV.memory_location = magma_location_t.Magma_CPU
                A_CONV.blocksize = self.blocksize
                A_CONV.alignment = self.alignment

                if self.dtype == np.complex128:
                    magma_zcsrset(A.shape[0], A.shape[1], A.indptr, A.indices, A.data, A_CPU, self.queue)
                    magma_zmconvert(A_CPU, A_CONV, magma_storage_t.Magma_CSR, self.storage, self.queue)
                    magma_zmtransfer(A_CONV, self.magma_t_matrix, magma_location_t.Magma_CPU, memory_location, self.queue)
                elif self.dtype == np.complex64:
                    magma_ccsrset(A.shape[0], A.shape[1], A.indptr, A.indices, A.data, A_CPU, self.queue)
                    magma_cmconvert(A_CPU, A_CONV, magma_storage_t.Magma_CSR, self.storage, self.queue)
                    magma_cmtransfer(A_CONV, self.magma_t_matrix, magma_location_t.Magma_CPU, memory_location, self.queue)
                elif self.dtype == np.float64:
                    magma_dcsrset(A.shape[0], A.shape[1], A.indptr, A.indices, A.data, A_CPU, self.queue)
                    magma_dmconvert(A_CPU, A_CONV, magma_storage_t.Magma_CSR, self.storage, self.queue)
                    magma_dmtransfer(A_CONV, self.magma_t_matrix, magma_location_t.Magma_CPU, memory_location, self.queue)
                elif self.dtype == np.float32:
                    magma_scsrset(A.shape[0], A.shape[1], A.indptr, A.indices, A.data, A_CPU, self.queue)
                    magma_smconvert(A_CPU, A_CONV, magma_storage_t.Magma_CSR, self.storage, self.queue)
                    magma_smtransfer(A_CONV, self.magma_t_matrix, magma_location_t.Magma_CPU, memory_location, self.queue)
                else:
                    raise ValueError('__init__: Type not supported.')


    def get_data(self):
        #if self.memory_location == magma_location_t.Magma_DEV:
        A = magma_t_matrix(magma_storage_t.Magma_CSR)
        A.memory_location = magma_location_t.Magma_CPU

        if self.dtype == np.complex128:
            magma_zmtransfer(self.magma_t_matrix, A, magma_location_t.Magma_DEV, magma_location_t.Magma_CPU, self.queue)
            m, n, val = magma_zvget(A, self.queue)
        elif self.dtype == np.complex64:
            magma_cmtransfer(self.magma_t_matrix, A, magma_location_t.Magma_DEV, magma_location_t.Magma_CPU, self.queue)
            m, n, val = magma_cvget(A, self.queue)
        elif self.dtype == np.float64:
            magma_dmtransfer(self.magma_t_matrix, A, magma_location_t.Magma_DEV, magma_location_t.Magma_CPU, self.queue)
            m, n, val = magma_dvget(A, self.queue)
        elif self.dtype == np.float32:
            magma_smtransfer(self.magma_t_matrix, A, magma_location_t.Magma_DEV, magma_location_t.Magma_CPU, self.queue)
            m, n, val = magma_svget(A, self.queue)
        else:
            raise ValueError('get_data: Type not supported.')

        return val


class magma_vector:
    def __init__(self, A, memory_space = 'device'):
        if isinstance(A, np.ndarray):
            self.memory_space = memory_space
            self.size   = A.shape[0]
            self.dtype  = A.dtype
            self.nbytes = self.size*self.dtype.itemsize

            if memory_space == 'device':
                self.pointer = cuda.cudaMalloc(self.nbytes)
                cuda.cudaMemcpy_htod(self.pointer, A.ctypes.data_as(ctypes.c_void_p) , self.nbytes)
            else:
                raise ValueError('magma_vector: not implemented.')
        else:
            raise ValueError('magma_vector: Type not supported for construction.')

    def to_ndarray(self):
        a = np.zeros((self.size,), dtype = self.dtype)
        if self.memory_space == 'device':
            cuda.cudaMemcpy_dtoh(a.ctypes.data_as(ctypes.c_void_p), self.pointer, self.nbytes)
        return a

    @property
    def dev_ptr(self):
        return self.pointer.value

    def __int__(self):
        return self.dev_ptr

    def iamax(self, incx = 1):
        """
        Index of maximum magnitude element.
        """
        if self.memory_space == 'device':
            if self.dtype == np.complex128:
                return magma_izamax(self.size, self, incx)
            elif self.dtype == np.complex64:
                return magma_icamax(self.size, self, incx)
            elif self.dtype == np.float64:
                return magma_idamax(self.size, self, incx)
            elif self.dtype == np.float32:
                return magma_isamax(self.size, self, incx)