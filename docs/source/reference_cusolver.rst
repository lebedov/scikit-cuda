.. -*- rst -*-

.. currentmodule:: skcuda.cusolver

CUSOLVER Routines
=================
These routines are only available in CUDA 7.0 and later.

Helper Routines
---------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   cusolverDnCreate
   cusolverDnCreateSyevjInfo
   cusolverDnGetStream
   cusolverDnDestroy
   cusolverDnDestroySyevjInfo
   cusolverDnSetStream
   cusolverDnXsyevjGetResidual
   cusolverDnXsyevjGetSweeps
   cusolverDnXsyevjSetMaxSweeps
   cusolverDnXsyevjSetSortEig
   cusolverDnXsyevjSetTolerance

Wrapper Routines
----------------

Single Precision Routines
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   cusolverDnSgeqrf_bufferSize
   cusolverDnSgeqrf
   cusolverDnSgesvd_bufferSize
   cusolverDnSgesvd
   cusolverDnSgetrf_bufferSize
   cusolverDnSgetrf
   cusolverDnSgetrs
   cusolverDnSorgqr_bufferSize
   cusolverDnSorgqr
   cusolverDnSpotrf_bufferSize
   cusolverDnSpotrf
   cusolverDnSsyevd_bufferSize
   cusolverDnSsyevd
   cusolverDnSsyevj_bufferSize
   cusolverDnSsyevj
   cusolverDnSsyevjBatched_bufferSize
   cusolverDnSsyevjBatched

   cusolverDnCgeqrf_bufferSize
   cusolverDnCgeqrf
   cusolverDnCgesvd_bufferSize
   cusolverDnCgesvd
   cusolverDnCgetrf_bufferSize
   cusolverDnCgetrf
   cusolverDnCgetrs
   cusolverDnCheevd_bufferSize
   cusolverDnCheevd
   cusolverDnCheevj_bufferSize
   cusolverDnCheevj
   cusolverDnCheevjBatched_bufferSize
   cusolverDnCheevjBatched
   cusolverDnCpotrf_bufferSize
   cusolverDnCpotrf
   cusolverDnCungqr_bufferSize
   cusolverDnCungqr

Double Precision Routines
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   cusolverDnDgeqrf_bufferSize
   cusolverDnDgeqrf
   cusolverDnDgesvd_bufferSize
   cusolverDnDgesvd
   cusolverDnDgetrf_bufferSize
   cusolverDnDgetrf
   cusolverDnDgetrs
   cusolverDnDorgqr_bufferSize
   cusolverDnDorgqr
   cusolverDnDpotrf_bufferSize
   cusolverDnDpotrf
   cusolverDnDsyevd_bufferSize
   cusolverDnDsyevd
   cusolverDnDsyevj_bufferSize
   cusolverDnDsyevj
   cusolverDnDsyevjBatched_bufferSize
   cusolverDnDsyevjBatched

   cusolverDnZgeqrf_bufferSize
   cusolverDnZgeqrf
   cusolverDnZgesvd_bufferSize
   cusolverDnZgesvd
   cusolverDnZgetrf_bufferSize
   cusolverDnZgetrf
   cusolverDnZgetrs
   cusolverDnZheevd_bufferSize
   cusolverDnZheevd
   cusolverDnZheevj_bufferSize
   cusolverDnZheevj
   cusolverDnZheevjBatched_bufferSize
   cusolverDnZheevjBatched
   cusolverDnZpotrf_bufferSize
   cusolverDnZpotrf
   cusolverDnZungqr_bufferSize
   cusolverDnZungqr
