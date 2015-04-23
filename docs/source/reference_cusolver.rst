.. -*- rst -*-

.. currentmodule:: scikits.cuda.cusolver

CUSOLVER Routines
=================
These routines are only available in CUDA 7.0 and later.

Helper Routines
---------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   cusolverDnCreate
   cusolverDnDestroy
   cusolverDnSetStream
   cusolverDnGetStream

Wrapper Routines
----------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   cusolverDnSgetrf_bufferSize
   cusolverDnSgetrf
   cusolverDnDgetrf_bufferSize
   cusolverDnDgetrf
   cusolverDnCgetrf_bufferSize
   cusolverDnCgetrf
   cusolverDnZgetrf_bufferSize
   cusolverDnZgetrf
   cusolverDnSgetrs
   cusolverDnDgetrs
   cusolverDnCgetrs
   cusolverDnZgetrs
   cusolverDnSgesvd_bufferSize
   cusolverDnSgesvd
   cusolverDnDgesvd_bufferSize
   cusolverDnDgesvd
   cusolverDnCgesvd_bufferSize
   cusolverDnCgesvd
   cusolverDnZgesvd_bufferSize
   cusolverDnZgesvd
