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
   cusolverDnGetStream
   cusolverDnDestroy
   cusolverDnSetStream

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
   cusolverDnSpotrf_bufferSize
   cusolverDnSpotrf

   cusolverDnCgeqrf_bufferSize
   cusolverDnCgeqrf
   cusolverDnCgesvd_bufferSize
   cusolverDnCgesvd
   cusolverDnCgetrf_bufferSize
   cusolverDnCgetrf
   cusolverDnCgetrs
   cusolverDnCpotrf_bufferSize
   cusolverDnCpotrf

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
   cusolverDnDpotrf_bufferSize
   cusolverDnDpotrf

   cusolverDnZgeqrf_bufferSize
   cusolverDnZgeqrf
   cusolverDnZgesvd_bufferSize
   cusolverDnZgesvd
   cusolverDnZgetrf_bufferSize
   cusolverDnZgetrf
   cusolverDnZgetrs
   cusolverDnZpotrf_bufferSize
   cusolverDnZpotrf
