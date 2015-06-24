.. -*- rst -*-

.. currentmodule:: skcuda.cula

CULA Routines
=============

Framework Routines
------------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   culaCheckStatus
   culaFreeBuffers
   culaGetCublasMinimumVersion
   culaGetCublasRuntimeVersion
   culaGetCudaDriverVersion
   culaGetCudaMinimumVersion
   culaGetCudaRuntimeVersion
   culaGetDeviceCount
   culaGetErrorInfo
   culaGetErrorInfoString
   culaGetExecutingDevice
   culaGetLastStatus
   culaGetStatusString
   culaGetVersion
   culaInitialize
   culaSelectDevice
   culaShutdown

Auxiliary Routines
------------------

Single Precision Real
^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   culaDeviceSgeNancheck
   culaDeviceSgeTranspose
   culaDeviceSgeTransposeInplace

Single Precision Complex
^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   culaDeviceCgeConjugate
   culaDeviceCgeNancheck
   culaDeviceCgeTranspose
   culaDeviceCgeTransposeConjugate
   culaDeviceCgeTransposeInplace
   culaDeviceCgeTransposeConjugateInplace

Double Precision Real
^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   culaDeviceDgeNancheck
   culaDeviceDgeTranspose
   culaDeviceDgeTransposeInplace

Double Precision Complex
^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   culaDeviceZgeConjugate
   culaDeviceZgeNancheck
   culaDeviceZgeTranspose
   culaDeviceZgeTransposeConjugate
   culaDeviceZgeTransposeInplace
   culaDeviceZgeTransposeConjugateInplace
    
BLAS Routines
-------------

Single Precision Real
^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   culaDeviceSgemm
   culaDeviceSgemv

Single Precision Complex
^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   culaDeviceCgemm
   culaDeviceCgemv

Double Precision Real
^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   culaDeviceDgemm
   culaDeviceDgemv

Double Precision Complex
^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   culaDeviceZgemm
   culaDeviceZgemv

LAPACK Routines
---------------

Single Precision Real
^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   culaDeviceSgels
   culaDeviceSgeqrf
   culaDeviceSgesv
   culaDeviceSgesvd
   culaDeviceSgetrf
   culaDeviceSgglse
   culaDeviceSposv
   culaDeviceSpotrf
   
Single Precision Complex
^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   culaDeviceCgels
   culaDeviceCgeqrf
   culaDeviceCgesv
   culaDeviceCgesvd
   culaDeviceCgetrf
   culaDeviceCgglse
   culaDeviceCposv
   culaDeviceCpotrf

Double Precision Real
^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   culaDeviceDgels
   culaDeviceDgeqrf
   culaDeviceDgesv
   culaDeviceDgesvd
   culaDeviceDgetrf
   culaDeviceDgglse
   culaDeviceDposv
   culaDeviceDpotrf

Double Precision Complex
^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   culaDeviceZgels
   culaDeviceZgeqrf
   culaDeviceZgesv
   culaDeviceZgesvd
   culaDeviceZgetrf
   culaDeviceZgglse
   culaDeviceZposv
   culaDeviceZpotrf
