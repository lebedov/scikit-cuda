.. -*- rst -*-

.. currentmodule:: scikits.cuda.cula

CULA Routines
=============

Helper Routines
---------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   culaCheckStatus
   culaGetErrorInfo
   culaGetLastStatus
   culaGetStatusString
   culaInitialize
   culaSelectDevice
   culaShutdown

Linear Algebra Routines
-----------------------

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
