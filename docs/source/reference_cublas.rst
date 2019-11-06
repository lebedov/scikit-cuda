.. -*- rst -*-

.. currentmodule:: skcuda.cublas

CUBLAS Routines
===============

Helper Routines
---------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   cublasCheckStatus
   cublasCreate
   cublasDestroy
   cublasGetPointerMode
   cublasGetStream
   cublasGetVersion
   cublasSetPointerMode
   cublasSetStream
   
Wrapper Routines
----------------

Single Precision BLAS1 Routines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   cublasIsamax
   cublasIsamin
   cublasSasum
   cublasSaxpy
   cublasScopy
   cublasSdot
   cublasSnrm2
   cublasSrot
   cublasSrotg
   cublasSrotm
   cublasSrotmg
   cublasSscal
   cublasSswap

   cublasCaxpy
   cublasCcopy
   cublasCdotc
   cublasCdotu
   cublasCrot
   cublasCrotg
   cublasCscal
   cublasCsrot
   cublasCsscal
   cublasCswap
   cublasIcamax
   cublasIcamin
   cublasScasum
   cublasScnrm2

Double Precision BLAS1 Routines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   cublasIdamax
   cublasIdamin
   cublasDasum
   cublasDaxpy
   cublasDcopy
   cublasDdot
   cublasDnrm2
   cublasDrot
   cublasDrotg
   cublasDrotm
   cublasDrotmg
   cublasDscal
   cublasDswap
   cublasDzasum
   cublasDznrm2
   cublasIzamax
   cublasIzamin
   
   cublasZaxpy
   cublasZcopy
   cublasZdotc
   cublasZdotu
   cublasZdrot
   cublasZdscal
   cublasZrot
   cublasZrotg
   cublasZscal
   cublasZswap

Single Precision BLAS2 Routines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   cublasSgbmv
   cublasSgemv
   cublasSger
   cublasSsbmv
   cublasSspmv
   cublasSspr
   cublasSspr2
   cublasSsymv
   cublasSsyr
   cublasSsyr2
   cublasStbmv
   cublasStbsv
   cublasStpmv
   cublasStpsv
   cublasStrmv
   cublasStrsv

   cublasCgbmv
   cublasCgemv
   cublasCgerc
   cublasCgeru
   cublasChbmv
   cublasChemv
   cublasCher
   cublasCher2
   cublasChpmv
   cublasChpr
   cublasChpr2
   cublasCtbmv
   cublasCtbsv
   cublasCtpmv
   cublasCtpsv
   cublasCtrmv
   cublasCtrsv

Double Precision BLAS2 Routines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   cublasDgbmv
   cublasDgemv
   cublasDger
   cublasDsbmv
   cublasDspmv
   cublasDspr
   cublasDspr2
   cublasDsymv
   cublasDsyr
   cublasDsyr2
   cublasDtbmv
   cublasDtbsv
   cublasDtpmv
   cublasDtpsv
   cublasDtrmv
   cublasDtrsv

   cublasZgbmv
   cublasZgemv
   cublasZgerc
   cublasZgeru
   cublasZhbmv
   cublasZhemv
   cublasZher
   cublasZher2
   cublasZhpmv
   cublasZhpr
   cublasZhpr2
   cublasZtbmv
   cublasZtbsv
   cublasZtpmv
   cublasZtpsv
   cublasZtrmv
   cublasZtrsv

Single Precision BLAS3 Routines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   cublasSgemm
   cublasSsymm
   cublasSsyrk
   cublasSsyr2k
   cublasStrmm
   cublasStrsm

   cublasCgemm
   cublasChemm
   cublasCherk
   cublasCher2k
   cublasCsymm
   cublasCsyrk
   cublasCsyr2k
   cublasCtrmm
   cublasCtrsm

Double Precision BLAS3 Routines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   cublasDgemm
   cublasDsymm
   cublasDsyrk
   cublasDsyr2k
   cublasDtrmm
   cublasDtrsm

   cublasZgemm
   cublasZhemm
   cublasZherk
   cublasZher2k
   cublasZsymm
   cublasZsyrk
   cublasZsyr2k
   cublasZtrmm
   cublasZtrsm

Single-Precision BLAS-like Extension Routines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   cublasSdgmm
   cublasSgeam
   cublasSgelsBatched
   cublasSgemmBatched
   cublasSgemmStridedBatched
   cublasSgelsBatched
   cublasCgemmBatched
   cublasCgemmStridedBatched
   cublasStrsmBatched
   cublasSgetrfBatched
   cublasCdgmm
   cublasCgeam

Double-Precision BLAS-like Extension Routines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   cublasDdgmm
   cublasDgeam
   cublasDgelsBatched
   cublasDgemmBatched
   cublasDgemmStridedBatched
   cublasZgelsBatched
   cublasZgemmBatched
   cublasZgemmStridedBatched
   cublasDtrsmBatched
   cublasDgetrfBatched
   cublasZdgmm
   cublasZgeam
