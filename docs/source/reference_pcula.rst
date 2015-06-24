.. -*- rst -*-

.. currentmodule:: skcuda.pcula

Multi-GPU CULA Routines
=======================

Framework Routines
------------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   pculaConfigInit

BLAS Routines
-------------

Single Precision Real
^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   pculaSgemm
   pculaStrsm

Single Precision Complex
^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   pculaCgemm
   pculaCtrsm

Double Precision Real
^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   pculaDgemm
   pculaDtrsm

Double Precision Complex
^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   pculaZgemm
   pculaZtrsm

LAPACK Routines
---------------

Single Precision Real
^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   pculaSgesv
   pculaSgetrf
   pculaSgetrs
   pculaSposv
   pculaSpotrf
   pculaSpotrs

Single Precision Complex
^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   pculaCgesv
   pculaCgetrf
   pculaCgetrs
   pculaCposv
   pculaCpotrf
   pculaCpotrs

Double Precision Real
^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   pculaDgesv
   pculaDgetrf
   pculaDgetrs
   pculaDposv
   pculaDpotrf
   pculaDpotrs

Double Precision Complex
^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   pculaZgesv
   pculaZgetrf
   pculaZgetrs
   pculaZposv
   pculaZpotrf
   pculaZpotrs
