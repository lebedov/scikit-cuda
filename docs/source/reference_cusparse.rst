.. -*- rst -*-

.. currentmodule:: skcuda.cusparse

CUSPARSE Routines
=================

Helper Routines
---------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   cusparseCreate
   cusparseDestroy
   cusparseGetVersion
   cusparseSetStream
   cusparseGetStream

Wrapper Routines
----------------

Single Precision Routines
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   cusparseSgtsv2StridedBatch_bufferSizeExt
   cusparseSgtsv2StridedBatch
   cusparseSgtsvInterleavedBatch_bufferSizeExt
   cusparseSgtsvInterleavedBatch

Double Precision Routines
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/
   :nosignatures:

   cusparseDgtsv2StridedBatch_bufferSizeExt
   cusparseDgtsv2StridedBatch
   cusparseDgtsvInterleavedBatch_bufferSizeExt
   cusparseDgtsvInterleavedBatch
