.. -*- rst -*-

.. currentmodule:: skcuda.cufft

CUFFT Routines
==============

Helper Routines
---------------
.. autosummary::
    :toctree: generated/
    :nosignatures:

    cufftCheckStatus
    cufftCreate
    cufftDestroy
    cufftSetAutoAllocation
    cufftSetCompatibilityMode
    cufftSetStream
    cufftSetWorkArea

Wrapper Routines
----------------
.. autosummary::
    :toctree: generated/
    :nosignatures:
    
    cufftPlan1d
    cufftPlan2d
    cufftPlan3d
    cufftPlanMany
    cufftDestroy
    cufftExecC2C
    cufftExecR2C
    cufftExecC2R
    cufftExecZ2Z
    cufftExecD2Z
    cufftExecZ2D
    cufftEstimate1d
    cufftEstimate2d
    cufftEstimate3d
    cufftEstimateMany
    cufftGetSize1d
    cufftGetSize2d
    cufftGetSize3d
    cufftGetSizeMany
    cufftGetSize
    cufftMakePlan1d
    cufftMakePlan2d
    cufftMakePlan3d
    cufftMakePlanMany
