#!/usr/bin/env python

"""
Python interface to CUBLAS functions.

Note: this module does not explicitly depend on PyCUDA.
"""

import sys
import warnings
import ctypes
import ctypes.util
import atexit
import numpy as np

from string import Template

import cuda

if sys.platform == 'linux2':
    _libcublas_libname_list = ['libcublas.so', 'libcublas.so.3',
                               'libcublas.so.4']
elif sys.platform == 'darwin':
    _libcublas_libname_list = ['libcublas.dylib']
else:
    raise RuntimeError('unsupported platform')

# Print understandable error message when library cannot be found:
_libcublas = None
for _libcublas_libname in _libcublas_libname_list:
    try:
        _libcublas = ctypes.cdll.LoadLibrary(_libcublas_libname)
    except OSError:
        pass
    else:
        break
if _libcublas == None:
    raise OSError('cublas library not found')

# Generic CUBLAS error:
class cublasError(Exception):
    """CUBLAS error"""
    pass

# Exceptions corresponding to different CUBLAS errors:
class cublasNotInitialized(cublasError):
    """CUBLAS library not initialized."""
    pass

class cublasAllocFailed(cublasError):
    """Resource allocation failed."""
    pass

class cublasInvalidValue(cublasError):
    """Unsupported numerical value was passed to function."""
    pass

class cublasArchMismatch(cublasError):
    """Function requires an architectural feature absent from the device."""
    pass

class cublasMappingError(cublasError):
    """Access to GPU memory space failed."""
    pass

class cublasExecutionFailed(cublasError):
    """GPU program failed to execute."""
    pass

class cublasInternalError(cublasError):
    """An internal CUBLAS operation failed."""
    pass

cublasExceptions = {
    0x1: cublasNotInitialized,
    0x3: cublasAllocFailed,
    0x7: cublasInvalidValue,
    0x8: cublasArchMismatch,
    0xb: cublasMappingError,
    0xd: cublasExecutionFailed,
    0xe: cublasInternalError,
    }

_CUBLAS_OP = {
    'n': 0, # CUBLAS_OP_N
    'N': 0,
    't': 1, # CUBLAS_OP_T
    'T': 1,
    'c': 2, # CUBLAS_OP_C
    'C': 2,
    }

_CUBLAS_FILL_MODE = {
    'l': 0, # CUBLAS_FILL_MODE_LOWER
    'L': 0,
    'u': 1, # CUBLAS_FILL_MODE_UPPER
    'U': 1,
    }

_CUBLAS_DIAG = {
    'n': 0, # CUBLAS_DIAG_NON_UNIT,
    'N': 0,
    'u': 1, # CUBLAS_DIAG_UNIT
    'U': 1,
    }

_libcublas.cublasGetError.restype = int
_libcublas.cublasGetError.argtypes = []
def cublasGetError():
    """
    Retrieve and reset CUBLAS error code.

    Retrieves the current CUBLAS error code and resets it in
    preparation for the next CUBLAS operation.

    Returns
    -------
    e : int
        Error code.

    See Also
    --------
    cublasExceptions
    
    """

    return _libcublas.cublasGetError()

def cublasCheckStatus(status):
    """
    Raise CUBLAS exception
    
    Raise an exception corresponding to the specified CUBLAS error
    code.
    
    Parameters
    ----------
    status : int
        CUBLAS error code.

    See Also
    --------
    cublasExceptions

    """
    
    if status != 0:
        try:
            raise cublasExceptions[status]
        except KeyError:
            raise cublasError

# Legacy functions:

_libcublas.cublasInit.restype = int
_libcublas.cublasInit.argtypes = []
def cublasInit():
    """
    Initialize CUBLAS.

    This function must be called before using any other CUBLAS functions.

    """

    if cuda.cudaDriverGetVersion() >= 4000:
        warnings.warn('cublasInit() is deprecated as of CUDA 4.0',
                      DeprecationWarning)
    status = _libcublas.cublasInit()
    cublasCheckStatus(status)

_libcublas.cublasShutdown.restype = int
_libcublas.cublasShutdown.argtypes = []
def cublasShutdown():
    """
    Shut down CUBLAS.

    This function must be called before an application that uses
    CUBLAS terminates.
    
    """

    if cuda.cudaDriverGetVersion() >= 4000:
        warnings.warn('cublasShutdown() is deprecated as of CUDA 4.0',
                      DeprecationWarning)
    status = _libcublas.cublasShutdown()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    atexit.register(_libcublas.cublasShutdown)

if cuda.cudaDriverGetVersion() < 4000:
    def cublasCreate():
        raise NotImplementedError(
            'cublasCreate() is only available in CUDA 4.0 and later')
else:
    _libcublas.cublasCreate_v2.restype = int
    _libcublas.cublasCreate_v2.argtypes = [ctypes.c_void_p]
    def cublasCreate():
        handle = ctypes.c_int()
        status = _libcublas.cublasCreate_v2(ctypes.byref(handle))
        cublasCheckStatus(status)
        return handle.value    
cublasCreate.__doc__ = \
    """
    Initialize CUBLAS.

    Initializes CUBLAS and creates a handle to a structure holding
    the CUBLAS library context.

    Returns
    -------
    handle : int
        CUBLAS library context.
        
    Notes
    -----
    This function is only available in CUDA 4.0 and later.
    
    """

if cuda.cudaDriverGetVersion() < 4000:
    def cublasDestroy(handle):
        raise NotImplementedError(
            'cublasDestroy() is only available in CUDA 4.0 and later')
else:
    _libcublas.cublasDestroy_v2.restype = int
    _libcublas.cublasDestroy_v2.argtypes = [ctypes.c_int]
    def cublasDestroy(handle):
        status = _libcublas.cublasCreate_v2(ctypes.c_int(handle))
        cublasCheckStatus(status)
cublasDestroy.__doc__ = \
    """
    Release CUBLAS resources.

    Releases hardware resources used by CUBLAS.

    Parameters
    ----------
    handle : int
        CUBLAS library context.
        
    Notes
    -----
    This function is only available in CUDA 4.0 and later.
    
    """

if cuda.cudaDriverGetVersion() < 4000:
    def cublasGetCurrentCtx():
        raise NotImplementedError(
            'cublasGetCurrentCtx() is only available in CUDA 4.0 and later')
else:
    _libcublas.cublasGetCurrentCtx.restype = int
    def cublasGetCurrentCtx():
        return _libcublas.cublasGetCurrentCtx()
cublasGetCurrentCtx.__doc__ = \
    """
    Get current CUBLAS context.

    Returns the current context used by CUBLAS.

    Returns
    -------
    context : int
        Current CUBLAS context.

    Notes
    -----
    This function is only available in CUDA 4.0 and later.

    """

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasSetKernelStream.restype = int
    _libcublas.cublasSetKernelStream.argtypes = [ctypes.c_int]
    def cublasSetKernelStream(id):
        status = _libcublas.cublasSetKernelStream(id)
        cublasCheckStatus(status)
    cublasSetStream = cublasSetKernelStream
else:
    _libcublas.cublasSetStream_v2.restype = int
    _libcublas.cublasSetStream_v2.argtypes = [ctypes.c_int,
                                              ctypes.c_int]
    def cublasSetStream(id):
        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasSetStream_v2(handle, id)
        cublasCheckStatus(status)
    cublasSetKernelStream = cublasSetStream
cublasSetStream.__doc__ = \
  """
  Set current CUBLAS library stream.

  Parameters
  ----------
  id : int
      Stream ID.

  """

if cuda.cudaDriverGetVersion() < 4000:
    def cublasGetStream():
        raise NotImplementedError(
            'cublasGetSTream() is only available in CUDA 4.0 and later')
else:
    _libcublas.cublasGetStream_v2.restype = int
    _libcublas.cublasGetStream_v2.argtypes = [ctypes.c_int,
                                              ctypes.c_void_p]
    def cublasGetStream():
        id = ctypes.c_int()
        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasGetStream_v2(handle, ctypes.byref(id))
        cublasCheckStatus(status)
        return id.value
cublasGetStream.__doc__ = \
  """
  Set current CUBLAS library stream.

  Returns
  -------
  id : int
      Stream ID.

  Notes
  -----
  This function is only available in CUDA 4.0 and later.
  
  """
### BLAS Level 1 Functions ###

# ISAMAX, IDAMAX, ICAMAX, IZAMAX
I_AMAX_doc = Template(
"""
    Index of maximum magnitude element.

    Finds the smallest index of the maximum magnitude element of a
    ${precision} ${real} vector.

    Parameters
    ----------
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input vector.
    incx : int
        Storage spacing between elements of `x`.

    Returns
    -------
    idx : int
        Index of maximum magnitude element.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data} 
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> m = ${func}(x_gpu.size, x_gpu.gpudata, 1)
    >>> np.allclose(m, np.argmax(np.abs(x)))
    True
    
    Notes
    -----
    This function returns a 0-based index.
    
""")

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasIsamax.restype = int
    _libcublas.cublasIsamax.argtypes = [ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
    def cublasIsamax(n, x, incx):
        result = _libcublas.cublasIsamax(n, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)	
        return result-1
else:
    _libcublas.cublasIsamax_v2.restype = int
    _libcublas.cublasIsamax_v2.argtypes = [ctypes.c_int,
                                           ctypes.c_int,
                                           ctypes.c_void_p,
                                           ctypes.c_int,
                                           ctypes.c_void_p]
    def cublasIsamax(n, x, incx):
        handle = cublasGetCurrentCtx()
        result = ctypes.c_int()
        status = \
               _libcublas.cublasIsamax_v2(handle,
                                          n, int(x), incx, ctypes.byref(result))
        cublasCheckStatus(status)
        return result.value-1

cublasIsamax.__doc__ = \
                     I_AMAX_doc.substitute(precision='single-precision',
                                           real='real',
                                           data='np.random.rand(5).astype(np.float32)',
                                           func='cublasIsamax')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasIdamax.restype = int
    _libcublas.cublasIdamax.argtypes = [ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
    def cublasIdamax(n, x, incx):
        result = _libcublas.cublasIdamax(n, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)
        return result-1
else:
    _libcublas.cublasIdamax_v2.restype = int
    _libcublas.cublasIdamax_v2.argtypes = [ctypes.c_int,
                                           ctypes.c_int,
                                           ctypes.c_void_p,
                                           ctypes.c_int,
                                           ctypes.c_void_p]
    def cublasIdamax(n, x, incx):
        handle = cublasGetCurrentCtx()
        result = ctypes.c_int()
        status = \
               _libcublas.cublasIdamax_v2(handle,
                                          n, int(x), incx, ctypes.byref(result))
        cublasCheckStatus(status)
        return result.value-1

cublasIdamax.__doc__ = \
                     I_AMAX_doc.substitute(precision='double-precision',
                                           real='real',
                                           data='np.random.rand(5).astype(np.float64)',
                                           func='cublasIdamax')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasIcamax.restype = int
    _libcublas.cublasIcamax.argtypes = [ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
    def cublasIcamax(n, x, incx):
        result = _libcublas.cublasIcamax(n, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)
        return result-1
else:
    _libcublas.cublasIcamax_v2.restype = int
    _libcublas.cublasIcamax_v2.argtypes = [ctypes.c_int,
                                           ctypes.c_int,
                                           ctypes.c_void_p,
                                           ctypes.c_int,
                                           ctypes.c_void_p]
    def cublasIcamax(n, x, incx):
        handle = cublasGetCurrentCtx()
        result = ctypes.c_int()
        status = \
               _libcublas.cublasIcamax_v2(handle,
                                          n, int(x), incx, ctypes.byref(result))
        cublasCheckStatus(status)
        return result.value-1

cublasIcamax.__doc__ = \
                     I_AMAX_doc.substitute(precision='single precision',
                                           real='complex',
                                           data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                           func='cublasIcamax')

if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasIzamax.restype = int
    _libcublas.cublasIzamax.argtypes = [ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
    def cublasIzamax(n, x, incx):    
        result = _libcublas.cublasIzamax(n, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)
        return result-1
else:
    _libcublas.cublasIzamax_v2.restype = int
    _libcublas.cublasIzamax_v2.argtypes = [ctypes.c_int,
                                           ctypes.c_int,
                                           ctypes.c_void_p,
                                           ctypes.c_int,
                                           ctypes.c_void_p]
    def cublasIzamax(n, x, incx):
        handle = cublasGetCurrentCtx()
        result = ctypes.c_int()
        status = \
               _libcublas.cublasIzamax_v2(handle,
                                          n, int(x), incx, ctypes.byref(result))
        cublasCheckStatus(status)
        return result.value-1
    
cublasIzamax.__doc__ = \
                     I_AMAX_doc.substitute(precision='double precision',
                                           real='complex',
                                           data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                           func='cublasIzamax')

# ISAMIN, IDAMIN, ICAMIN, IZAMIN
I_AMIN_doc = Template(
"""
    Index of minimum magnitude element (${precision} ${real}).

    Finds the smallest index of the minimum magnitude element of a
    ${precision} ${real} vector.

    Parameters
    ----------
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input vector.
    incx : int
        Storage spacing between elements of `x`.

    Returns
    -------
    idx : int
        Index of minimum magnitude element.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> m = ${func}(x_gpu.size, x_gpu.gpudata, 1)
    >>> np.allclose(m, np.argmin(x))
    True

    Notes
    -----
    This function returns a 0-based index.

    """
)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasIsamin.restype = int
    _libcublas.cublasIsamin.argtypes = [ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
    def cublasIsamin(n, x, incx):
        result = _libcublas.cublasIsamin(n, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)        
        return result-1
else:
    _libcublas.cublasIsamin_v2.restype = int
    _libcublas.cublasIsamin_v2.argtypes = [ctypes.c_int,
                                           ctypes.c_int,
                                           ctypes.c_void_p,
                                           ctypes.c_int,
                                           ctypes.c_void_p]
    def cublasIsamin(n, x, incx):
        handle = cublasGetCurrentCtx()
        result = ctypes.c_int()
        status = \
               _libcublas.cublasIsamin_v2(handle,
                                          n, int(x), incx, ctypes.byref(result))
        cublasCheckStatus(status)
        return result.value-1

cublasIsamin.__doc__ = \
                     I_AMIN_doc.substitute(precision='single-precision',
                                           real='real',
                                           data='np.random.rand(5).astype(np.float32)',
                                           func='cublasIsamin')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasIdamin.restype = int
    _libcublas.cublasIdamin.argtypes = [ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
    def cublasIdamin(n, x, incx):
        result = _libcublas.cublasIdamin(n, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)        
        return result-1
else:
    _libcublas.cublasIdamin_v2.restype = int
    _libcublas.cublasIdamin_v2.argtypes = [ctypes.c_int,
                                           ctypes.c_int,
                                           ctypes.c_void_p,
                                           ctypes.c_int,
                                           ctypes.c_void_p]
    def cublasIdamin(n, x, incx):
        handle = cublasGetCurrentCtx()
        result = ctypes.c_int()
        status = \
               _libcublas.cublasIdamin_v2(handle,
                                          n, int(x), incx, ctypes.byref(result))
        cublasCheckStatus(status)
        return result.value-1

cublasIdamin.__doc__ = \
                     I_AMIN_doc.substitute(precision='double-precision',
                                           real='real',
                                           data='np.random.rand(5).astype(np.float64)',
                                           func='cublasIdamin')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasIcamin.restype = int
    _libcublas.cublasIcamin.argtypes = [ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
    def cublasIcamin(n, x, incx):
        result = _libcublas.cublasIcamin(n, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)        
        return result-1
else:
    _libcublas.cublasIcamin_v2.restype = int
    _libcublas.cublasIcamin_v2.argtypes = [ctypes.c_int,
                                           ctypes.c_int,
                                           ctypes.c_void_p,
                                           ctypes.c_int,
                                           ctypes.c_void_p]
    def cublasIcamin(n, x, incx):
        handle = cublasGetCurrentCtx()
        result = ctypes.c_int()
        status = \
               _libcublas.cublasIcamin_v2(handle,
                                          n, int(x), incx, ctypes.byref(result))
        cublasCheckStatus(status)
        return result.value-1

cublasIcamin.__doc__ = \
                     I_AMIN_doc.substitute(precision='single-precision',
                                           real='complex',
                                           data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                           func='cublasIcamin')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasIzamin.restype = int
    _libcublas.cublasIzamin.argtypes = [ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
    def cublasIzamin(n, x, incx):
        result = _libcublas.cublasIzamin(n, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)        
        return result-1
else:
    _libcublas.cublasIzamin_v2.restype = int
    _libcublas.cublasIzamin_v2.argtypes = [ctypes.c_int,
                                           ctypes.c_int,
                                           ctypes.c_void_p,
                                           ctypes.c_int,
                                           ctypes.c_void_p]
    def cublasIzamin(n, x, incx):
        handle = cublasGetCurrentCtx()
        result = ctypes.c_int()
        status = \
               _libcublas.cublasIzamin_v2(handle,
                                          n, int(x), incx, ctypes.byref(result))
        cublasCheckStatus(status)
        return result.value-1

cublasIzamin.__doc__ = \
                     I_AMIN_doc.substitute(precision='double-precision',
                                           real='complex',
                                           data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                           func='cublasIzamin')

# SASUM, DASUM, SCASUM, DZASUM
_ASUM_doc = Template(                    
"""
    Sum of absolute values of ${precision} ${real} vector.

    Computes the sum of the absolute values of the elements of a
    ${precision} ${real} vector.

    Parameters
    ----------
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input vector.
    incx : int
        Storage spacing between elements of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> s = ${func}(x_gpu.size, x_gpu.gpudata, 1)
    >>> np.allclose(s, np.sum(np.abs(x)))
    True

    Returns
    -------
    s : ${ret_type}
        Sum of absolute values.
        
    """
)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasSasum.restype = ctypes.c_float
    _libcublas.cublasSasum.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasSasum(n, x, incx):
        result = _libcublas.cublasSasum(n, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)	
        return np.float32(result.value)
else:
    _libcublas.cublasSasum_v2.restype = int
    _libcublas.cublasSasum_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
    def cublasSasum(n, x, incx):
        handle = cublasGetCurrentCtx()
        result = ctypes.c_float()
        status = _libcublas.cublasSasum_v2(handle,
                                           n, int(x), incx, ctypes.byref(result))
        cublasCheckStatus(status)
        return np.float32(result.value)

cublasSasum.__doc__ = \
                    _ASUM_doc.substitute(precision='single-precision',
                                         real='real',
                                         data='np.random.rand(5).astype(np.float32)',
                                         func='cublasSasum',
                                         ret_type='numpy.float32')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDasum.restype = ctypes.c_double
    _libcublas.cublasDasum.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasDasum(n, x, incx):
        result = _libcublas.cublasDasum(n, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)
        return np.float64(result.value)
else:
    _libcublas.cublasDasum_v2.restype = int
    _libcublas.cublasDasum_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
    def cublasDasum(n, x, incx):
        handle = cublasGetCurrentCtx()
        result = ctypes.c_double()
        status = _libcublas.cublasDasum_v2(handle,
                                           n, int(x), incx, ctypes.byref(result))
        cublasCheckStatus(status)
        return np.float64(result.value)

cublasDasum.__doc__ = \
                    _ASUM_doc.substitute(precision='double-precision',
                                         real='real',
                                         data='np.random.rand(5).astype(np.float64)',
                                         func='cublasDasum',
                                         ret_type='numpy.float64')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasScasum.restype = ctypes.c_float
    _libcublas.cublasScasum.argtypes = [ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
    def cublasScasum(n, x, incx):    
        result = _libcublas.cublasScasum(n, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)
        return np.float32(result.value)
else:
    _libcublas.cublasScasum_v2.restype = int
    _libcublas.cublasScasum_v2.argtypes = [ctypes.c_int,
                                           ctypes.c_int,
                                           ctypes.c_void_p,
                                           ctypes.c_int,
                                           ctypes.c_void_p]
    def cublasScasum(n, x, incx):
        handle = cublasGetCurrentCtx()
        result = ctypes.c_float()
        status = _libcublas.cublasScasum_v2(handle,
                                            n, int(x), incx, ctypes.byref(result))
        cublasCheckStatus(status)
        return np.float32(result.value)
    
cublasScasum.__doc__ = \
                     _ASUM_doc.substitute(precision='single-precision',
                                          real='complex',
                                          data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                          func='cublasScasum',
                                          ret_type='numpy.float32')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDzasum.restype = ctypes.c_float
    _libcublas.cublasDzasum.argtypes = [ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
    def cublasDzasum(n, x, incx):    
        result = _libcublas.cublasDzasum(n, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)
        return np.float64(result.value)
else:
    _libcublas.cublasDzasum_v2.restype = int
    _libcublas.cublasDzasum_v2.argtypes = [ctypes.c_int,
                                           ctypes.c_int,
                                           ctypes.c_void_p,
                                           ctypes.c_int,
                                           ctypes.c_void_p]
    def cublasDzasum(n, x, incx):
        handle = cublasGetCurrentCtx()
        result = ctypes.c_double()
        status = _libcublas.cublasDzasum_v2(handle,
                                            n, int(x), incx, ctypes.byref(result))
        cublasCheckStatus(status)
        return np.float64(result.value)

cublasDzasum.__doc__ = \
                     _ASUM_doc.substitute(precision='double-precision',
                                          real='complex',
                                          data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                          func='cublasDzasum',
                                          ret_type='numpy.float64')

# SAXPY, DAXPY, CAXPY, ZAXPY
_AXPY_doc = Template(
"""
    Vector addition (${precision} ${real}).

    Computes the sum of a ${precision} ${real} vector scaled by a
    ${precision} ${real} scalar and another ${precision} ${real} vector.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    alpha : ${type}
        Scalar.
    x : ctypes.c_void_p
        Pointer to single-precision input vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to single-precision input/output vector.
    incy : int
        Storage spacing between elements of `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> alpha = ${alpha} 
    >>> x = ${data}
    >>> y = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> ${func}(x_gpu.size, alpha, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> np.allclose(y_gpu.get(), alpha*x+y)
    True

    Notes
    -----
    Both `x` and `y` must contain `n` elements.
    
    """
)

if cuda.cudaDriverGetVersion() < 4000: 
    _libcublas.cublasSaxpy.restype = None
    _libcublas.cublasSaxpy.argtypes = [ctypes.c_int,
                                       ctypes.c_float,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]

    def cublasSaxpy(n, alpha, x, incx, y, incy):
        _libcublas.cublasSaxpy(n, alpha, int(x), incx, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasSaxpy_v2.restype = int
    _libcublas.cublasSaxpy_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasSaxpy(n, alpha, x, incx, y, incy):
        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasSaxpy_v2(handle,
                                           n, ctypes.byref(ctypes.c_float(alpha)),
                                           int(x), incx, int(y), incy)
        cublasCheckStatus(status)

cublasSaxpy.__doc__ = \
                    _AXPY_doc.substitute(precision='single-precision',
                                         real='real',
                                         type='numpy.float32',
                                         alpha='np.float32(np.random.rand())',
                                         data='np.random.rand(5).astype(np.float32)',
                                         func='cublasSaxpy')

if cuda.cudaDriverGetVersion() < 4000: 
    _libcublas.cublasDaxpy.restype = None
    _libcublas.cublasDaxpy.argtypes = [ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]

    def cublasDaxpy(n, alpha, x, incx, y, incy):
        _libcublas.cublasDaxpy(n, alpha, int(x), incx, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasDaxpy_v2.restype = int
    _libcublas.cublasDaxpy_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasDaxpy(n, alpha, x, incx, y, incy):
        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasDaxpy_v2(handle,
                                           n, ctypes.byref(ctypes.c_double(alpha)),
                                           int(x), incx, int(y), incy)
        cublasCheckStatus(status)

cublasDaxpy.__doc__ = \
                    _AXPY_doc.substitute(precision='double-precision',
                                         real='real',
                                         type='numpy.float64',
                                         alpha='np.float64(np.random.rand())',
                                         data='np.random.rand(5).astype(np.float64)',
                                         func='cublasDaxpy')

if cuda.cudaDriverGetVersion() < 4000: 
    _libcublas.cublasCaxpy.restype = None
    _libcublas.cublasCaxpy.argtypes = [ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]

    def cublasCaxpy(n, alpha, x, incx, y, incy):
        _libcublas.cublasCaxpy(n, cuda.cuFloatComplex(alpha.real, alpha.imag), 
                               int(x), incx, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasCaxpy_v2.restype = int
    _libcublas.cublasCaxpy_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasCaxpy(n, alpha, x, incx, y, incy):
        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasCaxpy_v2(handle, n,
                                           ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)),
                                           int(x), incx, int(y), incy)
        cublasCheckStatus(status)

cublasCaxpy.__doc__ = \
                    _AXPY_doc.substitute(precision='single-precision',
                                         real='complex',
                                         type='numpy.complex64',
                                         alpha='(np.random.rand()+1j*np.random.rand()).astype(np.complex64)',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',             
                                         func='cublasCaxpy')

if cuda.cudaDriverGetVersion() < 4000: 
    _libcublas.cublasZaxpy.restype = None
    _libcublas.cublasZaxpy.argtypes = [ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]

    def cublasZaxpy(n, alpha, x, incx, y, incy):
        _libcublas.cublasZaxpy(n, cuda.cuDoubleComplex(alpha.real, alpha.imag), 
                               int(x), incx, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasZaxpy_v2.restype = int
    _libcublas.cublasZaxpy_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasZaxpy(n, alpha, x, incx, y, incy):
        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasZaxpy_v2(handle, n,
                                           ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)),
                                           int(x), incx, int(y), incy)
        cublasCheckStatus(status)

cublasZaxpy.__doc__ = \
                    _AXPY_doc.substitute(precision='double-precision',
                                         real='complex',
                                         type='numpy.complex128',
                                         alpha='(np.random.rand()+1j*np.random.rand()).astype(np.complex128)',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',             
                                         func='cublasZaxpy')

# SCOPY, DCOPY, CCOPY, ZCOPY
_COPY_doc = Template(
"""
    Vector copy (${precision} ${real})

    Copies a ${precision} ${real} vector to another ${precision} ${real}
    vector.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to ${precision} ${real} output vector.
    incy : int
        Storage spacing between elements of `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.zeros_like(x_gpu)
    >>> ${func}(x_gpu.size, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> np.allclose(y_gpu.get(), x_gpu.get())
    True
    
    Notes
    -----
    Both `x` and `y` must contain `n` elements.

""")

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasScopy.restype = None
    _libcublas.cublasScopy.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasScopy(n, x, incx, y, incy):
        _libcublas.cublasScopy(n, int(x), incx, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasScopy_v2.restype = int
    _libcublas.cublasScopy_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasScopy(n, x, incx, y, incy):
        handle = cublasGetCurrentCtx()
        status = \
               _libcublas.cublasScopy_v2(handle,
                                         n, int(x), incx, int(y), incy)
        cublasCheckStatus(status)
                
cublasScopy.__doc__ = \
                    _COPY_doc.substitute(precision='single-precision',
                                         real='real',
                                         data='np.random.rand(5).astype(np.float32)',
                                         func='cublasScopy')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDcopy.restype = None
    _libcublas.cublasDcopy.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasDcopy(n, x, incx, y, incy):
        _libcublas.cublasDcopy(n, int(x), incx, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasDcopy_v2.restype = int
    _libcublas.cublasDcopy_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasDcopy(n, x, incx, y, incy):
        handle = cublasGetCurrentCtx()
        status = \
               _libcublas.cublasDcopy_v2(handle,
                                         n, int(x), incx, int(y), incy)
        cublasCheckStatus(status)
                
cublasDcopy.__doc__ = \
                    _COPY_doc.substitute(precision='double-precision',
                                         real='real',
                                         data='np.random.rand(5).astype(np.float64)',
                                         func='cublasDcopy')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasCcopy.restype = None
    _libcublas.cublasCcopy.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasCcopy(n, x, incx, y, incy):
        _libcublas.cublasCcopy(n, int(x), incx, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasCcopy_v2.restype = int
    _libcublas.cublasCcopy_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasCcopy(n, x, incx, y, incy):
        handle = cublasGetCurrentCtx()
        status = \
               _libcublas.cublasCcopy_v2(handle,
                                         n, int(x), incx, int(y), incy)
        cublasCheckStatus(status)
                
cublasCcopy.__doc__ = \
                    _COPY_doc.substitute(precision='single-precision',
                                         real='complex',
                                         data='(np.random.rand(5)+np.random.rand(5).astype(np.complex64)',
                                         func='cublasCcopy')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZcopy.restype = None
    _libcublas.cublasZcopy.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasZcopy(n, x, incx, y, incy):
        _libcublas.cublasZcopy(n, int(x), incx, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasZcopy_v2.restype = int
    _libcublas.cublasZcopy_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasZcopy(n, x, incx, y, incy):
        handle = cublasGetCurrentCtx()
        status = \
               _libcublas.cublasZcopy_v2(handle,
                                         n, int(x), incx, int(y), incy)
        cublasCheckStatus(status)
                
cublasZcopy.__doc__ = \
                    _COPY_doc.substitute(precision='double-precision',
                                         real='complex',
                                         data='(np.random.rand(5)+np.random.rand(5).astype(np.complex128)',
                                         func='cublasZcopy')

# SDOT, DDOT, CDOTU, CDOTC, ZDOTU, ZDOTC
_DOT_doc = Template(
"""
    Vector dot product (${precision} ${real})

    Computes the dot product of two ${precision} ${real} vectors.
    cublasCdotc and cublasZdotc use the conjugate of the first vector
    when computing the dot product.
    
    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to ${precision} ${real} input/output vector.
    incy : int
        Storage spacing between elements of `y`.

    Returns
    -------
    d : ${ret_type}
        Dot product of `x` and `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> y = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> d = ${func}(x_gpu.size, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> ${check} 
    True

    Notes
    -----
    Both `x` and `y` must contain `n` elements.
    
""")

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasSdot.restype = ctypes.c_float
    _libcublas.cublasSdot.argtypes = [ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
    def cublasSdot(n, x, incx, y, incy):
        result = _libcublas.cublasSdot(n, int(x), incx, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
        return np.float32(result.value)
else:
    _libcublas.cublasSdot_v2.restype = int
    _libcublas.cublasSdot_v2.argtypes = [ctypes.c_int,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p]
    def cublasSdot(n, x, incx, y, incy):
        handle = cublasGetCurrentCtx()
        result = ctypes.c_float()
        status = _libcublas.cublasSdot_v2(handle, n,
                                          int(x), incx, int(y), incy,
                                          ctypes.byref(result))
        cublasCheckStatus(status)
        return np.float32(result.value)

cublasSdot.__doc__ = _DOT_doc.substitute(precision='single-precision',
                                         real='real',
                                         data='np.float32(np.random.rand(5))',
                                         ret_type='np.float32',
                                         func='cublasSdot',
                                         check='np.allclose(d, np.dot(x, y))')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDdot.restype = ctypes.c_double
    _libcublas.cublasDdot.argtypes = [ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
    def cublasDdot(n, x, incx, y, incy):
        result = _libcublas.cublasDdot(n, int(x), incx, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
        return np.float64(result.value)
else:
    _libcublas.cublasDdot_v2.restype = int
    _libcublas.cublasDdot_v2.argtypes = [ctypes.c_int,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p]
    def cublasDdot(n, x, incx, y, incy):
        handle = cublasGetCurrentCtx()
        result = ctypes.c_double()
        status = _libcublas.cublasDdot_v2(handle, n,
                                          int(x), incx, int(y), incy,
                                          ctypes.byref(result))
        cublasCheckStatus(status)
        return np.float64(result.value)

cublasDdot.__doc__ = _DOT_doc.substitute(precision='double-precision',
                                         real='real',
                                         data='np.float64(np.random.rand(5))',
                                         ret_type='np.float64',
                                         func='cublasDdot',
                                         check='np.allclose(d, np.dot(x, y))')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasCdotu.restype = cuda.cuFloatComplex
    _libcublas.cublasCdotu.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasCdotu(n, x, incx, y, incy):
        result = _libcublas.cublasCdotu(n, int(x), incx, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
        return np.complex64(result.value)
else:
    _libcublas.cublasCdotu_v2.restype = int
    _libcublas.cublasCdotu_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
    def cublasCdotu(n, x, incx, y, incy):
        handle = cublasGetCurrentCtx()
        result = cuda.cuFloatComplex()
        status = _libcublas.cublasCdotu_v2(handle, n,
                                           int(x), incx, int(y), incy,
                                           ctypes.byref(result))
        cublasCheckStatus(status)
        return np.complex64(result.value)

cublasCdotu.__doc__ = _DOT_doc.substitute(precision='single-precision',
                                         real='complex',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                         ret_type='np.complex64',
                                         func='cublasCdotu',
                                         check='np.allclose(d, np.dot(x, y))')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasCdotc.restype = cuda.cuFloatComplex
    _libcublas.cublasCdotc.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasCdotc(n, x, incx, y, incy):
        result = _libcublas.cublasCdotc(n, int(x), incx, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
        return np.complex64(result.value)
else:
    _libcublas.cublasCdotc_v2.restype = int
    _libcublas.cublasCdotc_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
    def cublasCdotc(n, x, incx, y, incy):
        handle = cublasGetCurrentCtx()
        result = cuda.cuFloatComplex()
        status = _libcublas.cublasCdotc_v2(handle, n,
                                           int(x), incx, int(y), incy,
                                           ctypes.byref(result))
        cublasCheckStatus(status)
        return np.complex64(result.value)

cublasCdotc.__doc__ = _DOT_doc.substitute(precision='single-precision',
                                         real='complex',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                         ret_type='np.complex64',
                                         func='cublasCdotc',
                                         check='np.allclose(d, np.dot(np.conj(x), y))')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZdotu.restype = cuda.cuDoubleComplex
    _libcublas.cublasZdotu.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasZdotu(n, x, incx, y, incy):
        result = _libcublas.cublasZdotu(n, int(x), incx, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
        return np.complex128(result.value)
else:
    _libcublas.cublasZdotu_v2.restype = int
    _libcublas.cublasZdotu_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
    def cublasZdotu(n, x, incx, y, incy):
        handle = cublasGetCurrentCtx()
        result = cuda.cuDoubleComplex()
        status = _libcublas.cublasZdotu_v2(handle, n,
                                           int(x), incx, int(y), incy,
                                           ctypes.byref(result))
        cublasCheckStatus(status)
        return np.complex128(result.value)

cublasZdotu.__doc__ = _DOT_doc.substitute(precision='double-precision',
                                          real='complex',
                                          data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                          ret_type='np.complex128',
                                          func='cublasZdotu',
                                          check='np.allclose(d, np.dot(x, y))')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZdotc.restype = cuda.cuDoubleComplex
    _libcublas.cublasZdotc.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasZdotc(n, x, incx, y, incy):
        result = _libcublas.cublasZdotc(n, int(x), incx, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
        return np.complex128(result.value)
else:
    _libcublas.cublasZdotc_v2.restype = int
    _libcublas.cublasZdotc_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
    def cublasZdotc(n, x, incx, y, incy):
        handle = cublasGetCurrentCtx()
        result = cuda.cuDoubleComplex()
        status = _libcublas.cublasZdotc_v2(handle, n,
                                           int(x), incx, int(y), incy,
                                           ctypes.byref(result))
        cublasCheckStatus(status)
        return np.complex128(result.value)

cublasZdotc.__doc__ = _DOT_doc.substitute(precision='double-precision',
                                         real='complex',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                         ret_type='np.complex128',
                                         func='cublasZdotc',
                                         check='np.allclose(d, np.dot(np.conj(x), y))')

# SNRM2, DNRM2, SCNRM2, DZNRM2
_NRM2_doc = Template(
"""
    Euclidean norm (2-norm) of real vector.

    Computes the Euclidean norm of a ${precision} ${real} vector.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input vector.
    incx : int
        Storage spacing between elements of `x`.

    Returns
    -------
    nrm : ${ret_type}
        Euclidean norm of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> nrm = cublasSnrm2(x.size, x_gpu.gpudata, 1)
    >>> np.allclose(nrm, np.linalg.norm(x))
    True
    
""")

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasSnrm2.restype = ctypes.c_float
    _libcublas.cublasSnrm2.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasSnrm2(n, x, incx):
        result = _libcublas.cublasSnrm2(n, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)
        return np.float32(result.value)
else:
    _libcublas.cublasSnrm2_v2.restype = int
    _libcublas.cublasSnrm2_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
    def cublasSnrm2(n, x, incx):
        handle = cublasGetCurrentCtx()
        result = ctypes.c_float()
        status = _libcublas.cublasSnrm2_v2(handle,
                                           n, int(x), incx,
                                           ctypes.byref(result))
        cublasCheckStatus(status)
        return np.float32(result.value)
    
cublasSnrm2.__doc__ = \
                    _NRM2_doc.substitute(precision='single-precision',
                                         real='real',
                                         data='np.float32(np.random.rand(5))',
                                         ret_type = 'numpy.float32',
                                         func='cublasSnrm2')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDnrm2.restype = ctypes.c_double
    _libcublas.cublasDnrm2.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasDnrm2(n, x, incx):
        result = _libcublas.cublasDnrm2(n, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)
        return np.float64(result.value)
else:
    _libcublas.cublasDnrm2_v2.restype = int
    _libcublas.cublasDnrm2_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
    def cublasDnrm2(n, x, incx):
        handle = cublasGetCurrentCtx()
        result = ctypes.c_double()
        status = _libcublas.cublasDnrm2_v2(handle,
                                           n, int(x), incx,
                                           ctypes.byref(result))
        cublasCheckStatus(status)
        return np.float64(result.value)
    
cublasDnrm2.__doc__ = \
                    _NRM2_doc.substitute(precision='double-precision',
                                         real='real',
                                         data='np.float64(np.random.rand(5))',
                                         ret_type = 'numpy.float64',
                                         func='cublasDnrm2')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasScnrm2.restype = cuda.cuFloatComplex
    _libcublas.cublasScnrm2.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasScnrm2(n, x, incx):
        result = _libcublas.cublasScnrm2(n, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)
        return np.complex64(result.value)
else:
    _libcublas.cublasScnrm2_v2.restype = int
    _libcublas.cublasScnrm2_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
    def cublasScnrm2(n, x, incx):
        handle = cublasGetCurrentCtx()
        result = cuda.cuFloatComplex()
        status = _libcublas.cublasScnrm2_v2(handle,
                                            n, int(x), incx,
                                            ctypes.byref(result))
        cublasCheckStatus(status)
        return np.complex64(result.value)
    
cublasScnrm2.__doc__ = \
                    _NRM2_doc.substitute(precision='single-precision',
                                         real='complex',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                         ret_type = 'numpy.complex64',
                                         func='cublasScnrm2')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDznrm2.restype = cuda.cuDoubleComplex
    _libcublas.cublasDznrm2.argtypes = [ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
    def cublasDznrm2(n, x, incx):
        result = _libcublas.cublasDznrm2(n, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)
        return np.complex128(result.value)
else:
    _libcublas.cublasDznrm2_v2.restype = int
    _libcublas.cublasDznrm2_v2.argtypes = [ctypes.c_int,
                                           ctypes.c_int,
                                           ctypes.c_void_p,
                                           ctypes.c_int,
                                           ctypes.c_void_p]
    def cublasDznrm2(n, x, incx):
        handle = cublasGetCurrentCtx()
        result = cuda.cuDoubleComplex()
        status = _libcublas.cublasDznrm2_v2(handle,
                                            n, int(x), incx,
                                            ctypes.byref(result))
        cublasCheckStatus(status)
        return np.complex128(result.value)
    
cublasDznrm2.__doc__ = \
                    _NRM2_doc.substitute(precision='double-precision',
                                         real='complex',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                         ret_type = 'numpy.complex128',
                                         func='cublasDznrm2')


# SROT, DROT, CROT, CSROT, ZROT, ZDROT
_ROT_doc = Template(
"""
    Apply a ${real} rotation to ${real} vectors (${precision})

    Multiplies the ${precision} matrix `[[c, s], [-s, c]]`
    with the 2 x `n` ${precision} matrix `[[x.T], [y.T]]`.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input/output vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to ${precision} ${real} input/output vector.
    incy : int
        Storage spacing between elements of `y`.
    c : ${c_type}
        Element of rotation matrix.
    s : ${s_type}
        Element of rotation matrix.

    Notes
    -----
    Both `x` and `y` must contain `n` elements.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> s = ${s_val}; c = ${c_val};
    >>> x = ${data}
    >>> y = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> ${func}(x.size, x_gpu.gpudata, 1, y_gpu.gpudata, 1, c, s)
    >>> np.allclose(x_gpu.get(), c*x+s*y)
    True
    >>> np.allclose(y_gpu.get(), -s*x+c*y)
    True
    
""")

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasSrot.restype = None
    _libcublas.cublasSrot.argtypes = [ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_float,
                                      ctypes.c_float]
    def cublasSrot(n, x, incx, y, incy, c, s):
        _libcublas.cublasSrot(n, int(x), incx, int(y), incy, c, s)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasSrot_v2.restype = int
    _libcublas.cublasSrot_v2.argtypes = [ctypes.c_int,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_void_p]
    def cublasSrot(n, x, incx, y, incy, c, s):
        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasSrot_v2(handle,
                                          n, int(x),
                                          incx, int(y), incy,
                                          ctypes.byref(ctypes.c_float(c)),
                                          ctypes.byref(ctypes.c_float(s)))
        cublasCheckStatus(status)
        
cublasSrot.__doc__ = _ROT_doc.substitute(precision='single-precision',
                                         real='real',
                                         c_type='numpy.float32',
                                         s_type='numpy.float32',
                                         c_val='np.float32(np.random.rand())',
                                         s_val='np.float32(np.random.rand())',
                                         data='np.random.rand(5).astype(np.float32)',
                                         func='cublasSrot')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDrot.restype = None
    _libcublas.cublasDrot.argtypes = [ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_double,
                                      ctypes.c_double]
    def cublasDrot(n, x, incx, y, incy, c, s):
        _libcublas.cublasDrot(n, int(x), incx, int(y), incy, c, s)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasDrot_v2.restype = int
    _libcublas.cublasDrot_v2.argtypes = [ctypes.c_int,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_void_p]
    def cublasDrot(n, x, incx, y, incy, c, s):
        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasDrot_v2(handle,
                                          n, int(x),
                                          incx, int(y), incy,
                                          ctypes.byref(ctypes.c_double(c)),
                                          ctypes.byref(ctypes.c_double(s)))
        cublasCheckStatus(status)
        
cublasDrot.__doc__ = _ROT_doc.substitute(precision='double-precision',
                                         real='real',
                                         c_type='numpy.float64',
                                         s_type='numpy.float64',
                                         c_val='np.float64(np.random.rand())',
                                         s_val='np.float64(np.random.rand())',
                                         data='np.random.rand(5).astype(np.float64)',
                                         func='cublasDrot')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasCrot.restype = None
    _libcublas.cublasCrot.argtypes = [ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_float,
                                      cuda.cuFloatComplex]
    def cublasCrot(n, x, incx, y, incy, c, s):
        _libcublas.cublasCrot(n, int(x), incx, int(y), incy,
                              c,
                              cuda.cuFloatComplex(s.real, s.imag))
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasCrot_v2.restype = int
    _libcublas.cublasCrot_v2.argtypes = [ctypes.c_int,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_void_p]
    def cublasCrot(n, x, incx, y, incy, c, s):
        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasCrot_v2(handle,
                                          n, int(x),
                                          incx, int(y), incy,
                                          ctypes.byref(ctypes.c_float(c)),
                                          ctypes.byref(cuda.cuFloatComplex(s.real,
                                                                           s.imag)))
        cublasCheckStatus(status)
        
cublasCrot.__doc__ = _ROT_doc.substitute(precision='single-precision',
                                         real='complex',
                                         c_type='numpy.float32',
                                         s_type='numpy.complex64',
                                         c_val='np.float32(np.random.rand())',
                                         s_val='np.complex64(np.random.rand()+1j*np.random.rand())',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                         func='cublasCrot')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasCsrot.restype = None
    _libcublas.cublasCsrot.argtypes = [ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_float,
                                      ctypes.c_float]
    def cublasCsrot(n, x, incx, y, incy, c, s):
        _libcublas.cublasCsrot(n, int(x), incx, int(y), incy,
                               c, s)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasCsrot_v2.restype = int
    _libcublas.cublasCsrot_v2.argtypes = [ctypes.c_int,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_void_p]
    def cublasCsrot(n, x, incx, y, incy, c, s):
        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasCsrot_v2(handle,
                                           n, int(x),
                                           incx, int(y), incy,
                                           c, s)
        cublasCheckStatus(status)
        
cublasCsrot.__doc__ = _ROT_doc.substitute(precision='single-precision',
                                          real='complex',
                                          c_type='numpy.float32',
                                          s_type='numpy.float32',
                                          c_val='np.float32(np.random.rand())',
                                          s_val='np.float32(np.random.rand())',
                                          data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                          func='cublasCsrot')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZrot.restype = None
    _libcublas.cublasZrot.argtypes = [ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_double,
                                      cuda.cuDoubleComplex]
    def cublasZrot(n, x, incx, y, incy, c, s):
        _libcublas.cublasZrot(n, int(x), incx, int(y), incy,
                               c, cuda.cuDoubleComplex(s.real, s.imag))
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasZrot_v2.restype = int
    _libcublas.cublasZrot_v2.argtypes = [ctypes.c_int,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_void_p]
    def cublasZrot(n, x, incx, y, incy, c, s):
        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasZrot_v2(handle,
                                          n, int(x),
                                          incx, int(y), incy,
                                          c,
                                          ctypes.byref(cuda.cuDoubleComplex(s.real, s.imag)))
        cublasCheckStatus(status)
        
cublasZrot.__doc__ = _ROT_doc.substitute(precision='double-precision',
                                         real='complex',
                                         c_type='numpy.float64',
                                         s_type='numpy.complex128',
                                         c_val='np.float64(np.random.rand())',
                                         s_val='np.complex128(np.random.rand()+1j*np.random.rand())',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                         func='cublasZrot')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZdrot.restype = None
    _libcublas.cublasZdrot.argtypes = [ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_double,
                                      ctypes.c_double]
    def cublasZdrot(n, x, incx, y, incy, c, s):
        _libcublas.cublasZdrot(n, int(x), incx, int(y), incy,
                               c, s)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasZdrot_v2.restype = int
    _libcublas.cublasZdrot_v2.argtypes = [ctypes.c_int,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_void_p]
    def cublasZdrot(n, x, incx, y, incy, c, s):
        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasZdrot_v2(handle,
                                           n, int(x),
                                           incx, int(y), incy,
                                           c, s)
        cublasCheckStatus(status)
        
cublasZdrot.__doc__ = _ROT_doc.substitute(precision='double-precision',
                                          real='complex',
                                          c_type='numpy.float64',
                                          s_type='numpy.float64',
                                          c_val='np.float64(np.random.rand())',
                                          s_val='np.float64(np.random.rand())',
                                          data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                          func='cublasZdrot')


# SROTG, DROTG, CROTG, ZROTG
_ROTG_doc = Template(
"""
    Construct a ${precision} ${real} Givens rotation matrix.

    Constructs the ${precision} ${real} Givens rotation matrix
    `G = [[c, s], [-s, c]]` such that
    `dot(G, [[a], [b]] == [[r], [0]]`, where
    `c**2+s**2 == 1` and `r == a**2+b**2` for real numbers and
    `c**2+(conj(s)*s) == 1` and `r ==
    (a/abs(a))*sqrt(abs(a)**2+abs(b)**2)` for `a != 0` and `r == b`
    for `a == 0`.

    Parameters
    ----------
    a, b : ${type}
        Entries of vector whose second entry should be zeroed
        out by the rotation.

    Returns
    -------
    r : ${type}
        Defined above.
    c : ${c_type}
        Cosine component of rotation matrix.
    s : ${s_type}
        Sine component of rotation matrix.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> a = ${a_val}
    >>> b = ${b_val}
    >>> r, c, s = ${func}(a, b)
    >>> np.allclose(np.dot(np.array([[c, s], [-np.conj(s), c]]), np.array([[a], [b]])), np.array([[r], [0.0]]))
    True

""")

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasSrotg.restype = None
    _libcublas.cublasSrotg.argtypes = [ctypes.c_void_p,
                                       ctypes.c_float,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p]
    def cublasSrotg(a, b):
        _a = ctypes.c_float(a)
        _b = ctypes.c_float(b)
        _c = ctypes.c_float()
        _s = ctypes.c_float()
        _libcublas.cublasSrotg(ctypes.byref(_a), _b,
                               ctypes.byref(_c), ctypes.byref(_s))
        status = cublasGetError()
        cublasCheckStatus(status)
        return np.float32(_a.value), np.float32(_c.value), np.float32(_s.value)
else:
    _libcublas.cublasSrotg_v2.restype = int
    _libcublas.cublasSrotg_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
    def cublasSrotg(a, b):
        handle = cublasGetCurrentCtx()
        _a = ctypes.c_float(a)
        _b = ctypes.c_float(b)
        _c = ctypes.c_float()
        _s = ctypes.c_float()
        status = _libcublas.cublasSrotg_v2(handle,
                                           ctypes.byref(_a), ctypes.byref(_b),
                                           ctypes.byref(_c), ctypes.byref(_s))
        cublasCheckStatus(status)
        return np.float32(_a.value), np.float32(_c.value), np.float32(_s.value)
                                  
cublasSrotg.__doc__ = \
                    _ROTG_doc.substitute(precision='single-precision',
                                         real='real',
                                         type='numpy.float32',
                                         c_type='numpy.float32',
                                         s_type='numpy.float32',
                                         a_val='np.float32(np.random.rand())',
                                         b_val='np.float32(np.random.rand())',
                                         func='cublasSrotg')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDrotg.restype = None
    _libcublas.cublasDrotg.argtypes = [ctypes.c_void_p,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p]
    def cublasDrotg(a, b):
        _a = ctypes.c_double(a)
        _b = ctypes.c_double(b)
        _c = ctypes.c_double()
        _s = ctypes.c_double()
        _libcublas.cublasDrotg(ctypes.byref(_a), _b,
                               ctypes.byref(_c), ctypes.byref(_s))
        status = cublasGetError()
        cublasCheckStatus(status)
        return np.float64(_a.value), np.float64(_c.value), np.float64(_s.value)
else:
    _libcublas.cublasDrotg_v2.restype = int
    _libcublas.cublasDrotg_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
    def cublasDrotg(a, b):
        handle = cublasGetCurrentCtx()
        _a = ctypes.c_double(a)
        _b = ctypes.c_double(b)
        _c = ctypes.c_double()
        _s = ctypes.c_double()
        status = _libcublas.cublasDrotg_v2(handle,
                                           ctypes.byref(_a), ctypes.byref(_b),
                                           ctypes.byref(_c), ctypes.byref(_s))
        cublasCheckStatus(status)
        return np.float64(_a.value), np.float64(_c.value), np.float64(_s.value)
                                  
cublasDrotg.__doc__ = \
                    _ROTG_doc.substitute(precision='double-precision',
                                         real='real',
                                         type='numpy.float64',
                                         c_type='numpy.float64',
                                         s_type='numpy.float64',
                                         a_val='np.float64(np.random.rand())',
                                         b_val='np.float64(np.random.rand())',
                                         func='cublasDrotg')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasCrotg.restype = None
    _libcublas.cublasCrotg.argtypes = [ctypes.c_void_p,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p]
    def cublasCrotg(a, b):
        _a = cuda.cuFloatComplex(a.real, a.imag)
        _b = cuda.cuFloatComplex(b.real, b.imag)
        _c = ctypes.c_float()
        _s = cuda.cuFloatComplex()
        _libcublas.cublasCrotg(ctypes.byref(_a), _b,
                               ctypes.byref(_c), ctypes.byref(_s))
        status = cublasGetError()
        cublasCheckStatus(status)
        return np.complex64(_a.value), np.float32(_c.value), np.complex64(_s.value)
else:
    _libcublas.cublasCrotg_v2.restype = int
    _libcublas.cublasCrotg_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
    def cublasCrotg(a, b):
        handle = cublasGetCurrentCtx()
        _a = cuda.cuFloatComplex(a.real, a.imag)
        _b = cuda.cuFloatComplex(b.real, b.imag)
        _c = ctypes.c_float()
        _s = cuda.cuFloatComplex()
        status = _libcublas.cublasCrotg_v2(handle,
                                           ctypes.byref(_a), _b,
                                           ctypes.byref(_c), ctypes.byref(_s))
        cublasCheckStatus(status)
        return np.complex64(_a.value), np.float32(_c.value), np.complex64(_s.value)
                                  
cublasCrotg.__doc__ = \
                    _ROTG_doc.substitute(precision='single-precision',
                                         real='complex',
                                         type='numpy.complex64',
                                         c_type='numpy.float32',
                                         s_type='numpy.complex64',
                                         a_val='np.complex64(np.random.rand()+1j*np.random.rand())',
                                         b_val='np.complex64(np.random.rand()+1j*np.random.rand())',
                                         func='cublasCrotg')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZrotg.restype = None
    _libcublas.cublasZrotg.argtypes = [ctypes.c_void_p,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p]
    def cublasZrotg(a, b):
        _a = cuda.cuDoubleComplex(a.real, a.imag)
        _b = cuda.cuDoubleComplex(b.real, b.imag)
        _c = ctypes.c_double()
        _s = cuda.cuDoubleComplex()
        _libcublas.cublasZrotg(ctypes.byref(_a), _b,
                               ctypes.byref(_c), ctypes.byref(_s))
        status = cublasGetError()
        cublasCheckStatus(status)
        return np.complex128(_a.value), np.float64(_c.value), np.complex128(_s.value)
else:
    _libcublas.cublasZrotg_v2.restype = int
    _libcublas.cublasZrotg_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p]
    def cublasZrotg(a, b):
        handle = cublasGetCurrentCtx()
        _a = cuda.cuDoubleComplex(a.real, a.imag)
        _b = cuda.cuDoubleComplex(b.real, b.imag)
        _c = ctypes.c_double()
        _s = cuda.cuDoubleComplex()
        status = _libcublas.cublasZrotg_v2(handle,
                                           ctypes.byref(_a), _b,
                                           ctypes.byref(_c), ctypes.byref(_s))
        cublasCheckStatus(status)
        return np.complex128(_a.value), np.float64(_c.value), np.complex128(_s.value)
                                  
cublasZrotg.__doc__ = \
                    _ROTG_doc.substitute(precision='double-precision',
                                         real='complex',
                                         type='numpy.complex128',
                                         c_type='numpy.float64',
                                         s_type='numpy.complex128',
                                         a_val='np.complex128(np.random.rand()+1j*np.random.rand())',
                                         b_val='np.complex128(np.random.rand()+1j*np.random.rand())',
                                         func='cublasZrotg')

# SROTM, DROTM (need to add example)
_ROTM_doc = Template(        
"""
    Apply a ${precision} real modified Givens rotation.

    Applies the ${precision} real modified Givens rotation matrix `h`
    to the 2 x `n` matrix `[[x.T], [y.T]]`.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to ${precision} real input/output vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to ${precision} real input/output vector.
    incy : int
        Storage spacing between elements of `y`.
    sparam : numpy.ndarray
        sparam[0] contains the `flag` described below;
        sparam[1:5] contains the values `[h00, h10, h01, h11]`
        that determine the rotation matrix `h`.

    Notes
    -----
    The rotation matrix may assume the following values:

    for `flag` == -1.0, `h` == `[[h00, h01], [h10, h11]]`
    for `flag` == 0.0,  `h` == `[[1.0, h01], [h10, 1.0]]`
    for `flag` == 1.0,  `h` == `[[h00, 1.0], [-1.0, h11]]`
    for `flag` == -2.0, `h` == `[[1.0, 0.0], [0.0, 1.0]]`

    Both `x` and `y` must contain `n` elements.
    
""")

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasSrotm.restype = None
    _libcublas.cublasSrotm.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
    def cublasSrotm(n, x, incx, y, incy, sparam):
        _libcublas.cublasSrotm(n, int(x), incx, int(y), incy,
                               int(sparam.ctypes.data))
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasSrotm_v2.restype = int
    _libcublas.cublasSrotm_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
    def cublasSrotm(n, x, incx, y, incy, sparam):
        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasSrotm_v2(handle,
                                           n, int(x), incx, int(y),
                                           incy, int(sparam.ctypes.data))
        cublasCheckStatus(status)

cublasSrotm.__doc__ = \
                    _ROTM_doc.substitute(precision='single-precision')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDrotm.restype = None
    _libcublas.cublasDrotm.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
    def cublasDrotm(n, x, incx, y, incy, sparam):
        _libcublas.cublasDrotm(n, int(x), incx, int(y), incy,
                               int(sparam.ctypes.data))
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasDrotm_v2.restype = int
    _libcublas.cublasDrotm_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p]
    def cublasDrotm(n, x, incx, y, incy, sparam):
        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasDrotm_v2(handle,
                                           n, int(x), incx, int(y),
                                           incy, int(sparam.ctypes.data))
        cublasCheckStatus(status)

cublasDrotm.__doc__ = \
                    _ROTM_doc.substitute(precision='double-precision')
                                        
# SROTMG, DROTMG (need to add example)
_ROTMG_doc = Template( 
"""
    Construct a ${precision} real modified Givens rotation matrix.

    Constructs the ${precision} real modified Givens rotation matrix
    `h = [[h11, h12], [h21, h22]]` that zeros out the second entry of
    the vector `[[sqrt(d1)*x1], [sqrt(d2)*x2]]`.

    Parameters
    ----------
    d1 : ${type}
        ${precision} real value.
    d2 : ${type}
        ${precision} real value.
    x1 : ${type}
        ${precision} real value.
    x2 : ${type}
        ${precision} real value.

    Returns
    -------
    sparam : numpy.ndarray
        sparam[0] contains the `flag` described below;
        sparam[1:5] contains the values `[h00, h10, h01, h11]`
        that determine the rotation matrix `h`.
        
    Notes
    -----
    The rotation matrix may assume the following values:

    for `flag` == -1.0, `h` == `[[h00, h01], [h10, h11]]`
    for `flag` == 0.0,  `h` == `[[1.0, h01], [h10, 1.0]]`
    for `flag` == 1.0,  `h` == `[[h00, 1.0], [-1.0, h11]]`
    for `flag` == -2.0, `h` == `[[1.0, 0.0], [0.0, 1.0]]`

""")

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasSrotmg.restype = None
    _libcublas.cublasSrotmg.argtypes = [ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p]                         
    def cublasSrotmg(d1, d2, x1, y1):
        _d1 = ctypes.c_float(d1)
        _d2 = ctypes.c_float(d2)
        _x1 = ctypes.c_float(x1)
        _y1 = ctypes.c_float(y1)
        sparam = np.empty(5, np.float32)
        
        _libcublas.cublasSrotmg(ctypes.byref(_d1), ctypes.byref(_d2),
                                ctypes.byref(_x1), ctypes.byref(_y1),
                                int(sparam.ctypes.data))
        status = cublasGetError()
        cublasCheckStatus(status)        
        return sparam
else:
    _libcublas.cublasSrotmg_v2.restype = int
    _libcublas.cublasSrotmg_v2.argtypes = [ctypes.c_int,
                                           ctypes.c_void_p,
                                           ctypes.c_void_p,
                                           ctypes.c_void_p,
                                           ctypes.c_void_p,
                                           ctypes.c_void_p]
    def cublasSrotmg(d1, d2, x1, y1):
        handle = cublasGetCurrentCtx()
        _d1 = ctypes.c_float(d1)
        _d2 = ctypes.c_float(d2)
        _x1 = ctypes.c_float(x1)
        _y1 = ctypes.c_float(y1)
        sparam = np.empty(5, np.float32)

        status = _libcublas.cublasSrotmg_v2(handle,
                                            ctypes.byref(_d1), ctypes.byref(_d2),
                                            ctypes.byref(_x1), ctypes.byref(_y1),
                                            int(sparam.ctypes.data))
        cublasCheckStatus(status)        
        return sparam

cublasSrotmg.__doc__ = \
                     _ROTMG_doc.substitute(precision='single-precision',
                                           type='numpy.float32')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDrotmg.restype = None
    _libcublas.cublasDrotmg.argtypes = [ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p]                         
    def cublasDrotmg(d1, d2, x1, y1):
        _d1 = ctypes.c_double(d1)
        _d2 = ctypes.c_double(d2)
        _x1 = ctypes.c_double(x1)
        _y1 = ctypes.c_double(y1)
        sparam = np.empty(5, np.float64)
        
        _libcublas.cublasDrotmg(ctypes.byref(_d1), ctypes.byref(_d2),
                                ctypes.byref(_x1), ctypes.byref(_y1),
                                int(sparam.ctypes.data))
        status = cublasGetError()
        cublasCheckStatus(status)        
        return sparam
else:
    _libcublas.cublasDrotmg_v2.restype = int
    _libcublas.cublasDrotmg_v2.argtypes = [ctypes.c_int,
                                           ctypes.c_void_p,
                                           ctypes.c_void_p,
                                           ctypes.c_void_p,
                                           ctypes.c_void_p,
                                           ctypes.c_void_p]
    def cublasDrotmg(d1, d2, x1, y1):
        handle = cublasGetCurrentCtx()
        _d1 = ctypes.c_double(d1)
        _d2 = ctypes.c_double(d2)
        _x1 = ctypes.c_double(x1)
        _y1 = ctypes.c_double(y1)
        sparam = np.empty(5, np.float64)

        status = _libcublas.cublasDrotmg_v2(handle,
                                            ctypes.byref(_d1), ctypes.byref(_d2),
                                            ctypes.byref(_x1), ctypes.byref(_y1),
                                            int(sparam.ctypes.data))
        cublasCheckStatus(status)        
        return sparam

cublasDrotmg.__doc__ = \
                     _ROTMG_doc.substitute(precision='double-precision',
                                           type='numpy.float64')

# SSCAL, DSCAL, CSCAL, CSCAL, CSSCAL, ZSCAL, ZDSCAL
_SCAL_doc = Template(
"""
    Scale a ${precision} ${real} vector by a ${precision} ${a_real} scalar.

    Replaces a ${precision} ${real} vector `x` with
    `alpha * x`.
    
    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    alpha : ${a_type}
        Scalar multiplier.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input/output vector.
    incx : int
        Storage spacing between elements of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> alpha = ${alpha}
    >>> ${func}(x.size, alpha, x_gpu.gpudata, 1)
    >>> np.allclose(x_gpu.get(), alpha*x)
    True
    
""")

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasSscal.restype = None
    _libcublas.cublasSscal.argtypes = [ctypes.c_int,
                                       ctypes.c_float,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasSscal(n, alpha, x, incx):
        _libcublas.cublasSscal(n, alpha, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasSscal_v2.restype = int
    _libcublas.cublasSscal_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasSscal(n, alpha, x, incx):
        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasSscal_v2(handle, n,
                                           ctypes.byref(ctypes.c_float(alpha)),
                                           int(x), incx)
        cublasCheckStatus(status)
        
cublasSscal.__doc__ = \
                    _SCAL_doc.substitute(precision='single-precision',
                                         real='real',
                                         a_real='real',
                                         a_type='numpy.float32',
                                         alpha='np.float32(np.random.rand())',
                                         data='np.random.rand(5).astype(np.float32)',
                                         func='cublasSscal')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDscal.restype = None
    _libcublas.cublasDscal.argtypes = [ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasDscal(n, alpha, x, incx):
        _libcublas.cublasDscal(n, alpha, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasDscal_v2.restype = int
    _libcublas.cublasDscal_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasDscal(n, alpha, x, incx):
        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasDscal_v2(handle, n,
                                           ctypes.byref(ctypes.c_double(alpha)),
                                           int(x), incx)
        cublasCheckStatus(status)
        
cublasDscal.__doc__ = \
                    _SCAL_doc.substitute(precision='double-precision',
                                         real='real',
                                         a_real='real',
                                         a_type='numpy.float64',
                                         alpha='np.float64(np.random.rand())',
                                         data='np.random.rand(5).astype(np.float64)',
                                         func='cublasDscal')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasCscal.restype = None
    _libcublas.cublasCscal.argtypes = [ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasCscal(n, alpha, x, incx):
        _libcublas.cublasCscal(n, cuda.cuFloatComplex(alpha.real, alpha.imag), int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasCscal_v2.restype = int
    _libcublas.cublasCscal_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasCscal(n, alpha, x, incx):
        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasCscal_v2(handle, n,
                                           ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                            alpha.imag)),
                                           int(x), incx)
        cublasCheckStatus(status)
        
cublasCscal.__doc__ = \
                    _SCAL_doc.substitute(precision='single-precision',
                                         real='complex',
                                         a_real='complex',
                                         a_type='numpy.complex64',
                                         alpha='np.complex64(np.random.rand()+1j*np.random.rand())',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                         func='cublasCscal')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasCsscal.restype = None
    _libcublas.cublasCsscal.argtypes = [ctypes.c_int,
                                        ctypes.c_float,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
    def cublasCsscal(n, alpha, x, incx):
        _libcublas.cublasCsscal(n, alpha, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasCsscal_v2.restype = int
    _libcublas.cublasCsscal_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasCsscal(n, alpha, x, incx):
        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasCsscal_v2(handle, n,
                                           ctypes.byref(ctypes.c_float(alpha)),
                                           int(x), incx)
        cublasCheckStatus(status)
        
cublasCsscal.__doc__ = \
                    _SCAL_doc.substitute(precision='single-precision',
                                         real='complex',
                                         a_real='real',
                                         a_type='numpy.float32',
                                         alpha='np.float32(np.random.rand())',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                         func='cublasCsscal')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZscal.restype = None
    _libcublas.cublasZscal.argtypes = [ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasZscal(n, alpha, x, incx):
        _libcublas.cublasZscal(n, cuda.cuDoubleComplex(alpha.real, alpha.imag), int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasZscal_v2.restype = int
    _libcublas.cublasZscal_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasZscal(n, alpha, x, incx):
        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasZscal_v2(handle, n,
                                           ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                             alpha.imag)),
                                           int(x), incx)
        cublasCheckStatus(status)
        
cublasZscal.__doc__ = \
                    _SCAL_doc.substitute(precision='double-precision',
                                         real='complex',
                                         a_real='complex',
                                         a_type='numpy.complex128',
                                         alpha='np.complex128(np.random.rand()+1j*np.random.rand())',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                         func='cublasZscal')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZdscal.restype = None
    _libcublas.cublasZdscal.argtypes = [ctypes.c_int,
                                        ctypes.c_double,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
    def cublasZdscal(n, alpha, x, incx):
        _libcublas.cublasZdscal(n, alpha, int(x), incx)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasZdscal_v2.restype = int
    _libcublas.cublasZdscal_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasZdscal(n, alpha, x, incx):
        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasZdscal_v2(handle, n,
                                           ctypes.byref(ctypes.c_double(alpha)),
                                           int(x), incx)
        cublasCheckStatus(status)
        
cublasZdscal.__doc__ = \
                    _SCAL_doc.substitute(precision='double-precision',
                                         real='complex',
                                         a_real='real',
                                         a_type='numpy.float64',
                                         alpha='np.float64(np.random.rand())',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                         func='cublasZdscal')

# SSWAP, DSWAP, CSWAP, ZSWAP
_SWAP_doc = Template(
"""
    Swap ${precision} ${real} vectors.

    Swaps the contents of one ${precision} ${real} vector with those
    of another ${precision} ${real} vector.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input/output vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to ${precision} ${real} input/output vector.
    incy : int
        Storage spacing between elements of `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> y = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> ${func}(x.size, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> np.allclose(x_gpu.get(), y)
    True
    >>> np.allclose(y_gpu.get(), x)
    True

    Notes
    -----
    Both `x` and `y` must contain `n` elements.

""")

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasSswap.restype = None
    _libcublas.cublasSswap.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasSswap(n, x, incx, y, incy):
        _libcublas.cublasSswap(n, int(x), incx, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasSswap_v2.restype = int
    _libcublas.cublasSswap_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasSswap(n, x, incx, y, incy):
        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasSswap_v2(handle,
                                           n, int(x), incx, int(y), incy)
        cublasCheckStatus(status)

cublasSswap.__doc__ = \
                    _SWAP_doc.substitute(precision='single-precision',
                                         real='real',
                                         data='np.random.rand(5).astype(np.float32)',
                                         func='cublasSswap')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDswap.restype = None
    _libcublas.cublasDswap.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasDswap(n, x, incx, y, incy):
        _libcublas.cublasDswap(n, int(x), incx, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasDswap_v2.restype = int
    _libcublas.cublasDswap_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int]    
    def cublasDswap(n, x, incx, y, incy):
        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasDswap_v2(handle,
                                           n, int(x), incx, int(y), incy)
        cublasCheckStatus(status)

cublasDswap.__doc__ = \
                    _SWAP_doc.substitute(precision='double-precision',
                                         real='real',
                                         data='np.random.rand(5).astype(np.float64)',
                                         func='cublasDswap')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasCswap.restype = None
    _libcublas.cublasCswap.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasCswap(n, x, incx, y, incy):
        _libcublas.cublasCswap(n, int(x), incx, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasCswap_v2.restype = int
    _libcublas.cublasCswap_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasCswap(n, x, incx, y, incy):
        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasCswap_v2(handle,
                                           n, int(x), incx, int(y), incy)
        cublasCheckStatus(status)

cublasCswap.__doc__ = \
                    _SWAP_doc.substitute(precision='single-precision',
                                         real='complex',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                         func='cublasCswap')

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZswap.restype = None
    _libcublas.cublasZswap.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasZswap(n, x, incx, y, incy):
        _libcublas.cublasZswap(n, int(x), incx, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasZswap_v2.restype = int
    _libcublas.cublasZswap_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasZswap(n, x, incx, y, incy):
        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasZswap_v2(handle,
                                           n, int(x), incx, int(y), incy)
        cublasCheckStatus(status)

cublasZswap.__doc__ = \
                    _SWAP_doc.substitute(precision='double-precision',
                                         real='complex',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                         func='cublasZswap')

### BLAS Level 2 Functions ###

# SGBMV, DGVMV, CGBMV, ZGBMV 
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasSgbmv.restype = None
    _libcublas.cublasSgbmv.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
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
    def cublasSgbmv(trans, m, n, kl, ku, alpha, A, lda,
                    x, incx, beta, y, incy):
        """
        Matrix-vector product for real general banded matrix.
        
        """
        _libcublas.cublasSgbmv(trans, m, n, kl, ku, alpha, int(A), lda,
                               int(x), incx, beta, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)

else:
    _libcublas.cublasSgbmv_v2.restype = int
    _libcublas.cublasSgbmv_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_char,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasSgbmv(trans, m, n, kl, ku, alpha, A, lda,
                    x, incx, beta, y, incy):
        """
        Matrix-vector product for real general banded matrix.
        
        """
        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasSgbmv_v2(handle,
                                           trans, m, n, kl, ku,
                                           ctypes.byref(ctypes.c_float(alpha)),
                                           int(A), lda,
                                           int(x), incx,
                                           ctypes.byref(ctypes.c_float(beta)),
                                           int(y), incy)
        cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasDgbmv.restype = None
    _libcublas.cublasDgbmv.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasDgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy):
        """
        Matrix-vector product for real general banded matrix.
        
        """

        _libcublas.cublasDgbmv(trans, m, n, kl, ku, alpha,
                               int(A), lda, int(x), incx, beta, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasDgbmv_v2.restype = int
    _libcublas.cublasDgbmv_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_char,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasDgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy):
        """
        Matrix-vector product for real general banded matrix.
        
        """

        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasDgbmv_v2(handle,
                                           trans, m, n, kl, ku,
                                           ctypes.byref(ctypes.c_float(alpha)),
                                           int(A), lda, int(x), incx,
                                           ctypes.byref(ctypes.c_float(beta)),
                                           int(y), incy)
        cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasCgbmv.restype = None
    _libcublas.cublasCgbmv.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasCgbmv(trans, m, n, kl, ku, alpha, A, lda,
                    x, incx, beta, y, incy):
        """
        Matrix-vector product for complex general banded matrix.
        
        """
    
        _libcublas.cublasCgbmv(trans, m, n, kl, ku,
                               cuda.cuFloatComplex(alpha.real,
                                                   alpha.imag),
                               int(A), lda, int(x), incx,
                               cuda.cuFloatComplex(beta.real,
                                                   beta.imag),
                               int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasCgbmv_v2.restype = int
    _libcublas.cublasCgbmv_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_char,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasCgbmv(trans, m, n, kl, ku, alpha, A, lda,
                    x, incx, beta, y, incy):
        """
        Matrix-vector product for complex general banded matrix.
        
        """

        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasCgbmv_v2(handle,
                                           trans, m, n, kl, ku,
                                           ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                            alpha.imag)),
                                           int(A), lda, int(x), incx,
                                           ctypes.byref(cuda.cuFloatComplex(beta.real,
                                                                            beta.imag)),
                                           int(y), incy)
        cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZgbmv.restype = None
    _libcublas.cublasZgbmv.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasZgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy):
        """
        Matrix-vector product for complex general banded matrix.
        
        """

        _libcublas.cublasZgbmv(trans, m, n, kl, ku,
                               cuda.cuDoubleComplex(alpha.real,
                                                    alpha.imag),
                               int(A), lda, int(x), incx,
                               cuda.cuDoubleComplex(beta.real,
                                                beta.imag),
                               int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasZgbmv_v2.restype = int
    _libcublas.cublasZgbmv_v2.argtypes = [ctypes.c_char,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasZgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy):
        """
        Matrix-vector product for complex general banded matrix.
        
        """

        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasZgbmv_v2(handle,
                                           trans, m, n, kl, ku,
                                           ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                             alpha.imag)),
                                           int(A), lda, int(x), incx,
                                           ctypes.byref(cuda.cuDoubleComplex(beta.real,
                                                                             beta.imag)),
                                  int(y), incy)
        cublasCheckStatus(status)
    
# SGEMV, DGEMV, CGEMV, ZGEMV
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasSgemv.restype = None
    _libcublas.cublasSgemv.argtypes = [ctypes.c_char,
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
    def cublasSgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
        """
        Matrix-vector product for real general matrix.
        
        Computes the product `alpha*op(A)*x+beta*y`, where `op(A)` == `A`
        or `op(A)` == `A.T`, and stores it in `y`.
        
        Parameters
        ----------
        trans : char
            If `upper(trans)` in `['T', 'C']`, assume that `A` is
            transposed.
        m : int
            Number of rows in `A`.
        n : int
            Number of columns in `A`.
        alpha : numpy.float32
            `A` is multiplied by this quantity. 
        A : ctypes.c_void_p
            Pointer to single-precision matrix. The matrix has
            shape `(lda, n)` if `upper(trans)` == 'N', `(lda, m)`
            otherwise.
        lda : int
            Leading dimension of `A`.
        X : ctypes.c_void_p
            Pointer to single-precision array of length at least
            `(1+(n-1)*abs(incx))` if `upper(trans) == 'N',
            `(1+(m+1)*abs(incx))` otherwise.
        incx : int
            Spacing between elements of `x`. Must be nonzero.
        beta : numpy.float32
            `y` is multiplied by this quantity. If zero, `y` is ignored.
        y : ctypes.c_void_p
            Pointer to single-precision array of length at least
            `(1+(m+1)*abs(incy))` if `upper(trans)` == `N`,
            `(1+(n+1)*abs(incy))` otherwise.
        incy : int
            Spacing between elements of `y`. Must be nonzero.

        Examples
        --------
        >>> import pycuda.autoinit
        >>> import pycuda.gpuarray as gpuarray
        >>> import numpy as np
        >>> a = np.random.rand(2, 3).astype(np.float32)
        >>> x = np.random.rand(3, 1).astype(np.float32)
        >>> a_gpu = gpuarray.to_gpu(a.T.copy())
        >>> x_gpu = gpuarray.to_gpu(x)
        >>> y_gpu = gpuarray.empty((2, 1), np.float32)
        >>> alpha = np.float32(1.0)
        >>> beta = np.float32(0)
        >>> cublasSgemv('n', 2, 3, alpha, a_gpu.gpudata, 2, x_gpu.gpudata, 1, beta, y_gpu.gpudata, 1)
        >>> np.allclose(y_gpu.get(), np.dot(a, x))
        True
    
        """
    
        _libcublas.cublasSgemv(trans, m, n, alpha, int(A), lda,
                               int(x), incx, beta, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasSgemv_v2.restype = int
    _libcublas.cublasSgemv_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasSgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
        """
        Matrix-vector product for real general matrix.

        """

        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasSgemv_v2(handle,
                                           _CUBLAS_OP[trans], m, n,
                                           ctypes.byref(ctypes.c_float(alpha)), int(A), lda,
                                           int(x), incx,
                                           ctypes.byref(ctypes.c_float(beta)), int(y), incy) 
        cublasCheckStatus(status)
    
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasDgemv.restype = None
    _libcublas.cublasDgemv.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasDgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
        """
        Matrix-vector product for real general matrix.
        
        """
        
        _libcublas.cublasDgemv(trans, m, n, alpha,
                               int(A), lda, int(x), incx, beta,
                               int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasDgemv_v2.restype = int
    _libcublas.cublasDgemv_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasDgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
        """
        Matrix-vector product for real general matrix.
        
        """

        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasDgemv_v2(handle,
                                           _CUBLAS_OP[trans], m, n,
                                           ctypes.byref(ctypes.c_double(alpha)),
                                           int(A), lda, int(x), incx,
                                           ctypes.byref(ctypes.c_double(beta)),
                                           int(y), incy)
        cublasCheckStatus(status)
    
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasCgemv.restype = None
    _libcublas.cublasCgemv.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasCgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
        """
        Matrix-vector product for complex general matrix.
        
        """
        
        _libcublas.cublasCgemv(trans, m, n,
                               cuda.cuFloatComplex(alpha.real,
                                                   alpha.imag),
                               int(A), lda, int(x), incx,
                               cuda.cuFloatComplex(beta.real,
                                                   beta.imag),
                               int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasCgemv_v2.restype = int
    _libcublas.cublasCgemv_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasCgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
        """
        Matrix-vector product for complex general matrix.
        
        """

        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasCgemv_v2(handle,
                                           _CUBLAS_OP[trans], m, n,
                                           ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                            alpha.imag)),
                                           int(A), lda, int(x), incx,
                                           ctypes.byref(cuda.cuFloatComplex(beta.real,
                                                                            beta.imag)),
                                           int(y), incy)
        cublasCheckStatus(status)
    
if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZgemv.restype = None
    _libcublas.cublasZgemv.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasZgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
        """
        Matrix-vector product for complex general matrix.
        
        """
        
        _libcublas.cublasZgemv(trans, m, n,
                               cuda.cuDoubleComplex(alpha.real,
                                                    alpha.imag),
                               int(A), lda, int(x), incx,
                               cuda.cuDoubleComplex(beta.real,
                                                    beta.imag),
                               int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasZgemv_v2.restype = int
    _libcublas.cublasZgemv_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,        
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasZgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
        """
        Matrix-vector product for complex general matrix.
        
        """

        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasZgemv_v2(handle,
                                           _CUBLAS_OP[trans], m, n,
                                           ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                             alpha.imag)),
                                           int(A), lda, int(x), incx,
                                           ctypes.byref(cuda.cuDoubleComplex(beta.real,
                                                                             beta.imag)),
                                           int(y), incy)
        cublasCheckStatus(status)

# SGER, DGER, CGERU, CGERC, ZGERU, ZGERC
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasSger.restype = None
    _libcublas.cublasSger.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_float,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
    def cublasSger(m, n, alpha, x, incx, y, incy, A, lda):
        """
        Rank-1 operation on real general matrix.
        
        """
        
        _libcublas.cublasSger(m, n, alpha, int(x), incx,
                              int(y), incy, int(A), lda)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasSger_v2.restype = int
    _libcublas.cublasSger_v2.argtypes = [ctypes.c_int,
                                         ctypes.c_int,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int]
    def cublasSger(m, n, alpha, x, incx, y, incy, A, lda):
        """
        Rank-1 operation on real general matrix.
        
        """
        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasSger_v2(handle,
                                          m, n,
                                          ctypes.byref(ctypes.c_float(alpha)),
                                          int(x), incx,
                                          int(y), incy, int(A), lda)
        cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasDger.restype = None
    _libcublas.cublasDger.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_double,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
    def cublasDger(m, n, alpha, x, incx, y, incy, A, lda):
        """
        Rank-1 operation on real general matrix.
        
        """
        
        _libcublas.cublasDger(m, n, alpha, int(x), incx,
                              int(y), incy, int(A), lda)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasDger_v2.restype = int
    _libcublas.cublasDger_v2.argtypes = [ctypes.c_int,
                                         ctypes.c_int,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_void_p,
                                         ctypes.c_int]
    def cublasDger(m, n, alpha, x, incx, y, incy, A, lda):
        """
        Rank-1 operation on real general matrix.
        
        """
        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasDger_v2(handle,
                                          m, n,
                                          ctypes.byref(ctypes.c_double(alpha)),
                                          int(x), incx,
                                          int(y), incy, int(A), lda)
        cublasCheckStatus(status)
    
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasCgerc.restype = None
    _libcublas.cublasCgerc.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasCgerc(m, n, alpha, x, incx, y, incy, A, lda):
        """
        Rank-1 operation on complex general matrix.
        
        """
        
        _libcublas.cublasCgerc(m, n, cuda.cuFloatComplex(alpha.real,
                                                         alpha.imag),
                               int(x), incx, int(y), incy, int(A), lda)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasCgerc_v2.restype = int
    _libcublas.cublasCgerc_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasCgerc(m, n, alpha, x, incx, y, incy, A, lda):
        """
        Rank-1 operation on complex general matrix.
        
        """
        
        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasCgerc_v2(handle,
                                           m, n, ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                                alpha.imag)),
                                           int(x), incx, int(y), incy, int(A), lda)
        cublasCheckStatus(status)
    
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasCgeru.restype = None
    _libcublas.cublasCgeru.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasCgeru(m, n, alpha, x, incx, y, incy, A, lda):
        """
        Rank-1 operation on complex general matrix.
        
        """
        
        _libcublas.cublasCgeru(m, n, cuda.cuFloatComplex(alpha.real,
                                                         alpha.imag),
                               int(x), incx, int(y), incy, int(A), lda)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasCgeru_v2.restype = int
    _libcublas.cublasCgeru_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasCgeru(m, n, alpha, x, incx, y, incy, A, lda):
        """
        Rank-1 operation on complex general matrix.
        
        """

        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasCgeru_v2(handle,
                                           m, n, ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                                  alpha.imag)),
                                           int(x), incx, int(y), incy, int(A), lda)
        cublasCheckStatus(status)
    
if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZgerc.restype = None
    _libcublas.cublasZgerc.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasZgerc(m, n, alpha, x, incx, y, incy, A, lda):
        """
        Rank-1 operation on complex general matrix.
        
        """
        
        _libcublas.cublasZgerc(m, n, cuda.cuDoubleComplex(alpha.real,
                                                          alpha.imag),
                               int(x), incx, int(y), incy, int(A), lda)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasZgerc_v2.restype = None
    _libcublas.cublasZgerc_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasZgerc(m, n, alpha, x, incx, y, incy, A, lda):
        """
        Rank-1 operation on complex general matrix.
        
        """

        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasZgerc_v2(handle,
                                           m, n, ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                                   alpha.imag)),
                                           int(x), incx, int(y), incy, int(A), lda)
        cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZgeru.restype = None
    _libcublas.cublasZgeru.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasZgeru(m, n, alpha, x, incx, y, incy, A, lda):
        """
        Rank-1 operation on complex general matrix.
        
        """
        
        _libcublas.cublasZgeru(m, n, cuda.cuDoubleComplex(alpha.real,
                                                          alpha.imag),
                               int(x), incx, int(y), incy, int(A), lda)

        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasZgeru_v2.restype = int
    _libcublas.cublasZgeru_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasZgeru(m, n, alpha, x, incx, y, incy, A, lda):
        """
        Rank-1 operation on complex general matrix.
        
        """

        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasZgeru_v2(handle,
                                           m, n, ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                                   alpha.imag)),
                                           int(x), incx, int(y), incy, int(A), lda)
        cublasCheckStatus(status)

# SSBMV, DSBMV 
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasSsbmv.restype = None
    _libcublas.cublasSsbmv.argtypes = [ctypes.c_char,
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
    def cublasSsbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
        """
        Matrix-vector product for real symmetric-banded matrix.
        
        """
    
        _libcublas.cublasSsbmv(uplo, n, k, alpha, int(A), lda,
                               int(x), incx, beta, int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasSsbmv_v2.restype = int
    _libcublas.cublasSsbmv_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]

    def cublasSsbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
        """
        Matrix-vector product for real symmetric-banded matrix.
        
        """

        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasSsbmv_v2(handle,
                                           _CUBLAS_FILL_MODE[uplo], n, k,
                                           ctypes.byref(ctypes.c_float(alpha)),
                                           int(A), lda, int(x), incx,
                                           ctypes.byref(ctypes.c_float(beta)),
                                           int(y), incy)
        cublasCheckStatus(status)
        
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasDsbmv.restype = None
    _libcublas.cublasDsbmv.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasDsbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
        """
        Matrix-vector product for real symmetric-banded matrix.
        
        """

        _libcublas.cublasDsbmv(uplo, n, k, alpha,
                               int(A), lda, int(x), incx, beta,
                               int(y), incy)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasDsbmv_v2.restype = int
    _libcublas.cublasDsbmv_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasDsbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
        """
        Matrix-vector product for real symmetric-banded matrix.
        
        """

        handle = cublasGetCurrentCtx()
        status = _libcublas.cublasDsbmv_v2(handle,
                                           _CUBLAS_FILL_MODE[uplo], n, k,
                                           ctypes.byref(ctypes.c_double(alpha)),
                                           int(A), lda, int(x), incx,
                                           ctypes.byref(ctypes.c_double(beta)),
                                           int(y), incy)
        cublasCheckStatus(status)
        
# SSPMV, DSPMV (need to make CUDA 4.0 compatible)
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasSspmv.restype = None
    _libcublas.cublasSspmv.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_float,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_float,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasSspmv(uplo, n, alpha, AP, x, incx, beta, y, incy):
    """
    Matrix-vector product for real symmetric-packed matrix.

    """
    
    _libcublas.cublasSspmv(uplo, n, alpha, int(AP),
                           int(x), incx, beta, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDspmv.restype = None
    _libcublas.cublasDspmv.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasDspmv(uplo, n, alpha, AP, x, incx, beta, y, incy):
    """
    Matrix-vector product for real symmetric-packed matrix.

    """

    _libcublas.cublasDspmv(uplo, n, alpha, int(AP),
                           int(x), incx, beta, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)

# SSPR, DSPR (need to make CUDA 4.0 compatible)
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasSspr.restype = None
    _libcublas.cublasSspr.argtypes = [ctypes.c_char,
                                      ctypes.c_int,
                                      ctypes.c_float,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cublasSspr(uplo, n, alpha, x, incx, AP):
    """
    Rank-1 operation on real symmetric-packed matrix.

    """
    
    _libcublas.cublasSspr(uplo, n, alpha, int(x), incx, int(AP))
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDspr.restype = None
    _libcublas.cublasDspr.argtypes = [ctypes.c_char,
                                      ctypes.c_int,
                                      ctypes.c_double,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cublasDspr(uplo, n, alpha, x, incx, AP):
    """
    Rank-1 operation on real symmetric-packed matrix.

    """

    _libcublas.cublasDspr(uplo, n, alpha, int(x), incx, int(AP))
    status = cublasGetError()
    cublasCheckStatus(status)

# SSPR2, DSPR2 (need to make CUDA 4.0 compatible)
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasSspr2.restype = None
    _libcublas.cublasSspr2.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_float,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def cublasSspr2(uplo, n, alpha, x, incx, y, incy, AP):
    """
    Rank-2 operation on real symmetric-packed matrix.

    """

    _libcublas.cublasSspr2(uplo, n, alpha,
                           int(x), incx, int(y), incy, int(AP))
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDspr2.restype = None
    _libcublas.cublasDspr2.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def cublasDspr2(uplo, n, alpha, x, incx, y, incy, AP):
    """
    Rank-2 operation on real symmetric-packed matrix.

    """

    _libcublas.cublasDspr2(uplo, n, alpha, int(x), incx, int(y), incy, int(AP))
    status = cublasGetError()
    cublasCheckStatus(status)

# SSYMV, DSYMV (need to make CUDA 4.0 compatible)
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasSsymv.restype = None
    _libcublas.cublasSsymv.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_float,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_float,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasSsymv(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for real symmetric matrix.
    
    """
    
    _libcublas.cublasSsymv(uplo, n, alpha,
                           int(A), lda, int(x), incx,
                           beta, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDsymv.restype = None
    _libcublas.cublasDsymv.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasDsymv(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for real symmetric matrix.
    
    """

    _libcublas.cublasDsymv(uplo, n, alpha, int(A), lda,
                           int(x), incx, beta, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)

# SSYR, DSYR (need to make CUDA 4.0 compatible)
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasSsyr.restype = None
    _libcublas.cublasSsyr.argtypes = [ctypes.c_char,
                                      ctypes.c_int,
                                      ctypes.c_float,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasSsyr(uplo, n, alpha, x, incx, A, lda): 
    """
    Rank-1 operation on real symmetric matrix.

    """
   
    _libcublas.cublasSsyr(uplo, n, alpha,
                          int(x), incx, int(A), lda)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDsyr.restype = None
    _libcublas.cublasDsyr.argtypes = [ctypes.c_char,
                                      ctypes.c_int,
                                      ctypes.c_double,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDsyr(uplo, n, alpha, x, incx, A, lda):
    """
    Rank-1 operation on real symmetric matrix.

    """

    _libcublas.cublasDsyr(uplo, n, alpha, int(x), incx, int(A), lda)
    status = cublasGetError()
    cublasCheckStatus(status)

# SSYR2, DSYR2 (need to make CUDA 4.0 compatible)
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasSsyr2.restype = None
    _libcublas.cublasSsyr2.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_float,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasSsyr2(uplo, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-2 operation on real symmetric matrix.

    """

    _libcublas.cublasSsyr2(uplo, n, alpha,
                           int(x), incx, int(y), incy,
                           int(A), lda)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDsyr2.restype = None
    _libcublas.cublasDsyr2.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasDsyr2(uplo, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-2 operation on real symmetric matrix.

    """

    _libcublas.cublasDsyr2(uplo, n, alpha, int(x), incx,
                           int(y), incy, int(A),lda)
    status = cublasGetError()
    cublasCheckStatus(status)

# STBMV, DTBMV, CTBMV, ZTBMV (need to make CUDA 4.0 compatible)
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasStbmv.restype = None
    _libcublas.cublasStbmv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasStbmv(uplo, trans, diag, n, k, A, lda, x, incx):
    """
    Matrix-vector product for real triangular-banded matrix.

    """
    
    _libcublas.cublasStbmv(uplo, trans, diag, n, k,
                           int(A), lda, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDtbmv.restype = None
    _libcublas.cublasDtbmv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasDtbmv(uplo, trans, diag, n, k, A, lda, x, incx):
    """
    Matrix-vector product for real triangular-banded matrix.

    """

    _libcublas.cublasDtbmv(uplo, trans, diag, n, k, int(A), lda,
                           int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasCtbmv.restype = None
    _libcublas.cublasCtbmv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasCtbmv(uplo, trans, diag, n, k, A, lda, x, incx):
    """
    Matrix-vector product for complex triangular-banded matrix.

    """
    
    _libcublas.cublasCtbmv(uplo, trans, diag, n, k,
                           int(A), lda, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZtbmv.restype = None
    _libcublas.cublasZtbmv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasZtbmv(uplo, trans, diag, n, k, A, lda, x, incx):
    """
    Matrix-vector product for complex triangular-banded matrix.

    """
    
    _libcublas.cublasZtbmv(uplo, trans, diag, n, k, int(A), lda, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

# STBSV, DTBSV, CTBSV, ZTBSV (need to make CUDA 4.0 compatible)
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasStbsv.restype = None
    _libcublas.cublasStbsv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasStbsv(uplo, trans, diag, n, k, A, lda, x, incx):
    """
    Solve real triangular-banded system with one right-hand side.

    """
    
    _libcublas.cublasStbsv(uplo, trans, diag, n, k,
                           int(A), lda, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDtbsv.restype = None
    _libcublas.cublasDtbsv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasDtbsv(uplo, trans, diag, n, k, A, lda, x, incx):
    """
    Solve real triangular-banded system with one right-hand side.

    """

    _libcublas.cublasDtbsv(uplo, trans, diag, n, k,
                           int(A), lda, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasCtbsv.restype = None
    _libcublas.cublasCtbsv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasCtbsv(uplo, trans, diag, n, k, A, lda, x, incx):
    """
    Solve complex triangular-banded system with one right-hand side.

    """
    
    _libcublas.cublasCtbsv(uplo, trans, diag, n, k,
                           int(A), lda, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZtbsv.restype = None
    _libcublas.cublasZtbsv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasZtbsv(uplo, trans, diag, n, k, A, lda, x, incx):
    """
    Solve complex triangular-banded system with one right-hand side.

    """
    
    _libcublas.cublasZtbsv(uplo, trans, diag, n, k, int(A), lda, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

# STPMV, DTPMV, CTPMV, ZTPMV (need to make CUDA 4.0 compatible)
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasStpmv.restype = None
    _libcublas.cublasStpmv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasStpmv(uplo, trans, diag, n, AP, x, incx):
    """
    Matrix-vector product for real triangular-packed matrix.

    """
    
    _libcublas.cublasStpmv(uplo, trans, diag, n,
                           int(AP), int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasCtpmv.restype = None
    _libcublas.cublasCtpmv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasCtpmv(uplo, trans, diag, n, AP, x, incx):
    """
    Matrix-vector product for complex triangular-packed matrix.

    """
    
    _libcublas.cublasCtpmv(uplo, trans, diag, n, int(AP), int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDtpmv.restype = None
    _libcublas.cublasDtpmv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasDtpmv(uplo, trans, diag, n, AP, x, incx):
    """
    Matrix-vector product for real triangular-packed matrix.

    """

    _libcublas.cublasDtpmv(uplo, trans, diag, n, int(AP), int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZtpmv.restype = None
    _libcublas.cublasZtpmv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasZtpmv(uplo, trans, diag, n, AP, x, incx):
    """
    Matrix-vector product for complex triangular-packed matrix.

    """
    
    _libcublas.cublasZtpmv(uplo, trans, diag, n, int(AP), int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

# STPSV, DTPSV, CTPSV, ZTPSV (need to convert to CUDA 4.0)
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasStpsv.restype = None
    _libcublas.cublasStpsv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasStpsv(uplo, trans, diag, n, AP, x, incx):
    """
    Solve real triangular-packed system with one right-hand side.

    """
    
    _libcublas.cublasStpsv(uplo, trans, diag, n,
                           int(AP), int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDtpsv.restype = None
    _libcublas.cublasDtpsv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasDtpsv(uplo, trans, diag, n, AP, x, incx):
    """
    Solve real triangular-packed system with one right-hand side.

    """

    _libcublas.cublasDtpsv(uplo, trans, diag, n, int(AP), int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasCtpsv.restype = None
    _libcublas.cublasCtpsv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasCtpsv(uplo, trans, diag, n, AP, x, incx):
    """
    Solve complex triangular-packed system with one right-hand side.
    
    """
    
    _libcublas.cublasCtpsv(uplo, trans, diag, n, int(AP), int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZtpsv.restype = None
    _libcublas.cublasZtpsv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasZtpsv(uplo, trans, diag, n, AP, x, incx):
    """
    Solve complex triangular-packed system with one right-hand size.

    """
    
    _libcublas.cublasZtpsv(uplo, trans, diag, n, int(AP), int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

# STRMV, DTRMV, CTRMV, ZTRMV (need to convert to CUDA 4.0) 
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasStrmv.restype = None
    _libcublas.cublasStrmv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasStrmv(uplo, trans, diag, n, A, lda, x, inx):
    """
    Matrix-vector product for real triangular matrix.

    """
    
    _libcublas.cublasStrmv(uplo, trans, diag, n,
                           int(A), lda, int(x), inx)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasCtrmv.restype = None
    _libcublas.cublasCtrmv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasCtrmv(uplo, trans, diag, n, A, lda, x, incx):
    """
    Matrix-vector product for complex triangular matrix.

    """
    
    _libcublas.cublasCtrmv(uplo, trans, diag, n, int(A), lda, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDtrmv.restype = None
    _libcublas.cublasDtrmv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasDtrmv(uplo, trans, diag, n, A, lda, x, inx):
    """
    Matrix-vector product for real triangular matrix.

    """

    _libcublas.cublasDtrmv(uplo, trans, diag, n, int(A), lda, int(x), inx)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZtrmv.restype = None
    _libcublas.cublasZtrmv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasZtrmv(uplo, trans, diag, n, A, lda, x, incx):
    """
    Matrix-vector product for complex triangular matrix.

    """
    
    _libcublas.cublasZtrmv(uplo, trans, diag, n, int(A), lda, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

# STRSV, DTRSV, CTRSV, ZTRSV (need to convert to CUDA 4.0)
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasStrsv.restype = None
    _libcublas.cublasStrsv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasStrsv(uplo, trans, diag, n, A, lda, x, incx):
    """
    Solve real triangular system with one right-hand side.

    """
    
    _libcublas.cublasStrsv(uplo, trans, diag, n,
                           int(A), lda, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDtrsv.restype = None
    _libcublas.cublasDtrsv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasDtrsv(uplo, trans, diag, n, A, lda, x, incx):
    """
    Solve real triangular system with one right-hand side.

    """

    _libcublas.cublasDtrsv(uplo, trans, diag, n, int(A), lda, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasCtrsv.restype = None
    _libcublas.cublasCtrsv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasCtrsv(uplo, trans, diag, n, A, lda, x, incx):
    """
    Solve complex triangular system with one right-hand side.

    """
    
    _libcublas.cublasCtrsv(uplo, trans, diag, n, int(A), lda, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZtrsv.restype = None
    _libcublas.cublasZtrsv.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasZtrsv(uplo, trans, diag, n, A, lda, x, incx):
    """
    Solve complex triangular system with one right-hand side.

    """
    
    _libcublas.cublasZtrsv(uplo, trans, diag, n, int(A), lda, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

# CHEMV, ZHEMV (need to convert to CUDA 4.0)
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasChemv.restype = None
    _libcublas.cublasChemv.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasChemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix vector product for Hermitian matrix.
    
    """
    
    _libcublas.cublasChemv(uplo, n, cuda.cuFloatComplex(alpha.real,
                                                        alpha.imag),
                           int(A), lda, int(x), incx,
                           cuda.cuFloatComplex(beta.real,
                                               beta.imag),
                           int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZhemv.restype = None
    _libcublas.cublasZhemv.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasZhemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for Hermitian matrix.

    """
    
    _libcublas.cublasZhemv(uplo, n, cuda.cuDoubleComplex(alpha.real,
                                                         alpha.imag),
                           int(A), lda, int(x), incx,
                           cuda.cuDoubleComplex(beta.real, beta.imag),
                           int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)

# CHBMV, ZHBMV (need to convert to CUDA 4.0) 
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasChbmv.restype = None
    _libcublas.cublasChbmv.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasChbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for Hermitian-banded matrix.

    """
    
    _libcublas.cublasChbmv(uplo, n, k,
                           cuda.cuFloatComplex(alpha.real,
                                               alpha.imag),
                           int(A), lda, int(x), incx,
                           cuda.cuFloatComplex(beta.real,
                                               beta.imag),
                           int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZhbmv.restype = None
    _libcublas.cublasZhbmv.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasZhbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for Hermitian banded matrix.

    """
    
    _libcublas.cublasZhbmv(uplo, n, k,
                           cuda.cuDoubleComplex(alpha.real,
                                                alpha.imag),
                           int(A), lda, int(x), incx,
                           cuda.cuDoubleComplex(beta.real, beta.imag),
                           int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)

# CHPMV, ZHPMV (need to convert to CUDA 4.0)
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasChpmv.restype = None
    _libcublas.cublasChpmv.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasChpmv(uplo, n, alpha, AP, x, incx, beta, y, incy):
    """
    Matrix-vector product for Hermitian-packed matrix.

    """
    
    _libcublas.cublasChpmv(uplo, n, cuda.cuFloatComplex(alpha.real,
                                                        alpha.imag),
                           int(AP), int(x), incx,
                           cuda.cuFloatComplex(beta.real,
                                               beta.imag),
                           int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZhpmv.restype = None
    _libcublas.cublasZhpmv.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasZhpmv(uplo, n, alpha, AP, x, incx, beta, y, incy):
    """
    Matrix-vector product for Hermitian-packed matrix.

    """
    
    _libcublas.cublasZhpmv(uplo, n, cuda.cuDoubleComplex(alpha.real,
                                                         alpha.imag),
                           int(AP), int(x), incx,
                           cuda.cuDoubleComplex(beta.real, beta.imag),
                           int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)

# CHER, ZHER (need to convert to CUDA 4.0) 
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasCher.restype = None
    _libcublas.cublasCher.argtypes = [ctypes.c_char,
                                      ctypes.c_int,
                                      ctypes.c_float,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCher(uplo, n, alpha, x, incx, A, lda):
    """
    Rank-1 operation on Hermitian matrix.

    """

    _libcublas.cublasCher(uplo, n, alpha, int(x), incx, int(A), lda)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZher.restype = None
    _libcublas.cublasZher.argtypes = [ctypes.c_char,
                                      ctypes.c_int,
                                      ctypes.c_double,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZher(uplo, n, alpha, x, incx, A, lda):
    """
    Rank-1 operation on Hermitian matrix.

    """
    
    _libcublas.cublasZher(uplo, n, alpha, int(x), incx, int(A), lda)
    status = cublasGetError()
    cublasCheckStatus(status)


# CHER2, ZHER2 (need to convert to CUDA 4.0)
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasCher2.restype = None
    _libcublas.cublasCher2.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasCher2(uplo, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-2 operation on Hermitian matrix.


    """
    
    _libcublas.cublasCher2(uplo, n, cuda.cuFloatComplex(alpha.real,
                                                        alpha.imag),
                           int(x), incx, int(y), incy,
                           int(A), lda)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZher2.restype = None
    _libcublas.cublasZher2.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasZher2(uplo, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-2 operation on Hermitian matrix.

    """
    
    _libcublas.cublasZher2(uplo, n, cuda.cuDoubleComplex(alpha.real,
                                                         alpha.imag),
                           int(x), incx, int(y), incy, int(A), lda)
    status = cublasGetError()
    cublasCheckStatus(status)


# CHPR, ZHPR (need to convert to CUDA 4.0) 
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasChpr.restype = None
    _libcublas.cublasChpr.argtypes = [ctypes.c_char,
                                      ctypes.c_int,
                                      ctypes.c_float,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cublasChpr(uplo, n, alpha, x, incx, AP):
    """
    Rank-1 operation on Hermitian-packed matrix.
    
    """
    
    _libcublas.cublasChpr(uplo, n, alpha, int(x), incx, int(AP))
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZhpr.restype = None
    _libcublas.cublasZhpr.argtypes = [ctypes.c_char,
                                      ctypes.c_int,
                                      ctypes.c_double,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cublasZhpr(uplo, n, alpha, x, incx, AP):
    """
    Rank-1 operation on Hermitian-packed matrix.

    """
    
    _libcublas.cublasZhpr(uplo, n, alpha, int(x), incx, int(AP))
    status = cublasGetError()
    cublasCheckStatus(status)

# CHPR2, ZHPR2 (need to convert to CUDA 4.0)
if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasChpr2.restype = None
    _libcublas.cublasChpr2.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def cublasChpr2(uplo, n, alpha, x, inx, y, incy, AP):
    """
    Rank-2 operation on Hermitian-packed matrix.
    
    """

    _libcublas.cublasChpr2(uplo, n, cuda.cuFloatComplex(alpha.real,
                                                        alpha.imag),
                           int(x), incx, int(y), incy, int(AP))
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZhpr2.restype = None
    _libcublas.cublasZhpr2.argtypes = [ctypes.c_char,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def cublasZhpr2(uplo, n, alpha, x, inx, y, incy, AP):
    """
    Rank-2 operation on Hermitian-packed matrix.

    """
    
    _libcublas.cublasZhpr2(uplo, n, cuda.cuDoubleComplex(alpha.real,
                                                         alpha.imag),
                           int(x), incx, int(y), incy, int(AP))
    status = cublasGetError()
    cublasCheckStatus(status)

# SGEMM, CGEMM, DGEMM, ZGEMM
if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasSgemm.restype = None
    _libcublas.cublasSgemm.argtypes = [ctypes.c_char,
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
    def cublasSgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
        """
        Matrix-matrix product for general matrix.
        
        """
        
        _libcublas.cublasSgemm(transa, transb, m, n, k, alpha,
                               int(A), lda, int(B), ldb, beta, int(C), ldc)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasSgemm_v2.restype = int
    _libcublas.cublasSgemm_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasSgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
        """
        Matrix-matrix product for real general matrix.
        
        """

        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasSgemm_v2(handle,
                                           _CUBLAS_OP[transa],
                                           _CUBLAS_OP[transb], m, n, k, 
                                           ctypes.byref(ctypes.c_float(alpha)),
                                           int(A), lda, int(B), ldb,
                                           ctypes.byref(ctypes.c_float(beta)),
                                           int(C), ldc)
        cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:    
    _libcublas.cublasCgemm.restype = None
    _libcublas.cublasCgemm.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasCgemm(transa, transb, m, n, k, alpha, x, lda, y, ldb, beta, C, ldc):
        """
        Matrix-matrix product for complex general matrix.
        
        """
        
        _libcublas.cublasCgemm(transa, transb, m, n, k,
                               cuda.cuFloatComplex(alpha.real,
                                                   alpha.imag),
                               int(x), lda, int(y), ldb,
                               cuda.cuFloatComplex(beta.real, beta.imag),
                               int(C), ldc)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasCgemm_v2.restype = int
    _libcublas.cublasCgemm_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasCgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
        """
        Matrix-matrix product for complex general matrix.
        
        """

        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasCgemm_v2(handle,
                                           _CUBLAS_OP[transa],
                                           _CUBLAS_OP[transb], m, n, k, 
                                           ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                            alpha.imag)),
                                           int(A), lda, int(B), ldb,
                                           ctypes.byref(cuda.cuFloatComplex(beta.real,
                                                                            beta.imag)),
                                           int(C), ldc)
        cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDgemm.restype = None
    _libcublas.cublasDgemm.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasDgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
        """
        Matrix-matrix product for real general matrix.
        
        """
        
        _libcublas.cublasDgemm(transa, transb, m, n, k, alpha,
                               int(A), lda, int(B), ldb, beta, int(C), ldc)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasDgemm_v2.restype = int
    _libcublas.cublasDgemm_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasDgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
        """
        Matrix-matrix product for real general matrix.
        
        """

        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasDgemm_v2(handle,
                                           _CUBLAS_OP[transa],
                                           _CUBLAS_OP[transb], m, n, k, 
                                           ctypes.byref(ctypes.c_double(alpha)),
                                           int(A), lda, int(B), ldb,
                                           ctypes.byref(ctypes.c_double(beta)),
                                           int(C), ldc)
        cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZgemm.restype = None
    _libcublas.cublasZgemm.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
    def cublasZgemm(transa, transb, m, n, k,
                    alpha, x, lda, y, ldb, beta, C, ldc):
        """
        Matrix-matrix product for complex general matrix.
        
        """
    
        _libcublas.cublasZgemm(transa, transb, m, n, k,
                               cuda.cuDoubleComplex(alpha.real,
                                                    alpha.imag),
                               int(x), lda, int(y), ldb,
                               cuda.cuDoubleComplex(beta.real,
                                                    beta.imag),
                               int(C), ldc)
        status = cublasGetError()
        cublasCheckStatus(status)
else:
    _libcublas.cublasZgemm_v2.restype = int
    _libcublas.cublasZgemm_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_int,
                                          ctypes.c_void_p,
                                          ctypes.c_void_p,
                                          ctypes.c_int]
    def cublasZgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
        """
        Matrix-matrix product for complex general matrix.
        
        """

        handle = cublasGetCurrentCtx()        
        status = _libcublas.cublasZgemm_v2(handle,
                                           _CUBLAS_OP[transa],
                                           _CUBLAS_OP[transb], m, n, k, 
                                           ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                             alpha.imag)),
                                           int(A), lda, int(B), ldb,
                                           ctypes.byref(cuda.cuDoubleComplex(beta.real,
                                                                             beta.imag)),
                                           int(C), ldc)
        cublasCheckStatus(status)
    
# SSYMM, DSYMM, CSYMM, ZSYMM (need to convert to CUDA 4.0) 
if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasSsymm.restype = None
    _libcublas.cublasSsymm.argtypes = [ctypes.c_char,
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
def cublasSsymm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for symmetric matrix.

    """
    
    _libcublas.cublasSsymm(side, uplo, m, n, alpha,
                           int(A), lda, int(B), ldb, beta, int(C), ldc)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDsymm.restype = None
    _libcublas.cublasDsymm.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int]

def cublasDsymm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for real symmetric matrix.

    """
    
    _libcublas.cublasDsymm(side, uplo, m, n, alpha,
                           int(A), lda, int(B), ldb, beta, int(C), ldc)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasCsymm.restype = None
    _libcublas.cublasCsymm.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasCsymm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for complex symmetric matrix.

    """
    
    _libcublas.cublasCsymm(side, uplo, m, n,
                           cuda.cuFloatComplex(alpha.real,
                                               alpha.imag),
                           int(A), lda, int(B), ldb,
                           cuda.cuFloatComplex(beta.real, beta.imag),
                           int(C), ldc)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZsymm.restype = None
    _libcublas.cublasZsymm.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasZsymm(side, uplo, m, n, alpha,
                A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for complex symmetric matrix.

    """
    
    _libcublas.cublasZsymm(side, uplo, m, n,
                           cuda.cuDoubleComplex(alpha.real,
                                                alpha.imag),
                           int(A), lda, int(B), ldb,
                           cuda.cuDoubleComplex(beta.real,
                                                beta.imag),
                           int(C), ldc)
    status = cublasGetError()
    cublasCheckStatus(status)

# SSYRK, DSYRK, CSYRK, ZSYRK (need to convert to CUDA 4.0)
if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasSsyrk.restype = None
    _libcublas.cublasSsyrk.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_float,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_float,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasSsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """
    Rank-k operation on real symmetric matrix.

    """
    
    _libcublas.cublasSsyrk(uplo, trans, n, k, alpha,
                           int(A), lda, beta, int(C), ldc)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDsyrk.restype = None
    _libcublas.cublasDsyrk.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasDsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """
    Rank-k operation on real symmetric matrix.

    """
    
    _libcublas.cublasDsyrk(uplo, trans, n, k, alpha,
                           int(A), lda, beta, int(C), ldc)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasCsyrk.restype = None
    _libcublas.cublasCsyrk.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasCsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """
    Rank-k operation on complex symmetric matrix.

    """
    
    _libcublas.cublasCsyrk(uplo, trans, n, k,
                           cuda.cuFloatComplex(alpha.real,
                                               alpha.imag),
                           int(A), lda,
                           cuda.cuFloatComplex(beta.real, beta.imag),
                           int(C), ldc)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZsyrk.restype = None
    _libcublas.cublasZsyrk.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasZsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """
    Rank-k operation on complex symmetric matrix.

    """
    
    _libcublas.cublasZsyrk(uplo, trans, n, k,
                           cuda.cuDoubleComplex(alpha.real,
                                                alpha.imag),
                           int(A), lda,
                           cuda.cuDoubleComplex(beta.real,
                                                beta.imag),
                           int(C), ldc)
    status = cublasGetError()
    cublasCheckStatus(status)

# SSYR2K, DSYR2K, CSYR2K, ZSYR2K (need to convert to CUDA 4.0)
if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasSsyr2k.restype = None
    _libcublas.cublasSsyr2k.argtypes = [ctypes.c_char,
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
def cublasSsyr2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Rank-2k operation on real symmetric matrix.

    """
    
    _libcublas.cublasSsyr2k(uplo, trans, n, k, alpha,
                            int(A), lda, int(B), ldb, beta, int(C), ldc)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDsyr2k.restype = None
    _libcublas.cublasDsyr2k.argtypes = [ctypes.c_char,
                                        ctypes.c_char,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_double,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_double,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
def cublasDsyr2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Rank-2k operation on real symmetric matrix.

    """

    _libcublas.cublasDsyr2k(uplo, trans, n, k, alpha,
                            int(A), lda, int(B), ldb, beta, int(C), ldc)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasCsyr2k.restype = None
    _libcublas.cublasCsyr2k.argtypes = [ctypes.c_char,
                                        ctypes.c_char,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        cuda.cuFloatComplex,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        cuda.cuFloatComplex,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
def cublasCsyr2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Rank-2k operation on complex symmetric matrix.

    """

    _libcublas.cublasCsyr2k(uplo, trans, n, k,
                            cuda.cuFloatComplex(alpha.real,
                                                alpha.imag),
                            int(A), lda, int(B), ldb,
                            cuda.cuFloatComplex(beta.real, beta.imag),
                            int(C), ldc)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZsyr2k.restype = None
    _libcublas.cublasZsyr2k.argtypes = [ctypes.c_char,
                                        ctypes.c_char,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        cuda.cuDoubleComplex,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        cuda.cuDoubleComplex,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
def cublasZsyr2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Rank-2k operation on complex symmetric matrix.
    
    """
    
    _libcublas.cublasZsyr2k(uplo, trans, n, k,
                            cuda.cuDoubleComplex(alpha.real,
                                                 alpha.imag),
                            int(A), lda, int(B), ldb,
                            cuda.cuDoubleComplex(beta.real,
                                                 beta.imag),
                            int(C), ldc)
    status = cublasGetError()
    cublasCheckStatus(status)

# STRMM, DTRMM, CTRMM, ZTRMM (need to convert to CUDA 4.0) 
if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasStrmm.restype = None
    _libcublas.cublasStrmm.argtypes = [ctypes.c_char,
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
def cublasStrmm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb):
    """
    Matrix-matrix product for real triangular matrix.

    """
    
    _libcublas.cublasStrmm(side, uplo, transa, diag, m, n, alpha,
                           int(A), lda, int(B), ldb)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDtrmm.restype = None
    _libcublas.cublasDtrmm.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasDtrmm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb):
    """
    Matrix-matrix product for real triangular matrix.

    """
    
    _libcublas.cublasDtrmm(side, uplo, transa, diag, m, n, alpha,
                           int(A), lda, int(B), ldb)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasCtrmm.restype = None
    _libcublas.cublasCtrmm.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasCtrmm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb):
    """
    Matrix-matrix product for complex triangular matrix.

    """
    
    _libcublas.cublasCtrmm(side, uplo, transa, diag, m, n,
                           cuda.cuFloatComplex(alpha.real,
                                               alpha.imag),
                           int(A), lda, int(B), ldb)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZtrmm.restype = None
    _libcublas.cublasZtrmm.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasZtrmm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb):
    """
    Matrix-matrix product for complex triangular matrix.

    """
    
    _libcublas.cublasZtrmm(side, uplo, transa, diag, m, n,
                           cuda.cuDoubleComplex(alpha.real,
                                                alpha.imag),
                           int(A), lda, int(B), ldb)
    status = cublasGetError()
    cublasCheckStatus(status)

# STRSM, DTRSM, CTRSM, ZTRSM (need to convert to CUDA 4.0)
if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasStrsm.restype = None
    _libcublas.cublasStrsm.argtypes = [ctypes.c_char,
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
def cublasStrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb):
    """
    Solve a real triangular system with multiple right-hand sides.

    """
    
    _libcublas.cublasStrsm(side, uplo, transa, diag, m, n, alpha,
                           int(A), lda, int(B), ldb)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasDtrsm.restype = None
    _libcublas.cublasDtrsm.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasDtrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb):
    """
    Solve a real triangular system with multiple right-hand sides.

    """
    
    _libcublas.cublasDtrsm(side, uplo, transa, diag, m, n, alpha,
                           int(A), lda, int(B), ldb)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasCtrsm.restype = None
    _libcublas.cublasCtrsm.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasCtrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb):
    """
    Solve a complex triangular system with multiple right-hand sides.

    """
    
    _libcublas.cublasCtrsm(side, uplo, transa, diag, m, n,
                           cuda.cuFloatComplex(alpha.real,
                                               alpha.imag),
                           int(A), lda, int(B), ldb)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZtrsm.restype = None
    _libcublas.cublasZtrsm.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasZtrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb):
    """
    Solve complex triangular system with multiple right-hand sides.

    """
    
    _libcublas.cublasZtrsm(side, uplo, transa, diag, m, n,
                           cuda.cuDoubleComplex(alpha.real,
                                                alpha.imag),                           
                           int(A), lda, int(B), ldb)
    status = cublasGetError()
    cublasCheckStatus(status)

# CHEMM, ZHEMM (need to convert to CUDA 4.0)
if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasChemm.restype = None
    _libcublas.cublasChemm.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       cuda.cuFloatComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasChemm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for complex Hermitian matrix.

    """
    
    _libcublas.cublasChemm(side, uplo, m, n,
                           cuda.cuFloatComplex(alpha.real,
                                               alpha.imag),
                           int(A), lda, int(B), ldb,
                           cuda.cuFloatComplex(beta.real, beta.imag),
                           int(C), ldc)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZhemm.restype = None
    _libcublas.cublasZhemm.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       cuda.cuDoubleComplex,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasZhemm(side, uplo, m, n, alpha,
                A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for Hermitian matrix.

    """
    
    _libcublas.cublasZhemm(side, uplo, m, n,
                           cuda.cuDoubleComplex(alpha.real,
                                                alpha.imag),
                           int(A), lda, int(B), ldb,
                           cuda.cuDoubleComplex(beta.real,
                                                beta.imag),
                           int(C), ldc)
    status = cublasGetError()
    cublasCheckStatus(status)

# CHERK, ZHERK (need to convert to CUDA 4.0)
if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasCherk.restype = None
    _libcublas.cublasCherk.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_float,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_float,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasCherk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """
    Rank-k operation on Hermitian matrix.

    """
    
    _libcublas.cublasCherk(uplo, trans, n, k, alpha,
                           int(A), lda, beta,
                           int(C), ldc)
    status = cublasGetError()
    cublasCheckStatus(status)

if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZherk.restype = None
    _libcublas.cublasZherk.argtypes = [ctypes.c_char,
                                       ctypes.c_char,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_double,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasZherk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """
    Rank-k operation on Hermitian matrix.

    """
    
    _libcublas.cublasZherk(uplo, trans, n, k,
                           alpha, int(A), lda, beta,
                           int(C), ldc)
    status = cublasGetError()
    cublasCheckStatus(status)

# CHER2K, ZHER2K (need to convert to CUDA 4.0)
if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasCher2k.restype = None
    _libcublas.cublasCher2k.argtypes = [ctypes.c_char,
                                        ctypes.c_char,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        cuda.cuFloatComplex,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_float,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
def cublasCher2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Rank-2k operation on Hermitian matrix.

    """
    
    _libcublas.cublasCher2k(uplo, trans, n, k,
                            cuda.cuFloatComplex(alpha.real,
                                                alpha.imag),
                            int(A), lda, int(B), ldb, beta,
                            int(C), ldc)
    status = cublasGetError()
    cublasCheckStatus(status)
        
if cuda.cudaDriverGetVersion() < 4000:
    _libcublas.cublasZher2k.restype = None
    _libcublas.cublasZher2k.argtypes = [ctypes.c_char,
                                        ctypes.c_char,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        cuda.cuDoubleComplex,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_double,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
def cublasZher2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Rank-2k operation on Hermitian matrix.

    """

    _libcublas.cublasZher2k(uplo, trans, n, k,
                            cuda.cuDoubleComplex(alpha.real,
                                                 alpha.imag),
                            int(A), lda, int(B), ldb,
                            beta, int(C), ldc)
    status = cublasGetError()
    cublasCheckStatus(status)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
