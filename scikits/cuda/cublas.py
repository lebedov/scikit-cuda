#!/usr/bin/env python

"""
Python interface to CUBLAS functions.

Note: this module does not explicitly depend on PyCUDA.
"""

import sys
import ctypes
import atexit
import numpy as np

import cuda

if sys.platform == 'linux2':
    _libcublas_libname_list = ['libcublas.so', 'libcublas.so.3']
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

_libcublas.cublasInit.restype = int
_libcublas.cublasInit.argtypes = []
def cublasInit():
    """
    Initialize CUBLAS.

    This function must be called before using any other CUBLAS functions.

    """
    
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

    status = _libcublas.cublasShutdown()
    cublasCheckStatus(status)

atexit.register(_libcublas.cublasShutdown)

# Single precision real BLAS1 functions:
_libcublas.cublasIsamax.restype = ctypes.c_int
_libcublas.cublasIsamax.argtypes = [ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int]
def cublasIsamax(n, x, incx):
    """
    Index of maximum absolute value.

    Finds the smallest index of the maximum magnitude element of a
    single-precision real vector.

    Parameters
    ----------
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to single-precision real input vector.
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
    >>> x = np.random.rand(5).astype(np.float32)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> m = cublasIsamax(x_gpu.size, x_gpu.gpudata, 1)
    >>> np.allclose(m, np.argmax(x))
    True
    
    Notes
    -----
    This function returns a 0-based index.
    
    """
    a = _libcublas.cublasIsamax(n, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)
	
    return a-1
	

_libcublas.cublasIsamin.restype = ctypes.c_int
_libcublas.cublasIsamin.argtypes = [ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int]
def cublasIsamin(n, x, incx):
    """
    Index of minimum absolute value.

    Finds the smallest index of the minimum magnitude element of a
    single-precision real vector.

    Parameters
    ----------
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to single-precision real input vector.
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
    >>> x = np.random.rand(5).astype(np.float32)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> m = cublasIsamin(x_gpu.size, x_gpu.gpudata, 1)
    >>> np.allclose(m, np.argmin(x))
    True

    Notes
    -----
    This function returns a 0-based index.

    """
    
    a = _libcublas.cublasIsamin(n, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

    return a-1


_libcublas.cublasSasum.restype = ctypes.c_float
_libcublas.cublasSasum.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasSasum(n, x, incx):
    """
    Sum of absolute values of real vector.

    Computes the sum of the absolute values of the elements of a
    single-precision real vector.

    Parameters
    ----------
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to single-precision input vector.
    incx : int
        Storage spacing between elements of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = np.random.rand(5).astype(np.float32)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> s = cublasSasum(x_gpu.size, x_gpu.gpudata, 1)
    >>> np.allclose(s, np.sum(np.abs(x)))
    True

    Returns
    -------
    s : numpy.float32
        Sum of absolute values.
        
    """
    
    s = _libcublas.cublasSasum(n, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)	

    return np.float32(s)

_libcublas.cublasSaxpy.restype = None
_libcublas.cublasSaxpy.argtypes = [ctypes.c_int,
                                   ctypes.c_float,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasSaxpy(n, alpha, x, incx, y, incy):
    """
    Real vector addition.

    Computes the sum of a single-precision vector scaled by a
    single-precision scalar and another single-precision vector.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    alpha : numpy.float32
        Single-precision scalar.
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
    >>> alpha = np.float32(np.random.rand())
    >>> x = np.random.rand(5).astype(np.float32)
    >>> y = np.random.rand(5).astype(np.float32)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> cublasSaxpy(x_gpu.size, alpha, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> np.allclose(y_gpu.get(), alpha*x+y)
    True

    Notes
    -----
    Both `x` and `y` must contain `n` elements.
    
    """
    
    _libcublas.cublasSaxpy(n, alpha, int(x), incx, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)


_libcublas.cublasScopy.restype = None
_libcublas.cublasScopy.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasScopy(n, x, incx, y, incy):
    """
    Real vector copy.

    Copies a single-precision vector to another single-precision
    vector.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to single-precision input vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to single-precision output vector.
    incy : int
        Storage spacing between elements of `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = np.random.rand(5).astype(np.float32)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.zeros_like(x_gpu)
    >>> cublasScopy(x_gpu.size, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> np.allclose(y_gpu.get(), x_gpu.get())
    True
    
    Notes
    -----
    Both `x` and `y` must contain `n` elements.

    """

    _libcublas.cublasScopy(n, int(x), incx, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)
    

_libcublas.cublasSdot.restype = ctypes.c_float
_libcublas.cublasSdot.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int]
def cublasSdot(n, x, incx, y, incy):
    """
    Real vector dot product.

    Computes the dot product of two single-precision vectors.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to single-precision input vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to single-precision input/output vector.
    incy : int
        Storage spacing between elements of `y`.

    Returns
    -------
    d : numpy.float32
        Dot product of `x` and `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = np.random.rand(5).astype(np.float32)
    >>> y = np.random.rand(5).astype(np.float32)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> d = cublasSdot(x_gpu.size, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> np.allclose(d, np.dot(x, y))
    True

    Notes
    -----
    Both `x` and `y` must contain `n` elements.
    
    """
    
    a = _libcublas.cublasSdot(n, int(x), incx, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)
    return np.float32(a)

_libcublas.cublasSnrm2.restype = ctypes.c_float
_libcublas.cublasSnrm2.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasSnrm2(n, x, incx):
    """
    Euclidean norm (2-norm) of real vector.

    Computes the Euclidean norm of a real single-precision vector.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to single-precision input vector.
    incx : int
        Storage spacing between elements of `x`.

    Returns
    -------
    nrm : numpy.float32
        Euclidean norm of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = np.random.rand(5).astype(np.float32)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> nrm = cublasSnrm2(x.size, x_gpu.gpudata, 1)
    >>> np.allclose(nrm, np.linalg.norm(x))
    True
    
    """
    
    a = _libcublas.cublasSnrm2(n, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)
    return np.float32(a)


_libcublas.cublasSrot.restype = None
_libcublas.cublasSrot.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  ctypes.c_float]
def cublasSrot(n, x, incx, y, incy, sc, ss):
    """
    Apply a real rotation to a real matrix.

    Multiplies the single-precision matrix `[[sc, ss], [-ss, sc]]`
    with the 2 x `n` single-precision matrix `[[x.T], [y.T]]`.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to single-precision input/output vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to single-precision input/output vector.
    incy : int
        Storage spacing between elements of `y`.
    sc : numpy.float32
        Element of rotation matrix.
    ss : numpy.float32
        Element of rotation matrix.

    Notes
    -----
    Both `x` and `y` must contain `n` elements.
    
    """
    
    _libcublas.cublasSrot(n, int(x), incx, int(y), incy, sc, ss)
    status = cublasGetError()
    cublasCheckStatus(status)

_libcublas.cublasSrotg.restype = None
_libcublas.cublasSrotg.argtypes = [ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def cublasSrotg(host_sa, host_sb):
    """
    Construct a real Givens rotation.

    Constructs the real single-precision Givens rotation matrix `G` =
    `[[sc, ss], [-ss, sc]]`, where `sc**2+ss**2 == 1`.

    Parameters
    ----------
    sa, sb : numpy.float32
        Values to use when constructing the rotation matrix.

    Returns
    -------
    sc, ss : numpy.float32
        Rotation matrix values.

    Notes
    -----
    This function runs on the host, not the GPU device.

    """

    sa = ctypes.c_float(host_sa)
    sb = ctypes.c_float(host_sb)
    sc = ctypes.c_float()
    ss = ctypes.c_float()
    _libcublas.cublasSrotg(ctypes.byref(sa), ctypes.byref(sb),
                           ctypes.byref(sc), ctypes.byref(ss))
    status = cublasGetError()
    cublasCheckStatus(status)
    return sc.value, ss.value

_libcublas.cublasSrotm.restype = None
_libcublas.cublasSrotm.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p]
def cublasSrotm(n, x, incx, y, incy, sparam):
    """
    Apply a real modified Givens rotation.

    Applies the modified Givens rotation `h` to the 2 x `n`
    matrix `[[x.T], [y.T]]`.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to single-precision input/output vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to single-precision input/output vector.
    incy : int
        Storage spacing between elements of `y`.
    sparam : numpy.ndarray
        sparam[0] contains the `sflag` described below;
        sparam[1:5] contains the values `[sh00, sh10, sh01, sh11]`
        that determine the rotation matrix `h`.

    Notes
    -----
    The rotation matrix may assume the following values:

    for `sflag` == -1.0, `h` == `[[sh00, sh01], [sh10, sh11]]`
    for `sflag` == 0.0,  `h` == `[[1.0, sh01], [sh10, 1.0]]`
    for `sflag` == 1.0,  `h` == `[[sh00, 1.0], [-1.0, sh11]]`
    for `sflag` == -2.0, `h` == `[[1.0, 0.0], [0.0, 1.0]]`

    Both `x` and `y` must contain `n` elements.
    
    """
    
    _libcublas.cublasSrotm(n, int(x), incx, int(y), incy,
                           int(sparam.ctypes.data))
    status = cublasGetError()
    cublasCheckStatus(status)
    

_libcublas.cublasSrotmg.restype = None
_libcublas.cublasSrotmg.argtypes = [ctypes.c_void_p,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p]
def cublasSrotmg(host_sd1, host_sd2, host_sx1, host_sy1):
    """
    Construct a real modified Givens rotation.

    Notes
    -----
    This function runs on the host, not the GPU device.

    """

    sd1 = ctypes.c_float(host_sd1)
    sd2 = ctypes.c_float(host_sd2)
    sx1 = ctypes.c_float(host_sx1)
    sy1 = ctypes.c_float(host_sy1)
    sparam = np.empty(5, np.float32)
    
    _libcublas.cublasSrotmg(ctypes.byref(sd1), ctypes.byref(sd2),
                            ctypes.byref(sx1), ctypes.byref(sy1),
                            int(sparam.ctypes.data))
    status = cublasGetError()
    cublasCheckStatus(status)

    return sparam


_libcublas.cublasSscal.restype = None
_libcublas.cublasSscal.argtypes = [ctypes.c_int,
                                   ctypes.c_float,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasSscal(n, alpha, x, incx):
    """
    Scale a real vector by a real scalar.

    Replaces a single-precision vector `x` with
    `alpha * x`.
    
    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    alpha : numpy.float32
        Scalar multiplier.
    x : ctypes.c_void_p
        Pointer to single-precision real input/output vector.
    incx : int
        Storage spacing between elements of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = np.random.rand(5).astype(np.float32)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> alpha = np.float32(np.random.rand())
    >>> cublasSscal(x.size, alpha, x_gpu.gpudata, 1)
    >>> np.allclose(x_gpu.get(), alpha*x)
    True
    
    """
    
    _libcublas.cublasSscal(n, alpha, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)
    

_libcublas.cublasSswap.restype = None
_libcublas.cublasSswap.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasSswap(n, x, incx, y, incy):
    """
    Swap real vectors.

    Swaps the contents of one real single-precision vector with those
    of another real single-precision vector.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to single-precision input/output vector.
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
    >>> x = np.random.rand(5).astype(np.float32)
    >>> y = np.random.rand(5).astype(np.float32)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> cublasSswap(x.size, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> np.allclose(x_gpu.get(), y)
    True
    >>> np.allclose(y_gpu.get(), x)
    True

    Notes
    -----
    Both `x` and `y` must contain `n` elements.
    
    """
    
    _libcublas.cublasSswap(n, int(x), incx, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)
    
# Single precision complex BLAS1 functions:
_libcublas.cublasCaxpy.restype = None
_libcublas.cublasCaxpy.argtypes = [ctypes.c_int,
                                   cuda.cuFloatComplex,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasCaxpy(n, alpha, x, incx, y, incy):
    """
    Complex vector addition.

    Computes the sum of a single-precision complex vector scaled by a
    single-precision complex scalar and another single-precision
    complex vector.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    alpha : numpy.complex64
        Single-precision complex scalar.
    x : ctypes.c_void_p
        Pointer to single-precision complex input vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to single-precision complex input/output vector.
    incy : int
        Storage spacing between elements of `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> alpha = np.complex64(np.random.rand())
    >>> x = np.random.rand(5).astype(np.complex64)
    >>> y = np.random.rand(5).astype(np.complex64)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> cublasCaxpy(x_gpu.size, alpha, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> np.allclose(y_gpu.get(), alpha*x+y)
    True

    Notes
    -----
    Both `x` and `y` must contain `n` elements.
    
    """

    _libcublas.cublasCaxpy(n, cuda.cuFloatComplex(alpha.real,
                                             alpha.imag),
                           int(x), incx, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)


_libcublas.cublasCcopy.restype = None
_libcublas.cublasCcopy.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasCcopy(n, x, incx, y, incy):
    """
    Complex vector copy.
    
    """
    
    _libcublas.cublasCcopy(n, int(x), incx, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)

_libcublas.cublasCdotc.restype = cuda.cuFloatComplex
_libcublas.cublasCdotc.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasCdotc(n, x, incx, y, incy):
    """
    Complex vector dot product.

    """
    
    a = _libcublas.cublasCdotc(n, int(x), incx, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)
    return np.float32(a.x) + np.complex64(1j)*np.float32(a.y)

_libcublas.cublasCdotu.restype = cuda.cuFloatComplex
_libcublas.cublasCdotu.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasCdotu(n, x, incx, y, incy):
    """
    Complex vector dot product.

    """
    
    a = _libcublas.cublasCdotu(n, int(x), incx, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)
    return np.float32(a.x) + np.complex64(1j)*np.float32(a.y)

_libcublas.cublasCrot.restype = None
_libcublas.cublasCrot.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_float,
                                  cuda.cuFloatComplex]
def cublasCrot(n, x, incx, y, incy, sc, cs):
    """
    Apply a complex rotation to a complex matrix.
    
    """
    
    _libcublas.cublasCrot(n, int(x), incx, int(y), incy, sc,
                          cuda.cuFloatComplex(cs.real,
                                              cs.imag))
    status = cublasGetError()
    cublasCheckStatus(status)
	
_libcublas.cublasCrotg.restype = None
_libcublas.cublasCrotg.argtypes = [ctypes.c_void_p,
                                   cuda.cuFloatComplex,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]

def cublasCrotg(host_ca, cb, host_sc, host_cs):
    """
    Construct a complex Givens rotation.

    """
    
    _libcublas.cublasCrotg(int(host_ca),
                           cuda.cuFloatComplex(cb.real,
                                               cb.imag),
                           int(host_sc), int(host_cs))
    status = cublasGetError()
    cublasCheckStatus(status)

_libcublas.cublasCscal.restype = None
_libcublas.cublasCscal.argtypes = [ctypes.c_int,
                                   cuda.cuFloatComplex,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasCscal(n, alpha, x, incx):
    """
    Scale a complex vector by a complex scalar.

    Replaces a single-precision vector `x` with
    `alpha * x`.
    
    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    alpha : numpy.complex64
        Scalar multiplier.
    x : ctypes.c_void_p
        Pointer to single-precision complex input/output vector.
    incx : int
        Storage spacing between elements of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> alpha = np.complex64(np.random.rand()+1j*np.random.rand())
    >>> cublasCscal(x.size, alpha, x_gpu.gpudata, 1)
    >>> np.allclose(x_gpu.get(), alpha*x)
    True

    """
    
    _libcublas.cublasCscal(n, cuda.cuFloatComplex(alpha.real, alpha.imag), int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

_libcublas.cublasCsrot.restype = None
_libcublas.cublasCsrot.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_float,
                                   ctypes.c_float]
def cublasCsrot(n, x, incx, y, incy, sc, ss):
    """
    Apply a real rotation to a complex matrix.
    
    """
    
    _libcublas.cublasCsrot(n, int(x), incx, int(y), incy, sc, ss)
    status = cublasGetError()
    cublasCheckStatus(status)
    
_libcublas.cublasCsscal.restype = None
_libcublas.cublasCsscal.argtypes = [ctypes.c_int,
                                    ctypes.c_float,
                                    ctypes.c_void_p,
                                    ctypes.c_int]
def cublasCsscal(n, alpha, x, incx):
    """
    Scale a complex vector by a real scalar.

    Replaces a single-precision vector `x` with
    `alpha * x`.
    
    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    alpha : numpy.float32
        Scalar multiplier.
    x : ctypes.c_void_p
        Pointer to single-precision complex input/output vector.
    incx : int
        Storage spacing between elements of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> alpha = np.float32(np.random.rand())
    >>> cublasCsscal(x.size, alpha, x_gpu.gpudata, 1)
    >>> np.allclose(x_gpu.get(), alpha*x)
    True

    """
    
    _libcublas.cublasCsscal(n, alpha, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

_libcublas.cublasCswap.restype = None
_libcublas.cublasCswap.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasCswap(n, x, incx, y, incy):
    """
    Swap complex vectors.

    """

    _libcublas.cublasCswap(n, int(x), incx, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)

_libcublas.cublasIcamax.restype = ctypes.c_int
_libcublas.cublasIcamax.argtypes = [ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int]
def cublasIcamax(n, x, incx):
    """
    Index of maximum absolute value.

    Finds the smallest index of the maximum magnitude element of a
    single-precision complex vector.

    Parameters
    ----------
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to single-precision complex input vector.
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
    >>> x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> m = cublasIcamax(x_gpu.size, x_gpu.gpudata, 1)
    >>> np.allclose(m, np.argmax(np.abs(x)))
    True
    
    Notes
    -----
    This function returns a 0-based index.

    """
    
    a = _libcublas.cublasIcamax(n, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)
    return a-1

_libcublas.cublasIcamin.restype = ctypes.c_int
_libcublas.cublasIcamin.argtypes = [ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int]
def cublasIcamin(n, x, incx):
    """
    Index of minimum absolute value.

    Finds the smallest index of the minimum magnitude element of a
    single-precision complex vector.

    Parameters
    ----------
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to single-precision complex input vector.
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
    >>> x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> m = cublasIcamin(x_gpu.size, x_gpu.gpudata, 1)
    >>> np.allclose(m, np.argmin(np.abs(x)))
    True

    Notes
    -----
    This function returns a 0-based index.
    
    """

    a = _libcublas.cublasIcamin(n, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)
    return a-1

_libcublas.cublasScasum.restype = ctypes.c_float
_libcublas.cublasScasum.argtypes = [ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int]
def cublasScasum(n, x, incx):
    """
    Sum of absolute values of complex vector.
    
    """
    
    a = _libcublas.cublasScasum(n, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)
    return np.float32(a)

_libcublas.cublasScnrm2.restype = ctypes.c_float
_libcublas.cublasScnrm2.argtypes = [ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int]

def cublasScnrm2(n, x, incx):
    """
    Euclidean norm (2-norm) of complex vector.

    """
    
    a = _libcublas.cublasScnrm2(n, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)
    return np.float32(a)

# Double precision real BLAS1 functions:
_libcublas.cublasIdamax.restype = ctypes.c_int
_libcublas.cublasIdamax.argtypes = [ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int]
def cublasIdamax(n, x, incx):
    """
    Index of maximum absolute value.

    Finds the smallest index of the maximum magnitude element of a
    double-precision real vector.

    Parameters
    ----------
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to double-precision real input vector.
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
    >>> x = np.random.rand(5).astype(np.float64)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> m = cublasIdamax(x_gpu.size, x_gpu.gpudata, 1)
    >>> np.allclose(m, np.argmax(x))
    True
    
    Notes
    -----
    This function returns a 0-based index.

    """
    
    a = _libcublas.cublasIdamax(n, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)
    return a-1

_libcublas.cublasIdamin.restype = ctypes.c_int
_libcublas.cublasIdamin.argtypes = [ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int]
def cublasIdamin(n, x, incx):
    """
    Index of minimum absolute value.

    Finds the smallest index of the minimum magnitude element of a
    double-precision vector.

    Parameters
    ----------
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to double-precision input vector.
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
    >>> x = np.random.rand(5).astype(np.float64)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> m = cublasIdamin(x_gpu.size, x_gpu.gpudata, 1)
    >>> np.allclose(m, np.argmin(x))
    True

    Notes
    -----
    This function returns a 0-based index.
    
    """
    
    a = _libcublas.cublasIdamin(n, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)
    return a-1

_libcublas.cublasDasum.restype = ctypes.c_double
_libcublas.cublasDasum.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasDasum(n, x, incx):
    """
    Sum of absolute values of real vector.

    Computes the sum of the absolute values of the elements of a
    double-precision vector.

    Parameters
    ----------
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to double-precision input vector.
    incx : int
        Storage spacing between elements of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = np.random.rand(5).astype(np.float64)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> s = cublasDasum(x_gpu.size, x_gpu.gpudata, 1)
    >>> np.allclose(s, np.sum(x))
    True

    Returns
    -------
    s : numpy.float64
        Sum of absolute values.

    """
    
    a = _libcublas.cublasDasum(n, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)
    return np.float64(a)

_libcublas.cublasDaxpy.restype = None
_libcublas.cublasDaxpy.argtypes = [ctypes.c_int,
                                   ctypes.c_double,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasDaxpy(n, alpha, x, incx, y, incy):
    """
    Real vector addition.

    Computes the sum of a double-precision vector scaled by a
    double-precision scalar and another double-precision vector.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    alpha : numpy.float64
        Double-precision scalar.
    x : ctypes.c_void_p
        Pointer to double-precision input vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to double-precision input/output vector.
    incy : int
        Storage spacing between elements of `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> alpha = np.float64(np.random.rand())
    >>> x = np.random.rand(5).astype(np.float64)
    >>> y = np.random.rand(5).astype(np.float64)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> cublasDaxpy(x_gpu.size, alpha, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> np.allclose(y_gpu.get(), alpha*x+y)
    True

    Notes
    -----
    Both `x` and `y` must contain `n` elements.

    """
    
    _libcublas.cublasDaxpy(n, alpha, int(x), incx, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)

_libcublas.cublasDcopy.restype = None
_libcublas.cublasDcopy.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasDcopy(n, x, incx, y, incy):
    """
    Real vector copy.

    Copies a double-precision vector to another double-precision
    vector.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to double-precision input vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to double-precision output vector.
    incy : int
        Storage spacing between elements of `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = np.random.rand(5).astype(np.float64)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.zeros_like(x_gpu)
    >>> cublasDcopy(x_gpu.size, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> np.allclose(y_gpu.get(), x_gpu.get())
    True
    
    Notes
    -----
    Both `x` and `y` must contain `n` elements.
    
    """
    
    _libcublas.cublasDcopy(n, int(x), incx, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)

_libcublas.cublasDdot.restype = ctypes.c_double
_libcublas.cublasDdot.argtypes = [ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int,
                                 ctypes.c_void_p,
                                 ctypes.c_int]
def cublasDdot(n, x, incx, y, incy):
    """
    Real vector dot product.

    Computes the dot product of two double-precision vectors.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to double-precision input vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to double-precision input/output vector.
    incy : int
        Storage spacing between elements of `y`.

    Returns
    -------
    d : numpy.float64
        Dot product of `x` and `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = np.random.rand(5).astype(np.float64)
    >>> y = np.random.rand(5).astype(np.float64)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> d = cublasDdot(x_gpu.size, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> np.allclose(d, np.dot(x, y))
    True

    Notes
    -----
    Both `x` and `y` must contain `n` elements.

    """
    
    a = _libcublas.cublasDdot(n, int(x), incx, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)
    return np.float64(a)

_libcublas.cublasDnrm2.restype = ctypes.c_double
_libcublas.cublasDnrm2.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasDnrm2(n, x, incx):
    """
    Euclidean norm (2-norm) of real vector.    

    Computes the Euclidean norm of a real double-precision vector.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to double-precision input vector.
    incx : int
        Storage spacing between elements of `x`.

    Returns
    -------
    nrm : numpy.float64
        Euclidean norm of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = np.random.rand(5).astype(np.float64)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> nrm = cublasDnrm2(x.size, x_gpu.gpudata, 1)
    >>> np.allclose(nrm, np.linalg.norm(x))
    True

    """
    
    a = _libcublas.cublasDnrm2(n, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)
    return np.float64(a)

_libcublas.cublasDrot.restype = None
_libcublas.cublasDrot.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_double,
                                  ctypes.c_double]
def cublasDrot(n, x, incx, y, incy, dc, ds):
    """
    Apply a real rotation to a real matrix.

    Multiplies the double-precision matrix `[[sc, ss], [-ss, sc]]`
    with the 2 x `n` double-precision matrix `[[x.T], [y.T]]`.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to double-precision input/output vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to double-precision input/output vector.
    incy : int
        Storage spacing between elements of `y`.
    sc : numpy.float64
        Element of rotation matrix.
    ss : numpy.float64
        Element of rotation matrix.

    Notes
    -----
    Both `x` and `y` must contain `n` elements.

    """
    _libcublas.cublasDrot(n, int(x), incx, int(y), incy, dc, ds)
    status = cublasGetError()
    cublasCheckStatus(status)

_libcublas.cublasDrotg.restype = None
_libcublas.cublasDrotg.argtypes = [ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def cublasDrotg(host_da, host_db):
    """
    Construct a real Givens rotation.

    Constructs the real Givens rotation matrix `G` =
    `[[dc, ds], [-ds, dc]]`, where `dc**2+ds**2 == 1`.

    Parameters
    ----------
    da, db : numpy.float64
        Values to use when constructing the rotation matrix.

    Returns
    -------
    dc, ds : numpy.float64
        Rotation matrix values.

    Notes
    -----
    This function runs on the host, not the GPU device.

    """

    da = ctypes.c_double(host_da)
    db = ctypes.c_double(host_db)
    dc = ctypes.c_double()
    ds = ctypes.c_double()
    _libcublas.cublasDrotg(ctypes.byref(da), ctypes.byref(db),
                           ctypes.byref(dc), ctypes.byref(ds))
    status = cublasGetError()
    cublasCheckStatus(status)
    return dc.value, ds.value
    
_libcublas.cublasDrotm.restype = None
_libcublas.cublasDrotm.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p]

def cublasDrotm(n, x, incx, y, incy, dparam):
    """
    Apply a real modified Givens rotation to a real matrix.

    """
    
    _libcublas.cublasDrotm(n, int(x), incx, int(y), incy,
                           int(dparam.ctypes.data))
    status = cublasGetError()
    cublasCheckStatus(status)

_libcublas.cublasDrotmg.restype = None
_libcublas.cublasDrotmg.argtypes = [ctypes.c_void_p,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p]
def cublasDrotmg(host_dd1, host_dd2, host_dx1, host_dy1):
    """
    Construct a real modified Givens rotation.

    Notes
    -----
    This function runs on the host, not the GPU device.
    
    """

    dd1 = ctypes.c_double(host_dd1)
    dd2 = ctypes.c_double(host_dd2)
    dx1 = ctypes.c_double(host_dx1)
    dy1 = ctypes.c_double(host_dy1)
    dparam = np.empty(5, np.float64)

    _libcublas.cublasDrotmg(ctypes.byref(dd1), ctypes.byref(dd2),
                            ctypes.byref(dx1), ctypes.byref(dy1),
                            int(dparam.ctypes.data))
    status = cublasGetError()
    cublasCheckStatus(status)

    return dparam

_libcublas.cublasDscal.restype = None
_libcublas.cublasDscal.argtypes = [ctypes.c_int,
                                   ctypes.c_double,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasDscal(n, alpha, x, incx):
    """
    Scale a real vector by a real scalar.

    Replaces a double-precision vector `x` with
    `alpha * x`.
    
    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    alpha : numpy.float64
        Scalar multiplier.
    x : ctypes.c_void_p
        Pointer to double-precision real input/output vector.
    incx : int
        Storage spacing between elements of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = np.random.rand(5).astype(np.float64)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> alpha = np.float64(np.random.rand())
    >>> cublasDscal(x.size, alpha, x_gpu.gpudata, 1)
    >>> np.allclose(x_gpu.get(), alpha*x)
    True

    """
    
    _libcublas.cublasDscal(n, alpha, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)

_libcublas.cublasDswap.restype = None
_libcublas.cublasDswap.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasDswap(n, x, incx, y, incy):
    """
    Swap real vectors.

    Swaps the contents of one real double-precision vector with those
    of another real double-precision vector.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to double-precision input/output vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to double-precision input/output vector.
    incy : int
        Storage spacing between elements of `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = np.random.rand(5).astype(np.float64)
    >>> y = np.random.rand(5).astype(np.float64)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> cublasDswap(x.size, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> np.allclose(x_gpu.get(), y)
    True
    >>> np.allclose(y_gpu.get(), x)
    True

    Notes
    -----
    Both `x` and `y` must contain `n` elements.

    """
    
    _libcublas.cublasDswap(n, int(x), incx, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)


# Double precision complex BLAS1
_libcublas.cublasDzasum.restype = ctypes.c_double
_libcublas.cublasDzasum.argtypes = [ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int]
def cublasDzasum(n, x, incx):
    """
    Sum of absolute values of complex vector.

    """
    
    a = _libcublas.cublasDzasum(n, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)
    return np.float64(a)

_libcublas.cublasDznrm2.restype = ctypes.c_double
_libcublas.cublasDznrm2.argtypes = [ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int]
def cublasDznrm2(n, x, incx):
    """
    Euclidean norm (2-norm) of complex vector.

    """
    
    a = _libcublas.cublasDznrm2(n, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)
    return np.float64(a)

_libcublas.cublasIzamax.restype = ctypes.c_int
_libcublas.cublasIzamax.argtypes = [ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int]
def cublasIzamax(n, x, incx):
    """
    Index of maximum absolute value.

    Finds the smallest index of the maximum magnitude element of a
    double-precision complex vector.

    Parameters
    ----------
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to double-precision complex input vector.
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
    >>> x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> m = cublasIzamax(x_gpu.size, x_gpu.gpudata, 1)
    >>> np.allclose(m, np.argmax(np.abs(x)))
    True
    
    Notes
    -----
    This function returns a 0-based index.
    
    """
    
    a = _libcublas.cublasIzamax(n, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)
    return a-1

_libcublas.cublasIzamin.restype = ctypes.c_int
_libcublas.cublasIzamin.argtypes = [ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int]

def cublasIzamin(n, x, incx):
    """
    Index of minimum absolute value.

    Finds the smallest index of the minimum magnitude element of a
    double-precision complex vector.

    Parameters
    ----------
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to double-precision complex input vector.
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
    >>> x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> m = cublasIzamin(x_gpu.size, x_gpu.gpudata, 1)
    >>> np.allclose(m, np.argmin(np.abs(x)))
    True

    Notes
    -----
    This function returns a 0-based index.

    """

    a = _libcublas.cublasIzamin(n, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)
    return a-1

_libcublas.cublasZaxpy.restype = None
_libcublas.cublasZaxpy.argtypes = [ctypes.c_int,
                                   cuda.cuDoubleComplex,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasZaxpy(n, alpha, x, incx, y, incy):
    """
    Complex vector addition.

    Computes the sum of a double-precision complex vector scaled by a
    double-precision complex scalar and another double-precision
    complex vector.

    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    alpha : numpy.complex128
        Double-precision complex scalar.
    x : ctypes.c_void_p
        Pointer to double-precision complex input vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to double-precision complex input/output vector.
    incy : int
        Storage spacing between elements of `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> alpha = np.complex128(np.random.rand())
    >>> x = np.random.rand(5).astype(np.complex128)
    >>> y = np.random.rand(5).astype(np.complex128)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> cublasZaxpy(x_gpu.size, alpha, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> np.allclose(y_gpu.get(), alpha*x+y)
    True

    Notes
    -----
    Both `x` and `y` must contain `n` elements.
    
    """
    
    _libcublas.cublasZaxpy(n, cuda.cuDoubleComplex(alpha.real,
                                                   alpha.imag),
                           int(x), incx, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)

_libcublas.cublasZcopy.restype = None
_libcublas.cublasZcopy.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasZcopy(n, x, incx, y, incy):
    """
    Complex vector copy.

    """
    
    _libcublas.cublasZcopy(n, int(x), incx, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)

_libcublas.cublasZdotc.restype = cuda.cuDoubleComplex
_libcublas.cublasZdotc.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasZdotc(n, x, incx, y, incy):
    """
    Complex vector dot product.

    """
    
    a = _libcublas.cublasZdotc(n, int(x), incx, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)
    return np.float64(a.x) + np.complex128(1j)*np.float64(a.y)


_libcublas.cublasZdotu.restype = cuda.cuDoubleComplex
_libcublas.cublasZdotu.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]

def cublasZdotu(n, x, incx, y, incy):
    """
    Complex vector dot product.

    """
    
    a = _libcublas.cublasZdotu(n, int(x), incx, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)
    return np.float64(a.x) + np.complex128(1j)*np.float64(a.y)

_libcublas.cublasZdrot.restype = None
_libcublas.cublasZdrot.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_double,
                                   ctypes.c_double]
def cublasZdrot(n, x, incx, y, incy, c, s):
    """
    Apply a real rotation to a complex matrix.

    """

    _libcublas.cublasZdrot(n, int(x), incx, int(y), incy, cs, s)
    status = cublasGetError()
    cublasCheckStatus(status)

_libcublas.cublasZdscal.restype = None
_libcublas.cublasZdscal.argtypes = [ctypes.c_int,
                                    ctypes.c_double,
                                    ctypes.c_void_p,
                                    ctypes.c_int]
def cublasZdscal(n, alpha, x, incx):
    """
    Scale a complex vector by a real scalar.

    Replaces a double-precision vector `x` with
    `alpha * x`.
    
    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    alpha : numpy.float64
        Scalar multiplier.
    x : ctypes.c_void_p
        Pointer to double-precision complex input/output vector.
    incx : int
        Storage spacing between elements of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> alpha = np.float64(np.random.rand())
    >>> cublasZdscal(x.size, alpha, x_gpu.gpudata, 1)
    >>> np.allclose(x_gpu.get(), alpha*x)
    True

    """
    
    _libcublas.cublasZdscal(n, alpha, int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)



_libcublas.cublasZrot.restype = None
_libcublas.cublasZrot.argtypes = [ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_double,
                                  cuda.cuDoubleComplex]
def cublasZrot(n, x, incx, y, incy, sc, cs):
    """
    Apply complex rotation to complex matrix.

    """
    
    _libcublas.cublasZrot(n, int(x), incx, int(y), incy,
                          sc, cuda.cuDoubleComplex(cs.real,
                                                   cs.imag))
    status = cublasGetError()
    cublasCheckStatus(status)

_libcublas.cublasZrotg.restype = None
_libcublas.cublasZrotg.argtypes = [ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p]
def cublasZrotg(host_ca, host_cb, host_sc, host_cs):
    """
    Construct a complex Givens rotation.

    """
    
    _libcublas.cublasZrotg(int(host_ca), int(host_cb),
                           int(host_sc), int(host_cs))
    status = cublasGetError()
    cublasCheckStatus(status)


_libcublas.cublasZscal.restype = None
_libcublas.cublasZscal.argtypes = [ctypes.c_int,
                                   cuda.cuDoubleComplex,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasZscal(n, alpha, x, incx):
    """
    Scale a complex vector by a complex scalar.

    Replaces a double-precision vector `x` with
    `alpha * x`.
    
    Parameters
    ----------
    n : int
        Number of elements in input vectors.
    alpha : numpy.complex128
        Scalar multiplier.
    x : ctypes.c_void_p
        Pointer to double-precision complex input/output vector.
    incx : int
        Storage spacing between elements of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> alpha = np.complex128(np.random.rand()+1j*np.random.rand())
    >>> cublasZscal(x.size, alpha, x_gpu.gpudata, 1)
    >>> np.allclose(x_gpu.get(), alpha*x)
    True

    """
    
    _libcublas.cublasZscal(n, cuda.cuDoubleComplex(alpha.real,
                                                   alpha.imag),
                           int(x), incx)
    status = cublasGetError()
    cublasCheckStatus(status)


_libcublas.cublasZswap.restype = None
_libcublas.cublasZswap.argtypes = [ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int]
def cublasZswap(n, x, incx, y, incy):
    """
    Swap complex vectors.

    """
    
    _libcublas.cublasZswap(n, int(x), incx, int(y), incy)
    status = cublasGetError()
    cublasCheckStatus(status)

# Single precision real BLAS2 functions:
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
    x : ctypes.c_void_p
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

# Single precision complex BLAS2 functions:
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
def cublasCgvmv(trans, m, n, kl, ku, alpha, A, lda,
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

# Double precision real BLAS2 functions:
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

# Double precision complex BLAS2 functions:
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

# Single precision real BLAS3:
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

# Single precision complex BLAS3 functions:
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

# Double precision real BLAS3:
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
    
# Double precision complex BLAS3 functions:
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

if __name__ == "__main__":
    import doctest
    doctest.testmod()
