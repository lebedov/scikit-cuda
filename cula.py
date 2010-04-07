#!/usr/bin/env python


"""
Python interface to CULA toolkit.
"""

import ctypes
import atexit
import numpy as np

from numpy.linalg import LinAlgError

try:
    _libcula = ctypes.cdll.LoadLibrary('libcula.so')
except OSError:
    print 'libcula.so not found'

# Function for retrieving string associated with specific CULA status code:
_culaGetStatusString = _libcula.culaGetStatusString
_culaGetStatusString.restype = ctypes.c_char_p
def culaGetStatusString(e):
    """Get string associated with the specified CULA error status code."""

    return _culaGetStatusString(e)

class culaError(Exception):
    """CULA error."""
    pass

# Exceptions corresponding to various CULA errors:
class culaNotFoundError(culaError):
    """CULA shared library not found"""
    pass

class culaNotInitializedError(culaError):
    __doc__ = _culaGetStatusString(1)
    pass

class culaNoHardwareError(culaError):
    __doc__ = _culaGetStatusString(2)
    pass

class culaInsufficientRuntimeError(culaError):
    __doc__ = _culaGetStatusString(3)
    pass

class culaInsufficientComputeCapabilityError(culaError):
    __doc__ = _culaGetStatusString(4)
    pass

class culaInsufficientMemoryError(culaError):
    __doc__ = _culaGetStatusString(5)
    pass

class culaFeatureNotImplementedError(culaError):
    __doc__ = _culaGetStatusString(6)
    pass

class culaArgumentError(culaError):
    __doc__ = _culaGetStatusString(7)
    pass

class culaDataError(culaError):
    __doc__ = _culaGetStatusString(8)
    pass

class culaBlasError(culaError):
    __doc__ = _culaGetStatusString(9)
    pass

class culaRuntimeError(culaError):
    __doc__ = _culaGetStatusString(10)
    pass

culaExceptions = {
    -1: culaNotFoundError,
    1: culaNotInitializedError,
    2: culaNoHardwareError,
    3: culaInsufficientRuntimeError,
    4: culaInsufficientComputeCapabilityError,
    5: culaInsufficientMemoryError,
    6: culaFeatureNotImplementedError,
    7: culaArgumentError,
    8: culaDataError,
    9: culaBlasError,
    10: culaRuntimeError,
    }

# CULA functions:
def culaCheckStatus(status):
    """Raise an exception corresponding to the specified CULA status
    code."""
    if status != 0:
        raise culaErrors[status]

def culaGetErrorInfo(e):
    """Returns extended information about the last CULA error."""

    return _libcula.culaGetErrorInfo(e)

def culaGetLastStatus():
    """Returns the last status code returned from a CULA function."""
    
    return _libcula.culaGetLastStatus()

def culaInitialize():
    """Must be called before using any other CULA function."""
    
    return _libcula.culaInitialize()

def culaShutdown():
    """Shuts down CULA."""
    
    return _libcula.culaShutdown()

# Shut down CULA upon exit:
atexit.register(culaShutdown)

# Unexported functions copied from numpy.linalg:
def _makearray(a):
    new = np.asarray(a)
    wrap = getattr(a, "__array_prepare__", new.__array_wrap__)
    return new, wrap

def _assertRank2(*arrays):
    for a in arrays:
        if len(a.shape) != 2:
            raise LinAlgError, '%d-dimensional array given. Array must be \
            two-dimensional' % len(a.shape)

def _assertNonEmpty(*arrays):
    for a in arrays:
        if np.size(a) == 0:
            raise LinAlgError("Arrays cannot be empty")

def _fastCopyAndTranspose(type, a):
    if a.dtype.type is type:
        return np.fastCopyAndTranspose(a)
    else:
        return np.fastCopyAndTranspose(a.astype(type))

def svd(a, full_matrices=1, compute_uv=1):
    """
    Singular Value Decomposition.

    Factorizes the matrix `a` into two unitary matrices, ``U`` and
    ``Vh``, and a 1-dimensional array of singular values, ``s`` (real,
    non-negative), such that ``a == U S Vh``, where ``S`` is the
    diagonal matrix ``np.diag(s)``.

    Parameters
    ----------
    a : array_like
        Matrix of shape ``(M, N)`` to decompose.
    full_matrices : bool, optional
        If True (default), ``u`` and ``v.H`` have the shapes
        ``(M, M)`` and ``(N, N)``, respectively.  Otherwise, the shapes
        are ``(M, K)`` and ``(K, N)``, resp., where ``K = min(M, N)``.
    compute_uv : bool, optional
        Whether or not to compute ``u`` and ``v.H`` in addition to ``s``.
        True by default.

    Returns
    -------
    u : ndarray
        Unitary matrix. The shape of `U` is ``(M, M)`` or ``(M, K)``
        depending on value of `full_matrices`.
    s : ndarray
        The singular values, sorted so that ``s[i] >= s[i+1]``.
        `S` is a 1-D array of length ``min(M, N)``
    v.H : ndarray
        Unitary matrix of shape ``(N, N)`` or ``(K, N)``, depending
        on `full_matrices`.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    Notes
    -----
    Because of the limitations of the free version of CULA, the
    argument is cast to single precision.
    
    If `a` is a matrix object (as opposed to an `ndarray`), then so are all
    the return values.

    Examples
    --------
    >>> import cula
    >>> a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
    >>> U, s, Vh = cula.svd(a)
    >>> U.shape, Vh.shape, s.shape
    ((9, 9), (6, 6), (6,))

    >>> U, s, Vh = cula.svd(a, full_matrices=False)
    >>> U.shape, Vh.shape, s.shape
    ((9, 6), (6, 6), (6,))
    >>> S = np.diag(s)
    >>> np.allclose(a, np.dot(U, np.dot(S, Vh)))
    True

    >>> s2 = cula.svd(a, compute_uv=False)
    >>> np.allclose(s, s2)
    True

    """

    _a, wrap = _makearray(a)

    # Set M and N:
    (_m, _n) = _a.shape

    # The free version of CULA only supports single precision floating
    # point numbers:
    real_t = np.float32
    if np.iscomplexobj(_a):
        t = np.complex64
        cula_func = _libcula.culaCgesvd        
    else:
        t = np.float32
        cula_func = _libcula.culaSgesvd

    _a = _fastCopyAndTranspose(t, _a)
    _assertRank2(_a)
    _assertNonEmpty(_a)
    
    # Set LDA:
    _lda = max(1, _m)

    # Set S:
    _s = np.zeros(min(_m, _n), real_t)

    # Set JOBU and JOBVT:
    if compute_uv:
        if full_matrices:
            _jobu = 'A'
            _jobvt = 'A'
        else:
            _jobu = 'S'
            _jobvt = 'S'
    else:
        _jobu = 'N'
        _jobvt = 'N'

    # Set LDU and transpose of U:
    _ldu = _m
    if _jobu == 'A':
        _u = np.zeros((_ldu, _m), t)
    elif _jobu == 'S':
        _u = np.zeros((min(_m, _n), _ldu), t)
    else:
        _ldu = 1
        _u = np.empty((1,1), t)

    # Set LDVT and transpose of VT:
    if _jobvt == 'A':
        _ldvt = _n
        _vt = np.zeros((_n, _n), t)
    elif _jobvt == 'S':
        _ldvt = min(_m, _n)
        _vt = np.zeros((_n, _ldvt), t)
    else:
        _ldvt = 1
        _vt = np.empty((1, 1), t)        

    m = ctypes.c_int(_m)
    n = ctypes.c_int(_n)
    jobu = ctypes.c_char(_jobu)
    jobvt = ctypes.c_char(_jobvt)
    lda = ctypes.c_int(_lda)
    ldu = ctypes.c_int(_ldu)
    ldvt = ctypes.c_int(_ldvt)
    a = _a.ctypes.data_as(ctypes.c_void_p)
    s = _s.ctypes.data_as(ctypes.c_void_p)
    u = _u.ctypes.data_as(ctypes.c_void_p)
    vt = _vt.ctypes.data_as(ctypes.c_void_p)

    status = cula_func(jobu, jobvt, m, n, a, lda, s,
                       u, ldu, vt, ldvt)
    if status != 0:
        status = culaInitialize()
        culaCheckStatus(status)
        status = cula_func(jobu, jobvt, m, n, a, lda, s,
                           u, ldu, vt, ldvt)
        
    if status > 0:
        raise LinAlgError, 'SVD did not converge'

    if compute_uv:
        return wrap(_u.transpose()), _s, wrap(_vt.transpose())
    else:
        return _s


if __name__ == "__main__":
    import doctest
    doctest.testmod()
