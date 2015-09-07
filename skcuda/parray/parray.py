#!/usr/bin/env python

"""
Pitched array analogue to pycuda.gpuarray.GPUArray.
"""

import numpy as np
import pycuda.driver as cuda
from pytools import memoize

import parray_utils as pu

""" utilities"""
@memoize
def _splay_backend(n, M):
    block_count = 6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT
    if M <= 1:
        block = (256, 1, 1)
    else:
        block = (32, 8, 1)
    return (block_count, 1), block


def splay(n, M):
    return _splay_backend(n, M)

def _get_common_dtype(obj1, obj2):
    """Return the 'least common multiple' of dtype of obj1 and obj2."""
    return (obj1.dtype.type(0) + obj2.dtype.type(0)).dtype

def _get_inplace_dtype(obj1, obj2):
    """
    Returns the dtype of obj1,
    Raise error if
    1) obj1 is real and obj2 is complex
    2) obj1 is integer and obj2 is floating

    Parameters
    ----------
    obj1 : numpy.ndarray like array
    obj2 : numpy.ndarray like array

    Returns
    -------
    out : np.dtype
    """
    if isrealobj(obj1):
        if iscomplexobj(obj2):
            raise TypeError("Cannot cast complex dtype to real dtype")
    if issubclass(obj1.dtype.type, np.integer):
        if issubclass(obj2.dtype.type, (np.floating, np.complexfloating)):
            raise TypeError("Cannot cast floating to integer")
    return obj1.dtype

def _get_common_dtype_with_scalar(scalar, obj1):
    """
    return the common dtype between a native scalar (int, float, complex)
    and the dtype of an ndarray like array.

    Parameters
    ----------
    scalar : { int, float, complex }
    obj1 : numpy.ndarray like array.

    """
    if issubclass(type(scalar), (int, float, np.integer, np.floating)):
        return obj1.dtype
    elif issubclass(type(scalar), (complex, np.complexfloating)):
        if isrealobj(obj1):
            return floattocomplex(obj1.dtype)
        else:
            return obj1.dtype
    else:
        raise TypeError("scalar type is not supported")


def _get_inplace_dtype_with_scalar(scalar, obj1):
    """
    Returns the dtype of obj1,
    Raise error if
    1) obj1 is real and obj2 is complex
    2) obj1 is integer and obj2 is floating

    Parameters
    ----------
    obj1 : numpy.ndarray like array
    obj2 : numpy.ndarray like array

    Returns
    -------
    out : np.dtype
    """

    if isrealobj(obj1):
        if issubclass(type(scalar), (complex, np.complexfloating)):
            raise TypeError("Cannot cast complex dtype to real dtype")
    if issubclass(obj1.dtype.type, np.integer):
        if issubclass(
                type(scalar),
                (float, complex, np.floating, np.complexfloating)):
            raise TypeError("Cannot cast floating to integer")
    return obj1.dtype

def _pd(shape):
    """ Returns the product of all element of shape except shape[0]. """
    s = 1
    for dim in shape[1:]:
        s *= dim
    return s

def _assignshape(shape, axis, value):
    a = []
    for i in range(len(shape)):
        if i == axis:
            a.append(value)
        else:
            a.append(shape[i])
    return tuple(a)

def PitchTrans(shape, dst, dst_ld, src, src_ld, dtype, aligned=False,
               async = False, stream = None):
    """
    Pitched memory transfer.

    Wraps pycuda.driver.Memcpy2D.

    Parameters
    ----------
    shape : tuple of ints
        shape of the 2D array to be transferred.
    dst : { cuda.DeviceAllocation, int, long }
        pointer to the device memory to be transferred to.
    dst_ld: int
        leading dimension (pitch) of destination.
    src : { pycuda.driver.DeviceAllocation, int, long }
        pointer to the device memory to be transferred from.
    src_ld : int
        leading dimension (pitch) of source.

    Optional Parameters
    -------------------
    aligned : bool
        if aligned is False, tolerate device-side misalignment for
        device-to-device copies that may lead to loss of copy bandwidth.
        (default: False).
    async : bool
        use asynchronous transfer (default: False).
    stream : pycuda.driver.stream
        specify the cuda stream (default: None).
    """

    size = np.dtype(dtype).itemsize

    trans = cuda.Memcpy2D()
    trans.src_pitch = src_ld * size
    if isinstance(src, (cuda.DeviceAllocation, int, long)):
        trans.set_src_device(src)
    else:
        trans.set_src_host(src)

    trans.dst_pitch = dst_ld * size
    if isinstance(dst, (cuda.DeviceAllocation, int, long)):
        trans.set_dst_device(dst)
    else:
        trans.set_dst_host(dst)

    trans.width_in_bytes = _pd(shape) * size
    trans.height = int(shape[0])

    if async:
        trans(stream)
    else:
        trans(aligned = aligned)

"""end of utilities"""

class PitchArray(object):
    def __init__(self, shape, dtype, gpudata=None, pitch = None, base = None):
        """
        Create a PitchArray

        Parameters
        ----------
        shape: tuple of ints
            shape of the array
        dtype: np.dtype
            dtype of the array
        gpudata: pycuda.driver.DeviceAllocation
            DeviceAllocation object indicating the device memory allocated
        pitch: int
            if gpudata is specified and pitch is True,
            gpudata will be treated as if it was allocated by
            cudaMallocPitch with pitch
        base: PitchArray
            base PitchArray

        Attributes:
        -----------
        .shape: shape of self
        .size:  number of elements of the array
        .mem_size: number of elements of total memory allocated
        .ld: leading dimension
        .M: 1 if self is a vector, shape[0] otherwise
        .N: self.size if self is a vector
            product of shape[1] and shape[2] otherwise
        .gpudata: DeviceAllocation
        .ndim: number of dimensions
        .dtype: dtype of array
        .nbytes: total memory allocated for the array in bytes
        .base: base PitchArray

        Note:
        -----
        1. any 1-dim shape will result in a row vector with
        new shape as (1, shape) operations of PitchArray
        is elementwise operation

        2. only support array of dimension up to 3.
        """

        try:
            tmpshape = []
            s = 1
            for dim in shape:
                dim = int(dim)
                assert isinstance(dim, int)
                s *= dim
                tmpshape.append(dim)

            self.shape = tuple(tmpshape)
        except TypeError:
            s = int(shape)
            assert isinstance(s, int)
            if s:
                self.shape = (1, s)
            else:
                self.shape = (0, 0)

        if len(self.shape) == 1:
            self.shape = (1, self.shape[0])

        self.ndim = len(self.shape)

        if self.ndim > 3:
            raise ValueError("Only support array of dimension leq 3")

        self.dtype = np.dtype(dtype)
        self.size = s

        if gpudata is None:
            if self.size:
                if _pd(self.shape) == 1 or self.shape[0] == 1:
                    self.gpudata = cuda.mem_alloc(
                                   self.size * self.dtype.itemsize)
                    self.mem_size = self.size
                    self.ld = _pd(self.shape)
                    self.M = 1
                    self.N = self.size
                else:
                    self.gpudata, pitch = cuda.mem_alloc_pitch(
                        int(_pd(self.shape) * np.dtype(dtype).itemsize),
                        self.shape[0], np.dtype(dtype).itemsize)
                    self.ld = pitch / np.dtype(dtype).itemsize
                    self.mem_size = self.ld * self.shape[0]
                    self.M = self.shape[0]
                    self.N = _pd(self.shape)
            else:
                self.gpudata = None
                self.M = 0
                self.N = 0
                self.ld = 0
                self.mem_size = 0
            self.base = base
        else:
            # assumed that the device memory was also allocated
            # by mem_alloc_pitch is required by the shape
            # Not performed to allow base
            # assert gpudata.__class__ == cuda.DeviceAllocation

            if self.size:
                self.gpudata = gpudata
                if _pd(self.shape) == 1 or self.shape[0] == 1:
                    self.mem_size = self.size
                    self.ld = _pd(self.shape)
                    self.M = 1
                    self.N = self.size
                else:
                    if pitch is None:
                        pitch = int(np.ceil(
                                float(_pd(self.shape) 
                                * np.dtype(dtype).itemsize)
                                / 512) * 512)
                    else:
                        assert pitch == int(np.ceil(
                                        float(_pd(self.shape) 
                                        * np.dtype(dtype).itemsize) 
                                        / 512) * 512)
                    
                    self.ld = pitch / np.dtype(dtype).itemsize
                    self.mem_size = self.ld * self.shape[0]
                    self.M = self.shape[0]
                    self.N = _pd(self.shape)
            else:
                self.gpudata = None
                self.M = 0
                self.N = 0
                self.ld = 0
                self.mem_size = 0
                print "warning: shape may not be assigned properly"
            self.base = base
        self.nbytes = self.dtype.itemsize * self.mem_size
        self._grid, self._block = splay(self.mem_size, self.M)
    	
    def set(self, ary):
        """
        Set PitchArray with an numpy.ndarray

        Parameter:
        ----------
        ary: num.ndarray
             must have the same shape as self if ndim > 2,
             and same length as self if ndim == 1
        """

        assert ary.ndim <= 3
        assert ary.dtype == ary.dtype
        assert ary.size == self.size

        if self.size:
            if self.M == 1:
                cuda.memcpy_htod(int(self.gpudata), ary)
            else:
                PitchTrans(self.shape, int(self.gpudata),
                           self.ld, ary, _pd(self.shape), self.dtype)

    def set_async(self, ary, stream=None):
        """
        Set PitchArray with an numpy.ndarray
        using asynchrous memroy transfer

        Parameter:
        ----------
        ary: num.ndarray pagelocked
             must have the same shape as self if ndim > 2,
             and same length as self if ndim == 1
             must be created by cuda.HostAllocation

        Optional Parameter:
        -------------------
        stream : pycuda.driver.Stream
        """

        assert ary.ndim <= 3
        assert ary.dtype == ary.dtype
        assert ary.size == self.size

        if ary.base.__class__ != cuda.HostAllocation:
            raise TypeError("asynchronous memory trasfer "
                            "requires pagelocked numpy array")

        if self.size:
            if self.M == 1:
                cuda.memcpy_htod_async(int(self.gpudata), ary, stream)
            else:
                PitchTrans(self.shape, int(self.gpudata), self.ld, ary,
                           _pd(self.shape), self.dtype, async = True,
                           stream = stream)

    def get(self, ary = None, pagelocked = False):
        """
        Get the PitchArray to an ndarray

        Optional Parameters:
        --------------------
        ary: numpy.ndarray
            if specified, will transfer device memory to ary's memory
            if None, create a new array
            (default: None).
        pagelocked: bool
            if True, create a pagelocked ndarray
            if False, create a regular ndarray
            (default: False)
        """

        if ary is None:
            if pagelocked:
                ary = cuda.pagelocked_empty(self.shape, self.dtype)
            else:
                ary = np.empty(self.shape, self.dtype)
        else:
            assert ary.size == self.size
            assert ary.dtype == ary.dtype

        if self.size:
            if self.M == 1:
                cuda.memcpy_dtoh(ary, int(self.gpudata))
            else:
                PitchTrans(self.shape, ary, _pd(self.shape),
                           int(self.gpudata), self.ld, self.dtype)

        return ary


    def get_async(self, stream = None, ary = None):
        """
        Get the PitchArray to an ndarray asynchronously

        Optional Parameters:
        --------------------
        stream: pycuda.driver.Stream
            (default: None)
        ary: numpy.ndarray
            if specified, will transfer device memory to ary's memory,
            must be pagelocked
            if None, create a new array
            (default: None)
        """

        if ary is None:
            ary = cuda.pagelocked_empty(self.shape, self.dtype)
        else:
            assert ary.size == self.size
            assert ary.dtype == ary.dtype
            if ary.base.__class__ != cuda.HostAllocation:
                raise TypeError("asynchronous memory trasfer "
                                "requires pagelocked numpy array")

        if self.size:
            if self.M == 1:
                cuda.memcpy_dtoh_async(ary, int(self.gpudata), stream)
            else:
                PitchTrans(self.shape, ary, _pd(self.shape),
                           int(self.gpudata), self.ld, self.dtype,
                           async = True, stream = stream)

        return ary

    def __str__(self):
        return str(self.get())

    def __repr__(self):
        return repr(self.get())

    def __hash__(self):
        raise TypeError("PitchArrays are not hashable.")

    def _new_like_me(self, dtype = None):
        if dtype is None:
            dtype = self.dtype
        return self.__class__(self.shape, dtype)

    """""""""
    Operators:
    operators defined by __op__(self, other) returns new PitchArray
    operators defined by op(self, other)  perform inplace operation
        if inplace cannot be done, error raises
    operators defined by __iop__(self, other) also perform inplace
        operation if possbile, otherwise returns a new PitchArray
    """""""""

    def __add__(self, other):
        """
        Addition

        Parameters
        ----------
        other: scalar or Pitcharray

        Returns
        -------
        out : PitchArray
            A new PitchArray
        """

        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            result = self._new_like_me(_get_common_dtype(self, other))
            if self.size:
                if self.M == 1:
                    func = pu.get_addarray_function(
                        self.dtype, other.dtype, result.dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, result.gpudata,
                        self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_addarray_function(
                        self.dtype, other.dtype, result.dtype, pitch = True)
                    func.prepared_call(
                        self._grid, self._block, self.M, self.N,
                        result.gpudata, result.ld, self.gpudata,
                        self.ld, other.gpudata, other.ld)
            return result
        elif issubclass(type(other), (float, int, complex, np.integer,
                                      np.floating, np.complexfloating)):
            dtype = _get_common_dtype_with_scalar(other, self)
            if other == 0:
                return self.astype(dtype)
            else:
                result = self._new_like_me(dtype)
                if self.size:
                    if self.M == 1:
                        func = pu.get_addscalar_function(
                            self.dtype, dtype, pitch = False)
                        func.prepared_call(
                            self._grid, self._block, result.gpudata,
                            self.gpudata, other, self.size)
                    else:
                        func = pu.get_addscalar_function(
                            self.dtype, dtype, pitch = True)
                        func.prepared_call(
                            self._grid, self._block, self.M, self.N,
                            result.gpudata, result.ld, self.gpudata,
                            self.ld, other)
                return result
        else:
            raise TypeError("type of object to be added is not supported")

    __radd__ = __add__
    
    def __sub__(self, other):
        """
        Substraction
        self - other
        
        Parameters
        ----------
        other : scalar or Pitcharray
        
        Returns
        -------
        out : PitchArray
            A new PitchArray
        """
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            result = self._new_like_me(_get_common_dtype(self, other))
            if self.size:
                if self.M == 1:
                    func = pu.get_subarray_function(
                        self.dtype, other.dtype, result.dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, result.gpudata,
                        self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_subarray_function(
                        self.dtype, other.dtype, result.dtype, pitch = True)
                    func.prepared_call(
                        self._grid, self._block, self.M, self.N,
                        result.gpudata, result.ld, self.gpudata,
                        self.ld, other.gpudata, other.ld)
            return result
        elif issubclass(type(other), (float, int, complex, np.integer,
                                      np.floating, np.complexfloating)):
            dtype = _get_common_dtype_with_scalar(other, self)
            if other == 0:
                return self.astype(dtype)
            else:
                result = self._new_like_me(dtype)
                if self.size:
                    if self.M == 1:
                        func = pu.get_subscalar_function(
                            self.dtype, dtype, pitch = False)
                        func.prepared_call(
                            self._grid, self._block, result.gpudata,
                            self.gpudata, other, self.size)
                    else:
                        func = pu.get_subscalar_function(
                            self.dtype, dtype, pitch = True)
                        func.prepared_call(
                            self._grid, self._block, self.M, self.N,
                            result.gpudata, result.ld, self.gpudata,
                            self.ld, other)
                return result
        else:
            raise TypeError("type of object to be substracted "
                            "is not supported")

    def __rsub__(self, other):
        """
        Being substracted
        Other - self
        
        Parameters
        ----------
        other: scalar or Pitcharray
        
        Returns
        -------
        out : PitchArray
            A new PitchArray
        """
        """
        # this part is not neccessary
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            result = self._new_like_me(_get_common_dtype(self, other))
            if self.size:
                if self.M == 1:
                    func = pu.get_subarray_function(
                        other.dtype, self.dtype, result.dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, result.gpudata,
                        other.gpudata, self.gpudata, self.size)
                else:
                    func = pu.get_subarray_function(
                        other.dtype, self.dtype, result.dtype, pitch = True)
                    func.prepared_call(
                        self._grid, self._block, self.M, self.N,
                        result.gpudata, result.ld, other.gpudata,
                        other.ld, self.gpudata, self.ld)
            return result
        """
        if issubclass(type(other), (float, int, complex, np.integer,
                                      np.floating, np.complexfloating)):
            dtype = _get_common_dtype_with_scalar(other, self)
            result = self._new_like_me(dtype)
            if self.size:
                if self.M == 1:
                    func = pu.get_scalarsub_function(
                        self.dtype, dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, result.gpudata,
                        self.gpudata, other, self.size)
                else:
                    func = pu.get_scalarsub_function(
                        self.dtype, dtype, pitch = True)
                    func.prepared_call(
                        self._grid, self._block, self.M, self.N,
                        result.gpudata, result.ld, self.gpudata,
                        self.ld, other)
            return result
        else:
            raise TypeError("type of object to substract from "
                            "is not supported")
    
    def __mul__(self, other):
        """
        Multiply
        
        Parameters
        ----------
        other: scalar or Pitcharray
        
        Returns
        -------
        out : PitchArray
            A new PitchArray
        """
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            result = self._new_like_me(_get_common_dtype(self, other))
            if self.size:
                if self.M == 1:
                    func = pu.get_mularray_function(
                        self.dtype, other.dtype, result.dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, result.gpudata,
                        self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_mularray_function(
                        self.dtype, other.dtype, result.dtype, pitch = True)
                    func.prepared_call(
                        self._grid, self._block, self.M, self.N,
                        result.gpudata, result.ld, self.gpudata,
                        self.ld, other.gpudata, other.ld)
            return result
        elif issubclass(type(other), (float, int, complex, np.integer,
                                      np.floating, np.complexfloating)):
            dtype = _get_common_dtype_with_scalar(other, self)
            if other == 1.0:
                return self.astype(dtype)
            else:
                result = self._new_like_me(dtype)
                if self.size:
                    if self.M == 1:
                        func = pu.get_mulscalar_function(
                            self.dtype, dtype, pitch = False)
                        func.prepared_call(
                            self._grid, self._block, result.gpudata,
                            self.gpudata, other, self.size)
                    else:
                        func = pu.get_mulscalar_function(
                            self.dtype, dtype, pitch = True)
                        func.prepared_call(
                            self._grid, self._block, self.M, self.N,
                            result.gpudata, result.ld, self.gpudata,
                            self.ld, other)
                return result
        else:
            raise TypeError("type of object to be multiplied "
                            "is not supported")
    
    __rmul__ = __mul__

    def __div__(self, other):
        """
        Division
        self / other
        
        Parameters
        ----------
        other: scalar or Pitcharray
        
        Returns
        -------
        out : PitchArray
            A new PitchArray
        """
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            result = self._new_like_me(_get_common_dtype(self, other))
            if self.size:
                if self.M == 1:
                    func = pu.get_divarray_function(
                        self.dtype, other.dtype, result.dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, result.gpudata,
                        self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_divarray_function(
                        self.dtype, other.dtype, result.dtype, pitch = True)
                    func.prepared_call(
                        self._grid, self._block, self.M, self.N,
                        result.gpudata, result.ld, self.gpudata,
                        self.ld, other.gpudata, other.ld)
            return result
        elif issubclass(type(other), (float, int, complex, np.integer,
                                      np.floating, np.complexfloating)):
            dtype = _get_common_dtype_with_scalar(other, self)
            if other == 1.0:
                return self.astype(dtype)
            else:
                result = self._new_like_me(dtype)
                if self.size:
                    if self.M == 1:
                        func = pu.get_divscalar_function(
                            self.dtype, dtype, pitch = False)
                        func.prepared_call(
                            self._grid, self._block, result.gpudata,
                            self.gpudata, other, self.size)
                    else:
                        func = pu.get_divscalar_function(
                            self.dtype, dtype, pitch = True)
                        func.prepared_call(
                            self._grid, self._block, self.M, self.N,
                            result.gpudata, result.ld, self.gpudata,
                            self.ld, other)
                return result
        else:
            raise TypeError("type of object to be divided "
                            "is not supported")

    def __rdiv__(self, other):
        """
        Being divided
        other / self
        
        Parameters
        ----------
        other: scalar or Pitcharray
        
        Returns
        -------
        out : PitchArray
            A new PitchArray
            
        """
        """
        # this part it not necessary
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            result = self._new_like_me(_get_common_dtype(self, other))
            if self.size:
                if self.M == 1:
                    func = pu.get_divarray_function(
                        other.dtype, self.dtype, result.dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, result.gpudata,
                        other.gpudata, self.gpudata, self.size)
                else:
                    func = pu.get_divarray_function(
                        other.dtype, self.dtype, result.dtype, pitch = True)
                    func.prepared_call(
                        self._grid, self._block, self.M, self.N,
                        result.gpudata, result.ld, other.gpudata,
                        other.ld, self.gpudata, self.ld)
            return result
        """
        if issubclass(type(other), (float, int, complex, np.integer,
                                      np.floating, np.complexfloating)):
            dtype = _get_common_dtype_with_scalar(other, self)
            result = self._new_like_me(dtype)
            if self.size:
                if self.M == 1:
                    func = pu.get_scalardiv_function(
                        self.dtype, dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, result.gpudata,
                        self.gpudata, other, self.size)
                else:
                    func = pu.get_scalardiv_function(
                        self.dtype, dtype, pitch = True)
                    func.prepared_call(
                        self._grid, self._block, self.M, self.N,
                        result.gpudata, result.ld, self.gpudata,
                        self.ld, other)
            return result
        else:
            raise TypeError("type of object to be divided from"
                            "is not supported")
    
    def __neg__(self):
        """
        Take negative value
        """
        return 0-self

    def __iadd__(self, other):
        """
        add to self inplace
        
        Parameters
        ----------
        other: scalar or Pitcharray
        
        Returns
        -------
        out : PitchArray (self)
        
        Note
        ----
        If other is complex, self is required to be a complex
        """
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            dtype = _get_common_dtype(self, other)
            if self.dtype == dtype:
                result = self
            else:
                result = self._new_like_me(dtype = dtype)
                
            if self.size:
                if self.M == 1:
                    func = pu.get_addarray_function(
                        self.dtype, other.dtype, result.dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, result.gpudata,
                        self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_addarray_function(
                        self.dtype, other.dtype, result.dtype, pitch = True)
                    func.prepared_call(self._grid, self._block,
                        self.M, self.N, result.gpudata, result.ld,
                        self.gpudata, self.ld, other.gpudata, other.ld)
            return result
        elif issubclass(type(other), (float, int, complex, np.integer,
                                      np.floating, np.complexfloating)):
            dtype = _get_common_dtype_with_scalar(other, self)
            if other == 0:
                return self.astype(dtype)
            else:
                if self.dtype != dtype:
                    result = self._new_like_me(dtype)
                else:
                    result = self
                if self.size:
                    if self.M == 1:
                        func = pu.get_addscalar_function(
                            self.dtype, dtype, pitch = False)
                        func.prepared_call(
                            self._grid, self._block, result.gpudata,
                            self.gpudata, other, self.size)
                    else:
                        func = pu.get_addscalar_function(
                            self.dtype, dtype, pitch = True)
                        func.prepared_call(
                            self._grid, self._block, self.M, self.N,
                            result.gpudata, self.ld, self.gpudata,
                            self.ld, other)
                return result
        else:
            raise TypeError("type of object to be added"
                            "is not supported")
    
    def __isub__(self, other):
        """
        Substracted other inplace
        
        Parameters
        ----------
        other: scalar or Pitcharray
        
        Returns
        -------
        out : PitchArray (self)
        """
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            dtype = _get_common_dtype(self, other)
            if self.dtype == dtype:
                result = self
            else:
                result = self._new_like_me(dtype = dtype)
                
            if self.size:
                if self.M == 1:
                    func = pu.get_subarray_function(
                        self.dtype, other.dtype, result.dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, result.gpudata,
                        self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_subarray_function(
                        self.dtype, other.dtype, result.dtype, pitch = True)
                    func.prepared_call(
                        self._grid, self._block, self.M, self.N,
                        result.gpudata, result.ld, self.gpudata,
                        self.ld, other.gpudata, other.ld)
            return result
        elif issubclass(type(other), (float, int, complex, np.integer,
                                      np.floating, np.complexfloating)):
            dtype = _get_common_dtype_with_scalar(other, self)
            if other == 0:
                return self.astype(dtype)
            else:
                if self.dtype != dtype:
                    result = self._new_like_me(dtype)
                else:
                    result = self
                if self.size:
                    if self.M == 1:
                        func = pu.get_subscalar_function(
                            self.dtype, dtype, pitch = False)
                        func.prepared_call(
                            self._grid, self._block, result.gpudata,
                            self.gpudata, other, self.size)
                    else:
                        func = pu.get_subscalar_function(
                            self.dtype, dtype, pitch = True)
                        func.prepared_call(
                            self._grid, self._block, self.M, self.N,
                            result.gpudata, self.ld, self.gpudata,
                            self.ld, other)
                return result
        else:
            raise TypeError("type of object to be substracted"
                            "is not supported")
    
    def __imul__(self, other):
        """
        Multiplied by other
        inplace if possible
        
        Parameters
        ----------
        other: scalar or Pitcharray
        
        Returns
        -------
        out : PitchArray (self)
        """
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            dtype = _get_common_dtype(self, other)
            if self.dtype == dtype:
                result = self
            else:
                result = self._new_like_me(dtype = dtype)
                
            if self.size:
                if self.M == 1:
                    func = pu.get_mularray_function(
                        self.dtype, other.dtype, result.dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, result.gpudata,
                        self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_mularray_function(
                        self.dtype, other.dtype, result.dtype, pitch = True)
                    func.prepared_call(
                        self._grid, self._block, self.M, self.N,
                        result.gpudata, result.ld, self.gpudata,
                        self.ld, other.gpudata, other.ld)
            return result
        elif issubclass(type(other), (float, int, complex, np.integer,
                                      np.floating, np.complexfloating)):
            dtype = _get_common_dtype_with_scalar(other, self)
            if other == 1.0:
                return self.astype(dtype)
            else:
                if self.dtype != dtype:
                    result = self._new_like_me(dtype)
                else:
                    result = self
                if self.size:
                    if self.M == 1:
                        func = pu.get_mulscalar_function(
                            self.dtype, dtype, pitch = False)
                        func.prepared_call(
                            self._grid, self._block, result.gpudata,
                            self.gpudata, other, self.size)
                    else:
                        func = pu.get_mulscalar_function(
                            self.dtype, dtype, pitch = True)
                        func.prepared_call(
                            self._grid, self._block, self.M, self.N,
                            result.gpudata, self.ld, self.gpudata,
                            self.ld, other)
                return result
        else:
            raise TypeError("type of object to be multiplied"
                            "is not supported")
    
    def __idiv__(self, other):
        """
        Divided by other
        inplace if possible
        
        Parameters
        ----------
        other: scalar or Pitcharray
        
        Returns
        -------
        out : PitchArray (self)
        """
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            dtype = _get_common_dtype(self, other)
            if self.dtype == dtype:
                result = self
            else:
                result = self._new_like_me(dtype = dtype)
                
            if self.size:
                if self.M == 1:
                    func = pu.get_divarray_function(
                        self.dtype, other.dtype, result.dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, result.gpudata,
                        self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_divarray_function(
                        self.dtype, other.dtype, result.dtype, pitch = True)
                    func.prepared_call(
                        self._grid, self._block, self.M, self.N,
                        result.gpudata, result.ld, self.gpudata,
                        self.ld, other.gpudata, other.ld)
            return result
        elif issubclass(type(other), (float, int, complex, np.integer,
                                      np.floating, np.complexfloating)):
            dtype = _get_common_dtype_with_scalar(other, self)
            if other == 1.0:
                return self.astype(dtype)
            else:
                if self.dtype != dtype:
                    result = self._new_like_me(dtype)
                else:
                    result = self
                if self.size:
                    if self.M == 1:
                        func = pu.get_divscalar_function(
                            self.dtype, dtype, pitch = False)
                        func.prepared_call(
                            self._grid, self._block, result.gpudata,
                            self.gpudata, other, self.size)
                    else:
                        func = pu.get_divscalar_function(
                            self.dtype, dtype, pitch = True)
                        func.prepared_call(
                            self._grid, self._block, self.M, self.N,
                            result.gpudata, self.ld, self.gpudata,
                            self.ld, other)
                return result
        else:
            raise TypeError("type of object to be divided"
                            "is not supported")

    def add(self, other):
        """
        add other to self
        inplace
        """
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            dtype = _get_inplace_dtype(self, other)
            if self.size:
                if self.M == 1:
                    func = pu.get_addarray_function(
                        self.dtype, other.dtype, self.dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, self.gpudata,
                        self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_addarray_function(
                        self.dtype, other.dtype, self.dtype, pitch = True)
                    func.prepared_call(self._grid, self._block,
                        self.M, self.N, self.gpudata, self.ld,
                        self.gpudata, self.ld, other.gpudata, other.ld)
            return self
        elif issubclass(type(other), (float, int, complex, np.integer,
                                      np.floating, np.complexfloating)):
            dtype = _get_inplace_dtype_with_scalar(other, self)
            if other != 0:
                if self.size:
                    if self.M == 1:
                        func = pu.get_addscalar_function(
                            self.dtype, self.dtype, pitch = False)
                        func.prepared_call(
                            self._grid, self._block, self.gpudata,
                            self.gpudata, other, self.size)
                    else:
                        func = pu.get_addscalar_function(
                            self.dtype, self.dtype, pitch = True)
                        func.prepared_call(
                            self._grid, self._block, self.M, self.N,
                            self.gpudata, self.ld, self.gpudata,
                            self.ld, other)
            return self
        else:
            raise TypeError("type of object to be added"
                            "is not supported")

    def sub(self, other):
        """
        substract other from self
        inplace
        """
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            dtype = _get_inplace_dtype(self, other)
            if self.size:
                if self.M == 1:
                    func = pu.get_subarray_function(
                        self.dtype, other.dtype, self.dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, self.gpudata,
                        self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_subarray_function(
                        self.dtype, other.dtype, self.dtype, pitch = True)
                    func.prepared_call(self._grid, self._block,
                        self.M, self.N, self.gpudata, self.ld,
                        self.gpudata, self.ld, other.gpudata, other.ld)
            return self
        elif issubclass(type(other), (float, int, complex, np.integer,
                                      np.floating, np.complexfloating)):
            dtype = _get_inplace_dtype_with_scalar(other, self)
            if other != 0:
                if self.size:
                    if self.M == 1:
                        func = pu.get_subscalar_function(
                            self.dtype, self.dtype, pitch = False)
                        func.prepared_call(
                            self._grid, self._block, self.gpudata,
                            self.gpudata, other, self.size)
                    else:
                        func = pu.get_subscalar_function(
                            self.dtype, self.dtype, pitch = True)
                        func.prepared_call(
                            self._grid, self._block, self.M, self.N,
                            self.gpudata, self.ld, self.gpudata,
                            self.ld, other)
            return self
        else:
            raise TypeError("type of object to be substracted"
                            "is not supported")

    def rsub(self, other):
        """
        substract other by self
        inplace
        """
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            dtype = _get_inplace_dtype(self, other)
            if self.size:
                if self.M == 1:
                    func = pu.get_subarray_function(
                        other.dtype, self.dtype, self.dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, self.gpudata,
                        other.gpudata, self.gpudata, self.size)
                else:
                    func = pu.get_subarray_function(
                        other.dtype, self.dtype, self.dtype, pitch = True)
                    func.prepared_call(
                        self._grid, self._block, self.M, self.N,
                        self.gpudata, self.ld, other.gpudata,
                        other.ld, self.gpudata, self.ld)
            return self
        elif issubclass(type(other), (float, int, complex, np.integer,
                                      np.floating, np.complexfloating)):
            dtype = _get_inplace_dtype_with_scalar(other, self)
            if self.size:
                if self.M == 1:
                    func = pu.get_scalarsub_function(
                        self.dtype, self.dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, self.gpudata,
                        self.gpudata, other, self.size)
                else:
                    func = pu.get_scalarsub_function(
                        self.dtype, self.dtype, pitch = True)
                    func.prepared_call(
                        self._grid, self._block, self.M, self.N,
                        self.gpudata, self.ld, self.gpudata,
                        self.ld, other)
            return self
        else:
            raise TypeError("type of object to substract from"
                            "is not supported")
    
    def mul(self, other):
        """
        multiply other with self
        inplace
        """
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            dtype = _get_inplace_dtype(self, other)
            if self.size:
                if self.M == 1:
                    func = pu.get_mularray_function(
                        self.dtype, other.dtype, self.dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, self.gpudata,
                        self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_mularray_function(
                        self.dtype, other.dtype, self.dtype, pitch = True)
                    func.prepared_call(self._grid, self._block,
                        self.M, self.N, self.gpudata, self.ld,
                        self.gpudata, self.ld, other.gpudata, other.ld)
            return self
        elif issubclass(type(other), (float, int, complex, np.integer,
                                      np.floating, np.complexfloating)):
            dtype = _get_inplace_dtype_with_scalar(other, self)
            if other != 1:
                if self.size:
                    if self.M == 1:
                        func = pu.get_mulscalar_function(
                            self.dtype, self.dtype, pitch = False)
                        func.prepared_call(
                            self._grid, self._block, self.gpudata,
                            self.gpudata, other, self.size)
                    else:
                        func = pu.get_mulscalar_function(
                            self.dtype, self.dtype, pitch = True)
                        func.prepared_call(
                            self._grid, self._block, self.M, self.N,
                            self.gpudata, self.ld, self.gpudata,
                            self.ld, other)
            return self
        else:
            raise TypeError("type of object to be multiplied"
                            "is not supported")

    def div(self, other):
        """
        divide other from self
        inplace
        """
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            dtype = _get_inplace_dtype(self, other)
            if self.size:
                if self.M == 1:
                    func = pu.get_divarray_function(
                        self.dtype, other.dtype, self.dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, self.gpudata,
                        self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_divarray_function(
                        self.dtype, other.dtype, self.dtype, pitch = True)
                    func.prepared_call(self._grid, self._block,
                        self.M, self.N, self.gpudata, self.ld,
                        self.gpudata, self.ld, other.gpudata, other.ld)
            return self
        elif issubclass(type(other), (float, int, complex, np.integer,
                                      np.floating, np.complexfloating)):
            dtype = _get_inplace_dtype_with_scalar(other, self)
            if other != 1:
                if self.size:
                    if self.M == 1:
                        func = pu.get_divscalar_function(
                            self.dtype, self.dtype, pitch = False)
                        func.prepared_call(
                            self._grid, self._block, self.gpudata,
                            self.gpudata, other, self.size)
                    else:
                        func = pu.get_divscalar_function(
                            self.dtype, self.dtype, pitch = True)
                        func.prepared_call(
                            self._grid, self._block, self.M, self.N,
                            self.gpudata, self.ld, self.gpudata,
                            self.ld, other)
            return self
        else:
            raise TypeError("type of object to be divided"
                            "is not supported")

    def rdiv(self, other):
        """
        divide other by self
        inplace
        """
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            dtype = _get_inplace_dtype(self, other)
            if self.size:
                if self.M == 1:
                    func = pu.get_divarray_function(
                        other.dtype, self.dtype, self.dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, self.gpudata,
                        other.gpudata, self.gpudata, self.size)
                else:
                    func = pu.get_divarray_function(
                        other.dtype, self.dtype, self.dtype, pitch = True)
                    func.prepared_call(
                        self._grid, self._block, self.M, self.N,
                        self.gpudata, self.ld, other.gpudata,
                        other.ld, self.gpudata, self.ld)
            return self
        elif issubclass(type(other), (float, int, complex, np.integer,
                                      np.floating, np.complexfloating)):
            dtype = _get_inplace_dtype_with_scalar(other, self)
            if self.size:
                if self.M == 1:
                    func = pu.get_scalardiv_function(
                        self.dtype, self.dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, self.gpudata,
                        self.gpudata, other, self.size)
                else:
                    func = pu.get_scalardiv_function(
                        self.dtype, dtype, pitch = True)
                    func.prepared_call(
                        self._grid, self._block, self.M, self.N,
                        self.gpudata, self.ld, self.gpudata,
                        self.ld, other)
            return self
        else:
            raise TypeError("type of object to divide from"
                            "is not supported")
    def neg(self):
        """
        Take the negative of self inplace
        
        Returns
        -------
        self
        """
        if self.size:
            if self.M == 1:
                func = pu.get_scalarsub_function(
                    self.dtype, self.dtype, pitch = False)
                func.prepared_call(
                    self._grid, self._block, self.gpudata,
                    self.gpudata, 0, self.size)
            else:
                func = pu.get_scalarsub_function(
                    self.dtype, self.dtype, pitch = True)
                func.prepared_call(
                    self._grid, self._block, self.M, self.N,
                    self.gpudata, self.ld, self.gpudata,
                    self.ld, 0)
        return self
    
    def fill(self, value, stream=None):
        """
        Fill all entries of self with value
        
        Parameters:
        ----------------------------------
        value: scalar
               Value to be filled
        stream: pycuda.driver.stream
                for asynchronous execution
        """
        if self.size:
            if self.M == 1:
                func = pu.get_fill_function(self.dtype, pitch = False)
                func.prepared_call(
                    self._grid, self._block, self.size, self.gpudata, value)
            else:
                func = pu.get_fill_function(
                    self.dtype, pitch = True)
                
                func.prepared_call(
                    self._grid, self._block, self.M, self.N,
                    self.gpudata, self.ld, value)
    
    def copy(self):
        """
        Returns a duplicated copy of self
        """
        result = self._new_like_me()
        if self.size:
            cuda.memcpy_dtod(
                result.gpudata, self.gpudata,
                self.mem_size * self.dtype.itemsize)
        return result
        
    def real(self):
        """
        Returns the real part of self
        """
        if self.dtype == np.complex128:
            result = self._new_like_me(dtype = np.float64)
        elif self.dtype == np.complex64:
            result = self._new_like_me(dtype = np.float32)
        else:
            return self
        
        if self.size:
            if self.M == 1:
                func = pu.get_realimag_function(
                    self.dtype, real = True, pitch = False)
                func.prepared_call(
                    self._grid, self._block, result.gpudata,
                    self.gpudata, self.size)
            else:
                func = pu.get_realimag_function(
                    self.dtype, real = True, pitch = True)
                func.prepared_call(
                    self._grid, self._block, self.M, self.N,
                    result.gpudata, result.ld, self.gpudata, self.ld)
        return result
    
    def imag(self):
        """
        returns the imaginary part of self
        """
        if self.dtype == np.complex128:
            result = self._new_like_me(dtype = np.float64)
        elif self.dtype == np.complex64:
            result = self._new_like_me(dtype = np.float32)
        else:
            return zeros_like(self)
        
        if self.size:
            if self.M == 1:
                func = pu.get_realimag_function(
                    self.dtype, real = False, pitch = False)
                func.prepared_call(
                    self._grid, self._block, result.gpudata,
                    self.gpudata, self.size)
            else:
                func = pu.get_realimag_function(
                    self.dtype, real = False, pitch = True)
                func.prepared_call(
                    self._grid, self._block, self.M, self.N,
                    result.gpudata, result.ld, self.gpudata, self.ld)
        return result
        
    def abs(self):
        """
        returns the absolute value of self
        """
        if self.dtype in [np.complex128, np.float64]:
            result = self._new_like_me(dtype = np.float64)
        elif self.dtype in [np.complex64, np.float32]:
            result = self._new_like_me(dtype = np.float32)
        else:
            result = self._new_like_me()
            
        if self.M == 1:
            func = pu.get_abs_function(self.dtype, pitch = False)
            func.prepared_call(
                self._grid, self._block, result.gpudata,
                self.gpudata, self.size)
        else:
            func = pu.get_abs_function(
                self.dtype, pitch = True)
            func.prepared_call(
                self._grid, self._block, self.M, self.N,
                result.gpudata, result.ld, self.gpudata, self.ld)
        return result
    
    def conj(self, inplace = True):
        """
        returns the conjuation of self.
        
        Paramters:
        ----------
        inplace: bool (optional)
            if inplace is True, conjugation will be performed in place
            (default: True)
        
        """
        if self.dtype in [np.complex64, np.complex128]:
            if inplace:
                result = self
            else:
                result = self._new_like_me()
            
            if self.M == 1:
                func = pu.get_conj_function(self.dtype, pitch = False)
                func.prepared_call(
                    self._grid, self._block, result.gpudata,
                    self.gpudata, self.size)
            else:
                func = pu.get_conj_function(
                    self.dtype, pitch = True)
                func.prepared_call(
                    self._grid, self._block, self.M, self.N,
                    result.gpudata, result.ld, self.gpudata, self.ld)
        else:
            result = self
        return result
    
    def reshape(self, shape, inplace = True):
        """
        reshape the shape of self to "shape"
        
        Paramters:
        ----------
        shape : tuple of ints
            the new shape to be reshaped to
        inplace : bool (optional)
            if True, enforce to keep the device memory of self if possible.
            If the above is not possible, e.g. the new shape requires larger
            memory size, or if inplace is False, return a new PitchArray.
            (default:  True)
        """
        
        sx = 1
        for dim in self.shape:
            sx *= dim
        
        s = 1
        flag = False
        n = 0
        axis = 0
        idx = -1
        for dim in shape:
            if dim == -1:
                flag = True
                n += 1
                idx = axis
            else:
                s *= dim
            axis += 1
        
        if flag:
            if n > 1:
                raise ValueError("can only specify one unknown dimension")
            else:
                if sx % s == 0:
                    shape = _assignshape(shape, idx, int(sx / s))
                else:
                    raise ValueError("cannot infer the size "
                                     "of the remaining axis")
        else:
            if s != sx:
                raise ValueError("total size of new array must be unchanged")

        if inplace:
            if shape[0] == self.shape[0]:
                #self.shape = shape
                return PitchArray(shape = shape,
                                  dtype = self.dtype,
                                  gpudata = int(self.gpudata),
                                  base = self)
            else:# The case when self is a vector
                if self.M == 1 and any(b == 1 for b in shape):
                    return PitchArray(shape = shape,
                                      dtype = self.dtype,
                                      gpudata = int(self.gpudata),
                                      base = self)
                else:
                    raise ValueError("cannot resize inplacely")

        
        result = PitchArray(shape, self.dtype)
        func = pu.get_resize_function(self.dtype)
        #func.set_block_shape(256,1,1)
        func.prepared_call(
            self._grid, (256,1,1), self.shape[0], _pd(self.shape),
            result.shape[0], _pd(result.shape), result.gpudata,
            result.ld, self.gpudata, self.ld)
        return result
    
    def astype(self, dtype):
        """
        Convert dtype of self to dtype 
        
        Parameters:
        -----------
        dtype: np.dtype
               dtype of the returned array
        
        """
        dtype = np.dtype(dtype)
        if self.dtype == dtype:
            return self.copy()
        else:
            result = self._new_like_me(dtype = dtype)
            
            if self.size:
                if self.M == 1:
                    func = pu.get_astype_function(
                        dtype, self.dtype, pitch = False)
                    func.prepared_call(
                        self._grid, self._block, result.gpudata,
                        self.gpudata, self.size)
                else:
                    func = pu.get_astype_function(
                        dtype, self.dtype, pitch = True)
                    func.prepared_call(
                        self._grid, self._block, self.M, self.N,
                        result.gpudata, result.ld, self.gpudata, self.ld)
            return result
    
    def T(self):
        """
        Returns the transpose
        PitchArray must be 2 dimensional
        """
        
        if len(self.shape) > 2:
            raise ValueError("transpose only apply to 2D matrix")
        
        shape_t = self.shape[::-1]
        
        if self.M == 1:
            result = self.copy()
            result.shape = shape_t
            result.ld = _pd(result.shape)
            return result

        result = PitchArray(shape_t, self.dtype)
        if self.size:
            func = pu.get_transpose_function(self.dtype)
            func.prepared_call(self._grid, self._block, self.shape[0],
                               self.shape[1], result.gpudata,
                               result.ld, self.gpudata, self.ld)
        return result
    
    def H(self):
        """
        Returns the conjugate transpose
        PitchArray must be 2 dimensional
        """
        if len(self.shape) > 2:
            raise ValueError("transpose only apply to 2D matrix")
        
        shape_t = self.shape[::-1]
        if self.M == 1:
            result = conj(self)
            result.shape = shape_t
            result.ld = _pd(result.shape)
            return result

        result = PitchArray(shape_t, self.dtype)

        if self.size:
            func = pu.get_transpose_function(self.dtype, conj = True)
            func.prepared_call(self._grid, self._block, self.shape[0],
                               self.shape[1], result.gpudata, result.ld,
                               self.gpudata, self.ld)
        return result
    
    def copy_rows(self, start, stop, step = 1):
        """
        Extract rows of self to form a new array.
        
        Parameters:
        -----------
        start: int
            first row to be extracted
        stop: int
            last row to be extracted
        step: int (optional, default: 1)
            extract every step row.
        
        """
        nrows = len(range(start,stop,step))
        if nrows:
            
            if self.ndim == 2:
                shape = (nrows, self.shape[1])
            else:
                shape = (nrows, self.shape[1], self.shape[2])
        else:
            if self.ndim == 2:
                shape = (nrows, 0)
            else:
                shape = (nrows, 0, 0)
        
        result = PitchArray(shape, self.dtype)
        
        if nrows > 1:
            PitchTrans(
                shape, result.gpudata, result.ld,
                int(self.gpudata) + start*self.ld*self.dtype.itemsize,
                self.ld * step, self.dtype)
        elif nrows == 1:
            cuda.memcpy_dtod(
                result.gpudata,
                int(int(self.gpudata) + start*self.ld*self.dtype.itemsize),
                self.dtype.itemsize * _pd(shape))
        return result
    
    def view(self, dtype = None):
        """
        New view of array with the same data (similar to numpy.ndarary.view)
        
        Optional Parameters:
        --------------------
        dtype : numpy.dtype
            Data-type descriptor of the returned view
        
        """
        
        if dtype is None:
            dtype = self.dtype
        old_itemsize = self.dtype.itemsize
        itemsize = np.dtype(dtype).itemsize
        
        if self.shape[-1] * old_itemsize % itemsize != 0:
            raise ValueError("new type not compatible with array")

        shape = self.shape[:-1] + (self.shape[-1] * old_itemsize // itemsize,)

        return PitchArray(shape=shape, dtype=dtype, gpudata=int(self.gpudata),
                          base=self)

    def __getitem__(self, idx):
        """
        only support slicing a chunk of consecutive rows
        """
        if idx == ():
            return self
        
        if isinstance(idx, slice):
            start = idx.start
            stop = idx.stop
            step = idx.step
            
            if start is None:
                start = 0
            if stop is None:
                stop = self.shape[0]
            if step is None:
                step = 1
            
            if step != 1:
                raise NotImplementedError("non-consecutive slicing is not "
                                          "implemented yet")
        elif isinstance(idx, tuple):
            if len(idx) > 1:
                axis = 1;
                for tmp in idx[1:]:
                    start = tmp.start
                    stop = tmp.stop
                    step = tmp.step
                    if start is None:
                        start = 0
                    if stop is None:
                        stop = self.shape[axis]
                    if step is None:
                        step = 1
                    
                    if (start != 0 or stop != self.shape[axis] or 
                        step != 1):
                        if self.M != 1:
                            raise NotImplementedError(
                            "slicing only supported on axis = 0")
                    axis += 1
                
                start = idx[0].start
                stop = idx[0].stop
                step = idx[0].step
                if start is None and stop is None and step is None:
                    if self.M == 1:
                        if len(idx) == 2:
                            start = idx[1].start
                            stop = idx[1].stop
                            step = idx[1].step
                if start is None:
                    start = 0
                if stop is None:
                    stop = self.shape[0]
                if step is None:
                    step = 1
                
                if step != 1:
                    raise NotImplementedError("non-consecutive slicing is "
                                              "not implemented yet")
        elif isinstance(idx, int):
            start = idx
            stop = idx + 1
            step = 1
        else:
            raise ValueError("non-slice indexing not supported: %s" % (idx,))
        
        maxshape = self.size if self.M == 1 else self.shape[0]
        if stop > maxshape:
            stop = maxshape
            from warnings import warn
            warn("array slicing larger than array size, "
                 "reduce to allowed size")

        if self.M == 1:
            if self.shape[0] == 1:
                shape = (1,stop-start)
            else:
                shape = (stop-start,1)
            return PitchArray(shape=shape, dtype = self.dtype,
                              gpudata = int(int(self.gpudata) +
                                    start*self.dtype.itemsize),
                              base = self)
        else:
            if self.ndim == 2:
                return PitchArray(shape=(stop-start,self.shape[1]),
                                  dtype = self.dtype,
                                  gpudata = int(int(self.gpudata) +
                                        start*self.dtype.itemsize*self.ld),
                                  base = self)
            else:
                return PitchArray(shape=(stop-start,self.shape[1],
                                          self.shape[2]),
                                  dtype = self.dtype,
                                  gpudata = int(int(self.gpudata) +
                                        start*self.dtype.itemsize*self.ld),
                                  base = self)
        

def to_gpu(ary):
    """
    Transfer a numpy ndarray to a PitchArray
    """
    result = PitchArray(ary.shape, ary.dtype)
    result.set(ary)
    return result


def to_gpu_async(ary, stream = None):
    """
    Transfer a pagelocked numpy ndarray to a PitchArray
    """
    result = PitchArray(ary.shape, ary.dtype)
    result.set_async(ary, stream)


empty = PitchArray


def empty_like(other_ary):
    """
    Create an empty PitchArray, whose shape
    and dtype is the same as other_ary
    """
    result = PitchArray(other_ary.shape, other_ary.dtype)
    return result


def zeros(shape, dtype):
    """
    Create a PitchArray with all entry equal 0
    
    Parameter:
    -------------------------------
    shape: tuple or list of int
           Shape of the new array
    dtype: np.dtype
           dtype of the new array
    """
    result = PitchArray(shape, dtype)
    result.fill(0)
    return result

def zeros_like(other_ary):
    """
    Create a PitchArray with all entry equal 0, whose shape
    and dtype is the same as other_ary
    """
    result = PitchArray(other_ary.shape, other_ary.dtype)
    result.fill(0)
    return result


def ones(shape, dtype):
    """
    Create a PitchArray with all entry equal 1
    
    Parameter:
    -------------------------------
    shape: tuple or list of int
           Shape of the new array
    dtype: np.dtype
           dtype of the new array
    """
    result = PitchArray(shape, dtype)
    result.fill(1)
    return result

def ones_like(other_ary):
    """
    Create a PitchArray with all entry equal 1, whose shape
    and dtype is the same as other_ary
    """
    result = PitchArray(other_ary.shape, other_ary.dtype)
    result.fill(1)
    return result    

    
def make_pitcharray(dptr, shape, dtype, linear = False, pitch=None):
    """
    Create a PitchArray from a DeviceAllocation pointer
    
    Parameters:
    -----------
    dptr : pycuda.driver.DeviceAllocation
        pointer to the device memory
    shape : tuple of ints
        shape of the array
    dtype : numpy.dtype
        data type of the array
    linear : bool (optional, default: False)
        "True" indicates the device memory is a linearly allocated
        "False" indicates the device memory is allocated by
        cudaMallocPitch, and pitch must be provided
    pitch : int
        pitch of the array
    
    """
    if linear:
        result = PitchArray(shape, dtype)
        if result.size:
            if result.M == 1:
                cuda.memcpy_dtod(
                    result.gpudata, dptr, result.mem_size * dtype.itemsize)
            else:
                PitchTrans(
                    shape, result.gpudata, result.ld, dptr, _pd(shape), dtype)
    else:
        result = PitchArray(shape, dtype, gpudata=dptr, pitch = pitch)
    return result


def arrayg2p(other_gpuarray):
    """
    Convert a GPUArray to a PitchArray
    
    Parameter:
    ---------------------------------
    other_gpuarray : pycuda.GPUArray
    
    """
    result = make_pitcharray(
        other_gpuarray.gpudata, other_gpuarray.shape,
        other_gpuarray.dtype, linear = True)
    return result


def arrayp2g(pary):
    """
    Convert a PitchArray to a GPUArray
    
    Parameter:
    ----------
    pary: PitchArray
    
    """
    from pycuda.gpuarray import GPUArray
    result = GPUArray(pary.shape, pary.dtype)
    if pary.size:
        if pary.M == 1:
            cuda.memcpy_dtod(
                result.gpudata, pary.gpudata,
                pary.mem_size * pary.dtype.itemsize)
        else:
            PitchTrans(pary.shape, result.gpudata, _pd(result.shape),
                       pary.gpudata, pary.ld, pary.dtype)
    return result


def conj(pary):
    """
    Returns the conjugation of 2D PitchArray
    same as PitchArray.conj(), but create a new copy
    
    Parameter:
    ----------
    pary: PitchArray
    
    """
    return pary.conj(inplace = False)


def reshape(pary, shape):
    """
    Returns reshaped of 2D PitchArray
    same as PitchArray.reshape(), but always create a new copy
    
    Parameter:
    ----------
    pary: PitchArray
    
    """
    return pary.reshape(shape, inplace = False)


def iscomplexobj(pary):
    """ Check if the array dtype is complex """
    return issubclass(pary.dtype.type, np.complexfloating)


def isrealobj(pary):
    """ Check if the array dtype is real """
    return not iscomplexobj(pary)


def issingle(dtype):
    """ Check if dtype is single floating point """
    if dtype in [np.float32, np.complex64]:
        return True
    elif dtype in [np.float64, np.complex128]:
        return False
    else:
        raise TypeError("input dtype " + str(dtype) +
                        "not understood")
        

def floattocomplex(dtype):
    """ Conver dtype from real to corresponding complex dtype """
    dtype = dtype.type if isinstance(dtype, np.dtype) else dtype
    if issubclass(dtype, np.complexfloating):
        outdtype = dtype
    elif dtype == np.float32:
        outdtype = np.complex64
    elif dtype == np.float64:
        outdtype = np.complex128
    else:
        raise TypeError("input dtype " + str(dtype) +
                        " cannot be translated to complex floating")
    return np.dtype(outdtype)


def complextofloat(dtype):
    """ convert dtype from complex to corresponding real dtype """
    dtype = dtype.type if isinstance(dtype, np.dtype) else dtype
    if not issubclass(dtype, np.complexfloating):
        outdtype = dtype
    elif dtype == np.complex64:
        outdtype = np.float32
    elif dtype == np.complex128:
        outdtype = np.float64
    else:
        raise TypeError("input dtype " + str(dtype) +
                        " cannot be translated to floating")
    return np.dtype(outdtype)

def make_complex(real, imag):
    """
    Create a complex array using two real arrays
    
    Parameters
    ---------------------------------
    real: PitchArray
          The real part
    imag: PitchArray
          The imaginary part
          shape of real and imag must be the same
    
    Returns
    ---------------------------------
    out: PitchArray
         The complex array
         
    """
    if isinstance(real, np.ndarray):
        real = to_gpu(real)
    if isinstance(imag, np.ndarray):
        imag = to_gpu(imag)
    
    if real.shape != imag.shape:
        raise ValueError("real and imaginary parts must have the same shape")
    if iscomplexobj(real) or iscomplexobj(imag):
        raise TypeError("real and imaginary parts must be real array")
    dtype = _get_common_dtype(real, imag)
    if dtype in [np.int32, np.float32]:
        dtype = np.dtype(np.complex64)
    elif dtype in [np.int64, np.float64]:
        dtype = np.dtype(np.complex128)
    else:
        dtype = np.dtype(np.complex64)
    
    result = empty(real.shape, dtype)
    if result.size:
        if result.M == 1:
            func = pu.get_complex_function(
                real.dtype, imag.dtype, dtype, pitch = False)
            func.prepared_call(
                result._grid, result._block, result.gpudata,
                real.gpudata, imag.gpudata, result.size)
        else:
            func = pu.get_complex_function(
                real.dtype, imag.dtype, dtype)
            func.prepared_call(
                result._grid, result._block, result.M, result.N,
                result.gpudata, result.ld, real.gpudata, imag.gpudata,
                real.ld)
    return result

def angle(array):
    """ Returns the angle of each element in a complex array """
    if isrealobj(array):
        if issingle(array.dtype):
            return parray.zeros(array.shape, np.float32)
        else:
            return parray.zeros(array.shape, np.double)
    else:
        if issingle(array.dtype):
            result = empty(array.shape, dtype = np.float32)
        else:
            result = empty(array.shape, dtype = np.double)
            
        if array.M == 1:
            func = pu.get_angle_function(array.dtype, result.dtype, pitch = False)
            func.prepared_call(
                array._grid, array._block, result.gpudata,
                array.gpudata, array.size)
        else:
            func = pu.get_angle_function(
                array.dtype, result.dtype, pitch = True)
            func.prepared_call(
                array._grid, array._block, array.M, array.N,
                result.gpudata, result.ld, array.gpudata, array.ld)
        return result
