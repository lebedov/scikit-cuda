#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
from pycuda.tools import dtype_to_ctype
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import scikits.cuda.cufft as cufft

import ypcutil.parray as parray

class fftplan(object):
    """
    This class is to facilitate taking fft
    of the same type for multiple times 
    """
    def __init__(self, shape, dtype, idist, odist,
                 forward = True, econ = False, batch_size = 1,
                 inembed = None, onembed = None):
        """
        Initialize fft plan.
        
        Parameters
        ----------
        shape : int or tuple or list
            The size of the fft in each dimension.
            The last dimension will be the leading dimension in the arrays.
        dtype : numpy.dtype
            The dtype of input array
        idist : int
            The amount of elements in memory between
            consecutive batches in the input data.
        odist : int
            The amount of elements in memory between
            consecutive batches in the output data.
        forward : bool, optional
            True if forward fft
            False if inverse fft
        econ : bool, optional
            For foward transform, when input is real,
            whether the output is stored in econ fashion,
            i.e. only half of the fft results will be returned.
            For inverse transform,
            whether the input is stored in econ fashion,
            if True, output ifft result will be real.
        batch_size : int
            The amount of batches in one transform call.
        inembed : list or tuple of int, optional
            Specifing the number of elements in each dimension
            in the input array.
        onembed : list or tuple of int, optional
            Specifing the number of elements in each dimension
            in the output array.
        See CUFFT user guide
        """
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.ndim = len(self.shape)
        self.dtype = dtype.type if isinstance(dtype, np.dtype) else dtype
        self.forward = forward
        self.idist = idist
        self.odist = odist
        self.econ = econ
        self.batch_size = batch_size
        (self.intype, self.outtype,
         self.ffttype, self.fftfunc, self.fftdir) = self.gettypes()
        self.setup_dim(inembed, onembed)
        self.create_plan(self.batch_size)
        
    def transform(self, d_in, d_out):
        """
        Perform fft or ifft transform on d_in, 
        and store the result in d_out,
        according to specified plan.
        
        Parameters
        ----------
        d_in : parray.PitchArray
            The input array.
        d_out : parray.PitchArray
            The output array.
        """
        assert d_in.dtype == self.intype
        assert d_out.dtype == self.outtype
        # TODO: check if d_in and d_out is the same,
        # if they are the same, check if idist, is larger than prod(shape)
        # to avoid error
        if int(d_in.gpudata) == int(d_out.gpudata):
            from warnings import warn
            warn("Doing FFT inplace, make sure you know what you are doing.")
        
        if self.fftdir is None:
            self.fftfunc(self.plan, int(d_in.gpudata), int(d_out.gpudata))
        else:
            self.fftfunc(self.plan, int(d_in.gpudata),
                         int(d_out.gpudata), self.fftdir)
    
    def transform_p(self, d_in, d_out):
        if self.fftdir is None:
            self.fftfunc(self.plan, d_in, d_out)
        else:
            self.fftfunc(self.plan, d_in, d_out, self.fftdir)
    
    def __del__(self):
        if self.planned:
            self.destroy_plan()
        
    def setup_dim(self, inembed = None, onembed = None):
        """
        Setup the dimensions of the fft.
        
        Parameters
        ----------
        inembed : list or tuple of int, optional
            Specifing the number of elements in each dimension
            in the input array.
        onembed : list or tuple of int, optional
            Specifing the number of elements in each dimension
            in the output array.
        
        See CUFFT user guide
        """
        self.n = np.asarray(self.shape ,np.int32)
        
        if inembed is None:
            inembed = np.asarray(self.shape, np.int32)
            if self.econ and not self.forward:
                inembed[-1] = inembed[-1]/2+1
        else:
            if len(inembed) != self.ndim:
                raise ValueError("size of inembed speicified not correct")
            else:
                inembed = np.asarray(inembed, np.int32)
        
        if onembed is None:
            onembed = np.asarray(self.shape, np.int32)
            if self.econ and self.forward:
                onembed[-1] = onembed[-1]/2+1
        else:
            if len(onembed) != self.ndim:
                raise ValueError("size of onembed speicified not correct")
            else:
                onembed = np.asarray(onembed, np.int32)
        
        self.inembed = inembed
        self.onembed = onembed
        
    def destroy_plan(self):
        """
        Destroy the plan.
        """
        cufft.cufftDestroy(self.plan)
        self.planned = False

    def create_plan(self, batch_size):
        """
        Create the fft plan, with specified batch_size.
        
        Parameters
        ----------
        batch_size : int
            The number of batches to be transformed together.
        """
        self.plan = cufft.cufftPlanMany(
            self.ndim, self.n.ctypes.data,
            self.inembed.ctypes.data, 1, self.idist,
            self.onembed.ctypes.data, 1, self.odist,
            self.ffttype, batch_size)
        self.planned = True

    def gettypes(self):
        """
        Obtain the input, output array type,
        and the fft type, function and direction
        according to the parameters set in initialization
        
        Returns
        -------
        intype : numpy.dtype
            Dtype of the input array.
        outtype : numpy.dtype
            Dtype of the output array.
        ffttype : cufft transform types
        fftfunc : function
            The cufftExecx2x function to be executed in transform.
        fftdir : The direction of fft
            None is either input or output is real (can be inferred).
            Otherwise, the direction of complex to complex transform.
        """
        dtype = self.dtype
        forward = self.forward
        econ = self.econ
        single = parray.issingle(dtype)
        if issubclass(dtype, np.complexfloating):
            intype = dtype
            if forward:
                outtype = dtype
                if single:
                    ffttype = cufft.CUFFT_C2C
                    fftfunc = cufft.cufftExecC2C
                else:
                    ffttype = cufft.CUFFT_Z2Z
                    fftfunc = cufft.cufftExecZ2Z
                fftdir = cufft.CUFFT_FORWARD
            else:
                if econ:
                    outtype = parray.complextofloat(dtype)
                    if single:
                        ffttype = cufft.CUFFT_C2R
                        fftfunc = cufft.cufftExecC2R
                    else:
                        ffttype = cufft.CUFFT_Z2D
                        fftfunc = cufft.cufftExecZ2D
                    fftdir = None
                else:
                    outtype = dtype
                    if single:
                        ffttype = cufft.CUFFT_C2C
                        fftfunc = cufft.cufftExecC2C
                    else:
                        ffttype = cufft.CUFFT_Z2Z
                        fftfunc = cufft.cufftExecZ2Z
                    fftdir = cufft.CUFFT_INVERSE
        else:
            intype = dtype
            if not forward:
                forward = True
                from warnings import warn
                warn("real input will be forward transform by default")
            outtype = parray.floattocomplex(dtype)
            if single:
                ffttype = cufft.CUFFT_R2C
                fftfunc = cufft.cufftExecR2C
            else:
                ffttype = cufft.CUFFT_D2Z
                fftfunc = cufft.cufftExecD2Z
            fftdir = None
        return intype, outtype, ffttype, fftfunc, fftdir
        

class function_holder(object):
    def __init__(self):
        pass


_kernels = function_holder()


def fft(d_A, econ = False):
    """
    Perform 1D fft on each row of d_A
    can accept only 2D array
    
    Parameters
    ----------
    d_A : pycuda.gpuarray.GPUArray or parray.PitchArray
        Input array, complex or real
    econ : bool, optional
        Only applies when d_A is real
        If True, the output only contains half of the fft result,
        the other half can be inferred.
    
    Returns
    -------
    out : same type as d_A
        Each row containing the fft of corresponding to the input row.
    
    Note: 1) for batch job when size = d_A.ld-1, the result is incorrect
            due to cufft bug. Bug reported.
          2) The precision when econ = True when using GPUArray is
            slightly worse than using PitchArray.
    """
    if type(d_A) is parray.PitchArray:
        return _fft_parray(d_A, econ)
    elif type(d_A) is gpuarray.GPUArray:
        return _fft_gpuarray(d_A, econ)
    else:
        raise TypeError("FFT: Only PitchArray and GPUArray are supported")

def _fft_gpuarray(d_A, econ = False):
    """
    Perform 1D fft on each row of d_A
    can accept only 2D array
    
    Parameters
    ----------
    d_A : pycuda.gpuarray.GPUArray
        Input array, complex or real
    econ : bool, optional
        Only applies when d_A is real
        If True, the output only contains half of the fft result,
        the other half can be inferred.
    
    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        Each row containing the fft of corresponding to the input row.
        
    Note: The precision when econ = True is slightly worse than
        using PitchArray.
    """
    ndim = len(d_A.shape)
    reshaped = False
    A = d_A
    if ndim == 1:
        total_inputs = 1
        size = A.shape[0]
    elif ndim == 2:
        if any([b == 1 for b in A.shape]):
            total_inputs = 1
            size = max(A.shape)
            if A.shape[1] == 1:
                A = d_A.reshape(1, size)
                reshaped = True
        else:
            total_inputs = A.shape[0]
            size = A.shape[1]
    else:
        raise ValueError(
            "FFT: Only 1D and 2D array is supported for GPUArray")
    
    realA = parray.isrealobj(A)
    
    if econ and not realA:
        print ("Warning ypcutil.fft.fft: "
               "requested econ outputs, but getting complex inputs. "
               "econ is neglected")
    outdtype = parray.floattocomplex(A.dtype)
    d_output = gpuarray.empty(
        (total_inputs, size/2+1 if econ and realA else size),
        outdtype)
    
    batch_size = min(total_inputs, 128)
    plan = fftplan(size, A.dtype, size, d_output.shape[1],
                   forward = True, econ = realA,
                   batch_size = batch_size)
    for i in range(0, total_inputs, batch_size):
        ntransform = min(batch_size, total_inputs-i)
        if ntransform != batch_size:
            del plan
            plan = fftplan(size, A.dtype, size, d_output.shape[1],
                           forward = True, econ = realA,
                           batch_size = ntransform)
        plan.transform_p(int(A.gpudata) + i*size*A.dtype.itemsize,
                         int(d_output.gpudata) + 
                         i*outdtype.itemsize*d_output.shape[1])
    del plan
    if realA and not econ:
        global _kernels
        id = cuda.Context.get_device().PCI_BUS_ID
        func_name = 'get_1d'+'_'+str(id)+'_'+outdtype.name
        if hasattr(_kernels, func_name):
            pad_func = getattr(_kernels, func_name)
        else:
            pad_func = _get_1d_pad_func(outdtype)
            setattr(_kernels, func_name, pad_func)
        pad_func.prepared_call(
            (6*cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1),
            (256, 1, 1), d_output.gpudata, d_output.shape[1],
            size, total_inputs)
    
    if ndim == 1:
        return d_output.reshape(d_output.size)
    else:
        return d_output.reshape((d_output.size,1)) if reshaped else d_output



def _fft_parray(d_A, econ = False):
    """
    Perform 1D fft on each row of d_A
    can accept only 2D array
    
    Parameters
    ----------
    d_A : parray.PitchArray
        Input array, complex or real
    econ : bool, optional
        Only applies when d_A is real
        If True, the output only contains half of the fft result,
        the other half can be inferred.
    
    Returns
    -------
    out : parray.PitchArray, complex
        Each row containing the fft of corresponding to the input row.
        
    Note: for batch job when size = d_A.ld-1, the result is incorrect
          due to cufft bug. Bug reported.
    """
    assert len(d_A.shape) <= 2
    A = d_A
    reshaped = False
    if any([b == 1 for b in A.shape]):
        total_inputs = 1
        size = max(A.shape)
        if A.shape[1] == 1:
            A = d_A.reshape((1, size))
            reshaped = True
    else:
        total_inputs = A.shape[0]
        size = A.shape[1]
    realA = parray.isrealobj(A)
    
    if econ and not realA:
        print ("Warning ypcutil.fft.fft: "
               "requested econ outputs, but getting complex inputs. "
               "econ is neglected")
    outdtype = parray.floattocomplex(A.dtype)
    d_output = parray.empty(
        (total_inputs, size/2+1 if econ and realA else size),
        outdtype)
    
    batch_size = min(total_inputs, 128)
    plan = fftplan(size, A.dtype, A.ld, d_output.ld,
                   forward = True, econ = realA,
                   batch_size = batch_size)
    for i in range(0, total_inputs, batch_size):
        ntransform = min(batch_size, total_inputs-i)
        if ntransform != batch_size:
            del plan
            plan = fftplan(size, A.dtype, A.ld, d_output.ld,
                           forward = True, econ = realA,
                           batch_size = ntransform)
        plan.transform(A[i:i+ntransform], d_output[i:i+ntransform])
    del plan
    if realA and not econ:
        global _kernels
        id = cuda.Context.get_device().PCI_BUS_ID
        func_name = 'get_1d'+'_'+str(id)+'_'+outdtype.name
        if hasattr(_kernels, func_name):
            pad_func = getattr(_kernels, func_name)
        else:
            pad_func = _get_1d_pad_func(outdtype)
            setattr(_kernels, func_name, pad_func)
        pad_func.prepared_call(
            (6*cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1),
            (256, 1, 1), d_output.gpudata, d_output.ld,
            size, total_inputs)
    return d_output.reshape((d_A.shape[0],1)) if reshaped else d_output


def ifft(d_A, econ = False, even_size = None,
         scale = True, scalevalue = None):
    """
    Perform 1D inverse fft on each row of d_A
    can accept only 2D array, may be vector
    
    Parameters
    ----------
    d_A : parray.PitchArray or pycuda.gpuarray.GPUArray
        Input array, complex
    econ : bool, optional
        Whether the dft is stored in econ fashion in d_A.
    even_size : bool or None, optional
        Only effects when econ is True.
        If None, the size of fft is inferred.
        from element in d_A if d_A is real.
        If True, size of fft is even, else odd.
    scale : bool, optional
        Whether to scale the ifft to true ifft results.
        If false, the returned ifft is not normalize by N,
        where N is the size of ifft.
        If True, the ifft is normalized to be the true idft.
    scalevalue : float or None, optioinal
        Only takes effect when scale if True.
        If None, scale to the default size 1/N.
        If float, scale by value float.
        
    Returns
    -------
    out : same type as dd_A
        If econ is True, returns real array
        Otherwise, returns complex array.
    """
    if type(d_A) is parray.PitchArray:
        return _ifft_parray(d_A, econ = econ, even_size = even_size,
                            scale = scale, scalevalue = scalevalue)
    elif type(d_A) is gpuarray.GPUArray:
        return _ifft_gpuarray(d_A, econ = econ, even_size = even_size,
                              scale = scale, scalevalue = scalevalue)
    else:
        raise TypeError("FFT: Only PitchArray and GPUArray are supported")


def _ifft_parray(d_A, econ = False, even_size = None,
         scale = True, scalevalue = None):
    """
    Perform 1D inverse fft on each row of d_A
    can accept only 2D array, may be vector
    
    Parameters
    ----------
    d_A : parray.PitchArray
        Input array, complex
    econ : bool, optional
        Whether the dft is stored in econ fashion in d_A.
    even_size : bool or None, optional
        Only effects when econ is True.
        If None, the size of fft is inferred.
        from element in d_A if d_A is real.
        If True, size of fft is even, else odd.
    scale : bool, optional
        Whether to scale the ifft to true ifft results.
        If false, the returned ifft is not normalize by N,
        where N is the size of ifft.
        If True, the ifft is normalized to be the true idft.
    scalevalue : float or None, optioinal
        Only takes effect when scale if True.
        If None, scale to the default size 1/N.
        If float, scale by value float.
        
    Returns
    -------
    out : parray.PitchArray
        If econ is True, returns real array
        Otherwise, returns complex array.
    """
    assert len(d_A.shape) <= 2
    A = d_A
    reshaped = False
    if any([b == 1 for b in A.shape]):
        total_inputs = 1
        if econ:
            if even_size is None:
                even_size = _check_even_econ_1d(A, max(A.shape))
            size = (max(A.shape)-1)*2 if even_size else (max(A.shape)-1)*2+1
        else:
            size = max(A.shape)
        if A.shape[1] == 1:
            A = d_A.reshape((1, max(A.shape)))
            reshaped = True
    else:
        total_inputs = A.shape[0]
        if econ:
            if even_size is None:
                even_size = _check_even_econ_1d(A, A.shape[1])
            size = (A.shape[1]-1)*2 if even_size else (A.shape[1]-1)*2+1
        else:
            size = A.shape[1]
    
    outdtype = parray.complextofloat(A.dtype) if econ else A.dtype        
    d_output = parray.empty((total_inputs, size), outdtype)
    
    batch_size = min(total_inputs, 128)
    # Even for vectors d_output is alway of shape (1, size),
    # so d_output.ld should be correct
    plan = fftplan(size, A.dtype, A.ld, d_output.ld,
                 forward = False, econ = econ,
                 batch_size = batch_size)
    for i in range(0, total_inputs, batch_size):
        ntransform = min(batch_size, total_inputs-i)
        if ntransform != batch_size:
            del plan
            plan = fftplan(size, A.dtype, A.ld, d_output.ld,
                         forward = False, econ = econ,
                         batch_size = ntransform)
        plan.transform(A[i:i+ntransform], d_output[i:i+ntransform])
    del plan
    
    if scale:
        if scalevalue is None:
            scalevalue = 1./size
        d_output *= scalevalue
    return d_output.reshape(d_A.shape) if reshaped else d_output


def _ifft_gpuarray(d_A, econ = False, even_size = None,
         scale = True, scalevalue = None):
    """
    Perform 1D inverse fft on each row of d_A
    can accept only 2D array, may be vector
    
    Parameters
    ----------
    d_A : pycuda.gpuarray.GPUArray
        Input array, complex
    econ : bool, optional
        Whether the dft is stored in econ fashion in d_A.
    even_size : bool or None, optional
        Only effects when econ is True.
        If None, the size of fft is inferred.
        from element in d_A if d_A is real.
        If True, size of fft is even, else odd.
    scale : bool, optional
        Whether to scale the ifft to true ifft results.
        If false, the returned ifft is not normalize by N,
        where N is the size of ifft.
        If True, the ifft is normalized to be the true idft.
    scalevalue : float or None, optioinal
        Only takes effect when scale if True.
        If None, scale to the default size 1/N.
        If float, scale by value float.
        
    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        If econ is True, returns real array
        Otherwise, returns complex array.
    """
    ndim = len(d_A.shape)
    reshaped = False
    A = d_A
    if ndim == 1:
        total_inputs = 1
        size = A.shape[0]
    elif ndim == 2:
        if any([b == 1 for b in A.shape]):
            total_inputs = 1
            if econ:
                if even_size is None:
                    even_size = _check_even_econ_1d(A, max(A.shape))
                size = ((max(A.shape)-1)*2 if
                        even_size else (max(A.shape)-1)*2+1)
            else:
                size = max(A.shape)
            if A.shape[1] == 1:
                A = d_A.reshape(1, size)
                reshaped = True
        else:
            total_inputs = A.shape[0]
            if econ:
                if even_size is None:
                    even_size = _check_even_econ_1d(A, A.shape[1])
                size = ((A.shape[1]-1)*2 if 
                        even_size else (A.shape[1]-1)*2+1)
            else:
                size = A.shape[1]
    else:
        raise ValueError(
            "FFT: Only 1D and 2D array is supported for GPUArray")
    
    outdtype = parray.complextofloat(A.dtype) if econ else A.dtype        
    d_output = gpuarray.empty((total_inputs, size), outdtype)
    
    batch_size = min(total_inputs, 128)
    # Even for vectors d_output is alway of shape (1, size),
    # so d_output.ld should be correct
    plan = fftplan(size, A.dtype, A.shape[1], d_output.shape[1],
                 forward = False, econ = econ,
                 batch_size = batch_size)
    for i in range(0, total_inputs, batch_size):
        ntransform = min(batch_size, total_inputs-i)
        if ntransform != batch_size:
            del plan
            plan = fftplan(size, A.dtype, A.shape[1], d_output.shape[1],
                         forward = False, econ = econ,
                         batch_size = ntransform)
        plan.transform_p(int(A.gpudata) + i*A.shape[1]*A.dtype.itemsize,
                         int(d_output.gpudata) + 
                         i*outdtype.itemsize*d_output.shape[1])
    del plan
    if scale:
        if scalevalue is None:
            scalevalue = 1./size
        d_output *= scalevalue
    if ndim == 1:
        return d_output.reshape(d_output.size)
    else:
        return d_output.reshape((d_output.size,1)) if reshaped else d_output


def fft2(d_A, econ = False):
    """
    Perform 2D fft on the last two axis of d_A
    can accept only 2D or 3D array
    
    Parameters
    ----------
    d_A : parray.PitchArray or pycuda.gpuarray.GPUArray
        Input array, complex or real
    econ : bool, optional
        Only applies when d_A is real
        If True, the output only contains half of the fft result,
        the other half can be inferred.
    
    Returns
    -------
    out : smae type as d_A, complex
        Containing the fft of corresponding to the inputs.
    """
    if type(d_A) is parray.PitchArray:
        return _fft2_parray(d_A, econ)
    elif type(d_A) is gpuarray.GPUArray:
        return _fft2_gpuarray(d_A, econ)
    else:
        raise TypeError("FFT2: Only PitchArray and GPUArray are supported")


def _fft2_parray(d_A, econ = False):
    """
    Perform 2D fft on the last two axis of d_A
    can accept only 2D or 3D array
    
    Parameters
    ----------
    d_A : parray.PitchArray
        Input array, complex or real
    econ : bool, optional
        Only applies when d_A is real
        If True, the output only contains half of the fft result,
        the other half can be inferred.
    
    Returns
    -------
    out : parray.PitchArray, complex
        Containing the fft of corresponding to the inputs.
    
    Note: content destroyed when econ = True
    """
    ndim = len(d_A.shape)
    if ndim == 2:
        total_inputs = 1
        size = d_A.shape
    elif ndim == 3:
        total_inputs = d_A.shape[0]
        size = d_A.shape[1:3]
    else:
        raise ValueError("Input to fft2 must be of 2D or 3D")
    realA = parray.isrealobj(d_A)
    
    if econ and not realA:
        print ("Warning ypcutil.fft.fft2: "
               "requested econ outputs, but getting complex inputs. "
               "econ is neglected")
    
    outdtype = parray.floattocomplex(d_A.dtype)
    outshape = [b for b in d_A.shape]
    if econ and realA:
        outshape[-1] = outshape[-1]/2+1
    d_output = parray.empty(outshape, outdtype)
    batch_size = min(total_inputs, 8)
    
    plan = fftplan(
        size, d_A.dtype, d_A.ld, d_output.ld, forward = True, econ = realA,
        batch_size = batch_size, 
        inembed = (d_A.shape[0], d_A.ld) if ndim == 2 else None,
        onembed = ((d_output.shape[0], d_output.ld) if
                       ndim == 2 else (d_output.ld, outshape[-1])))
    for i in range(0, total_inputs, batch_size):
        ntransform = min(batch_size, total_inputs-i)
        if ntransform != batch_size:
            del plan
            plan = fftplan(
                size, d_A.dtype, d_A.ld, d_output.ld, forward = True,
                econ = realA, batch_size = ntransform, 
                inembed = (d_A.shape[0], d_A.ld) if ndim == 2 else None,
                onembed = ((d_output.shape[0], d_output.ld) if
                            ndim == 2 else (d_output.ld, outshape[-1])))
        plan.transform(d_A if ndim == 2 else d_A[i:i+ntransform],
                       d_output if ndim == 2 else d_output[i:i+ntransform])
    del plan
    if realA and not econ:
        global _kernels
        id = cuda.Context.get_device().PCI_BUS_ID
        func_name = 'get_2d'+'_'+str(id)+'_'+outdtype.name+'_'+str(ndim)
        if hasattr(_kernels, func_name):
            pad_func = getattr(_kernels, func_name)
        else:
            pad_func = _get_2d_pad_func(outdtype, ndim)
            setattr(_kernels, func_name, pad_func)
        pad_func.prepared_call(
            (6*cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1),
            (256, 1, 1), d_output.gpudata, d_output.ld,
            size[1], size[0], total_inputs)
    return d_output
        

def _fft2_gpuarray(d_A, econ = False):
    """
    Perform 2D fft on the last two axis of d_A
    can accept only 2D or 3D array
    
    Parameters
    ----------
    d_A : pycuda.gpuarray.GPUArray
        Input array, complex or real
    econ : bool, optional
        Only applies when d_A is real
        If True, the output only contains half of the fft result,
        the other half can be inferred.
    
    Returns
    -------
    out : pycuda.gpuarray.GPUArray, complex
        Containing the fft of corresponding to the inputs.
    """
    ndim = len(d_A.shape)
    if ndim == 2:
        total_inputs = 1
        size = d_A.shape
    elif ndim == 3:
        total_inputs = d_A.shape[0]
        size = d_A.shape[1:3]
    else:
        raise ValueError("Input to fft2 must be of 2D or 3D")
    realA = parray.isrealobj(d_A)
    
    if econ and not realA:
        print ("Warning ypcutil.fft.fft2: "
               "requested econ outputs, but getting complex inputs. "
               "econ is neglected")
    
    outdtype = parray.floattocomplex(d_A.dtype)
    outshape = [b for b in d_A.shape]
    if econ and realA:
        outshape[-1] = outshape[-1]/2+1
    d_output = gpuarray.empty(outshape, outdtype)
    batch_size = min(total_inputs, 8)
    input_total_elements = size[0]*size[1]
    output_total_elements = outshape[-1]*outshape[-2]
    
    plan = fftplan(
        size, d_A.dtype, input_total_elements, output_total_elements,
        forward = True, econ = realA, batch_size = batch_size, 
        #inembed = (d_A.shape[0], d_A.ld) if ndim == 2 else None,
        #onembed = ((d_output.shape[0], d_output.ld) if
        #               ndim == 2 else (d_output.ld, outshape[-1])))
        onembed = (1, size[1]) if realA and not econ else None)
    for i in range(0, total_inputs, batch_size):
        ntransform = min(batch_size, total_inputs-i)
        if ntransform != batch_size:
            del plan
            plan = fftplan(
                size, d_A.dtype, input_total_elements, output_total_elements,
                forward = True, econ = realA, batch_size = ntransform, 
                onembed = (1, size[1]) if realA and not econ else None)
        plan.transform_p(
            int(d_A.gpudata) + i*input_total_elements*d_A.dtype.itemsize,
            int(d_output.gpudata) + i*outdtype.itemsize*output_total_elements)
    del plan
    if realA and not econ:
        global _kernels
        id = cuda.Context.get_device().PCI_BUS_ID
        func_name = 'get_2d'+'_'+str(id)+'_'+outdtype.name+'_'+str(ndim)
        if hasattr(_kernels, func_name):
            pad_func = getattr(_kernels, func_name)
        else:
            pad_func = _get_2d_pad_func(outdtype, ndim)
            setattr(_kernels, func_name, pad_func)
        pad_func.prepared_call(
            (6*cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1),
            (256, 1, 1), d_output.gpudata,
            output_total_elements if ndim == 3 else outshape[-1],
            size[1], size[0], total_inputs)
    return d_output


def ifft2(d_A, econ = False, even_size = None,
         scale = True, scalevalue = None):
    """
    Perform 2D inverse fft on the last two axes of d_A
    can accept only 2D or 3D array
    
    Parameters
    ----------
    d_A : parray.PitchArray or pycuda.gpuarray.GPUArray
        Input array, complex, 2D or 3D
        For 3D arrays, ifft will be performed on the last two axes.
    econ : bool, optional
        Whether the dft is stored in econ fashion in d_A.
    even_size : bool or None, optional
        Only effects when econ is True.
        If None, the size of fft is inferred.
        from element in d_A if d_A is real.
        If True, size of fft is even, else odd.
    scale : bool, optional
        Whether to scale the ifft to true ifft results.
        If false, the returned ifft is not normalize by N,
        where N is the size of ifft.
        If True, the ifft is normalized to be the true idft.
    scalevalue : float or None, optioinal
        Only takes effect when scale if True.
        If None, scale to the default size 1/N.
        If float, scale by value float.
        
    Returns
    -------
    out : same type as d_A
        If econ is True, returns real array
        Otherwise, returns complex array.
    """
    if type(d_A) is parray.PitchArray:
        return _ifft2_parray(d_A, econ = econ, even_size = even_size,
                             scale = scale, scalevalue = scalevalue)
    elif type(d_A) is gpuarray.GPUArray:
        return _ifft2_gpuarray(d_A, econ = econ, even_size = even_size,
                               scale = scale, scalevalue = scalevalue)
    else:
        raise TypeError("IFFT2: Only PitchArray and GPUArray are supported")


def _ifft2_parray(d_A, econ = False, even_size = None,
          scale = True, scalevalue = None):
    """
    Perform 2D inverse fft on the last two axes of d_A
    can accept only 2D or 3D array
    
    Parameters
    ----------
    d_A : parray.PitchArray 
        Input array, complex, 2D or 3D
        For 3D arrays, ifft will be performed on the last two axes.
    econ : bool, optional
        Whether the dft is stored in econ fashion in d_A.
    even_size : bool or None, optional
        Only effects when econ is True.
        If None, the size of fft is inferred.
        from element in d_A if d_A is real.
        If True, size of fft is even, else odd.
    scale : bool, optional
        Whether to scale the ifft to true ifft results.
        If false, the returned ifft is not normalize by N,
        where N is the size of ifft.
        If True, the ifft is normalized to be the true idft.
    scalevalue : float or None, optioinal
        Only takes effect when scale if True.
        If None, scale to the default size 1/N.
        If float, scale by value float.
        
    Returns
    -------
    out : parray.PitchArray
        If econ is True, returns real array
        Otherwise, returns complex array.
    """
    ndim = len(d_A.shape)
    if ndim == 2:
        total_inputs = 1
        if econ:
            if even_size is None:
                even_size = _check_even_econ_1d(d_A, d_A.shape[1])
            size = (d_A.shape[0],
                    (d_A.shape[1]-1)*2 + (0 if even_size else 1))
        else:
            size = d_A.shape
    elif ndim == 3:
        total_inputs = d_A.shape[0]
        if econ:
            if even_size is None:
                even_size = _check_even_econ_1d(d_A, d_A.shape[2])
            size = (d_A.shape[1],
                    (d_A.shape[2]-1)*2 + (0 if even_size else 1))
        else:
            size = d_A.shape[1:3]
    else:
        raise ValueError("Input to ifft2 must be 2D or 3D")
    outdtype = parray.complextofloat(d_A.dtype) if econ else d_A.dtype        
    d_output = (parray.empty((total_inputs, size[0], size[1]), outdtype) if
                    ndim == 3 else parray.empty(size, outdtype))
    
    batch_size = min(total_inputs, 8)
    plan = fftplan(size, d_A.dtype, d_A.ld, d_output.ld,
                   forward = False, econ = econ,
                   batch_size = batch_size,
                   inembed = ((1, d_A.ld) if
                              ndim == 2 else (d_A.ld, d_A.shape[2])),
                   onembed = ((1, d_output.ld) if
                              ndim == 2 else None))
    for i in range(0, total_inputs, batch_size):
        ntransform = min(batch_size, total_inputs-i)
        if ntransform != batch_size:
            del plan
            plan = fftplan(size, d_A.dtype, d_A.ld, d_output.ld,
                           forward = False, econ = econ,
                           batch_size = ntransform,
                           inembed = ((1, d_A.ld) if
                                      ndim == 2 else (d_A.ld, d_A.shape[2])),
                           onembed = ((1, d_output.ld) if
                                      ndim == 2 else None))
        plan.transform(d_A if ndim == 2 else d_A[i:i+ntransform],
                       d_output if ndim == 2 else d_output[i:i+ntransform])
    del plan
    if scale:
        if scalevalue is None:
            scalevalue = 1./size[1]/size[0]
        d_output *= scalevalue
    return d_output


def _ifft2_gpuarray(d_A, econ = False, even_size = None,
          scale = True, scalevalue = None):
    """
    Perform 2D inverse fft on the last two axes of d_A
    can accept only 2D or 3D array
    
    Parameters
    ----------
    d_A : pycuda.gpuarray.GPUArray
        Input array, complex, 2D or 3D
        For 3D arrays, ifft will be performed on the last two axes.
    econ : bool, optional
        Whether the dft is stored in econ fashion in d_A.
    even_size : bool or None, optional
        Only effects when econ is True.
        If None, the size of fft is inferred.
        from element in d_A if d_A is real.
        If True, size of fft is even, else odd.
    scale : bool, optional
        Whether to scale the ifft to true ifft results.
        If false, the returned ifft is not normalize by N,
        where N is the size of ifft.
        If True, the ifft is normalized to be the true idft.
    scalevalue : float or None, optioinal
        Only takes effect when scale if True.
        If None, scale to the default size 1/N.
        If float, scale by value float.
        
    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        If econ is True, returns real array
        Otherwise, returns complex array.
    """
    ndim = len(d_A.shape)
    if ndim == 2:
        total_inputs = 1
        if econ:
            if even_size is None:
                even_size = _check_even_econ_1d(d_A, d_A.shape[1])
            size = (d_A.shape[0],
                    (d_A.shape[1]-1)*2 + (0 if even_size else 1))
        else:
            size = d_A.shape
    elif ndim == 3:
        total_inputs = d_A.shape[0]
        if econ:
            if even_size is None:
                even_size = _check_even_econ_1d(d_A, d_A.shape[2])
            size = (d_A.shape[1],
                    (d_A.shape[2]-1)*2 + (0 if even_size else 1))
        else:
            size = d_A.shape[1:3]
    else:
        raise ValueError("Input to ifft2 must be 2D or 3D")
    outdtype = parray.complextofloat(d_A.dtype) if econ else d_A.dtype        
    d_output = (gpuarray.empty((total_inputs, size[0], size[1]), outdtype) if
                    ndim == 3 else gpuarray.empty(size, outdtype))
    
    input_total_elements = size[0]*(size[1]/2+1 if econ else size[1])
    output_total_elements = size[0]*size[1]
    batch_size = min(total_inputs, 8)
    plan = fftplan(size, d_A.dtype, input_total_elements,
                   output_total_elements, forward = False,
                   econ = econ, batch_size = batch_size)
                   #inembed = ((d_A.shape[0], d_A.ld) if
                   #           ndim == 2 else (d_A.ld, d_A.shape[2])),
                   #onembed = ((d_output.shape[0], d_output.ld) if
                   #           ndim == 2 else None))
    for i in range(0, total_inputs, batch_size):
        ntransform = min(batch_size, total_inputs-i)
        if ntransform != batch_size:
            del plan
            plan = fftplan(size, d_A.dtype, input_total_elements,
                           output_total_elements, forward = False,
                           econ = econ, batch_size = ntransform)
        plan.transform_p(
            int(d_A.gpudata) + i*input_total_elements*d_A.dtype.itemsize,
            int(d_output.gpudata) + i*outdtype.itemsize*output_total_elements)
    del plan
    if scale:
        if scalevalue is None:
            scalevalue = 1./size[1]/size[0]
        d_output *= scalevalue
    return d_output


def fft3(d_A, econ = False, shape = None):
    """
    Perform 3D fft on d_A
    
    Parameters
    ----------
    d_A : parray.PitchArray or pycuda.gpuarray.GPUArray
        If using PitchArray: real or complex, 2D
        The last axes of PitchArray contains rasterized 3 dimensional data
        If using GPUArray: real or complex, 3D or 4D
        For 4D GPUArrays, ifft will be performed on the last three axes.
    econ : bool, optional
        Only applies when d_A is real
        If True, the output only contains half of the fft result,
        the other half can be inferred.
    shape : tuple of ints
        If d_A is PitchArray, then shape must be specified.
    
    Returns
    -------
    out : parray.PitchArray, complex
        Containing the fft of corresponding to the inputs.
    """

    if type(d_A) is parray.PitchArray:
        if shape is None:
            raise LogicError("FFT3: shape must be specified "
                             "when input is PitchArray")
        return _fft3_parray(d_A, shape, econ)
    elif type(d_A) is gpuarray.GPUArray:
        return _fft3_gpuarray(d_A, econ)
    else:
        raise TypeError("FFT3: Only PitchArray and GPUArray are supported")


def _fft3_parray(d_A, shape, econ = False):
    """
    Perform 3D fft on the last three axis of d_A
    can accept only 2D array, where axis 0 is indicate the
    batch number, and axis 1 is all the data for one batch
    in three dimension.
    
    Parameters
    ----------
    d_A : parray.PitchArray or pycuda.gpuarray.GPUArray
        Data to be transformed.
        If using PitchArray: complex, 2D
        For 3D arrays, ifft will be performed on the last two axes.
        If using GPUArray: complex, 3D or 4D
        For 4D arrays, ifft will be performed on the last three axes.
        Input array, complex or real
    econ : bool, optional
        Only applies when d_A is real
        If True, the output only contains half of the fft result,
        the other half can be inferred.
    
    Returns
    -------
    out : parray.PitchArray, complex
        Containing the fft of corresponding to the inputs.
    """
    ndim = len(d_A.shape)
    if ndim == 2:
        total_inputs = d_A.shape[0]
        size = shape
    else:
        raise ValueError("PitchArray input to fft3 must be a 2D")
    realA = parray.isrealobj(d_A)
    
    if econ and not realA:
        print ("Warning ypcutil.fft.fft3: "
               "requested econ outputs, but getting complex inputs. "
               "econ is neglected")
    
    outdtype = parray.floattocomplex(d_A.dtype)
    outsize = [b for b in shape]
    if econ and realA:
        outsize[-1] = outsize[-1]/2+1
    output_total_elements = outsize[0]*outsize[1]*outsize[2]
    d_output = parray.empty((total_inputs, output_total_elements), outdtype)
    batch_size = min(total_inputs, 4)
    
    plan = fftplan(
        size, d_A.dtype, d_A.ld, d_output.ld, forward = True, econ = realA,
        batch_size = batch_size, 
        #inembed = (d_A.mem_size, d_A.ld, d_A.shape[2]) if ndim == 3 else None,
        onembed = (1, size[1], size[2]) if realA and not econ else None)
    for i in range(0, total_inputs, batch_size):
        ntransform = min(batch_size, total_inputs-i)
        if ntransform != batch_size:
            del plan
            plan = fftplan(
                size, d_A.dtype, d_A.ld, d_output.ld,
                forward = True, econ = realA, batch_size = ntransform,
                onembed = ((1, size[1], size[2]) if
                           realA and not econ else None))
        plan.transform(d_A[i:i+ntransform], d_output[i:i+ntransform])
    del plan
    if realA and not econ:
        global _kernels
        id = cuda.Context.get_device().PCI_BUS_ID
        func_name = 'get_3d'+'_'+str(id)+'_'+outdtype.name+'_'+str(4)
        if hasattr(_kernels, func_name):
            pad_func = getattr(_kernels, func_name)
        else:
            pad_func = _get_3d_pad_func(outdtype, 4)
            setattr(_kernels, func_name, pad_func)
        pad_func.prepared_call(
            (6*cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1),
            (256, 1, 1), d_output.gpudata, d_output.ld, size[2],
            size[1], size[0], total_inputs)
    return d_output


def _fft3_gpuarray(d_A, econ = False):
    """
    Perform 3D fft on the last three axis of d_A
    can accept only 2D array
    
    Parameters
    ----------
    d_A : parray.PitchArray
        Input array, complex or real
    econ : bool, optional
        Only applies when d_A is real
        If True, the output only contains half of the fft result,
        the other half can be inferred.
    
    Returns
    -------
    out : parray.PitchArray, complex
        Containing the fft of corresponding to the inputs.
    """
    ndim = len(d_A.shape)
    if ndim == 3:
        total_inputs = 1
        size = d_A.shape
    elif ndim == 4:
        total_inputs = d_A.shape[0]
        size = d_A.shape[1:4]
    else:
        raise ValueError("Input to fft3 must be of 3D or 4D")
    realA = parray.isrealobj(d_A)
    
    if econ and not realA:
        print ("Warning ypcutil.fft.fft3: "
               "requested econ outputs, but getting complex inputs. "
               "econ is neglected")
    
    outdtype = parray.floattocomplex(d_A.dtype)
    outshape = [b for b in d_A.shape]
    if econ and realA:
        outshape[-1] = outshape[-1]/2+1
    d_output = gpuarray.empty(outshape, outdtype)
    batch_size = min(total_inputs, 4)
    input_total_elements = size[0]*size[1]*size[2]
    output_total_elements = outshape[-1]*outshape[-2]*outshape[-3]
    plan = fftplan(
        size, d_A.dtype, input_total_elements, output_total_elements,
        forward = True, econ = realA, batch_size = batch_size,
        onembed = (1, size[1], size[2]) if realA and not econ else None)
        #inembed = ((input_total_elements, d_A.shape[1], d_A.shape[2]) if 
        #           ndim == 3 else None),
        #onembed = ((d_output.mem_size, outshape[-2], outshape[-1]) if
        #               ndim == 3 else
        #               (d_output.ld, outshape[-2], outshape[-1])))
    for i in range(0, total_inputs, batch_size):
        ntransform = min(batch_size, total_inputs-i)
        if ntransform != batch_size:
            del plan
            plan = fftplan(
                size, d_A.dtype, input_total_elements,
                output_total_elements, forward = True,
                econ = realA, batch_size = ntransform,
                onembed = ((1, size[1], size[2]) if
                            realA and not econ else None))
                #inembed = ((d_A.shape[0], d_A.shape[1], d_A.shape[2]) if
                #            ndim == 3 else None),
                #onembed = ((d_output.shape[0], outshape[-2],
                #            outshape[-1]) if
                #           ndim == 3 else
                #           (d_output.ld, outshape[-2],
                #            outshape[-1])))
        plan.transform_p(
            int(d_A.gpudata) + i*input_total_elements*d_A.dtype.itemsize,
            int(d_output.gpudata) + i*outdtype.itemsize*output_total_elements)
    del plan
    if realA and not econ:
        global _kernels
        id = cuda.Context.get_device().PCI_BUS_ID
        func_name = 'get_3d'+'_'+str(id)+'_'+outdtype.name+'_'+str(ndim)
        if hasattr(_kernels, func_name):
            pad_func = getattr(_kernels, func_name)
        else:
            pad_func = _get_3d_pad_func(outdtype, ndim)
            setattr(_kernels, func_name, pad_func)
        pad_func.prepared_call(
            (6*cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1),
            (256, 1, 1), d_output.gpudata, 
            output_total_elements if ndim == 4 else outshape[-1]*outshape[-2],
            size[2], size[1], size[0], total_inputs)
    return d_output


def ifft3(d_A, econ = False, even_size = None,
         scale = True, scalevalue = None, shape = None):
    """
    Perform 3D inverse fft on d_A
    
    Parameters
    ----------
    d_A : parray.PitchArray or pycuda.gpuarray.GPUArray
        Data to be inverse transformed.
        If using PitchArray: complex, 2D
        The last axes of PitchArray contains rasterized 3 dimensional data
        If using GPUArray: complex, 3D or 4D
        For 4D arrays, ifft will be performed on the last three axes.
    econ : bool, optional
        Whether the dft is stored in econ fashion in d_A.
    even_size : bool or None, optional
        Only effects when econ is True.
        If None, the size of fft is inferred.
        from element in d_A if d_A is real.
        If True, size of fft is even, else odd.
    scale : bool, optional
        Whether to scale the ifft to true ifft results.
        If false, the returned ifft is not normalize by N,
        where N is the size of ifft.
        If True, the ifft is normalized to be the true idft.
    scalevalue : float or None, optioinal
        Only takes effect when scale if True.
        If None, scale to the default size 1/N.
        If float, scale by value float.
    shape : tuple of ints
        If d_A is PitchArray, then shape must be specified.
    
    Returns
    -------
    out : smae type as d_A, complex
        Containing the ifft of corresponding to the inputs.
    """
    if type(d_A) is parray.PitchArray:
        if shape is None:
            raise LogicError("IFFT3: shape must be specified "
                             "when input is PitchArray")
        return _ifft3_parray(d_A, shape, econ = econ, even_size = even_size,
                             scale = scale, scalevalue = scalevalue)
    elif type(d_A) is gpuarray.GPUArray:
        return _ifft3_gpuarray(d_A, econ = econ, even_size = even_size,
                               scale = scale, scalevalue = scalevalue)
    else:
        raise TypeError("FFT3: Only PitchArray and GPUArray are supported")


def _ifft3_parray(d_A, shape, econ = False, even_size = None,
          scale = True, scalevalue = None):
    """
    Perform 3D inverse fft on d_A
    can accept only 2D array, where axis 0 is indicate the
    batch number, and axis 1 is all the data for one batch
    in three dimension.
    
    Parameters
    ----------
    d_A : parray.PitchArray
        Input array, complex, 2D
    econ : bool, optional
        Whether the dft is stored in econ fashion in d_A.
    even_size : bool or None, optional
        Only effects when econ is True.
        If None, the size of fft is inferred.
        from element in d_A if d_A is real.
        If True, size of fft is even, else odd.
    scale : bool, optional
        Whether to scale the ifft to true ifft results.
        If false, the returned ifft is not normalize by N,
        where N is the size of ifft.
        If True, the ifft is normalized to be the true idft.
    scalevalue : float or None, optioinal
        Only takes effect when scale if True.
        If None, scale to the default size 1/N.
        If float, scale by value float.
        
    Returns
    -------
    out : parray.PitchArray
        If econ is True, returns real array
        Otherwise, returns complex array.
    """
    ndim = len(d_A.shape)
    size = shape
    if ndim == 2:
        total_inputs = d_A.shape[0]
        size = shape
    else:
        raise ValueError("PitchArray input to ifft3 must be 2D")
    outdtype = parray.complextofloat(d_A.dtype) if econ else d_A.dtype        
    d_output = (
        parray.empty((total_inputs, size[0]*size[1]*size[2]), outdtype))
    
    batch_size = min(total_inputs, 4)
    plan = fftplan(size, d_A.dtype, d_A.ld, d_output.ld,
                   forward = False, econ = econ,
                   batch_size = batch_size)
                   #inembed = ((d_A.shape[0], d_A.ld, d_A.shape[2]) if
                   #           ndim == 3 else
                   #           (d_A.ld, d_A.shape[2],
                   #            d_A.shape[3])),
                   #onembed = ((d_output.shape[0], d_output.ld,
                   #            d_output.shape[2]) if
                   #           ndim == 3 else None))
    for i in range(0, total_inputs, batch_size):
        ntransform = min(batch_size, total_inputs-i)
        if ntransform != batch_size:
            del plan
            plan = fftplan(size, d_A.dtype, d_A.ld, d_output.ld,
                 forward = False, econ = econ,
                 batch_size = ntransform)
        plan.transform(d_A[i:i+ntransform], d_output[i:i+ntransform])
    del plan
    if scale:
        if scalevalue is None:
            scalevalue = 1./size[2]/size[1]/size[0]
        d_output *= scalevalue
    return d_output


def _ifft3_gpuarray(d_A, econ = False, even_size = None,
          scale = True, scalevalue = None):
    """
    Perform 3D inverse fft on the last three axes of d_A
    can accept only 3D or 4D array
    
    Parameters
    ----------
    d_A : pycuda.gpuarray.GPUArray
        Input array, complex, 3D or 4D
        For 3D arrays, ifft will be performed on the last two axes.
    econ : bool, optional
        Whether the dft is stored in econ fashion in d_A.
    even_size : bool or None, optional
        Only effects when econ is True.
        If None, the size of fft is inferred.
        from element in d_A if d_A is real.
        If True, size of fft is even, else odd.
    scale : bool, optional
        Whether to scale the ifft to true ifft results.
        If false, the returned ifft is not normalize by N,
        where N is the size of ifft.
        If True, the ifft is normalized to be the true idft.
    scalevalue : float or None, optioinal
        Only takes effect when scale if True.
        If None, scale to the default size 1/N.
        If float, scale by value float.
        
    Returns
    -------
    out : parray.PitchArray
        If econ is True, returns real array
        Otherwise, returns complex array.
    """
    ndim = len(d_A.shape)
    
    if ndim == 3:
        total_inputs = 1
        if econ:
            if even_size is None:
                even_size = _check_even_econ_1d(d_A, d_A.shape[2])
            size = (d_A.shape[0], d_A.shape[1],
                    (d_A.shape[2]-1)*2 + (0 if even_size else 1))
        else:
            size = d_A.shape
    elif ndim == 4:
        total_inputs = d_A.shape[0]
        if econ:
            if even_size is None:
                even_size = _check_even_econ_1d(d_A, d_A.shape[3])
            size = (d_A.shape[1], d_A.shape[2],
                    (d_A.shape[3]-1)*2 + (0 if even_size else 1))
        else:
            size = d_A.shape[1:4]
    else:
        raise ValueError("GPUArray Input to ifft3 must be 3D or 4D")
    outdtype = parray.complextofloat(d_A.dtype) if econ else d_A.dtype        
    d_output = (
        gpuarray.empty((total_inputs, size[0], size[1], size[2]), outdtype) if
        ndim == 4 else gpuarray.empty(size, outdtype))
    
    input_total_elements = size[0]*size[1]*(size[2]/2+1 if econ else size[2])
    output_total_elements = size[0]*size[1]*size[2]
    
    batch_size = min(total_inputs, 4)
    plan = fftplan(size, d_A.dtype, input_total_elements,
                   output_total_elements,
                   forward = False, econ = econ,
                   batch_size = batch_size)
                   #inembed = ((d_A.shape[0], d_A.ld, d_A.shape[2]) if
                   #           ndim == 3 else
                   #           (d_A.ld, d_A.shape[2],
                   #            d_A.shape[3])),
                   #onembed = ((d_output.shape[0], d_output.ld,
                   #           d_output.shape[2]) if
                   #           ndim == 3 else None))
    for i in range(0, total_inputs, batch_size):
        ntransform = min(batch_size, total_inputs-i)
        if ntransform != batch_size:
            del plan
            plan = fftplan(size, d_A.dtype, input_total_elements,
                           output_total_elements,
                           forward = False, econ = econ,
                           batch_size = ntransform)
        plan.transform_p(
            int(d_A.gpudata) + i*input_total_elements*d_A.dtype.itemsize,
            int(d_output.gpudata) + i*outdtype.itemsize*output_total_elements)
    del plan
    if scale:
        if scalevalue is None:
            scalevalue = 1./size[2]/size[1]/size[0]
        d_output *= scalevalue
    return d_output


def _check_even_econ_1d(A, size):
    """
    Check whether the fft size of A is even or odd.
    
    Parameters
    ----------
    A: parray.PitchArray
       The array A contains the reduced storage
       of fft for some real sequences
    size: int
          The size of reduced storage
          
    Returns
    -------
    out: bool
         True if the size is even, i.e. the last entry of fft is real
         False if the size is odd, i.e. the last entry of fft is complex
         
    Note that this assumes 0 error in the forward fft calculation,
    otherwise, the last entry of reduced storage fft will has very small
    imaginary part
    """
    a = np.empty(1, A.dtype)
    cuda.memcpy_dtoh(a, int(int(A.gpudata) + A.dtype.itemsize*(size-1)))
    return a.imag == 0.0


def _get_1d_pad_func(dtype):
    """
    Assumes that the array is already allocated and the half of
    the entry is filled with half of the fft results
    Kernel not optimized.
    """
    template = """
#include <pycuda-complex.hpp>
__global__ void
get_1d_pad_kernel(%(type)s* input, int ld, int fftsize, int nbatch)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;
    int entry_per_row = fftsize - 1 - (fftsize>>1);
    int total_entry = entry_per_row*nbatch;
    int row, col;
    
    for(int i = tid; i < total_entry; i += total_threads)
    {
        row = i / entry_per_row;
        col = i %% entry_per_row;
        input[row*ld + fftsize-col-1] = conj(input[row*ld + col+1]);
    }
}

    """
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype)},
                       options = ["--ptxas-options=-v"])
    func = mod.get_function('get_1d_pad_kernel')
    func.prepare([np.intp, np.int32, np.int32, np.int32])
    #grid = (6*cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)    
    #block = (256, 1, 1)
    #Used 19 registers, 52 bytes cmem[0], 168 bytes cmem[2]
    return func


def _get_2d_pad_func(dtype, ndim):
    """
    Assumes that the array is already allocated and the half of
    the entry is filled with half of the fft results
    Kernel not optimized.
    """
    if ndim == 3:
        template = """
#include <pycuda-complex.hpp>
__global__ void
get_2d_pad_kernel(%(type)s* input, int ld, int fftsize_x,
                  int fftsize_y, int nbatch)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;
    int entry_per_row = fftsize_x - 1 - (fftsize_x>>1);
    int entry_per_batch = entry_per_row*fftsize_y;
    int total_entry = entry_per_batch*nbatch;
    int row, col, batch, tmp, ind;
    
    for(int i = tid; i < total_entry; i += total_threads)
    {
        batch = i / entry_per_batch;
        tmp = i %% entry_per_batch;
        row = tmp / entry_per_row;
        col = tmp %% entry_per_row + 1;
        ind = batch*ld;
        input[ind + (((tmp=fftsize_y-row) == fftsize_y)?0:tmp)*fftsize_x \
            + fftsize_x-col] = conj(input[ind + col + fftsize_x*row]);
    }
}

        """
    elif ndim == 2:
        template = """
#include <pycuda-complex.hpp>
__global__ void
get_2d_pad_kernel(%(type)s* input, int ld, int fftsize_x,
                  int fftsize_y, int nbatch)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;
    int entry_per_row = fftsize_x - 1 - (fftsize_x>>1);
    int total_entry = entry_per_row*fftsize_y;
    int row, col, tmp;
    
    for(int i = tid; i < total_entry; i += total_threads)
    {
        row = i / entry_per_row;
        col = i %% entry_per_row + 1;
        input[ (((tmp=fftsize_y-row) == fftsize_y)?0:tmp)*ld \
            + fftsize_x-col] = conj(input[col + ld*row]);
    }
}    

        """
    else:
        raise ValueError("Wrong ndim to get_2d_pad_func. ndim = " + str(ndim))
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype)},
                       options = ["--ptxas-options=-v"])
    func = mod.get_function('get_2d_pad_kernel')
    func.prepare([np.intp, np.int32, np.int32, np.int32, np.int32])
    #grid = (6*cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)    
    #block = (256, 1, 1)
    #Used 19 registers, 52 bytes cmem[0], 168 bytes cmem[2]
    return func


def _get_3d_pad_func(dtype, ndim):
    """
    Assumes that the array is already allocated and the half of
    the entry is filled with half of the fft results
    Kernel not optimized.
    """
    if ndim == 4:
        template = """
#include <pycuda-complex.hpp>
__global__ void
get_3d_pad_kernel(%(type)s* input, int ld, int fftsize_x,
                  int fftsize_y, int fftsize_z, int nbatch)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;
    int entry_per_row = fftsize_x - 1 - (fftsize_x>>1);
    int entry_per_z = entry_per_row * fftsize_y;
    int entry_per_batch = entry_per_z*fftsize_z;
    int total_entry = entry_per_batch*nbatch;
    int x, y, z, batch, tmp, ind, tmp2;
    
    for(int i = tid; i < total_entry; i += total_threads)
    {
        batch = i / entry_per_batch;
        tmp = i %% entry_per_batch;
        z = tmp / entry_per_z;
        tmp = tmp %% entry_per_z;
        y = tmp / entry_per_row;
        x = tmp %% entry_per_row + 1;
        ind = batch*ld;
        
        input[ind + \
            (((tmp2=fftsize_z-z) == fftsize_z)?0:tmp2)*fftsize_x*fftsize_y +\
            (((tmp=fftsize_y-y) == fftsize_y)?0:tmp)*fftsize_x + fftsize_x-x]\
        = conj(input[ind + x + fftsize_x*y + fftsize_x*fftsize_y*z]);
    }
}

        """
    elif ndim == 3:
        template = """
#include <pycuda-complex.hpp>
__global__ void
get_3d_pad_kernel(%(type)s* input, int ld, int fftsize_x,
                  int fftsize_y, int fftsize_z, int nbatch)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;
    int entry_per_row = fftsize_x - 1 - (fftsize_x>>1);
    int entry_per_z = entry_per_row * fftsize_y;
    int total_entry = entry_per_z*fftsize_z;
    int x, y, z, tmp, tmp2;
    
    for(int i = tid; i < total_entry; i += total_threads)
    {
        z = i / entry_per_z;
        tmp = i %% entry_per_z;
        y = tmp / entry_per_row;
        x = tmp %% entry_per_row + 1;
        input[(((tmp2=fftsize_z-z) == fftsize_z)?0:tmp2)*ld +\
            (((tmp=fftsize_y-y) == fftsize_y)?0:tmp)*fftsize_x + fftsize_x-x]\
        = conj(input[x + fftsize_x*y + ld*z]);
    }
}    

        """
    else:
        raise ValueError("Wrong ndim to get_3d_pad_func. ndim = " + str(ndim))
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype)},
                       options = ["--ptxas-options=-v"])
    func = mod.get_function('get_3d_pad_kernel')
    func.prepare([np.intp, np.int32, np.int32, np.int32, np.int32, np.int32])
    #grid = (6*cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)    
    #block = (256, 1, 1)
    #Used 19 registers, 52 bytes cmem[0], 168 bytes cmem[2]
    return func

