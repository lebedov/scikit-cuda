#!/usr/bin/env python

"""
Miscellaneous PyCUDA functions.
"""

from __future__ import absolute_import, division

import atexit
import numbers
from string import Template

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.elementwise as elementwise
import pycuda.reduction as reduction
import pycuda.scan as scan
import pycuda.tools as tools
from pycuda.tools import context_dependent_memoize, dtype_to_ctype
from pycuda.compiler import SourceModule
from pytools import memoize
import numpy as np

from . import cuda
from . import cublas

import sys
if sys.version_info < (3,):
    range = xrange

try:
    from . import cula
    _has_cula = True
except (ImportError, OSError):
    _has_cula = False

try:
    from . import cusolver
    _has_cusolver = True
except (ImportError, OSError):
    _has_cusolver = False

try:
    from . import magma
    _has_magma = True
except (ImportError, OSError):
    _has_magma = False

isdoubletype = lambda x : True if x == np.float64 or \
               x == np.complex128 else False
isdoubletype.__doc__ = """
Check whether a type has double precision.

Parameters
----------
t : numpy float type
    Type to test.

Returns
-------
result : bool
    Result.
"""

iscomplextype = lambda x : True if x == np.complex64 or \
                x == np.complex128 else False
iscomplextype.__doc__ = """
Check whether a type is complex.

Parameters
----------
t : numpy float type
    Type to test.

Returns
-------
result : bool
    Result.

"""

def init_device(n=0):
    """
    Initialize a GPU device.

    Initialize a specified GPU device rather than the default device
    found by `pycuda.autoinit`.

    Parameters
    ----------
    n : int
        Device number.

    Returns
    -------
    dev : pycuda.driver.Device
        Initialized device.
    """

    drv.init()
    dev = drv.Device(n)
    return dev

def init_context(dev):
    """
    Create a context that will be cleaned up properly.

    Create a context on the specified device and register its pop()
    method with atexit.

    Parameters
    ----------
    dev : pycuda.driver.Device
        GPU device.

    Returns
    -------
    ctx : pycuda.driver.Context
        Created context.

    """

    ctx = dev.make_context()
    atexit.register(ctx.pop)
    return ctx

def done_context(ctx):
    """
    Detach from a context cleanly.

    Detach from a context and remove its pop() from atexit.

    Parameters
    ----------
    ctx : pycuda.driver.Context
        Context from which to detach.
    """

    for i in range(len(atexit._exithandlers)):
        if atexit._exithandlers[i][0] == ctx.pop:
            del atexit._exithandlers[i]
            break
    ctx.detach()

global _global_cublas_handle
_global_cublas_handle = None
global _global_cusolver_handle
_global_cusolver_handle = None
global _global_cublas_allocator
_global_cublas_allocator = None
def init(allocator=drv.mem_alloc):
    """
    Initialize libraries used by scikit-cuda.

    Initialize the CUBLAS, CULA, CUSOLVER, and MAGMA libraries used by 
    high-level functions provided by scikit-cuda.

    Parameters
    ----------
    allocator : an allocator used internally by some of the high-level
        functions.

    Notes
    -----
    This function does not initialize PyCUDA; it uses whatever device
    and context were initialized in the current host thread.
    """

    # CUBLAS uses whatever device is being used by the host thread:
    global _global_cublas_handle, _global_cublas_allocator
    if not _global_cublas_handle:
        from . import cublas  # nest to avoid requiring cublas e.g. for FFT
        _global_cublas_handle = cublas.cublasCreate()
    
    if _global_cublas_allocator is None:
        _global_cublas_allocator = allocator

    # Initializing MAGMA after CUSOLVER causes some functions in the latter to
    # fail with internal errors:
    if _has_magma:
        magma.magma_init()

    global _global_cusolver_handle
    if not _global_cusolver_handle:
        from . import cusolver
        _global_cusolver_handle = cusolver.cusolverDnCreate()

    # culaSelectDevice() need not (and, in fact, cannot) be called
    # here because the host thread has already been bound to a GPU
    # device:
    if _has_cula:
        cula.culaInitialize()

def shutdown():
    """
    Shutdown libraries used by scikit-cuda.

    Shutdown the CUBLAS, CULA, CUSOLVER, and MAGMA libraries used by 
    high-level functions provided by scikits-cuda.

    Notes
    -----
    This function does not shutdown PyCUDA.
    """

    global _global_cublas_handle
    if _global_cublas_handle:
        from . import cublas  # nest to avoid requiring cublas e.g. for FFT
        cublas.cublasDestroy(_global_cublas_handle)
        _global_cublas_handle = None

    global _global_cusolver_handle
    if _global_cusolver_handle:
        from . import cusolver
        cusolver.cusolverDnDestroy(_global_cusolver_handle)
        _global_cusolver_handle = None

    if _has_magma:
        magma.magma_finalize()
    if _has_cula:
        cula.culaShutdown()

def get_compute_capability(dev):
    """
    Get the compute capability of the specified device.

    Retrieve the compute capability of the specified CUDA device and
    return it as a floating point value.

    Parameters
    ----------
    d : pycuda.driver.Device
        Device object to examine.

    Returns
    -------
    c : float
        Compute capability.
    """

    return np.float('.'.join([str(i) for i in
                                 dev.compute_capability()]))

def get_current_device():
    """
    Get the device in use by the current context.

    Returns
    -------
    d : pycuda.driver.Device
        Device in use by current context.
    """

    return drv.Device(cuda.cudaGetDevice())

@memoize
def get_dev_attrs(dev):
    """
    Get select CUDA device attributes.

    Retrieve select attributes of the specified CUDA device that
    relate to maximum thread block and grid sizes.

    Parameters
    ----------
    d : pycuda.driver.Device
        Device object to examine.

    Returns
    -------
    attrs : list
        List containing [MAX_THREADS_PER_BLOCK,
        (MAX_BLOCK_DIM_X, MAX_BLOCK_DIM_Y, MAX_BLOCK_DIM_Z),
        (MAX_GRID_DIM_X, MAX_GRID_DIM_Y, MAX_GRID_DIM_Z)]
    """

    attrs = dev.get_attributes()
    return [attrs[drv.device_attribute.MAX_THREADS_PER_BLOCK],
            (attrs[drv.device_attribute.MAX_BLOCK_DIM_X],
             attrs[drv.device_attribute.MAX_BLOCK_DIM_Y],
             attrs[drv.device_attribute.MAX_BLOCK_DIM_Z]),
            (attrs[drv.device_attribute.MAX_GRID_DIM_X],
             attrs[drv.device_attribute.MAX_GRID_DIM_Y],
             attrs[drv.device_attribute.MAX_GRID_DIM_Z])]

iceil = lambda n: int(np.ceil(n))

@memoize
def select_block_grid_sizes(dev, data_shape, threads_per_block=None):
    """
    Determine CUDA block and grid dimensions given device constraints.

    Determine the CUDA block and grid dimensions allowed by a GPU
    device that are sufficient for processing every element of an
    array in a separate thread.

    Parameters
    ----------
    d : pycuda.driver.Device
        Device object to be used.
    data_shape : tuple
        Shape of input data array. Must be of length 2.
    threads_per_block : int, optional
        Number of threads to execute in each block. If this is None,
        the maximum number of threads per block allowed by device `d`
        is used.

    Returns
    -------
    block_dim : tuple
        X, Y, and Z dimensions of minimal required thread block.
    grid_dim : tuple
        X and Y dimensions of minimal required block grid.

    Notes
    -----
    Using the scheme in this function, all of the threads in the grid can be enumerated
    as `i = blockIdx.y*max_threads_per_block*max_blocks_per_grid+
    blockIdx.x*max_threads_per_block+threadIdx.x`.

    For 2D shapes, the subscripts of the element `data[a, b]` where `data.shape == (A, B)`
    can be computed as
    `a = i/B`
    `b = mod(i,B)`.

    For 3D shapes, the subscripts of the element `data[a, b, c]` where
    `data.shape == (A, B, C)` can be computed as
    `a = i/(B*C)`
    `b = mod(i, B*C)/C`
    `c = mod(mod(i, B*C), C)`.

    For 4D shapes, the subscripts of the element `data[a, b, c, d]`
    where `data.shape == (A, B, C, D)` can be computed as
    `a = i/(B*C*D)`
    `b = mod(i, B*C*D)/(C*D)`
    `c = mod(mod(i, B*C*D)%(C*D))/D`
    `d = mod(mod(mod(i, B*C*D)%(C*D)), D)`

    It is advisable that the number of threads per block be a multiple
    of the warp size to fully utilize a device's computing resources.
    """

    # Sanity checks:
    if np.isscalar(data_shape):
        data_shape = (data_shape,)

    # Number of elements to process; we need to cast the result of
    # np.prod to a Python int to prevent PyCUDA's kernel execution
    # framework from getting confused when
    N = int(np.prod(data_shape))

    # Get device constraints:
    max_threads_per_block, max_block_dim, max_grid_dim = get_dev_attrs(dev)

    if threads_per_block is not None:
        if threads_per_block > max_threads_per_block:
            raise ValueError('threads per block exceeds device maximum')
        else:
            max_threads_per_block = threads_per_block

    # Actual number of thread blocks needed:
    blocks_needed = iceil(N/float(max_threads_per_block))

    if blocks_needed <= max_grid_dim[0]:
        return (max_threads_per_block, 1, 1), (blocks_needed, 1, 1)
    elif blocks_needed > max_grid_dim[0] and \
         blocks_needed <= max_grid_dim[0]*max_grid_dim[1]:
        return (max_threads_per_block, 1, 1), \
            (max_grid_dim[0], iceil(blocks_needed/float(max_grid_dim[0])), 1)
    elif blocks_needed > max_grid_dim[0]*max_grid_dim[1] and \
         blocks_needed <= max_grid_dim[0]*max_grid_dim[1]*max_grid_dim[2]:
        return (max_threads_per_block, 1, 1), \
            (max_grid_dim[0], max_grid_dim[1],
             iceil(blocks_needed/float(max_grid_dim[0]*max_grid_dim[1])))
    else:
        raise ValueError('array size too large')

def zeros(shape, dtype, order='C', allocator=drv.mem_alloc):
    """
    Return an array of the given shape and dtype filled with zeros.

    Parameters
    ----------
    shape : tuple
        Array shape.
    dtype : data-type
        Data type for the array.
    order : {'C', 'F'}, optional
        Create array using row-major or column-major format.
    allocator : callable, optional
        Returns an object that represents the memory allocated for
        the requested array.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        Array of zeros with the given shape, dtype, and order.

    Notes
    -----
    This function exists to work around the following numpy bug that
    prevents pycuda.gpuarray.zeros() from working properly with
    complex types in pycuda 2011.1.2:
    http://projects.scipy.org/numpy/ticket/1898
    """

    out = gpuarray.GPUArray(shape, dtype, allocator, order=order)
    z = np.zeros((), dtype)
    out.fill(z)
    return out

def zeros_like(a):
    """
    Return an array of zeros with the same shape and type as a given
    array.

    Parameters
    ----------
    a : array_like
        The shape and data type of `a` determine the corresponding
        attributes of the returned array.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        Array of zeros with the shape, dtype, and strides of `a`.
    """

    out = gpuarray.GPUArray(a.shape, a.dtype, drv.mem_alloc,
                            strides=a.strides)
    z = np.zeros((), a.dtype)
    out.fill(z)
    return out

def ones(shape, dtype, order='C', allocator=drv.mem_alloc):
    """
    Return an array of the given shape and dtype filled with ones.

    Parameters
    ----------
    shape : tuple
        Array shape.
    dtype : data-type
        Data type for the array.
    order : {'C', 'F'}, optional
        Create array using row-major or column-major format.
    allocator : callable, optional
        Returns an object that represents the memory allocated for
        the requested array.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        Array of ones with the given shape, dtype, and order.
    """

    out = gpuarray.GPUArray(shape, dtype, allocator, order=order)
    o = np.ones((), dtype)
    out.fill(o)
    return out

def ones_like(a):
    """
    Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data type of `a` determine the corresponding
        attributes of the returned array.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        Array of ones with the shape, dtype, and strides of `other`.
    """

    out = gpuarray.GPUArray(a.shape, a.dtype,
                            a.allocator, strides=a.strides)
    o = np.ones((), a.dtype)
    out.fill(o)
    return out

def inf(shape, dtype, order='C', allocator=drv.mem_alloc):
    """
    Return an array of the given shape and dtype filled with infs.

    Parameters
    ----------
    shape : tuple
        Array shape.
    dtype : data-type
        Data type for the array.
    order : {'C', 'F'}, optional
        Create array using row-major or column-major format.
    allocator : callable, optional
        Returns an object that represents the memory allocated for
        the requested array.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        Array of infs with the given shape, dtype, and order.
    """

    out = gpuarray.GPUArray(shape, dtype, allocator, order=order)
    i = np.array(np.inf, dtype)
    out.fill(i)
    return out

def maxabs(x_gpu):
    """
    Get maximum absolute value.

    Find maximum absolute value in the specified array.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Input array.

    Returns
    -------
    m_gpu : pycuda.gpuarray.GPUArray
        Array containing maximum absolute value in `x_gpu`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import misc
    >>> x_gpu = gpuarray.to_gpu(np.array([-1, 2, -3], np.float32))
    >>> m_gpu = misc.maxabs(x_gpu)
    >>> np.allclose(m_gpu.get(), 3.0)
    True
    """

    try:
        func = maxabs.cache[x_gpu.dtype]
    except KeyError:
        ctype = tools.dtype_to_ctype(x_gpu.dtype)
        use_double = int(x_gpu.dtype in [np.float64, np.complex128])
        ret_type = np.float64 if use_double else np.float32
        func = reduction.ReductionKernel(ret_type, neutral="0",
                                           reduce_expr="max(a,b)",
                                           map_expr="abs(x[i])",
                                           arguments="{ctype} *x".format(ctype=ctype))
        maxabs.cache[x_gpu.dtype] = func
    return func(x_gpu)
maxabs.cache = {}

def cumsum(x_gpu):
    """
    Cumulative sum.

    Return the cumulative sum of the elements in the specified array.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Input array.

    Returns
    -------
    c_gpu : pycuda.gpuarray.GPUArray
        Output array containing cumulative sum of `x_gpu`.

    Notes
    -----
    Higher dimensional arrays are implicitly flattened row-wise by this function.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import misc
    >>> x_gpu = gpuarray.to_gpu(np.random.rand(5).astype(np.float32))
    >>> c_gpu = misc.cumsum(x_gpu)
    >>> np.allclose(c_gpu.get(), np.cumsum(x_gpu.get()))
    True
    """

    try:
        func = cumsum.cache[x_gpu.dtype]
    except KeyError:
        func = scan.InclusiveScanKernel(x_gpu.dtype, 'a+b',
                                        preamble='#include <pycuda-complex.hpp>')
        cumsum.cache[x_gpu.dtype] = func
    return func(x_gpu)
cumsum.cache = {}

def diff(x_gpu):
    """
    Calculate the discrete difference.

    Calculates the first order difference between the successive
    entries of a vector.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Input vector.

    Returns
    -------
    y_gpu : pycuda.gpuarray.GPUArray
        Discrete difference.

    Examples
    --------
    >>> import pycuda.driver as drv
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import misc
    >>> x = np.asarray(np.random.rand(5), np.float32)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = misc.diff(x_gpu)
    >>> np.allclose(np.diff(x), y_gpu.get())
    True
    """

    y_gpu = gpuarray.empty(len(x_gpu)-1, x_gpu.dtype)
    try:
        func = diff.cache[x_gpu.dtype]
    except KeyError:
        ctype = tools.dtype_to_ctype(x_gpu.dtype)
        func = elementwise.ElementwiseKernel("{ctype} *a, {ctype} *b".format(ctype=ctype),
                                             "b[i] = a[i+1]-a[i]")
        diff.cache[x_gpu.dtype] = func
    func(x_gpu, y_gpu)
    return y_gpu
diff.cache = {}


# List of available numerical types provided by numpy:
num_types = [np.typeDict[t] for t in \
             np.typecodes['AllInteger']+np.typecodes['AllFloat']]

# Numbers of bytes occupied by each numerical type:
num_nbytes = dict((np.dtype(t),t(1).nbytes) for t in num_types)


def set_realloc(x_gpu, data):
    """
    Transfer data into a GPUArray instance.

    Copies the contents of a numpy array into a GPUArray instance. If
    the array has a different type or dimensions than the instance,
    the GPU memory used by the instance is reallocated and the
    instance updated appropriately.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        GPUArray instance to modify.
    data : numpy.ndarray
        Array of data to transfer to the GPU.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import misc
    >>> x = np.asarray(np.random.rand(5), np.float32)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> x = np.asarray(np.random.rand(10, 1), np.float64)
    >>> set_realloc(x_gpu, x)
    >>> np.allclose(x, x_gpu.get())
    True
    """

    # Only reallocate if absolutely necessary:
    if x_gpu.shape != data.shape or x_gpu.size != data.size or \
        x_gpu.strides != data.strides or x_gpu.dtype != data.dtype:

        # Free old memory:
        x_gpu.gpudata.free()

        # Allocate new memory:
        nbytes = num_nbytes[data.dtype]
        x_gpu.gpudata = drv.mem_alloc(nbytes*data.size)

        # Set array attributes:
        x_gpu.shape = data.shape
        x_gpu.size = data.size
        x_gpu.strides = data.strides
        x_gpu.dtype = data.dtype

    # Update the GPU memory:
    x_gpu.set(data)

def get_by_index(src_gpu, ind):
    """
    Get values in a GPUArray by index.

    Parameters
    ----------
    src_gpu : pycuda.gpuarray.GPUArray
        GPUArray instance from which to extract values.
    ind : pycuda.gpuarray.GPUArray or numpy.ndarray
        Array of element indices to set. Must have an integer dtype.

    Returns
    -------
    res_gpu : pycuda.gpuarray.GPUArray
        GPUArray with length of `ind` and dtype of `src_gpu` containing
        selected values.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import misc
    >>> src = np.random.rand(5).astype(np.float32)
    >>> src_gpu = gpuarray.to_gpu(src)
    >>> ind = gpuarray.to_gpu(np.array([0, 2, 4]))
    >>> res_gpu = misc.get_by_index(src_gpu, ind)
    >>> np.allclose(res_gpu.get(), src[[0, 2, 4]])
    True

    Notes
    -----
    Only supports 1D index arrays.

    May not be efficient for certain index patterns because of lack of inability
    to coalesce memory operations.
    """

    # Only support 1D index arrays:
    assert len(np.shape(ind)) == 1
    assert issubclass(ind.dtype.type, numbers.Integral)
    N = len(ind)
    if not isinstance(ind, gpuarray.GPUArray):
        ind = gpuarray.to_gpu(ind)
    dest_gpu = gpuarray.empty(N, dtype=src_gpu.dtype)

    # Manually handle empty index array because it will cause the kernel to
    # fail if processed:
    if N == 0:
        return dest_gpu
    try:
        func = get_by_index.cache[(src_gpu.dtype, ind.dtype)]
    except KeyError:
        data_ctype = tools.dtype_to_ctype(src_gpu.dtype)
        ind_ctype = tools.dtype_to_ctype(ind.dtype)
        v = "{data_ctype} *dest, {ind_ctype} *ind, {data_ctype} *src".format(data_ctype=data_ctype, ind_ctype=ind_ctype)
        func = elementwise.ElementwiseKernel(v, "dest[i] = src[ind[i]]")
        get_by_index.cache[(src_gpu.dtype, ind.dtype)] = func
    func(dest_gpu, ind, src_gpu, range=slice(0, N, 1))
    return dest_gpu
get_by_index.cache = {}

def set_by_index(dest_gpu, ind, src_gpu, ind_which='dest'):
    """
    Set values in a GPUArray by index.

    Parameters
    ----------
    dest_gpu : pycuda.gpuarray.GPUArray
        GPUArray instance to modify.
    ind : pycuda.gpuarray.GPUArray or numpy.ndarray
        1D array of element indices to set. Must have an integer dtype.
    src_gpu : pycuda.gpuarray.GPUArray
        GPUArray instance from which to set values.
    ind_which : str
        If set to 'dest', set the elements in `dest_gpu` with indices `ind`
        to the successive values in `src_gpu`; the lengths of `ind` and
        `src_gpu` must be equal. If set to 'src', set the
        successive values in `dest_gpu` to the values in `src_gpu` with indices
        `ind`; the lengths of `ind` and `dest_gpu` must be equal.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import misc
    >>> dest_gpu = gpuarray.to_gpu(np.arange(5, dtype=np.float32))
    >>> ind = gpuarray.to_gpu(np.array([0, 2, 4]))
    >>> src_gpu = gpuarray.to_gpu(np.array([1, 1, 1], dtype=np.float32))
    >>> misc.set_by_index(dest_gpu, ind, src_gpu, 'dest')
    >>> np.allclose(dest_gpu.get(), np.array([1, 1, 1, 3, 1], dtype=np.float32))
    True
    >>> dest_gpu = gpuarray.to_gpu(np.zeros(3, dtype=np.float32))
    >>> ind = gpuarray.to_gpu(np.array([0, 2, 4]))
    >>> src_gpu = gpuarray.to_gpu(np.arange(5, dtype=np.float32))
    >>> misc.set_by_index(dest_gpu, ind, src_gpu)
    >>> np.allclose(dest_gpu.get(), np.array([0, 2, 4], dtype=np.float32))
    True

    Notes
    -----
    Only supports 1D index arrays.

    May not be efficient for certain index patterns because of lack of inability
    to coalesce memory operations.
    """

    # Only support 1D index arrays:
    assert len(np.shape(ind)) == 1
    assert dest_gpu.dtype == src_gpu.dtype
    assert issubclass(ind.dtype.type, numbers.Integral)
    N = len(ind)

    # Manually handle empty index array because it will cause the kernel to
    # fail if processed:
    if N == 0:
        return
    if ind_which == 'dest':
        assert N == len(src_gpu)
    elif ind_which == 'src':
        assert N == len(dest_gpu)
    else:
        raise ValueError('invalid value for `ind_which`')
    if not isinstance(ind, gpuarray.GPUArray):
        ind = gpuarray.to_gpu(ind)
    try:
        func = set_by_index.cache[(dest_gpu.dtype, ind.dtype, ind_which)]
    except KeyError:
        data_ctype = tools.dtype_to_ctype(dest_gpu.dtype)
        ind_ctype = tools.dtype_to_ctype(ind.dtype)
        v = "{data_ctype} *dest, {ind_ctype} *ind, {data_ctype} *src".format(data_ctype=data_ctype, ind_ctype=ind_ctype)

        if ind_which == 'dest':
            func = elementwise.ElementwiseKernel(v, "dest[ind[i]] = src[i]")
        else:
            func = elementwise.ElementwiseKernel(v, "dest[i] = src[ind[i]]")
        set_by_index.cache[(dest_gpu.dtype, ind.dtype, ind_which)] = func
    func(dest_gpu, ind, src_gpu, range=slice(0, N, 1))
set_by_index.cache = {}


@context_dependent_memoize
def _get_binaryop_vecmat_kernel(dtype, binary_op):
    template = Template("""
    #include <pycuda-complex.hpp>

    __global__ void opColVecToMat(const ${type} *mat, const ${type} *vec, ${type} *out,
                                   const int n, const int m){
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
        const int tidy = blockIdx.y * blockDim.y + threadIdx.y;

        extern __shared__ ${type} shared_vec[];

        if ((ty == 0) & (tidx < n))
            shared_vec[tx] = vec[tidx];
        __syncthreads();

        if ((tidy < m) & (tidx < n)) {
            out[tidx*m+tidy] = mat[tidx*m+tidy] ${binary_op} shared_vec[tx];
        }
    }

    __global__ void opRowVecToMat(const ${type}* mat, const ${type}* vec, ${type}* out,
                                   const int n, const int m){
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
        const int tidy = blockIdx.y * blockDim.y + threadIdx.y;

        extern __shared__ ${type} shared_vec[];

        if ((tx == 0) & (tidy < m))
            shared_vec[ty] = vec[tidy];
        __syncthreads();

        if ((tidy < m) & (tidx < n)) {
            out[tidx*m+tidy] = mat[tidx*m+tidy] ${binary_op} shared_vec[ty];
        }
    }""")
    cache_dir=None
    ctype = dtype_to_ctype(dtype)
    tmpl = template.substitute(type=ctype, binary_op=binary_op)
    mod = SourceModule(tmpl)

    add_row_vec_kernel = mod.get_function('opRowVecToMat')
    add_col_vec_kernel = mod.get_function('opColVecToMat')
    return add_row_vec_kernel, add_col_vec_kernel


def binaryop_matvec(binary_op, x_gpu, a_gpu, axis=None, out=None, stream=None):
    """
    Applies a binary operation to a vector and each column/row of a matrix.

    The numpy broadcasting rules apply so this would yield the same result
    as `x_gpu.get()` op `a_gpu.get()` in host-code.

    Parameters
    ----------
    binary_op : string, ['+', '-', '/', '*' '%']
        The operator to apply
    x_gpu : pycuda.gpuarray.GPUArray
        Matrix to which to add the vector.
    a_gpu : pycuda.gpuarray.GPUArray
        Vector to add to `x_gpu`.
    axis : int (optional)
        The axis onto which the vector is added. By default this is
        determined automatically by using the first axis with the correct
        dimensionality.
    out : pycuda.gpuarray.GPUArray (optional)
        Optional destination matrix.
    stream : pycuda.driver.Stream (optional)
        Optional Stream in which to perform this calculation.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        result of `x_gpu` + `a_gpu`
    """
    if axis is None:
        if len(a_gpu.shape) == 1:
            if a_gpu.shape[0] == x_gpu.shape[1]:
                axis = 1
            else:
                raise ValueError(
                    "operands could not be broadcast together "
                    "with shapes %s %s" % (x_gpu.shape, a_gpu.shape))
        elif a_gpu.shape[1] == x_gpu.shape[1]:  # numpy matches inner axes first
            axis = 1
        elif a_gpu.shape[0] == x_gpu.shape[0]:
            axis = 0
        else:
                raise ValueError(
                    "operands could not be broadcast together "
                    "with shapes %s %s" % (x_gpu.shape, a_gpu.shape))
    else:
        if axis < 0:
            axis += 2
        if axis > 1:
            raise ValueError('invalid axis')

    if binary_op not in ['+', '-', '/', '*', '%']:
        raise ValueError('invalid operator')

    row_kernel, col_kernel = _get_binaryop_vecmat_kernel(x_gpu.dtype, binary_op)
    n, m = np.int32(x_gpu.shape[0]), np.int32(x_gpu.shape[1])

    block = (24, 24, 1)
    gridx = int(n // block[0] + 1 * (n % block[0] != 0))
    gridy = int(m // block[1] + 1 * (m % block[1] != 0))
    grid = (gridx, gridy, 1)

    if out is None:
        alloc = _global_cublas_allocator
        out = gpuarray.empty_like(x_gpu)
    else:
        assert out.dtype == x_gpu.dtype
        assert out.shape == x_gpu.shape

    if x_gpu.flags.c_contiguous:
        if axis == 0:
            col_kernel(x_gpu, a_gpu, out, n, m,
                       block=block, grid=grid, stream=stream,
                       shared=24*x_gpu.dtype.itemsize)
        elif axis == 1:
            row_kernel(x_gpu, a_gpu, out, n, m,
                       block=block, grid=grid, stream=stream,
                       shared=24*x_gpu.dtype.itemsize)
    else:
        if axis == 0:
            row_kernel(x_gpu, a_gpu, out, m, n,
                       block=block, grid=grid, stream=stream,
                       shared=24*x_gpu.dtype.itemsize)
        elif axis == 1:
            col_kernel(x_gpu, a_gpu, out, m, n,
                       block=block, grid=grid, stream=stream,
                       shared=24*x_gpu.dtype.itemsize)
    return out

import operator

def binaryop_2d(c_op, py_op, commutative, x_gpu, y_gpu):
    if x_gpu.flags.c_contiguous != y_gpu.flags.c_contiguous:
        raise ValueError('unsupported combination of input order')

    if x_gpu.shape == y_gpu.shape:
        return py_op(x_gpu, y_gpu)
    elif x_gpu.size == 1:
        return py_op(x_gpu.get().reshape(()), y_gpu)
    elif y_gpu.size == 1:
        return py_op(x_gpu, y_gpu.get().reshape(()))

    if len(x_gpu.shape) == 2:
        m, n = x_gpu.shape
        if y_gpu.shape == (n,):
            return binaryop_matvec(c_op, x_gpu, y_gpu, axis=1)
        elif y_gpu.shape == (1, n):
            return binaryop_matvec(c_op, x_gpu, y_gpu[0], axis=1)
        elif y_gpu.shape == (m, 1):
            return binaryop_matvec(c_op, x_gpu, y_gpu.ravel(), axis=0)

    if len(y_gpu.shape) == 2 and commutative:
        m, n = y_gpu.shape
        if x_gpu.shape == (n,):
            return binaryop_matvec(c_op, y_gpu, x_gpu, axis=1)
        elif x_gpu.shape == (1, n):
            return binaryop_matvec(c_op, y_gpu, x_gpu[0], axis=1)
        elif x_gpu.shape == (m, 1):
            return binaryop_matvec(c_op, y_gpu, x_gpu.ravel(), axis=0)

    raise TypeError("unsupported combination of shapes")

def add(x_gpu, y_gpu):
    """
    Adds two scalars, vectors, or matrices.

    The numpy broadcasting rules apply so this would yield the same result
    as `x_gpu.get()` + `y_gpu.get()` in host code.

    Parameters
    ----------
    x_gpu, y_gpu : pycuda.gpuarray.GPUArray
        The arrays to be added.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        Equivalent to `x_gpu.get()` + `y_gpu.get()`.

    Notes
    -----
    The `out` and `stream` options are not supported because `GPUArray.__add__`
    doesn't provide them.
    """

    return binaryop_2d("+", operator.add, True, x_gpu, y_gpu)

def subtract(x_gpu, y_gpu):
    """
    Subtracts two scalars, vectors, or matrices with broadcasting.

    The numpy broadcasting rules apply so this would yield the same result
    as `x_gpu.get()` - `y_gpu.get()` in host code.

    Parameters
    ----------
    x_gpu, y_gpu : pycuda.gpuarray.GPUArray
        The arrays to be subtracted.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        Equivalent to `x_gpu.get()` - `y_gpu.get()`.

    Notes
    -----
    The `out` and `stream` options are not supported because `GPUArray.__sub__`
    doesn't provide them.
    """

    return binaryop_2d("-", operator.sub, False, x_gpu, y_gpu)

def multiply(x_gpu, y_gpu):
    """
    Multiplies two scalars, vectors, or matrices with broadcasting.

    The numpy broadcasting rules apply so this would yield the same result
    as `x_gpu.get()` * `y_gpu.get()` in host code.

    Parameters
    ----------
    x_gpu, y_gpu : pycuda.gpuarray.GPUArray
        The arrays to be multiplied.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        Equivalent to `x_gpu.get()` * `y_gpu.get()`.

    Notes
    -----
    The `out` and `stream` options are not supported because `GPUArray.__mul__`
    doesn't provide them.
    """

    return binaryop_2d("*", operator.mul, True, x_gpu, y_gpu)

def divide(x_gpu, y_gpu):
    """
    Divides two scalars, vectors, or matrices with broadcasting.

    The numpy broadcasting rules apply so this would yield the same result
    as `x_gpu.get()` / `y_gpu.get()` in host code.

    Parameters
    ----------
    x_gpu, y_gpu : pycuda.gpuarray.GPUArray
        The arrays to be divided.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        Equivalent to `x_gpu.get()` / `y_gpu.get()`.

    Notes
    -----
    The `out` and `stream` options are not supported because `GPUArray.__div__`
    doesn't provide them.
    """

    return binaryop_2d("/", operator.truediv, False, x_gpu, y_gpu)

def add_matvec(x_gpu, a_gpu, axis=None, out=None, stream=None):
    """
    Adds a vector to each column/row of the matrix.

    The numpy broadcasting rules apply so this would yield the same result
    as `x_gpu.get()` + `a_gpu.get()` in host-code.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Matrix to which to add the vector.
    a_gpu : pycuda.gpuarray.GPUArray
        Vector to add to `x_gpu`.
    axis : int (optional)
        The axis onto which the vector is added. By default this is
        determined automatically by using the first axis with the correct
        dimensionality.
    out : pycuda.gpuarray.GPUArray (optional)
        Optional destination matrix.
    stream : pycuda.driver.Stream (optional)
        Optional Stream in which to perform this calculation.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        Result of `x_gpu` + `a_gpu`
    """

    return binaryop_matvec('+', x_gpu, a_gpu, axis, out, stream)


def div_matvec(x_gpu, a_gpu, axis=None, out=None, stream=None):
    """
    Divides each column/row of a matrix by a vector.

    The numpy broadcasting rules apply so this would yield the same result
    as `x_gpu.get()` / `a_gpu.get()` in host-code.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Matrix to divide by the vector `a_gpu`.
    a_gpu : pycuda.gpuarray.GPUArray
        The matrix `x_gpu` will be divided by this vector.
    axis : int (optional)
        The axis on which division occurs. By default this is
        determined automatically by using the first axis with the correct
        dimensionality.
    out : pycuda.gpuarray.GPUArray (optional)
        Optional destination matrix.
    stream : pycuda.driver.Stream (optional)
        Optional Stream in which to perform this calculation.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        result of `x_gpu` / `a_gpu`
    """
    return binaryop_matvec('/', x_gpu, a_gpu, axis, out, stream)


def mult_matvec(x_gpu, a_gpu, axis=None, out=None, stream=None):
    """
    Multiplies a vector elementwise with each column/row of the matrix.

    The numpy broadcasting rules apply so this would yield the same result
    as `x_gpu.get()` * `a_gpu.get()` in host-code.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Matrix to multiply by the vector `a_gpu`.
    a_gpu : pycuda.gpuarray.GPUArray
        The matrix `x_gpu` will be multiplied by this vector.
    axis : int (optional)
        The axis on which multiplication occurs. By default this is
        determined automatically by using the first axis with the correct
        dimensionality.
    out : pycuda.gpuarray.GPUArray (optional)
        Optional destination matrix.
    stream : pycuda.driver.Stream (optional)
        Optional Stream in which to perform this calculation.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        result of `x_gpu` * `a_gpu`
    """
    return binaryop_matvec('*', x_gpu, a_gpu, axis, out, stream)

def _sum_axis(x_gpu, axis=None, out=None, calc_mean=False, ddof=0,
              keepdims=False):
    global _global_cublas_allocator
    assert isinstance(ddof, numbers.Integral)

    if axis is None or len(x_gpu.shape) <= 1:
        out_shape = (1,)*len(x_gpu.shape) if keepdims else ()
        if calc_mean == False:
            return gpuarray.sum(x_gpu).reshape(out_shape)
        else:
            return gpuarray.sum(x_gpu).reshape(out_shape) / (x_gpu.dtype.type(x_gpu.size-ddof))

    if axis < 0:
        axis += 2
    if axis > 1:
        raise ValueError('invalid axis')

    if x_gpu.flags.c_contiguous:
        n, m = x_gpu.shape[1], x_gpu.shape[0]
        lda = x_gpu.shape[1]
        trans = "n" if axis == 0 else "t"
        sum_axis, out_axis = (m, n) if axis == 0 else (n, m)
    else:
        n, m = x_gpu.shape[0], x_gpu.shape[1]
        lda = x_gpu.shape[0]
        trans = "t" if axis == 0 else "n"
        sum_axis, out_axis = (n, m) if axis == 0 else (m, n)

    if calc_mean:
        alpha = (1.0 / (sum_axis-ddof))
    else:
        alpha = 1.0
    if (x_gpu.dtype == np.complex64):
        gemv = cublas.cublasCgemv
    elif (x_gpu.dtype == np.float32):
        gemv = cublas.cublasSgemv
    elif (x_gpu.dtype == np.complex128):
        gemv = cublas.cublasZgemv
    elif (x_gpu.dtype == np.float64):
        gemv = cublas.cublasDgemv

    alloc = _global_cublas_allocator
    ons = ones((sum_axis, ), x_gpu.dtype, allocator=alloc)

    if keepdims:
        out_shape = (1, out_axis) if axis == 0 else (out_axis, 1)
    else:
        out_shape = (out_axis,)

    if out is None:
        out = gpuarray.empty(out_shape, x_gpu.dtype, alloc)
    else:
        assert out.dtype == x_gpu.dtype
        assert out.size >= out_axis

    gemv(_global_cublas_handle, trans, n, m,
         alpha, x_gpu.gpudata, lda,
         ons.gpudata, 1, 0.0, out.gpudata, 1)
    return out


def sum(x_gpu, axis=None, out=None, keepdims=False):
    """
    Compute the sum along the specified axis.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Array containing numbers whose sum is desired.
    axis : int (optional)
        Axis along which the sums are computed. The default is to
        compute the sum of the flattened array.
    out : pycuda.gpuarray.GPUArray (optional)
        Output array in which to place the result.
    keepdims : bool (optional, default False)
        If True, the axes which are reduced are left in the result as
        dimensions with size one.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        sum of elements, or sums of elements along the desired axis.
    """
    return _sum_axis(x_gpu, axis, out=out, keepdims=keepdims)


def mean(x_gpu, axis=None, out=None, keepdims=False):
    """
    Compute the arithmetic means along the specified axis.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Array containing numbers whose mean is desired.
    axis : int (optional)
        Axis along which the means are computed. The default is to
        compute the mean of the flattened array.
    out : pycuda.gpuarray.GPUArray (optional)
        Output array in which to place the result.
    keepdims : bool (optional, default False)
        If True, the axes which are reduced are left in the result as
        dimensions with size one.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        mean of elements, or means of elements along the desired axis.
    """
    return _sum_axis(x_gpu, axis, calc_mean=True, out=out, keepdims=keepdims)

def var(x_gpu, ddof=0, axis=None, stream=None, keepdims=False):
    """
    Compute the variance along the specified axis.

    Returns the variance of the array elements, a measure of the spread of a
    distribution. The variance is computed for the flattened array by default,
    otherwise over the specified axis.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Array containing numbers whose variance is desired.
    ddof : int (optional)
        "Delta Degrees of Freedom": the divisor used in computing the 
        variance is ``N - ddof``, where ``N`` is the number of elements.
        Setting ``ddof = 1`` is equivalent to applying Bessel's
        correction.
    axis : int (optional)
        Axis along which the variance are computed. The default is to
        compute the variance of the flattened array.
    stream : pycuda.driver.Stream (optional)
        Optional CUDA stream in which to perform this calculation
    keepdims : bool (optional, default False)
        If True, the axes which are reduced are left in the result as
        dimensions with size one.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        variance of elements, or variances of elements along the desired axis.
    """
    def _inplace_pow(x_gpu, p, stream):
        func = elementwise.get_pow_kernel(x_gpu.dtype)
        func.prepared_async_call(x_gpu._grid, x_gpu._block, stream,
                    p, x_gpu.gpudata, x_gpu.gpudata, x_gpu.mem_size)

    if axis is None:
        m = mean(x_gpu).get()
        out = x_gpu - m
        out **= 2
        out = _sum_axis(out, axis=None, calc_mean=True,
                        ddof=ddof, out=None, keepdims=keepdims)
    else:
        if axis < 0:
            axis += 2
        m = mean(x_gpu, axis=axis)
        out = add_matvec(x_gpu, -m, axis=1-axis, stream=stream)
        _inplace_pow(out, 2, stream)
        out = _sum_axis(out, axis=axis, calc_mean=True,
                        ddof=ddof, out=None, keepdims=keepdims)
    return out


def std(x_gpu, ddof=0, axis=None, stream=None, keepdims=False):
    """
    Compute the standard deviation along the specified axis.

    Returns the standard deviation of the array elements, a measure of the
    spread of a distribution. The standard deviation is computed for the
    flattened array by default, otherwise over the specified axis.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Array containing numbers whose std is desired.
    ddof : int (optional)
        "Delta Degrees of Freedom": the divisor used in computing the 
        variance is ``N - ddof``, where ``N`` is the number of elements.
        Setting ``ddof = 1`` is equivalent to applying Bessel's
        correction.
    axis : int (optional)
        Axis along which the std are computed. The default is to
        compute the std of the flattened array.
    stream : pycuda.driver.Stream (optional)
        Optional CUDA stream in which to perform this calculation
    keepdims : bool (optional, default False)
        If True, the axes which are reduced are left in the result as
        dimensions with size one.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray or float
        std of elements, or stds of elements along the desired axis.
    """
    def _inplace_pow(x_gpu, p, stream):
        func = elementwise.get_pow_kernel(x_gpu.dtype)
        func.prepared_async_call(x_gpu._grid, x_gpu._block, stream,
                    p, x_gpu.gpudata, x_gpu.gpudata, x_gpu.mem_size)

    if axis is None:
        return var(x_gpu, ddof=ddof, stream=stream, keepdims=keepdims) ** 0.5
    else:
        out = var(x_gpu, ddof=ddof, axis=axis, stream=stream, keepdims=keepdims)
        _inplace_pow(out, 0.5, stream)
    return out


@context_dependent_memoize
def _get_minmax_kernel(dtype, min_or_max):
    template = Template("""
    #include <pycuda-complex.hpp>

    __global__ void minmax_column_kernel(${type}* mat, ${type}* target,
                                         unsigned int *idx_target,
                                         unsigned int width,
                                         unsigned int height) {
        __shared__ ${type} max_vals[32];
        __shared__ unsigned int max_idxs[32];
        ${type} cur_max = ${init_value};
        unsigned int cur_idx = 0;
        ${type} val = 0;

        for (unsigned int i = threadIdx.x; i < height; i += 32) {
            val = mat[blockIdx.x + i * width];

            if (val ${cmp_op} cur_max) {
                cur_max = val;
                cur_idx = i;
            }
        }
        max_vals[threadIdx.x] = cur_max;
        max_idxs[threadIdx.x] = cur_idx;
        __syncthreads();

        if (threadIdx.x == 0) {
            cur_max = ${init_value};
            cur_idx = 0;

            for (unsigned int i = 0; i < 32; i++)
                if (max_vals[i] ${cmp_op} cur_max) {
                    cur_max = max_vals[i];
                    cur_idx = max_idxs[i];
                }

            target[blockIdx.x] = cur_max;
            idx_target[blockIdx.x] = cur_idx;
        }
    }

    __global__ void minmax_row_kernel(${type}* mat, ${type}* target,
                                      unsigned int* idx_target,
                                      unsigned int width,
                                      unsigned int height) {
        __shared__ ${type} max_vals[32];
        __shared__ unsigned int max_idxs[32];
        ${type} cur_max = ${init_value};
        unsigned int cur_idx = 0;
        ${type} val = 0;

        for (unsigned int i = threadIdx.x; i < width; i += 32) {
            val = mat[blockIdx.x * width + i];

            if (val ${cmp_op} cur_max) {
                cur_max = val;
                cur_idx = i;
            }
        }
        max_vals[threadIdx.x] = cur_max;
        max_idxs[threadIdx.x] = cur_idx;
        __syncthreads();

        if (threadIdx.x == 0) {
            cur_max = ${init_value};
            cur_idx = 0;

            for (unsigned int i = 0; i < 32; i++)
                if (max_vals[i] ${cmp_op} cur_max) {
                    cur_max = max_vals[i];
                    cur_idx = max_idxs[i];
                }

            target[blockIdx.x] = cur_max;
            idx_target[blockIdx.x] = cur_idx;
        }
    }
""")
    cache_dir=None
    ctype = dtype_to_ctype(dtype)
    if min_or_max=='max':
        iv = str(np.finfo(dtype).min)
        tmpl = template.substitute(type=ctype, cmp_op='>', init_value=iv)
    elif min_or_max=='min':
        iv = str(np.finfo(dtype).max)
        tmpl = template.substitute(type=ctype, cmp_op='<', init_value=iv)
    else:
        raise ValueError('invalid argument')
    mod = SourceModule(tmpl)

    minmax_col_kernel = mod.get_function('minmax_column_kernel')
    minmax_row_kernel = mod.get_function('minmax_row_kernel')
    return minmax_col_kernel, minmax_row_kernel


def _minmax_impl(a_gpu, axis, min_or_max, stream=None, keepdims=False):
    ''' Returns both max and argmax (min/argmin) along an axis.'''
    assert len(a_gpu.shape) < 3
    if iscomplextype(a_gpu.dtype):
        raise ValueError("Cannot compute min/max of complex values")

    if axis is None or len(a_gpu.shape) <= 1:  ## Note: PyCUDA doesn't have an overall argmax/argmin!
        out_shape = (1,) * len(a_gpu.shape)
        if min_or_max == 'max':
            return gpuarray.max(a_gpu).reshape(out_shape), None
        else:
            return gpuarray.min(a_gpu).reshape(out_shape), None
    else:
        if axis < 0:
            axis += 2
    assert axis in (0, 1)

    global _global_cublas_allocator
    alloc = _global_cublas_allocator

    n, m = a_gpu.shape if a_gpu.flags.c_contiguous else (a_gpu.shape[1], a_gpu.shape[0])
    col_kernel, row_kernel = _get_minmax_kernel(a_gpu.dtype, min_or_max)
    if (axis == 0 and a_gpu.flags.c_contiguous) or (axis == 1 and a_gpu.flags.f_contiguous):
        if keepdims:
            out_shape = (1, m) if axis == 0 else (m, 1)
        else:
            out_shape = (m,)
        target = gpuarray.empty(out_shape, dtype=a_gpu.dtype, allocator=alloc)
        idx = gpuarray.empty(out_shape, dtype=np.uint32, allocator=alloc)
        col_kernel(a_gpu, target, idx, np.uint32(m), np.uint32(n),
                   block=(32, 1, 1), grid=(m, 1, 1), stream=stream)
    else:
        if keepdims:
            out_shape = (1, n) if axis == 0 else (n, 1)
        else:
            out_shape = (n,)
        target = gpuarray.empty(out_shape, dtype=a_gpu, allocator=alloc)
        idx = gpuarray.empty(out_shape, dtype=np.uint32, allocator=alloc)
        row_kernel(a_gpu, target, idx, np.uint32(m), np.uint32(n),
                block=(32, 1, 1), grid=(n, 1, 1), stream=stream)
    return target, idx


def max(a_gpu, axis=None, keepdims=False):
    '''
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input array
    axis : int (optional)
        Axis along which the maxima are computed. The default is to
        compute the maximum of the flattened array.
    keepdims : bool (optional, default False)
        If True, the axes which are reduced are left in the result as
        dimensions with size one.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray or float
        maximum of elements, or maxima of elements along the desired axis.
    '''
    return _minmax_impl(a_gpu, axis, "max", keepdims=keepdims)[0]


def min(a_gpu, axis=None, keepdims=False):
    '''
    Return the minimum of an array or minimum along an axis.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input array
    axis : int (optional)
        Axis along which the minima are computed. The default is to
        compute the minimum of the flattened array.
    keepdims : bool (optional, default False)
        If True, the axes which are reduced are left in the result as
        dimensions with size one.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray or float
        minimum of elements, or minima of elements along the desired axis.
    '''
    return _minmax_impl(a_gpu, axis, "min", keepdims=keepdims)[0]


def argmax(a_gpu, axis, keepdims=False):
    '''
    Indices of the maximum values along an axis.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input array
    axis : int
        Axis along which the maxima are computed.
    keepdims : bool (optional, default False)
        If True, the axes which are reduced are left in the result as
        dimensions with size one.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        Array of indices into the array.
    '''
    if axis is None:
        raise NotImplementedError("Can't compute global argmax")
    return _minmax_impl(a_gpu, axis, "max", keepdims=keepdims)[1]


def argmin(a_gpu, axis, keepdims=False):
    '''
    Indices of the minimum values along an axis.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input array
    axis : int
        Axis along which the minima are computed.
    keepdims : bool (optional, default False)
        If True, the axes which are reduced are left in the result as
        dimensions with size one.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        Array of indices into the array.
    '''
    if axis is None:
        raise NotImplementedError("Can't compute global argmax")
    return _minmax_impl(a_gpu, axis, "min", keepdims=keepdims)[1]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
