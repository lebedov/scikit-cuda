#!/usr/bin/env python

"""
Miscellaneous PyCUDA functions.
"""

import string
from string import Template
import atexit

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np

import cuda
import cublas

try:
    import cula
    _has_cula = True
except (ImportError, OSError):
    _has_cula = False

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

    for i in xrange(len(atexit._exithandlers)):
        if atexit._exithandlers[i][0] == ctx.pop:
            del atexit._exithandlers[i]
            break
    ctx.detach()

global _global_cublas_handle
_global_cublas_handle = None
def init():
    """
    Initialize libraries used by scikits.cuda.

    Initialize the CUBLAS and CULA libraries used by high-level functions
    provided by scikits.cuda.
    
    Notes
    -----
    This function does not initialize PyCUDA; it uses whatever device
    and context were initialized in the current host thread.

    """

    # CUBLAS uses whatever device is being used by the host thread:
    global _global_cublas_handle
    _global_cublas_handle = cublas.cublasCreate()

    # culaSelectDevice() need not (and, in fact, cannot) be called
    # here because the host thread has already been bound to a GPU
    # device:
    if _has_cula:
        cula.culaInitialize()

def shutdown():
    """
    Shutdown libraries used by scikits.cuda.

    Shutdown the CUBLAS and CULA libraries used by high-level functions provided
    by scikits.cuda.

    Notes
    -----
    This function does not shutdown PyCUDA.

    """

    global _global_cublas_handle
    cublasDestroy(_global_cublas_handle)
    
    if _has_cula:
        cula.culaShutdown()

    _global_cublas_handle = None
    
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

    return np.float(string.join([str(i) for i in
                                 dev.compute_capability()], '.'))

def get_current_device():
    """
    Get the device in use by the current context.

    Returns
    -------
    d : pycuda.driver.Device
        Device in use by current context.

    """

    return drv.Device(cuda.cudaGetDevice())

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
        (MAX_GRID_DIM_X, MAX_GRID_DIM_Y)]

    """

    attrs = dev.get_attributes()
    return [attrs[drv.device_attribute.MAX_THREADS_PER_BLOCK],
            (attrs[drv.device_attribute.MAX_BLOCK_DIM_X],
             attrs[drv.device_attribute.MAX_BLOCK_DIM_Y],
             attrs[drv.device_attribute.MAX_BLOCK_DIM_Z]),
            (attrs[drv.device_attribute.MAX_GRID_DIM_X],
            attrs[drv.device_attribute.MAX_GRID_DIM_Y])]


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

    if threads_per_block != None:
        max_threads_per_block = threads_per_block

    # Assume that the maximum number of threads per block is no larger
    # than the maximum X and Y dimension of a thread block:
    assert max_threads_per_block <= max_block_dim[0]
    assert max_threads_per_block <= max_block_dim[1]

    # Assume that the maximum X and Y dimensions of a grid are the
    # same:
    max_blocks_per_grid_dim = max(max_grid_dim)
    assert max_blocks_per_grid_dim == max_grid_dim[0]
    assert max_blocks_per_grid_dim == max_grid_dim[1]

    # Actual number of thread blocks needed:
    blocks_needed = N/max_threads_per_block+1

    if blocks_needed*max_threads_per_block < max_threads_per_block*max_blocks_per_grid_dim:
        grid_x = blocks_needed
        grid_y = 1
    elif blocks_needed*max_threads_per_block < max_threads_per_block*max_blocks_per_grid_dim**2:
        grid_x = max_blocks_per_grid_dim
        grid_y = blocks_needed/max_blocks_per_grid_dim+1
    else:
        raise ValueError('array size too large')

    return (max_threads_per_block, 1, 1), (grid_x, grid_y)

def zeros(shape, dtype, allocator=drv.mem_alloc):
    """
    Return an array of the given shape and dtype filled with zeros.

    Parameters
    ----------
    shape : tuple
        Array shape.
    dtype : data-type
        Data type for the array.
    allocator : callable
        Returns an object that represents the memory allocated for
        the requested array.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        Array of zeros with the given shape and dtype.

    Notes
    -----
    This function exists to work around the following numpy bug that
    prevents pycuda.gpuarray.zeros() from working properly with
    complex types in pycuda 2011.1.2:
    http://projects.scipy.org/numpy/ticket/1898

    """

    out = gpuarray.GPUArray(shape, dtype, allocator)
    out.fill(0)
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
        Array of zeros with the shape and dtype of `a`.

    """

    out = gpuarray.GPUArray(a.shape, a.dtype, drv.mem_alloc)
    out.fill(0)
    return out

def ones(shape, dtype, allocator=drv.mem_alloc):
    """
    Return an array of the given shape and dtype filled with ones.

    Parameters
    ----------
    shape : tuple
        Array shape.
    dtype : data-type
        Data type for the array.
    allocator : callable
        Returns an object that represents the memory allocated for
        the requested array.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        Array of ones with the given shape and dtype.

    """

    out = gpuarray.GPUArray(shape, dtype, allocator)
    out.fill(1)
    return out

def ones_like(other):
    """
    Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    other : pycuda.gpuarray.GPUArray
        Array whose shape and dtype are to be used to allocate a new array.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        Array of ones with the shape and dtype of `other`.

    """

    out = gpuarray.GPUArray(other.shape, other.dtype,
                            other.allocator)
    out.fill(1)
    return out

def inf(shape, dtype, allocator=drv.mem_alloc):
    """
    Return an array of the given shape and dtype filled with infs.

    Parameters
    ----------
    shape : tuple
        Array shape.
    dtype : data-type
        Data type for the array.
    allocator : callable
        Returns an object that represents the memory allocated for
        the requested array.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        Array of infs with the given shape and dtype.

    """

    out = gpuarray.GPUArray(shape, dtype, allocator)
    out.fill(np.inf)
    return out

maxabs_mod_template = Template("""
#include <pycuda-complex.hpp>

#if ${use_double}
#define REAL_TYPE double
#if ${use_complex}
#define TYPE pycuda::complex<double>
#else
#define TYPE double
#endif
#else
#define REAL_TYPE float
#if ${use_complex}
#define TYPE pycuda::complex<float>
#else
#define TYPE float
#endif
#endif

// This kernel is only meant to be run in one thread;
// N must contain the length of x:
__global__ void maxabs(TYPE *x, REAL_TYPE *m, unsigned int N) {
    unsigned int idx = threadIdx.x;

    REAL_TYPE result, temp;

    if (idx == 0) {
        result = abs(x[0]);
        for (unsigned int i = 1; i < N; i++) {
           temp = abs(x[i]);
           if (temp > result)
               result = temp;
        }
        m[0] = result;
    }
}
""")

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
        Length 1 array containing the maximum absolute value in
        `x_gpu`.

    Notes
    -----
    This function could be made faster by computing the absolute
    values of the input array in parallel.

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

    use_double = int(x_gpu.dtype in [np.float64, np.complex128])
    use_complex = int(x_gpu.dtype in [np.complex64, np.complex128])
    real_type = np.float64 if use_double else np.float32

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir = None
    maxabs_mod = \
               SourceModule(maxabs_mod_template.substitute(use_double=use_double,
                                                           use_complex=use_complex),
                            cache_dir=cache_dir)

    maxabs = maxabs_mod.get_function("maxabs")
    m_gpu = gpuarray.empty(1, real_type)
    maxabs(x_gpu, m_gpu, np.uint32(x_gpu.size),
           block=(1, 1, 1), grid=(1, 1))

    return m_gpu

cumsum_template = Template("""
#include <pycuda-complex.hpp>

#if ${use_double}
#define REAL_TYPE double
#if ${use_complex}
#define TYPE pycuda::complex<double>
#else
#define TYPE double
#endif
#else
#define REAL_TYPE float
#if ${use_complex}
#define TYPE pycuda::complex<float>
#else
#define TYPE float
#endif
#endif

// This kernel should only be invoked on a single thread:
__global__ void cumsum(TYPE *x, TYPE *c, unsigned int N) {
    unsigned int idx = threadIdx.x;

    if (idx == 0) {
        TYPE sum_curr = 0;
        for (unsigned i = 0; i < N; i++) {
            sum_curr += x[i];
            c[i] = sum_curr;
        }
    }
}
""")

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
    This function could be made faster by using a parallel prefix sum.

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

    use_double = int(x_gpu.dtype in [np.float64, np.complex128])
    use_complex = int(x_gpu.dtype in [np.complex64, np.complex128])

    cumsum_mod = \
                SourceModule(cumsum_template.substitute(use_double=use_double,
                                                        use_complex=use_complex))

    cumsum = cumsum_mod.get_function("cumsum")
    c_gpu = gpuarray.empty_like(x_gpu)
    cumsum(x_gpu, c_gpu, np.uint32(x_gpu.size),
           block=(1, 1, 1), grid=(1, 1))

    return c_gpu

diff_mod_template = Template("""
#include <pycuda-complex.hpp>

#if ${use_double}
#define REAL_TYPE double
#if ${use_complex}
#define TYPE pycuda::complex<double>
#else
#define TYPE double
#endif
#else
#define REAL_TYPE float
#if ${use_complex}
#define TYPE pycuda::complex<float>
#else
#define TYPE float
#endif
#endif

__global__ void diff(TYPE *x, TYPE *y, unsigned int N) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < N-1) {
        y[idx] = x[idx+1]-x[idx];
    }
}
""")

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

    if len(x_gpu.shape) > 1:
        raise ValueError('input must be 1D vector')

    use_double = int(x_gpu.dtype in [np.float64, np.complex128])
    use_complex = int(x_gpu.dtype in [np.complex64, np.complex128])

    # Get block/grid sizes:
    dev = get_current_device()
    block_dim, grid_dim = select_block_grid_sizes(dev, x_gpu.shape)

    # Set this to False when debugging to make sure the compiled kernel is
    # not cached:
    cache_dir=None
    diff_mod = \
             SourceModule(diff_mod_template.substitute(use_double=use_double,
                                                       use_complex=use_complex),
                          cache_dir=cache_dir)
    diff = diff_mod.get_function("diff")

    N = x_gpu.size
    y_gpu = gpuarray.empty((N-1,), x_gpu.dtype)
    diff(x_gpu, y_gpu, np.uint32(N),
         block=block_dim,
         grid=grid_dim)

    return y_gpu

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
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
