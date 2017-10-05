#!/usr/bin/env python

"""
PyCUDA-based randomized linear algebra functions.
"""

from __future__ import absolute_import, division

from pprint import pprint
from string import Template
from pycuda.tools import context_dependent_memoize
from pycuda.compiler import SourceModule
from pycuda.reduction import ReductionKernel
from pycuda import curandom

from pycuda import cumath

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.elementwise as el
import pycuda.tools as tools
import numpy as np

from . import cublas
from . import misc
from . import linalg

rand = curandom.MRG32k3aRandomNumberGenerator()

import sys
if sys.version_info < (3,):
    range = xrange


class LinAlgError(Exception):
    """Randomized Linear Algebra Error."""
    pass


try:
    from . import cula
    _has_cula = True
except (ImportError, OSError):
    _has_cula = False

from .misc import init, add_matvec, div_matvec, mult_matvec
from .linalg import hermitian, transpose

# Get installation location of C headers:
from . import install_headers



def rsvd(a_gpu, k=None, p=0, q=0, method="standard", handle=None):
    """
    Randomized Singular Value Decomposition.
    
    Randomized algorithm for computing the approximate low-rank singular value 
    decomposition of a rectangular (m, n) matrix `a` with target rank `k << n`. 
    The input matrix a is factored as `a = U * diag(s) * Vt`. The right singluar 
    vectors are the columns of the real or complex unitary matrix `U`. The left 
    singular vectors are the columns of the real or complex unitary matrix `V`. 
    The singular values `s` are non-negative and real numbers.

    The paramter `p` is a oversampling parameter to improve the approximation. 
    A value between 2 and 10 is recommended.
    
    The paramter `q` specifies the number of normlized power iterations
    (subspace iterations) to reduce the approximation error. This is recommended 
    if the the singular values decay slowly and in practice 1 or 2 iterations 
    achive good results. However, computing power iterations is increasing the
    computational time. 
    
    If k > (n/1.5), partial SVD or trancated SVD might be faster. 
    
    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Real/complex input matrix  `a` with dimensions `(m, n)`.
    k : int
        `k` is the target rank of the low-rank decomposition, k << min(m,n). 
    p : int
        `p` sets the oversampling parameter (default k=0).
    q : int
        `q` sets the number of power iterations (default=0).
    method : `{'standard', 'fast'}`
        'standard' : Standard algorithm as described in [1, 2]
        'fast' : Version II algorithm as described in [2]   
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.                

    Returns
    -------
    u_gpu : pycuda.gpuarray
        Right singular values, array of shape `(m, k)`.
    s_gpu : pycuda.gpuarray
        Singular values, 1-d array of length `k`.
    vt_gpu : pycuda.gpuarray
        Left singular values, array of shape `(k, n)`.

    Notes
    -----
    Double precision is only supported if the standard version of the
    CULA Dense toolkit is installed.

    This function destroys the contents of the input matrix.
    
    Arrays are assumed to be stored in column-major order, i.e., order='F'.
    
    Input matrix of shape `(m, n)`, where `n>m` is not supported yet.

    References
    ----------
    N. Halko, P. Martinsson, and J. Tropp.
    "Finding structure with randomness: probabilistic
    algorithms for constructing approximate matrix
    decompositions" (2009).
    (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).
    
    S. Voronin and P.Martinsson. 
    "RSVDPACK: Subroutines for computing partial singular value 
    decompositions via randomized sampling on single core, multi core, 
    and GPU architectures" (2015).
    (available at `arXiv <http://arxiv.org/abs/1502.05366>`_).

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> from skcuda import linalg, rlinalg
    >>> linalg.init()
    >>> rlinalg.init()
    
    >>> #Randomized SVD decomposition of the square matrix `a` with single precision.
    >>> #Note: There is no gain to use rsvd if k > int(n/1.5)
    >>> a = np.array(np.random.randn(5, 5), np.float32, order='F')
    >>> a_gpu = gpuarray.to_gpu(a)  
    >>> U, s, Vt = rlinalg.rsvd(a_gpu, k=5, method='standard')
    >>> np.allclose(a, np.dot(U.get(), np.dot(np.diag(s.get()), Vt.get())), 1e-4)
    True
    
    >>> #Low-rank SVD decomposition with target rank k=2
    >>> a = np.array(np.random.randn(5, 5), np.float32, order='F')
    >>> a_gpu = gpuarray.to_gpu(a)  
    >>> U, s, Vt = rlinalg.rsvd(a_gpu, k=2, method='standard')
    
    """
    
    #*************************************************************************
    #***        Author: N. Benjamin Erichson <nbe@st-andrews.ac.uk>        ***
    #***                         <September, 2015>                         ***
    #***                       License: BSD 3 clause                       ***
    #*************************************************************************

    if not _has_cula:
        raise NotImplementedError('CULA not installed')

    if handle is None:
        handle = misc._global_cublas_handle

    alloc = misc._global_cublas_allocator

    # The free version of CULA only supports single precision floating
    data_type = a_gpu.dtype.type
    real_type = np.float32

    if data_type == np.complex64:
        cula_func_gesvd = cula.culaDeviceCgesvd
        cublas_func_gemm = cublas.cublasCgemm
        copy_func = cublas.cublasCcopy
        alpha = np.complex64(1.0)
        beta = np.complex64(0.0)
        TRANS_type = 'C'
        isreal = False
    elif data_type == np.float32:
        cula_func_gesvd = cula.culaDeviceSgesvd
        cublas_func_gemm = cublas.cublasSgemm
        copy_func = cublas.cublasScopy
        alpha = np.float32(1.0)
        beta = np.float32(0.0)
        TRANS_type = 'T'
        isreal = True
    else:
        if cula._libcula_toolkit == 'standard':
            if data_type == np.complex128:
                cula_func_gesvd = cula.culaDeviceZgesvd
                cublas_func_gemm = cublas.cublasZgemm
                copy_func = cublas.cublasZcopy
                alpha = np.complex128(1.0)
                beta = np.complex128(0.0)
                TRANS_type = 'C'
                isreal = False
            elif data_type == np.float64:
                cula_func_gesvd = cula.culaDeviceDgesvd
                cublas_func_gemm = cublas.cublasDgemm
                copy_func = cublas.cublasDcopy
                alpha = np.float64(1.0)
                beta = np.float64(0.0)
                TRANS_type = 'T'
                isreal = True
            else:
                raise ValueError('unsupported type')
            real_type = np.float64
        else:
            raise ValueError('double precision not supported')

    #CUDA assumes that arrays are stored in column-major order
    m, n = np.array(a_gpu.shape, int)
    if n>m : raise ValueError('input matrix of shape (m,n), where n>m is not supported')    
    
    #Set k 
    if k == None : raise ValueError('k must be provided')
    if k > n or k < 1: raise ValueError('k must be 0 < k <= n')
    kt = k
    k = k + p
    if k > n: k=n

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Generate a random sampling matrix O
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if isreal==False: 
        Oimag_gpu = gpuarray.empty((n,k), real_type, order="F", allocator=alloc)
        Oreal_gpu = gpuarray.empty((n,k), real_type, order="F", allocator=alloc) 
        O_gpu = gpuarray.empty((n,k), data_type, order="F", allocator=alloc) 
        rand.fill_uniform(Oimag_gpu)
        rand.fill_uniform(Oreal_gpu)
        O_gpu = Oreal_gpu + 1j * Oimag_gpu
        O_gpu = O_gpu.T * 2 - 1 #Scale to [-1,1] 
    else:
        O_gpu = gpuarray.empty((n,k), real_type, order="F", allocator=alloc) 
        rand.fill_uniform(O_gpu) #Draw random samples from a ~ Uniform(-1,1) distribution
        O_gpu = O_gpu * 2 - 1 #Scale to [-1,1]  
    
               
       
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Build sample matrix Y : Y = A * O
    #Note: Y should approximate the range of A
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    #Allocate Y    
    Y_gpu = gpuarray.zeros((m,k), data_type, order="F", allocator=alloc)    
    #Dot product Y = A * O    
    cublas_func_gemm(handle, 'n', 'n', m, k, n, alpha, 
                         a_gpu.gpudata, m, O_gpu.gpudata, n, 
                         beta, Y_gpu.gpudata, m  )  
      
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Orthogonalize Y using economic QR decomposition: Y=QR
    #If q > 0 perfrom q subspace iterations
    #Note: economic QR just returns Q, and destroys Y_gpu
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    if q > 0:
        Z_gpu = gpuarray.empty((n,k), data_type, order="F", allocator=alloc)    

        for i in np.arange(1, q+1 ):
            if( (2*i-2)%q == 0 ):
                Y_gpu = linalg.qr(Y_gpu, 'economic')
            
            cublas_func_gemm(handle, TRANS_type, 'n', n, k, m, alpha, 
                         a_gpu.gpudata, m, Y_gpu.gpudata, m, 
                         beta, Z_gpu.gpudata, n  )

            if( (2*i-1)%q == 0 ):
                Z_gpu = linalg.qr(Z_gpu, 'economic')
       
            cublas_func_gemm(handle, 'n', 'n', m, k, n, alpha, 
                         a_gpu.gpudata, m, Z_gpu.gpudata, n, 
                         beta, Y_gpu.gpudata, m  )
                         
        #End for
     #End if   
    
    Q_gpu = linalg.qr(Y_gpu, 'economic')
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Project the data matrix a into a lower dimensional subspace
    #B = Q.T * A 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    #Allocate B    
    B_gpu = gpuarray.empty((k,n), data_type, order="F", allocator=alloc)    
    cublas_func_gemm(handle, TRANS_type, 'n', k, n, m, alpha, 
                         Q_gpu.gpudata, m, a_gpu.gpudata, m, 
                         beta, B_gpu.gpudata, k  )
    
    if method == 'standard':
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Singular Value Decomposition
        #Note: B = U" * S * Vt
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
        #gesvd(jobu, jobvt, m, n, int(a), lda, int(s), int(u), ldu, int(vt), ldvt)
        #Allocate s, U, Vt for economic SVD
        #Note: singular values are always real
        s_gpu = gpuarray.empty(k, real_type, order="F", allocator=alloc)
        U_gpu = gpuarray.empty((k,k), data_type, order="F", allocator=alloc)
        Vt_gpu = gpuarray.empty((k,n), data_type, order="F", allocator=alloc)
        
        #Economic SVD
        cula_func_gesvd('S', 'S', k, n, int(B_gpu.gpudata), k, int(s_gpu.gpudata), 
                        int(U_gpu.gpudata), k, int(Vt_gpu.gpudata), k)
    
        #Compute right singular vectors as U = Q * U"
        cublas_func_gemm(handle, 'n', 'n', m, k, k, alpha, 
                         Q_gpu.gpudata, m, U_gpu.gpudata, k, 
                         beta, Q_gpu.gpudata, m  )
        U_gpu =  Q_gpu   #Set pointer            

        # Free internal CULA memory:
        cula.culaFreeBuffers()      
         
        #Return
        return U_gpu[ : , 0:kt ], s_gpu[ 0:kt ], Vt_gpu[ 0:kt , : ]
    
        
    elif method == 'fast':
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Orthogonalize B.T using reduced QR decomposition: B.T = Q" * R"
        #Note: reduced QR returns Q and R, and destroys B_gpu
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                
        if isreal==True: 
            B_gpu = transpose(B_gpu) #transpose B
        else:
            B_gpu = hermitian(B_gpu) #transpose B
        
        Qstar_gpu, Rstar_gpu = linalg.qr(B_gpu, 'reduced')

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Singular Value Decomposition of R"
        #Note: R" = U" * S" * Vt"
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
        #gesvd(jobu, jobvt, m, n, int(a), lda, int(s), int(u), ldu, int(vt), ldvt)
        #Allocate s, U, Vt for economic SVD
        #Note: singular values are always real
        s_gpu = gpuarray.empty(k, real_type, order="F", allocator=alloc)
        Ustar_gpu = gpuarray.empty((k,k), data_type, order="F", allocator=alloc)
        Vtstar_gpu = gpuarray.empty((k,k), data_type, order="F", allocator=alloc)
        
        #Economic SVD
        cula_func_gesvd('A', 'A', k, k, int(Rstar_gpu.gpudata), k, int(s_gpu.gpudata), 
                        int(Ustar_gpu.gpudata), k, int(Vtstar_gpu.gpudata), k)
    
   
        #Compute right singular vectors as U = Q * Vt.T"
        cublas_func_gemm(handle, 'n', TRANS_type, m, k, k, alpha, 
                         Q_gpu.gpudata, m, Vtstar_gpu.gpudata, k, 
                         beta, Q_gpu.gpudata, m  )
        U_gpu =  Q_gpu   #Set pointer  

        #Compute left singular vectors as Vt = U".T * Q".T  
        Vt_gpu = gpuarray.empty((k,n), data_type, order="F", allocator=alloc)

        cublas_func_gemm(handle, TRANS_type, TRANS_type, k, n, k, alpha, 
                         Ustar_gpu.gpudata, k, Qstar_gpu.gpudata, n, 
                         beta, Vt_gpu.gpudata, k  )
    
       

        # Free internal CULA memory:
        cula.culaFreeBuffers()      
         
        #Return
        return U_gpu[ : , 0:kt ], s_gpu[ 0:kt ], Vt_gpu[ 0:kt , : ]    
    #End if



def rdmd(a_gpu, k=None, p=5, q=1, modes='exact', method_rsvd='standard', return_amplitudes=False, return_vandermonde=False, handle=None):
    """
    Randomized Dynamic Mode Decomposition.

    Dynamic Mode Decomposition (DMD) is a data processing algorithm which
    allows to decompose a matrix `a` in space and time.
    The matrix `a` is decomposed as `a = FBV`, where the columns of `F`
    contain the dynamic modes. The modes are ordered corresponding
    to the amplitudes stored in the diagonal matrix `B`. `V` is a Vandermonde
    matrix describing the temporal evolution.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Real/complex input matrix  `a` with dimensions `(m, n)`.
    k : int, optional
        If `k < (n-1)` low-rank Dynamic Mode Decomposition is computed.
    p : int
        `p` sets the oversampling parameter for rSVD (default k=5).
    q : int
        `q` sets the number of power iterations for rSVD (default=1).
    modes : `{'standard', 'exact'}`
        'standard' : uses the standard definition to compute the dynamic modes,
                    `F = U * W`.
        'exact' : computes the exact dynamic modes, `F = Y * V * (S**-1) * W`.    
    method_rsvd : `{'standard', 'fast'}`
        'standard' : (default) Standard algorithm as described in [1, 2] 
        'fast' : Version II algorithm as described in [2]
    return_amplitudes : bool `{True, False}` 
        True: return amplitudes in addition to dynamic modes. 
    return_vandermonde : bool `{True, False}`
        True: return Vandermonde matrix in addition to dynamic modes and amplitudes.    
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.

    Returns
    -------
    f_gpu : pycuda.gpuarray.GPUArray
        Matrix containing the dynamic modes of shape `(m, n-1)`  or `(m, k)`.
    b_gpu : pycuda.gpuarray.GPUArray
        1-D array containing the amplitudes of length `min(n-1, k)`.
    v_gpu : pycuda.gpuarray.GPUArray
        Vandermonde matrix of shape `(n-1, n-1)`  or `(k, n-1)`.

    Notes
    -----
    Double precision is only supported if the standard version of the
    CULA Dense toolkit is installed.

    This function destroys the contents of the input matrix.

    Arrays are assumed to be stored in column-major order, i.e., order='F'.

    References
    ----------
    N. B. Erichson and C. Donovan.
    "Randomized Low-Rank Dynamic Mode Decomposition for Motion Detection"
    Under Review.    
    
    N. Halko, P. Martinsson, and J. Tropp.
    "Finding structure with randomness: probabilistic
    algorithms for constructing approximate matrix
    decompositions" (2009).
    (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).

    J. H. Tu, et al.
    "On dynamic mode decomposition: theory and applications."
    arXiv preprint arXiv:1312.0041 (2013).
    
    Examples
    --------
    >>> #Numpy
    >>> import numpy as np
    >>> #Plot libs
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> from matplotlib import cm
    >>> #GPU DMD libs
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> from skcuda import linalg, rlinalg
    >>> linalg.init()
    >>> rlinalg.init()
    
    >>> # Define time and space discretizations
    >>> x=np.linspace( -15, 15, 200)
    >>> t=np.linspace(0, 8*np.pi , 80) 
    >>> dt=t[2]-t[1]
    >>> X, T = np.meshgrid(x,t)
    >>> # Create two patio-temporal patterns
    >>> F1 = 0.5* np.cos(X)*(1.+0.* T)
    >>> F2 = ( (1./np.cosh(X)) * np.tanh(X)) *(2.*np.exp(1j*2.8*T))
    >>> # Add both signals
    >>> F = (F1+F2)
    
    >>> #Plot dataset
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(231, projection='3d')
    >>> ax = fig.gca(projection='3d')
    >>> surf = ax.plot_surface(X, T, F, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    >>> ax.set_zlim(-1, 1)
    >>> plt.title('F')
    >>> ax = fig.add_subplot(232, projection='3d')
    >>> ax = fig.gca(projection='3d')
    >>> surf = ax.plot_surface(X, T, F1, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    >>> ax.set_zlim(-1, 1)
    >>> plt.title('F1')
    >>> ax = fig.add_subplot(233, projection='3d')
    >>> ax = fig.gca(projection='3d')
    >>> surf = ax.plot_surface(X, T, F2, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    >>> ax.set_zlim(-1, 1)
    >>> plt.title('F2')
    
    >>> #Dynamic Mode Decomposition
    >>> F_gpu = np.array(F.T, np.complex64, order='F')
    >>> F_gpu = gpuarray.to_gpu(F_gpu) 
    >>> Fmodes_gpu, b_gpu, V_gpu, omega_gpu = rlinalg.rdmd(F_gpu, k=2, p=0, q=1, modes='exact', return_amplitudes=True, return_vandermonde=True)
    >>> omega = omega_gpu.get()
    >>> plt.scatter(omega.real, omega.imag, marker='o', c='r')

    >>> #Recover original signal
    >>> F1tilde = np.dot(Fmodes_gpu[:,0:1].get() , np.dot(b_gpu[0].get(), V_gpu[0:1,:].get() ) )
    >>> F2tilde = np.dot(Fmodes_gpu[:,1:2].get() , np.dot(b_gpu[1].get(), V_gpu[1:2,:].get() ) )
    
    >>> #Plot DMD modes
    >>> #Mode 0
    >>> ax = fig.add_subplot(235, projection='3d')
    >>> ax = fig.gca(projection='3d')
    >>> surf = ax.plot_surface(X[0:F1tilde.shape[1],:], T[0:F1tilde.shape[1],:], F1tilde.T, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    >>> ax.set_zlim(-1, 1)
    >>> plt.title('F1_tilde')
    >>> #Mode 1
    >>> ax = fig.add_subplot(236, projection='3d')
    >>> ax = fig.gca(projection='3d')
    >>> surf = ax.plot_surface(X[0:F2tilde.shape[1],:], T[0:F2tilde.shape[1],:], F2tilde.T, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    >>> ax.set_zlim(-1, 1)
    >>> plt.title('F2_tilde')
    >>> plt.show()     
    """

    #*************************************************************************
    #***        Author: N. Benjamin Erichson <nbe@st-andrews.ac.uk>        ***
    #***                              <2015>                               ***
    #***                       License: BSD 3 clause                       ***
    #*************************************************************************

    if not _has_cula:
        raise NotImplementedError('CULA not installed')

    if handle is None:
        handle = misc._global_cublas_handle

    alloc = misc._global_cublas_allocator

    # The free version of CULA only supports single precision floating
    data_type = a_gpu.dtype.type
    real_type = np.float32

    if data_type == np.complex64:
        cula_func_gesvd = cula.culaDeviceCgesvd
        cublas_func_gemm = cublas.cublasCgemm
        cublas_func_dgmm = cublas.cublasCdgmm
        cula_func_gels = cula.culaDeviceCgels
        copy_func = cublas.cublasCcopy
        transpose_func = cublas.cublasCgeam
        alpha = np.complex64(1.0)
        beta = np.complex64(0.0)
        TRANS_type = 'C'
        isreal = False
    elif data_type == np.float32:
        cula_func_gesvd = cula.culaDeviceSgesvd
        cublas_func_gemm = cublas.cublasSgemm
        cublas_func_dgmm = cublas.cublasSdgmm
        cula_func_gels = cula.culaDeviceSgels
        copy_func = cublas.cublasScopy
        transpose_func = cublas.cublasSgeam
        alpha = np.float32(1.0)
        beta = np.float32(0.0)
        TRANS_type = 'T'
        isreal = True
    else:
        if cula._libcula_toolkit == 'standard':
            if data_type == np.complex128:
                cula_func_gesvd = cula.culaDeviceZgesvd
                cublas_func_gemm = cublas.cublasZgemm
                cublas_func_dgmm = cublas.cublasZdgmm
                cula_func_gels = cula.culaDeviceZgels
                copy_func = cublas.cublasZcopy
                transpose_func = cublas.cublasZgeam
                alpha = np.complex128(1.0)
                beta = np.complex128(0.0)
                TRANS_type = 'C'
                isreal = False
            elif data_type == np.float64:
                cula_func_gesvd = cula.culaDeviceDgesvd
                cublas_func_gemm = cublas.cublasDgemm
                cublas_func_dgmm = cublas.cublasDdgmm
                cula_func_gels = cula.culaDeviceDgels
                copy_func = cublas.cublasDcopy
                transpose_func = cublas.cublasDgeam
                alpha = np.float64(1.0)
                beta = np.float64(0.0)
                TRANS_type = 'T'
                isreal = True
            else:
                raise ValueError('unsupported type')
            real_type = np.float64
        else:
            raise ValueError('double precision not supported')

    #CUDA assumes that arrays are stored in column-major order
    m, n = np.array(a_gpu.shape, int)
    nx = n-1
    #Set k     
    if k == None : k = nx
    if k > nx or k < 1: raise ValueError('k is not valid')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Split data into lef and right snapshot sequence
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Note: we need a copy of X_gpu, because SVD destroys X_gpu
    #While Y_gpu is just a pointer
    X_gpu = gpuarray.empty((m, n), data_type, order="F", allocator=alloc) 
    copy_func(handle, X_gpu.size, int(a_gpu.gpudata), 1, int(X_gpu.gpudata), 1)
    X_gpu = X_gpu[:, :nx]    
    Y_gpu = a_gpu[:, 1:] 

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Randomized Singular Value Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    U_gpu, s_gpu, Vh_gpu = rsvd(X_gpu, k=k, p=p, q=q, 
                                method=method_rsvd, handle=handle)
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Solve the LS problem to find estimate for M using the pseudo-inverse    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    #real: M = U.T * Y * Vt.T * S**-1
    #complex: M = U.H * Y * Vt.H * S**-1
    #Let G = Y * Vt.H * S**-1, hence M = M * G
    
    #Allocate G and M  
    G_gpu = gpuarray.empty((m,k), data_type, order="F", allocator=alloc)    
    M_gpu = gpuarray.empty((k,k), data_type, order="F", allocator=alloc)     
    
    #i) s = s **-1 (inverse)
    if data_type == np.complex64 or data_type == np.complex128:
        s_gpu = 1/s_gpu
        s_gpu = s_gpu + 1j * gpuarray.zeros_like(s_gpu)
    else:
        s_gpu = 1.0/s_gpu

    
    #ii) real/complex: scale Vs =  Vt* x diag(s**-1) 
    Vs_gpu = gpuarray.empty((nx,k), data_type, order="F", allocator=alloc) 
    lda = max(1, Vh_gpu.strides[1] // Vh_gpu.dtype.itemsize)
    ldb = max(1, Vs_gpu.strides[1] // Vs_gpu.dtype.itemsize)     
    transpose_func(handle, TRANS_type, TRANS_type, nx, k,
                   alpha, int(Vh_gpu.gpudata), lda, beta, int(Vh_gpu.gpudata), lda,
                   int(Vs_gpu.gpudata), ldb)    
    
    
    
    cublas_func_dgmm(handle, 'r', nx, k, int(Vs_gpu.gpudata), nx, 
                     int(s_gpu.gpudata), 1 , int(Vs_gpu.gpudata), nx)
   

    #iii) real: G = Y * Vs , complex: G = Y x Vs
    cublas_func_gemm(handle, 'n', 'n', m, k, nx, alpha, 
                     int(Y_gpu.gpudata), m, int(Vs_gpu.gpudata), nx, 
                        beta, int(G_gpu.gpudata), m )      
   
    
    #iv) real/complex: M = U* x G 
    cublas_func_gemm(handle, TRANS_type, 'n', k, k, m, alpha, 
                     int(U_gpu.gpudata), m, int(G_gpu.gpudata), m, 
                    beta, int(M_gpu.gpudata), k )     

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Eigen Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Note: If a_gpu is real the imag part is omitted
    Vr_gpu, w_gpu = linalg.eig(M_gpu, 'N', 'V', 'F')
    omega = cumath.log(w_gpu)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    #Compute DMD Modes 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    F_gpu = gpuarray.empty((m,k), data_type, order="F", allocator=alloc)    
    modes = modes.lower()
    if modes == 'exact': #Compute (exact) DMD modes: F = Y * V * S**-1 * W = G * W
        cublas_func_gemm(handle, 'n', 'n', m, k, k, alpha, 
                         G_gpu.gpudata, m, Vr_gpu.gpudata, k, 
                         beta, G_gpu.gpudata, m  )     
        F_gpu_temp = G_gpu
    
    elif modes == 'standard': #Compute (standard) DMD modes: F = U * W
        cublas_func_gemm(handle, 'n', 'n', m, k, k, 
                         alpha, U_gpu.gpudata, m, Vr_gpu.gpudata, k, 
                         beta, U_gpu.gpudata, m  )     
        F_gpu_temp = U_gpu
    else: 
        raise ValueError('Type of modes is not supported, choose "exact" or "standard".')
    
    #Copy is required, because gels destroys input    
    copy_func(handle, F_gpu_temp.size, int(F_gpu_temp.gpudata), 
              1, int(F_gpu.gpudata), 1)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute amplitueds b using least-squares: Fb=x1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if return_amplitudes==True:    
        #x1_gpu = a_gpu[:,0].copy() 
        x1_gpu = gpuarray.empty(m, data_type, order="F", allocator=alloc) 
        copy_func(handle, x1_gpu.size, int(a_gpu[:,0].gpudata), 1, int(x1_gpu.gpudata), 1)
        cula_func_gels( 'N', m, k, int(1) , F_gpu_temp.gpudata, m, x1_gpu.gpudata, m)
        b_gpu = x1_gpu
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute Vandermonde matrix (CPU)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if return_vandermonde==True:      
        V_gpu = linalg.vander(w_gpu, n=nx)
    
    # Free internal CULA memory:
    cula.culaFreeBuffers()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Return 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    if return_amplitudes==True and return_vandermonde==True:
        return F_gpu, b_gpu[:k], V_gpu, omega
    elif return_amplitudes==True and return_vandermonde==False:
        return F_gpu, b_gpu[:k], omega
    elif return_amplitudes==False and return_vandermonde==True:
        return F_gpu, V_gpu, omega
    else:
        return F_gpu, omega   
  


def cdmd(a_gpu, k=None, c=None,  modes='exact', return_amplitudes=False, return_vandermonde=False, handle=None):
    """
    Compressed Dynamic Mode Decomposition.

    Dynamic Mode Decomposition (DMD) is a data processing algorithm which
    allows to decompose a matrix `a` in space and time.
    The matrix `a` is decomposed as `a = FBV`, where the columns of `F`
    contain the dynamic modes. The modes are ordered corresponding
    to the amplitudes stored in the diagonal matrix `B`. `V` is a Vandermonde
    matrix describing the temporal evolution.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Real/complex input matrix  `a` with dimensions `(m, n)`.
    k : int, optional
        If `k < (n-1)` low-rank Dynamic Mode Decomposition is computed.
    c : int
        `p` sets the number of measurements sensors.
    modes : `{'exact'}`
        'exact' : computes the exact dynamic modes, `F = Y * V * (S**-1) * W`.
    return_amplitudes : bool `{True, False}` 
        True: return amplitudes in addition to dynamic modes. 
    return_vandermonde : bool `{True, False}`
        True: return Vandermonde matrix in addition to dynamic modes and amplitudes.          
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.

    Returns
    -------
    f_gpu : pycuda.gpuarray.GPUArray
        Matrix containing the dynamic modes of shape `(m, n-1)`  or `(m, k)`.
    b_gpu : pycuda.gpuarray.GPUArray
        1-D array containing the amplitudes of length `min(n-1, k)`.
    v_gpu : pycuda.gpuarray.GPUArray
        Vandermonde matrix of shape `(n-1, n-1)`  or `(k, n-1)`.

    Notes
    -----
    Double precision is only supported if the standard version of the
    CULA Dense toolkit is installed.

    This function destroys the contents of the input matrix.

    Arrays are assumed to be stored in column-major order, i.e., order='F'.

    References
    ----------
    S. L. Brunton, et al.
    "Compressed sampling and dynamic mode decomposition."  
    arXiv preprint arXiv:1312.5186 (2013).
    
    J. H. Tu, et al.
    "On dynamic mode decomposition: theory and applications."
    arXiv preprint arXiv:1312.0041 (2013).
    
    Examples
    --------
    >>> #Numpy
    >>> import numpy as np
    >>> #Plot libs
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> from matplotlib import cm
    >>> #GPU DMD libs
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> from skcuda import linalg, rlinalg
    >>> linalg.init()
    >>> rlinalg.init()
    
    >>> # Define time and space discretizations
    >>> x=np.linspace( -15, 15, 200)
    >>> t=np.linspace(0, 8*np.pi , 80) 
    >>> dt=t[2]-t[1]
    >>> X, T = np.meshgrid(x,t)
    >>> # Create two patio-temporal patterns
    >>> F1 = 0.5* np.cos(X)*(1.+0.* T)
    >>> F2 = ( (1./np.cosh(X)) * np.tanh(X)) *(2.*np.exp(1j*2.8*T))
    >>> # Add both signals
    >>> F = (F1+F2)
    
    >>> #Plot dataset
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(231, projection='3d')
    >>> ax = fig.gca(projection='3d')
    >>> surf = ax.plot_surface(X, T, F, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    >>> ax.set_zlim(-1, 1)
    >>> plt.title('F')
    >>> ax = fig.add_subplot(232, projection='3d')
    >>> ax = fig.gca(projection='3d')
    >>> surf = ax.plot_surface(X, T, F1, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    >>> ax.set_zlim(-1, 1)
    >>> plt.title('F1')
    >>> ax = fig.add_subplot(233, projection='3d')
    >>> ax = fig.gca(projection='3d')
    >>> surf = ax.plot_surface(X, T, F2, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    >>> ax.set_zlim(-1, 1)
    >>> plt.title('F2')
    
    
    >>> #Dynamic Mode Decomposition
    >>> F_gpu = np.array(F.T, np.complex64, order='F')
    >>> F_gpu = gpuarray.to_gpu(F_gpu) 
    >>> Fmodes_gpu, b_gpu, V_gpu, omega_gpu = rlinalg.cdmd(F_gpu, k=2, c=20, modes='exact', return_amplitudes=True, return_vandermonde=True)
    >>> omega = omega_gpu.get()
    >>> plt.scatter(omega.real, omega.imag, marker='o', c='r')
    
    >>> #Recover original signal
    >>> F1tilde = np.dot(Fmodes_gpu[:,0:1].get() , np.dot(b_gpu[0].get(), V_gpu[0:1,:].get() ) )
    >>> F2tilde = np.dot(Fmodes_gpu[:,1:2].get() , np.dot(b_gpu[1].get(), V_gpu[1:2,:].get() ) )
    
    >>> # Plot DMD modes
    >>> #Mode 0
    >>> ax = fig.add_subplot(235, projection='3d')
    >>> ax = fig.gca(projection='3d')
    >>> surf = ax.plot_surface(X[0:F1tilde.shape[1],:], T[0:F1tilde.shape[1],:], F1tilde.T, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    >>> ax.set_zlim(-1, 1)
    >>> plt.title('F1_tilde')
    >>> #Mode 1
    >>> ax = fig.add_subplot(236, projection='3d')
    >>> ax = fig.gca(projection='3d')
    >>> surf = ax.plot_surface(X[0:F2tilde.shape[1],:], T[0:F2tilde.shape[1],:], F2tilde.T, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    >>> ax.set_zlim(-1, 1)
    >>> plt.title('F2_tilde')
    >>> plt.show()     
    """

    #*************************************************************************
    #***        Author: N. Benjamin Erichson <nbe@st-andrews.ac.uk>        ***
    #***                              <2015>                               ***
    #***                       License: BSD 3 clause                       ***
    #*************************************************************************

    if not _has_cula:
        raise NotImplementedError('CULA not installed')

    if handle is None:
        handle = misc._global_cublas_handle

    alloc = misc._global_cublas_allocator

    # The free version of CULA only supports single precision floating
    data_type = a_gpu.dtype.type
    real_type = np.float32

    if data_type == np.complex64:
        cula_func_gesvd = cula.culaDeviceCgesvd
        cublas_func_gemm = cublas.cublasCgemm
        cublas_func_dgmm = cublas.cublasCdgmm
        cula_func_gels = cula.culaDeviceCgels
        copy_func = cublas.cublasCcopy
        transpose_func = cublas.cublasCgeam
        alpha = np.complex64(1.0)
        beta = np.complex64(0.0)
        TRANS_type = 'C'
        isreal = False
    elif data_type == np.float32:
        cula_func_gesvd = cula.culaDeviceSgesvd
        cublas_func_gemm = cublas.cublasSgemm
        cublas_func_dgmm = cublas.cublasSdgmm
        cula_func_gels = cula.culaDeviceSgels
        copy_func = cublas.cublasScopy
        transpose_func = cublas.cublasSgeam
        alpha = np.float32(1.0)
        beta = np.float32(0.0)
        TRANS_type = 'T'
        isreal = True
    else:
        if cula._libcula_toolkit == 'standard':
            if data_type == np.complex128:
                cula_func_gesvd = cula.culaDeviceZgesvd
                cublas_func_gemm = cublas.cublasZgemm
                cublas_func_dgmm = cublas.cublasZdgmm
                cula_func_gels = cula.culaDeviceZgels
                copy_func = cublas.cublasZcopy
                transpose_func = cublas.cublasZgeam
                alpha = np.complex128(1.0)
                beta = np.complex128(0.0)
                TRANS_type = 'C'
                isreal = False
            elif data_type == np.float64:
                cula_func_gesvd = cula.culaDeviceDgesvd
                cublas_func_gemm = cublas.cublasDgemm
                cublas_func_dgmm = cublas.cublasDdgmm
                cula_func_gels = cula.culaDeviceDgels
                copy_func = cublas.cublasDcopy
                transpose_func = cublas.cublasDgeam
                alpha = np.float64(1.0)
                beta = np.float64(0.0)
                TRANS_type = 'T'
                isreal = True
            else:
                raise ValueError('unsupported type')
            real_type = np.float64
        else:
            raise ValueError('double precision not supported')

    #CUDA assumes that arrays are stored in column-major order
    m, n = np.array(a_gpu.shape, int)
    nx = n-1
    #Set k     
    if k == None : k = nx
    if k > nx or k < 1: raise ValueError('k is not valid')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compress
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if c==None:
        Ac_gpu = A
        c=m
    else:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Generate a random sensing matrix S
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if isreal==False: 
            Simag_gpu = gpuarray.empty((m,c), real_type, order="F", allocator=alloc)
            Sreal_gpu = gpuarray.empty((m,c), real_type, order="F", allocator=alloc) 
            S_gpu = gpuarray.empty((c,m), data_type, order="F", allocator=alloc) 
            rand.fill_uniform(Simag_gpu)
            rand.fill_uniform(Sreal_gpu)
            S_gpu = Sreal_gpu + 1j * Simag_gpu
            S_gpu = S_gpu.T * 2 -1 #Scale to [-1,1] 
        else:
            S_gpu = gpuarray.empty((c,m), real_type, order="F", allocator=alloc) 
            rand.fill_uniform(S_gpu) #Draw random samples from a ~ Uniform(-1,1) distribution
            S_gpu = S_gpu * 2 - 1 #Scale to [-1,1]  
        
            
        #Allocate Ac 
        Ac_gpu = gpuarray.empty((c,n), data_type, order="F", allocator=alloc)   

        #Compress input matrix
        cublas_func_gemm(handle, 'n', 'n', c, n, m, alpha, 
                     int(S_gpu.gpudata), c, int(a_gpu.gpudata), m, 
                        beta, int(Ac_gpu.gpudata), c )         
                        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Split data into lef and right snapshot sequence
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Note: we need a copy of X_gpu, because SVD destroys X_gpu
    #While Y_gpu is just a pointer
    X_gpu = gpuarray.empty((c, n), data_type, order="F", allocator=alloc) 
    copy_func(handle, X_gpu.size, int(Ac_gpu.gpudata), 1, int(X_gpu.gpudata), 1)
    X_gpu = X_gpu[:, :nx]    
    Y_gpu = Ac_gpu[:, 1:] 
    Yorig_gpu = a_gpu[:, 1:]     
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Singular Value Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
    #Allocate s, U, Vt for economic SVD
    #Note: singular values are always real
    min_s = min(nx,c)
    s_gpu = gpuarray.zeros(min_s, real_type, order="F", allocator=alloc)
    U_gpu = gpuarray.zeros((c,min_s), data_type, order="F", allocator=alloc)
    Vh_gpu = gpuarray.zeros((min_s,nx), data_type, order="F", allocator=alloc)
    
   #Economic SVD
    cula_func_gesvd('S', 'S', c, nx, int(X_gpu.gpudata), c, int(s_gpu.gpudata), 
                    int(U_gpu.gpudata), c, int(Vh_gpu.gpudata), min_s)
     
    #Low-rank DMD: trancate SVD if k < nx
    if k != nx:
        s_gpu = s_gpu[:k]
        U_gpu = U_gpu[: , :k]
        Vh_gpu = Vh_gpu[:k , : ]
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Solve the LS problem to find estimate for M using the pseudo-inverse    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    #real: M = U.T * Y * Vt.T * S**-1
    #complex: M = U.H * Y * Vt.H * S**-1
    #Let G = Y * Vt.H * S**-1, hence M = M * G
    
    #Allocate G and M  
    G_gpu = gpuarray.zeros((c,k), data_type, order="F", allocator=alloc)    
    M_gpu = gpuarray.zeros((k,k), data_type, order="F", allocator=alloc)    
    
    #i) s = s **-1 (inverse)
    if data_type == np.complex64 or data_type == np.complex128:
        s_gpu = 1/s_gpu
        s_gpu = s_gpu + 1j * gpuarray.zeros_like(s_gpu)
    else:
        s_gpu = 1/s_gpu
    
    
    #ii) real/complex: scale Vs =  Vt* x diag(s**-1) 
    Vs_gpu = gpuarray.zeros((nx,k), data_type, order="F", allocator=alloc)      
    lda = max(1, Vh_gpu.strides[1] // Vh_gpu.dtype.itemsize)
    ldb = max(1, Vs_gpu.strides[1] // Vs_gpu.dtype.itemsize)
    transpose_func(handle, TRANS_type, TRANS_type, nx, k,
                   1.0, int(Vh_gpu.gpudata), lda, 0.0, int(Vh_gpu.gpudata), lda,
                   int(Vs_gpu.gpudata), ldb)       
    #End Transpose
    
    cublas_func_dgmm(handle, 'r', nx, k, int(Vs_gpu.gpudata), nx, 
                     int(s_gpu.gpudata), 1 , int(Vs_gpu.gpudata), nx)
   

    #iii) real: G = Y * Vs , complex: G = Y x Vs
    cublas_func_gemm(handle, 'n', 'n', c, k, nx, alpha, 
                     int(Y_gpu.gpudata), c, int(Vs_gpu.gpudata), nx, 
                        beta, int(G_gpu.gpudata), c )      
   
    
    #iv) real/complex: M = U* x G 
    cublas_func_gemm(handle, TRANS_type, 'n', k, k, c, alpha, 
                     int(U_gpu.gpudata), c, int(G_gpu.gpudata), c, 
                    beta, int(M_gpu.gpudata), k )   

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Eigen Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Note: If a_gpu is real the imag part is omitted
    Vr_gpu, w_gpu = linalg.eig(M_gpu, 'N', 'V', 'F')

    omega = cumath.log(w_gpu)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    #Compute DMD Modes 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    F_gpu = gpuarray.empty((m,k), data_type, order="F", allocator=alloc)    
    modes = modes.lower()
    if modes == 'exact': #Compute (exact) DMD modes: F = Y * V * S**-1 * W = G * W
        
        cublas_func_gemm(handle, 'n' , 'n', nx, k, k, alpha, 
                     int(Vs_gpu.gpudata), nx, int(Vr_gpu.gpudata), k, 
                        beta, int(Vs_gpu.gpudata), nx ) 
        
        cublas_func_gemm(handle, 'n', 'n', m, k, nx, alpha, 
                         Yorig_gpu.gpudata, m, Vs_gpu.gpudata, nx, 
                         beta, F_gpu.gpudata, m  )     
    
    else: 
        raise ValueError('Type of modes is not supported, choose "exact" or "standard".')
    


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute amplitueds b using least-squares: Fb=x1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if return_amplitudes==True:  
        F_gpu_temp = gpuarray.empty((m,k), data_type, order="F", allocator=alloc)        
        
        #Copy is required, because gels destroys input    
        copy_func(handle, F_gpu.size, int(F_gpu.gpudata), 
              1, int(F_gpu_temp.gpudata), 1)
        
        #x1_gpu = a_gpu[:,0].copy() 
        x1_gpu = gpuarray.empty(m, data_type, order="F", allocator=alloc) 
        copy_func(handle, x1_gpu.size, int(a_gpu[:,0].gpudata), 1, int(x1_gpu.gpudata), 1)
        cula_func_gels( 'N', m, k, int(1) , F_gpu_temp.gpudata, m, x1_gpu.gpudata, m)
        b_gpu = x1_gpu
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute Vandermonde matrix (CPU)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if return_vandermonde==True:      
        V_gpu = linalg.vander(w_gpu, n=nx)
    
    # Free internal CULA memory:
    cula.culaFreeBuffers()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Return 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    if return_amplitudes==True and return_vandermonde==True:
        return F_gpu, b_gpu[:k], V_gpu, omega
    elif return_amplitudes==True and return_vandermonde==False:
        return F_gpu, b_gpu[:k], omega
    elif return_amplitudes==False and return_vandermonde==True:
        return F_gpu, V_gpu, omega
    else:
        return F_gpu, omega   

        
if __name__ == "__main__":
    import doctest
    doctest.testmod()
