"""
Demo of how to call low-level MAGMA wrappers to solve non-symmetric eigenvalue problem.

Note MAGMA's GEEV implementation is a hybrid of CPU/GPU code; the inputs
therefore must be in host memory.
"""

import numpy as np
import scipy.linalg
import skcuda.magma as magma
import time
import importlib
importlib.reload(magma)
typedict = {'s': np.float32, 'd': np.float64, 'c': np.complex64, 'z': np.complex128}

def test_cpu_gpu(N, t='z'):
    """
        N     : dimension
        dtype : type (default complex)
    """
    assert t in typedict.keys()
    
    dtype = typedict[t]
    
    
    if t in ['s', 'd']:
        M_gpu = np.random.random((N,N))
    elif t in ['c', 'z']:
        M_gpu = np.random.random((N,N))+1j*np.random.random((N,N))

    M_gpu = M_gpu.astype(dtype)
    M_cpu = M_gpu.copy()
    
    # GPU (skcuda + Magma)
    # Set up output buffers:
    if t in ['s', 'd']:
        wr = np.zeros((N,), dtype) # eigenvalues
        wi = np.zeros((N,), dtype) # eigenvalues
    elif t in ['c', 'z']:
        w = np.zeros((N,), dtype) # eigenvalues
        
    vl = np.zeros((N, N), dtype)
    vr = np.zeros((N, N), dtype)

    # Set up workspace:
    if t == 's':
        nb = magma.magma_get_sgeqrf_nb(N,N)
    if t == 'd':
        nb = magma.magma_get_dgeqrf_nb(N,N)
    if t == 'c':
        nb = magma.magma_get_cgeqrf_nb(N,N)
    if t == 'z':
        nb = magma.magma_get_zgeqrf_nb(N,N)
    
    lwork = N*(1 + 2*nb)
    work = np.zeros((lwork,), dtype)
    if t in ['c', 'z']:
        rwork= np.zeros((2*N,), dtype)

    # Compute:
    gpu_time = time.time();
    if t == 's':
        status = magma.magma_sgeev('N', 'V', N, M_gpu.ctypes.data, N, 
                                   wr.ctypes.data, wi.ctypes.data, 
                                   vl.ctypes.data, N, vr.ctypes.data, N, 
                                   work.ctypes.data, lwork)
    if t == 'd':
        status = magma.magma_dgeev('N', 'V', N, M_gpu.ctypes.data, N, 
                                   wr.ctypes.data, wi.ctypes.data, 
                                   vl.ctypes.data, N, vr.ctypes.data, N, 
                                   work.ctypes.data, lwork)
    if t == 'c':
        status = magma.magma_cgeev('N', 'V', N, M_gpu.ctypes.data, N, 
                                   w.ctypes.data, vl.ctypes.data, N, vr.ctypes.data, N, 
                                   work.ctypes.data, lwork, rwork.ctypes.data)
    if t == 'z':
        status = magma.magma_zgeev('N', 'V', N, M_gpu.ctypes.data, N, 
                                   w.ctypes.data, vl.ctypes.data, N, vr.ctypes.data, N, 
                                   work.ctypes.data, lwork, rwork.ctypes.data)
    gpu_time = time.time() - gpu_time;
    
    # CPU
    cpu_time = time.time()
    W, V = scipy.linalg.eig(M_cpu)
    cpu_time = time.time() - cpu_time
    
    
    # Compare
    if t in ['s', 'd']:
        W_gpu = wr + 1j*wi
    elif t in ['c', 'z']:
        W_gpu = w
        
    W_gpu.sort()
    W.sort()
    status = np.allclose(W[:int(N/4)], W_gpu[:int(N/4)], 1e-3)
    
    return gpu_time, cpu_time, status
    
    
    
if __name__=='__main__':

    magma.magma_init()

    N=1000
    
    print("%10a %10a %10a %10a" % ('type', "GPU", "CPU", "Equal?"))
    for t in ['z', 'c', 's', 'd']:
        gpu_time, cpu_time, status = test_cpu_gpu(N, t=t)
        print("%10a %10.3g, %10.3g, %10s" % (t, gpu_time, cpu_time, status))
        
    magma.magma_finalize()
