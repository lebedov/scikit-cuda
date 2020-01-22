"""
Demo of using ssyedx_m 
"""
import numpy as np
from skcuda import magma
import time

typedict = {'s': np.float32, 'd': np.float64, 'c': np.complex64, 'z': np.complex128}
typedict_= {v: k for k, v in typedict.items()}

def eigs(a_gpu, k=None, which='LM', imag=False, return_eigenvectors=True):
    """
    Driver for eigenvalue solver for symmetric matrix
    """

    if len(a_gpu.shape) != 2:
        raise ValueError("M needs to be a rank 2 square array for eig.")

    magma.magma_init()
    ngpu=1

    dtype = a_gpu.dtype.type
    t = typedict_[dtype]
    N = a_gpu.shape[1]

    if k is None: k=int(np.ceil(N/2))

    if 'L' in which:
        a_gpu = -a_gpu

    if return_eigenvectors:
        jobz = 'V'
    else:
        jobz = 'N'

    rnge = 'I'; uplo = 'U'
    vl = 0; vu = 1e10
    il = 1; iu = k; m = np.zeros((1,), dtype=int)
    print(f"k = {k}, iu = {iu}")

    w_gpu = np.empty((N,), dtype, order='F') # eigenvalues

    if t == 's':
        nb = magma.magma_get_ssytrd_nb(N)
    elif t == 'd':
        nb = magma.magma_get_dsytrd_nb(N)
    else:
        raise ValueError('unsupported type')

    lwork = N*(1 + 2*nb)
    if jobz:
        lwork = max(lwork, 1+6*N+2*N**2)
    work = np.empty(lwork, dtype)
    liwork = 3+5*N if jobz else 1
    iwork = np.empty(liwork, dtype)

    if t == 's':
        status = magma.magma_ssyevdx_m(ngpu, jobz, rnge, uplo, N, a_gpu.ctypes.data, N,
                                    vl, vu, il, iu,
                                    m.ctypes.data, 
                                    w_gpu.ctypes.data, work.ctypes.data, lwork, iwork.ctypes.data, liwork)
    elif t == 'd':
        status = magma.magma_dsyevdx_m(ngpu, jobz, rnge, uplo, N, a_gpu.ctypes.data, N,
                                    vl, vu, il, iu,
                                    m.ctypes.data,  
                                    w_gpu.ctypes.data, work.ctypes.data, lwork, iwork.ctypes.data, liwork)
    else:
        raise ValueError('unsupported type')

    print(f"Number of eigenvalues found: {m}")
    print(f"{work[:5]}")

    if 'L' in which:
        w_gpu = -w_gpu

    magma.magma_finalize()

    if jobz:
        return w_gpu, a_gpu
    else:
        return w_gpu


if __name__=='__main__':
    import sys
    N = int(sys.argv[1])

    # not symmetric, but only side of the diagonal is used
    M_gpu = np.random.random((N, N)) 
    M_gpu = M_gpu.astype(np.float32)
    M_cpu = M_gpu.copy()
    
    # GPU
    t1 = time.time()
    W_gpu, V_gpu = eigs(M_gpu, k=10)
    W_gpu = np.sort(W_gpu)
    t2 = time.time()

    # CPU
    t3 = time.time()
    W_cpu, V_cpu = np.linalg.eigh(M_cpu)
    W_cpu = np.sort(W_cpu)
    t4 = time.time()

    print("First 10 eigenvalues")
    print(f"GPU: {W_gpu[:10]}")
    print(f"CPU: {W_cpu[:10]}")
    print("Time")
    print(f"GPU: {t2-t1}")
    print(f"CPU: {t4-t3}")
