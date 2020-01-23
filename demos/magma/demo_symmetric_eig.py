"""
Author: Kit Lee (wklee4993@gmail.com)
Demo of using symmetric eigenvalue solver in Magma

The following functions can be tested:
    ssyevd, ssyevd_m, ssyevd_gpu,
    ssyevdx, ssyevdx_m, ssyevdx_gpu
and their 'double' variants.
"""
import numpy as np
import time
from skcuda.magma import (
    magma_init, magma_finalize,
    magma_get_ssytrd_nb, magma_get_dsytrd_nb,
    magma_ssyevdx, magma_ssyevdx_gpu, magma_ssyevdx_m,
    magma_dsyevdx, magma_dsyevdx_gpu, magma_dsyevdx_m,
    magma_ssyevd, magma_ssyevd_gpu, magma_ssyevd_m,
    magma_dsyevd, magma_dsyevd_gpu, magma_dsyevd_m
)
import time

typedict = {'s': np.float32, 'd': np.float64, 'c': np.complex64, 'z': np.complex128}
typedict_= {v: k for k, v in typedict.items()}

def _test_syev(N, type_key, data_gpu=False, ngpu=1, expert=False, keigs=None):
    """A simple test of the eigenvalue solver for symmetric matrix. 
    Random matrix is used.

    :param N: size of the testing matrix
    :param type_key: data type: one of 's', 'd'
    :type type_key: string
    :param data_gpu: whether data is stored on the gpu initially, i.e., `_gpu`
    :type data_gpu: bool, optional
    :param ngpu: number of gpu, use -1 for running single gpu with `_m` functions
    :type ngpu: int, optional
    :param expert: use of expert variant (```x```)
    :type expert: bool, optional
    :param keigs: number of eigenvalue to find; find all when ```keigs=-1```
    :type keigs: int, optional
    """
    if data_gpu:
        try:
            import pycuda.autoinit
            import pycuda.driver as drv
            import pycuda.gpuarray as gpuarray
            print("Imported pycuda.gpuarray")
        except:
            raise ImportError("Cannot import pycuda.gpuarray!")
    
    dtype = typedict[type_key]
    if dtype in [np.complex64, np.complex128]:
        raise ValueError("Complex types are not supported yet.")

    # not symmetric, but only side of the diagonal is used
    np.random.seed(1234)
    M = np.random.random((N, N)).astype(dtype)
    M_cpu = M.copy() # for comparison
    
    # GPU 
    jobz = 'N' # do not compute eigenvectors
    uplo = 'U' # use upper diagonal
    if expert:
        rnge = 'I'
        vl = 0; vu = 1e10
        if keigs is None:
            keigs = int(np.ceil(N/2))
        il = 1; iu = keigs; 
        m = np.zeros((1,), dtype=int) # no. of eigenvalues found

    # allocate memory for eigenvalues
    w_gpu = np.empty((N,), dtype, order='F')

    if type_key == 's':
        nb = magma_get_ssytrd_nb(N)
    elif type_key == 'd':
        nb = magma_get_dsytrd_nb(N)
    else:
        raise ValueError('unsupported type')

    lwork = N*(1 + 2*nb)
    if jobz:
        lwork = max(lwork, 1+6*N+2*N**2)
    work = np.empty(lwork, dtype)
    liwork = 3+5*N if jobz else 1
    iwork = np.empty(liwork, dtype)
    if data_gpu:
        ldwa = 2*max(1, N)
        worka = np.empty((ldwa, N), order='F')
        #worka = np.empty((2*N), dtype)


    if data_gpu:
        ngpu = 1
        if type_key == 's':
            magma_function = magma_ssyevdx_gpu if expert else magma_ssyevd_gpu
        elif type_key == 'd':
            magma_function = magma_dsyevdx_gpu if expert else magma_dsyevd_gpu
    else:
        if ngpu > 1 or ngpu == -1:
            ngpu = abs(ngpu)
            if type_key == 's':
                magma_function = magma_ssyevdx_m if expert else magma_ssyevd_m
            elif type_key == 'd':
                magma_function = magma_dsyevdx_m if expert else magma_dsyevd_m
        elif ngpu == 1:
            if type_key == 's':
                magma_function = magma_ssyevdx if expert else magma_ssyevd
            elif type_key == 'd':
                magma_function = magma_dsyevdx if expert else magma_dsyevd
        else:
            raise ValueError(f"ngpu={ngpu} is not supported, it must be -1 or >= 1")


    print(f"calling {magma_function.__name__}")

    t1 = time.time()
    if '_m' in magma_function.__name__:
        # multi-gpu
        if 'x' in magma_function.__name__:
            # expert
            status = magma_function(ngpu, jobz, rnge, uplo, N, M.ctypes.data, N,
                                    vl, vu, il, iu, m.ctypes.data, 
                                    w_gpu.ctypes.data, work.ctypes.data, lwork, iwork.ctypes.data, liwork)
        else:
            # non-expert
            status = magma_function(ngpu, jobz, uplo, N, M.ctypes.data, N,
                                w_gpu.ctypes.data, work.ctypes.data, lwork, iwork.ctypes.data, liwork)
    elif '_gpu' in magma_function.__name__:
        # data-on-gpu
        print("_gpu cases are not ready yet!")
        M_gpu = gpuarray.to_gpu(M)
        if 'x' in magma_function.__name__:
            status = magma_function(jobz, rnge, uplo, N, M_gpu.gpudata, N,
                                    vl, vu, il, iu, m.ctypes.data, 
                                    w_gpu.ctypes.data, worka.ctypes.data, ldwa,
                                    work.ctypes.data, lwork, iwork.ctypes.data, liwork)
        else:
            status = magma_function(jobz, uplo, N, M_gpu.gpudata, N,
                                    w_gpu.ctypes.data, worka.ctypes.data, ldwa,
                                    work.ctypes.data, lwork, iwork.ctypes.data, liwork)
    else:
        # single-gpu
        if 'x' in magma_function.__name__:
            # expert
            status = magma_function(jobz, rnge, uplo, N, M.ctypes.data, N,
                                    vl, vu, il, iu, m.ctypes.data, 
                                    w_gpu.ctypes.data, work.ctypes.data, lwork, iwork.ctypes.data, liwork)
        else:
            # non-expert
            status = magma_function(jobz, uplo, N, M.ctypes.data, N,
                                    w_gpu.ctypes.data, work.ctypes.data, lwork, iwork.ctypes.data, liwork)
    W_gpu = np.sort(w_gpu)
    t2 = time.time()

    if expert:
        print(f"Number of eigenvalues found: {m}")
        print(f"{work[:5]}")

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

def main():
    import argparse
    parser = argparse.ArgumentParser("Demo for Magma symmetric eigen-solver")
    parser.add_argument('-N', default=256, type=int, help='size of matrix')
    parser.add_argument('-t', default='s', type=str, help='data type: s or d, default s')
    parser.add_argument('-ongpu', action='store_true', help='whether initial data on gpu (using _gpu functions), default False')
    parser.add_argument('-ngpu', default=1, type=int, help='number of GPU (default 1), use -1 for running multi-gpu functions with single gpu')
    parser.add_argument('-expert', action='store_true', help='use the expert function (ssyevdx, dsyevdx)')
    parser.add_argument('-fulltest', action='store_true', help='full test except data-on-gpu functions')
    args = parser.parse_args()

    magma_init()
    if not args.fulltest:
        _test_syev(N=args.N, type_key=args.t,
                    data_gpu=args.ongpu,
                    ngpu=args.ngpu,
                    expert=args.expert,
                    keigs=None)
    else:
        for N in [200, 2000]:
            for t in ['s', 'd']:
                print("Data_on_host")
                for expert in [False, True]:
                    for ngpu in [-1, 1]:
                        print(f"N={N}; t={t}; ngpu={ngpu}; expert={expert}")
                        _test_syev(N=N, type_key=t, data_gpu=False,
                                    ngpu=ngpu, expert=expert)
                print("Data_on_device")
                print("\n\n")
                for expert in [False, True]:
                    print(f"N={N}; t={t}; expert={expert}")
                    _test_syev(N=N, type_key=t, data_gpu=True,
                                ngpu=1, expert=expert)
    magma_finalize()
   
if __name__=='__main__':
    main()
