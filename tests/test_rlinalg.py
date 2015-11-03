#!/usr/bin/env python

"""
Unit tests for scikits.cuda.linalg
"""

from unittest import main, makeSuite, TestCase, TestSuite

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

from numpy.testing import assert_raises

import skcuda.linalg as linalg
import skcuda.rlinalg as rlinalg
import skcuda.misc as misc

atol_float32 = 1e-4
atol_float64 = 1e-8

class test_rlinalg(TestCase):
    def setUp(self):
        np.random.seed(0)
        linalg.init()
        rlinalg.init()

    def test_rsvd_float32(self):
        m, n = 5, 4
        a = np.array(np.random.randn(m, n), np.float32, order='F')
        a_gpu = gpuarray.to_gpu(a)
        U, s, Vt = rlinalg.rsvd(a_gpu, k=n, p=0, q=2, method='standard')
        assert np.allclose(a, np.dot(U.get(), np.dot(np.diag(s.get()), Vt.get())), atol_float32)
       
    def test_rsvd_float64(self):
        m, n = 5, 4
        a = np.array(np.random.randn(m, n), np.float64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        U, s, Vt = rlinalg.rsvd(a_gpu, k=n, p=0, q=2, method='standard')
        assert np.allclose(a, np.dot(U.get(), np.dot(np.diag(s.get()), Vt.get())), atol_float64)
    
    def test_rsvd_complex64(self):
        m, n = 5, 4
        a = np.array(np.random.randn(m, n) + 1j*np.random.randn(m, n), np.complex64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        U, s, Vt = rlinalg.rsvd(a_gpu, k=n, p=0, q=2, method='standard')
        assert np.allclose(a, np.dot(U.get(), np.dot(np.diag(s.get()), Vt.get())), atol_float32)
        
    def test_rsvd_complex128(self):
        m, n = 5, 4
        a = np.array(np.random.randn(m, n) + 1j*np.random.randn(m, n), np.complex128, order='F')
        a_gpu = gpuarray.to_gpu(a)
        U, s, Vt = rlinalg.rsvd(a_gpu, k=n, p=0, q=2, method='standard')
        assert np.allclose(a, np.dot(U.get(), np.dot(np.diag(s.get()), Vt.get())), atol_float64)
     
    def test_rsvdf_float32(self):
        m, n = 5, 4
        a = np.array(np.random.randn(m, n), np.float32, order='F')
        a_gpu = gpuarray.to_gpu(a)
        U, s, Vt = rlinalg.rsvd(a_gpu, k=n, p=0, q=2, method='fast')
        assert np.allclose(a, np.dot(U.get(), np.dot(np.diag(s.get()), Vt.get())), atol_float32)

    def test_rsvdf_float64(self):
        m, n = 5, 4
        a = np.array(np.random.randn(m, n), np.float64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        U, s, Vt = rlinalg.rsvd(a_gpu, k=n, p=0, q=2, method='fast')
        assert np.allclose(a, np.dot(U.get(), np.dot(np.diag(s.get()), Vt.get())), atol_float64)
    
    def test_rsvdf_complex64(self):
        m, n = 5, 4
        a = np.array(np.random.randn(m, n) + 1j*np.random.randn(m, n), np.complex64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        U, s, Vt = rlinalg.rsvd(a_gpu, k=n, p=0, q=2, method='fast')
        assert np.allclose(a, np.dot(U.get(), np.dot(np.diag(s.get()), Vt.get())), atol_float32)
        
    def test_rsvdf_complex128(self):
        m, n = 5, 4
        a = np.array(np.random.randn(m, n) + 1j*np.random.randn(m, n), np.complex128, order='F')
        a_gpu = gpuarray.to_gpu(a)
        U, s, Vt = rlinalg.rsvd(a_gpu, k=n, p=0, q=2, method='fast')
        assert np.allclose(a, np.dot(U.get(), np.dot(np.diag(s.get()), Vt.get())), atol_float64) 
        
    def test_rdmd_float32(self):
        m, n = 6, 4
        a = np.array(np.fliplr(np.vander(np.random.rand(m)+1, n)), np.float32, order='F')
        a_gpu = gpuarray.to_gpu(a)
        f_gpu, b_gpu, v_gpu, omega = rlinalg.rdmd(a_gpu, k=(n-1), p=1, q=2, modes='standard', return_amplitudes=True, return_vandermonde=True)
        assert np.allclose(a[:,:(n-1)], np.dot(f_gpu.get(), np.dot(np.diag(b_gpu.get()), v_gpu.get()) ), atol_float32)   

    def test_rdmd_float64(self):
        m, n = 9, 7
        a = np.array(np.fliplr(np.vander(np.random.rand(m)+1, n)), np.float64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        f_gpu, b_gpu, v_gpu, omega = rlinalg.rdmd(a_gpu, k=(n-1), p=1, q=2, modes='standard', return_amplitudes=True, return_vandermonde=True)
        assert np.allclose(a[:,:(n-1)], np.dot(f_gpu.get(), np.dot(np.diag(b_gpu.get()), v_gpu.get()) ), atol_float64)  

    def test_rdmd_complex64(self):
        m, n = 9, 7
        a = np.array(np.fliplr(np.vander(np.random.rand(m)+1, n)) + 1j*np.fliplr(np.vander(np.random.rand(m)+1, n)), 
                     np.complex64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        f_gpu, b_gpu, v_gpu, omega = rlinalg.rdmd(a_gpu, k=(n-1), p=1, q=1, modes='standard', return_amplitudes=True, return_vandermonde=True)
        assert np.allclose(a[:,:(n-1)], np.dot(f_gpu.get(), np.dot(np.diag(b_gpu.get()), v_gpu.get()) ), atol_float32)
        
    def test_rdmd_complex128(self):
        m, n = 9, 7
        a = np.array(np.fliplr(np.vander(np.random.rand(m)+1, n)) + 1j*np.fliplr(np.vander(np.random.rand(m)+1, n)), 
                     np.complex128, order='F')
        a_gpu = gpuarray.to_gpu(a)
        f_gpu, b_gpu, v_gpu, omega = rlinalg.rdmd(a_gpu, k=(n-1), p=1, q=1, modes='standard', return_amplitudes=True, return_vandermonde=True)
        assert np.allclose(a[:,:(n-1)], np.dot(f_gpu.get(), np.dot(np.diag(b_gpu.get()), v_gpu.get()) ), atol_float64)

    def test_cdmd_float32(self):
        # Define time and space discretizations
        x=np.linspace( -5, 5, 50)
        t=np.linspace(0, 8*np.pi , 20) 
        dt=t[2]-t[1]
        X, T = np.meshgrid(x,t)
        # Create two patio-temporal patterns
        F1 = 0.5* np.cos(X)*(1.+0.* T)
        F2 = ( (1./np.cosh(X)) * np.tanh(X)) *(2.*np.exp(1j*2.8*T))
        D = F1+F2
        a = np.array(D.real, np.float32, order='F')
        a_gpu = gpuarray.to_gpu(a)
        dmdf_gpu, dmdb_gpu, dmdv_gpu, dmdomega = linalg.dmd(a_gpu, k=2, modes='exact', return_amplitudes=True, return_vandermonde=True)
        f_gpu, b_gpu, v_gpu, omega = rlinalg.cdmd(a_gpu, k=2, c=10, modes='exact', return_amplitudes=True, return_vandermonde=True)
        assert np.allclose(dmdomega.get(), omega.get(), atol_float32)   

    def test_cdmd_float64(self):
        # Define time and space discretizations
        x=np.linspace( -5, 5, 50)
        t=np.linspace(0, 8*np.pi , 20) 
        dt=t[2]-t[1]
        X, T = np.meshgrid(x,t)
        # Create two patio-temporal patterns
        F1 = 0.5* np.cos(X)*(1.+0.* T)
        F2 = ( (1./np.cosh(X)) * np.tanh(X)) *(2.*np.exp(1j*2.8*T))
        D = F1+F2
        a = np.array(D.real, np.float64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        dmdf_gpu, dmdb_gpu, dmdv_gpu, dmdomega = linalg.dmd(a_gpu, k=2, modes='exact', return_amplitudes=True, return_vandermonde=True)
        f_gpu, b_gpu, v_gpu, omega = rlinalg.cdmd(a_gpu, k=2, c=10, modes='exact', return_amplitudes=True, return_vandermonde=True)
        assert np.allclose(dmdomega.get(), omega.get(), atol_float64)    

    def test_cdmd_complex64(self):
        # Define time and space discretizations
        x=np.linspace( -5, 5, 50)
        t=np.linspace(0, 8*np.pi , 20) 
        dt=t[2]-t[1]
        X, T = np.meshgrid(x,t)
        # Create two patio-temporal patterns
        F1 = 0.5* np.cos(X)*(1.+0.* T)
        F2 = ( (1./np.cosh(X)) * np.tanh(X)) *(2.*np.exp(1j*2.8*T))
        D = F1+F2
        a = np.array(D.real, np.complex64, order='F')
        a_gpu = gpuarray.to_gpu(a)
        dmdf_gpu, dmdb_gpu, dmdv_gpu, dmdomega = linalg.dmd(a_gpu, k=2, modes='exact', return_amplitudes=True, return_vandermonde=True)
        f_gpu, b_gpu, v_gpu, omega = rlinalg.cdmd(a_gpu, k=2, c=10, modes='exact', return_amplitudes=True, return_vandermonde=True)
        assert np.allclose(dmdomega.get().real, omega.get().real, atol_float32)   
        
        
    def test_cdmd_complex128(self):
        # Define time and space discretizations
        x=np.linspace( -5, 5, 50)
        t=np.linspace(0, 8*np.pi , 20) 
        dt=t[2]-t[1]
        X, T = np.meshgrid(x,t)
        # Create two patio-temporal patterns
        F1 = 0.5* np.cos(X)*(1.+0.* T)
        F2 = ( (1./np.cosh(X)) * np.tanh(X)) *(2.*np.exp(1j*2.8*T))
        D = F1+F2
        a = np.array(D.real, np.complex128, order='F')
        a_gpu = gpuarray.to_gpu(a)
        dmdf_gpu, dmdb_gpu, dmdv_gpu, dmdomega = linalg.dmd(a_gpu, k=2, modes='exact', return_amplitudes=True, return_vandermonde=True)
        f_gpu, b_gpu, v_gpu, omega = rlinalg.cdmd(a_gpu, k=2, c=10, modes='exact', return_amplitudes=True, return_vandermonde=True)
        assert np.allclose(dmdomega.get().real, omega.get().real, atol_float64)   
        
        
def suite():
    s = TestSuite()
    s.addTest(test_rlinalg('test_rsvd_float32'))
    s.addTest(test_rlinalg('test_rsvd_float64'))
    s.addTest(test_rlinalg('test_rsvd_complex64'))
    s.addTest(test_rlinalg('test_rsvd_complex128'))
    s.addTest(test_rlinalg('test_rsvdf_float32'))
    s.addTest(test_rlinalg('test_rsvdf_float64'))
    s.addTest(test_rlinalg('test_rsvdf_complex64'))
    s.addTest(test_rlinalg('test_rsvdf_complex128'))
    s.addTest(test_rlinalg('test_rdmd_float32'))
    s.addTest(test_rlinalg('test_rdmd_float64'))
    s.addTest(test_rlinalg('test_rdmd_complex64'))
    s.addTest(test_rlinalg('test_rdmd_complex128'))
    s.addTest(test_rlinalg('test_cdmd_float32'))
    s.addTest(test_rlinalg('test_cdmd_float64'))
    s.addTest(test_rlinalg('test_cdmd_complex64'))
    s.addTest(test_rlinalg('test_cdmd_complex128'))   
    
    if misc.get_compute_capability(pycuda.autoinit.device) >= 1.3:
        s.addTest(test_rlinalg('test_rsvd_float32'))
        s.addTest(test_rlinalg('test_rsvd_float64'))
        s.addTest(test_rlinalg('test_rsvd_complex64'))
        s.addTest(test_rlinalg('test_rsvd_complex128'))  
        s.addTest(test_rlinalg('test_rsvdf_float32'))
        s.addTest(test_rlinalg('test_rsvdf_float64'))
        s.addTest(test_rlinalg('test_rsvdf_complex64'))
        s.addTest(test_rlinalg('test_rsvdf_complex128'))
        s.addTest(test_rlinalg('test_rdmd_float32'))
        s.addTest(test_rlinalg('test_rdmd_float64'))
        s.addTest(test_rlinalg('test_rdmd_complex64'))
        s.addTest(test_rlinalg('test_rdmd_complex128'))
        s.addTest(test_rlinalg('test_cdmd_float32'))
        s.addTest(test_rlinalg('test_cdmd_float64'))
        s.addTest(test_rlinalg('test_cdmd_complex64'))
        s.addTest(test_rlinalg('test_cdmd_complex128')) 
        
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
