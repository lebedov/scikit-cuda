import numpy as np
from pycuda import gpuarray, cumath, autoinit
import cublas, misc, linalg


class PCA():

	"""
	Principal Component Analysis with similar API to sklearn.decomposition.PCA 

	The algorithm implemented here was first implemented with cuda in [Andrecut, 2008]. 
	It performs nonlinear dimensionality reduction for a data matrix, mapping the data
	to a lower dimensional space of orthogonal vectors, while keeping most of the variance. 
	Read more in the reference. 

	Parameters
	----------
	n_components: int, default=None
		The number of principal component column vectors to compute in the output 
		matrix.

	epsilon: float, default=1e-7	
		The maximum error tolerance for eigen value approximation.

	max_iter: int, default=10000
		The maximum number of iterations in approximating each eigenvalue  



	References
	----------
	`[Andrecut, 2008] <https://arxiv.org/pdf/0811.1081.pdf>`_
	

	Examples
	--------
	
	>>> import pycuda.autoinit
	>>> import pycuda.gpuarray as gpuarray
	>>> import numpy as np
	>>> import skcuda.linalg as linalg
	>>> import skcuda.pca import PCA as cuPCA 
	>>> pca = cuPCA(n_components=4) # map the data to 4 dimensions
	>>> X = np.random.rand(1000,100) # 1000 samples of 100-dimensional data vectors
	>>> X_gpu = gpuarray.GPUArray((1000,100), np.float64, order="F") # note that order="F" or a transpose is necessary. fit_transform requires row-major matrices, and column-major is the default
	>>> X_gpu.set(X) # copy data to gpu
	>>> T_gpu = pca.fit_transform(X_gpu) # calculate the principal components
	>>> linalg.dot(T_gpu[:,0], T_gpu[:,1]) # show that the resulting eigenvectors are orthogonal
	3.637978807091713e-12



	"""
		
	def __init__(self, n_components=None, handle=None, epsilon=1e-7, max_iter=10000):
		
		self.n_components = n_components
		self.epsilon = epsilon
		self.max_iter = max_iter	
		misc.init()
		
		if handle is None:
			self.h = misc._global_cublas_handle # create a handle to initialize cublas
		else:	
			self.h = handle
			

	def fit_transform(self, X_gpu):

		"""
		Fit the Principal Component Analysis model, and return the dimension-reduced matrix.

		Compute the first K principal components of R_gpu using the
		Gram-Schmidt orthogonalization algorithm provided by [Andrecut, 2008].

		Parameters
		----------
		R_gpu: pycuda.gpuarray.GPUArray
			NxP (N = number of samples, P = number of variables) data matrix that needs 
			to be reduced. R_gpu can be of type numpy.float32 or numpy.float64.
			Note that if R_gpu is not instantiated with the kwarg 'order="F"', 
			specifying a fortran-contiguous (row-major) array structure,
			fit_transform will throw an error.	

		Returns
		-------
		T_gpu: pycuda.gpuarray.GPUArray
			`NxK` matrix of the first K principal components of R_gpu. 

		References
		----------
		`[Andrecut, 2008] <https://arxiv.org/pdf/0811.1081.pdf>`_
		

		Notes
		-----
		If n_components was not set, then `K = min(N, P)`. Otherwise, `K = min(n_components, N, P)`

		Examples
		--------
		
		>>> import pycuda.autoinit
		>>> import pycuda.gpuarray as gpuarray
		>>> import numpy as np
		>>> import skcuda.linalg as linalg
		>>> linalg.init()
		>>> pca = linalg.PCA(n_components=4) # map the data to 4 dimensions
		>>> X = np.random.rand(1000,100) # 1000 samples of 100-dimensional data vectors
		>>> X_gpu = gpuarray.GPUArray((1000,100), np.float64, order="F") # note that order="F" or a transpose is necessary. fit_transform requires row-major matrices, and column-major is the default
		>>> X_gpu.set(X) # copy data to gpu
		>>> T_gpu = pca.fit_transform(X_gpu) # calculate the principal components
		>>> linalg.dot(T_gpu[:,0], T_gpu[:,1]) # show that the resulting eigenvectors are orthogonal
		3.637978807091713e-12


		"""

		
		if X_gpu.flags.c_contiguous:
			raise ValueError("Array must be fortran-contiguous. Please instantiate with 'order=\"F\"' or use the transpose of a C-ordered array.")

		R_gpu = X_gpu.copy() # copy X, because it will be altered internally otherwise
	
		n = R_gpu.shape[0] # num samples

		p = R_gpu.shape[1] # num features


		# choose either single or double precision cublas functions
		if R_gpu.dtype == 'float32':

			inpt_dtype = np.float32			

			cuAxpy = cublas.cublasSaxpy
			cuCopy = cublas.cublasScopy
			cuGemv = cublas.cublasSgemv
			cuNrm2 = cublas.cublasSnrm2
			cuScal = cublas.cublasSscal
			cuGer =	cublas.cublasSger

		elif R_gpu.dtype == 'float64':

			inpt_dtype = np.float64

			cuAxpy = cublas.cublasDaxpy
			cuCopy = cublas.cublasDcopy
			cuGemv = cublas.cublasDgemv
			cuNrm2 = cublas.cublasDnrm2
			cuScal = cublas.cublasDscal
			cuGer =	cublas.cublasDger



		else:
			raise ValueError("Array must be of type numpy.float32 or numpy.float64, not '" + R_gpu.dtype + "'") 

		n_components = self.n_components

		if n_components == None or n_components > n or n_components > p:
			n_components = min(n, p)	


		Lambda = np.zeros((n_components,1), inpt_dtype, order="F") # kx1

		P_gpu = gpuarray.zeros((p, n_components), inpt_dtype, order="F") # pxk

		T_gpu = gpuarray.zeros((n, n_components), inpt_dtype, order="F") # nxk


		# mean centering data
		U_gpu = gpuarray.zeros((n,1), np.float32, order="F")
		U_gpu = misc.sum(R_gpu,axis=1) # nx1 sum the columns of R

		for i in xrange(p):
			cuAxpy(self.h, n, -1.0/p, U_gpu.gpudata, 1, R_gpu[:,i].gpudata, 1) 	


		for k in xrange(n_components):

			mu = 0.0

			cuCopy(self.h, n, R_gpu[:,k].gpudata, 1, T_gpu[:,k].gpudata, 1)

			for j in xrange(self.max_iter):

			
				cuGemv(self.h, 't', n, p, 1.0, R_gpu.gpudata, n, T_gpu[:,k].gpudata, 1, 0.0, P_gpu[:,k].gpudata, 1)
		
							
				if k > 0:

					cuGemv(self.h,'t', p, k, 1.0, P_gpu.gpudata, p, P_gpu[:,k].gpudata, 1, 0.0, U_gpu.gpudata, 1)  

					cuGemv (self.h, 'n', p, k, 0.0-1.0, P_gpu.gpudata, p, U_gpu.gpudata, 1, 1.0, P_gpu[:,k].gpudata, 1)


				l2 = cuNrm2(self.h, p, P_gpu[:,k].gpudata, 1)
				cuScal(self.h, p, 1.0/l2, P_gpu[:,k].gpudata, 1)

				cuGemv(self.h, 'n', n, p, 1.0, R_gpu.gpudata, n, P_gpu[:,k].gpudata, 1, 0.0, T_gpu[:,k].gpudata, 1)

				if k > 0:

					cuGemv(self.h, 't', n, k, 1.0, T_gpu.gpudata, n, T_gpu[:,k].gpudata, 1, 0.0, U_gpu.gpudata, 1)
					cuGemv(self.h, 'n', n, k, 0.0-1.0, T_gpu.gpudata, n, U_gpu.gpudata, 1, 1.0, T_gpu[:,k].gpudata, 1)
			

				Lambda[k] = cuNrm2(self.h, n, T_gpu[:,k].gpudata, 1)

				cuScal(self.h, n, 1.0/Lambda[k], T_gpu[:,k].gpudata, 1)
							

				if abs(Lambda[k] - mu) < self.epsilon*Lambda[k]:
					break


				mu = Lambda[k]

			# end for j

			cuGer(self.h, n, p, (0.0-Lambda[k]), T_gpu[:,k].gpudata, 1, P_gpu[:,k].gpudata, 1, R_gpu.gpudata, n)

		# end for k

		for k in xrange(n_components):
			cuScal(self.h, n, Lambda[k], T_gpu[:,k].gpudata, 1) 

		# free gpu memory
		P_gpu.gpudata.free()
		U_gpu.gpudata.free()

		return T_gpu # return the gpu array


	def set_n_components(self, n_components):

		"""
		n_components setter.


		Parameters
		----------
			
		n_components: int
			The new number of principal components to return in fit_transform. 
			Must be None or greater than 0
		"""
	
		if n_components > 0 or n_components == None:
			self.n_components = n_components
		else:
			raise ValueError("n_components can only be greater than 0 or None")


	def get_n_components(self):
		"""
		n_components getter.


		Returns
		-------
		n_components: int
			The current value of self.n_components
		"""

		return self.n_components


