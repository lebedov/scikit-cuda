#!/usr/bin/env python

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg
from skcuda.linalg import PCA as cuPCA
import skcuda.misc as cumisc
 
pca = cuPCA() # take all principal components

demo_types = [np.float32, np.float64] # we can use single or double precision
precisions = ['single', 'double']

print("Principal Component Analysis Demo!")
print("Compute all 100 principal components of a 1000x100 data matrix")
print("Lets test if the first two resulting eigenvectors (principal components) are orthogonal, by dotting them and seeing if it is about zero, then we can see the amount of the origial variance explained by just two of the original 100 dimensions.\n\n\n")

for i in range(len(demo_types)):

	demo_type = demo_types[i]
	X = np.random.rand(1000,100).astype(demo_type) # 1000 samples of 100-dimensional data vectors
	X_gpu = gpuarray.GPUArray((1000,100), demo_type, order="F") # note that order="F" or a transpose is necessary. fit_transform requires row-major matrices, and column-major is the default
	X_gpu.set(X) # copy data to gpu
	T_gpu = pca.fit_transform(X_gpu) # calculate the principal components
	dot_product = linalg.dot(T_gpu[:,0], T_gpu[:,1]) # show that the resulting eigenvectors are orthogonal 
	print("The dot product of the two " + str(precisions[i]) + " precision eigenvectors is: " + str(dot_product))
	# now get the variance of each eigenvector so we can see the percent explained by the first two
	std_vec = np.std(T_gpu.get(), axis=0)
	print("We explained " + str(100 * np.sum(std_vec[:2]) / np.sum(std_vec)) + "% of the variance with 2 principal components in " +  str(precisions[i]) +  " precision\n\n")


















