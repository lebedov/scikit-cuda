#!/usr/bin/env python

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg
from skcuda.linalg import PCA as cuPCA
from matplotlib import pyplot as plt 
from sklearn import datasets

iris = datasets.load_iris()
X_orig = iris.data
y = iris.target

pca = cuPCA(4) # take all 4 principal components

demo_types = [np.float32, np.float64] # we can use single or double precision
precisions = ['single', 'double']

print("Principal Component Analysis Demo!")
print("Compute 2 principal components of a 1000x4 IRIS data matrix")
print("Lets test if the first two resulting eigenvectors (principal components) are orthogonal,"
      " by dotting them and seeing if it is about zero, then we can see the amount of the origial"
      " variance explained by just two of the original 4 dimensions. Then we will plot the reults"
      " for the double precision experiment.\n\n\n")

for i in range(len(demo_types)):

    demo_type = demo_types[i]

    # 1000 samples of 4-dimensional data vectors
    X = X_orig.astype(demo_type) 

    X_gpu = gpuarray.to_gpu(X) # copy data to gpu

    T_gpu = pca.fit_transform(X_gpu) # calculate the principal components

    # show that the resulting eigenvectors are orthogonal 
    # Note that GPUArray.copy() is necessary to create a contiguous array 
    # from the array slice, otherwise there will be undefined behavior
    dot_product = linalg.dot(T_gpu[:,0].copy(), T_gpu[:,1].copy()) 	
    T = T_gpu.get()

    print("The dot product of the first two " + str(precisions[i]) + 
            " precision eigenvectors is: " + str(dot_product))
    
    # now get the variance of the eigenvectors and create the ratio explained from the total
    std_vec = np.std(T, axis=0)
    print("We explained " + str(100 * np.sum(std_vec[:2]) / np.sum(std_vec)) + 
            "% of the variance with 2 principal components in " +  
            str(precisions[i]) +  " precision\n\n")

    # Plot results for double precision
    if i == len(demo_types)-1:
        # Different color means different IRIS class
        plt.scatter(T[:,0], T[:,1], c=y, cmap=plt.cm.Set1, edgecolor='k', s=20)
        plt.show()















