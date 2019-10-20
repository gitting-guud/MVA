from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *
from build_similarity_graph import build_similarity_graph
from spectral_clustering import build_laplacian, spectral_clustering



def image_segmentation(input_img='four_elements.bmp'):
    """
    TO BE COMPLETED

    Function to perform image segmentation.

    :param input_img: name of the image file in /data (e.g. 'four_elements.bmp')
    """
    filename = os.path.join('data', input_img)

    X = io.imread(filename)
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    
    im_side = np.size(X, 1)
    Xr = X.reshape(im_side ** 2, 3)
    print(Xr.shape)
    """
    Y_rec should contain an index from 0 to c-1 where c is the     
     number of segments you want to split the image into          
    """

    """
    Choose parameters
    """
    var = 10# Tried multiple values before fixing it to this
    k = 60 # Tried multiple values before fixing it to this
    laplacian_normalization = 'sym'
    chosen_eig_indices = [1,2]# The eigen vectors to consider for the clustering, not used if using adaptive spectral clustering
    num_classes = 5 # Fixed by hand, we can also look at the values of the eigenvectors to infer this number


    # First we build the graph using KNN (this is what takes few minutes)
#    W = build_similarity_graph(Xr, var=var, k=0, eps= 0.5)
    W = build_similarity_graph(Xr, var=var, k=0, eps=0.7)
    # Build le laplacian matrix
    L = build_laplacian(W, laplacian_normalization)
    # Perform the spectral clustering where we choose automatically the number
    # of eigenvectors to consider using first order derivatives or by hand
    
    
    Y_rec = spectral_clustering(L,chosen_eig_indices, num_classes=num_classes)
#    Y_rec = spectral_clustering_adaptive(L, num_classes=num_classes)

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(X)
#    plt.imshow(X[:30,:30,:])
    plt.subplot(1, 2, 2)
    Y_rec = Y_rec.reshape(im_side, im_side)
#    Y_rec = Y_rec.reshape(30,30)
    plt.imshow(Y_rec)

    plt.show()


if __name__ == '__main__':
    image_segmentation()
