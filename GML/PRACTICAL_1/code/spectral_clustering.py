import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as skm
import scipy
import scipy.spatial.distance as sd

from utils import plot_clustering_result, plot_the_bend, min_span_tree
from build_similarity_graph import build_similarity_graph
from generate_data import blobs, two_moons, point_and_circle


def build_laplacian(W, laplacian_normalization=""):
    """
    Compute graph Laplacian.

    :param W: adjacency matrix
    :param laplacian_normalization:  string selecting which version of the laplacian matrix to construct
                                     'unn':  unnormalized,
                                     'sym': symmetric normalization
                                     'rw':  random-walk normalization
    :return: L: (n x n) dimensional matrix representing the Laplacian of the graph
    """
    
    # The degree matrix is diagonal.
    # The i-th diagonal component is the sum of the weights of the i-th row of the W matrix
    D = np.zeros(W.shape)
    for i in range(W.shape[0]):
        D[i,i] = np.sum(W[i,:])
    
    if laplacian_normalization == "unn":
        L = D - W
    elif laplacian_normalization == "sym":
        diag = np.linalg.inv(D)
        sqrtdiag = np.sqrt(diag)
        L = np.eye(W.shape[0]) - np.dot(np.dot(sqrtdiag,W),sqrtdiag)
    elif laplacian_normalization == "rw":
        diag = np.linalg.inv(D)
        sqrtdiag = np.sqrt(diag)
        L = np.eye(W.shape[0]) - np.dot(sqrtdiag,W)
    else :
        return "Problem with the normalisation"
    return L


def spectral_clustering(L, chosen_eig_indices, num_classes=2):
    """
    :param L: Graph Laplacian (standard or normalized)
    :param chosen_eig_indices: indices of eigenvectors to use for clustering
    :param num_classes: number of clusters to compute (defaults to 2)
    :return: Y: Cluster assignments
    """

    """
    Use the function scipy.linalg.eig or the function scipy.sparse.linalg.eigs to compute:
    U = (n x n) eigenvector matrix           (sorted)
    E = (n x n) eigenvalue diagonal matrix   (sorted)
    """
    eigenvals, eigenvect = scipy.linalg.eig(L)
#    eigenvals, eigenvect = scipy.sparse.linalg.eigs(L, k = len(chosen_eig_indices))
    ind_sorted = np.argsort(eigenvals)
    E = eigenvals[ind_sorted]
    U = eigenvect[:,ind_sorted]
    
    
    print(U)
    """
    compute the clustering assignment from the eigenvectors        
    Y = (n x 1) cluster assignments [0,1,...,c-1]                   
    """
    Y = KMeans(num_classes).fit_predict(np.real(U)[:,chosen_eig_indices])
    return Y


def two_blobs_clustering():
    """
    TO BE COMPLETED

    Clustering of two blobs. Used in questions 2.1 and 2.2
    """

    # Get data and compute number of classes
    X, Y = blobs(600, n_blobs=2, blob_var=0.15, surplus=0)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 0
    var = 1.0  # exponential_euclidean's sigma^2

    laplacian_normalization = 'unn'
    chosen_eig_indices = [0,1]    # indices of the ordered eigenvalues to pick

    # build laplacian
    if k == 0 :
        dists = sd.cdist(X,X, metric="euclidean")
        min_tree = min_span_tree(dists)   
        distance_threshold = dists[min_tree].max()
        eps = np.exp(-distance_threshold**2.0/(2*var))
        
        W = build_similarity_graph(X, var=var, k=k, eps=eps)
    else :
        W = build_similarity_graph(X, var=var, k=k)
        
    L = build_laplacian(W, laplacian_normalization)

    # run spectral clustering
    Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes=num_classes)

    # Plot results
    plot_clustering_result(X, Y, L, Y_rec, KMeans(num_classes).fit_predict(X))


def choose_eigenvalues(eigenvalues):
    """
    Function to choose the indices of which eigenvalues to use for clustering.

    :param eigenvalues: sorted eigenvalues (in ascending order)
    :return: indices of the eigenvalues to use
    """
    differences = np.diff(eigenvalues.real[1:])
    eig_ind = list(range(0,np.argmax(differences)+2))
    
    print(eig_ind)
    return eig_ind


def spectral_clustering_adaptive(L, num_classes=2):
    """
    Spectral clustering that adaptively chooses which eigenvalues to use.
    :param L: Graph Laplacian (standard or normalized)
    :param num_classes: number of clusters to compute (defaults to 2)
    :return: Y: Cluster assignments
    """

    """
    Use the function scipy.linalg.eig or the function scipy.linalg.eigs to compute:
    U = (n x n) eigenvector matrix           (sorted)
    E = (n x n) eigenvalue diagonal matrix   (sorted)
    """
    eigenvals, eigenvect = scipy.linalg.eig(L)
    ind_sorted = np.argsort(eigenvals)
    E = eigenvals[ind_sorted]
    U = eigenvect[:,ind_sorted]
    

    """
    compute the clustering assignment from the eigenvectors        
    Y = (n x 1) cluster assignments [0,1,...,c-1]                   
    """
    ind = choose_eigenvalues(E)
    Y = KMeans(num_classes).fit_predict(np.real(U)[:,ind])
    return Y


def find_the_bend():
    """
    TO BE COMPLETED

    Used in question 2.3
    :return:
    """

    # the number of samples to generate
    num_samples = 600

    # Generate blobs and compute number of clusters
    X, Y = blobs(num_samples, 4, 0.2)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 0
    var = 1  # exponential_euclidean's sigma^2
    laplacian_normalization = 'unn'  # either 'unn'normalized, 'sym'metric normalization or 'rw' random-walk normalization


    # build laplacian
    if k == 0 :
        dists = sd.cdist(X,X, metric="euclidean")
        min_tree = min_span_tree(dists)   
        distance_threshold = dists[min_tree].max()
        eps = np.exp(-distance_threshold**2.0/(2*var))
        print(eps)
        W = build_similarity_graph(X, var=var, k=k, eps=eps)
    else :
        W = build_similarity_graph(X, var=var, k=k)
    L = build_laplacian(W, laplacian_normalization)

    """
    compute first 15 eigenvalues and call choose_eigenvalues() to choose which ones to use. 
    """
    eigenvalues, vects = scipy.linalg.eig(L)
    eigenvalues = sorted(eigenvalues.real)
#    for ind,val in enumerate(eigenvalues[:15]):
#        plt.scatter(ind, val)
#    plt.xlabel("index of the eigenvalue")
#    plt.ylabel("value of the eigenvalue")
#    chosen_eig_indices =  [0,1,2,3]  # indices of the ordered eigenvalues to pick


    """
    compute spectral clustering solution using a non-adaptive method first, and an adaptive one after (see handout) 
    Y_rec = (n x 1) cluster assignments [0,1,..., c-1]    
    """
    # run spectral clustering
#    Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes=num_classes)
    Y_rec_adaptive = spectral_clustering_adaptive(L, num_classes=num_classes)

#    plot_the_bend(X, Y, L, Y_rec, eigenvalues)
    plot_the_bend(X, Y, L, Y_rec_adaptive, eigenvalues)

def two_moons_clustering():
    """
    TO BE COMPLETED.

    Used in question 2.7
    """
    # Generate data and compute number of clusters
    X, Y = two_moons(600)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 0
    var = 1.0  # exponential_euclidean's sigma^2

    laplacian_normalization = 'unn'
    chosen_eig_indices = [0, 1]    # indices of the ordered eigenvalues to pick


    # build laplacian
    # build laplacian
    if k == 0 :
        dists = sd.cdist(X,X, metric="euclidean")
        min_tree = min_span_tree(dists)   
        distance_threshold = dists[min_tree].max()
        eps = np.exp(-distance_threshold**2.0/(2*var))
        print(eps)
        W = build_similarity_graph(X, var=var, k=k, eps=eps)
    else :
        W = build_similarity_graph(X, var=var, k=k)
    L = build_laplacian(W, laplacian_normalization)

#    Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes=num_classes)
#
#    plot_clustering_result(X, Y, L, Y_rec, KMeans(num_classes).fit_predict(X))

    Y_rec_adaptive = spectral_clustering_adaptive(L, num_classes=num_classes)
    
    plot_clustering_result(X, Y, L, Y_rec_adaptive, KMeans(num_classes).fit_predict(X))

def point_and_circle_clustering():
    """
    TO BE COMPLETED.

    Used in question 2.8
    """
    # Generate data and compute number of clusters
    X, Y = point_and_circle(600)
    num_classes = len(np.unique(Y))
    

    """
    Choose parameters
    """
    k = 50
    var = 1.0  # exponential_euclidean's sigma^2

    chosen_eig_indices = [0, 1]    # indices of the ordered eigenvalues to pick


    # build laplacian
    W = build_similarity_graph(X, var=var, k=k)
    
    
    L_unn = build_laplacian(W, 'unn')
    L_norm = build_laplacian(W, 'rw')

    
    Y_unn = spectral_clustering(L_unn, chosen_eig_indices, num_classes=num_classes)
    Y_norm = spectral_clustering(L_norm, chosen_eig_indices, num_classes=num_classes)

    plot_clustering_result(X, Y, L_unn, Y_unn, Y_norm, 1)


def parameter_sensitivity():
    """
    TO BE COMPLETED.

    A function to test spectral clustering sensitivity to parameter choice.
print
    Used in question 2.9
    """
    # the number of samples to generate
    num_samples = 500

    """
    Choose parameters
    """
    var = 1.0  # exponential_euclidean's sigma^2
    laplacian_normalization = 'rw'
    chosen_eig_indices = [0, 1]

    """
    Choose candidate parameters
    """
#    parameter_candidate = range(1,30) # the number of neighbours for the graph or the epsilon threshold
    parameter_candidate = np.linspace(0.05,1,20)
    parameter_performance = []

    for eps in parameter_candidate:
        # Generate data
        X, Y = two_moons(num_samples, 1, 0.02)
        num_classes = len(np.unique(Y))
        
        
        W = build_similarity_graph(X, k=0, eps=eps)
        L = build_laplacian(W, laplacian_normalization)
        
        Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes)
                
        parameter_performance.append(skm.adjusted_rand_score(Y, Y_rec))
        
    plt.figure()
    plt.plot(parameter_candidate, parameter_performance)
    plt.title('parameter sensitivity')
    plt.show()
    print(parameter_performance)
