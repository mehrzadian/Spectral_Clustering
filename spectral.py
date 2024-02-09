from kmeans import k_means_clustering
from numpy import linalg as LA
import numpy as np

def laplacian(A):
    D = np.diag(np.sum(A, axis=1))

    
    Dinvsqrt = np.diag(1.0 / np.sqrt(np.diag(D)))

    
    L_sym = np.eye(A.shape[0]) - np.dot(Dinvsqrt, np.dot(A, Dinvsqrt))

    return L_sym

def spectral_clustering(affinity, k):
    """
    Perform spectral clustering on the given affinity matrix.

    Parameters:
    - affinity: numpy array, affinity matrix capturing pairwise relationships between data points.
    - k: int, number of clusters.

    Returns:
    - labels: numpy array, cluster labels assigned by the spectral clustering algorithm.
    """

    # TODO: Compute Laplacian matrix
    L = laplacian(affinity)
    # TODO: Compute the first k eigenvectors of the Laplacian matrix
    _, eig_vecs = np.linalg.eigh(L)
    X = eig_vecs[:, :k]
    # TODO: Apply K-means clustering on the selected eigenvectors
    l,_ = k_means_clustering(X, k)
    # TODO: Return cluster labels
    return l
    
