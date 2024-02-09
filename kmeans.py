import numpy as np


def k_means_clustering(data, k, max_iterations=1000):
    """
    Perform K-means clustering on the given dataset.

    Parameters:
    - data: numpy array, mxn representing m points in an n-dimensional dataset.
    - k: int, the number of resulting clusters.
    - max_iterations: int, optional parameter to prevent potential infinite loops (default: 100).

    Returns:
    - labels: numpy array, cluster labels for each data point.
    - centroids: numpy array, final centroids of the clusters.
    """

    centroids = data[np.random.choice(data.shape[0], size=k, replace=False)]

    for _ in range(max_iterations):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        # convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids
