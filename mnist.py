import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

from spectral import spectral_clustering
from metrics import clustering_score


def chamfer_distance(point_cloud1, point_cloud2):
    """
    Calculate the Chamfer distance between two point clouds.

    Parameters:
    - point_cloud1: numpy array, shape (N1, D), representing the first point cloud.
    - point_cloud2: numpy array, shape (N2, D), representing the second point cloud.

    Returns:
    - dist: float, the Chamfer distance between the two point clouds.
    """
    N1, _ = point_cloud1.shape
    N2, _ = point_cloud2.shape
    # norm
    dists1 = (
        np.sum(point_cloud1**2, axis=1).reshape(N1, 1)
        + np.sum(point_cloud2**2, axis=1)
        - 2 * np.dot(point_cloud1, point_cloud2.T)
    )
    dists2 = (
        np.sum(point_cloud2**2, axis=1).reshape(N2, 1)
        + np.sum(point_cloud1**2, axis=1)
        - 2 * np.dot(point_cloud2, point_cloud1.T)
    )
    # TODO: Calculate distances from each point in point_cloud2 to the nearest point in point_cloud1
    chamfer_dist1 = np.mean(np.min(dists1, axis=1))
    chamfer_dist2 = np.mean(np.min(dists2, axis=1))
    # TODO: Return Chamfer distance, sum of the average distances in both directions
    chamfer_dist = (chamfer_dist1 + chamfer_dist2) / 2
    return chamfer_dist


def rigid_transform(A, B):
    """
    Find the rigid (translation + rotation) transformation between two sets of points.

    Parameters:
    - A: numpy array, mxn representing m points in an n-dimensional space.
    - B: numpy array, mxn representing m points in an n-dimensional space.

    Returns:
    - R: numpy array, n x n rotation matrix.
    - t: numpy array, translation vector.
    """
    # TODO: Subtract centroids to center the point clouds A and B
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # TODO: Construct Cross-Covariance matrix
    H = np.dot(AA.T, BB)

    # TODO: Apply SVD to the Cross-Covariance matrix
    U, S, Vt = np.linalg.svd(H)
    # TODO: Calculate the rotation matrix
    R = np.dot(Vt.T, U.T)
    # TODO: Calculate the translation vector
    t = centroid_B.T - np.dot(R, centroid_A.T)
    # TODO: Return rotation and translation matrices
    return R, t


def nearest_neighbor(src, dst):
    dists = np.linalg.norm(src[:, np.newaxis] - dst, axis=2)

    # the index of the nearest neighbor
    indices = np.argmin(dists, axis=1)
    return indices


def icp(source, target, max_iterations=100, tolerance=1e-5):
    """
    Perform ICP (Iterative Closest Point) between two sets of points.

    Parameters:
    - source: numpy array, mxn representing m source points in an n-dimensional space.
    - target: numpy array, mxn representing m target points in an n-dimensional space.
    - max_iterations: int, maximum number of iterations for ICP.
    - tolerance: float, convergence threshold for ICP.

    Returns:
    - R: numpy array, n x n rotation matrix.
    - t: numpy array, translation vector.
    - transformed_source: numpy array, mxn representing the transformed source points.
    """
    prev_error = 0
    # TODO: Iterate until convergence
    for _ in range(max_iterations):

        indices = nearest_neighbor(source, target)

        # TODO: Calculate rigid transformation
        R, t = rigid_transform(source, target[indices])

        # TODO: Apply transformation to source points
        source = np.dot(R, source.T).T + t
        # TODO: Calculate Chamfer distance
        mean_error = chamfer_distance(source, target)
        # TODO: Check for convergence
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    # TODO: Return the transformed source
    return R, t, source


def construct_affinity_matrix(point_clouds):
    """
    Construct the affinity matrix for spectral clustering based on the given data.

    Parameters:
    - point_clouds: numpy array, mxnxd representing m point clouds each containing n points in a d-dimensional space.

    Returns:
    - affinity_matrix: numpy array, the constructed affinity matrix using Chamfer distance.
    """

    # TODO: Iterate over point clouds to fill affinity matrix
    m = point_clouds.shape[0]

    affinity_matrix = np.zeros((m, m))

    # TODO: For each pair of point clouds, register them with each other
    # and calculate the Chamfer distance between the registered clouds
    for i in range(m):
        for j in range(i + 1, m):
            R, _, transformed_source = icp(point_clouds[i], point_clouds[j])
            dist = chamfer_distance(transformed_source, point_clouds[j])
            affinity_matrix[i, j] = affinity_matrix[j, i] = dist
    # distances to similarities using Gaussian
    sigma = np.mean(affinity_matrix)
    affinity_matrix = np.exp(-(affinity_matrix**2) / (2.0 * sigma**2))

    return affinity_matrix


if __name__ == "__main__":
    dataset = "mnist"

    dataset = np.load("datasets/%s.npz" % dataset)
    X = dataset["data"]  # feature points
    y = dataset["target"]  # ground truth labels
    n = len(np.unique(y))  # number of clusters

    Ach = construct_affinity_matrix(X)
    y_pred = spectral_clustering(Ach, n)
    print("Chamfer affinity on %s:" % dataset, c_y := clustering_score(y, y))

    print("Chamfer affinity on %s:" % dataset, c_ypred := clustering_score(y, y_pred))
    print("The accuracy percentage is", (c_ypred / c_y) * 100)
    # TODO: Plot Ach using its first 3 eigenvectors
    eigvals, eigvecs = np.linalg.eigh(Ach)
    idx = np.argsort(eigvals)
    eigvecs = eigvecs[:, idx]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(eigvecs[:, -1], eigvecs[:, -2], eigvecs[:, -3], c=y_pred)
    plt.show()
