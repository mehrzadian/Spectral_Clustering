from spectral import spectral_clustering as spectral_clustering_old
from mnist import construct_affinity_matrix as construct_affinity_matrix_old
from kmeans import k_means_clustering as k_means_clustering_old
from metrics import clustering_score


from numba import jit, njit, prange, vectorize, guvectorize, cuda

import numpy as np
from numpy import linalg as LA
import timeit


# TODO: Rewrite the k_means_clustering function
@jit(nopython=True, parallel=True, fastmath=True)
def k_means_clustering(data, k, max_iterations=100):

    centroids = data[np.random.choice(data.shape[0], size=k, replace=False)]
    k = float(k)
    for _ in prange(max_iterations):
        distances = np.empty((data.shape[0], int(k)))

        for i in prange(data.shape[0]):
            i = float(i)
            for j in prange(k):
                j = float(j)
                distances[i, j] = np.linalg.norm(data[i] - centroids[j])

        labels = np.empty(data.shape[0], dtype=np.int64)
        for i in prange(data.shape[0]):
            labels[i] = np.argmin(distances[i])

        new_centroids = np.empty((k, data.shape[1]))
        for i in prange(k):
            new_centroids[i] = np.mean(data[labels == i], axis=0)

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


# TODO: Rewrite the laplacian function
@jit(nopython=True, parallel=True, fastmath=True)
def laplacian(A):

    D = np.zeros(A.shape[0])
    for i in prange(A.shape[0]):
        D[i] = np.sum(A[i, :])

    Dis = np.empty(A.shape[0])
    for i in prange(A.shape[0]):
        Dis[i] = 1.0 / np.sqrt(D[i])

    L_sym = np.eye(A.shape[0])
    for i in prange(A.shape[0]):
        for j in prange(A.shape[0]):
            L_sym[i, j] -= Dis[i] * A[i, j] * Dis[j]

    return L_sym


# TODO: Rewrite the spectral_clustering function
@jit(nopython=True, parallel=True, fastmath=True)
def spectral_clustering(affinity, k):
    L = laplacian(affinity)

    _, eig_vecs = np.linalg.eigh(L, k)
    X = eig_vecs[:, :k]
    l, _ = k_means_clustering(X, k)

    return l


# TODO: Rewrite the chamfer_distance function
@jit(nopython=True, parallel=True, fastmath=True)
def chamfer_distance(point_cloud1, point_cloud2):
    N1, D = point_cloud1.shape
    N2, D = point_cloud2.shape
    dists1 = np.empty((N1, N2))
    for i in prange(N1):
        for j in prange(N2):
            dists1[i, j] = np.sum((point_cloud1[i] - point_cloud2[j]) ** 2)
    dists2 = np.empty((N2, N1))
    for i in prange(N2):
        for j in prange(N1):
            dists2[i, j] = np.sum((point_cloud2[i] - point_cloud1[j]) ** 2)
    chamfer_dist1 = np.mean(np.min(dists1, axis=1))
    chamfer_dist2 = np.mean(np.min(dists2, axis=1))
    chamfer_dist = (chamfer_dist1 + chamfer_dist2) / 2
    return chamfer_dist


# TODO: Rewrite the rigid_transform function
@jit(nopython=True, parallel=True, fastmath=True)
def rigid_transform(A, B):
    assert len(A) == len(B)
    centroid_A = np.zeros(A.shape[1])
    for i in prange(A.shape[1]):
        centroid_A[i] = np.mean(A[:, i])
    centroid_B = np.zeros(B.shape[1])
    for i in prange(B.shape[1]):
        centroid_B[i] = np.mean(B[:, i])
    AA = A - centroid_A
    BB = B - centroid_B
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)

    # reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = centroid_B.T - np.dot(R, centroid_A.T)

    return R, t


# TODO: Rewrite the icp function
@jit(nopython=True, parallel=True, fastmath=True)
def nearest_neighbor(src, dst):
    num_src, dim = src.shape
    num_dst, _ = dst.shape

    min_distances = np.empty(num_src)
    indices = np.empty(num_src, dtype=np.int64)
    for i in prange(num_src):
        min_dist = np.inf
        min_index = 0
        for j in prange(num_dst):
            dist = 0.0
            for k in prange(dim):
                diff = src[i, k] - dst[j, k]
                dist += diff * diff
            if dist < min_dist:
                min_dist = dist
                min_index = j
        min_distances[i] = min_dist
        indices[i] = min_index

    return min_distances, indices


@jit(nopython=True, parallel=True, fastmath=True)
def icp(source, target, max_iterations=100, tolerance=1e-5):
    prev_error = 0

    for i in prange(max_iterations):

        distances, indices = nearest_neighbor(source, target)

        R, t = rigid_transform(source, target[indices])

        source = np.dot(R, source.T).T + t
        mean_error = chamfer_distance(source, target)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return R, t, source


# TODO: Rewrite the construct_affinity_matrix function
@jit(nopython=True, parallel=True, fastmath=True)
def euclidean_distance(point1, point2):
    distance = 0.0
    for i in prange(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return np.sqrt(distance)


@jit(nopython=True, parallel=True, fastmath=True)
def construct_affinity_matrix(data, affinity_type, k=3, sigma=1.0):
    num_points = data.shape[0]
    affinity_matrix = np.zeros((num_points, num_points))
    if affinity_type == "knn":
        for i in prange(num_points):
            distances = np.array(
                [euclidean_distance(data[i], data[j]) for j in prange(num_points)]
            )
            knn_indices = np.argpartition(distances, k)[: k + 1]

            for index in knn_indices:
                if index != i:
                    affinity_matrix[i, index] = 1
                    affinity_matrix[index, i] = 1

        return affinity_matrix
    elif affinity_type == "rbf":
        for i in prange(num_points):
            for j in prange(i, num_points):
                distance = euclidean_distance(data[i], data[j])
                affinity = np.exp(-1 * distance * distance / (2 * sigma * sigma))
                affinity_matrix[i, j] = affinity_matrix[j, i] = affinity

        return affinity_matrix
    else:
        raise Exception("invalid affinity matrix type")


if __name__ == "__main__":
    dataset = "mnist"

    dataset = np.load("datasets/%s.npz" % dataset)
    X = dataset["data"]  # feature points
    y = dataset["target"]  # ground truth labels
    n = len(np.unique(y))  # number of clusters

    # TODO: Run both the old and speed up version of your algorithms and capture running time
    # K-means clustering
    start = timeit.default_timer()
    y_pred_old, _ = k_means_clustering_old(X, n)
    end = timeit.default_timer()
    print("Old K-means clustering time:", end - start)
    start = timeit.default_timer()
    y_pred, _ = k_means_clustering(X, n)
    end = timeit.default_timer()
    print("K-means clustering time:", end - start)
    # print("Old K-means on %s:" % dataset, clustering_score(y, y_pred_old))
    print("K-means on %s:" % dataset, clustering_score(y, y_pred))
    # print time
    # print("Old K-means clustering time:", timeit("k_means_clustering_old(X, n)", globals=globals(), number=10))
    print(
        "K-means clustering time:",
        timeit("k_means_clustering(X, n)", globals=globals(), number=10),
    )

    # Spectral clustering
    Ach_old = construct_affinity_matrix_old(X)
    y_pred_old = spectral_clustering_old(Ach_old, n)
    Ach = construct_affinity_matrix(X, "rbf", sigma=1.0)
    y_pred = spectral_clustering(Ach, n)
    print("Old spectral on %s:" % dataset, clustering_score(y, y_pred_old))
    print("Spectral on %s:" % dataset, clustering_score(y, y_pred))

    # print("Old Chamfer affinity on %s:" % dataset, clustering_score(y, y_pred_old))
    # print("Chamfer affinity on %s:" % dataset, clustering_score(y, y_pred))

    # TODO: Compare the running time using timeit module

    pass
