import numpy as np
import matplotlib.pyplot as plt

from kmeans import k_means_clustering
from spectral import spectral_clustering
from metrics import clustering_score


def construct_affinity_matrix(data, affinity_type, *, k=3, sigma=1.0):
    """
    Construct the affinity matrix for spectral clustering based on the given data.

    Parameters:
    - data: numpy array, mxn representing m points in an n-dimensional dataset.
    - affinity_type: str, type of affinity matrix to construct. Options: 'knn' or 'rbf'.
    - k: int, the number of nearest neighbors for the KNN affinity matrix (default: 3).
    - sigma: float, bandwidth parameter for the RBF kernel (default: 1.0).

    Returns:
    - affinity_matrix: numpy array, the constructed affinity matrix based on the specified type.
    """

    # TODO: Compute pairwise distances
    distances = np.linalg.norm(data[:, np.newaxis] - data, axis=2)

    if affinity_type == "knn":
        # TODO: Find k nearest neighbors for each point

        knn_indices = np.argpartition(distances, k, axis=1)[:, :k]
        row_indices = np.repeat(np.arange(data.shape[0]), k)
        affinity_matrix = np.zeros_like(distances)
        affinity_matrix[row_indices, knn_indices.flatten()] = 1
        affinity_matrix[knn_indices.flatten(), row_indices] = 1
        return affinity_matrix

    elif affinity_type == "rbf":
        # TODO: Apply RBF kernel
        affinity_matrix = np.exp(-(distances**2) / (2.0 * sigma**2))
        # TODO: Return affinity matrix
        return affinity_matrix

    else:
        raise Exception("invalid affinity matrix type")


if __name__ == "__main__":
    datasets = ["blobs", "circles", "moons"]

    # TODO: Create and configure plot
    fig, axes = plt.subplots(3, 4, figsize=(12, 16))
    plt.subplots_adjust(hspace=0.5)
    plt.setp(axes, xticks=(), yticks=())
    # axes[0, 0].set_title('K-means ')
    # axes[0, 1].set_title('RBF affinity')
    # axes[0, 2].set_title('KNN affinity')
    # axes[0, 3].set_title('Ground truth')
    for ds_name in datasets:
        dataset = np.load("datasets/%s.npz" % ds_name)
        X = dataset["data"]  # feature points
        y = dataset["target"]  # ground truth labels
        n = len(np.unique(y))  # number of clusters

        k = 11
        sigma = 0.06

        y_km, _ = k_means_clustering(X, n)
        Arbf = construct_affinity_matrix(X, "rbf", sigma=sigma)
        y_rbf = spectral_clustering(Arbf, n)
        Aknn = construct_affinity_matrix(X, "knn", k=k)
        y_knn = spectral_clustering(Aknn, n)
        print("Ground Truth on %s:" % ds_name, clustering_score(y, y, metric="ari"))
        print("K-means on %s:" % ds_name, clustering_score(y, y_km, metric="ari"))
        print("RBF affinity on %s:" % ds_name, clustering_score(y, y_rbf, metric="ari"))
        print("KNN affinity on %s:" % ds_name, clustering_score(y, y_knn, metric="ari"))
        # library spectral clustering
        # spectral_model = SpectralClustering(n_clusters= 2, affinity='precomputed')
        # spectral_labels = spectral_model.fit_predict(Arbf)
        # print("Spectral Clustering on %s:" % ds_name, clustering_score(y, spectral_labels))

        # TODO: Create subplots
        axes[datasets.index(ds_name), 0].scatter(X[:, 0], X[:, 1], c=y_km)
        axes[datasets.index(ds_name), 0].set_title(
            f"K-means on {ds_name} \n {clustering_score(y, y_km)}"
        )
        axes[datasets.index(ds_name), 1].scatter(X[:, 0], X[:, 1], c=y_rbf)
        axes[datasets.index(ds_name), 1].set_title(
            f"RBF affinity on {ds_name} \n {clustering_score(y, y_rbf)}"
        )
        axes[datasets.index(ds_name), 2].scatter(X[:, 0], X[:, 1], c=y_knn)
        axes[datasets.index(ds_name), 2].set_title(
            f"KNN affinity on {ds_name} \n {clustering_score(y, y_knn)}"
        )
        axes[datasets.index(ds_name), 3].scatter(X[:, 0], X[:, 1], c=y)
        axes[datasets.index(ds_name), 3].set_title(
            f"Ground truth on {ds_name} \n {clustering_score(y, y)}"
        )

        # spectral clustering
        # axes[datasets.index(ds_name), 4].scatter(X[:, 0], X[:, 1], c=spectral_labels, cmap='viridis')
        # axes[datasets.index(ds_name), 4].set_title('Spectral Clustering on %s' % ds_name)

    # TODO: Show subplots
    plt.show()
