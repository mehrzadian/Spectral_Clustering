import numpy as np


# https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
def mutual_information(c1, c2):

    contingency = np.histogram2d(c1, c2)[0]
    margins_entropy = entropy(np.sum(contingency, axis=0)) + entropy(
        np.sum(contingency, axis=1)
    )
    return (
        np.sum(contingency * np.log(contingency + np.finfo(float).eps))
        - margins_entropy
    )


def entropy(c):
    n = np.sum(c)
    p = c / n  # probs
    return -np.sum(p * np.log(p + np.finfo(float).eps))


def nmi(true_labels, predicted_labels):
    sqrt_product_of_entrpies = np.sqrt(entropy(true_labels) * entropy(predicted_labels))
    return mutual_information(true_labels, predicted_labels) / sqrt_product_of_entrpies


def ari(true_labels, predicted_labels):

    contingency = np.histogram2d(true_labels, predicted_labels)[0]
    sum_row = np.sum(np.sum(contingency, axis=1) ** 2)
    sum_col = np.sum(np.sum(contingency, axis=0) ** 2)

    sum_ = np.sum(contingency**2)

    total = contingency.sum() ** 2

    index = sum_ - total
    max_index = 0.5 * (sum_row + sum_col) - total
    min_index = min(sum_row, sum_col) - total

    ari = (index - min_index) / (max_index - min_index + np.finfo(float).eps)

    return ari


def clustering_score(true_labels, predicted_labels, metric="nmi"):
    """
    Calculate the clustering score to assess the accuracy of predicted labels compared to true labels.

    Parameters:
    - true_labels: List or numpy array, true cluster labels for each data point.
    - predicted_labels: List or numpy array, predicted cluster labels for each data point.

    Returns:
    - score: float, clustering score indicating the accuracy of predicted labels.
    """

    # TODO: Calculate and return clustering score
    if metric == "nmi":
        score = nmi(true_labels, predicted_labels)
    else:
        score = ari(true_labels, predicted_labels)

    return score


if __name__ == "__main__":
    true_labels = np.array([0, 0, 1, 1, 2, 2])
    predicted_labels = np.array([0, 0, 1, 1, 1, 1])
    print(clustering_score(true_labels, predicted_labels))
