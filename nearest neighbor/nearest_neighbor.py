from collections import Counter

import numpy
from sklearn.datasets import fetch_openml


def _get_euclidean_distances() -> float:
    pass


def _find_k_neighbors(data, k):
    pass


def k_nearest_neighbors(train_images: set = None, label_vector: set = None, query_image=None, k: int = 1) -> int:
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']  # images of 28 X 28 (matrix)
    labels = mnist['target']  # for each image it has a label - the number in the picture

    idx = numpy.random.RandomState(0).choice(70000, 11000)  # 11000 numbers from 0 to 70000

    train = data[idx[:10000], :].astype(int)  # 10000 train images
    train_labels = labels[idx[:10000]]

    test = data[idx[10000:], :].astype(int)  # 1000 test images
    test_labels = labels[idx[1000:]]

    k_neighbors = _find_k_neighbors()
    counter = Counter(k_neighbors)  # count labels

    pass


if __name__ == '__main__':
    print(k_nearest_neighbors())
