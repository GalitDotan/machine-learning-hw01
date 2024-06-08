from collections import Counter

import numpy as np
from numpy import ndarray
from scipy.spatial.distance import euclidean
from sklearn.datasets import fetch_openml

_MNIST = fetch_openml('mnist_784', as_frame=False)
DATA = _MNIST['data']  # images of 28 X 28 (matrix)
LABELS = _MNIST['target']  # for each image it has a label - the number in the picture

idx = np.random.RandomState(0).choice(70000, 11000)  # 11000 numbers from 0 to 70000


def _get_k_nearest_neighbors_indexes(images: ndarray[ndarray],
                                     image: ndarray[int],
                                     k: int) -> list[int]:
    distances = np.array([(i, euclidean(image, image2)) for i, image2 in enumerate(images)])
    sorted_distances = sorted(distances, key=lambda item: item[1])
    k_nearest = sorted_distances[:k]
    return [int(image_tuple[0]) for image_tuple in k_nearest]


def k_nearest_neighbors(train_images: ndarray[ndarray[int]],
                        train_labels: ndarray[int],
                        query_image: ndarray[int],
                        k: int = 1) -> int:
    k_neighbors_indexes = _get_k_nearest_neighbors_indexes(train_images, query_image, k)
    k_labels = [train_labels[i] for i in k_neighbors_indexes]
    counter = Counter(k_labels)
    result: tuple[int, int] = counter.most_common(1)[0]  # (<label>, count)
    return result[0]


if __name__ == '__main__':
    TRAIN_SIZE = 1000
    TEST_SIZE = 1

    train_images = DATA[idx[:TRAIN_SIZE], :].astype(int)
    train_labels = LABELS[idx[:TRAIN_SIZE]]

    test_images = DATA[idx[TRAIN_SIZE:], :].astype(int)
    expected_labels = LABELS[idx[TRAIN_SIZE:]]

    correct = 0
    wrong = 0

    for image, expected_label in zip(test_images, expected_labels):
        actual_label = k_nearest_neighbors(train_images=train_images,
                                           train_labels=train_labels,
                                           query_image=image,
                                           k=20)
        print(f'Expected: {expected_label}, Actual: {actual_label}')
        if expected_label == actual_label:
            correct += 1
        else:
            wrong += 1
    print(f'Correct: {correct}, Wrong: {wrong}. Accuracy: {correct / len(test_images)}')
