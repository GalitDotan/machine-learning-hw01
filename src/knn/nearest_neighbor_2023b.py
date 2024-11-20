from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.spatial.distance import euclidean

from knn.nearest_neighbor import TEST_IMAGES, TEST_LABELS, TRAIN_IMAGES, TRAIN_LABELS


def _get_k_nearest_neighbors_indexes(images: ndarray[ndarray],
                                     image: ndarray[int],
                                     k: int) -> list[int]:
    distances = np.array([(i, euclidean(image, image2)) for i, image2 in enumerate(images)])
    sorted_distances = sorted(distances, key=lambda item: item[1])
    k_nearest = sorted_distances[:k]
    return [int(image_tuple[0]) for image_tuple in k_nearest]


def k_nearest_neighbors(images: ndarray[ndarray[int]],
                        labels: ndarray[int],
                        query_image: ndarray[int],
                        k: int = 1) -> int:
    k_neighbors_indexes = _get_k_nearest_neighbors_indexes(images, query_image, k)
    k_labels = [labels[i] for i in k_neighbors_indexes]
    counter = Counter(k_labels)
    result: tuple[int, int] = counter.most_common(1)[0]  # (<label>, count)
    return result[0]


def run_simulation(n: int, k: int) -> float:
    """
    Runs the nearest neighbor and calculates accuracy percentage.

    Args:
        n: number of training images. Would take the first n of `train`.
        k: amount of nearest neighbors.

    Returns:
        accuracy percentage.
    """
    correct, wrong = 0, 0

    for image, expected_label in zip(TEST_IMAGES, TEST_LABELS):
        actual_label = k_nearest_neighbors(images=TRAIN_IMAGES[:n], labels=TRAIN_LABELS[:n], query_image=image, k=k)
        # print(f'Expected: {expected_label}, Actual: {actual_label}')
        if expected_label == actual_label:
            correct += 1
        else:
            wrong += 1
    accuracy = 100 * correct / len(TEST_IMAGES)
    print(f'n={n}, k={k}, accuracy={accuracy}')
    return accuracy


def run_question_b():
    n, k = 1000, 10
    accuracy = run_simulation(n=n, k=k)
    print(f'Result for n={n}, k={k} is {accuracy}')


def run_question_c():
    n = 1000
    k_values = np.array(range(1, 101))
    accuracy = [run_simulation(n, k) for k in k_values]
    plt.plot(k_values, accuracy)
    plt.xlabel('k')
    plt.ylabel('Accuracy Percentage')
    plt.legend()
    plt.show()


def run_question_d():
    k = 1
    n_values = range(100, 5001, 100)  # (100,200,...,5000)
    accuracy = [run_simulation(n, k) for n in n_values]
    plt.plot(n_values, accuracy)
    plt.xlabel('n')
    plt.ylabel('Accuracy Percentage')
    plt.legend()
    plt.show()
