from collections import Counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray, dtype
from sklearn.datasets import fetch_openml
from sklearn.metrics import euclidean_distances

_MNIST = fetch_openml('mnist_784', as_frame=False)
DATA = _MNIST['data']  # images of 28 X 28 (matrix)
LABELS = _MNIST['target']  # for each image it has a label - the number in the picture

RAND_INDEXES = np.random.RandomState(0).choice(70000, 11000)  # 11,000 numbers from 0 to 70,000

# split data into training data and labels and test data and labels
TRAIN_SIZE = 10000

TRAIN_IMAGES = DATA[RAND_INDEXES[:TRAIN_SIZE], :].astype(int)
TRAIN_LABELS = LABELS[RAND_INDEXES[:TRAIN_SIZE]]

TEST_IMAGES = DATA[RAND_INDEXES[TRAIN_SIZE:], :].astype(int)
TEST_LABELS = LABELS[RAND_INDEXES[TRAIN_SIZE:]]


def knn_predict(n: int = TRAIN_SIZE,
                k: int = 1) -> ndarray[Any, dtype[Any]]:
    """
    Predict a label for each query image using the KNN algorithm.

    Args:
        n: Number of training images to use (subset size of the training set).
        k: Number of neighbors to consider.

    Returns:
        The majority label of the k nearest neighbors for each test image.
    """
    train_images = TRAIN_IMAGES[:n]
    train_labels = TRAIN_LABELS[:n]

    distances = euclidean_distances(TEST_IMAGES, train_images)

    k_nearest_indexes = np.argsort(distances, axis=1)[:, :k]
    k_labels = train_labels[k_nearest_indexes]
    mode_labels = np.array([Counter(row).most_common(1)[0][0] for row in k_labels])
    return mode_labels


def run_simulation(n: int, k: int) -> float:
    """
    Runs the nearest neighbor and calculates accuracy percentage.

    Args:
        n: number of training images. Would take the first n of `train`.
        k: amount of nearest neighbors.

    Returns:
        (float) accuracy percentage

    """
    predicted_labels = knn_predict(n, k)
    accuracy = 100 * np.mean(predicted_labels == TEST_LABELS)
    print(f'n={n}, k={k}, accuracy={accuracy}')
    return accuracy


def run_question_b(n: int = 1000, k: int = 10):
    """
    This prints the answer of question (b)
    """
    print('\n*** Started part (b) ***')
    accuracy = run_simulation(n=n, k=k)
    print(f'Result for n={n}, k={k} is {accuracy}')
    print('Finished part (b) ***')


def _plot_results(x_values: np.array, accuracy_values: np.array, x_label: str,
                  plot_name: str):
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, accuracy_values, label='accuracy')
    plt.xlabel(x_label, fontweight='bold')
    plt.ylabel('Accuracy Percentage', fontweight='bold')
    plt.title(plot_name, fontsize=16, color='brown', fontweight='bold',
              fontstyle='italic')
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_question_c(n: int = 1000):
    """
    This generated the plot requested in question (c)
    I.e., the accuracy of KNN for k values from 1 to 100, with a constant size of input (n).
    """
    print('\n*** Started part (c) ***')
    k_values = np.array(range(1, 101))
    accuracy = np.array([run_simulation(n, k) for k in k_values])
    best_k_index = np.argmax(accuracy)
    print(f'Best accuracy is {accuracy[best_k_index]}, for k = {k_values[best_k_index]}')
    _plot_results(x_values=k_values,
                  accuracy_values=accuracy,
                  x_label='k',
                  plot_name='Result (c)')
    print('Finished part (c) ***')


def run_question_d(k: int = 1):
    """
    This generated the plot requested in question (d)
    I.e., the accuracy of KNN for a given k and a changing size od training set: (100,200,...,5000)
    """
    print('\n*** Started part (d) ***')
    n_values = np.array(range(100, 5001, 100))
    accuracy = np.array([run_simulation(n, k) for n in n_values])
    _plot_results(x_values=n_values,
                  accuracy_values=accuracy,
                  x_label='n',
                  plot_name='Result (d)')
    print('Finished part (d) ***')


if __name__ == '__main__':
    print('Starting...')
    run_question_b()
    run_question_c()
    run_question_d()
    print('Done.')
