import numpy as np
from matplotlib import pyplot as plt

import src.knn.nearest_neighbor as knn2024
import src.knn.nearest_neighbor_2023b as knn2023


def test_question_b(n: int = 1000, k: int = 10):
    """
    This prints the answer of question (b)
    """
    print('\n*** Started part (b) ***')
    accuracy2023 = knn2023.run_simulation(n=n, k=k)
    accuracy2024 = knn2024.run_simulation(n=n, k=k)
    print(f'2023: Result for n={n}, k={k} is {accuracy2023}')
    print(f'2024:Result for n={n}, k={k} is {accuracy2024}')
    assert accuracy2023 == accuracy2024
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
    # plt.savefig(f'{plot_name}.png', format='png', dpi=300)
    plt.show()


def test_question_c(n: int = 1000):
    """
    This generated the plot requested in question (c)
    I.e., the accuracy of KNN for k values from 1 to 100, with a constant size of input (n).
    """
    print('\n*** Started part (c) ***')
    k_values = range(1, 101)
    accuracy_2023 = np.array([knn2023.run_simulation(n, k) for k in k_values])
    accuracy_2024 = np.array([knn2024.run_simulation(n, k) for k in k_values])
    best_k_index_2023 = np.argmax(accuracy_2023)
    best_k_index_2024 = np.argmax(accuracy_2024)
    print(f'2023: Best accuracy is {accuracy_2023[best_k_index_2023]}, for k = {k_values[best_k_index_2023]}')
    print(f'2024: Best accuracy is {accuracy_2024[best_k_index_2024]}, for k = {k_values[best_k_index_2024]}')
    _plot_results(x_values=k_values,
                  accuracy_values=accuracy_2023,
                  x_label='k',
                  plot_name='Result (c)-2023')
    _plot_results(x_values=k_values,
                  accuracy_values=accuracy_2024,
                  x_label='k',
                  plot_name='Result (c)')
    assert best_k_index_2023 == best_k_index_2024
    assert accuracy_2023 == accuracy_2024
    print('Finished part (c) ***')


def test_question_d(k: int = 1):
    """
    This generated the plot requested in question (d)
    I.e., the accuracy of KNN for a given k and a changing size od training set: (100,200,...,5000)
    """
    print('\n*** Started part (d) ***')
    n_values = np.array(range(100, 5001, 100))
    accuracy_2023 = np.array([knn2023.run_simulation(n, k) for n in n_values])
    accuracy_2024 = np.array([knn2024.run_simulation(n, k) for n in n_values])
    _plot_results(x_values=n_values,
                  accuracy_values=accuracy_2023,
                  x_label='n',
                  plot_name='Result (d)-2023')
    _plot_results(x_values=n_values,
                  accuracy_values=accuracy_2024,
                  x_label='n',
                  plot_name='Result (d)')
    assert accuracy_2023 == accuracy_2024
    print('Finished part (d) ***')


def test_run_simulation(n: int = 1000, k: int = 10):
    """
    Runs the nearest neighbor and calculates accuracy percentage.

    Args:
        n: number of training images. Would take the first n of `train`.
        k: amount of nearest neighbors.

    Returns:
        (float) accuracy percentage

    """
    predicted_labels_2023 = np.array([
        knn2023.k_nearest_neighbors(images=knn2024.TRAIN_IMAGES[:n], labels=knn2024.TRAIN_LABELS[:n],
                                    query_image=image, k=k) for image in knn2024.TEST_IMAGES])
    predicted_labels_2024 = knn2024.knn_predict(n, k)
    diff = len(knn2024.TEST_IMAGES) - np.sum(predicted_labels_2024 == predicted_labels_2023)
    print(diff)
