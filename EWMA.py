"""
Exponentially Weighted Moving Average (EWMA) function

By: Rui Nian
"""

import numpy as np
import matplotlib.pyplot as plt


def ewma(beta, vector):
    """
    Description
       ---
          Exponential smoothing
          Note: Exponentially smoothed vector has 0 for the first value.  During implementation, omit first value.

    Inputs
       ---
          beta: Smoothing factor.  Higher = more smoothing
        vector: Vector of numbers to be smoothed

    Returns
       ---
      ewma_num: Exponentially smoothed vector

    """
    ewma_num = np.zeros(len(vector))

    for j in range(len(ewma_num)):

        if j == (len(ewma_num) - 1):
            break

        ewma_num[j + 1] = beta * ewma_num[j] + (1 - beta) * vector[j + 1]

        # Bias correction
        if j == 0:
            ewma_num[j + 1] = ewma_num[j + 1] / (1 - np.power(beta, j + 1))

    return ewma_num


if __name__ == "__main__":

    np.random.seed(1)

    numbers = np.zeros(500)

    for i, number in enumerate(numbers):
        numbers[i] = i + np.random.uniform(-i / 5, i / 5)

    ewma_values = ewma(0.9, numbers)

    plt.plot(numbers)
    plt.plot(ewma_values)

    plt.show()
