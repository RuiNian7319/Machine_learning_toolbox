"""
Part 1 of Feature Importance: Permutation Importance.

By: Rui Nian

Date: January 10th, 2019

Idea:  Shuffle one column while keeping others the same, for each column in columns.
          - If accuracy drops tremendously, that feature was useful
          - If accuracy does not drop, feature was not useful

Output: 0.0785 +/- 0.03.  First number represents accuracy/loss decrease after shuffle.
                          Second number represents confidence interval: mean(x) +/- Z * sigma / sqrt(n)
                            - Both values are an average after x amount of shuffles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gc
gc.enable()


def perm_importance(labels, features, shuffle_num=5):
    """
    Inputs
      ----
           labels: Vector of labels
         features: Matrix of features
      shuffle_num: Amount of shuffles to generate the accuracy / confidence intervals


    Returns
      ----
       importance: Importance of each feature
    """




    return importance


if __name__ == "__main__":

    data = pd.read_csv('test_datasets/AirQualityUCI.csv')
