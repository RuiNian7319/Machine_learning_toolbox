"""
SMOTE: Synthetic Minority Oversampling Technique

Used to generate data non-adaptively

Developed by Rui Nian
Date of last edit: December 23th, 2018

Patch Notes: Updated Doc Strings

Known Issues: -
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors

seed = 10
np.random.seed(seed)


def adasyn():

    """
    Adaptively generating minority data samples according to their distributions.
    More synthetic data is generated for minority class samples that are harder to learn.
    Harder to learn data is defined as positive examples with not many examples for in their respective neighbourhood.

    Inputs
         -----
         X:  Input features, X, sorted by the minority examples on top.  Minority example should also be labeled as 1
         y:  Labels, with minority example labeled as 1
      beta:  Degree of imbalance desired.  A 1 means the positive and negative examples are perfectly balanced.
         K:  Amount of neighbours to look at
 threshold:  Amount of imbalance rebalance required for algorithm

    Variables
         -----
         xi:  Minority example
        xzi:  A minority example inside the neighbourhood of xi
         ms:  Amount of data in minority class
         ml:  Amount of data in majority class
        clf:  k-NN classifier model
          d:  Ratio of minority : majority
       beta:  Degree of imbalance desired
          G:  Amount of data to generate
         Ri:  Ratio of majority data / neighbourhood size.  Larger ratio means the neighbourhood is harder to learn,
              thus generating more data.
     Minority_per_xi:  All the minority data's index by neighbourhood
     Rhat_i:  Normalized Ri, where sum = 1
         Gi:  Amount of data to generate per neighbourhood (indexed by neighbourhoods corresponding to xi)

    Returns
         -----
  syn_data:  New synthetic minority data created
    """

    pass


