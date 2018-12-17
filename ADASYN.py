import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors

seed = 5
np.random.seed(seed)


class MinMaxNormalization:

    """
    axis:  0 is by columns, 1 is by rows
    """

    def __init__(self, data, axis=0):
        self.row_min = np.min(data, axis=axis)
        self.row_max = np.max(data, axis=axis)
        self.denominator = abs(self.row_max - self.row_min)

        # Fix divide by zero, replace value with 1 because these usually happen for boolean columns
        for index, value in enumerate(self.denominator):
            if value == 0:
                self.denominator[index] = 1

    def __call__(self, data):
        return np.divide((data - self.row_min), self.denominator)


def adasyn(X, y, beta, K, threshold=1):
    
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

    ms = sum(y)
    ml = len(y) - ms

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X, y)

    # Step 1, calculate the degree of class imbalance.  If degree of class imbalance is violated, continue.
    d = np.divide(ms, ml)

    if d > threshold:
        return print("The data set is not imbalanced enough.")

    # Step 2a, if the minority data set is below the maximum tolerated threshold, generate data.
    # Beta is the desired balance level parameter.  Beta > 1 means u want more of the imbalanced type, vice versa.
    G = (ml - ms) * beta

    # Step 2b, find the K nearest neighbours of each minority class example in euclidean distance.
    # Find the ratio ri = majority_class in neighbourhood / K
    Ri = []
    Minority_per_xi = []
    for i in range(ms):
        xi = X[i, :].reshape(1, -1)
        # Returns indices of the closest neighbours, and return it as a list
        neighbours = clf.kneighbors(xi, n_neighbors=K, return_distance=False)[0]
        # Skip classifying itself as one of its own neighbours
        neighbours = neighbours[1:]

        # Count how many belongs to the majority class
        count = 0
        for value in neighbours:
            if value > ms:
                count += 1

        Ri.append(count / K)

        # Find all the minority examples
        minority = []
        for value in neighbours:
            if value <= ms:
                minority.append(value)

        Minority_per_xi.append(minority)

    # Step 2c, normalize ri's so their sum equals to 1
    Rhat_i = []
    for ri in Ri:
        rhat_i = ri / sum(Ri)
        Rhat_i.append(rhat_i)

    assert (sum(Rhat_i) > 0.99)

    # Step 2d, calculate the number of synthetic data examples that will be generated for each minority example
    Gi = []
    for rhat_i in Rhat_i:
        gi = round(rhat_i * G)
        Gi.append(int(gi))

    # # Step 2e, generate synthetic examples
    syn_data = []
    for i in range(ms):
        xi = X[i, :].reshape(1, -1)
        for j in range(Gi[i]):
            if Minority_per_xi[i]:
                index = np.random.choice(Minority_per_xi[i])
                xzi = X[index, :].reshape(1, -1)
                si = xi + (xzi - xi) * np.random.uniform(0, 1)
                syn_data.append(si)

    # Test the new generated data
    test = []
    for values in syn_data:
        a = clf.predict(values)
        test.append(a)

    print("Using the old classifier, {} out of {} would be classified as minority.".format(sum(test), G))

    return syn_data


df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.drop(['id'], axis=1, inplace=True)
df.replace('?', -99999, inplace=True)
df['class'].replace([2, 4], [0, 1], inplace=True)
df.sort_values(by=['class'], ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)

X = df.drop(['class'], axis=1).values
X = X.astype('int')
y = df['class'].values

Syn_data = adasyn(X, y, beta=1, K=10, threshold=1)
