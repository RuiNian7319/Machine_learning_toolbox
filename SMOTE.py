"""
SMOTE: Synthetic Minority Oversampling Technique

Used to generate data non-adaptively

Developed by Rui Nian
Date of last edit: December 25th, 2018

Patch Notes: -

Known Issues: -
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors


def smote(X, y, N=100, K=5):

    """
    Synthetic minority oversampling technique.

    Generates synthetic examples in a less application-specific manner by operating in the "feature space" rather than
    "data space" by generating data from interpolating between minority examples.

    Inputs
         -----
           X:  Input features, X, sorted by the minority examples on top.  Minority example should also be labeled as 1
           y:  Labels, with minority example labeled as 1
           N:  Amount of SMOTE N%
           K:  Amount of neighbours to look at

    Variables
         -----
          ms:  # of minority example
          ml:  # of majority example
     min_cls:  All features that are in minority class
   s_min_cls:  Synthetic examples
          xi:  Minority example
neighborhood:  All minority example indices per neighbourhood
      syn_ex:  Total amount of synthetic examples to be generated
          si:  Synthetic example generated
    syn_data:  List of all synthetic generated examples

    Returns
         -----
    syn_data:  New synthetic minority data created
    """

    seed = 1
    np.random.seed(seed)

    # Step 1.  Define minority and majority class examples, and minority class features
    ms = int(sum(y))
    min_cls = X[0:ms, :]

    # Step 2.  If N is less than 100, then only a random percent will be smoted.
    if N < 100:
        np.random.shuffle(min_cls)
        ms = int((N / 100) * ms)
        N = 100

    syn_ex = int(N / 100) * ms

    # Step 3.  Compute the k-NN for each minority class
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X, y)

    neighborhoods = []
    for i in range(ms):
        xi = X[i, :].reshape(1, -1)
        neighbours = clf.kneighbors(xi, n_neighbors=K, return_distance=False)[0]
        # Skip itself in the neighborhood
        neighbours = neighbours[1:]

        # Find all the minority examples
        neighborhood = []
        for index in neighbours:
            if index <= ms - 1:
                neighborhood.append(index)

        neighborhoods.append(neighborhood)

    # Step 4.  Determine the amount of SMOTE examples to develop per neighbourhood.

    num_ex = int(syn_ex / len(neighborhoods))

    # Step 5.  Generate SMOTE examples
    syn_data = []
    for i in range(ms):
        xi = X[i, :].reshape(1, -1)
        for j in range(num_ex):
            # if the neighbourhood is not empty
            if neighborhoods[i]:
                index = np.random.choice(neighborhoods[i])
                xzi = X[index, :].reshape(1, -1)
                si = xi + (xzi - xi) * np.random.uniform(0, 1)
                syn_data.append(si)

    # Build the data matrix
    data = []
    for values in syn_data:
        data.append(values[0])

    print("{} amount of minority class samples generated".format(len(data)))

    # Step 6.  Re-build the data set with synthetic data added

    # Concatenate the positive labels with the newly made data
    labels = np.ones([len(data), 1])
    data = np.concatenate([labels, data], axis=1)

    # Concatenate with old data
    org_data = np.concatenate([y.reshape(-1, 1), X], axis=1)
    data = np.concatenate([data, org_data])

    # Test the new generated data
    test = []
    for values in syn_data:
        a = clf.predict(values)
        test.append(a)

    print("Using the old classifier, {} out of {} would be classified as minority.".format(np.sum(test), len(syn_data)))

    return data, neighborhoods


if __name__ == "__main__":
    df = pd.read_csv("test_datasets/breast-cancer-wisconsin.data.txt")
    # Remove missing values
    df.replace('?', -99999, inplace=True)
    # Remove ID column
    df.drop(['id'], axis=1, inplace=True)
    df.iloc[:, 5] = pd.to_numeric(df.iloc[:, 5])
    # Replace labels booleans column with 0 and 1
    df.loc[:, 'class'].replace(2, 0, inplace=True)
    df.loc[:, 'class'].replace(4, 1, inplace=True)
    # Sort labels, with minority class on top
    df.sort_values(['class'], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    X = df.iloc[:, 0:-1].values
    y = df.iloc[:, -1].values

    Syn_data, _ = smote(X, y, N=50, K=10)
