"""
Statistical Event Detection Patch 1.1

Patch notes:  Initialized the file

Date of last edit: Jan-15th-2018
Rui Nian

Current issues: -
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from copy import deepcopy

import gc

import warnings
warnings.filterwarnings("ignore")


class AnomalyDetection:
    """
    Class for Anomaly Detection.

    Attributes:
                     mean:  The mean of the data
                      std:  The standard deviation of th data
           z_score_labels:  Labels based on z_score threshold

                   median:  Median of the data
                      mad:  Median absolute deviation of each feature.  Robust measurement of deviation.
       mod_z_score_labels:  Labels based on modified z_score threshold

                quartiles:  Includes the 25% and the 75% quartiles
                      iqr:  75% Quartile - 25% Quartile
                   bounds:  25% Quartile - threshold * IQR | 75% Quartile + threshold * IQR
      iqr_outliers_labels:  Any points outside of the bounds


    Methods:

           z_score_method: Univariate method, finds any z-score outside "threshold", and marks as outlier.
                           Not robust to outlier, not a good method.  Theortically cannot detect outlier
                           in data sets under 12 examples.

             mod_z_method: Univariate method.  Uses median and median absolute deviation instead of mean
                           and standard deviation.  Method is more robust to outliers.  Any point outside 3.5
                           median absolute deviation is considered an outlier.

               iqr_method: Univariate method.  Robust to outliers as well. The interquartile range method divides
                           the data into 5 quartiles starting from the 0%, and goes up 25% each.  The box represents
                           the 25% - 75% quartile.  Any data ±1.5 IQR from Q1 or Q3 is considered an outlier.
                           Uses a robust measurement of dispersion to be robust to outliers.


    """

    def __init__(self):
        """
        Inputs
           ---
           data:  Data used to initialize the mean, median, std and data shapes
        """

        # Z-score attributes
        self.mean = None
        self.std = None
        self.z_score_labels = None

        # Modified Z-score attributes
        self.median = None
        self.mad = None
        self.mod_z_score_labels = None

        # IQR Methods
        self.quartiles = None
        self.iqr = None
        self.bounds = None
        self.iqr_outliers = None

    def z_score_method(self, data, columns=None, threshold=3):

        """
        Inputs
           ---

                  data:  Input data
               columns:  Different columns that are to be extracted
             threshold:  Amount of std the data must be outside of to be considered anomalous

        Returns
           ---

         outlier_count: amount of anomalous data in each column
                  mean: Mean of current data
                   std: Standard deviation of current data
               columns: Returns columns that mean and std were calculated for. During live operations, those EXACT
                        columns must be called.
        """

        # Do Z-score on whole data
        if columns is None:

            self.mean = data.mean().transpose()
            self.std = data.std().transpose()

            # Name of columns
            names = list(data)
            for i, name in enumerate(names):
                names[i] = name + "_label"

            self.z_score_labels = pd.DataFrame(np.zeros(data.shape), columns=names)

            for j in range(data.shape[1]):
                if j % 100 == 0:
                    print("Currently on column {}.".format(j))
                self.z_score_labels.iloc[:, j] = [1 if abs((y - self.mean[j]) / self.std[j])
                                                  >= threshold else 0 for y in data.iloc[:, j]]

        # Z-score on selected columns
        else:

            # Name of columns
            names = deepcopy(columns)
            for i, name in enumerate(names):
                names[i] = name + "_label"

            self.z_score_labels = pd.DataFrame(np.zeros((data.shape[0], len(columns))), columns=names)
            
            self.mean = []
            self.std = []
            
            for col in columns:
                mean = data.mean(axis=0)[col]
                std = data.std(axis=0)[col]
                self.z_score_labels.loc[:, col + "_label"] = [1 if abs((y - mean) / std) > threshold
                                                              else 0 for y in data.loc[:, col]]
                
                # Appends mean and std of each column to the total mean / std
                self.mean.append(mean)
                self.std.append(std)

        # Concatenate the labels with the features
        data = pd.concat([self.z_score_labels, data], axis=1)

        # Anomaly count per column
        outlier_count = self.z_score_labels.sum()

        return data, outlier_count, self.mean, self.std, columns, threshold

    def mod_z_method(self, data, columns=None, threshold=3.5):

        """
        Inputs
           ---

                  data:  Input data
               columns:  Different columns that are to be extracted
             threshold:  Amount of modified std the data must be outside of to be considered anomalous

        Returns
           ---

         outlier_count: amount of anomalous data in each column
                median: Median of current data
                   mad: Mean absolute deviation calculated based on current data
               columns: Returns columns that mean and std were calculated for. During live operations, those EXACT
                        columns must be called.
        """

        self.mad = []

        # Do modified Z-score on whole data
        if columns is None:

            self.median = data.median()

            # Make names for each label column
            names = list(data)

            for i, name in enumerate(names):
                names[i] = name + "_label"

            self.mod_z_score_labels = pd.DataFrame(np.zeros(data.shape), columns=names)

            outlier_count = np.zeros(data.shape[1])

            # Calculate all the mean absolute deviations
            for j in range(self.mod_z_score_labels.shape[1]):
                median_absolute_deviation = np.median([np.abs(y - self.median[j]) for y in data.iloc[:, j]])
                self.mad.append(median_absolute_deviation)

            for j in range(data.shape[1]):
                if j % 100 == 0:
                    print("Currently on column {} for modified Z calculations.".format(j))
                self.mod_z_score_labels.iloc[:, j] = [1 if abs(0.6745 * (y - self.median[j]) / self.mad[j]) >
                                                      threshold else 0 for y in data.iloc[:, j]]

                outlier_count = self.mod_z_score_labels.sum()

        else:

            self.median = []

            # Make names for each label column
            names = deepcopy(columns)

            for i, name in enumerate(names):
                names[i] = name + "_label"

            self.mod_z_score_labels = pd.DataFrame(np.zeros((data.shape[0], len(columns))), columns=names)

            for col in columns:
                # Find the median of that column
                median = data.median(axis=0)[col]
                median_absolute_deviation = np.median([np.abs(y - median) for y in data.loc[:, col]])

                self.mad.append(median_absolute_deviation)
                self.median.append(median)

            for i, col in enumerate(columns):
                median = data.median(axis=0)[col]
                self.mod_z_score_labels.loc[:, col + "_label"] = [1 if abs(0.6745 * (y - median) / self.mad[i]) >
                                                                  threshold else 0 for y in data.loc[:, col]]
            outlier_count = self.mod_z_score_labels.sum()

            data = pd.concat([self.mod_z_score_labels, data], axis=1)

        return data, outlier_count, self.median, self.mad, columns, threshold

    def iqr_method(self, data, columns=None, threshold=1.5):

        """
        Inputs
           ---

                  data:  Input data
               columns:  Different columns that are to be extracted
             threshold:  X * IQR to be considered anomalous

        Returns
           ---

         outlier_count: amount of anomalous data in each column
                bounds: Upper and lower bounds of outliers
               columns: Returns columns that mean and std were calculated for. During live operations, those EXACT
                        columns must be called.
        """

        # If columns is not given
        if columns is None:
            self.iqr = np.zeros(data.shape[1])
            self.quartiles = np.zeros((2, data.shape[1]))
            self.bounds = np.zeros((2, data.shape[1]))

            names = list(data)
            for i, name in enumerate(names):
                names[i] = name + "_label"

            self.iqr_outliers = pd.DataFrame(np.zeros(data.shape), columns=names)

            # Find the 25th and 75th quartiles for each column
            for j in range(data.shape[1]):
                self.quartiles[:, j] = np.percentile(data.iloc[:, j], [25, 75])

                # IQR calculations
                self.iqr[j] = self.quartiles[1, j] - self.quartiles[0, j]

                # Upper and Lower bounds calculations
                self.bounds[0, j] = self.quartiles[0, j] - threshold * self.iqr[j]
                self.bounds[1, j] = self.quartiles[1, j] + threshold * self.iqr[j]

            # Outliers calculation
            for j in range(data.shape[1]):
                if j % 100 == 0:
                    print("On {}th column.".format(j))
                self.iqr_outliers.iloc[:, j] = [1 if (y < self.bounds[0, j] or y > self.bounds[1, j]) else 0
                                                for y in data.iloc[:, j]]

        # If a unique set of columns is given
        else:
            self.iqr = np.zeros(len(columns))
            self.quartiles = np.zeros((2, len(columns)))
            self.bounds = np.zeros((2, len(columns)))

            names = deepcopy(columns)
            for i, name in enumerate(names):
                names[i] = name + "_label"

            self.iqr_outliers = pd.DataFrame(np.zeros((data.shape[0], len(columns))), columns=names)

            # Find the 25th and 75th quartiles for each column
            for i, col in enumerate(columns):
                self.quartiles[:, i] = np.percentile(data.loc[:, col], [25, 75])

                # IQR calculations
                self.iqr[i] = self.quartiles[1, i] - self.quartiles[0, i]

                # Upper and Lower bounds calculations
                self.bounds[0, i] = self.quartiles[0, i] - threshold * self.iqr[i]
                self.bounds[1, i] = self.quartiles[1, i] + threshold * self.iqr[i]

            # Outlier calculation
            for i, col in enumerate(columns):
                self.iqr_outliers.loc[:, col + "_label"] = [1 if (y < self.bounds[0, i] or y > self.bounds[1, i]) else 0
                                                            for y in data.loc[:, col]]

        # Calculate sum of outliers for each feature
        outlier_count = self.iqr_outliers.sum()

        data = pd.concat([self.iqr_outliers, data], axis=1)

        return data, outlier_count, self.bounds, columns

    
class LiveAnomalyDetection:
    
    def __init__(self):
        """
        Attributes:
                         mean:  The mean of the data
                          std:  The standard deviation of th data
               z_score_labels:  Labels based on z_score threshold

                       median:  Median of the data
                          mad:  Median absolute deviation of each feature.  Robust measurement of deviation.
           mod_z_score_labels:  Labels based on modified z_score threshold

                    quartiles:  Includes the 25% and the 75% quartiles
                       bounds:  25% Quartile - threshold * IQR | 75% Quartile + threshold * IQR
        """

        # Z-score attributes
        self.mean = None
        self.std = None
        self.z_score_labels = None

        # Modified Z-score attributes
        self.median = None
        self.mad = None
        self.mod_z_score_labels = None

        # IQR Methods
        self.bounds = None
        self.iqr_outliers = None

    def live_z_score(self, data, mean, std, threshold, columns=None):
        """
        Inputs
            ---

                  data: Input data
                  mean: Mean of data from training data set
                   std: Standard deviation of data from training data set
             threshold: Threshold of data from training data set
               columns: Columns to be labeled from training data set

        Returns
            ---
                  data: Labeled data
            z_outliers:
        """
        self.mean = mean
        self.std = std

        # If no unique columns was passed, 
        if columns is None:
            # Ensure shape compatibility
            # assert(data.shape[1] == len(mean))
            # assert(data.shape[1] == len(std))

            # Name of columns
            names = list(data)
            for i, name in enumerate(names):
                names[i] = name + "_label"

            self.z_score_labels = pd.DataFrame(np.zeros(data.shape), columns=names)

            for j in range(data.shape[1]):
                if j % 100 == 0:
                    print("Currently on column {}.".format(j))
                self.z_score_labels.iloc[:, j] = [1 if abs((y - self.mean[j]) / self.std[j])
                                                  >= threshold else 0 for y in data.iloc[:, j]]
        
        # If unique columns was given        
        else:
            # Ensure shape compatibility
            # assert(len(columns) == len(mean))
            # assert(len(columns) == len(std))

            # Name of columns
            names = deepcopy(columns)
            for i, name in enumerate(names):
                names[i] = name + "_label"

            self.z_score_labels = pd.DataFrame(np.zeros((data.shape[0], len(columns))), columns=names)

            for i, col in enumerate(columns):
                self.z_score_labels.loc[:, col + "_label"] = [1 if abs((y - mean[i]) / std[i]) > threshold
                                                              else 0 for y in data.loc[:, col]]

        z_outliers = self.z_score_labels.sum()

        # Concatenate the labels with the features
        data = pd.concat([self.z_score_labels, data], axis=1)

        return data, z_outliers
    
    def live_mod_z(self, data, median, mad, threshold, columns=None):
        """
        Inputs
            ---
           
                  data: Input data
                median: Median of data from training data set
                   mad: Median absolute deviation of the training data set
             threshold: Threshold of data from training data set
               columns: Columns to be labeled from training data set
               
        Returns
            ---
                  data: Labeled data
         modz_outliers:
        """

        self.median = median
        self.mad = mad

        # Do modified Z-score on whole data
        if columns is None:
            # Ensure shape compatibility
            # assert(data.shape[1] == len(median))
            # assert(data.shape[1] == len(mad))

            # Make names for each label column
            names = list(data)

            for i, name in enumerate(names):
                names[i] = name + "_label"

            self.mod_z_score_labels = pd.DataFrame(np.zeros(data.shape), columns=names)

            for j in range(data.shape[1]):
                if j % 100 == 0:
                    print("Currently on column {} for modified Z calculations.".format(j))
                self.mod_z_score_labels.iloc[:, j] = [1 if abs(0.6745 * (y - self.median[j]) / self.mad[j]) >
                                                      threshold else 0 for y in data.iloc[:, j]]

        else:
            # Ensure shape compatibility
            # assert(len(columns) == len(median))
            # assert(len(columns) == len(mad))

            # Make names for each label column
            names = deepcopy(columns)

            for i, name in enumerate(names):
                names[i] = name + "_label"

            self.mod_z_score_labels = pd.DataFrame(np.zeros((data.shape[0], len(columns))), columns=names)

            for i, col in enumerate(columns):
                self.mod_z_score_labels.loc[:, col + "_label"] = [1 if abs(0.6745 * (y - self.median[i]) / self.mad[i])
                                                                  > threshold else 0 for y in data.loc[:, col]]

        modz_outliers = self.mod_z_score_labels.sum()

        data = pd.concat([self.mod_z_score_labels, data], axis=1)
        
        return data, modz_outliers
    
    def live_iqr(self, data, bounds, columns=None):
        """
        Inputs
            ---
           
                  data: Input data
                bounds: The upper and lower bounds of each column to be labeled
               columns: Columns to be labeled from training data set
               
        Returns
            ---
                  data: Labeled data
          iqr_outliers:
        """

        self.bounds = bounds

        if columns is None:
            # Ensure shapes are compatible
            # assert(data.shape[1] == len(bounds))

            names = list(data)
            for i, name in enumerate(names):
                names[i] = name + "_label"

            self.iqr_outliers = pd.DataFrame(np.zeros(data.shape), columns=names)

            # Outliers calculation
            for j in range(data.shape[1]):
                if j % 100 == 0:
                    print("On {}th column.".format(j))
                self.iqr_outliers.iloc[:, j] = [1 if (y < self.bounds[0, j] or y > self.bounds[1, j]) else 0
                                                for y in data.iloc[:, j]]

        # If a unique set of columns is given
        else:
            # Ensure shapes are compatible
            # assert(data.shape[1] == len(bounds))

            names = deepcopy(columns)
            for i, name in enumerate(names):
                names[i] = name + "_label"

            self.iqr_outliers = pd.DataFrame(np.zeros((data.shape[0], len(columns))), columns=names)

            # Outlier calculation
            for i, col in enumerate(columns):
                self.iqr_outliers.loc[:, col + "_label"] = [1 if (y < self.bounds[0, i] or y > self.bounds[1, i]) else 0
                                                            for y in data.loc[:, col]]

        iqr_outliers = self.iqr_outliers.sum()

        data = pd.concat([self.iqr_outliers, data], axis=1)
        
        return data, iqr_outliers


if __name__ == "__main__":

    # Load filtered data
    Data1 = pd.read_csv('test_datasets/CoffeeBeanData.csv')
    print('The original data has {} training examples'.format(Data1.shape[0]))

    # Build the stat_analysis object
    stat_analysis = AnomalyDetection()

    # data_z is the new output file [Labels | Features] using Z score, anomaly_count_z is the amount of anomalous data
    data_z, count_z, Mean, std_dev, cols_z, thres_z = stat_analysis.z_score_method(Data1, columns=['175642862_630',
                                                                                                   '175642865_630'])

    # data_modz is the new file [Labels | Features] using mod Z score, anomaly_count_modz is the amount of
    # anomalous data
    data_modz, count_modz, Median, MAD, cols_modz, thres_modz = stat_analysis.mod_z_method(Data1,
                                                                                           columns=['175642862_630',
                                                                                                    '175642865_630'])

    # data_iqr is the new output file [Labels | Features] using IQR, anomaly_count_iqr is the amount of anomalous data
    data_iqr, count_iqr, Bounds, cols_iqr = stat_analysis.iqr_method(Data1, columns=['175642862_630', '175642865_630'])

    """
    Online evaluation
    """

    Data2 = pd.read_csv('test_datasets/CoffeeBeanDatav2.csv')

    # Live anomaly detection
    live_stat_analysis = LiveAnomalyDetection()

    # Live z-score labelling
    data_z2, count_z2 = live_stat_analysis.live_z_score(Data2, Mean, std_dev, thres_z, cols_z)

    # Live mod z-score labelling
    data_modz2, count_modz2 = live_stat_analysis.live_mod_z(Data2, Median, MAD, thres_modz, cols_modz)

    # Live IQR labelling
    data_iqr2, count_iqr2 = live_stat_analysis.live_iqr(Data2, Bounds, cols_iqr)
