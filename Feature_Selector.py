"""
Feature Selector Class for Willowglen Project

Rui Nian

Last Updated: Jan-14-2018
"""


import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from copy import deepcopy
from Data_loader import data_loader

import random

import gc

from itertools import chain
import os


class FeatureSelector:
    """
    Attributes:
    ------
            features_removed:  A list of all the removed features
                 removal_ops:  Dictionary of removal operations ran and associated features for removal identified by
                               that removal method

              record_missing:  Records fraction of missing values for features with missing fraction
                               above threshold
           missing_threshold:  Threshold for missing values for column to be considered "removal quality"
               missing_stats:  Index are all the features, first column is all the % missing stats

        record_single_unique:  Records features with single unique value
                unique_stats:  Data frame of each feature and its corresponding number of unique values

            record_collinear:  Records pairs of features with collinear features about threshold
       correlation_threshold:  Correlation required between 2 features to be identified as highly correlated
                 corr_matrix:  Correlation matrix between all features within the data set

          record_near_unique:  Data frame of all features that have a very imbalanced boolean distribution


    Methods:
    -------

             identify_mssing:  Identifies all columns with missing columns above threshold
                plot_missing:  Plots histograms of # of features vs. % missing value
      identify_single_unique:  Identifies all columns with a single unique value
          plot_single_unique:  Plots histogram of # of features vs. # of unique values in that column
          identify_collinear:  Identifies all correlations between all features within the data set
              plot_collinear:  Plots a correlation heat map between all features
          remove_near_unique:  Identifies all boolean data types whose data is highly imbalanced
            plot_near_unique:  Plots # of features vs the %  of data in the positive class
            identify_feature:  Removes any SINGLE features selected by operator
               readd_feature:  Re-adds any SINGLE features selected by operator
            check_identified:  Returns all features that are currently marked for deletion
                     removal:  Removes the features marked for deletion
                unique_value:  Static Method.  Used to find a unique set of the original data
               remove_idtype:  Method only applicable to Suncor/Willowglen.  Removes all features with a certain IDType
    """

    def __init__(self):

        self.features_removed = []

        # Dictionary to hold removal operations
        self.removal_ops = {'custom': []}

        # Missing threshold attributes
        self.record_missing = None
        self.missing_threshold = None
        self.missing_stats = None

        # Unique values attributes
        self.record_single_unique = None
        self.unique_stats = None

        # Collinear attributes
        self.record_collinear = None
        self.correlation_threshold = None
        self.corr_matrix = None

        # Custom ID removal attributes
        self.record_id_removal = None

        # Near unique attributes
        self.record_near_unique = None

    def __str__(self):
        return "Select featured based on missing values, single unique values, and collinearity"

    def __repr__(self):
        return "FeatureSelector()"

    def identify_missing(self, data, missing_threshold, json_file=False):
        """
        Method that removes columns with missing value percentage above a threshold defined by: missing_threshold
        """

        self.missing_threshold = missing_threshold

        # Calculates fraction of missing in each column
        missing_series = data.isnull().sum() / data.shape[0]

        # Make the missing_series into a data frame
        self.missing_stats = pd.DataFrame(missing_series).rename(columns={'index': 'feature', 0: 'missing_fraction'})

        # Find columns with missing stats above threshold
        record_missing = pd.DataFrame(missing_series[missing_series > missing_threshold]).reset_index().rename(
            columns={'index': 'feature', 0: 'missing_fraction'})

        # Make a list of all the headers in the record_missing data frame
        drop = list(record_missing['feature'])

        self.record_missing = record_missing
        self.removal_ops['missing'] = drop

        print('{} features with greater than {}% of missing values. \n'.format(len(self.removal_ops['missing']),
                                                                               self.missing_threshold * 100))

        if json_file:
            bin_list = np.linspace(-0.001, 1, 21)
            histogram_json = self.missing_stats.groupby(pd.cut(self.missing_stats.iloc[:, 0], bin_list)).count()
            print(histogram_json)
            histogram_json.to_json('JSON/missing_values_plot.json', orient='split')

    def plot_missing(self):

        """
        Method for visualization of missing values
        """

        if self.missing_stats is None:
            raise NotImplementedError("No missing data, or the identify missing method was not ran")

        plt.figure()
        self.missing_stats.plot.hist(color='red', edgecolor='k', figsize=(6, 4), fontsize=14)

        plt.ylabel('# of Features', size=18)
        plt.xlabel('% of Missing Data', size=18)

        plt.xlim([0, 1])

        plt.title('Missing Data Histogram', size=18)

        plt.show()

    def identify_single_unique(self, data, json_file=False):

        """
        Method to identify columns with single unique values, meaning they have no predictive power
        """

        # Calculates the unique counts in each column
        unique_counts = data.nunique()

        self.unique_stats = pd.DataFrame(unique_counts).rename(columns={'index': 'features', 0: 'nunique'})

        # Find columns with only one value
        record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(columns={
            'index': 'features',
            0: 'Number Unique'
        })

        # Find columns with two unique values
        record_boolean_values = pd.DataFrame(unique_counts[unique_counts == 2]).reset_index().rename(columns={
            'index': 'features',
            0: 'Number Boolean'
        })

        drop = list(record_single_unique['features'])

        self.record_single_unique = record_single_unique
        self.removal_ops['single_unique'] = drop

        print('{} features has only one unique value. \n'.format(len(self.removal_ops['single_unique'])))
        print('{} features are boolean values. \n'.format(len(record_boolean_values)))

        if json_file:
            # Take the log space
            bin_list = np.logspace(0, np.log10(self.unique_stats.iloc[:, 0].max()), 15)
            histogram_json = self.unique_stats.groupby(pd.cut(self.unique_stats.iloc[:, 0], bin_list)).count()
            histogram_json.to_json('JSON/unique_value_plot.json', orient='split')
            print(histogram_json)

    def plot_single_unique(self):

        """
        Histogram of # of unique values in each column
        """

        if self.unique_stats is None:
            raise NotImplementedError("No features are unique, or single unique method was not ran.")

        plt.figure()
        self.unique_stats.plot.hist(color='blue', edgecolor='k', figsize=(6, 4), fontsize=14,
                                    bins=np.logspace(0, np.log10(self.unique_stats.iloc[:, 0].max()), 15))

        plt.ylabel('# of Features')
        plt.xlabel('Unique Values')

        plt.gca().set_xscale('log')

        plt.title("Unique Value Histogram")

        plt.show()

    def identify_collinear(self, data, correlation_threshold, json_file=False):

        """
        Identify highly collinear features.  Collinear features highly reduce predictive powers.

        Reference: https://www.quora.com/Why-is-multicollinearity-bad-in-laymans-terms-In-feature-selection-for-a-
        regression-model-intended-for-use-in-prediction-why-is-it-a-bad-thing-to-have-multicollinearity-or-highly-
        correlated-independent-variables

        Identifies any columns of data with highly correlated data, correlation threshold is determined by using
        using the correlation_threshold.
        """

        self.correlation_threshold = correlation_threshold

        # Calculates the correlations between each column. https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
        corr_matrix = data.corr()

        self.corr_matrix = corr_matrix

        # Extract upper triangular of the correlation matrix, since the matrix is symmetrical.  Does not take diagonal
        # since diagonal is all 1
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Select the features with correlations above the threshold (absolute value is used in case of neg corr)
        drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

        # Remember correlated pairs
        record_collinear = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])

        # Iterate through columns to identify which to drop
        for column in drop:
            # Find correlated features
            corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

            # Find the correlated values
            corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]

            # Record the information
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                              'corr_feature': corr_features,
                                              'corr_value': corr_values})

            # Add to DataFrame
            record_collinear = record_collinear.append(temp_df, ignore_index=True, sort=True)

        self.record_collinear = record_collinear
        self.removal_ops['collinear'] = drop

        print("{} features with a correlation greater than {}. \n".format(len(self.removal_ops['collinear']),
                                                                          correlation_threshold * 100))
        if json_file:
            collinear_json = self.corr_matrix.loc[list(self.record_collinear['corr_feature']),
                                                  list(self.record_collinear['drop_feature'])]
            collinear_json.to_json("JSON/collinear_plot.json", orient='split')

    def plot_collinear(self):
        """
        Heatmap of the features with correlations above the correlated threshold in the data.

        Notes
        --------
            - Not all of the plotted correlations are above the threshold because this plots
            all the variables that have been identified as having even one correlation above the threshold
            - The features on the x-axis are those that will be removed. The features on the y-axis
            are the correlated feature with those on the x-axis

        """

        if self.record_collinear is None:
            raise NotImplementedError('Collinear features have not been idenfitied. Run `identify_collinear`.')

        # Identify the correlations that were above the threshold
        corr_matrix_plot = self.corr_matrix.loc[list(self.record_collinear['corr_feature']),
                                                list(self.record_collinear['drop_feature'])]

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(10, 8))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr_matrix_plot, cmap=cmap, center=0,
                    linewidths=.25, cbar_kws={"shrink": 0.6})

        ax.set_yticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[0]))])
        ax.set_yticklabels(list(corr_matrix_plot.index), size=int(160 / corr_matrix_plot.shape[0]))

        ax.set_xticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[1]))])
        ax.set_xticklabels(list(corr_matrix_plot.columns), size=int(160 / corr_matrix_plot.shape[1]))

        plt.xlabel('Features to Remove', size=8)
        plt.ylabel('Correlated Feature', size=8)
        plt.title("Correlations Above Threshold", size=14)

        plt.show()

    def remove_near_unique(self, data, threshold, json_file=False):

        """
        Removes binary, trinary, etc. columns that are mostly one value

        threshold: % of total being equal to 1.  For a column with 100 examples, at a threshold of 0.03 (3%), any columns
        with less than 3% of values being 1 or greater than 97% of values being 1 will be set for deletion.

        DOES NOT FILTER OUT MULTI-CLASS CATEGORIES AT THE MOMENT
        """

        drop = []
        positives = []

        # Iterate over the different columns
        for i in range(data.shape[1]):
            # Pass any columns that are not boolean.  Greater than 3 means it filters floats that are higher than 3.
            # Less than 0.99 means it filters any floats that possibly have very low values or negative values.
            if abs(data.iloc[:, i].max()) > 1 or abs(data.iloc[:, i].min()) > 1 or abs(data.iloc[:, i].max()) < 0.99:
                pass
            else:
                # Sum up that column
                total_count = sum(data.iloc[:, i])
                # If the total of a class is less than 2% or higher than 98%, append column
                if total_count < (threshold * data.shape[0]) or total_count > ((1 - threshold) * data.shape[0]):
                    drop.append(data.columns[i])
                    positives.append(total_count / data.shape[0])

        # Construct the record_near_unique data frame
        self.record_near_unique = pd.DataFrame([drop, positives]).T.rename(columns={0: 'features', 1: 'pos'})

        self.removal_ops['near_unique'] = drop

        if json_file:
            bin_json = np.linspace(0, 1, 21)
            near_unique = self.record_near_unique.groupby(pd.cut(self.record_near_unique.iloc[:, 1], bin_json)).count()
            near_unique.drop(columns=['features'])
            near_unique.to_json('data/near_unique_plot.json', orient='split')
            print(near_unique)

    def plot_near_unique(self):

        """
        Plots the near unique histogram, showing how the positive class is distributed
        """

        if self.record_near_unique is None:
            raise NotImplementedError('No near unique values, perhaps the method was not ran')

        plt.figure()
        self.record_near_unique.plot.hist(color='red', edgecolor='k', figsize=(6, 4), fontsize=14)

        plt.xlim([0, 1])

        plt.title('Near Unique Histogram', size=18)

        plt.xlabel("% of Positive Data Class")
        plt.ylabel("# Of Features")

        plt.show()

    def identify_feature(self, data, col_name):

        """
        Allows user to add additional features to the removal list
        """

        if col_name in list(data):
            self.removal_ops['custom'].append(col_name)

        else:
            print("Column name not found!")

    def readd_feature(self, col_name):

        """
        Removes feature(s) from the removal list if the operator decides that the feature is important
        """

        # Iterate through all the keys in the dictionary
        for key in self.removal_ops.keys():
            # If the col_name is found in one of the removal ops
            if col_name in self.removal_ops[key]:
                # Enumerate across all col names in that removal ops list
                for i, col_names in enumerate(self.removal_ops[key]):
                    # If the col name is found, delete it
                    if col_names == col_name:
                        del self.removal_ops[key][i]

    def check_identified(self):

        """
        Prints out a list of all the variables set for removal
        """

        # Identifies all variables set for deletion
        all_identified = list(chain(*list(self.removal_ops.values())))

        print("{} features are identified for removal".format(len(all_identified)))

        return all_identified

    def removal(self, data, methods):
        """
        Removes the columns set for removal with accordance to the specific method.

        Methods: Delete all features identified with this particular method, if all is passed, deletes all columns in the
                 removal ops dictionary
        keys: ['missing], ['single_unique'], ['collinear'], ['id_type'], ['near_unique'], ['custom']
        """

        features_to_drop = []

        # data = pd.get_dummies(data)

        if methods == 'all':

            print('{} method(s) has been ran.'.format(list(self.removal_ops.keys())))

            # Find the unique features to drop
            features_to_drop = list(chain(*list(self.removal_ops.values())))

        else:
            # Iterate through the specified methods
            for method in methods:
                # Check to make sure the method has been run
                if method not in self.removal_ops.keys():
                    raise NotFittedError('{} method has not been ran.'.format(method))

                # Append the features identified for removal
                else:
                    features_to_drop.append(self.removal_ops[method])

            # Find the unique features to drop
            features_to_drop = set(list(chain(*features_to_drop)))

        # Find unique features, instead of trying to drop the same column twice
        features_to_drop, _ = self.unique_value(features_to_drop)

        # Remove the features and return the data
        data = data.drop(columns=features_to_drop)
        self.features_removed = features_to_drop

        print('Removed {} features.  The new data has {} features.'.format(len(features_to_drop), data.shape[1]))
        return data, self.features_removed

    @staticmethod
    def online_removal(data, features_to_drop):
        """
        Feature removal online to ensure the ML models are receiving correct number of inputs

        Inputs
             ---
                       data: Data from SCADA
           features_to_drop: Useless features identified from previous


        Returns
             ---
                      data:  Data with "features_to_drop" removed.
        """
        data = data.drop(columns=features_to_drop)

        return data

    """
    Identifies unique values
    """

    @staticmethod
    def unique_value(data):
        s = []
        duplicate = 0
        for x in data:
            if x in s:
                duplicate += 1
                pass
            else:
                s.append(x)
        return s, duplicate

    # Methods below this line are only applicable to Suncor Pipeline

    def remove_idtype(self, data, col_name="name"):

        """
        Deletes custom columns
        """

        drop = []

        # If a list of values to drop was given
        if type(col_name) == list:

            for value in col_name:
                # If the value does not have _, because all the headers in pipeline are 37879814_302, we need to add _
                if value[0] != "_":
                    # Add the _
                    value = "_" + value

                # Create a list of drop columns
                drop = drop + list(data.filter(regex=value))

        # If only 1 value was given
        else:
            # If the col_name is not specified with _ in front, add the _
            if col_name[0] != "_":
                col_name = "_" + col_name

            # Create a list of drop columns
            drop = drop + list(data.filter(regex=col_name))

        self.removal_ops['id_type'] = drop


if __name__ == "__main__":

    # day_minutes = 60 * 24
    #
    # Data, Original_data = data_loader('data/downsampled_data.csv', chunk_size=day_minutes, num_of_chunks=1000)

    Data = pd.read_csv('test_datasets/CoffeeBeanData.csv')

    # Builds Feature Selector object
    feature_selection = FeatureSelector()

    """
    From Google Docs:

    input: 0 (missing value method), % missing, data file from previously
    """
    # Identifies missing values
    feature_selection.identify_missing(Data, missing_threshold=0.3, json_file=False)
    feature_selection.plot_missing()

    """
    From Google Docs:

    input: 1 (unique value method), data file from previously
    """
    # Identifies unique values
    feature_selection.identify_single_unique(Data, json_file=False)
    feature_selection.plot_single_unique()

    """
    From Google Docs:

    input: 2 (collinear value method), correlation, data file from previously
    """
    # Identifies missing values
    feature_selection.identify_collinear(Data, correlation_threshold=0.90, json_file=False)
    feature_selection.plot_collinear()

    """
    From Google Docs:

    input: 3 (near unique method), amount of one class, data file from previously
    """
    # Identifies near unique columns
    feature_selection.remove_near_unique(Data, threshold=0.02, json_file=False)
    feature_selection.plot_near_unique()

    """
    Delete all the _319 and _322 IDType features

    input: col IDTypes, data file from previously
    """
    feature_selection.remove_idtype(Data, col_name=['319', '322'])

    """
    Removes one custom entry

    input: column name, data file from previously
    """
    feature_selection.identify_feature(Data, '175643118_630')

    """
    Re-add a feature that was accidently deleted
    """
    feature_selection.readd_feature('175643118_630')

    """
    Remove the features
    """
    Data, features_removed = feature_selection.removal(Data, 'all')

    # Online feature selection test
    Data2 = pd.read_csv('test_datasets/CoffeeBeanData.csv')
    Data2 = feature_selection.online_removal(Data2, features_removed)

    assert(Data2.shape[1] == Data.shape[1])
