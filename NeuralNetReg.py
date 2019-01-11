"""
Artificial Neural Network Regression Code v1.1 (Feed-forward neural network)

By: Rui Nian

Date of last edit: January 3rd, 2018

Patch Notes: Added RMSE, R2
             Added Train / Test Accuracy during runtime
             Added Live evaluation with batch normalization

Known Issues: -

Features:
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import pickle

from sklearn.model_selection import train_test_split

from EVAL_ValveVsPress import ValvePresEval

import gc

from copy import deepcopy

import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)


# Min max normalization
class MinMaxNormalization:
    """
    Inputs
       -----
            data:  Input feature vectors from the training data

    Attributes
       -----
         col_min:  The minimum value per feature
         col_max:  The maximum value per feature
     denominator:  col_max - col_min

     Methods
        -----
     init:  Builds the col_min, col_max, and denominator
     call:  Normalizes data based on init attributes
    """

    def __init__(self, data):
        self.col_min = np.min(data, axis=0).reshape(1, data.shape[1])
        self.col_max = np.max(data, axis=0).reshape(1, data.shape[1])
        self.denominator = abs(self.col_max - self.col_min)

        # Fix divide by zero, replace value with 1 because these usually happen for boolean columns
        for index, value in enumerate(self.denominator[0]):
            if value == 0:
                self.denominator[0][index] = 1

    def __call__(self, data):
        return np.divide((data - self.col_min), self.denominator)

    def unnormalize(self, data):
        return np.multiply((data + self.col_min), self.denominator)


# Load data
# path = '/Users/ruinian/Documents/Willowglen/valve_pressure_data/'
path = '/home/rui/Documents/Willowglen/'

raw_data = pd.read_csv(path + 'valve_pressure_data/valve_pressure_data.csv')
raw_data.sort_values(by=['175642874_630'], inplace=True)
raw_data = raw_data.values

assert(not np.isnan(raw_data).any())

train_X, test_X, train_y, test_y = train_test_split(raw_data[:, 1], raw_data[:, 0],
                                                    test_size=0.1, random_state=1, shuffle=True)

# Reshaping data to ensure compatibility with tensorflow
train_X = train_X.reshape(-1, 1)
test_X = test_X.reshape(-1, 1)

train_y = train_y.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)

# Normalization of input space
pickle_in = open(path + 'neural_net_tf/pickles/norm_reg.pickle', 'rb')
min_max_normalization = pickle.load(pickle_in)

# min_max_normalization = MinMaxNormalization(train_X)
train_X = min_max_normalization(train_X)
test_X = min_max_normalization(test_X)

input_size = train_y.shape[1]
h1_nodes = 30
h2_nodes = 30
h3_nodes = 30
output_size = 1

batch_size = 256
total_batch_number = int(train_X.shape[0] / batch_size)

X = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
y = tf.placeholder(dtype=tf.float32, shape=[None, output_size])

# Batch normalization
training = False
is_train = tf.placeholder(dtype=tf.bool, name='is_train')

hidden_layer_1 = {'weights': tf.get_variable('h1_weights', shape=[input_size, h1_nodes],
                                             initializer=tf.contrib.layers.variance_scaling_initializer()),
                  'biases': tf.get_variable('h1_biases', shape=[h1_nodes],
                                            initializer=tf.contrib.layers.variance_scaling_initializer())}

hidden_layer_2 = {'weights': tf.get_variable('h2_weights', shape=[h1_nodes, h2_nodes],
                                             initializer=tf.contrib.layers.variance_scaling_initializer()),
                  'biases': tf.get_variable('h2_biases', shape=[h2_nodes],
                                            initializer=tf.contrib.layers.variance_scaling_initializer())}

hidden_layer_3 = {'weights': tf.get_variable('h3_weights', shape=[h2_nodes, h3_nodes],
                                             initializer=tf.contrib.layers.variance_scaling_initializer()),
                  'biases': tf.get_variable('h3_biases', shape=[h3_nodes],
                                            initializer=tf.contrib.layers.variance_scaling_initializer())}

output_layer = {'weights': tf.get_variable('output_weights', shape=[h3_nodes, output_size],
                                           initializer=tf.contrib.layers.variance_scaling_initializer()),
                'biases': tf.get_variable('output_biases', shape=[output_size],
                                          initializer=tf.contrib.layers.variance_scaling_initializer())}

l1 = tf.add(tf.matmul(X, hidden_layer_1['weights']), hidden_layer_1['biases'])
l1 = tf.nn.relu(l1)
l1 = tf.layers.batch_normalization(l1, training=is_train)

l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
l2 = tf.nn.relu(l2)
l2 = tf.layers.batch_normalization(l2, training=is_train)

l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
l3 = tf.nn.relu(l3)
l3 = tf.layers.batch_normalization(l3, training=is_train)

output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

# L2 Regularization
lambd = 0.01
trainable_vars = tf.trainable_variables()
lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name])

loss = tf.math.sqrt(tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=output)))

# Batch Normalization
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)

# Evaluation metrics
RMSE = tf.reduce_mean(tf.math.square(tf.math.subtract(y, output)))
MSE = tf.reduce_sum(tf.math.square(tf.math.subtract(y, output)))
Syy = tf.reduce_sum(tf.math.square(tf.math.subtract(y, tf.reduce_mean(y))))
R2 = 1 - np.divide(MSE, Syy)

init = tf.global_variables_initializer()

epochs = 1
loss_history = []

saver = tf.train.Saver()

with tf.Session() as sess:

    saver.restore(sess, path + 'neural_net_tf/checkpoints/test_reg.ckpt')

    # sess.run(init)
    #
    # for epoch in range(epochs):
    #
    #     for i in range(total_batch_number + 1):
    #
    #         # If the batch size is too big for the last bit of data
    #         if i == total_batch_number:
    #             batch_X = train_X[i * batch_size:, :]
    #             batch_y = train_y[i * batch_size:, :]
    #         else:
    #             batch_index = i * batch_size
    #             batch_X = train_X[batch_index:batch_index + batch_size, :]
    #             batch_y = train_y[batch_index:batch_index + batch_size, :]
    #
    #         sess.run(optimizer, feed_dict={X: batch_X, y: batch_y, is_train: training})
    #         current_loss = sess.run(loss, feed_dict={X: batch_X, y: batch_y, is_train: training})
    #         loss_history.append(current_loss)
    #
    #         if i % 100 == 0:
    #             train_RMSE = sess.run(RMSE, feed_dict={X: train_X, y: train_y, is_train: training})
    #             test_RMSE = sess.run(RMSE, feed_dict={X: test_X, y: test_y, is_train: training})
    #             print('Epoch: {} | Loss: {:5f} | Train RMSE: {:5f} | Test RMSE: {:5f}'.format(epoch, current_loss,
    #                                                                                           train_RMSE, test_RMSE))

    # Save Model
    # saver.save(sess, path + 'neural_net_tf/checkpoints/test_reg.ckpt')

    # Overall performance metrics
    overall_X = raw_data[:, 1].reshape(-1, 1)
    overall_X = min_max_normalization(overall_X)
    overall_y = raw_data[:, 0].reshape(-1, 1)

    # predictions = sess.run(output, feed_dict={X: overall_X, y: overall_y,
    #                                           is_train: training})

    # Evaluation, broken into pieces because of batch normalization
    predictions = tf.constant([], shape=(0, 1))
    for i in range(int(raw_data.shape[0] / batch_size) + 1):

        # If the batch size is too big for the last bit of data
        if i == (int(raw_data.shape[0] / batch_size)):
            batch_X = overall_X[i * batch_size:, :]
            batch_y = overall_y[i * batch_size:, :]

        else:
            batch_index = i * batch_size
            batch_X = overall_X[batch_index:batch_index + batch_size, :]
            batch_y = overall_y[batch_index:batch_index + batch_size, :]

        prediction = sess.run(output, feed_dict={X: batch_X, y: batch_y,
                                                 is_train: training})
        predictions = tf.concat([predictions, prediction], axis=0)

    predictions = sess.run(predictions)

    print('Overall R2: {:5f} | Overall RMSE: {:5f}'.format(sess.run(R2, feed_dict={X: overall_X,
                                                                                   y: overall_y,
                                                                                   is_train: training}),
                                                           sess.run(RMSE, feed_dict={X: overall_X,
                                                                                     y: overall_y,
                                                                                     is_train: training})))

    # Unnormalize the features
    overall_X = min_max_normalization.unnormalize(overall_X)

Online_Eval = ValvePresEval(overall_X, predictions)
Online_Eval.live_plots(overall_X, overall_y, time_start=500000, time_end=510000)
