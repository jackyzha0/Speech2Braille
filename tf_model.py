# !/usr/local/bin/python
from __future__ import print_function
import tensorflow as tf
import os
import sys
import string
import numpy as np
import data_util
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#PARAMS
FLAGS = None
path = '/home/jacky/Desktop/Spyre/__data/TIMIT/*/*/*'
num_mfccs = 13
batchsize = 10
preprocess = 1
max_stepsize = 5
sample_rate = 32000
num_hidden = 50
learning_rate = 1e-2
momentum = 0.9
#0th indice + End indice + space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 4

inputs = []
labels = []

def preprocess(rawsnd,stdev) :
    """
    #INCOMPLETE
    If preprocess == 1, add additional white noise with stdev (default 0.6)
    """

def BiRNN(x, weights, biases):
    """
    #Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    """
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    #Load paths
    dr = data_util.load_dir(path)
    print('Getting Data.',end='')
    #Training Loop
    for i in range(0,4700/batchsize):
        print('.',end='')
        data_util.getData(path, i*batchsize, 4700/batchsize, num_mfccs, num_classes-1)

    print('Done!')
    #print(dict)
    sess.run(init)
    print('Starting Tensorflow session')
    features_placeholder = tf.placeholder(tf.float32, shape=(None, num_mfccs+4))
    labels_placeholder = tf.placeholder(tf.string, shape=(batchsize))

    #train
    #eval
