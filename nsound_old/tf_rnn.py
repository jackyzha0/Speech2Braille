
#!/usr/bin/env python

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os

print('Loading Data...')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# PARAMS
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10
#Data Shape
n_input = 28
n_steps = 28
n_hidden = 128 # hidden layer num of features
n_classtypes = 10 # MNIST total classes (0-9 digits)

print(input_data)

#Add sumaries to TensorBoard for weights and biases
def var_summaries(var):
    with tf.name_scope('summary'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

# tf Graph input
input_vector = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classtypes])

weights = tf.Variable(tf.random_normal([2*n_hidden, n_classtypes]))
biases = tf.Variable(tf.random_normal([n_classtypes]))
var_summaries(weights)
var_summaries(biases)

def BiRNN(input_vector, weights, biases):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    input_vector = tf.unstack(input_vector, n_steps, 1)


    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs,_,_ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input_vector, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    #return tf.matmul(outputs[-1], weights['out']) + biases['out']
    return tf.matmul(outputs[-1], weights) + biases

pred = BiRNN(input_vector, weights, biases)

# Define loss and optimizer
with tf.name_scope('cross_entropy'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    var_summaries(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    var_summaries(accuracy)

merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1

    #Tensorboard
    print("Starting Tensorboard...")
    writer = tf.summary.FileWriter("totalsummary", sess.graph)

    print("Training...")
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={input_vector: batch_x, y: batch_y})
        if step % display_step == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={input_vector: batch_x, y: batch_y})
            i = tf.convert_to_tensor(step, dtype = tf.int8)
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={input_vector: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

            trainsummary = sess.run(merged, feed_dict={input_vector: batch_x, y: batch_y})
            writer.add_summary(trainsummary, step)
            writer.flush()
        step += 1

    print("Optimization Finished!")
    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={input_vector: test_data, y: test_label}))



writer.close()
