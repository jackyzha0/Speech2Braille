#!/usr/bin/env python
print('Init MNIST ...')
import os
import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#MNIST Classifier

#import MNIST data
print('Getting Data..')
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print('Done')

#image
x = tf.placeholder(tf.float32, [None, 784])

#init arrays of zeros
#Weights
W = tf.Variable(tf.zeros([784, 10]))
#Biases
b = tf.Variable(tf.zeros([10]))

#predicted distribution
y = tf.nn.softmax(tf.matmul(x, W) + b)

#true distribution
y_ = tf.placeholder(tf.float32, [None, 10])

#calculating the difference between predicted and true
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#optimization
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

#start session and init vars
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#1000 epochs
print('Training...')
for i in range(5000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print('Done!')

#get accuracy
print('Calculating Accuracy')
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
