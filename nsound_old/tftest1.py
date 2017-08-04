#!/usr/bin/env python
print('Iniatilizing TensorFlow Python..')
import os
import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#list of inputs
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

#estimator to train and evaluate
estimator = tf.contrib.learn.LinearRegressor(feature_columns = features)

#datasets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
#evaluation data should be kept seperate from training data
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

#epoch is how many batches of data there are
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_train}, y_train, batch_size=4, num_epochs=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_eval}, y_eval, batch_size=4, num_epochs=1000)

#invoke 1000 steps
estimator.fit(input_fn=input_fn, steps=1000)

#calc loss and print results
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r"% train_loss)
print("eval loss: %r"% eval_loss)
