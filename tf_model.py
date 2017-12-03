# !/usr/local/bin/python
from __future__ import print_function
import time
print(time.strftime('[%H:%M:%S]'), 'Starting network... ')
import tensorflow as tf
import os
import sys
import data_util
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#PARAMS
FLAGS = None
path = '/home/jacky/2kx/Spyre/__data/TIMIT/*/*/*'
num_mfccs = 13
batchsize = 10
max_timesteps = 150
timesteplen = 50
preprocess = 1
num_hidden = 150
learning_rate = 1e-2
momentum = 0.9
num_layers = 2
#0th indice + End indice + space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 4

print(time.strftime('[%H:%M:%S]'), 'Loading network functions... ')

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

def preprocess(rawsnd,stdev) :
    """
    If preprocess == 1, add additional white noise with stdev (default 0.6)
    """


graph = tf.Graph()
with graph.as_default():
    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(num_hidden)

    #Network Code is heavily influenced by igormq's ctc_tensorflow example
    # e.g: log filter bank or MFCC features
    # Has size [batch_size, max_stepsize, num_features], but the
    # batch_size and max_stepsize can vary along each step
    inputs = tf.placeholder(tf.float32, [None, None, num_mfccs+28])

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32)

    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])

    # Stacking rnn cells
    stack = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])

    # The second output is the last state and we will not use that
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

    shape = tf.shape(inputs)
    batch_s, TF_max_timesteps = shape[0], shape[1]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])

    W = tf.Variable(tf.truncated_normal([num_hidden,num_classes],stddev=0.1))
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cost)

    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),targets))

# Launch the graph
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    #Load paths
    print(time.strftime('[%H:%M:%S]'), 'Passing params... ')
    data_util.setParams(batchsize, num_mfccs, num_classes, max_timesteps, timesteplen)
    print(time.strftime('[%H:%M:%S]'), 'Parsing directory... ')
    dr = data_util.load_dir(path)
    lr = dr[1]
    #Training Loop
    for i in range(0,4700/batchsize):
        print(time.strftime('[%H:%M:%S]'),'Loading batch',i)
        minibatch = data_util.next_miniBatch(i*batchsize,dr[0])
        minibatch_targets = data_util.next_target_miniBatch(i*batchsize,dr[1])
        #print(minibatch,minibatch_targets)
        for j in range(0,batchsize):
            indexes = [i % batchsize for i in range(j * batchsize, (j + 1) * batchsize)]
            batch_train_targets = data_util.sparse_tuple_from(minibatch_targets[j])
            batch_train_inputs = minibatch[indexes]
            feed = {inputs: batch_train_inputs, targets: batch_train_targets}

            #Batch
            batch_cost, _ = sess.run([cost, optimizer], feed)
            train_cost += batch_cost*batch_size
            train_ler += sess.run(ler, feed_dict=feed)*batch_size
        train_cost /= num_examples
        train_ler /= num_examples
        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler, time.time() - start))

    #Decoding
    feed = {inputs: batch_train_inputs, targets: batch_train_targets, seq_len: batch_train_seq_len}
    d = session.run(decoded[0], feed_dict=feed)
    dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=session)
    for i, seq in enumerate(dense_decoded):

        seq = [s for s in seq if s != -1]

        print('Sequence %d' % i)
        print('\t Original:\n%s' % train_targets[i])
        print('\t Decoded:\n%s' % seq)
        print('Done!')
