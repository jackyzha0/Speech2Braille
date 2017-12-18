# !/usr/local/bin/python
from __future__ import print_function
import time
print(time.strftime('[%H:%M:%S]'), 'Starting network... ')
import tensorflow as tf
import os
import sys
import data_util
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#PARAMS
FLAGS = None
path = '/home/jacky/2kx/Spyre/__data/TIMIT/*/*/*'
#path = '/home/jacky/2kx/Spyre/pract_data/*'
num_mfccs = 4
batchsize = 32
num_hidden = 50
learning_rate = 1e-2
momentum = 0.9
num_layers = 2
prevcost = 0.00
#0th indice + End indice + space + blank label = 27 characters

num_classes = 63

print(time.strftime('[%H:%M:%S]'), 'Loading network functions... ')
graph = tf.Graph()
with graph.as_default():
    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(num_hidden)

    #Network Code is heavily influenced by igormq's ctc_tensorflow example
    # Has size [batch_size, max_stepsize, num_features], but the
    # batch_size and max_stepsize can vary along each step
    inputs = tf.placeholder(tf.float32, [batchsize, None,num_mfccs+28])

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32)
    #var_summaries(targets)

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

    W = tf.Variable(tf.truncated_normal([num_hidden,num_classes],stddev=0.01))
    #var_summaries(W)
    # Zero initialization
    b = tf.Variable(tf.constant(0., shape=[num_classes]))
    #var_summaries(b)

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    loss = tf.nn.ctc_loss(targets, logits, seq_len,ignore_longer_outputs_than_inputs=True)
    cost = tf.reduce_mean(loss)
    #var_summaries(loss)

    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum).minimize(cost)

    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),targets))

def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    '''Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.
        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths

print(time.strftime('[%H:%M:%S]'), 'Parsing directory... ')
dr = data_util.load_dir(path)
datasetsize = len(dr[0])
lr = dr[1]

# Launch the graph
with tf.Session(graph=graph) as sess:
    print("Starting Tensorboard...")
    writer = tf.summary.FileWriter("totalsummary", sess.graph)
    tf.global_variables_initializer().run()

    #Load paths
    print(time.strftime('[%H:%M:%S]'), 'Passing params... ')
    data_util.setParams(batchsize, num_mfccs, num_classes)
    #Training Loop
    train_cost = train_ler = 0
    start = prev = time.time()
    for i in range(0,datasetsize/batchsize):
        #print(datasetsize,batchsize)
        minibatch = data_util.next_miniBatch(i*batchsize,dr[0])
        minibatch_targets = data_util.next_target_miniBatch(i*batchsize,dr[1])
        indexes = [j % batchsize for j in range(i * batchsize, (i + 1) * batchsize)]
        batch_train_targets = data_util.sparse_tuple_from(minibatch_targets)
        batch_train_inputs = minibatch[indexes]
        batch_train_inputs, batch_train_seq_len = pad_sequences(batch_train_inputs)
        print(batch_train_seq_len)
        print(time.strftime('[%H:%M:%S]'),'Updating Weights of batch',i)
        feed = {inputs: batch_train_inputs, targets: batch_train_targets, seq_len: batch_train_seq_len}
        #Batch
        batch_cost, _ = sess.run([cost, optimizer], feed)
        train_cost += batch_cost*batchsize
        train_ler += sess.run(ler, feed_dict=feed)*batchsize
        time_fm = "{:.4f} seconds"

        d = sess.run(decoded[0], feed_dict=feed)
        dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=sess)
        for i, seq in enumerate(dense_decoded):

            seq = [s for s in seq if s != -1]
            print('Sequence %d' % i)
            print('\t Original:\n%s' % minibatch_targets[i])
            print('\t Decoded:\n%s' % seq)
        print(time.strftime('[%H:%M:%S]'),'Batch',i,'trained in',time_fm.format(time.time()-prev))
        print('>>> Cost:', batch_cost)
        print('>>> Cost Diff:', batch_cost-prevcost)
        prevcost = batch_cost
        prev = time.time()


    train_cost /= datasetsize
    train_ler /= datasetsize
    log = "Cost: {:.3f}, Label Error Rate: {:.3f}, Time taken: {:.3f}"
    print(time.strftime('[%H:%M:%S]'),log.format(train_cost, train_ler, time.time() - start))
