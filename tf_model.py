# !/usr/local/bin/python
from __future__ import print_function
import time
print(time.strftime('[%H:%M:%S]'), 'Starting network... ')
import tensorflow as tf
import datetime
import os
import sys
import data_util
import numpy as np
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#PARAMS
FLAGS = None
#path = '/home/jacky/2kx/Spyre/__data/TIMIT/*/*/*'
#path = '/home/jacky/2kx/Spyre/pract_data/*'
path = '/home/jacky/2kx/Spyre/larger_pract/*/*'
logs_path = '/home/jacky/2kx/Spyre/git/totalsummary/logs/'+datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
num_mfccs = 13
prevcost = 0

num_classes = 28

num_hidden = 50
learning_rate = 1e-3
momentum = 0.9
num_layers = 1

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

print(time.strftime('[%H:%M:%S]'), 'Parsing directory... ')
dr = data_util.load_dir(path)
datasetsize = len(dr[0])
lr = dr[1]

num_examples = 64#dr[2]
num_epochs = 40
batchsize = 2
num_batches_per_epoch = int(num_examples/batchsize)

print(time.strftime('[%H:%M:%S]'), 'Loading network functions... ')
graph = tf.Graph()
with graph.as_default():
    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(num_hidden)

    #Network Code is heavily influenced by igormq's ctc_tensorflow example
    with tf.name_scope('inputLength'):
        seq_len = tf.placeholder(tf.int32, [None])

    with tf.name_scope('input'):
        inputs = tf.placeholder(tf.float32, [None, None, num_mfccs])
        targets = tf.sparse_placeholder(tf.int32)
        tf.summary.histogram("input",inputs)
        tf.summary.histogram("targets",tf.sparse_to_dense(targets.indices,targets.dense_shape,targets.values))

    with tf.name_scope('mfccs'):
        t = tf.placeholder(tf.float32, [None, None, None,None])
        tf.summary.image("mfccs",t,batchsize)

    # Stacking rnn cells
    with tf.name_scope('cellStack'):
        #stack = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
        cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
                                            state_is_tuple=True)

    # The second output is the last state and we will not use that
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
    shape = tf.shape(inputs)
    batch_s, TF_max_timesteps = shape[0], shape[1]

    # Reshaping to apply the same weights over the timesteps
    with tf.name_scope('outputs'):
        outputs = tf.reshape(outputs, [-1, num_hidden])

    with tf.name_scope('weights'):
        initializer = tf.contrib.layers.xavier_initializer()
        W = tf.Variable(initializer([num_hidden,num_classes]))
        #W = tf.Variable(tf.truncated_normal([num_hidden,num_classes],stddev=1))
        #W = tf.Variable(tf.truncated_normal([num_hidden,num_classes],stddev=0.1))
        #W = tf.Variable(tf.random_normal([num_hidden, num_classes]))
        tf.summary.histogram('weightsHistogram', W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.constant(0., shape=[num_classes]))
        tf.summary.histogram('biasesHistogram', b)

    with tf.name_scope('logits'):
        # Doing the affine projection
        logits = tf.matmul(outputs, W) + b
        # Reshaping back to the original shape
        logits = tf.reshape(logits, [batch_s, -1, num_classes])
        # Time major
        logits = tf.transpose(logits, (1, 0, 2))
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.ctc_loss(targets, logits, seq_len, ignore_longer_outputs_than_inputs=True))
    with tf.name_scope('cost'):
        cost = tf.reduce_mean(loss)
    tf.summary.scalar("cost", cost)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate,momentum).minimize(cost)
        #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    with tf.name_scope('decoder'):
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
        #decoded = tf.to_int32(ctc.ctc_beam_search_decoder(logits, seq_len)[0][0])
        tf.summary.histogram("outputs",tf.sparse_to_dense(decoded[0].indices,decoded[0].dense_shape,decoded[0].values))
    with tf.name_scope('LER'):
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),targets))
    tf.summary.scalar("LER", ler)

    merged = tf.summary.merge_all()

# Launch the graph
with tf.Session(graph=graph) as sess:
    print("Starting Tensorboard...")
    writer = tf.summary.FileWriter(logs_path, graph=sess.graph)
    tf.global_variables_initializer().run()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    #Load paths
    print(time.strftime('[%H:%M:%S]'), 'Passing params... ')
    data_util.setParams(batchsize, num_mfccs, num_classes)
    for curr_epoch in range(num_epochs):
        print('>>>',time.strftime('[%H:%M:%S]'), 'Epoch',curr_epoch,'/',num_epochs)
        train_cost = train_ler = 0
        start = time.time()
        for batch in range(num_batches_per_epoch):
            # Getting the index
            indexes = [i % num_examples for i in range(batch * batchsize, (batch + 1) * batchsize)]
            train_targets = data_util.next_target_miniBatch(indexes,dr[1])
            train_inputs = data_util.next_miniBatch(indexes,dr[0])
            #train_inputs,train_targets = data_util.fake_data(num_examples,num_mfccs,num_classes-1)
            newindex = [i % num_examples for i in range(batchsize)]
            batch_train_inputs = train_inputs[newindex]
            # Padding input to max_time_step of this batch
            batch_train_inputs, batch_train_seq_len = data_util.pad_sequences(batch_train_inputs)
            # Converting to sparse representation so as to to feed SparseTensor input
            batch_train_targets = data_util.sparse_tuple_from(train_targets[newindex])
            batch_train_mfccs_img = []
            data_util.saveImg(batch_train_inputs)
            for n in range(0,batchsize):
                batch_train_mfccs_img.append(data_util.jpg_image_to_array('/home/jacky/2kx/Spyre/git/img/'+str(n)+'.jpg'))
            feed = {inputs: batch_train_inputs,
                    targets: batch_train_targets,
                    seq_len: batch_train_seq_len,
                    t: batch_train_mfccs_img
                    }
            batch_cost, _ = sess.run([cost, optimizer], feed)
            train_cost += batch_cost*batchsize
            train_ler += sess.run(ler, feed_dict=feed)*batchsize
            print('      >>>',time.strftime('[%H:%M:%S]'), 'Batch',batch,'/',num_batches_per_epoch,'@Cost',batch_cost/batchsize)
            summary = sess.run(merged, feed_dict=feed, options=run_options, run_metadata=run_metadata)
            writer.add_summary(summary, batch+(curr_epoch*num_batches_per_epoch))
            writer.add_run_metadata(run_metadata, 'step%03d' % int(batch+(curr_epoch*num_batches_per_epoch)))
            writer.flush()

        # Metrics mean
        train_cost /= num_examples
        train_ler /= num_examples

        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler, time.time() - start))
        d = sess.run(decoded[0], feed_dict=feed)
        dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=sess)
        #print(dense_decoded)
        for i, seq in enumerate(dense_decoded):

            seq = [s for s in seq if s != -1]

            print('Sequence %d' % i)
            print('\t Original:\n%s' % train_targets[i])
            print('\t Decoded:\n%s' % seq)
