# !/usr/local/bin/python
from __future__ import print_function
import time
print(time.strftime('[%H:%M:%S]'), 'Starting network... ')
import tensorflow as tf
import datetime
import os
import sys
import random
import data_util
import numpy as np
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#PARAMS
FLAGS = None
test_path = '/home/jacky/2kx/Spyre/__data/TIMIT/TEST/*/*'
path = '/home/jacky/2kx/Spyre/__data/TIMIT/TRAIN/*/*'
#path = '/home/jacky/2kx/Spyre/pract_data/*'
#path = '/home/jacky/2kx/Spyre/larger_pract/*/*'
num_mfccs = 13
prevcost = 0

num_classes = 28

num_hidden = 100
learning_rate = 1e-3
momentum = 0.9
num_layers = 2

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

print(time.strftime('[%H:%M:%S]'), 'Parsing training directory... ')
dr = data_util.load_dir(path)
datasetsize = len(dr[0])

num_examples = 1#dr[2]
num_epochs = 200
batchsize = 1
num_batches_per_epoch = int(num_examples/batchsize)

print(time.strftime('[%H:%M:%S]'), 'Parsing testing directory... ')
t_dr = data_util.load_dir(test_path)
testsetsize = len(t_dr[0])
testbatchsize = batchsize*10

logs_path = '/home/jacky/2kx/Spyre/nsound_git/totalsummary/logs/'+datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')+'_'+str(batchsize)+'_'+str(num_epochs)

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

    #with tf.name_scope('mfccs'):
    #    t = tf.placeholder(tf.float32, [None, None, None,None])
    #    tf.summary.image("mfccs",t,batchsize)

    # Stacking rnn cells
    with tf.name_scope('cellStack'):
        stack = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
        cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        #stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,state_is_tuple=True)

    # The second output is the last state and we will not use that
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
    shape = tf.shape(inputs)
    batch_s, TF_max_timesteps = shape[0], shape[1]

    # Reshaping to apply the same weights over the timesteps
    with tf.name_scope('outputs'):
        outputs = tf.reshape(outputs, [-1, num_hidden])

    with tf.name_scope('weights'):
        initializer = tf.contrib.layers.xavier_initializer()
        #W = tf.Variable(initializer([num_hidden,num_classes]))
        #W = tf.Variable(tf.truncated_normal([num_hidden,num_classes],stddev=1))
        #W = tf.Variable(tf.truncated_normal([num_hidden,num_classes],stddev=0.1))
        W = tf.Variable(tf.random_normal([num_hidden, num_classes]))
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
        #optimizer = tf.train.MomentumOptimizer(learning_rate,momentum).minimize(cost)
        optimizer = tf.train.AdamOptimizer(learning_rate,momentum).minimize(cost)
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
    initstart = time.time()
    train_writer = tf.summary.FileWriter(logs_path+'/TRAIN', graph=sess.graph)
    test_writer = tf.summary.FileWriter(logs_path+'/TEST', graph=sess.graph)
    tf.global_variables_initializer().run()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    #Load paths
    print(time.strftime('[%H:%M:%S]'), 'Passing params... ')
    data_util.setParams(batchsize, num_mfccs, num_classes)
    for curr_epoch in range(num_epochs):
        print('>>>',time.strftime('[%H:%M:%S]'), 'Epoch',curr_epoch+1,'/',num_epochs)
        train_cost = train_ler = 0
        start = time.time()
        for batch in range(num_batches_per_epoch):
            # Getting the index
            indexes = [i % num_examples for i in range(batch * batchsize, (batch + 1) * batchsize)]
            train_targets = data_util.next_target_miniBatch(indexes,dr[1])
            train_inputs = data_util.next_miniBatch(indexes,dr[0])
            #train_inputs,train_targets = data_util.fake_data(num_examples,num_mfccs,num_classes-1)
            newindex = [i % num_examples for i in range(batchsize)]
            random.shuffle(newindex)
            batch_train_inputs = train_inputs[newindex]
            # Padding input to max_time_step of this batch
            batch_train_inputs, batch_train_seq_len = data_util.pad_sequences(batch_train_inputs)
            # Converting to sparse representation so as to to feed SparseTensor input
            batch_train_targets = data_util.sparse_tuple_from(train_targets[newindex])
            #batch_train_mfccs_img = []
            #data_util.saveImg(batch_train_inputs)
            #for n in range(0,batchsize):
            #    batch_train_mfccs_img.append(data_util.jpg_image_to_array('/home/jacky/2kx/Spyre/nsound_git/img/'+str(n)+'.jpg'))
            feed = {inputs: batch_train_inputs,
                    targets: batch_train_targets,
                    seq_len: batch_train_seq_len
            #        ,t: batch_train_mfccs_img
                    }
            batch_cost, _ = sess.run([cost, optimizer], feed)
            train_cost += batch_cost*batchsize
            train_ler += sess.run(ler, feed_dict=feed)*batchsize
            print('      >>>',time.strftime('[%H:%M:%S]'), 'Batch',batch+1,'/',num_batches_per_epoch,'@Cost',batch_cost,'\r',)
            summary = sess.run(merged, feed_dict=feed, options=run_options, run_metadata=run_metadata)
            train_writer.add_summary(summary, int(batch)+int(curr_epoch*num_batches_per_epoch))
            train_writer.add_run_metadata(run_metadata, 'step%03d' % int(batch+(curr_epoch*num_batches_per_epoch)))
            train_writer.flush()

        # Metrics mean
        train_cost /= num_examples
        train_ler /= num_examples

        if (curr_epoch % 1 == 0):
            #Testing
            print('>>>',time.strftime('[%H:%M:%S]'), 'Evaluating Test Accuracy...')
            t_index = random.sample(range(0, testsetsize), testbatchsize)
            test_targets = data_util.next_target_miniBatch(t_index,t_dr[1])
            test_inputs = data_util.next_miniBatch(t_index,t_dr[0])
            newindex = [i % testbatchsize for i in range(testbatchsize)]
            batch_test_inputs = test_inputs[newindex]
            batch_test_inputs, batch_test_seq_len = data_util.pad_sequences(batch_test_inputs)
            batch_test_targets = data_util.sparse_tuple_from(test_targets[newindex])
            t_feed = {inputs: batch_test_inputs,
                    targets: batch_test_targets,
                    seq_len: batch_test_seq_len
                    }
            test_ler = sess.run(ler, feed_dict=t_feed)
            print(test_ler,train_ler)
            log = "Epoch {}/{}  |  Batch Cost : {:.3f}  |  Train Accuracy : {:.3f}%  |  Test Accuracy : {:.3f}%  |  Time Elapsed : {:.3f}s"
            print(log.format(curr_epoch+1, num_epochs, train_cost, 100-(train_ler*100), 100-(test_ler*100), time.time() - start))
        else:
            log = "Epoch {}/{}  |  Batch Cost : {:.3f}  |  Train Accuracy : {:.3f}%  |  Time Elapsed : {:.3f}s"
            print(log.format(curr_epoch+1, num_epochs, train_cost, 100-(train_ler*100), time.time() - start))
    print('Total Training Time: '+str(time.time() - initstart)+'s')
