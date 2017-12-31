# !/usr/local/bin/python
'''
Author: github.com/jackyzha0
All code is self-written unless explicitly stated
'''
from __future__ import print_function
import time
print('[OK] time')
print(time.strftime('[%H:%M:%S]'), 'Starting network... ')
import sugartensor as tf
print('[OK] tensorflow ')
import datetime
import sys
print('[OK] sys ')
import random
print('[OK] random ')
import numpy as np
print('[OK] numpy ')
import string
import librosa.display as lib_disp
import librosa.feature as lib_feat
import librosa
print('[OK] librosa ')
import glob
print('[OK] glob ')
from PIL import Image
print('[OK] PIL ')
import scipy
print('[OK] scipy ')
import os
print('[OK] os ')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

with tf.device("/cpu:0"):

    #PARAMS
    FLAGS = None
    SPACE_TOKEN = '<space>'
    SPACE_INDEX = 0
    FIRST_INDEX = ord('a') - 1  # 0 is reserved to space
    test_path = '/home/jacky/2kx/Spyre/__data/TIMIT/TEST/*/*'
    path = '/home/jacky/2kx/Spyre/__data/TIMIT/TRAIN/*/*'

    logImages = False

    # Network Params #
    num_mfccs = 13
    num_classes = 28
    num_hidden = 100
    learning_rate = 1e-4
    momentum = 0.3
    num_layers = 3
    ##############

    # Pickle Settings #
    pickle_path = 'timit_pickle'
    repickle = False
    print(time.strftime('[%H:%M:%S]'), 'Checking for pickled data... ')
    ##############
    def load_dir(fp):
        """Load raw paths data into arrays and returns important info
        Args:
            fp : string path to data
        Returns:
            Returns array of loaded files
            loaded[0] = sound
            loaded[1] = text
            loaded[2] = dataset size
        """
        with tf.name_scope('raw_data'):
            ind = 0
            raw_audio = []
            phonemes = []
            words = []
            text = []
            for __file in glob.iglob(fp + '/*.*'):
                    if not ("SA" in __file):
                        ind+=1
                        if (".wav" in __file):
                            raw_audio.append(__file)
                            __targ = __file[:-4]+str('.TXT')
                            with open(__targ) as f:
                                for line in f:
                                    res = ''.join([i for i in line if not i.isdigit()])
                            res = (list(res[2:-1].lower().translate(None, string.punctuation)))
                            tmp_res = []
                            for r in res:
                                if r==' ':
                                    tmp_res.append(0)
                                else:
                                    tmp_res.append(ord(r)-96)
                            text.append(tmp_res)
            print(time.strftime('[%H:%M:%S]'), 'Succesfully loaded data set of size',len(raw_audio))
            return raw_audio,text,len(raw_audio)
    ##############


    print(time.strftime('[%H:%M:%S]'), 'Parsing training directory... ')
    dr = load_dir(path)
    datasetsize = len(dr[0])

    # Training Params #
    num_examples = dr[2]
    num_epochs = 30
    batchsize = 8
    dropout_keep_prob = 1.0
    num_batches_per_epoch = int(num_examples/batchsize)
    ##############

    # Log Params #
    logs_path = 'totalsummary/logs/'+datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')+'_'+str(batchsize)+'_'+str(num_epochs)+'_'+str(dropout_keep_prob)
    savepath = 'totalsummary/ckpt'
    RESTOREMODEL = False
    RESTOREMODEL = True
    ##############

    print(time.strftime('[%H:%M:%S]'), 'Parsing testing directory... ')
    t_dr = load_dir(test_path)
    testsetsize = len(t_dr[0])
    testbatchsize = 16

    # Functions #
    def next_miniBatch(index,patharr,test=False):
        """Returns array of size batchsize with features for training
        Args:
            index: current position in training
        Returns:
            features: rank 3 tensor of batchsize * maxsize * num_features
        """
        minibatch = []
        for j in index:
            #tmp = next_Data(patharr[j])
            tmp = next_Data(patharr[j],test)
            minibatch.append(np.array(tmp[0]))
        minibatch = np.array(minibatch)
        #saveImg(minibatch)
        return np.asarray(minibatch)
    def next_target_miniBatch(index,patharr):
        minibatch = []
        for j in index:
            #tmp = patharr[j]
            tmp = patharr[j]
            tmp_k = []
            for k in range(0,len(tmp)):
                tmp_k.append(int(tmp[k]))
            minibatch.append(np.array(tmp_k))
            #print(time.strftime('[%H:%M:%S]'), 'Passed target tensor',np.asarray(tmp_k).shape)
        return np.asarray(minibatch)
    def next_Data(path,test):
        """Returns array of features for training
        Args:
            path: path to audio file to compute
        Returns:
            featurearr: rank 2 tensor of maxsize * num_features
        """
        z = path.replace('/','').split("TIMIT")[1][:-4]
        if test or repickle or not os.path.exists(pickle_path+'/'+z+'.npy'):
            featurearr = []
            ftrtmp=features(path, num_mfccs)
            featurearr.append(ftrtmp)
            np.save(pickle_path+'/'+z,featurearr)
            print(time.strftime('[%H:%M:%S]'), 'Pickle saved to',pickle_path+'/'+z[:-4])
        else:
            featurearr = np.load(pickle_path+'/'+z+'.npy')
        return featurearr
    def features(rawsnd, num) :
        """Compute num amount of audio features of a sound
        Args:
            rawsnd : array with string paths to .wav files
            num : numbers of mfccs to compute
        Returns:
            Return a num x max_stepsize*32 feature vector
        """
        x, sample_rate = librosa.load(rawsnd)
        #s_tft = np.abs(librosa.stft(x))
        ft = lib_feat.mfcc(y=x, sr=sample_rate, n_mfcc=num+1).T
        #t_ft = ft
        #ft = np.delete(np.append(ft,lib_feat.delta(t_ft),axis=1),0,1)
        #ft = np.delete(np.append(ft,lib_feat.delta(t_ft,order=2),axis=1),0,1)
        ft = np.delete(ft,0,1)
        ft -= (np.mean(ft,axis=0) + 1e-8)
        return (ft)
    def r_saveImg(arr):
        """Saves testing batch images to disk
        Args:
            arr: array to save
        Returns:
            nothing
        """
        n=0
        for i in arr:
            im = scipy.misc.toimage(i)
            im.save('r_img/'+str(n)+'.jpg')
            n+=1
    def saveImg(arr):
        """Saves training images to disk
        Args:
            arr: array to save
        Returns:
            nothing
        """
        n=0
        for i in arr:
            im = scipy.misc.toimage(i)
            im.save('img/'+str(n)+'.jpg')
            n+=1
    def sparse_tuple_from(sequences, dtype=np.int32):
        """
        Author: github.com/igormq
        Create a sparse representention of input array. For handling one-hot vector in targets
        Args:
            sequences: a list of lists of type dtype where each element is a sequence
        Returns:
            A tuple with (indices, values, shape)"""
        indices = []
        values = []
        for i, seq in enumerate(sequences):
            indices.extend(zip([i]*len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

        return indices, values, shape
    def jpg_image_to_array(image_path):
        """Creates an array given path to image
        Args:
            image_path: system path to image
        Returns:
            height x width 2D array with values from 0 to 255
        """
        img = Image.open(image_path).convert('L')  # convert image to 8-bit grayscale
        WIDTH, HEIGHT = img.size
        data = list(img.getdata()) # convert image data to a list of integers
        # convert that to 2D list (list of lists of integers)
        data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
        return np.expand_dims(data, axis=4)
    def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                      padding='post', truncating='post', value=0.):
        '''
        Author: github.com/igormq
        Pads each sequence to the same length: the length of the longest
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
    def fake_data(num_examples, num_features, num_labels, min_size = 10, max_size=100):
        """Generates random noise as input (for debug purposes)
        Args:
            num_examples: number of instances of data
            num_features: number of fake features to generate
            num_labels: number of classes-1
            min_size: minimum timestep length of fake data
            max_size: max timestep length
        Returns:
            inputs: num_examples x num_features x max_size
            labels: num_examples x max_size [from 0 -> num_labels]
        """
        np.random.seed(0)
        timesteps = np.random.randint(min_size, max_size, (num_examples,))
        inputs = np.asarray([np.random.randn(t, num_features).astype(np.float32) for t in timesteps])
        #inputs = np.asarray([np.ones((t, num_features))*255 for t in timesteps])
        labels = np.asarray([np.random.randint(0, num_labels, np.random.randint(1, inputs[i].shape[0], (1,))).astype(np.int64) for i, _ in enumerate(timesteps)])
        return inputs,labels
    ####################

with tf.device("/gpu:0"):
    print(time.strftime('[%H:%M:%S]'), 'Loading network functions... ')
    graph = tf.Graph()
    with graph.as_default():
        #tf.set_random_seed(0)
        keep_prob = tf.placeholder(tf.float32)
        initializer = tf.contrib.layers.xavier_initializer()
        def lstm_cell():
            #return tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=keep_prob, activation=tf.tanh)
            return tf.contrib.rnn.LayerNormBasicLSTMCell(num_hidden,forget_bias=1.0,activation=tf.tanh,dropout_keep_prob=keep_prob)
            #return tf.contrib.rnn.LSTMCell(num_hidden,initializer=initializer,forget_bias=keep_prob, activation=tf.tanh)

        #Network Code is heavily influenced by igormq's ctc_tensorflow example
        with tf.name_scope('inputLength'):
            seq_len = tf.placeholder(tf.int32, [None])

        with tf.name_scope('input'):
            inputs = tf.placeholder(tf.float32, [None, None, num_mfccs])
            targets = tf.sparse_placeholder(tf.int32)
            tf.summary.histogram("input",inputs)
            tf.summary.histogram("targets",tf.sparse_to_dense(targets.indices,targets.dense_shape,targets.values))

        if logImages:
            with tf.name_scope('mfccs'):
                t = tf.placeholder(tf.float32, [None, None, None,None])
                tf.summary.image("mfccs",t,batchsize)

        # Stacking rnn cells
        with tf.name_scope('cellStack'):
            #stack = tf.contrib.rnn.LayerNormBasicLSTMCell(num_hidden,forget_bias=1.0,activation=tf.sigmoid,dropout_keep_prob=keep_prob)
            #stack = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers,num_hidden,num_mfccs,direction=CUDNN_RNN_BIDIRECTIONAL,dropout=dropout_keep_prob)
            #cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
            #stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,state_is_tuple=True)
            stack = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])

        outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
        shape = tf.shape(inputs)
        batch_s, TF_max_timesteps = shape[0], shape[1]

        # Reshaping to apply the same weights over the timesteps
        with tf.name_scope('outputs'):
            outputs = tf.reshape(outputs, [-1, num_hidden])

        with tf.name_scope('weights'):
            #W = tf.Variable(initializer([num_hidden,num_classes]))
            #W = tf.Variable(tf.truncated_normal([num_hidden,num_classes],stddev=0.1))
            W = tf.Variable(tf.random_normal([num_hidden, num_classes],stddev=0.3))
            tf.summary.histogram('weightsHistogram', W)
        with tf.name_scope('biases'):
            #b = tf.Variable(tf.constant(0., shape=[num_classes]))
            b = tf.Variable(initializer([num_classes]))
            tf.summary.histogram('biasesHistogram', b)

        with tf.name_scope('logits'):
            # Doing the affine projection
            logits = tf.matmul(outputs, W) + b
            # Reshaping back to the original shape
            logits = tf.reshape(logits, [batch_s, -1, num_classes])
            # Time major
            logits = tf.transpose(logits, (1, 0, 2))
        with tf.name_scope('loss'):
            x = tf.nn.ctc_loss(targets, logits, seq_len, ignore_longer_outputs_than_inputs=True)
            #loss = tf.reduce_mean(tf.boolean_mask(x, tf.logical_not(tf.is_inf(x))))
            loss = tf.reduce_mean(x)
        with tf.name_scope('cost'):
            cost = tf.reduce_mean(loss)
        tf.summary.scalar("cost", cost)
        with tf.name_scope('optimizer'):
            #optimizer = tf.train.MomentumOptimizer(learning_rate,momentum).minimize(cost)
            optimizer = tf.train.AdamOptimizer(learning_rate,momentum)#.minimize(loss)
            #optimizer = tf.train.AdagradOptimizer(learning_rate)
            gvs = optimizer.compute_gradients(cost)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            train_optimizer = optimizer.apply_gradients(capped_gvs)

        with tf.name_scope('decoder'):
            #decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
            decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)

        with tf.name_scope('LER'):
            ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),targets))
        tf.summary.scalar("LER", ler)

        merged = tf.summary.merge_all()

with tf.device("/cpu:0"):
    # Launch the graph
    with tf.Session(graph=graph, config=config) as sess:
        print("Starting Tensorboard...")
        initstart = time.time()
        train_writer = tf.summary.FileWriter(logs_path+'/TRAIN', graph=sess.graph)
        test_writer = tf.summary.FileWriter(logs_path+'/TEST', graph=sess.graph)
        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        if RESTOREMODEL == True:
            if os.path.exists(savepath):
                saver.restore(sess, savepath+'/model.ckpt')
                print('>>>',time.strftime('[%H:%M:%S]'),'Model Restored Succesfully')
            else:
                print('>>>',time.strftime('[%H:%M:%S]'),'Model Restore Failed! Make sure the chkpt file exists')
        #Load paths
        for curr_epoch in range(num_epochs):
            print('>>>',time.strftime('[%H:%M:%S]'), 'Epoch',curr_epoch+1,'/',num_epochs)
            train_cost = train_ler = 0
            start = t_time = time.time()
            for batch in range(num_batches_per_epoch):
                # Getting the index
                indexes = [i % num_examples for i in range(batch * batchsize, (batch + 1) * batchsize)]
                train_targets = next_target_miniBatch(indexes,dr[1])
                train_inputs = next_miniBatch(indexes,dr[0])
                #train_inputs,train_targets = fake_data(num_examples,num_mfccs,num_classes-1)
                newindex = [i % num_examples for i in range(batchsize)]
                random.shuffle(newindex)
                batch_train_inputs = train_inputs[newindex]
                # Padding input to max_time_step of this batch
                batch_train_inputs, batch_train_seq_len = pad_sequences(batch_train_inputs)
                # Converting to sparse representation so as to to feed SparseTensor input
                batch_train_targets = sparse_tuple_from(train_targets[newindex])
                if logImages:
                    batch_train_mfccs_img = []
                    saveImg(batch_train_inputs)
                    for n in range(0,batchsize):
                        batch_train_mfccs_img.append(jpg_image_to_array('img/'+str(n)+'.jpg'))
                feed = {inputs: batch_train_inputs,
                        targets: batch_train_targets,
                        seq_len: batch_train_seq_len,
                        keep_prob: dropout_keep_prob
                #        ,t: batch_train_mfccs_img
                        }
                batch_cost, _, l = sess.run([cost, train_optimizer, ler], feed)
                train_cost += batch_cost*batchsize
                train_ler += l*batchsize
                print('['+str(curr_epoch)+']','  >>>',time.strftime('[%H:%M:%S]'), 'Batch',batch+1,'/',num_batches_per_epoch,'@Cost',batch_cost,'Time Elapsed',time.time()-t_time,'s')
                t_time=time.time()

            # Metrics mean
            train_cost /= num_examples
            train_ler /= num_examples
            summary = sess.run(merged, feed_dict=feed, options=run_options, run_metadata=run_metadata)
            train_writer.add_summary(summary, int(batch)+int(curr_epoch*num_batches_per_epoch))
            train_writer.add_run_metadata(run_metadata, 'step%03d' % int(batch+(curr_epoch*num_batches_per_epoch)))
            train_writer.flush()

            if (curr_epoch % 1 == 0):
                #Testing
                print('>>>',time.strftime('[%H:%M:%S]'), 'Evaluating Test Accuracy...')
                t_index = random.sample(range(0, testsetsize), testbatchsize)
                test_targets = next_target_miniBatch(t_index,t_dr[1])
                test_inputs = next_miniBatch(t_index,t_dr[0],test=True)
                newindex = [i % testbatchsize for i in range(testbatchsize)]
                batch_test_inputs = test_inputs[newindex]
                batch_test_inputs, batch_test_seq_len = pad_sequences(batch_test_inputs)
                batch_test_targets = sparse_tuple_from(test_targets[newindex])
                if logImages:
                    batch_test_mfccs_img = []
                    r_saveImg(batch_test_inputs)
                    for n in range(0,testbatchsize):
                        batch_test_mfccs_img.append(jpg_image_to_array('r_img/'+str(n)+'.jpg'))
                t_feed = {inputs: batch_test_inputs,
                        targets: batch_test_targets,
                        seq_len: batch_test_seq_len,
                        keep_prob: 1.0
                #        ,t: batch_test_mfccs_img
                        }
                test_ler,d = sess.run((ler,decoded[0]), feed_dict=t_feed)
                dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=sess)
                for i, seq in enumerate(dense_decoded):
                    seq = [s for s in seq if s != -1]
                    tmp_o = ""
                    for q in test_targets[i]:
                        if q==0:
                            tmp_o+=" "
                        else:
                            tmp_o+=chr(q+96)
                    tmp_d = ""
                    for k in seq:
                        if k==0:
                            tmp_d+=" "
                        else:
                            tmp_d+=chr(k+96)
                    print('Sequence %d' % i)
                    print('\t Original:\n%s' % tmp_o)
                    print('\t Decoded:\n%s' % tmp_d)
                    print('Done!')
                log = "Epoch {}/{}  |  Batch Cost : {:.3f}  |  Train Accuracy : {:.3f}%  |  Test Accuracy : {:.3f}%  |  Time Elapsed : {:.3f}s"
                print(log.format(curr_epoch+1, num_epochs, train_cost, 100-(train_ler*100), 100-(test_ler*100), time.time() - start))
                t_summary = sess.run(merged, feed_dict=t_feed, options=run_options, run_metadata=run_metadata)
                test_writer.add_summary(t_summary, int(batch)+int(curr_epoch*num_batches_per_epoch))
                test_writer.add_run_metadata(run_metadata, 'step%03d' % int(batch+(curr_epoch*num_batches_per_epoch)))
                test_writer.flush()
            else:
                log = "Epoch {}/{}  |  Batch Cost : {:.3f}  |  Train Accuracy : {:.3f}%  |  Time Elapsed : {:.3f}s"
                print(log.format(curr_epoch+1, num_epochs, train_cost, 100-(train_ler*100), time.time() - start))
            save_path = saver.save(sess, savepath+'/model.ckpt')
            print(">>> Model saved succesfully")
        print('Total Training Time: '+str(time.time() - initstart)+'s')
