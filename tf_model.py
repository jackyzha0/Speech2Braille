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
import python_speech_features as feat
print('[OK] features ')
import glob
print('[OK] glob ')
from PIL import Image
print('[OK] PIL ')
import scipy
import scipy.io.wavfile as wav
print('[OK] scipy ')
import os
print('[OK] os ')
from tensorpack import *
print('[OK] tensorpack ')
import BN_LSTMCell
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

with tf.device("/cpu:0"):

    #PARAMS
    FLAGS = None
    #SPACE_TOKEN = '<space>'
    #SPACE_INDEX = 0
    #FIRST_INDEX = ord('a')-96-1  # 0 is reserved to space
    #test_path = '/home/jacky/2kx/Spyre/__data/LibriSpeech/test-clean/*/*'
    #path = '/home/jacky/2kx/Spyre/__data/LibriSpeech/train-clean-100/*/*'
    test_path = '/home/jacky/2kx/Spyre/__data/TIMIT/TEST/*/*'
    path = '/home/jacky/2kx/Spyre/__data/TIMIT/TRAIN/*/*'

    logImages = False

    # Network Params #
    with tf.variable_scope('main'):
        num_mfccs = 13
        num_classes = 28
        num_hidden = 128
        learning_rate = 1e-3
        momentum = 0.3
        num_layers = 1#2
        BLSTM = False
        BatchNorm = False#True
        LayerNorm = False
        input_noise = False
        opt_momentum = True
        opt_adam = False
        opt_adag = False
        opt_sgd = False
    ##############

    # Pickle Settings #
    pickle_path = './timit_pickle'
    repickle = False
    print(time.strftime('[%H:%M:%S]'), 'Checking for pickled data... ')
    ##############
    def _load_dir(fp):
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
            text = []
            for __file in glob.iglob(fp + '/*.*'):
                    ind+=1
                    if (".trans.txt" in __file):
                        with open(__file) as f:
                            _prefix = __file[:-10]
                            for line in f:
                                _id = line.split(' ')[0][-5:]
                                _line = line.split(' ', 1)[-1]
                                raw_audio.append(_prefix+_id+'.wav')
                                res = (list(_line[:-1].lower().translate(None, string.punctuation)))
                                tmp_res = []
                                for r in res:
                                    if r==' ':
                                        tmp_res.append(0)
                                    else:
                                        tmp_res.append(ord(r)-96)
                                text.append(tmp_res)
            print(time.strftime('[%H:%M:%S]'), 'Succesfully loaded data set of size',len(raw_audio))
            return raw_audio,text,len(raw_audio)
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
    num_epochs = 200
    batchsize = 32
    num_batches_per_epoch = int(num_examples/batchsize)
    ##############

    # Log Params #
    logs_path = 'totalsummary/logs/'+datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')+'_'+str(batchsize)+'_'+str(num_epochs)
    savepath = 'totalsummary/ckpt'
    RESTOREMODEL = False
    ##############

    print(time.strftime('[%H:%M:%S]'), 'Parsing testing directory... ')
    t_dr = load_dir(test_path)
    testsetsize = len(t_dr[0])
    testbatchsize = 32

    # Functions #
    def decode_to_chars(test_targets):
        tmp_o = ""
        for q in test_targets:
            if q==0:
                tmp_o+=" "
            else:
                tmp_o+=chr(q+96)
        return tmp_o
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
            #print(decode_to_chars(tmp_k))
        return np.asarray(minibatch)
    def next_Data(path,test):
        """Returns array of features for training
        Args:
            path: path to audio file to compute
        Returns:
            featurearr: rank 2 tensor of maxsize * num_features
        """
        #z = path.replace('/','').split("LibriSpeech")[1][:-5]
        z = path.replace('/','').split("TIMIT")[1][:-4]
        if test or repickle or not os.path.exists(pickle_path+'/'+z+'.npy'):
            featurearr = []
            ftrtmp=features(path, num_mfccs)
            featurearr.append(ftrtmp)
            np.save(pickle_path+'/'+z,featurearr)
            print(time.strftime('[%H:%M:%S]'), 'Pickle saved to',pickle_path+'/'+z[:-4])
        else:
            featurearr = np.load(pickle_path+'/'+z+'.npy')
            if input_noise:
                featurearr += np.random.normal(scale=0.6,size=featurearr.shape)
        #print(path)
        return featurearr
    def features(rawsnd, num) :
        """Compute num amount of audio features of a sound
        Args:
            rawsnd : array with string paths to .wav files
            num : numbers of mfccs to compute
        Returns:
            Return a num x max_stepsize*32 feature vector
        """
        (rate,sample) = wav.read(rawsnd)
        mfcc = feat.mfcc(sample,rate,winlen=0.025,winstep=0.01,numcep=num_mfccs,nfilt=26,preemph=0.95)
        #derivative = np.zeros(mfcc.shape)
        #for i in range(1, mfcc.shape[0] - 1):
        #    derivative[i, :] = mfcc[i + 1, :] - mfcc[i - 1, :]
        #mfcc_derivative = np.concatenate((mfcc, derivative), axis=1)
        #derivative2 = np.zeros(derivative.shape)
        #for i in range(1, derivative.shape[0] - 1):
        #    derivative2[i, :] = derivative[i + 1, :] - derivative[i - 1, :]
        #out = np.concatenate((mfcc, derivative, derivative2), axis=1)
        mfcc -= (np.mean(mfcc,axis=0) + 1e-8)
        #print(out.shape)
        return (mfcc - np.max(mfcc))/-np.ptp(mfcc)
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
        inputs = tf.placeholder(tf.float32, [None, None, num_mfccs])
        targets = tf.sparse_placeholder(tf.int32)
        seq_len = tf.placeholder(tf.int32, [None])
        def lstm_cell():
            return tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        stack = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)], state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
        shape = tf.shape(inputs)
        batch_s, max_timesteps = shape[0], shape[1]
        outputs = tf.reshape(outputs, [-1, num_hidden])
        W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[num_classes]))
        logits = tf.matmul(outputs, W) + b
        logits = tf.reshape(logits, [batch_s, -1, num_classes])
        logits = tf.transpose(logits, (1, 0, 2))
        loss = tf.nn.ctc_loss(targets, logits, seq_len, ignore_longer_outputs_than_inputs=True)
        cost = tf.reduce_mean(loss)
        tf.summary.scalar("cost", cost)
        train_optimizer = tf.train.AdamOptimizer(learning_rate,
                                           momentum).minimize(loss)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))
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
            index_list = range(0,datasetsize)
            for batch in range(num_batches_per_epoch):
                # Getting the index
                indexes = random.sample(index_list,batchsize)
                index_list = [x for x in index_list if x not in indexes]
                train_inputs = next_miniBatch(indexes,dr[0])
                train_targets = next_target_miniBatch(indexes,dr[1])
                #train_inputs,train_targets = fake_data(num_examples,num_mfccs,num_classes-1)
                newindex = [i % num_examples for i in range(batchsize)]
                random.shuffle(newindex)
                batch_train_inputs = train_inputs[newindex]
                # Padding input to max_time_step of this batch
                batch_train_inputs, batch_train_seq_len = pad_sequences(batch_train_inputs)
                # Converting to sparse representation so as to to feed SparseTensor input
                batch_train_targets = sparse_tuple_from(train_targets[newindex])
                feed = {inputs: batch_train_inputs,
                        targets: batch_train_targets,
                        seq_len: batch_train_seq_len
                #        ,training: True
                        }
                batch_cost, _, l = sess.run([cost, train_optimizer, ler], feed)
                train_cost += batch_cost*batchsize
                train_ler += l*batchsize
                print('['+str(curr_epoch)+']','  >>>',time.strftime('[%H:%M:%S]'), 'Batch',batch+1,'/',num_batches_per_epoch,'@Cost',batch_cost,'Time Elapsed',time.time()-t_time,'s')
                t_time=time.time()
                if (batch % 32 == 0):
                    summary = sess.run(merged, feed_dict=feed, options=run_options, run_metadata=run_metadata)
                    train_writer.add_summary(summary, int(batch)+int(curr_epoch*num_batches_per_epoch))
                    train_writer.flush()

            # Metrics mean
            train_cost /= num_examples
            train_ler /= num_examples
            #Testing
            print('>>>',time.strftime('[%H:%M:%S]'), 'Evaluating Test Accuracy...')
            t_index = random.sample(range(0, testsetsize), testbatchsize)
            test_inputs = next_miniBatch(t_index,t_dr[0],test=True)
            test_targets = next_target_miniBatch(t_index,t_dr[1])
            newindex = [i % testbatchsize for i in range(testbatchsize)]
            batch_test_inputs = test_inputs[newindex]
            batch_test_inputs, batch_test_seq_len = pad_sequences(batch_test_inputs)
            batch_test_targets = sparse_tuple_from(test_targets[newindex])
            t_feed = {inputs: batch_test_inputs,
                    targets: batch_test_targets,
                    seq_len: batch_test_seq_len
            #        ,training: False
                    }
            test_ler,d = sess.run((ler,decoded[0]), feed_dict=t_feed)
            dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=sess)
            for i, seq in enumerate(dense_decoded):
                seq = [s for s in seq if s != -1]
                tmp_o = decode_to_chars(test_targets[i])
                tmp_d = decode_to_chars(seq)
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
            save_path = saver.save(sess, savepath+'/model.ckpt')
            print(">>> Model saved succesfully")
        print('Total Training Time: '+str(time.time() - initstart)+'s')
