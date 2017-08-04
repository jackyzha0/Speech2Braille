#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
import librosa
import librosa.display as lib_disp
import librosa.feature as lib_feat
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#TODO
"""
Implement Connectionist Temporal Classification as
an optimization function
"""

#PARAMS
FLAGS = None
path = '/home/jacky/Desktop/Spyre/__data/TIMIT/*/*/*'
num_mfccs = 13
batchsize = 10
preprocess = 1
learning_rate = 0.001
#0th indice +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 3

def load_dir(fp):
    """
    Load raw data into arrays
    fp is a string path
    Returns array of loaded files
    loaded[0] = sound
    loaded[1] = phonemes
    loaded[2] = words
    loaded[3] = text
    """
    with tf.name_scope('raw_data'):
        ind = 0
        raw_audio = []
        phonemes = []
        words = []
        text = []
        for __file in glob.iglob(path + '/*.*'):
                if not ("SA" in __file):
                    ind+=1
                    if (ind%500==0):
                        print(".", end="")
                    if (".wav" in __file):
                        raw_audio.append(__file)
                    if (".PHN" in __file):
                        with open(__file) as f:
                            tmp_phn_file = []
                            for line in f:
                                tmp_phn = line.split(" ")
                                tmp_phn_file.append(tmp_phn)
                        phonemes.append(tmp_phn_file)
                    if (".WRD" in __file):
                        with open(__file) as f:
                            tmp_wrd_file = []
                            for line in f:
                                tmp_wrd = line.split(" ")
                                tmp_wrd_file.append(tmp_wrd)
                        words.append(tmp_wrd_file)
                    if (".TXT" in __file):
                        with open(__file) as f:
                            for line in f:
                                res = ''.join([i for i in line if not i.isdigit()])
                        text.append(res)

        return raw_audio,phonemes,words,text

def features(rawsnd, num) :
    """
    Compute and return a (n+4)-dimensional Tensorflow feature vector
        n Amount of Mel Frequency Ceptral Coefficients
        Zero Crossing Rate
        Spectral Rolloff
        Spectrall Contrast
        Spectral Centroid
    """
    x, sample_rate = librosa.load(rawsnd)
    mf = lib_feat.mfcc(y=x, sr=sample_rate, n_mfcc=num)
    zcr = lib_feat.zero_crossing_rate(y=x)
    srolloff = lib_feat.spectral_rolloff(y=x,sr=sample_rate)
    scontrast = lib_feat.spectral_contrast(y=x,sr=sample_rate)
    scentroid = lib_feat.spectral_centroid(y=x,sr=sample_rate)
    return (mf,zcr[0],srolloff[0],scontrast[0],scentroid[0])

def preprocess(rawsnd) :
    """
    If preprocess == 1, add additional white noise and
    random sample from Urban8K
    """

def BLSTM():
    """
    """

def fill_feed_dict(data,truth_label,index):
    """
    Outputs feed dictionary for training
    Form of:
    feed_dict = {<placeholder>: <tensor values>}
    """
    print('Constructing input dictionary of size %d' % len(data),end='')
    feature_vec = []
    label_vec = []
    if i > (len(data)-batchsize)-(len(data)%10):
        raise ValueError('Out of Bounds')
    else:
        for i in range(index,index+batchsize):
            feature_vec.append(features(data[i],num_mfccs))
            label_vec.append(label_vec[i])
    return dict(zip(feature_vec, label_vec))

def plot(spec):
    """
    Creates librosa plot given output of features(_,_)
    """
    plt.subplot(5,1,1)
    librosa.display.specshow(spec[0], x_axis='time')
    for i in range(2,6):
        plt.subplot(5,1,i)
        plt.plot(spec[i-1])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.37)
    plt.show()

def train():
    """
    Trains the model
    """

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()

    with tf.Session() as sess:
        print('Loading data', end='')
        loaded = load_dir(path)
        raw_sound = loaded[0]
        print('Done!')
        dict = fill_feed_dict(raw_sound,loaded[1],0)
        print(dict)

        features_placeholder = tf.placeholder(tf.float32, shape=(None, num_mfccs+4))
        labels_placeholder = tf.placeholder(tf.string, shape=(batch_size))

        #train
        #eval
