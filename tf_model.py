#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
import librosa
import numpy as np
import librosa.display as lib_disp
import librosa.feature as lib_feat
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#TODO
"""
-Finish Batching
-Add Classes
-Add blank symbol
-Implement CTC
-Complete namescopes for Tensorflow
-Implement Connectionist Temporal Classification as
an optimization function
-Add HMM / N-Gram Language Model
"""

#PARAMS
FLAGS = None
path = '/home/jacky/Desktop/Spyre/__data/TIMIT/*/*/*'
num_mfccs = 13
batchsize = 10
preprocess = 1
learning_rate = 0.001
window_cutoff = 5
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
    1 second = 32 units
    Compute and return a (num+28,(window_cutoff*32))-dimensional Tensorflow feature vector, and length in
        num Amount of Mel Frequency Ceptral Coefficients
        12 Chromagrams
        7 bands of Spectral Contrast
        6 bands of Tonal Centroid Features (Tonnetz)
        Zero Crossing Rate
        Spectral Rolloff
        Spectral Centroid
    """
    x, sample_rate = librosa.load(rawsnd,sr=16000, duration=window_cutoff)
    s_tft = np.abs(librosa.stft(x))
    ft = lib_feat.mfcc(y=x, sr=sample_rate, n_mfcc=num)
    ft = np.append(ft,lib_feat.chroma_stft(S=s_tft, sr=sample_rate),axis=0)
    ft = np.append(ft,lib_feat.spectral_contrast(S=s_tft,sr=sample_rate),axis=0)
    ft = np.append(ft,lib_feat.tonnetz(y=librosa.effects.harmonic(x), sr=sample_rate),axis=0)
    ft = np.append(ft,lib_feat.zero_crossing_rate(y=x),axis=0)
    ft = np.append(ft,lib_feat.spectral_rolloff(y=x,sr=sample_rate),axis=0)
    ft = np.append(ft,lib_feat.spectral_centroid(y=x,sr=sample_rate),axis=0)
    z = np.zeros((num+12+7+6+3,(window_cutoff*32)-ft.shape[1]))
    ft = np.concatenate((ft,z),axis=1)
    return (ft)

def preprocess(rawsnd) :
    """
    If preprocess == 1, add additional white noise and
    random sample from Urban8K
    """

def BLSTM():
    """
    """

def batch(data,truth_label,index):
    """
    Outputs feed dictionary for training
    Form of:
    feed_dict = {<placeholder>: <tensor values>}
    """
    print('Constructing input dictionary of size %d' % len(data))
    feature_vec = []
    label_vec = []
    largest = 0;
    if index > (len(data)-batchsize)-(len(data)%batchsize):
        raise ValueError('Out of Bounds')
    else:
        for i in range(index,index+batchsize):
            feature_vec.append(features(data[i],num_mfccs))
            for f in truth_label[i]:
                label_vec.append(f[1])
    #return dict(zip(feature_vec, label_vec))
    return 0

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

init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    print('Loading data', end='')
    loaded = load_dir(path)
    raw_sound = loaded[0]
    print('Done!')
    dict = batch(raw_sound,loaded[1],0)
    #print(dict)
    sess.run(init)
    print('Starting Tensorflow session')
    features_placeholder = tf.placeholder(tf.float32, shape=(None, num_mfccs+4))
    labels_placeholder = tf.placeholder(tf.string, shape=(batchsize))

    #train
    #eval
