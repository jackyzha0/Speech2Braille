# !/usr/local/bin/python
from __future__ import print_function

print('Importing libraries... ')

import numpy as np
import librosa.display as lib_disp
import librosa.feature as lib_feat
import tensorflow as tf
import glob
import librosa

sample_rate = 32000
batchsize = -1
num_mfccs = -1
num_classes = -1
max_timestepsize = -1
max_timesteplen = -1

def setParams(_batchsize, _num_mfccs, _num_classes, _max_timesteps, _timesteplen):
    """Set Training Parameters
    Args:
        _batchsize: size of mini batches
        _num_mfccs: number of mel-spectrum ceptral coefficients to compute
        _num_classes: total output types - 1
        _max_timesteps: max timesteps for minibatch/enveloping
        _timesteplen: max length of timesteps for minibatch/enveloping
        """
    global batchsize
    global num_mfccs
    global num_classes
    global max_timestepsize
    global max_timesteplen
    batchsize = _batchsize
    num_mfccs = _num_mfccs
    num_classes = _num_classes
    max_timestepsize = _max_timesteps
    max_timesteplen = _timesteplen

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x. For handling one-hot vector
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

def features(rawsnd, num) :
    """Compute num amount of audio features of a sound
    Args:
        rawsnd : array with string paths to .wav files
        num : numbers of mfccs to compute
    Returns:
        Return a (num+28,max_stepsize*32)-dimensional Tensorflow feature vector, and length in
        *num Amount of Mel Frequency Ceptral Coefficients
        *12 Chromagrams
        *7 bands of Spectral Contrast
        *6 bands of Tonal Centroid Features (Tonnetz)
        *Zero Crossing Rate
        *Spectral Rolloff
        *Spectral Centroid"""
    x, _ = librosa.load(rawsnd,sr=sample_rate, duration=max_timesteplen*max_timestepsize)
    s_tft = np.abs(librosa.stft(x))
    ft = lib_feat.mfcc(y=x, sr=sample_rate, n_mfcc=num)
    ft = np.append(ft,lib_feat.chroma_stft(S=s_tft, sr=sample_rate),axis=0)
    ft = np.append(ft,lib_feat.spectral_contrast(S=s_tft,sr=sample_rate),axis=0)
    ft = np.append(ft,lib_feat.tonnetz(y=librosa.effects.harmonic(x), sr=sample_rate),axis=0)
    ft = np.append(ft,lib_feat.zero_crossing_rate(y=x),axis=0)
    ft = np.append(ft,lib_feat.spectral_rolloff(y=x,sr=sample_rate),axis=0)
    ft = np.append(ft,lib_feat.spectral_centroid(y=x,sr=sample_rate),axis=0)
    z = np.zeros((num+12+7+6+3,(max_timestepsize*((sample_rate/1000)*2))-ft.shape[1]))
    ft = np.concatenate((ft,z),axis=1)
    #print(ft[13].astype(np.float16))
    return (ft)

def load_dir(fp):
    """Load raw paths data into arrays
    Args:
        fp : string path to data
    Returns:
        Returns array of loaded files
        loaded[0] = sound
        loaded[1] = phonemes
        loaded[2] = words
        loaded[3] = text"""
    with tf.name_scope('raw_data'):
        ind = 0
        raw_audio = []
        phonemes = []
        words = []
        text = []
        for __file in glob.iglob(fp + '/*.*'):
                if not ("SA" in __file):
                    ind+=1
                    if (ind%500==0):
                        print(".", end='')
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

        return raw_audio,phonemes,words,text,ind

def pad_seq():
    return 0

def next_Data(index,path):
    """Returns array of size batchsize with features and labels for training
    Args:
        index: current position
    Returns:
        features = rank 3 tensor of batchsize * maxsize * num_features
        """
    featurearr = []
    featurearr.append(features(path, num_mfccs))
    return featurearr

def next_miniBatch(index,patharr):
    minibatch = np.array()
    for j in range(0,batchsize):
        print(index+j,'/',len(patharr))
        np.append(minibatch,next_Data(index,patharr[index+j]),axis=0)
        print(minibatch.shape)
