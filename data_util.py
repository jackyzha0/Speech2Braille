# !/usr/local/bin/python
from __future__ import print_function

import time
print(time.strftime('[%H:%M:%S]'), 'Starting data loader... ')
print(time.strftime('[%H:%M:%S]'), 'Importing libraries... ')
import numpy as np
print('[OK] numpy ')
import librosa.display as lib_disp
import librosa.feature as lib_feat
import librosa
print('[OK] librosa ')
import tensorflow as tf
print('[OK] tensorflow ')
import glob
print('[OK] glob ')


sample_rate = 16000
batchsize = -1
num_mfccs = -1
num_classes = -1
path_phonetable = '/home/jacky/2kx/Spyre/git/phon_table.txt'

print(time.strftime('[%H:%M:%S]'), 'Constructing Phone Conversion Table...')
with open(path_phonetable) as f:
    phn_lookup = []
    for line in f:
        tmp_phn = line.split("\n")[0]
        phn_lookup.append(tmp_phn)
    print(phn_lookup)
    print(len(phn_lookup))
print(time.strftime('[%H:%M:%S]'),'Loaded phone table')

print(time.strftime('[%H:%M:%S]'), 'Loading helper functions...')

def setParams(_batchsize, _num_mfccs, _num_classes):
    """Set Training Parameters
    Args:+28
        _batchsize: size of mini batches
        _num_mfccs: number of mel-spectrum ceptral coefficients to compute
        _num_classes: total output types - 1
        _max_timesteps: max timesteps for minibatch/enveloping
        _timesteplen: max length of timesteps for minibatch/enveloping
        """
    global batchsize
    global num_mfccs
    global num_classes
    batchsize = _batchsize
    num_mfccs = _num_mfccs
    num_classes = _num_classes

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
    x, _ = librosa.load(rawsnd)
    s_tft = np.abs(librosa.stft(x))
    ft = lib_feat.mfcc(y=x, sr=sample_rate, n_mfcc=num)
    ft = np.append(ft,lib_feat.chroma_stft(S=s_tft, sr=sample_rate),axis=0)
    ft = np.append(ft,lib_feat.spectral_contrast(S=s_tft,sr=sample_rate),axis=0)
    ft = np.append(ft,lib_feat.tonnetz(y=librosa.effects.harmonic(x), sr=sample_rate),axis=0)
    ft = np.append(ft,lib_feat.zero_crossing_rate(y=x),axis=0)
    ft = np.append(ft,lib_feat.spectral_rolloff(y=x,sr=sample_rate),axis=0)
    ft = np.append(ft,lib_feat.spectral_centroid(y=x,sr=sample_rate),axis=0)
    ft = np.swapaxes(ft,0,1)
    print(ft.shape)
    return (ft)

def phn_to_int(inp):
    return phn_lookup.index(inp)

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
                    if (".wav" in __file):
                        raw_audio.append(__file)
                    if (".PHN" in __file):
                        with open(__file) as f:
                            tmp_phn_file = []
                            for line in f:
                                tmp_phn = line.split(" ")
                                tmp_phn[2] = phn_to_int(tmp_phn[2][:-1])
                                tmp_phn_file.append(tmp_phn)
                        print(time.strftime('[%H:%M:%S]'),'Phone file',__file[35:],'loaded, size',len(tmp_phn_file))
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
        print(time.strftime('[%H:%M:%S]'), 'Succesfully loaded data set of size',len(raw_audio))
        return raw_audio,phonemes,words,text,ind

def next_Data(path):
    """Returns array of features for training
    Args:
        path: path to audio file to compute
    Returns:
        features = rank 2 tensor of maxsize * num_features+28
    """
    featurearr = []
    ftrtmp=features(path, num_mfccs)
    featurearr.append(ftrtmp)
    return featurearr

def next_miniBatch(index,patharr):
    """Returns array of size batchsize with features for training
    Args:
        index: current position in training
    Returns:
        features = rank 3 tensor of batchsize * maxsize * num_features+28
    """
    minibatch = []
    for j in range(0,batchsize):
        tmp = next_Data(patharr[index+j])
        minibatch.append(np.array(tmp[0]))
        print(time.strftime('[%H:%M:%S]'), 'Passed input tensor with rank...',np.array(tmp[0]).shape,j+1,'/',batchsize)
    minibatch = np.array(minibatch)
    #minibatch = np.swapaxes(minibatch,2,1)
    print(time.strftime('[%H:%M:%S]'), 'Succesfully loaded minibatch of rank',minibatch.shape)
    return minibatch
def next_target_miniBatch(index,patharr):
    minibatch = []
    for j in range(0,batchsize):
        tmp = patharr[index+j]
        tmp_k = []
        for k in range(0,len(tmp)):
            tmp_k.append(int(tmp[k][2]))
        minibatch.append(np.array(tmp_k))
        print(time.strftime('[%H:%M:%S]'), 'Passed target tensor of rank',np.array(tmp_k).ndim,j+1,'/',batchsize)
    return np.asarray(minibatch)
