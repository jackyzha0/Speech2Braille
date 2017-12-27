# !/usr/local/bin/python
from __future__ import print_function

import time
print(time.strftime('[%H:%M:%S]'), 'Starting data loader... ')
print(time.strftime('[%H:%M:%S]'), 'Importing libraries... ')
import numpy as np
import string
print('[OK] numpy ')
import librosa.display as lib_disp
import librosa.feature as lib_feat
import librosa
print('[OK] librosa ')
import tensorflow as tf
print('[OK] tensorflow ')
import glob
print('[OK] glob ')
from PIL import Image
import scipy

batchsize = -1
num_mfccs = -1
num_classes = -1

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

def jpg_image_to_array(image_path):
    img = Image.open(image_path).convert('L')  # convert image to 8-bit grayscale
    WIDTH, HEIGHT = img.size
    data = list(img.getdata()) # convert image data to a list of integers
    # convert that to 2D list (list of lists of integers)
    data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
    return np.expand_dims(data, axis=4)

def r_saveImg(arr):
    n=0
    for i in arr:
        im = scipy.misc.toimage(i)
        im.save('/home/jacky/2kx/Spyre/nsound_git/r_img/'+str(n)+'.jpg')
        n+=1
def saveImg(arr):
    n=0
    for i in arr:
        im = scipy.misc.toimage(i)
        im.save('/home/jacky/2kx/Spyre/nsound_git/img/'+str(n)+'.jpg')
        n+=1

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
    x, sample_rate = librosa.load(rawsnd)
    s_tft = np.abs(librosa.stft(x))
    ft = lib_feat.mfcc(y=x, sr=sample_rate, n_mfcc=num).T
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
                    if (".wav" in __file):
                        raw_audio.append(__file)
                    if (".TXT" in __file):
                        with open(__file) as f:
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
    for j in index:
        #tmp = next_Data(patharr[j])
        tmp = next_Data(patharr[j])
        minibatch.append(np.array(tmp[0]))
        #print(time.strftime('[%H:%M:%S]'), 'Passed input tensor',batchsize)
        #print(time.strftime('[%H:%M:%S]'), 'Passed input tensor',np.asarray(np.array(tmp[0])).shape)
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

def fake_data(num_examples, num_features, num_labels, min_size = 10, max_size=100):
    np.random.seed(0)
    # Generating different timesteps for each fake data
    timesteps = np.random.randint(min_size, max_size, (num_examples,))
    # Generating random input
    inputs = np.asarray([np.random.randn(t, num_features).astype(np.float32) for t in timesteps])
    #inputs = np.asarray([np.ones((t, num_features))*255 for t in timesteps])

    # Generating random label, the size must be less or equal than timestep in order to achieve the end of the lattice in max timestep
    labels = np.asarray([np.random.randint(0, num_labels, np.random.randint(1, inputs[i].shape[0], (1,))).astype(np.int64) for i, _ in enumerate(timesteps)])
    r_saveImg(inputs)
    return inputs,labels
