# !/usr/local/bin/python
import os
import glob
import sys,argparse
import braille_util
import time

import RPi.GPIO as GPIO
GPIO.setup(17,GPIO.OUT,initial=0)

start_time = time.time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
bcolors =  {
        "red":   '\033[95m',
        "blue":  '\033[94m',
        "green":        '\033[92m',
        "yellow":       '\033[93m',
        "fail":   '\033[91m',
        "end":   '\033[0m'
        }

def color_print(my_str, color="yellow"):
        print(bcolors[color]  + my_str  + bcolors["end"])

print("\033c")
color_print(str(time.strftime('[%H:%M:%S]'))+' '+"Preloading Modules into memory ... ")
import subprocess
import tensorflow as tf
import numpy as np

#PARAMS#
num_mfccs = 13
__file = "_dir/tmp.wav"
supress = 0.24
savepath = 'best_chkpt'
graph_dir = "best_chkpt/model.meta"

color_print(str(time.strftime('[%H:%M:%S]'))+' '+"Starting Tensorflow Session ...")

sess=tf.Session()

color_print(str(time.strftime('[%H:%M:%S]'))+' '+"Importing Graph Metadata, this may take a while ...")

saver = tf.train.import_meta_graph(graph_dir)

color_print(str(time.strftime('[%H:%M:%S]'))+' '+"Restoring Model ...")

saver.restore(sess, savepath+'/model')

color_print(str(time.strftime('[%H:%M:%S]'))+' '+"Defining operations ...")

graph = tf.get_default_graph()
op = sess.graph.get_operations()
inp = graph.get_tensor_by_name("input/Placeholder:0")
seq = graph.get_tensor_by_name("inputLength/Placeholder:0")


out = graph.get_tensor_by_name("decoder/CTCGreedyDecoder:0")
out1 = graph.get_tensor_by_name("decoder/CTCGreedyDecoder:1")
out2 = graph.get_tensor_by_name("decoder/CTCGreedyDecoder:2")
out3 = graph.get_tensor_by_name("decoder/CTCGreedyDecoder:3")


#color_print("\n\nWaiting for audio...")

def features(rawsnd,num):    
    import librosa
    import librosa.feature as lib_feat
    x, sample_rate = librosa.load(rawsnd, sr=16000)
    ft = lib_feat.mfcc(y=x, sr=sample_rate, n_mfcc=num, n_fft=int(sample_rate*0.025), hop_length=int(sample_rate*0.010))
    ft[0] = lib_feat.rmse(y=x, hop_length=int(0.010*sample_rate),n_fft=int(0.025*sample_rate))
    deltas = librosa.feature.delta(ft)
    ft_plus_deltas = np.vstack([ft,deltas])
    ft_plus_deltas /= np.max(np.abs(ft_plus_deltas),axis=0)
    return (ft_plus_deltas.T)

def decode_to_chars(test_targets):
    tmp_o = ""
    for q in test_targets:
        if q==0:
            tmp_o+=" "
        else:
            tmp_o+=chr(q+96)
    return tmp_o

def process(f,i,s):
    print("...Denoising  [size=%s]" %os.path.getsize(f))
    #subprocess.check_call(["./denoise.sh", f, str(supress)])
    print("...Calculating Features")
    dat=features(str(f),num_mfccs)
    feed_dict = {i:[dat],s:[len(dat)]}
    print("...Speech Recognizing")
    z = sess.run((out,out1,out2,out3),feed_dict)
    d = decode_to_chars(z[1])
    print("\n\nRecognized Speech: " + bcolors["green"]+str(d)+ bcolors["end"])
    seq2 = braille_util.seq2braille(d)
    braille_util.disp(seq2,0.1)
    #print("Execution time: %s seconds" % (time.time() - start_time))

process(sys.argv[1],inp,seq)
