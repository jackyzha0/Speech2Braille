# !/usr/local/bin/python
import subprocess
import tensorflow as tf
import os
import numpy as np
import glob
import sys,argparse
#import braille_util
import time
start_time = time.time()
#PARAMS#
num_mfccs = 13

__file = sys.argv[1]
supress = sys.argv[2]
subprocess.check_call(["./denoise.sh", __file, supress])

print("Getting file directory")

def features(rawsnd,num):
    import librosa
    import librosa.feature as lib_feat
    x, sample_rate = librosa.load(rawsnd, sr=16000)
    #s_tft = np.abs(librosa.stft(x))
    ft = lib_feat.mfcc(y=x, sr=sample_rate, n_mfcc=num, n_fft=int(sample_rate*0.025), hop_length=int(sample_rate*0.010)).T
    #ft = lib_feat.melspectrogram(y=x, sr=sample_rate, n_fft=1024, hop_length=256, power=1.0).T
    t_ft = ft
    ft = np.append(ft,lib_feat.delta(t_ft),axis=1)
    ft = np.append(ft,lib_feat.delta(t_ft,order=2),axis=1)
    ft /= np.max(np.abs(ft),axis=0)
    #ft += np.random.normal(scale=0.5,size=ft.shape)
    #import matplotlib.pyplot as plt
    #import librosa.display
    #plt.figure(figsize=(12,4))
    #librosa.display.specshow(ft, sr=sample_rate, x_axis='time', y_axis='mel')
    #plt.colorbar(format='%+02.0f dB')
    #plt.tight_layout()
    #plt.show()
    return (ft)

def decode_to_chars(test_targets):
    tmp_o = ""
    for q in test_targets:
        if q==0:
            tmp_o+=" "
        else:
            tmp_o+=chr(q+96)
    return tmp_o
print('>>>'+str(time.strftime('[%H:%M:%S]'))+' '+"Calculating Features")
dat=features('_'+str(__file),num_mfccs)
#print(dat.shape)

savepath = 'best_chkpt'
graph_dir = "best_chkpt/model.meta"

print('>>>'+str(time.strftime('[%H:%M:%S]'))+' '+"Starting Tensorflow Session")

sess=tf.Session()

print('>>>'+str(time.strftime('[%H:%M:%S]'))+' '+"Importing Graph Metadata")

saver = tf.train.import_meta_graph(graph_dir)

print('>>>'+str(time.strftime('[%H:%M:%S]'))+' '+"Restoring Model")

saver.restore(sess, savepath+'/model')

print('>>>'+str(time.strftime('[%H:%M:%S]'))+' '+"Defining operations")

graph = tf.get_default_graph()
op = sess.graph.get_operations()
inp = graph.get_tensor_by_name("input/Placeholder:0")
seq = graph.get_tensor_by_name("inputLength/Placeholder:0")
feed_dict = {inp: [dat],seq: [len(dat)]}

out = graph.get_tensor_by_name("decoder/CTCGreedyDecoder:0")
out1 = graph.get_tensor_by_name("decoder/CTCGreedyDecoder:1")
out2 = graph.get_tensor_by_name("decoder/CTCGreedyDecoder:2")
out3 = graph.get_tensor_by_name("decoder/CTCGreedyDecoder:3")
print('>>>'+str(time.strftime('[%H:%M:%S]'))+' '+"Running Outputs")
z = sess.run((out,out1,out2,out3),feed_dict)
d = decode_to_chars(z[1])
print('>>>'+str(time.strftime('[%H:%M:%S]'))+' '+" > > > Output:",d)
#seq = braille_util.seq2braille(d)
#braille_util.disp(seq,0.1)
print("Execution time: %s seconds" % (time.time() - start_time))
