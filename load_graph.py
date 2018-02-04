# !/usr/local/bin/python
import tensorflow as tf
import os
import numpy as np
import glob
import sys,argparse
import librosa
import librosa.feature as lib_feat
#PARAMS#
num_mfccs = 13
graph_dir = "best_chkpt/frozen_model.pb"

def features(rawsnd,num):
    x, sample_rate = librosa.load(rawsnd, sr=16000)
    #s_tft = np.abs(librosa.stft(x))
    ft = lib_feat.mfcc(y=x, sr=sample_rate, n_mfcc=num, n_fft=int(sample_rate*0.025), hop_length=int(sample_rate*0.010)).T
    #ft = lib_feat.melspectrogram(y=x, sr=sample_rate, n_fft=1024, hop_length=256, power=1.0).T
    t_ft = ft
    ft = np.append(ft,lib_feat.delta(t_ft),axis=1)
    ft = np.append(ft,lib_feat.delta(t_ft,order=2),axis=1)
    ft /= np.max(np.abs(ft),axis=0)
    return (ft)

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        for node in graph_def.node:
            print(node.op)
            if node.op == 'RefEnter':
                node.op = 'Enter'
                for index in xrange(len(node.input)):
                    node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best_chkpt/frozen_model.pb", type=str, help="Frozen model file to import")
    parser.add_argument("--wav", default="eg.wav", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.model)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions

    # We access the input and output nodes
    x = graph.get_tensor_by_name('prefix/Placeholder/inputs_placeholder:0')
    y = graph.get_tensor_by_name('prefix/Accuracy/predictions:0')

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants
        y_out = sess.run(y, feed_dict={
            x: [[3, 5, 7, 4, 5, 1, 1, 1, 1, 1]] # < 45
        })
        # I taught a neural net to recognise when a sum of numbers is bigger than 45
        # it should return False in this case
        print(y_out) # [[ False ]] Yay, it works!
