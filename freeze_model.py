# !/usr/local/bin/python
import os, argparse
import tensorflow as tf
import time
print('[OK] time')

# print('[OK]',time.strftime('[%H:%M:%S]'),'Directory')
# saver = tf.train.import_meta_graph(path+'/model.ckpt.meta', clear_devices=True)
# print('[OK]',time.strftime('[%H:%M:%S]'),'Metagraph Import')
# graph = tf.get_default_graph()
# input_graph_def = graph.as_graph_def()
# sess = tf.Session()
# saver.restore(sess, path+'/model.ckpt')
# output_node_names="decoder/CTCGreedyDecoder"
# print([n.name for n in tf.get_default_graph().as_graph_def().node])
# output_graph_def = tf.graph_util.convert_variables_to_constants(
#             sess, # The session
#             input_graph_def, # input_graph_def is useful for retrieving the nodes
#             output_node_names.split(",")
# )
# output_graph="/test.pb"

def freeze(model_dir,output_node_names):
    assert tf.gfile.Exists(model_dir)
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"
    clear_devices = True
    print(input_checkpoint,absolute_model_dir,output_graph)
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def

if __name__ == '__main__':
    model_dir = "best_chkpt"
    output_node_names="decoder/CTCGreedyDecoder"
    freeze(model_dir,output_node_names)
