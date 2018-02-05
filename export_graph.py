#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 10:56:06 2018

@author: sw
"""

import tensorflow as tf

import model


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint_dir', 'ckpt', 'checkpoints directory path')
tf.flags.DEFINE_string('model', 'model.pb', 'image matching model name, default: model.pb')

tf.flags.DEFINE_integer('image_size', '64', 'image size, default: 256')

batch_size = 500



def export_graph(model_name):
  graph = tf.Graph()

  with graph.as_default():
    

    input_image = tf.placeholder(tf.float32, shape=[batch_size, FLAGS.image_size, 2 * FLAGS.image_size, 1], name='input_image')
   
    features1,features2,_ = model.get_features(input_image, reuse = False)
    logits, _, _ = model.get_logits(features1,features2,keep_prob=1)
    output = tf.nn.softmax(logits)[:,1]  
    output_image = tf.identity(output, name='output')
    restore_saver = tf.train.Saver()


  with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    restore_saver.restore(sess, latest_ckpt)
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [output_image.op.name])

    tf.train.write_graph(output_graph_def, 'pretrained', model_name, as_text=False)


def main(unused_argv):
  print('Export XtoY model...')
  export_graph(FLAGS.model)


if __name__ == '__main__':
  tf.app.run()