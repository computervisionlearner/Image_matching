#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 11:15:41 2018

@author: sw
"""

import numpy as np
#import cv2
import time

import dataset
import tensorflow as tf
import sys
from datetime import datetime
from sklearn.metrics import roc_curve,auc
from matplotlib import pyplot as plt


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', '500', 'batch size, default: 64')
tf.flags.DEFINE_string('model', 'pretrained/model.pb', 'model path (.pb)')
tf.flags.DEFINE_integer('image_size', '64', 'image size, default: 64')
tf.flags.DEFINE_string('dataset_dir', '/home/sw/Documents/qd_fang1_9_2/field_2ch.npz', 'data path (.pb)')


def view_bar(message, num, total):
  rate = num / total
  rate_num = int(rate * 40)
  rate_nums = np.ceil(rate * 100)
  r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
  sys.stdout.write(r)
  sys.stdout.flush()
    
def draw_roc(outputs,labels):
  fpr,tpr,thresh = roc_curve(labels,outputs)
  roc_auc = auc(fpr,tpr)
  plt.plot(fpr, tpr, lw=1, label='AUC = %0.4f' %  roc_auc)
  plt.xlim([0, 0.2])  
  plt.ylim([0.6, 1])  
  plt.xlabel('False Positive Rate')  
  plt.ylabel('True Positive Rate') 
  plt.title('ROC curve')  
  plt.legend(loc="lower right")  
  plt.savefig('roc.png')
  return fpr, tpr

def get_fpr95(fpr, tpr):
  index = np.argmin(np.abs(tpr-0.95))
  return fpr[index]
  
if __name__ == '__main__':
    
  test_data = dataset.read_data_sets(dataset_dir = FLAGS.dataset_dir)
  graph = tf.Graph()
  with graph.as_default():
    with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(model_file.read())
      tf.import_graph_def(graph_def,name='')
  predicts = []
  labels = []
  with tf.Session(graph=graph) as sess:
    output = sess.graph.get_tensor_by_name('output:0')
    steps_per_epoch = test_data.num_examples // FLAGS.batch_size
    start_time = time.time()
    for i in range(steps_per_epoch):
      images_feed, labels_feed = test_data.next_batch(FLAGS.batch_size,shuffle = False)
      images_feed = np.concatenate(np.split(images_feed,2,axis=3),axis=2)
      predict = sess.run(output, {'input_image:0':images_feed})#numpy file fake opt      
      predicts.extend(predict)
      labels.extend(labels_feed)
      view_bar('processing:', i, steps_per_epoch)

    fpr, tpr = draw_roc(predicts,labels)
    fpr95 = get_fpr95(fpr, tpr)
    duration = time.time() - start_time

    print('fpr95 = %.4f , cost time = %.3f (sec)'%(fpr95, duration))