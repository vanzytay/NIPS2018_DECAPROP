#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ..lib.stat import *
from ..lib.seq_op import *
from ..lib.nn import *
from ..lib.att_op import *
from ..lib.loss import *
from ..lib.compose_op import *
from ..lib.opt import *

import gzip
import json
from tqdm import tqdm
import numpy as np
import random
from collections import Counter
import numpy as np
import operator
import timeit
import time
import datetime


class BaseModel(object):
    ''' Base Model for TensorFlow Experiments

    This class is meant to be extended.
    '''
    def __init__(self, vocab_size, args,
                    char_size=None,
                    sdmax=None, f_len=None):
        self.vocab_size = vocab_size
        self.char_size = char_size
        self.graph = tf.Graph()
        self.args = args
        # Max doc length for flat models
        self.sdmax = sdmax
        if(f_len is None):
            self.f_len = 0
        else:
            self.f_len = f_len
        self.imap = {}
        # Default Settings

        if(self.args.init_type=='uniform'):
            self.init = tf.random_uniform_initializer(
                                    minval=-self.args.init,
                                    maxval=self.args.init)
        elif(self.args.init_type=='xavier'):
            self.init = tf.contrib.layers.xavier_initializer()
        elif(self.args.init_type=='normal'):
            self.init = tf.random_normal_initializer(0.0,
                                    self.args.init)
        elif(self.args.init_type=='tnormal'):
            self.init = tf.truncated_normal_initializer(0.0,
                                    self.args.init)
        if(self.args.init_type=='uniform'):
            self.rnn_init = tf.random_uniform_initializer(
                                    minval=-self.args.init,
                                    maxval=self.args.init)
        elif(self.args.rnn_init_type=='xavier'):
            self.rnn_init = tf.contrib.layers.xavier_initializer()
        elif(self.args.rnn_init_type=='normal'):
            self.rnn_init = tf.random_normal_initializer(0.0,
                                    self.args.rnn_init)
        elif(self.args.rnn_init_type=='orth'):
            self.rnn_init = tf.orthogonal_initializer()

        self.l2_reg = tf.contrib.layers.l2_regularizer(self.args.l2_reg)
        self.activation = tf.nn.relu

    def register_index_map(self, idx, target):
        self.imap[target] = idx

    def _build_base_placeholders(self):
        ''' Builds base placeholders
        '''
        self.true = tf.constant(True, dtype=tf.bool)
        self.false = tf.constant(False, dtype=tf.bool)

        with tf.name_scope('dropout'):
            self.dropout = tf.placeholder(tf.float32,
                                name='dropout')
            tf.add_to_collection('dropout', self.dropout)

        with tf.name_scope('emb_dropout'):
            self.emb_dropout = tf.placeholder(tf.float32,
                                name='emb_dropout')
            tf.add_to_collection('emb_dropout', self.emb_dropout)

        with tf.name_scope('hdropout'):
            self.hdropout = tf.placeholder(tf.float32,
                                name='hdropout')
            tf.add_to_collection('hdropout', self.hdropout)

        with tf.name_scope('hdropout'):
            self.hdropout = tf.placeholder(tf.float32,
                                name='hdropout')
            tf.add_to_collection('hdropout', self.hdropout)

        if(self.args.pretrained==1):
            self.emb_placeholder = tf.placeholder(tf.float32,
                        [self.vocab_size, self.args.emb_size])
        with tf.name_scope("labels"):
            # one hot encoding?
            self.label = tf.placeholder(tf.int32, shape=[None, 3],
                                                name='labels')

        self.learn_rate = tf.placeholder(tf.float32,
                                name='learn_rate')

    def _get_learn_rate(self):

        if(self.args.dev_lr>0):
            print("Using Dev-LR")
            print("-----------------")
            return self.learn_rate

        if(self.args.decay_steps>0):
            lr = tf.train.exponential_decay(self.args.lr,
                          self.global_step,
                          self.args.decay_steps,
                           self.args.decay_lr,
                           staircase=self.args.decay_stairs)
        elif(self.args.decay_lr>0 and self.args.decay_epoch>0):
            decay_epoch = self.args.decay_epoch
            lr = tf.train.exponential_decay(self.args.lr,
                          self.global_step,
                          decay_epoch * self.args.batch_size,
                           self.args.decay_lr, staircase=True)
        else:
            lr = self.args.lr
        return lr

    def _get_optimizer(self, lr):
        """ Get optimizer only
        """
        #global_step = self.global_step

        with tf.name_scope('optimizer'):
            # global_step = tf.Variable(0, trainable=False)
            if(self.args.opt=='SGD'):
                opt = tf.train.GradientDescentOptimizer(
                                learning_rate=lr)
            elif(self.args.opt=='Adam'):
                opt = tf.train.AdamOptimizer(
                                learning_rate=lr)
            elif(self.args.opt=='Adadelta'):
                opt = tf.train.AdadeltaOptimizer(
                                learning_rate=lr, epsilon=1E-6)
            elif(self.args.opt=='Adagrad'):
                opt = tf.train.AdagradOptimizer(
                                learning_rate=lr,
                                initial_accumulator_value=0.1)
            elif(self.args.opt=='RMS'):
                opt = tf.train.RMSPropOptimizer(
                                learning_rate=lr)
            elif(self.args.opt=='Moment'):
                opt = tf.train.MomentumOptimizer(
                                lr, 0.9)
            elif(self.args.opt=='Adamax'):
                # this is found in lib.opt
                # opt = AdamaxOptimizer(lr)
                opt = tf.keras.optimizers.Adamax(lr=lr)
        return opt

    def _build_train_ops(self, grads, opt, global_step):
        ''' General Optimization Function
        '''
        # Define loss and optimizer

        # lr = tf.train.exponential_decay(self.args.learn_rate, global_step,
        #                    100000, 0.96, staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print("Update ops={}".format(update_ops))
        with tf.control_dependencies(update_ops):
            with tf.variable_scope("train"):
                # tf.summary.scalar("cost_function", self.cost)

                def ClipIfNotNone(grad):
                    if grad is None:
                        return grad
                    # grad = tf.clip_by_value(grad, -10, 10, name=None)
                    # grad = tf.clip_by_global_norm(grad, self.args.clip_norm)
                    grad = tf.clip_by_norm(grad, self.args.clip_norm)
                    return grad

                if(self.args.clip_norm>0):
                    print("Clipping norm")
                    clipped_gradients = [(ClipIfNotNone(grad), var) \
                                            for grad, var in grads]
                else:
                    clipped_gradients = [(grad,var) for grad,var in grads]

                train_op = opt.apply_gradients(clipped_gradients,
                                global_step=global_step)
                # self.train_op = optimizer

                # self.train_op = opt.minimize(self.cost)
                # self.tb_summary = tf.summary.merge_all(
                #                     key=tf.GraphKeys.SUMMARIES)

            # clipped_gradients = [x for x in clipped_gradients if x[0] is not None]
            # grads = [x for x in grads if x[0] is not None]
            return train_op
