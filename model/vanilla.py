#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import gzip
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import random
from collections import Counter
import numpy as np
import operator
import tensorflow.contrib.slim as slim

from tylib.lib.seq_op import *
from tylib.lib.nn import *
from tylib.lib.att_op import *
from tylib.lib.loss import *
from tylib.lib.rnn import *
from tylib.lib.compose_op import *
from tylib.lib.cnn import *
from tylib.models.base_model import BaseModel
from tylib.lib.sim_op import *
from tylib.lib.stat import *
from tylib.lib.rec_op import *
from tylib.lib.enhanced import *
from tylib.lib.choice import *
from tylib.exp.multi_gpu import *
from tylib.lib.bimpm import *
from tylib.exp.utilities import *
from tylib.lib.func import *
from tylib.lib.pointer import *
from .decaprop import *


def _build_vanilla_model(self, doc_hidden, doc_len, query_hidden,
                            query_len, cmask, qmask):
    """ vanilla model that is experimentally customisable
    """

    if('LSTM' in self.args.rnn_type or 'GRU' in self.args.rnn_type):
        """ Add a LSTM or GRU layer
        """
        print("Adding LSTM/GRU layer of {} dims".format(
                        self.args.rnn_type))
        doc_hidden, _ = build_rnn(doc_hidden, doc_len,
                            rnn_type=self.args.rnn_type,
                            reuse=False,
                            rnn_dim=self.args.rnn_size,
                            dropout=self.args.dropout,
                            initializer=self.rnn_init,
                            use_cudnn=self.args.use_cudnn,
                            is_train=self.is_train)
        query_hidden, _ = build_rnn(query_hidden, query_len,
                            rnn_type=self.args.rnn_type,
                            reuse=True,
                            rnn_dim=self.args.rnn_size,
                            dropout=self.args.dropout,
                            initializer=self.init,
                            use_cudnn=self.args.use_cudnn,
                            is_train=self.is_train)

    _dim = doc_hidden.get_shape().as_list()[1]
    output = self.match_op(query_hidden, doc_hidden, query_len,
                        doc_len, '',
                        mask_a=self.qmask, mask_b=self.doc_mask)
    G = output

    if('INTRA' in self.args.rnn_type):
        output2, _, _, _, _ = co_attention(
                        output, output,
                        att_type='DOT',
                        pooling='MATRIX',
                        kernel_initializer=self.init,
                        dropout=None,
                        seq_lens=[doc_len, doc_len],
                        name='intra',
                        mask_a=self.doc_mask,
                        mask_b=self.doc_mask
                                )
        output = tf.concat([output, output2], 2)

    output_list = []

    _dim = output.get_shape().as_list()[2]

    """ Answer Pointer Layers.
    Generally, we can pass the output of doc_len to 2 encoder blocks.
    The first block is used to learn the start pointers. The output
    from the 1st block is then passed to learn the end pointer. softmax
    is applied to both for optimization.
    """

    if('MLP' in self.args.ptr_type):
        """ Simple MLP pointer layers
        """
        start_ptr = projection_layer(output, _dim,
                            name='ptr_start',
                            reuse=None,
                            activation=None,
                            initializer=self.init,
                            dropout=None, mode='None',
                            num_layers=1)
        end_ptr =  projection_layer(start_ptr, _dim,
                            name='ptr_end',
                            reuse=False,
                            activation=None,
                            initializer=self.init,
                            dropout=None, mode='None',
                            num_layers=1)
    elif('BIDAF' in self.args.ptr_type):
        """ BiDAF style pointer layer
        """
        output2,_  = build_rnn(output, doc_len,
                            rnn_type=self.args.rnn_type,
                            reuse=False,
                            name='mdl_rnn',
                            num_layers=1,
                            rnn_dim=self.args.rnn_size,
                            dropout=self.args.dropout,
                            initializer=self.rnn_init,
                            use_cudnn=self.args.use_cudnn,
                            is_train=self.is_train
                            )
        start_ptr, _ = build_rnn(output2, doc_len,
                            rnn_type=self.args.ptr_type,
                            reuse=False,
                            name='start_rnn',
                            num_layers=1,
                            rnn_dim=self.args.rnn_size,
                            dropout=self.args.dropout,
                            initializer=self.rnn_init,
                            use_cudnn=self.args.use_cudnn,
                            is_train=self.is_train
                            )
        att0 = projection_layer(tf.concat([start_ptr, output],2), 1,
                            name='ptr_proj_start',
                            reuse=None,
                            activation=None,
                            initializer=self.init,
                            dropout=self.args.dropout, mode='None',
                            num_layers=1,
                            is_train=self.is_train)
        # bsz x dim x 1
        att = tf.nn.softmax(att0)
        att_ptr = tf.reduce_sum(start_ptr * att, 1, keepdims=True)
        seq_len = tf.shape(output)[1]
        att_ptr = tf.tile(att_ptr, [1, seq_len, 1])

        end_ptr = tf.concat([output, start_ptr, att_ptr,
                            att_ptr * start_ptr], 2)
        end_ptr, _ = build_rnn(end_ptr, doc_len,
                            rnn_type=self.args.ptr_type,
                            reuse=False,
                            num_layers=1,
                            name='end_rnn',
                            rnn_dim=self.args.rnn_size,
                            dropout=None,
                            initializer=self.rnn_init,
                            use_cudnn=self.args.use_cudnn,
                            is_train=self.is_train
                            )
        end_ptr =  projection_layer(tf.concat([end_ptr, output],2), 1,
                            name='ptr_proj_end',
                            reuse=False,
                            activation=None,
                            initializer=self.init,
                            dropout=None, mode='None',
                            num_layers=1, is_train=self.is_train
                            )
        start_ptr = att0
    elif('RP' in self.args.ptr_type):
        start_ptr, end_ptr = recurrent_pointer_layer(self, output,
                                                    doc_len)

    seq_len = tf.shape(start_ptr)[1]
    start_ptr = tf.reshape(start_ptr, [-1, seq_len])
    end_ptr = tf.reshape(end_ptr, [-1, seq_len])

    self.start_ptr = softmax_mask(start_ptr, cmask)
    self.end_ptr = softmax_mask(end_ptr, cmask)
    self.predict_start = tf.nn.softmax(self.start_ptr)
    self.predict_end = tf.nn.softmax(self.end_ptr)
    return self
