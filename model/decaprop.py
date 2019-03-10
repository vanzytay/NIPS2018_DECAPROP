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

import datetime

from tylib.lib.seq_op import *
from tylib.lib.nn import *
from tylib.lib.att_op import *
from tylib.lib.rnn import *
from tylib.lib.compose_op import *
from tylib.lib.cnn import *
from tylib.models.base_model import BaseModel
from tylib.lib.sim_op import *
from tylib.exp.multi_gpu import *
from tylib.exp.utilities import *
from tylib.lib.func import *


""" Functions to build DecaProp Model
"""

def build_deca_prop(self, c, q, c_len, q_len, cmask, qmask,
                    return_pointer=True):
    """ Densely Connected Attention Propagation model
    (DecaProp)

    Self is the span_model class.
    """
    print("Building DecaProp architecture..")
    bsz = tf.shape(c)[0]
    d = self.args.rnn_size
    couts,qouts=[],[]
    aggr = []

    for i in range(self.args.rnn_layers):
        with tf.variable_scope("encoding{}".format(i)):
            rnn = self.rnn(num_layers=1, num_units=d,
                        batch_size=bsz,
                        input_size=c.get_shape().as_list()[-1],
                        keep_prob=self.args.dropout,
                        is_train=self.is_train,
                        rnn_type=self.args.rnn_type,
                        init=self.init)
            c = rnn(c, seq_len=c_len, var_drop=self.args.var_drop)
            q = rnn(q, seq_len=q_len, var_drop=self.args.var_drop)
            couts.append(c)
            qouts.append(q)
            # this should be optional.
            # c, q, ff = bidirectional_attention_connector(
            #             c, q, c_len, q_len,
            #             None, None,
            #             mask_a=cmask, mask_b=qmask,
            #             initializer=self.init,
            #             factor=self.args.factor, factor2=32,
            #             name='enccafe{}'.format(i))
            # aggr.append(ff[0])

    last_q = q
    c = tf.concat(couts, 2)
    q = tf.concat(qouts, 2)

    cff, qff = [], []
    with tf.variable_scope("deca{}".format(i)):
        for i in range(self.args.rnn_layers):
            for j in range(self.args.rnn_layers):
                _, _, ff = bidirectional_attention_connector(
                            couts[i],
                            qouts[j],
                            c_len, q_len,
                            None, None,
                            mask_a=cmask, mask_b=qmask,
                            initializer=self.init,
                            factor=self.args.factor, factor2=32,
                            name='{}_{}'.format(i, j),
                            use_intra=True)
                cff.append(ff[0])
                qff.append(ff[1])

    cff = tf.concat(cff, 2)
    qff = tf.concat(qff, 2)
    c = tf.concat([c, cff], 2)
    q = tf.concat([q, qff], 2)

    elf = []
    with tf.variable_scope("attention"):
        if('SM' in self.args.rnn_type):
            print("Using SubMul mode")
            c2, q2, _, _, _ = co_attention(
                            q, c,
                            att_type='DOT',
                            pooling='MATRIX',
                            kernel_initializer=self.init,
                            dropout=None,
                            seq_lens=[q_len, c_len],
                            name='qc_att',
                            mask_a=qmask,
                            mask_b=cmask
                                    )
            qc_att = sub_mult_nn(c2, c)
        elif('SYM' in self.args.rnn_type):
            print("Using Gated Symmetric Dot Attention")
            qc_att =  symmetric_dot_attention(c, q,
                                   mask=qmask,
                                   hidden=d,
                                   keep_prob=self.args.dropout,
                                   is_train=self.is_train,
                                   init=self.init)
        elif('NBA' in self.args.rnn_type):
            print("Disabling Bi-Attention")
            qc_att = c
        else:
            qc_att = dot_attention(c, q, mask=qmask,
                                   hidden=d,
                                   keep_prob=self.args.dropout,
                                   is_train=self.is_train)

    if('NMR' in self.args.rnn_type):
        print("Disabling Middle RNN")
        att = qc_att
        elf.append(att)
    else:
        rnn = self.rnn(num_layers=1, num_units=d,
                    batch_size=bsz,
                    input_size=qc_att.get_shape().as_list()[-1],
                    keep_prob=self.args.dropout, is_train=self.is_train,
                    init=self.rnn_init)
        att = rnn(qc_att, seq_len=c_len, var_drop=self.args.var_drop)
        elf.append(att)
    if('NSA' in self.args.rnn_type):
        print("Disabling Self-Attention..")
        match = att
    else:
        with tf.variable_scope("match"):
            self_att = dot_attention(att, att,
                    mask=cmask, hidden=d,
                    keep_prob=self.args.dropout, is_train=self.is_train)
            rnn = self.rnn(num_layers=1, num_units=d,
                        batch_size=bsz,
                        input_size=self_att.get_shape().as_list()[-1],
                        keep_prob=self.args.dropout, is_train=self.is_train,
                        init=self.rnn_init)
            match = rnn(self_att, seq_len=c_len, var_drop=self.args.var_drop)
            elf.append(match)

    if('FULL' in self.args.rnn_type):
        print("Full Dense Att Prop")
        elf_feats = []
        print("Making {} connections".format(len(elf) * len(qouts)))
        with tf.variable_scope("decafull2"):
            for i in range(len(elf)):
                for j in range(len(qouts)):
                    _, _, ff = factor_flow2(elf[i], qouts[j],
                            c_len, q_len,
                            None, None,
                            mask_a=cmask, mask_b=qmask,
                            initializer=self.init,
                            factor=self.args.factor, factor2=32,
                            name='{}_{}'.format(i, j), use_intra=False)
                    elf_feats.append(ff[0])
        elf_feats = tf.concat(elf_feats, 2)
        match = tf.concat(elf_feats, 2)

    if(return_pointer==False):
        return match

    with tf.variable_scope("pointer"):
        if('RP' in self.args.rnn_type):
        # Recurrent Pointer
            with tf.variable_scope("start"):
                rnn = self.rnn(num_layers=1, num_units=d,
                            batch_size=bsz,
                            input_size=match.get_shape().as_list()[-1],
                            keep_prob=self.args.dropout, is_train=self.is_train,
                            init=self.rnn_init)
                start_rnn = rnn(match, seq_len=c_len,
                            var_drop=self.args.var_drop)
            with tf.variable_scope("end"):
                rnn = self.rnn(num_layers=1, num_units=d,
                            batch_size=bsz,
                            input_size=start_rnn.get_shape().as_list()[-1],
                            keep_prob=self.args.dropout, is_train=self.is_train,
                            init=self.rnn_init)
                end_rnn = rnn(start_rnn, seq_len=c_len,
                            var_drop=self.args.var_drop)

            logits1 = projection_layer(start_rnn, 1,
                                name='ptr_start',
                                reuse=None,
                                activation=None,
                                initializer=self.init,
                                dropout=None, mode='None',
                                num_layers=1)
            logits2 =  projection_layer(end_rnn, 1,
                                name='ptr_end',
                                reuse=False,
                                activation=None,
                                initializer=self.init,
                                dropout=None, mode='None',
                                num_layers=1)
            seq_len = tf.shape(end_rnn)[1]
            logits1 = tf.reshape(logits1, [-1, seq_len])
            logits2 = tf.reshape(logits2, [-1, seq_len])
        else:
            init = summ(q[:, :, -2 * d:], d, mask=qmask,
                        keep_prob=self.args.dropout,
                        is_train=self.is_train)
            pointer = ptr_net_v2(batch=bsz,
                        hidden=init.get_shape().as_list()[-1],
                        keep_prob=self.args.dropout, is_train=self.is_train)
            logits1, logits2 = pointer(init, match, d, cmask,
                                lengths=q_len)

        self.start_ptr = logits1
        self.end_ptr = logits2
        self.predict_start = tf.nn.softmax(logits1)
        self.predict_end = tf.nn.softmax(logits2)
    return
