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
from tylib.lib.compose_op import *
from tylib.lib.cnn import *
from tylib.models.base_model import BaseModel
from tylib.lib.sim_op import *
from tylib.lib.enhanced import *
from tylib.exp.multi_gpu import *
from tylib.lib.bimpm import *
from tylib.exp.utilities import *
from tylib.lib.func import *
from tylib.lib.cudnn_cove_lstm import *
from .decaprop import *

class SpanModel(BaseModel):
    ''' TensorFlow Span Prediction Model
    '''
    def __init__(self, vocab_size, args,
                    char_size=None, sdmax=None,
                    f_len=None, num_features=0,
                    num_global_features=0):
        """ Initializes a Span Prediction model
        """

        super(SpanModel, self).\
            __init__(vocab_size, args,
                        char_size=char_size,
                        sdmax=sdmax, f_len=f_len)
        self.num_features = num_features
        self.num_global_features = num_global_features

        if(self.args.num_gpu>1):
            self._build_multi_gpu_model()
        else:
            self._build_model()

    def get_feed_dict_v2(self, feed_holder, batch,
                            mode='training', lr=None):
        """ Get feed dict by feed holders
        """
        batch = zip(*batch)
        feed_holder = {value:key for key, value in feed_holder.items()}
        feed_dict = {}
        if(mode!='training'):
            feed_dict[feed_holder['emb_dropout']] = 1.0
            feed_dict[feed_holder['hdropout']] = 1.0
            feed_dict[feed_holder['dropout']] = 1.0
        else:
            feed_dict[feed_holder['emb_dropout']] = self.args.emb_dropout
            feed_dict[feed_holder['hdropout']] = self.args.hdrop
            feed_dict[feed_holder['dropout']] = self.args.dropout

        if(self.args.dev_lr>0):
            feed_dict[feed_holder['learn_rate']] = lr

        for key, value in feed_holder.items():
            # keys are strings
            if(key not in self.imap):
                continue
            list_id = self.imap[key]
            feed_dict[value] = batch[list_id]
        return feed_dict

    def _make_feed_holder(self):
        """ Builds a feed holder with reference strings
        """
        feed_holder = {
            self.doc_inputs:'doc_inputs',
            self.doc_len:'doc_len',
            self.query_inputs:'query_inputs',
            self.query_len:'query_len',
            self.start_label:'start_label',
            self.end_label:'end_label',
            self.doc_char_inputs:'doc_char_inputs',
            self.query_char_inputs:'query_char_inputs',
            self.doc_features:'doc_feats',
            self.query_features:'query_feats',
            self.doc_fq:'doc_fq',
            self.query_fq:'query_fq',
            self.dropout:'dropout',
            self.emb_dropout:'emb_dropout',
            self.hdropout:'hdropout',
            self.learn_rate:'learn_rate',
            self.qt_feats:'qt_feats'
        }
        return feed_holder

    def _build_placeholders(self):
        ''' Span Model placeholders
        '''

        self.output_pos, self.output_neg = None, None

        with tf.name_scope('doc_input'):
            self.doc_inputs = tf.placeholder(tf.int32,
                                    shape=[None, None],
                                    name='doc_inputs')
            tf.add_to_collection('doc_inputs', self.doc_inputs)
            self.doc_features = tf.placeholder(tf.float32,
                                    shape=[None, None, self.num_features],
                                    name='doc_features')
            self.doc_fq = tf.placeholder(tf.float32,
                                    shape=[None, None, 1],
                                    name='doc_fq')

        with tf.name_scope('doc_char_input'):
            self.doc_char_inputs = tf.placeholder(tf.int32,
                                    shape=[None, None,
                                            self.args.char_max],
                                    name='doc_char_inputs')
            tf.add_to_collection('doc_char_inputs', self.doc_char_inputs)

        with tf.name_scope('query_input'):
            self.query_inputs = tf.placeholder(tf.int32,
                                    shape=[None, None],
                                    name='query_inputs')
            tf.add_to_collection('query_inputs', self.query_inputs)
            self.query_features = tf.placeholder(tf.float32,
                                    shape=[None, None,
                                            self.num_features],
                                    name='query_features')
            self.qt_feats = tf.placeholder(tf.int32,
                                shape=[None,1],
                                name='qt_feats')
            self.query_fq = tf.placeholder(tf.float32,
                                    shape=[None, None, 1],
                                    name='query_fq')

        with tf.name_scope('query_char_input'):
            self.query_char_inputs = tf.placeholder(tf.int32,
                                    shape=[None, None,
                                        self.args.char_max],
                                    name='query_char_inputs')
            tf.add_to_collection('query_char_inputs', self.query_char_inputs)

        with tf.name_scope('doc_lengths'):
            self.doc_len = tf.placeholder(tf.int32, shape=[None])
            tf.add_to_collection('doc_len', self.doc_len)

        with tf.name_scope('query_lengths'):
            self.query_len = tf.placeholder(tf.int32, shape=[None])
            tf.add_to_collection('query_len', self.query_len)

        with tf.name_scope('label_vector'):
            self.start_label = tf.placeholder(tf.int32,
                            shape=[None])
            tf.add_to_collection('start_label', self.start_label)
            self.end_label = tf.placeholder(tf.int32,
                            shape=[None])
            tf.add_to_collection('start_label', self.end_label)

    def match_op(self, a, b, alen, blen, name,
                reuse=None, mask_a=None, mask_b=None):
        """ Align and match operation
        a = query
        b = document
        """
        pooling = self.args.dq_pooling
        if('BIDAF' in self.args.rnn_type):
            # Use Bidaf style Co-Attention
            pooling = 'BIDAF'
        c, d, _, _, _ = co_attention(
                        a, b,
                        att_type=self.args.dq_att,
                        pooling=pooling,
                        kernel_initializer=self.init,
                        dropout=None,
                        seq_lens=[alen, blen],
                        name=name,
                        mask_a=mask_a,
                        mask_b=mask_b
                                )
        if('SM' in self.args.rnn_type):
            """
            SubMult Op
            https://arxiv.org/pdf/1611.01747.pdf
            """
            match = sub_mult_nn(c, b, reuse=reuse, init=self.init)
        elif('EN' in self.args.rnn_type):
            """ Enhanced Reprensentations
            """
            _dim = c.get_shape().as_list()[2]
            match = tf.concat([c*b, c-b, c, b], 2)
            match = projection_layer(match, _dim,
                                name='enNN',
                                reuse=False,
                                activation=tf.nn.relu,
                                initializer=self.init,
                                dropout=None, mode='FC',
                                num_layers=1)
        elif('BIDAF' in self.args.rnn_type):
            _dim = c.get_shape().as_list()[2]
            d = tf.reduce_sum(d, 1, keep_dims=True)
            seq_len =tf.shape(b)[1]
            d = tf.tile(d, [1, seq_len, 1])
            match = tf.concat([b, c, c * b, b * d], 2)
        else:
            match = c * b
        return match


    def _build_graph(self, model_id=0):
        """ Builds graph
        """
        feature_store = []

        if(self.args.use_cudnn==1):
            self.rnn = cudnn_rnn
        else:
            self.rnn = native_gru

        self.doc_mask = tf.cast(self.doc_inputs, tf.bool)
        self.qmask = tf.cast(self.query_inputs, tf.bool)

        with tf.variable_scope('embedding_layer'):
            if(self.args.pretrained==1):
                self.embeddings = tf.get_variable('embeddings',
                                    shape=[self.vocab_size,
                                    self.args.emb_size],
                                    trainable=self.args.trainable)
                self.embeddings_init = self.embeddings.assign(
                                            self.emb_placeholder)
            else:
                self.embeddings = tf.get_variable('embeddings',
                                    shape=[self.vocab_size,
                                    self.args.emb_size],
                                    initializer=self.init)
            if('CHAR' in self.args.rnn_type):
                self.char_embeddings = tf.get_variable('char_embed',
                                    shape=[self.char_size,
                                    self.args.char_emb_size])

        #doc_inputs, clip_smax = clip_sentence(self.doc_inputs, self.doc_len)
        #query_inputs, clip_qmax = clip_sentence(self.query_inputs, self.query_len)
        doc_inputs = self.doc_inputs
        query_inputs = self.query_inputs

        """ Character Encoders
        """

        if('CHAR' in self.args.rnn_type):
            printc("Using Char Features",'yellow')
            seq_len = tf.shape(doc_inputs)[1]
            with tf.variable_scope("char_enc"):
                bsz = tf.shape(self.doc_char_inputs)[0]
                if('RNN' in self.args.char_enc):
                    char_rnn = self.rnn(
                                num_layers=1,
                                num_units=self.args.rnn_size,
                                batch_size=bsz,
                                input_size=self.args.char_emb_size,
                                keep_prob=self.args.dropout,
                                is_train=self.is_train)
                else:
                    char_rnn = None
                doc_char_embed =  char_encoder(self.doc_char_inputs, seq_len,
                                    self.char_embeddings, self.args.cnn_size,
                                    self.doc_len, self.args.char_enc,
                                    self.args.char_max, self.args.conv_size,
                                    self.init, clip=False,
                                    char_rnn=char_rnn,
                                    var_drop=self.args.var_drop)
                qmax = tf.shape(self.query_char_inputs)[1]
                query_char_embed =  char_encoder(self.query_char_inputs, qmax,
                                    self.char_embeddings, self.args.cnn_size,
                                    self.query_len, self.args.char_enc,
                                    self.args.char_max, self.args.conv_size,
                                    self.init, reuse=True, clip=False,
                                    char_rnn=char_rnn,
                                    var_drop=self.args.var_drop)
        else:
            doc_char_embed = None
            query_char_embed = None

        if('EM' in self.args.add_features):
            # Add EM add_features
            printc("Using EM Features",'yellow')
            doc_features, query_features = self.doc_features, self.query_features
        else:
            doc_features = None
            query_features = None

        if('FQ' in self.args.add_features):
            printc("Using FQ features", 'yellow')
            doc_fq, query_fq = self.doc_fq, self.query_fq
        else:
            doc_fq = None
            query_fq = None

        if("QT" in self.args.add_features):
            """ Adds question type features
            """
            printc("Using QT Features",'yellow')
            qt_feats = self.qt_feats
            self.qt_embeddings = tf.get_variable('qt_embed',
                                shape=[12, self.args.qt_emb_size])
            qt_embed = tf.nn.embedding_lookup(self.qt_embeddings, qt_feats)
            doc_qt = tf.tile(qt_embed, [1, tf.shape(doc_inputs)[1], 1])
            query_qt = tf.tile(qt_embed, [1, tf.shape(query_inputs)[1], 1])
        else:
            doc_qt, query_qt = None, None


        """ Input encoders takes in word embedding and optionally
        char and other features. It optionally passes into n layers
        of projection layers. Proj layers can also be optionally
        highway layers.
        """

        doc_embed = input_encoder(self.embeddings,
                                [doc_inputs],
                                proj=self.args.translate_proj,
                                proj_dim=self.args.proj_size,
                                dropout=self.args.emb_dropout,
                                init=self.init,
                                proj_mode=self.args.proj_mode,
                                num_proj=self.args.num_proj,
                                extras=[[doc_char_embed,
                                    doc_features, doc_qt,
                                    doc_fq]],
                                is_train=self.is_train,
                                use_cove=self.args.use_cove
                                )[0]
        query_embed = input_encoder(self.embeddings,
                                [query_inputs],
                                proj=self.args.translate_proj,
                                proj_dim=self.args.proj_size,
                                dropout=self.args.emb_dropout,
                                init=self.init, reuse=True,
                                proj_mode=self.args.proj_mode,
                                num_proj=self.args.num_proj,
                                is_train=self.is_train,
                                use_cove=self.args.use_cove,
                                extras=[[query_char_embed,
                                    query_features, query_qt,
                                    query_fq]])[0]

        doc_len = self.doc_len
        query_len = self.query_len
        raw_doc_embed = doc_embed

        _dim = doc_embed.get_shape().as_list()[2]

        query_hidden = query_embed
        doc_hidden = doc_embed

        """ This is the main body of the model.
        It accepts doc (passage) and query embeddings
        with lengths and masks and then outputs the
        start and end pointer
        """

        if('DECAPROP' in self.args.rnn_type):
            self = build_deca_prop(self, doc_hidden, query_hidden,
                            doc_len, query_len,
                            self.doc_mask, self.qmask)

        else:
            self = build_vanilla_model(self, doc_hidden, doc_len, query_hidden,
                                query_len, self.doc_mask, self.qmask)
            return

    def _build_multi_gpu_model(self):
        """ Supports for multi_gpu_model
        """
        print("Attempting Multi-GPU setup...")
        tower_grads = []
        self.multi_feed_dicts = []

        with self.graph.as_default(), tf.device('/cpu:0'):
            # with tf.variable_scope(tf.get_variable_scope()):
            lr = self._get_learn_rate()
            opt = self._get_optimizer(lr)
            for i in xrange(self.args.num_gpu):
                with tf.device('/gpu:%d' % i):
                    # with tf.variable_scope('model', reuse=(i>0),
                    #         caching_device='/cpu:0') as scope:
                    with slim.arg_scope([slim.variable], device='/cpu:0') as scope:
                        with tf.variable_scope('model', reuse=(i>0)) as scope:
                             print("Creating model for gpu={}".format(i))
                             loss = self._create_model(loss_only=True)
                             self.multi_feed_dicts.append(self._make_feed_holder())
                             grads = opt.compute_gradients(loss)
                             tower_grads.append(grads)
            grads = average_gradients(tower_grads)

            self.train_op = self._build_train_ops(grads, opt, self.global_step)
            self._build_predict_ops()
            # Use 1st one for prediction
            self.feed_holder = self.multi_feed_dicts[-1]

    def _create_model(self, show_stats=True, loss_only=False, model_id=0):

        self._build_base_placeholders()
        self._build_placeholders()
        self.is_train = tf.get_variable("is_train",
                                        shape=[],
                                        dtype=tf.bool,
                                        trainable=False)
        self.global_step = tf.Variable(0, trainable=False)
        self.output = self._build_graph(model_id=model_id)
        self.feed_holder = self._make_feed_holder()
        lr = self._get_learn_rate()
        opt = self._get_optimizer(lr)
        loss = self._cost_function()
        if(show_stats==True):
            model_stats()
        if(loss_only):
            return loss
        grads = opt.compute_gradients(loss)
        op = self._build_train_ops(grads, opt, self.global_step)
        self.train_op = op
        self._build_predict_ops()

    def _build_model(self):
        '''  Build Model is the main function where the whole graph is init.

        Creates placeholders, the actual network and also cost and
        train_ops. Everything becomes an attribute of the model class which
        can be called via tensorflow's session.

        '''
        with self.graph.as_default():
            self._create_model()

    def _build_predict_ops(self):
        ''' Creates predict ops
        '''
        if(self.args.mask_softmax==1):
            print("masking predict op...")
            self.predict_op = mask_zeros(self.predict_op, self.choice_mask,
                                    self.args.num_choice)
        else:
            self.predict_op = [self.predict_start, self.predict_end]

        outer = tf.matmul(tf.expand_dims(self.predict_start, axis=2),
                        tf.expand_dims(self.predict_end, axis=1))
        outer = tf.matrix_band_part(outer, 0, self.args.max_span)
        self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
        self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)


    def _cost_function(self):
        ''' Create Span-based Loss Function and adds L2 reg
        '''

        with tf.name_scope("cost_function"):
            """ Builds joint loss for start and end pointers.
            """
            seq_len = tf.shape(self.start_ptr)[1]
            self.start_label = tf.one_hot(self.start_label, depth=seq_len)
            self.end_label = tf.one_hot(self.end_label, depth=seq_len)
            loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.start_ptr, labels=tf.stop_gradient(self.start_label))
            loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.end_ptr, labels=tf.stop_gradient(self.end_label))
            self.cost = tf.reduce_mean(loss1 + loss2)

        with tf.name_scope('regularization'):
            if(self.args.l2_reg>0):
                print("Adding L2 Reg of {}".format(self.args.l2_reg))
                vars = tf.trainable_variables()
                _v = [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]
                lossL2 = tf.add_n(_v)
                lossL2 = lossL2 * self.args.l2_reg
                self.cost = self.cost + lossL2

        control_ops = []
        if(self.args.ema_decay>0):
            ema = tf.train.ExponentialMovingAverage(0.9999)
            ema_op = ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.cost = tf.identity(self.cost)


        return self.cost
