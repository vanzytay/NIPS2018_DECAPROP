from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np

import argparse
import json
import gzip
import pickle
from tqdm import tqdm
from datetime import datetime
import os
import random
import csv
import sys
from multiprocessing import Pool
from termcolor import colored
from collections import Counter

from parser import *

from keras.utils import np_utils
from keras.preprocessing import sequence
from tensorflow.contrib.tensorboard.plugins import projector

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from tylib.exp.experiment import Experiment
from tylib.exp.exp_ops import *
from tylib.exp.utilities import *
from tylib.prep.loaders import *
from tylib.exp.metrics import *

from model.span_model import SpanModel
from span_evaluate import *
from utilities import *


class SpanExperiment(Experiment):
    '''

    This experiment conducts Span-based Experiments
    (e.g., NewsQA, TriviaQA, Squad etc.)
    '''

    def __init__(self):
        ''' Initializes a Span Experiment Class
        '''
        super(SpanExperiment, self).__init__()
        parser = build_parser()
        self._build_char_index()
        self.args = parser.parse_args()

        self.model_name = self.args.rnn_type
        self.has_test=True
        # self.hierarchical = False
        self.patience = 0
        self.end_format = False
        self.return_idx = False

        # Use RNN for char-embed for now
        self.args.char_enc = 'RNN'
        self.args.cnn_size = self.args.rnn_size

        """ Dataset-specific settings
        """

        if('Squad' in self.args.dataset):
            self.has_test = False
            self.end_format = True
            self.args.align_spans = True
            self.args.use_lower = 0
        if('NewsQA' in self.args.dataset):
            self.end_format = True
            self.args.align_spans = True
            self.args.use_lower = 0
        if('Quasar' in self.args.dataset):
            self.end_format = True
            self.args.align_spans = True
            self.args.use_lower = 0
        if('TriviaQA' in self.args.dataset):
            self.end_format = True
            self.args.align_spans = True
            self.has_test =False
            self.args.use_lower = 0
        if('NarrativeQA' in self.args.dataset):
            self.end_format = True
            self.args.align_spans = False
            self.has_test = True
            self.args.use_lower = 1
        if('SearchQA' in self.args.dataset):
            self.end_format = True
            self.args.align_spans = False
            self.has_test = True
            self.args.use_lower =1

        printc("====================================", 'green')
        printc("[Start] Training Span Prediction task", 'green')
        printc('[{} Dataset]'.format(self.args.dataset), 'green')
        printc("Tensorflow {}".format(tf.__version__), 'green')

        self._setup()
        self.sdmax = None
        self.f_len = None
        self.num_choice = 0
        self.query_max = 0
        self.test_set2, self.dev_set2 = None, None
        self.num_features = 0
        self.num_global_features = 0

        print("Loading environment...")

        try:
            self.env = fast_load(
                    './datasets/{}/env.gz'.format(
                            self.args.dataset))
        except:
            print("Can't find GZ file. loading pure json instead..")
            self.env = fast_load('./datasets/{}/env.json'.format(
                                    self.args.dataset))
        if(self.args.add_features!=""):
            print("Loading Features..")
            self.feats =  fast_load(
                    './datasets/{}/feats.gz'.format(
                            self.args.dataset))
            if('EM' in self.args.add_features):
                self.num_features +=1
        else:
            self.feats = {'train':None, 'test':None, 'dev':None,
                            'dev2':None, 'test2': None}

        # Build Word Index and Inverse Index
        print("Word Index={}".format(len(self.env['word_index'])))
        self.index_word = {key: value for value,
                        key in self.env['word_index'].items()}
        self.word_index = self.env['word_index']

        self.mdl = SpanModel(len(self.env['word_index']),
                        self.args,
                        char_size=len(self.char_index),
                        sdmax=self.sdmax,
                        f_len=self.f_len,
                        num_features=self.num_features,
                        num_global_features=self.num_global_features
                            )
        self._setup_tf()

        if('Baidu' in self.args.dataset):
            self = prep_all_baidu(self)
        else:
            self.dev_set, self.dev_eval = self._prepare_set(
                                        self.env['dev'],
                                        set_type='dev',
                                        features=self.feats['dev'])
            self.train_set, self.train_eval = self._prepare_set(
                                        self.env['train'],
                                        set_type='train',
                                        features=self.feats['train'])

            print('Train={} Dev={}'.format(len(self.train_set),
                                        len(self.dev_set)))

            if('dev2' in self.env):
                self.dev_set2, self.dev_eval2 = self._prepare_set(
                                        self.env['dev2'],
                                        set_type='dev',
                                        features=self.feats['dev2'])
            else:
                self.dev_set2, self.dev_eval2 = None, None

        if(self.has_test):
            self.test_set, self.test_eval = self._prepare_set(self.env['test'],
                                        set_type='test',
                                        features=self.feats['test'])
            print("Test={}".format(len(self.test_set)))

        print("Loaded environment")
        print("Vocab size={}".format(len(self.env['word_index'])))

        self._make_dir()

        # Primary metric to use to align dev and test sets
        if('NarrativeQA' in self.args.dataset):
            self.eval_primary = 'Rouge'
            self.show_metrics  = ['Bleu1','Bleu4','Meteor','Rouge']
        else:
            self.eval_primary = 'EM'
            self.show_metrics = ['EM','F1']

    def compute_metrics(self, spans, passage, passage_str, labels, qids,
                        maxspan=15, questions=None, set_type='',
                        align_spans=None, spans2=None):
        """ Compute EM and F1 Metrics
        """

        all_em, all_f1 = [], []
        all_em2, all_f12 = [], []
        all_rouge, all_b4, all_b1, all_meteor = [],[],[],[]

        if(type(passage_str[0]) is list):
            # print(passage[0])
            def passage2id(passage):
                return [self.index_word[x] for x in passage if x>0]

            passage_str = [passage2id(x) for x in passage]
            passage_str = [' '.join(x) for x in passage_str]

        max_span = self.args.max_span

        assert(len(spans)==len(passage)==len(passage_str)==len(labels))

        predict_dict, ans_dict = {}, {}

        error_rate = 0

        if(align_spans is not None):
            assert(len(align_spans)==len(spans))

        for i in tqdm(range(len(passage)), desc='Evaluating'):
            _span = spans[i]
            if(spans2 is not None):
                _spans2 = spans2[i]
            else:
                _spans2 = None
            _passage_str = passage_str[i]
            ans_start = np.array(_span[0]).reshape((-1)).tolist()
            ans_end = np.array(_span[1]).reshape((-1)).tolist()
            _qids = qids[i]
            _passage = passage[i]
            if(len(labels[i])==0):
                continue

            _passage_words = _passage_str.split(' ')[:self.args.tmax]

            if(self.args.dataset=='SearchQA'):
                # Unigram only
                if(len(labels[i][0].split())==1):
                    # Unigram sample
                    ans, _ = get_ans_string_single_post_pad_search_updated(
                                                    _passage_str,
                                                    _passage_words,
                                                    ans_start, ans_end,
                                                    maxspan=1
                                                    )
                    _em = metric_max_over_ground_truths(exact_match_score,
                                                    ans, labels[i])
                    all_em2.append(_em)
                else:
                    # Ngram sample
                    ans, _ = get_ans_string_single_post_pad_search_updated(
                                                    _passage_str,
                                                    _passage_words,
                                                    ans_start, ans_end,
                                                    maxspan=3
                                                    )
                    _f1 = metric_max_over_ground_truths(f1_score, ans,
                                                            labels[i])
                    all_f12.append(_f1)

            if(align_spans is not None):
                _align_spans = align_spans[i]
            else:
                _align_spans = None

            ans, error = get_ans_string_single_post_pad_search_updated(
                                                _passage_str,
                                                _passage_words,
                                                ans_start, ans_end,
                                                maxspan=max_span,
                                                align_spans=_align_spans,
                                                spans2=_spans2,
                                                return_idx=self.return_idx
                                                )

            error_rate += error

            predict_dict[str(_qids)] = [ans]
            ans_dict[str(_qids)] = [x for x in labels[i]]

            _em = metric_max_over_ground_truths(exact_match_score,
                                                    ans, labels[i])
            _f1 = metric_max_over_ground_truths(f1_score, ans,
                                                    labels[i])
            all_em.append(_em)
            all_f1.append(_f1)

        # Merge dicts
        merge_dict = {}
        for key, value in predict_dict.items():
            _ans = ans_dict[key]
            _ans = [x.encode('utf-8') for x in _ans]
            value = [x.encode('utf-8') for x in value]
            merge_dict[key] = [value, _ans]

        print("errors={} out of {}".format(error_rate, len(passage)))
        try:
            with open(self.out_dir+'./{}.pred_ans.json'.format(set_type), 'w+') as f:
                json.dump(merge_dict, f, indent=4, ensure_ascii=False)
        except:
            print("Can't find write due to some reason..")

        if(self.args.dataset=='SearchQA'):
            metric1 = 100 * np.mean(all_em2)
            metric2 = 100 * np.mean(all_f12)
            self.write_to_file('[2nd Eval] EM={} F1={}'.format(100 * np.mean(all_em),
                                    100 * np.mean(all_f1)))
            return [metric1, metric2]
        elif('NarrativeQA' in self.args.dataset):
            bleu = batch_bleu_score(ans_dict, predict_dict, n=4)
            metric1 = 100 * bleu[0]
            metric2 = 100 * bleu[3]
            # metric2 = batch_bleu_score(ans_dict, predict_dict, n=4)
            metric3 = 100 * batch_meteor_score(ans_dict, predict_dict)
            metric4 = 100 * batch_rouge_score(ans_dict, predict_dict)
            return [metric1, metric2, metric3, metric4]
        elif('Baidu' in self.args.dataset):
            bleu = batch_bleu_score(ans_dict, predict_dict, n=4)
            bleu4 = 100 * bleu[3]
            metric4 = 100 * batch_rouge_score(ans_dict, predict_dict)
            return [bleu4, metric4]
        else:
            metric1 = 100 * np.mean(all_em)
            metric2 = 100 * np.mean(all_f1)
            return [metric1, metric2]

    def evaluate(self, epoch, data, original_data, name='', set_type=''):
        """ Main evaluation function
        """
        if('NarrativeQA2' in self.args.dataset):
            metrics = evaluate_nqa2(self, epoch, data, original_data, name=name,
                                    set_type=set_type)
            return metrics
        # Training Iteration
        losses, accuracies = [],[]

        batch_size = int(self.args.batch_size/self.args.test_bsz_div)

        num_batches = int(len(data) / batch_size)
        accuracies = 0
        all_start = []
        all_end = []

        ground_truth = [x[1] for x in original_data]
        passages_str = [x[0] for x in original_data]
        if(self.args.align_spans==1):
            # Spans exist
            print("Found align spans, using them")
            align_spans = [x[3] for x in original_data]
        else:
            align_spans = None
        qids = [x[2] for x in original_data]

        passages = [x[0] for x in data]
        all_yp1 = []
        all_yp2 = []

        for i in tqdm(range(0, num_batches + 1), desc='predicting'):
            batch = make_batch(data, batch_size, i)
            if(batch is None):
                continue
            feed_dict = self.mdl.get_feed_dict_v2(
                                        self.mdl.feed_holder,
                                        batch,
                                        mode='testing')

            loss, p = self.sess.run(
                    [self.mdl.cost, self.mdl.predict_op],
                    feed_dict)

            yp1, yp2 = self.sess.run([self.mdl.yp1, self.mdl.yp2],
                                feed_dict)

            start_p = p[0]
            end_p = p[1]
            all_start += [x for x in start_p]
            all_end += [x for x in end_p]
            all_yp1 += yp1.tolist()
            all_yp2 += yp2.tolist()
            losses.append(loss)

        assert(len(all_start)==len(all_end))
        assert(len(all_start)==len(data))
        spans = zip(all_start, all_end)

        if('Baidu' in self.args.dataset):
            original = passages_str
            metrics = compute_baidu_metrics(self, spans, original,
                                        ground_truth, qids, self.pmax,
                                        set_type=set_type)
        else:
            metrics = self.compute_metrics(spans, passages,
                                    passages_str, ground_truth, qids,
                                    questions=[x[1] for x in original_data],
                                    set_type=set_type,
                                    align_spans=align_spans,
                                    spans2=zip(all_yp1, all_yp2))
        acc = 0
        self.write_to_file("[{}] Loss={}".format(
                            name, np.mean(losses)))
        if('NarrativeQA' in self.args.dataset):
            self._register_eval_score(epoch, set_type, 'Bleu1',metrics[0])
            self._register_eval_score(epoch, set_type, 'Bleu4', metrics[1])
            self._register_eval_score(epoch, set_type, 'Meteor',metrics[2])
            self._register_eval_score(epoch, set_type, 'Rouge', metrics[3])
        elif('Baidu' in self.args.dataset):
            self._register_eval_score(epoch, set_type, 'Bleu4',metrics[0])
            self._register_eval_score(epoch, set_type, 'Rouge', metrics[1])
        else:
            self._register_eval_score(epoch, set_type, 'F1', metrics[1])
            self._register_eval_score(epoch, set_type, 'EM', metrics[0])
        return metrics

    def get_predictions(self, epoch, data, name='', set_type=''):
        """ Same as evaluate but do not compute metrics
        """
        num_batches = int(len(data) / self.args.batch_size)
        accuracies = 0
        all_p = []

        for i in range(0, num_batches + 1):
            batch = make_batch(data,
                    self.args.batch_size, i)
            if(batch is None):
                continue
            feed_dict = self.mdl.get_feed_dict_v2(self.mdl.feed_holder,
                                        batch,
                                        mode='testing')
            p = self.sess.run([self.mdl.predictions], feed_dict)
            all_p += [x for x in p[0]]
        # print(all_p)
        assert(len(all_p)==len(data))
        return all_p

    def _write_predictions(self, preds, set_type):
        with open(self.out_dir + './{}_pred.txt'.format(set_type), 'w+') as f:
            for p in preds:
                f.write(str(p) + '\n')

    def _prepare_set(self, data,
                    set_type='', features=None, ans_features=None):
        """ Prepares set.
        Takes in raw processed data (env.gz) and loads them into
        arrays for passing into feed_dict.
        """
        # data = data


        if(set_type=='train' and self.args.adjust==0):
            print("Removing all samples with ptr more than {}".format(
                                                    self.args.smax))
            print("Original Samples={}".format(len(data)))
            if(self.end_format==1):
                data = [x for x in data if x[3]<self.args.smax]
            else:
                data = [x for x in data if x[2]+x[3]<self.args.smax]
            print("Reduced Samples={}".format(len(data)))

        if(self.args.align_spans==1):
            eval_data = [[x[6], x[5],x[4],x[7]] for x in data]
            asp = [x[7] for x in data]
            show_stats('align spans', [len(x) for x in asp])

        else:
            eval_data = [[x[0],x[5],x[4]] for x in data]

        self.char_pad_token = [0 for i in range(self.args.char_max)]

        def flatten_list(l):
            flat_list = [item for sublist in l for item in sublist]
            return flat_list

        print("Preparing {}".format(set_type))

        def w2i(w):
            try:
                return self.word_index[w]
            except:
                return 1

        def tokenize(s, vmax, pad=True):
            if(self.args.use_lower):
                s = [x.lower() for x in s]
            s = s[:vmax]
            tokens = [w2i(x) for x in s]
            lengths = [len(x) for x in s]
            if(pad==True):
                tokens = pad_to_max(tokens, vmax)
            return tokens

        def hierarchical_tokenize(s, vmax):
            s = [[tokenize(x, vmax) for x in y] for y in s]
            s = [x for x in s]
            return s

        def clip_len(s, vmax):
            if(s>vmax):
                return vmax
            else:
                return s

        if(set_type=='train'):
            smax, qmax = self.args.smax, self.args.qmax
        else:
            smax, qmax = self.args.tmax, self.args.qmax
        q = [x[1] for x in data]
        q_raw = [x[1] for x in data]
        q = [x.split(' ') for x in q]
        q_raw = [x.split(' ') for x in q_raw]
        qlen_raw = [len(x) for x in q]

        qlen = [clip_len(x, qmax) for x in qlen_raw]

        q = [tokenize(x, qmax) for x in tqdm(q, desc='tokenizing qns')]

        start = [x[2] for x in data]

        passages = [x[0] for x in data]
        _passages = [x.split(' ') for x in passages]
        if(self.args.adjust==1 and set_type=='train'):
            print("Adjusting passages")
            _passages, start, align = adjust_passages(_passages, start,
                                    self.args.smax,
                                    span=int(self.args.smax/2))
        # print(_passages[0])
        dlen_raw = [len(x) for x in _passages]
        dlen = [clip_len(x, smax) for x in dlen_raw]
        docs = [tokenize(x, smax) for x in tqdm(_passages,
                                        desc='tokenizing docs')]

        if(self.args.align_spans==1):
            show_stats('pointer', [x[3] for x in data])
            end = [x[3] for x in data]
            start = [min(x, smax-1) for x in start]
            end = [min(x, smax-1) for x in end]
        else:
            show_stats('pointer', [x[2]+x[3]-1 for x in data])
            label_len = [x[3] for x in data]
            end = np.array(start) + np.array(label_len)
            start = [min(x, smax-1) for x in start]
            end = [min(x-1, smax-1) for x in end]

        print("================================")
        printc('Showing passage stats {}'.format(set_type),'cyan')
        show_stats('passage', dlen_raw)
        show_stats('question', qlen_raw)
        print("=================================")

        output = [docs, dlen, q, qlen, start, end]

        # print(dlen)
        self.mdl.register_index_map(0, 'doc_inputs')
        self.mdl.register_index_map(1, 'doc_len')
        self.mdl.register_index_map(2, 'query_inputs')
        self.mdl.register_index_map(3, 'query_len')
        self.mdl.register_index_map(4, 'start_label')
        self.mdl.register_index_map(5, 'end_label')

        if('CHAR' in self.args.rnn_type):
            # print("Preparing Chars...")
            char_index = self.char_index
            char_pad_token = self.char_pad_token
            char_max = self.args.char_max

            def char_idx(x, idx):
                try:
                    return idx[x]
                except:
                    return 1

            def char_ids(d, smax):
                txt = d.split(' ')[:smax]
                _txt = [[char_idx(y, char_index) for y in x] for x in txt]
                _txt = [pad_to_max(x, char_max) for x in _txt]
                _txt = pad_to_max(_txt, smax,
                        pad_token=char_pad_token)
                return _txt

            queries = [x[1] for x in data]
            qc = [char_ids(x[0], self.args.qmax) for x in tqdm(data,
                                                    desc='Prep char query')]
            pc = [char_ids(x, smax) for x in tqdm(passages,
                                                    desc="Prep char doc")]
            qc = np.array(qc).reshape((-1,
                    self.args.qmax, self.args.char_max))
            pc = np.array(pc).reshape((-1,
                    smax, self.args.char_max))

            print("Constructed Char Inputs")
            print(pc.shape)
            print(qc.shape)
            self.mdl.register_index_map(len(output),
                            'doc_char_inputs')
            output.append(pc)
            self.mdl.register_index_map(len(output),
                            'query_char_inputs')
            output.append(qc)

        if(self.args.add_features!=''):
            if('EM' in self.args.add_features):
                qem = [x[1] for x in features]
                pem = [x[0] for x in features]
                if(self.args.adjust==1 and set_type=='train'):
                    qem = apply_alignment(qem, align)
                pem = [pad_to_max(x, smax) for x in pem]
                qem = [pad_to_max(x, qmax) for x in qem]
                pem = np.array(pem).reshape((-1, smax, 1))
                qem = np.array(qem).reshape((-1, qmax, 1))
                self.mdl.register_index_map(len(output), 'doc_feats')
                output.append(pem)
                self.mdl.register_index_map(len(output), 'query_feats')
                output.append(qem)
            if('QT' in self.args.add_features):
                # Question type features
                qt = [question_type(x[1]) for x in data]
                qt = np.array(qt).reshape((-1, 1))
                self.mdl.register_index_map(len(output), 'qt_feats')
                output.append(qt)
            if('FQ' in self.args.add_features):
                freq = [two_way_frequency(x[1], x[0]) for x in data]
                freq_q = [x[0] for x in freq]
                freq_c = [x[1] for x in freq]
                freq_q = [pad_to_max(x, qmax) for x in freq_q]
                freq_c = [pad_to_max(x, smax) for x in freq_c]
                freq_q = np.array(freq_q).reshape((-1, qmax, 1))
                freq_c = np.array(freq_c).reshape((-1, smax, 1))
                self.mdl.register_index_map(len(output), 'doc_fq')
                output.append(freq_c)
                self.mdl.register_index_map(len(output), 'query_fq')
                output.append(freq_q)

        new_data = zip(*output)
        return new_data, eval_data

    def train(self):
        """ Main Training Function
        """

        print("Starting training")
        data = self.train_set
        lr = self.args.lr
        for epoch in range(1, self.args.epochs + 1):
            self.write_to_file('===================================')
            losses, accuracies = [],[]

            if(self.args.shuffle==1):
                random.shuffle(data)

            num_batches = int(len(data) / self.args.batch_size)
            accuracies = 0
            all_p = []
            self.sess.run(tf.assign(self.mdl.is_train, self.mdl.true))
            for i in tqdm(range(0, num_batches + 1)):
                batch = make_batch(data,
                        self.args.batch_size, i)
                if(batch is None):
                    continue
                if(self.args.num_gpu>1):
                    # Multi-GPU feed dict
                    feed_dict = {}
                    gpu_bsz = int(len(batch) / self.args.num_gpu)
                    for gid, feed_holder in enumerate(self.mdl.multi_feed_dicts):
                        mbatch = make_batch(batch, gpu_bsz, gid)
                        fd = self.mdl.get_feed_dict_v2(feed_holder,
                                                    mbatch,
                                                    mode='training', lr=lr)
                        feed_dict.update(fd)
                else:
                    feed_dict = self.mdl.get_feed_dict_v2(
                                                self.mdl.feed_holder,
                                                batch,
                                                mode='training', lr=lr)

                _, loss = self.sess.run(
                    [self.mdl.train_op, self.mdl.cost],
                    feed_dict)

                losses.append(loss)

                if(self.args.tensorboard):
                    self.train_writer.add_summary(
                        summary, epoch * num_batches + i)

            self.write_to_file("[{}] [{}] Epoch [{}] Loss={}".format(
                 self.args.dataset, self.model_name, epoch, np.mean(losses)))
            self.write_to_file('[smax={}] [rnn={}] [lr={}] [f={}] [cove={}]'.format(
                                        self.args.smax,
                                        self.args.rnn_size,
                                        lr,self.args.add_features,
                                        self.args.use_cove
                                        ))
            print("[GPU={}]".format(self.args.gpu))

            self.sess.run(tf.assign(self.mdl.is_train,
                                  self.mdl.false))
            lr = self._run_evaluation(epoch, lr)


    def _run_evaluation(self, epoch, lr):
        """ Run Evaluation on test set
        """

        dev_metrics = self.evaluate(epoch, self.dev_set, self.dev_eval,
                                    name='dev',
                                    set_type='dev')
        self._show_metrics(epoch, self.eval_dev,
                                    self.show_metrics,
                                        name='Dev')

        best_epoch, _ = self._select_test_by_dev(epoch, self.eval_dev,
                                    None,
                                    no_test=True,
                                    name='dev')

        if(self.dev_set2 is not None):
            dev_metrics2 = self.evaluate(epoch, self.dev_set2, self.dev_eval2,
                                        name='dev2',
                                        set_type='dev2')
            self._show_metrics(epoch, self.eval_dev2,
                                        self.show_metrics,
                                            name='Dev2')
            best_epoch, _ = self._select_test_by_dev(epoch, self.eval_dev2,
                                        None,
                                        no_test=True,
                                        name='dev2')

        if(self.args.dev_lr>0 and best_epoch!=epoch):
            self.patience +=1
            print('Patience={}'.format(self.patience))
            if(self.patience>=self.args.patience):
                print("Reducing LR by {} times".format(self.args.dev_lr))
                lr = lr / self.args.dev_lr
                print("LR={}".format(lr))
                self.patience = 0

        """ Evaluation on Test Set Locally
        All other datasets which require submission should use test_eval==0
        """
        if(self.has_test==0):
            if('Baidu' in self.args.dataset and epoch==best_epoch):
                print("Best epoch! Writing predictions to file!")
                generate_baidu_test(self, epoch, self.test_set, self.test_eval,
                                            name='test',
                                            set_type='test')
            return lr

        test_metrics = self.evaluate(epoch, self.test_set,
                                    self.test_eval,
                                    name='test',
                                    set_type='test')
        self._show_metrics(epoch, self.eval_test,
                                    self.show_metrics,
                                        name='Test')
        _, max_ep, best_ep = self._select_test_by_dev(epoch, self.eval_dev,
                                    self.eval_test,
                                    no_test=False, name='test')
        return lr


if __name__ == '__main__':
    exp = SpanExperiment()
    exp.train()
    print("Finished Running")
