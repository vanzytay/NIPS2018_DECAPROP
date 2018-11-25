# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import numpy as np
from collections import Counter
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
import json
import os
import gzip
from utilities import *
from utils import *
from nltk.stem.porter import PorterStemmer
import argparse
from nus_utilities import *
from common_v2 import *
import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.dont_write_bytecode = True

parser = argparse.ArgumentParser()
ps = parser.add_argument
ps("--mode", dest="mode", type=str,  default='all', help="mode")
ps("--vocab_count", dest="vocab_count", type=int,
    default=0, help="set >0 to activate")
args =  parser.parse_args()
mode = args.mode


def word_level_em_features(s1, s2, lower=True, stem=True):
    em1 = []
    em2 = []
    #print(s1)
    #print(s2)
    s1 = s1.split(' ')
    s2 = s2.split(' ')
    if(lower):
        s1 = [x.lower() for x in s1]
        s2 = [x.lower() for x in s2]
    if(stem):
        s1 = [porter_stemmer.stem(x) for x in s1]
        s2 = [porter_stemmer.stem(x) for x in s2]
    for w1 in s1:
        if(w1 in s2):
            em1.append(1)
        else:
            em1.append(0)
    for w2 in s2:
        if(w2 in s1):
            em2.append(1)
        else:
            em2.append(0)
    return em1, em2


def convert_paragraph(para):
    words = []
    context = para['context.tokens']
    # print(len(para['qas']))
    data = []
    for qa in para['qas']:
        qid = qa['id']
        question = qa['question.tokens']
        words += question
        label_start = int(qa['answers'][0]['answer_start'])
        label_length = len(qa['answers'][0]['text.tokens'])
        question = ' '.join(question)
        answer = ' '.join(qa['answers'][0]['text.tokens'])
        #print(qa['answers'][0])
        # print(context[label_start:label_start+label_length])
    words += context
    context = ' '.join(context)
    ground_truths = list(map(lambda x: x['text'], qa['answers']))
    #print(ground_truths)
    data = [context, question, label_start, label_length, qid, ground_truths]

    # print(data)
    return data, words

def load_set(fp, datatype='train'):
    parsed_file = load_json(fp)
    # print(parsed_file)
    all_words = []
    all_data = []
    all_feats = []
    # print(parsed_file[0])
    for p in tqdm(parsed_file, desc='parsing file'):
        pdata, words = convert_paragraph(p)
        qem, pem =  word_level_em_features(pdata[1], pdata[0])
        all_words += words
        all_data.append(pdata)
        all_feats.append([pem, qem])
        # print(qem)
    # print(' Collected {} words'.format(len(all_words)))
    return all_words, all_data, all_feats


train_words, train_data, train_feats = load_set('./corpus/SearchQA/tokenized-train.json')
dev_words, dev_data, dev_feats = load_set('./corpus/SearchQA/tokenized-dev.json')
test_words, test_data, test_feats = load_set('./corpus/SearchQA/tokenized-test.json')

all_words = train_words + dev_words + test_words

if(args.vocab_count>0):
    print("Using Vocab Count of {}".format(args.vocab_count))
    word_index, index_word = build_word_index(all_words, min_count=0,
                                                vocab_count=args.vocab_count,
                                                lower=True)
else:
    word_index, index_word = build_word_index(all_words, min_count=0,
                                                lower=True)

print("Vocab Size={}".format(len(word_index)))

# Convert passages to tokens
# passages = dict(train_passage.items() + test_passage.items() + dev_passage.items())

fp = './datasets/SearchQA/'

if not os.path.exists(fp):
    os.makedirs(fp)

build_embeddings(word_index, index_word,
  out_dir=fp,
  init_type='zero', init_val=0.01,
  emb_types=[('glove',300)],
  normalize=False)

passages = {}

env = {
    'train':train_data,
    'test':test_data,
    'dev':dev_data,
    'passages':passages,
    'word_index':word_index
}

feature_env = {
    'train':train_feats,
    'test':test_feats,
    'dev':dev_feats
    }

dictToFile(env,'./datasets/SearchQA/env.gz'.format(mode))
dictToFile(feature_env,'./datasets/SearchQA/feats.gz'.format(mode))
