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
import spacy
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

def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

nlp = spacy.blank("en")

def word_tokenize(sent):
    doc = nlp(sent)
    # print(doc)
    # return doc.split(' ')
    return [token.text for token in doc]

def convert_paragraph_v2(para):
    data = []
    words = []
    context = para["context"].replace(
        "''", '" ').replace("``", '" ')
    context_tokens = word_tokenize(context)
    context_tokens_str = ' '.join(context_tokens)
    # print(context_tokens)
    words += context_tokens
    spans = convert_idx(context, context_tokens)
    for qa in para["qas"]:
        qid = qa['id']
        ques = qa["question"].replace(
            "''", '" ').replace("``", '" ')
        ques_tokens = word_tokenize(ques)
        words += ques_tokens
        y1s, y2s = [], []
        answer_texts = []
        for answer in qa["answers"]:
            answer_text = answer["text"]
            answer_start = answer['answer_start']
            answer_end = answer_start + len(answer_text)
            answer_texts.append(answer_text)
            answer_span = []
            for idx, span in enumerate(spans):
                if not (answer_end <= span[0] or answer_start >= span[1]):
                    answer_span.append(idx)
            y1, y2 = answer_span[0], answer_span[-1]
            y1s.append(y1)
            y2s.append(y2)
        # ground_truths = list(map(lambda x: x['text'], qa['answers']))
        #print('y1={} y2={}'.format(y1, y2))
        # print(context_tokens)

        # ques_tokens = ' '.join(ques_tokens)
        #print(spans)
        data.append([context_tokens_str, ' '.join(ques_tokens), y1, y2,
                    qid, answer_texts,
                    context, spans])
    return data, words

def convert_paragraph(para):
    words = []
    context = para['context.tokens']
    context_raw = para['context']
    spans = convert_idx(context_raw, context)
    words += context
    context = ' '.join(context)
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
        ground_truths = list(map(lambda x: x['text'], qa['answers']))
        data.append([context, question, label_start, label_length, qid, ground_truths,
                    context_raw])
    return data, words

def load_set(fp, datatype='train'):
    # parsed_file = load_json(fp)
    # # print(parsed_file)
    all_words = []
    all_data = []
    all_feats = []
    # print(parsed_file[0])
    with open(fp, 'r') as f:
        source = json.load(f)
        for article in tqdm(source['data'], desc='parsing file'):
            for p in article['paragraphs']:
                pdata, words = convert_paragraph_v2(p)
                # print(pdata)
                for d in pdata:
                    qem, pem =  word_level_em_features(d[1], d[0])
                    all_feats.append([pem, qem])
                all_words += words
                all_data += pdata

        # print(qem)
    # print(' Collected {} words'.format(len(all_words)))
    return all_words, all_data, all_feats


train_words, train_data, train_feats = load_set('./corpus/squad/train-v1.1.json')
dev_words, dev_data, dev_feats = load_set('./corpus/squad/dev-v1.1.json')
# test_words, test_data, test_feats = load_set('./corpus/NewsQA/tokenized-test-v1.1.json')

print('train={}'.format(len(train_data)))
print('dev={}'.format(len(dev_data)))
all_words = train_words + dev_words

if(args.vocab_count>0):
    print("Using Vocab Count of {}".format(args.vocab_count))
    word_index, index_word = build_word_index(all_words, min_count=0,
                                                vocab_count=args.vocab_count,
                                                lower=False)
else:
    word_index, index_word = build_word_index(all_words, min_count=0,
                                                lower=False)

print("Vocab Size={}".format(len(word_index)))

# Convert passages to tokens
# passages = dict(train_passage.items() + test_passage.items() + dev_passage.items())

fp = './datasets/Squad/'

if not os.path.exists(fp):
    os.makedirs(fp)

build_embeddings(word_index, index_word,
  out_dir=fp,
  init_type='zero', init_val=0.01,
  emb_types=[('glove',300),('glove',100)],
  normalize=False)

passages = {}

env = {
    'train':train_data,
    'test':[],
    'dev':dev_data,
    'passages':passages,
    'word_index':word_index
}

feature_env = {
    'train':train_feats,
    'test':[],
    'dev':dev_feats
    }

dictToFile(env,'./datasets/Squad/env.gz'.format(mode))
dictToFile(feature_env,'./datasets/Squad/feats.gz'.format(mode))
