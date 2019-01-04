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
import csv
from utilities import *
from utils import *
from nltk.stem.porter import PorterStemmer
import argparse
import operator
import random
from nltk.corpus import stopwords
from nus_utilities import *
from common_v2 import *
import sys
from rouge import *

reload(sys)
sys.setdefaultencoding('utf8')
sys.dont_write_bytecode = True

""" Script to convert NarrativeQA to Answer Span-prediction
Following the original paper, we use the best rouge-L matching score
for span selection.
"""

# Load passages / summaries
passages = {}
with open('./corpus/narrativeqa/third_party/wikipedia/summaries.csv','r') as f:
    reader = csv.reader(f, delimiter=',')
    reader.next()
    for r in reader:
        passages[r[0]] = r[3]

print("Collected {} Passages".format(len(passages)))

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results

max_span = 6

fp = './corpus/narrativeqa/qaps.csv'
stoplist = set(['the','a','.',','])
train, dev, test = [],[],[]
ignored_train, ignored_eval = 0, 0

with open(fp, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    reader.next()   # Skip Header
    for idx, r in tqdm(enumerate(reader)):
        # if(idx>500):
        #     break
        # data = {}
        _id = r[0]
        set_type = r[1]
        # print(set_type)
        question = r[5]
        answer1 = r[6]
        answer2 = r[7]
        passage =passages[_id]
        p = passage.split(' ')
        results = {}
        for i in range(0, max_span):
            ngrams = get_ngrams(i+1, p)
            ngrams = [' '.join(x) for x in ngrams]
            for n in ngrams:
                r = Rouge().calc_score([n], [answer1, answer2])
                if(r>0):
                    results[n] = r
        sorted_results = sorted(results.items(),
                        key=operator.itemgetter(1), reverse=True)

        new_results = []
        for s in sorted_results:
            if(s[0].lower() in stoplist):
                continue
            else:
                new_results.append(s[0])
        if(len(new_results)==0):
            if(set_type=='train'):
                ignored_train +=1
                continue
            else:
                # dummy ans for dev/test
                rand = random.randint(0, len(p)-2)
                choosen_ans = p[rand:rand+1]
                ignored_eval +=1
        else:
            choosen_ans = new_results[0].split(' ')
        spans = find_sub_list(choosen_ans, p)
        # train_labels = [new_results[0], spans]
        span_ans = p[spans[0][0]:spans[0][1]+1]
        ans_str = ' '.join(choosen_ans)
        assert(' '.join(span_ans)==ans_str)

        answers = [ans_str, spans]
        # print(train_labels)
        data = {
            '_id':_id,
            'question.tokens':question,
            'ground_truths':[answer1, answer2],
            'context.tokens':passage,
            'answers':answers
        }
        if(set_type=='train'):
            train.append(data)
        elif(set_type=='valid'):
            dev.append(data)
        elif(set_type=='test'):
            test.append(data)

print("Train={} Dev={} Test={}".format(
                            len(train), len(dev),
                            len(test)))
print("Ignored Train={} Ignored Eval={}".format(ignored_train,
                                            ignored_eval))


with open('./corpus/narrativeqa/train.json', 'w') as f:
    f.write(json.dumps(train,  indent=4,))

with open('./corpus/narrativeqa/dev.json', 'w') as f:
    f.write(json.dumps(dev,  indent=4,))

with open('./corpus/narrativeqa/test.json', 'w') as f:
    f.write(json.dumps(test,  indent=4,))
