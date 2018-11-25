from __future__ import division
import gzip
import json
import numpy as np
from collections import Counter
import io

def get_ngrams(n, text):
  """Calcualtes n-grams.

  Args:
    n: which n-grams to calculate
    text: An array of tokens

  Returns:
    A set of n-grams
  """
  ngram_set = set()
  text_length = len(text)
  max_index_ngram_start = text_length - n
  for i in range(max_index_ngram_start + 1):
    ngram_set.add(tuple(text[i:i + n]))
  return ngram_set

def get_ngrams_with_ids(n, text):
  """Calcualtes n-grams.

  Args:
    n: which n-grams to calculate
    text: An array of tokens

  Returns:
    A set of n-grams
  """

  text_length = len(text)
  max_index_ngram_start = text_length - n
  output = []
  for i in range(max_index_ngram_start + 1):
    output.append([(i,i+n), text[i:i + n]])
  return output

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results

def dictToFile(dict,path):
	print("Writing to {}".format(path))
	with gzip.open(path, 'w') as f:
		f.write(json.dumps(dict))

def fast_load(path):
	'''
	Read js file:
	key ->  unicode keys
	string values -> unicode value
	'''
	print("Fast Loading {}".format(path))
	try:
		gz = gzip.open(path, 'rb')
		f = io.BufferedReader(gz)
		# with gzip.open(path, 'r') as f:
		return json.loads(f.read())
	except:
		print("Can't find Gzip. loading pure json")
		with open(path, 'r') as f:
			return json.loads(f.read())

def show_stats(name, x):
	print("{} max={} mean={} min={}".format(name, np.max(x),
										np.mean(x), np.min(x)))

def question_type(data):
	""" Returns question type
	"""
	data = data.lower()
	if('what' in data):
		return 0
	elif('where' in data):
		return 1
	elif('how' in data):
		return 2
	elif('why' in data):
		return 3
	elif('when' in data):
		return 4
	elif('who' in data):
		return 5
	elif('which' in data):
		return 6
	elif('is it' in data or 'is there' in data):
		return 7
	elif('can' in data):
		return 8
	elif('are' in data):
		return 9
	elif('do' in data):
		return 10
	else:
		return 11

def get_frequency(tokens, counter=None):
	if(counter is None):
		cnt = Counter(tokens)
	else:
		cnt = counter
	cnt_tokens = []
	for t in tokens:
		c = cnt[t]
		# normalized frequency
		c /= len(tokens)
		cnt_tokens.append(c)
	return cnt_tokens

def two_way_frequency(tokens_a, tokens_b):
	tokens_a = tokens_a.split(' ')
	tokens_b = tokens_b.split(' ')
	total_tokens = tokens_a + tokens_b
	cnt = Counter(total_tokens)
	cnt_a, cnt_b = [], []
	cnt_a = get_frequency(tokens_a, counter=cnt)
	cnt_b = get_frequency(tokens_b, counter=cnt)
	return cnt_a, cnt_b
