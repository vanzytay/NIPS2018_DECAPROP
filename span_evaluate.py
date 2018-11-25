""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import numpy as np

from tylib.pycocoevalcap.bleu.bleu import Bleu
from tylib.pycocoevalcap.rouge.rouge import Rouge
from tylib.pycocoevalcap.cider.cider import Cider
from tylib.pycocoevalcap.meteor.meteor import Meteor


def get_ans_string_single_post_pad_search_updated(context, context_words,
						  ans_start_pred, ans_end_pred,
						  maxspan=15, align_spans=None,
						  spans2=None, return_idx=False,
						  return_score=False):
	if(align_spans is not None and spans2 is not None):
		error = 0
		start_ind = spans2[0]
		end_ind = spans2[1]
		try:
			s = align_spans[start_ind]
			e = align_spans[end_ind]
		except:
			s = align_spans[-1]
			e = align_spans[-1]
			error = 1
		try:
			s = s[0]
		except:
			print("Can't find S")
			s=0
		try:
			e = e[1]
		except:
			print("Can't find E")
			e = 1

		return context[s:e], error

	ans_start = ans_start_pred[:len(context_words)]
	ans_end = ans_end_pred[:len(context_words)]
	p = np.zeros((len(context_words), len(context_words)))
	for i in range(len(context_words)):
		for j in range(i, min(i + maxspan, len(context_words))):
			p[i, j] = ans_start[i] * ans_end[j]
	loc = np.argmax(p)
	start_ind = int(loc / len(context_words))
	end_ind = loc - start_ind * len(context_words)
	indices = [start_ind, end_ind]

	if(return_idx):
		return ''.join(context_words[indices[0]:indices[1]+1]), 0

	context = context.replace("``", '"').replace("''", '"')
	char_idx = 0
	char_start, char_stop = None, None
	for word_idx, word in enumerate(context_words):
		word = word.replace("``", '"').replace("''", '"')
		# print word
		char_idx = context.find(word, char_idx)
		assert char_idx >= 0
		if word_idx == indices[0]:
			char_start = char_idx
		char_idx += len(word)
		if word_idx == indices[1]:
			char_stop = char_idx
			break

	assert char_start is not None
	assert char_stop is not None

	if(return_score):
		score = np.max(p)
		return context[char_start:char_stop], 1, score
	else:
		return context[char_start:char_stop], 1

def apply_alignment(data, align):
	new_data = []
	for i in range(len(data)):
		_align = align[i]
		_data = data[i]
		new_data.append(_data[_align[0]:_align[1]])
	return new_data

def adjust_passages(passages, ptrs, limits, span=150):
	""" Crop passages based on answers
	"""
	data = zip(passages, ptrs)
	new_passages = []
	new_spans = []
	span = span
	align = []
	for d in data:
		original = d[0][d[1]]
		ptr_left = max(0, int(d[1] - span))
		ptr_right = int(d[1] + span + 1)
		#print('{}:{}'.format(ptr_left, ptr_right))
		p = d[0][ptr_left:ptr_right]
		new_passages.append(p)
		ans = max(0, span)
		if(ptr_left==0):
			ans = d[1]
		assert(original==p[ans])
		new_spans.append(ans)
		align.append([ptr_left, ptr_right])
	return new_passages, new_spans, align

def normalize_answer(s):
	"""Lower text and remove punctuation, articles and extra whitespace."""
	def remove_articles(text):
		return re.sub(r'\b(a|an|the)\b', ' ', text)

	def white_space_fix(text):
		return ' '.join(text.split())

	def remove_punc(text):
		exclude = set(string.punctuation)
		return ''.join(ch for ch in text if ch not in exclude)

	def lower(text):
		return text.lower()

	return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
	prediction_tokens = normalize_answer(prediction).split()
	ground_truth_tokens = normalize_answer(ground_truth).split()
	#print('======================================')
	#print(prediction_tokens)
	#print(ground_truth_tokens)
	common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
	num_same = sum(common.values())
	if num_same == 0:
		return 0
	precision = 1.0 * num_same / len(prediction_tokens)
	recall = 1.0 * num_same / len(ground_truth_tokens)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1

# def meteor_score(prediction, ground_truth):
# 	prediction = {'dummy':[prediction]}
# 	ground_truth = {'dummy':[ground_truth]}
# 	score, _ = Meteor().compute_score(prediction, ground_truth)
# 	return score

def batch_meteor_score(ref, hyp):
	score, _ = Meteor().compute_score(ref, hyp)
	return score

def batch_bleu_score(ref, hyp, n=4):
	score, _ = Bleu(n=n).compute_score(ref, hyp)
	return score

def batch_rouge_score(ref, hyp):
	score, _ = Rouge().compute_score(ref, hyp)
	return score

def bleu_score4(prediction, ground_truth, progress=True):
	prediction = {'dummy':[prediction]}
	ground_truth = {'dummy':[ground_truth]}
	score, _ = Bleu(n=4).compute_score(prediction, ground_truth, progress=False)
	return score

def bleu_score1(prediction, ground_truth):
	prediction = {'dummy':[prediction]}
	ground_truth = {'dummy':[ground_truth]}
	score, _ = Bleu(n=1).compute_score(prediction, ground_truth)
	return score

def rouge_score(prediction, ground_truth):
	return Rouge().calc_score([prediction], [ground_truth])

def exact_match_score(prediction, ground_truth):
	return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
	scores_for_ground_truths = []
	for ground_truth in ground_truths:
		score = metric_fn(prediction, ground_truth)
		scores_for_ground_truths.append(score)
	return max(scores_for_ground_truths)

def evaluate(dataset, predictions):
	f1 = exact_match = total = 0
	for article in dataset:
		for paragraph in article['paragraphs']:
			for qa in paragraph['qas']:
				total += 1
				if qa['id'] not in predictions:
					message = 'Unanswered question ' + qa['id'] + \
							  ' will receive score 0.'
					print(message, file=sys.stderr)
					continue
				ground_truths = list(map(lambda x: x['text'], qa['answers']))
				prediction = predictions[qa['id']]
				exact_match += metric_max_over_ground_truths(
					exact_match_score, prediction, ground_truths)
				f1 += metric_max_over_ground_truths(
					f1_score, prediction, ground_truths)

	exact_match = 100.0 * exact_match / total
	f1 = 100.0 * f1 / total

	return {'exact_match': exact_match, 'f1': f1}


if __name__ == '__main__':
	expected_version = '1.1'
	parser = argparse.ArgumentParser(
		description='Evaluation for SQuAD ' + expected_version)
	parser.add_argument('dataset_file', help='Dataset file')
	parser.add_argument('prediction_file', help='Prediction File')
	args = parser.parse_args()
	with open(args.dataset_file) as dataset_file:
		dataset_json = json.load(dataset_file)
		if (dataset_json['version'] != expected_version):
			print('Evaluation expects v-' + expected_version +
				  ', but got dataset with v-' + dataset_json['version'],
				  file=sys.stderr)
		dataset = dataset_json['data']
	with open(args.prediction_file) as prediction_file:
		predictions = json.load(prediction_file)
	print(json.dumps(evaluate(dataset, predictions)))
