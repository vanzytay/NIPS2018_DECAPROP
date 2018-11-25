from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

def eval_to_json(eval_list, fp):
	""" Evaluation scores to json file
	"""
	with open(fp, 'w+') as f:
		json.dump(eval_list)

