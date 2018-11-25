from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from .nn import *
from .compose_op import *
from .complex_op import *
from .cnn import *
from .sim_op import *
from .qrnn import *


def get_distance_biases(time_steps, reuse_weights=False, dist_bias=10):
	""" Return a 2-d tensor with the values of the distance biases to be applied
	on the intra-attention matrix of size sentence_size

	This is for intra-attention

	Args:
		time_steps: `tensor` scalar

	Returns:
		 2-d `tensor` (time_steps, time_steps)
	"""
	with tf.variable_scope('distance-bias', reuse=reuse_weights):
		# this is d_{i-j}
		distance_bias = tf.get_variable('dist_bias', [dist_bias],
										initializer=tf.zeros_initializer())

		# messy tensor manipulation for indexing the biases
		r = tf.range(0, time_steps)
		r_matrix = tf.tile(tf.reshape(r, [1, -1]),
						   tf.stack([time_steps, 1]))
		raw_inds = r_matrix - tf.reshape(r, [-1, 1])
		clipped_inds = tf.clip_by_value(raw_inds, 0,
										dist_bias - 1)
		values = tf.nn.embedding_lookup(distance_bias, clipped_inds)

	return values

def intra_attention(sentence, dim, initializer=None, activation=None,
					reuse=None, dist_bias=10, dropout=None,
					weights_regularizer=None, pooling='MATRIX'):
	''' Computes intra-attention

	Follows IA model of https://arxiv.org/pdf/1606.01933.pdf

	Args:
		sentence: `tensor` [bsz x time_steps x dim]
		dim: `int` projected dimensions
		initializer: tensorflow initializer
		activation: tensorflow activation (i.e., tf.nn.relu)
		reuse: `bool` To reuse params or not
		dist_bias: `int` value of dist bias
		dropout: Tensorflow dropout placeholder
		weights_regularizer: Regularization for projection layers

	Returns:
		attended: `tensor [bsz x timesteps x (dim+original dim)]
		attention: attention vector

	'''
	with tf.variable_scope('intra_att') as scope:
		time_steps = tf.shape(sentence)[1]
		dist_biases = get_distance_biases(time_steps, dist_bias=dist_bias,
											reuse_weights=reuse)
		sentence = projection_layer(sentence,
									dim,
									name='intra_att_proj',
									activation=activation,
									weights_regularizer=weights_regularizer,
									initializer=initializer,
									dropout=dropout,
									use_fc=False,
									num_layers=2,
									reuse=reuse)
		sentence2 = tf.transpose(sentence, [0,2,1])
		raw_att = tf.matmul(sentence, sentence2)
		raw_att += dist_biases
		attention = matrix_softmax(raw_att)
		attended = tf.matmul(attention, sentence)
		return tf.concat([sentence, attended], 2), attention


def mask_3d(values, sentence_sizes, mask_value, dimension=2):
	""" Given a batch of matrices, each with shape m x n, mask the values in each
	row after the positions indicated in sentence_sizes.
	This function is supposed to mask the last columns in the raw attention
	matrix (e_{i, j}) in cases where the sentence2 is smaller than the
	maximum.

	Source https://github.com/erickrf/multiffn-nli/

	Args:
		values: `tensor` with shape (batch_size, m, n)
		sentence_sizes: `tensor` with shape (batch_size) containing the
			sentence sizes that should be limited
		mask_value: `float` to assign to items after sentence size
		dimension: `int` over which dimension to mask values

	Returns
		A tensor with the same shape as `values`
	"""
	if dimension == 1:
		values = tf.transpose(values, [0, 2, 1])
	time_steps1 = tf.shape(values)[1]
	time_steps2 = tf.shape(values)[2]

	ones = tf.ones_like(values, dtype=tf.int32)
	pad_values = mask_value * tf.cast(ones, tf.float32)
	mask = tf.sequence_mask(sentence_sizes, time_steps2)

	# mask is (batch_size, sentence2_size). we have to tile it for 3d
	mask3d = tf.expand_dims(mask, 1)
	mask3d = tf.tile(mask3d, (1, time_steps1, 1))
	mask3d = tf.cast(mask3d, tf.float32)

	masked = values * mask3d
	# masked = tf.where(mask3d, values, pad_values)

	if dimension == 1:
		masked = tf.transpose(masked, [0, 2, 1])

	return masked

def matrix_softmax(values):
	''' Implements a matrix-styled softmax

	Args:
		values `tensor` [bsz x a_len, b_len]

	Returns:
		A tensor of the same shape
	'''
	original_shape = tf.shape(values)
	num_units = original_shape[2]
	reshaped = tf.reshape(values, tf.stack([-1, num_units]))
	softmaxed = tf.nn.softmax(reshaped)
	return tf.reshape(softmaxed, original_shape)

def softmax_mask(val, mask, value=-1E30):
	return value * (1 - tf.cast(mask, tf.float32)) + val

def pointer(inputs, state, hidden, mask, scope="pointer"):
	with tf.variable_scope(scope):
		u = tf.concat([tf.tile(tf.expand_dims(state, axis=1), [
			1, tf.shape(inputs)[1], 1]), inputs], axis=2)
		s0 = tf.nn.tanh(dense(u, hidden, use_bias=False, scope="s0"))
		s = dense(s0, 1, use_bias=False, scope="s")
		s1 = softmax_mask(tf.squeeze(s, [2]), mask)
		a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
		res = tf.reduce_sum(a * inputs, axis=1)
		return res, s1

def co_attention(input_a, input_b, reuse=False, name='', att_type='TENSOR',
				pooling='MEAN', k=10, mask_diag=False, kernel_initializer=None,
				dropout=None, activation=None, seq_lens=[], clipped=False,
				transform_layers=0, proj_activation=tf.nn.relu,
				dist_bias=0, gumbel=False, temp=0.5, hard=1,
				model_type="", mask_a=None, mask_b=None,
				val_transform=False, proj_dim=None, proj_nalu=False):
	''' Implements a Co-Attention Mechanism

	This attention uses tiling method (this uses more RAM, but enables
	MLP or special types of interaction functions between vectors.)

	Note: For self-attention, set input_a and input_b to be same tensor.

	Args:
		input_a: `tensor`. Shape=[bsz x max_steps x dim]
		input_b: `tensor`. Shape=[bsz x max_steps x dim]
		reuse:  `bool`. To reuse weights or not
		name:   `str`. Variable name
		att_type: `str`. Supports 'BILINEAR','TENSOR','MLP' and 'MD'
		pooling: 'str'. supports "MEAN",'MAX' and 'SUM' pooling
		k:  `int`. For multi-dimensional. Num_slice tensor or hidden
			layer.
		mask_diag: `bool` Supports masking against diagonal for self-att
		kernel_initializer: `Initializer function
		dropout: `tensor` dropout placeholder (default is disabled)
		activation: Activation function
		seq_lens: `list of 2 tensors` actual seq_lens for
			input_a and input_b

	Returns:
		final_a: `tensor` Weighted representation of input_a.
		final_b: `tensor` Weighted representation of input_b.
		max_row: `tensor` Row-based attention weights.
		max_col: `tensor` Col-based attention weights.
		y:  `tensor` Affinity matrix

	'''

	if(kernel_initializer is None):
		kernel_initializer = tf.random_uniform_initializer()

	if(len(input_a.get_shape().as_list())<=2):
		# expand dims
		input_a = tf.expand_dims(input_a, 2)
		input_b = tf.expand_dims(input_b, 2)
		readjust = True
	else:
		readjust = False

	# print(input_a)
	orig_a = input_a
	orig_b = input_b
	a_len = tf.shape(input_a)[1]
	b_len = tf.shape(input_b)[1]
	input_dim = tf.shape(input_a)[2]
	if(clipped):
		max_len = tf.reduce_max([tf.shape(input_a)[1],
								tf.shape(input_b)[2]])
	else:
		max_len = a_len

	shape = input_a.get_shape().as_list()
	if(proj_dim is not None):
		dim = proj_dim
		if(proj_dim!=dim):
			print("Proj dim is not equal Dim, setting val_trans to true.")
			val_transform = True
			if(transform_layers==0):
				transform_layers = 1
	else:
		dim = shape[2]

	if(dist_bias>0):
		time_steps = tf.shape(input_a)[1]
		dist_biases = get_distance_biases(time_steps, dist_bias=dist_bias,
											reuse_weights=reuse)

	if(proj_nalu==True):
		use_mode='NALU'
		proj_activation=None
	else:
		use_mode='None'

	if(transform_layers>=1):
		input_a = projection_layer(input_a,
								dim,
								name='att_proj_{}'.format(name),
								activation=proj_activation,
								initializer=kernel_initializer,
								dropout=None,
								reuse=reuse,
								num_layers=transform_layers,
								use_mode=use_mode)
		input_b = projection_layer(input_b,
								dim,
								name='att_proj_{}'.format(name),
								activation=proj_activation,
								reuse=True,
								initializer=kernel_initializer,
								dropout=None,
								num_layers=transform_layers,
								use_mode=use_mode)
		if(val_transform):
			val_a = projection_layer(input_a,
									dim,
									name='val_att_proj_{}'.format(name),
									activation=proj_activation,
									initializer=kernel_initializer,
									dropout=None,
									reuse=reuse,
									num_layers=transform_layers,
									use_mode=use_mode)
			val_b = projection_layer(input_b,
									dim,
									name='val_att_proj_{}'.format(name),
									activation=proj_activation,
									reuse=True,
									initializer=kernel_initializer,
									dropout=None,
									num_layers=transform_layers,
									use_mode=use_mode)
		else:
			val_a, val_b = input_a, input_b
	else:
		val_a, val_b = input_a, input_b

	if(att_type == 'BILINEAR'):
		# Bilinear Attention
		with tf.variable_scope('att_{}'.format(name), reuse=reuse) as f:
			weights_U = tf.get_variable("weights_U", [dim, dim],
										initializer=kernel_initializer)
		_a = tf.reshape(input_a, [-1, dim])
		z = tf.matmul(_a, weights_U)
		z = tf.reshape(z, [-1, a_len, dim])
		y = tf.matmul(z, tf.transpose(input_b, [0, 2, 1]))
	elif(att_type == 'TENSOR'):
		# Tensor based Co-Attention
		with tf.variable_scope('att_{}'.format(name),
								reuse=reuse) as f:
			weights_U = tf.get_variable(
					"weights_T", [dim, dim * k],
					initializer=kernel_initializer)
			_a = tf.reshape(input_a, [-1, dim])
			z = tf.matmul(_a, weights_U)
			z = tf.reshape(z, [-1, a_len * k, dim])
			y = tf.matmul(z, tf.transpose(input_b, [0, 2, 1]))
			y = tf.reshape(y, [-1, a_len, b_len, k])
			y = tf.reduce_max(y, 3)
	elif(att_type=='SOFT'):
		# Soft match without parameters
		_b = tf.transpose(input_b, [0,2,1])
		z = tf.matmul(input_a, _b)
		y = z
	elif(att_type=='DOT'):
		# print("Using DOT-ATT")
		input_a = projection_layer(input_a,
								dim,
								name='dotatt_{}'.format(name),
								activation=tf.nn.relu,
								initializer=kernel_initializer,
								dropout=None,
								reuse=reuse,
								num_layers=1,
								use_mode='None')
		input_b = projection_layer(input_b,
								dim,
								name='dotatt_{}'.format(name),
								activation=tf.nn.relu,
								reuse=True,
								initializer=kernel_initializer,
								dropout=None,
								num_layers=1,
								use_mode='None')
		_b = tf.transpose(input_b, [0,2,1])
		z = tf.matmul(input_a, _b)
		z = z / (dim ** 0.5)
		y = tf.reshape(z, [-1, a_len, b_len])
	else:
		a_aug = tf.tile(input_a, [1, b_len, 1])
		b_aug = tf.tile(input_b, [1, a_len, 1])
		output = tf.concat([a_aug, b_aug], 2)
		if(att_type == 'MLP'):
			# MLP-based Attention
			sim = projection_layer(output, 1,
								name='{}_co_att'.format(name),
								reuse=reuse,
								num_layers=1,
								activation=None)
			y = tf.reshape(sim, [-1, a_len, b_len])
		elif(att_type=='DOTMLP'):
			# Add dot product
			_dim = input_a.get_shape().as_list()[2]
			sim = projection_layer(output, 1,
								name='dotmlp',
								reuse=reuse,
								num_layers=1,
								activation=None)

			y = tf.reshape(sim, [-1, a_len, b_len])

	if(activation is not None):
		y = activation(y)

	if(mask_diag):
		# Create mask to prevent matching against itself
		mask = tf.ones([a_len, b_len])
		mask = tf.matrix_set_diag(mask, tf.zeros([max_len]))
		y = y * mask

	if(dist_bias>0):
		print("Adding Distance Bias..")
		y += dist_biases

	if(pooling=='MATRIX'):
		_y = tf.transpose(y, [0,2,1])
		if(mask_a is not None and mask_b is not None):
			mask_b = tf.expand_dims(mask_b, 1)
			mask_a = tf.expand_dims(mask_a, 1)
			# bsz x 1 x b_len
			mask_a = tf.tile(mask_a, [1, b_len, 1])
			mask_b = tf.tile(mask_b, [1, a_len, 1])
			val = -1E-30
			_y = softmax_mask(_y, mask_a, value=val)
			y = softmax_mask(y, mask_b, value=val)
		else:
			print("[Warning] Using Co-Attention without Mask!")

		att2 = tf.nn.softmax(_y)
		att1 = tf.nn.softmax(y)
		final_a = tf.matmul(att2, orig_a)
		final_b = tf.matmul(att1, orig_b)
		_a2 = att2
		_a1 = att1
	elif(pooling=='BIDAF'):
		""" Uses Pooling like Bidaf
		https://arxiv.org/pdf/1611.01603.pdf
		"""
		_y = tf.transpose(y, [0,2,1])
		mask_b = tf.expand_dims(mask_b, 1)
		mask_a = tf.expand_dims(mask_a, 1)
		# bsz x 1 x b_len
		mask_a = tf.tile(mask_a, [1, b_len, 1])
		mask_b = tf.tile(mask_b, [1, a_len, 1])
		_y = softmax_mask(_y, mask_a)
		y = softmax_mask(y, mask_b)
		att2 = tf.nn.softmax(_y)
		final_a = tf.matmul(att2, orig_a)   # this is U
		att1 = tf.nn.softmax(tf.reduce_max(y, 1))
		att1 = tf.expand_dims(att1, 2)
		final_b = att1 * input_b
		_a2 = att2
		_a1 = att1
	elif(pooling=='ALIGN'):
		y = tf.nn.softmax(y)
		_a = tf.matmul(y, orig_a)
		_b = tf.matmul(y, orig_b)
		final_b = tf.concat([_a, orig_b], 2)
		final_a = tf.concat([_b, orig_a], 2)
	elif(pooling=='DCN'):
		"""
		Dynamic Co-Attention Networks
		https://arxiv.org/pdf/1611.01604.pdf
		"""
		D = orig_a
		Q = orig_b
		_dim = D.get_shape().as_list()[2]
		_y = tf.reshape(y, [-1, b_len, a_len])

		AD = tf.nn.softmax(_y)
		AQ = tf.nn.softmax(y)
		CQ = tf.matmul(D, AQ, transpose_a=True)   # repr of Q
		CD = tf.matmul(CQ, AD)
		CD = tf.reshape(CD, [-1, a_len, _dim])
		CQ = tf.reshape(CQ, [-1, b_len, _dim])
		"""
		final_a is bsz x a_len x dim => QAD
		final_b is bsz x b_len x dim => CQ
		"""
		_a1 = AD
		_a2 = AQ
		final_a = CD
		final_b = CQ
	else:
		if(pooling=='MAX'):
			att_row = tf.reduce_max(y, 1)
			att_col = tf.reduce_max(y, 2)
		elif(pooling=='MIN'):
			att_row = tf.reduce_min(y, 1)
			att_col = tf.reduce_min(y, 2)
		elif(pooling=='SUM'):
			att_row = tf.reduce_sum(y, 1)
			att_col = tf.reduce_sum(y, 2)
		elif(pooling=='MEAN'):
			att_row = tf.reduce_mean(y, 1)
			att_col = tf.reduce_mean(y, 2)

		# Get attention weights
		if(gumbel):
			att_row = gumbel_softmax(att_row, temp, hard=hard)
			att_col = gumbel_softmax(att_col, temp, hard=hard)
		else:
			att_row = tf.nn.softmax(att_row)
			att_col = tf.nn.softmax(att_col)

		_a2 = att_row
		_a1 = att_col

		att_col = tf.expand_dims(att_col, 2)
		att_row = tf.expand_dims(att_row, 2)

		# Weighted Representations
		final_a = att_col * val_a
		final_b = att_row * val_b

	y = tf.reshape(y, tf.stack([-1, a_len, b_len]))

	if(dropout is not None):
		final_a = tf.nn.dropout(final_a, dropout)
		final_b = tf.nn.dropout(final_b, dropout)

	if(readjust):
		final_a = tf.squeeze(final_a, 2)
		final_b = tf.squeeze(final_b, 2)

	return final_a, final_b, _a1, _a2, y

def feat_compare(q1_compare, q2_compare,
				name='', reuse=None, compress='FM', factor=32,
				initializer=None, dropout=None):
	if(compress=='NFF'):
		# Use nonlinear layer baseline
		q1_fm = projection_layer(q1_compare,
				1,
				name='{}_compare_mlp'.format(name),
				activation=tf.nn.relu,
				initializer=initializer,
				dropout=dropout,
				reuse=reuse,
				num_layers=1)
		q2_fm = projection_layer(q2_compare,
				1,
				name='{}_compare_mlp'.format(name),
				activation=tf.nn.relu,
				initializer=initializer,
				dropout=dropout,
				reuse=True,
				num_layers=1)
	elif(compress=='SFF'):
		# Sum Encoding
		q1_fm = tf.reduce_sum(q1_compare, 2, keep_dims=True)
		q2_fm = tf.reduce_sum(q2_compare, 2, keep_dims=True)
	elif(compress=='FM'):
		# Factorization based encoding
		q1_fm, q1_latent = build_fm(q1_compare, k=factor,
							name='{}_com_fm'.format(name),
							initializer=initializer,
							reshape=True, reuse=reuse)
		q2_fm, q2_latent = build_fm(q2_compare, k=factor,
							name='{}_com_fm'.format(name),
							initializer=initializer, reuse=True,
							reshape=True)
	return q1_fm, q2_fm, q1_fm, q2_fm

def alignment_compare(q1, q2, _q1, _q2, reuse=None,
						feature_list=[], name='',
						factor=32,
						dropout=None, initializer=None,
						compress='FM'):
	""" Compares and build features between alignments

	q1 will compare with _q2
	q2 will compare with _q1
	"""

	features1 = []
	features2 = []
	latent_features1 = []
	latent_features2 = []

	# print(feature_list)
	for mode in feature_list:
		mname = name + mode
		if(mode=='CAT'):
			fv1 = tf.concat([q1, _q2], 2)
			fv2 = tf.concat([q2, _q1], 2)
			f1, f2, l1, l2 = feat_compare(fv1, fv2, name=mname,
									reuse=reuse, dropout=dropout,
									initializer=initializer,
									factor=factor,
									compress=compress)
		elif(mode=='MUL'):
			fv1 = q1 * _q2
			fv2 = q2 * _q1
			f1, f2, l1, l2 = feat_compare(fv1, fv2, name=mname,
									dropout=dropout,
									reuse=reuse,
									initializer=initializer,
									factor=factor,
									compress=compress)
		elif(mode=='SUB'):
			fv1 = q1 - _q2
			fv2 = q2 - _q1
			f1, f2, l1, l2 = feat_compare(fv1, fv2, name=mname,
									dropout=dropout,
									reuse=reuse, initializer=initializer,
									factor=factor,
									compress=compress)
		elif(mode=='ADD'):
			fv1 = q1 + _q2
			fv2 = q2 + _q1
			f1, f2, l1, l2 = feat_compare(fv1, fv2, name=mname,
									dropout=dropout,
									reuse=reuse, initializer=initializer,
									factor=factor,
									compress=compress)

		features1.append(f1)
		features2.append(f2)
		latent_features1.append(l1)
		latent_features2.append(l2)
	return features1, features2, latent_features1, latent_features2


def bidirectional_attention_connector(q1_embed,
        q2_embed, q1_len, q2_len, q1_max, q2_max,
		factor=32, factor2=32, reuse=None, name='', use_intra=True,
		feature_list='MUL_CAT_SUB', initializer=None, dropout=None,
		mask_a=None, mask_b=None, meta_aggregate=False, att_type='DOT',
		pooling='MATRIX', temp=0.5, compress='FM'):
	""" BAC layer
	"""

	features1 = []
	features2 = []
	la_feats1 = []
	la_feats2 = []

	# ___ Inter Attention Features
	_q1_embed, _q2_embed, _, _, afm = co_attention(
						q1_embed,
						q2_embed,
						att_type=att_type,
						pooling=pooling,
						mask_diag=False,
						kernel_initializer=initializer,
						activation=None,
						dropout=dropout,
						seq_lens=[q1_len, q2_len],
						transform_layers=1,
						name='{}_ca'.format(name),
						reuse=reuse,
						mask_a=mask_a,
						mask_b=mask_b,
						temp=0.5,
						hard=0,
						proj_nalu=False
						)

	feature_list = feature_list.split('_')

	f1, f2, l1, l2 = alignment_compare(q1_embed, q2_embed,
						 _q1_embed, _q2_embed,
						reuse=reuse,
						feature_list=feature_list,
						name='{}_inter'.format(name),
						initializer=initializer,
						factor=factor,
						dropout=dropout,
						compress=compress)
	features1 += f1
	features2 += f2
	la_feats1 += l1
	la_feats2 += l2

	# ___ Intra Attention
	if(use_intra):
		_i1_embed, _i2_embed =  intra_attention(
										q1_embed, q2_embed,
										q1_len, q2_len,
										att_type=att_type,
										name='intra_{}'.format(name),
										reuse=reuse,
										dist_bias=0,
										initializer=initializer,
										dropout=dropout,
										mask_a=mask_a,
										mask_b=mask_b,
										transform_layers=0,
										pooling=pooling
										)

		f1, f2, l1, l2 = alignment_compare(
								q1_embed, q2_embed,
								 _i2_embed, _i1_embed,
								reuse=reuse,
								factor=factor,
								feature_list=feature_list,
								name='{}_intra'.format(name),
								compress=compress
								)
		features1 += f1
		features2 += f2
		la_feats1 += l1
		la_feats2 += l2

	features1 = tf.concat(features1, 2)
	features2 = tf.concat(features2, 2)

	q2_embed = tf.concat([q2_embed, features2], 2)
	q1_embed = tf.concat([q1_embed, features1], 2)
	return q1_embed, q2_embed, [features1, features2]

def sample_gumbel(shape, eps=1e-20):
	U = tf.random_uniform(shape, minval=0, maxval=1)
	return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax(logits, temperature, hard=1):
	gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
	# print(gumbel_softmax_sample)
	y = tf.nn.softmax(gumbel_softmax_sample / temperature)

	if hard==1:
		k = tf.shape(logits)[-1]
		y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)),
						 y.dtype)
		y = tf.stop_gradient(y_hard - y) + y

	return y

def attention(inputs, context=None, reuse=False, name='',
			  kernel_initializer=None, dropout=None, gumbel=False,
			  actual_len=None, temperature=1.0, hard=1, reuse2=None,
			  return_raw=False):
	''' Implements Vanilla Attention Mechanism

	Context is for conditioned attentions (on last vector or topic
	vectors)

	Args:
		inputs: `tensor`. input seq of [bsz x timesteps x dim]
		context: `tensor`. input vector of [bsz x dim]
		reuse: `bool`. whether to reuse parameters
		kernel_initializer: intializer function
		dropout: tensor placeholder for dropout keep prob

	Returns:
		h_final: `tensor`. output representation [bsz x dim]
		att: `tensor`. vector of attention weights

	'''
	if(kernel_initializer is None):
		kernel_initializer = tf.random_uniform_initializer()

	shape = inputs.get_shape().as_list()
	dim = shape[2]
	# seq_len = shape[1]
	seq_len = tf.shape(inputs)[1]
	with tf.variable_scope('attention_{}'.format(name), reuse=reuse) as f:
		weights_Y = tf.get_variable(
			"weights_Y", [dim, dim], initializer=kernel_initializer,
					validate_shape=False)
		weights_w = tf.get_variable(
			"weights_w", [dim, 1], initializer=kernel_initializer,
					validate_shape=False)
		tmp_inputs = tf.reshape(inputs, [-1, dim])
		H = inputs
		Y = tf.matmul(tmp_inputs, weights_Y)
		Y = tf.reshape(Y, [-1, seq_len, dim])

	if(context is not None):
		# Add context for conditioned attention
		with tf.variable_scope('att_context_{}'.format(name), reuse=reuse2) as f:
			weights_h = tf.get_variable(
				"weights_h_{}".format(name), [dim, dim], initializer=kernel_initializer)
			Wh = tf.expand_dims(context, 1)
			Wh = tf.tile(Wh, [1, seq_len, 1], name='tiled_state')
			Wh = tf.reshape(Wh, [-1, dim])
			HN = tf.matmul(Wh, weights_h)
			HN = tf.reshape(HN, [-1, seq_len, dim])
			Y = tf.add(Y, HN)

	Y = tf.tanh(Y, name='M_matrix')
	Y = tf.reshape(Y, [-1, dim])
	a = tf.matmul(Y, weights_w)
	a = tf.reshape(a, [-1, seq_len])

	if(actual_len is not None):
		a = mask_zeros_1(a, actual_len, seq_len, expand=False)
	if(gumbel):
		a = gumbel_softmax(a, temperature, hard=hard)
	else:
		a = tf.nn.softmax(a, name='attention_vector')

	att = tf.expand_dims(a, 2)

	r = tf.reduce_sum(inputs * att, 1)

	h_final = r
	if(context is not None):
		# Projection Layer
		with tf.variable_scope('att_context_{}'.format(name),
												reuse=reuse2) as f:
			weights_P = tf.get_variable(
				"weights_P", [dim, dim], initializer=kernel_initializer)
			weights_X = tf.get_variable(
				"weights_X", [dim, dim], initializer=kernel_initializer)
			Wr = tf.matmul(r, weights_P)
			Wx = tf.matmul(context, weights_X)
			h_final = tf.tanh(tf.add(Wr, Wx))

	return h_final, att
