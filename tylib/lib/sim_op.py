from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .att_op import *
from .nn import *

def gesd(x, y, eps=1E-6):
    """ Computes geometric mean of sigmoid and euclidean

    Reference - http://bingning.wang/site_media/download/ACL2016.pdf
    """
    dot = tf.reduce_sum(x * y, 1, keep_dims=True) + 1.0
    denom = 1.0 + tf.exp(-dot)
    sig_dot = tf.div(1.0, denom + eps)

    denom2 = tf.reduce_sum(tf.abs(x-y), 1, keep_dims=True) + 1.0
    euclid = tf.div(1.0, denom2 + eps)

    output = sig_dot * euclid
    return output

def cosine_similarity(x, y, axis=1):
    ''' Implements a simple cosine similarity function

    Args:
        x: `tensor`. input vec of [bsz x dim]
        y: `tensor`. input vec of [bsz x dim]

    Returns:
        dist: `tensor: output vec of [bsz x 1]
    '''
    x = tf.nn.l2_normalize(x, axis)
    y = tf.nn.l2_normalize(y, axis)
    dist = tf.reduce_sum(tf.multiply(x,y), axis=axis,
                keep_dims=True,
                name='cos_sim')
    return dist


def euc_cos(a, b):
    cos_sim = cosine_similarity(a, b, axis=2)
    euclid = tf.sqrt(tf.reduce_sum(tf.square(a-b),2,
                            keep_dims=True)) + 1E-5
    return tf.concat([cos_sim, euclid], 2)

def sub_mult_nn(a,b, name='', reuse=None, init=None):
    left = (a - b) * (a-b)
    right = a * b
    _dim = a.get_shape().as_list()[2]
    c = tf.concat([left, right], 2)
    output = projection_layer(c, _dim,
            name='submultNN',
            reuse=reuse,
            activation=tf.nn.relu,
            initializer=init,
            dropout=None, use_mode='FC',
            num_layers=1)
    return output

def match_compare(input_a, input_b):
    """ Match compare
    """
    cos_sim = cosine_similarity(input_a, input_b, axis=2)
    # print(cos_sim)
    feat_vec = cos_sim
    diff = input_a - input_b
    euclid = tf.sqrt(tf.reduce_sum(tf.square(diff),2,
                            keep_dims=True)) + 1E-5

    mul = input_a * input_b
    sub = input_a - input_b
    # print(euclid)
    # _mul = _ce * _c
    feat_vec = mul
    # feat_vec = tf.concat([mul, sub], 2)
    return feat_vec
