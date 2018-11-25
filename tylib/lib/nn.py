from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from .compose_op import *
from .cudnn_cove_lstm import *

''' Random utilities
'''

def gather_tree_py(values, parents):
  """Gathers path through a tree backwards from the leave nodes. Used
  to reconstruct beams given their parents."""
  beam_length = values.shape[0]
  num_beams = values.shape[1]
  res = np.zeros_like(values)
  res[-1, :] = values[-1, :]
  for beam_id in range(num_beams):
    parent = parents[-1][beam_id]
    for level in reversed(range(beam_length - 1)):
      res[level, beam_id] = values[level][parent]
      parent = parents[level][parent]
  return np.array(res).astype(values.dtype)

def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def gather_tree(values, parents):
  """Tensor version of gather_tree_py"""
  res = tf.py_func(
      func=gather_tree_py, inp=[values, parents], Tout=values.dtype)
  res.set_shape(values.get_shape().as_list())
  return res

def mask_probs(probs, end_token, finished):
    """
    Args:
        probs: tensor of shape [batch_size, beam_size, vocab_size]
        end_token: (int)
        finished: tensor of shape [batch_size, beam_size], dtype = tf.bool
    """
    # one hot of shape [vocab_size]
    vocab_size = probs.shape[-1].value
    one_hot = tf.one_hot(end_token, vocab_size, on_value=0.,
            off_value=probs.dtype.min, dtype=probs.dtype)
    # expand dims of shape [batch_size, beam_size, 1]
    finished = tf.expand_dims(tf.cast(finished, probs.dtype), axis=-1)

    return (1. - finished) * probs + finished * one_hot

def gather_helper(t, indices, batch_size, beam_size):
    """
    Args:
        t: tensor of shape = [batch_size, beam_size, d]
        indices: tensor of shape = [batch_size, beam_size]
    Returns:
        new_t: tensor w shape as t but new_t[:, i] = t[:, new_parents[:, i]]
    """
    range_  = tf.expand_dims(tf.range(batch_size) * beam_size, axis=1)
    indices = tf.reshape(indices + range_, [-1])
    output  = tf.gather(
        tf.reshape(t, [batch_size*beam_size, -1]),
        indices)

    if t.shape.ndims == 2:
        return tf.reshape(output, [batch_size, beam_size])

    elif t.shape.ndims == 3:
        d = t.shape[-1].value
        return tf.reshape(output, [batch_size, beam_size, d])

def get_position_encoding(
    length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.
    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.
    Args:
    length: Sequence length.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position
    Returns:
    Tensor with shape [length, hidden_size]
    """
    import math
    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal

def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.
    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    N, T = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units**0.5

        return outputs

def normalize(inputs,
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta

    return outputs

def pos_feedforward(inputs,
                num_units=[2048, 512],
                scope="pos_ffn",
                reuse=None, residual=True,
                num_layers=2):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        for i in range(num_layers-1):
        # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

        # Residual connection
        if(residual==True):
            outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs

def highway_layer(input_data, dim, init, name='', reuse=None,
                    activation=tf.nn.relu):
    """ Creates a highway layer
    """
    print("Constructing highway layer..")
    trans = linear(input_data, dim, init,  name='trans_{}'.format(name),
                        reuse=reuse)
    trans = activation(trans)
    gate = linear(input_data, dim, init, name='gate_{}'.format(name),
                        reuse=reuse)
    gate = tf.nn.sigmoid(gate)
    if(dim!=input_data.get_shape()[-1]):
        input_data = linear(input_data, dim, init,name='trans2_{}'.format(name),
                            reuse=reuse)
    output = gate * trans + (1-gate) * input_data
    return output

def ffn(input_data, dim, initializer, name='', reuse=None, num_layers=2,
        dropout=None, activation=None):
    for i in range(num_layers):
        input_data = linear(input_data, dim, initializer,
                            name='{}_{}'.format(name, i), reuse=reuse)
        if(activation is not None):
            input_data = activation(input_data)
        if(dropout is not None):
            input_data = tf.nn.dropout(input_data, dropout)
    return input_data

def linear(input_data, dim, initializer, name='', reuse=None,
            bias=True, bias_val=0.1):
    """ Default linear layer
    """
    input_shape = input_data.get_shape().as_list()[1]
    with tf.variable_scope('linear', reuse=reuse) as scope:
        _weights = tf.get_variable(
                "W_{}".format(name),
                shape=[input_shape, dim],
                initializer=initializer)
        if(bias==True):
            _bias = tf.get_variable('bias_{}'.format(name),
                    shape=[dim],
                    initializer=tf.constant_initializer([bias_val]))
            output_data = tf.nn.xw_plus_b(input_data, _weights, _bias)
        else:
            output_data = tf.matmul(input_data, _weights)
    return output_data

def mask_zeros_1(embed, lens, max_len, expand=True):
    mask = tf.sequence_mask(lens, max_len)
    mask = tf.cast(mask, tf.float32)
    if(expand):
        mask = tf.expand_dims(mask, 2)
    embed = embed * mask
    return embed

def mask_zeros(embed, lens, max_len, expand_dims=-1):
    # Needs refactor
    mask = tf.sequence_mask(lens, max_len)
    mask = tf.cast(mask, tf.float32)
    if(expand_dims>0):
        mask = tf.expand_dims(mask, expand_dims)
    embed = embed * mask
    return embed

def mask_dim(embed, lens, max_len, expand_dims=[1]):
    # Needs refactor
    mask = tf.sequence_mask(lens, max_len)
    mask = tf.cast(mask, tf.float32)
    for e in expand_dims:
        mask = tf.expand_dims(mask, e)
    embed = embed * mask
    return embed

def hierarchical_flatten(embed, lengths, smax):
    """ Flattens embedding for hierarchical processing.

    Args:
        embed: `tensor` [bsz x (num_docs * seq_len) x dim]
        lengths: `tensor` [bsz x num_docs]
        smax: `int` - maximum number of words in sentence

    Returns:
        embed: `tensor` [bsz x seq_len x dim] flattened input
        lengths: `tensor` [bsz] flattend lengths
    """

    _dims = embed.get_shape().as_list()[2]
    embed = tf.reshape(embed, [-1, smax, _dims])
    lengths = tf.reshape(lengths, [-1])
    return embed, lengths

def input_encoder(embeddings, inputs_list, dropout=None,
                proj=0, proj_dim=None, init=None, name="",
                reuse=None, proj_mode='FC', num_proj=1,
                extras=[], is_train=None, use_cove=0):
    ''' New Wrapper in same style as embed_and_dropout

    Supports adding extra features like char or EM match.
    '''
    if(is_train is None):
        is_train = tf.constant(False)

    output_list = []
    if(use_cove):
        print("Using CovE")
        cove_lstm = load_cudnn_cove_lstm()
    with tf.variable_scope('input_encoder'):
        for idx, _input in enumerate(inputs_list):
            embed = tf.nn.embedding_lookup(embeddings, _input)
            if(use_cove):
                cove_embed = cove_lstm(embed)
                embed = tf.concat([embed, cove_embed], 2)
            if(dropout is not None and dropout <1.0):
                # embed = tf.nn.dropout(embed, dropout)
                embed = dropoutz(embed, dropout, is_train,
                            mode='recurrent')
            if(len(extras)>0):
                _extra = extras[idx]
                for extra in _extra:
                    if(extra is not None):
                        embed = tf.concat([embed, extra], 2)
            print("[Input Encoder] Dim={}".format(
                        embed.get_shape().as_list()[2]))
            if(proj==1):
                print("Projecting Embedding..")
                embed = projection_layer(embed,
                                        proj_dim,
                                        name='embed_proj',
                                        activation=tf.nn.relu,
                                        initializer=init,
                                        dropout=dropout,
                                        use_mode=proj_mode,
                                        reuse=reuse,
                                        is_train=is_train,
                                        num_layers=num_proj)
            output_list.append(embed)
    return output_list

def dropoutz(args, keep_prob, is_train, mode="recurrent"):
    if(keep_prob is None):
        return args
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args


def embed_and_dropout(embeddings, inputs_list, dropout=None,
                        proj=0, proj_dim=None, init=None, name="",
                        reuse=None, proj_mode='FC', num_proj=1):
    ''' Passes through all inputs into embedding layer

    Convenience wrapper for embeddings

    Args:
        embeddings: `tensor` [vocab x dim]
        inputs_list: `list` of tensors each of [bsz x time_steps]
        dropout: tensorflow dropout placeholder

    Returns:
        output_list: `list` of tensors each of [bsz x time_steps x dim]
    '''

    output_list = []
    with tf.variable_scope('embedding_lookup'):
        for _input in inputs_list:
            embed = tf.nn.embedding_lookup(embeddings, _input)
            if(dropout is not None):
                embed = tf.nn.dropout(embed, dropout)
            if(proj==1):
                print("Projecting Embedding..[{}]".format(proj_mode))
                embed = projection_layer(embed,
                                        proj_dim,
                                        name='embed_proj',
                                        activation=tf.nn.relu,
                                        initializer=init,
                                        dropout=dropout,
                                        use_mode=proj_mode,
                                        reuse=reuse,
                                        num_layers=num_proj)
            output_list.append(embed)
    return output_list

def feed_forward(inputs, output_dim, name='', initializer=None):
    """ Simple Single Layer Feed-Forward
    """
    _dim = inputs.get_shape().as_list()[1]
    weights = tf.get_variable('weights_{}'.format(name),
                              [_dim, output_dim],
                              initializer=initializer)
    zero_init = tf.zeros_initializer()
    bias = tf.get_variable('bias_{}'.format(name), shape=output_dim,
                                dtype=tf.float32,
                                initializer=zero_init)
    output = tf.nn.xw_plus_b(inputs, weights, bias)
    return output

def projection_layer(inputs, output_dim, name='', reuse=None,
                    activation=None, weights_regularizer=None,
                    initializer=None, dropout=None, use_mode='FC',
                    num_layers=2, mode='', return_weights=False,
                    is_train=False):
    """ Simple Projection layer

    Args:
        x: `tensor`. vectors to be projected
            Shape is [batch_size x time_steps x emb_size]
        output_dim: `int`. dimensions of input embeddings
        rname: `str`. variable scope name
        reuse: `bool`. whether to reuse parameters within same
            scope
        activation: tensorflow activation function
        initializer: initializer
        dropout: dropout placeholder
        use_fc: `bool` to use fc layer api or matmul
        num_layers: `int` number layers of projection

    Returns:
        A 3D `Tensor` of shape [batch, time_steps, output_dim]
    """
    # input_dim = tf.shape(inputs)[2]
    if(initializer is None):
        initializer = tf.contrib.layers.xavier_initializer()
    shape = inputs.get_shape().as_list()
    if(len(shape)==3):
        input_dim = inputs.get_shape().as_list()[2]
        time_steps = tf.shape(inputs)[1]
    else:
        input_dim = inputs.get_shape().as_list()[1]
    with tf.variable_scope('proj_{}'.format(name), reuse=reuse) as scope:
        x = tf.reshape(inputs, [-1, input_dim])
        output = x
        for i in range(num_layers):
            if(dropout is not None and dropout < 1.0):
                output = dropoutz(output, dropout, is_train)
            _dim = output.get_shape().as_list()[1]
            if(use_mode=='FC'):
                weights = tf.get_variable('weights_{}'.format(i),
                              [_dim, output_dim],
                              initializer=initializer)
                zero_init = tf.zeros_initializer()
                bias = tf.get_variable('bias_{}'.format(i), shape=output_dim,
                                            dtype=tf.float32,
                                            initializer=zero_init)
                output = tf.nn.xw_plus_b(output, weights, bias)
            elif(use_mode=='HIGH'):
                output = highway_layer(output, output_dim, initializer,
                                name='proj_{}'.format(i), reuse=reuse)
            elif(use_mode=='NALU'):
                output = nalu(output, output_dim, initializer, reuse=reuse,
                                name='proj_{}'.format(i))
            else:
                weights = tf.get_variable('weights_{}_{}'.format(i, name),
                              [_dim, output_dim],
                              initializer=initializer)
                output = tf.matmul(output, weights)
            if(activation is not None and use_mode!='HIGH'):
                output = activation(output)


        if(len(shape)==3):
            output = tf.reshape(output, [-1, time_steps, output_dim])


        return output
