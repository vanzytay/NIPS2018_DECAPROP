# # Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================
# """Cudnn RNN operators."""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# from tensorflow.contrib.cudnn_rnn.ops import gen_cudnn_rnn_ops
#
# # from tensorflow.contrib.cudnn_rnn_ops import cudnn_rnn_canonical_to_opaque_params
# from tensorflow.contrib.rnn.python.ops import lstm_ops
# from tensorflow.contrib.util import loader
# from tensorflow.python.framework import common_shapes
# from tensorflow.python.framework import dtypes
# from tensorflow.python.framework import ops
# from tensorflow.python.framework import random_seed
# from tensorflow.python.layers import base as base_layer
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import init_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import rnn_cell_impl
# from tensorflow.python.ops import state_ops
# from tensorflow.python.ops import variable_scope as vs
# from tensorflow.python.platform import resource_loader
# from tensorflow.python.training import saver
# import tensorflow as tf
#
# """ (TY) TODO: This shouldn't be here, but I could not find a way
# to get around the missing lack of canonical_to_params missing in
# TF v1.7. This is ripped from Tensorflow v1.4 to make it
# work with tylib.cudnn_cove_lstm.
# """
#
# CUDNN_RNN_UNIDIRECTION = "unidirectional"
# CUDNN_RNN_BIDIRECTION = "bidirectional"
# CUDNN_LSTM = "lstm"
# CUDNN_GRU = "gru"
# CUDNN_RNN_RELU = "rnn_relu"
# CUDNN_RNN_TANH = "rnn_tanh"
#
# # Half for cell input, half for hidden states.
# CUDNN_LSTM_PARAMS_PER_LAYER = 8
# CUDNN_GRU_PARAMS_PER_LAYER = 6
# CUDNN_RNN_TANH_PARAMS_PER_LAYER = 2
# CUDNN_RNN_RELU_PARAMS_PER_LAYER = 2
#
# CUDNN_INPUT_LINEAR_MODE = "linear_input"
# CUDNN_INPUT_SKIP_MODE = "skip_input"
# CUDNN_INPUT_AUTO_MODE = "auto_select"
#
# class _CudnnRNN(object):
#   """Creates an RNN model using the underlying Cudnn implementation.
#
#   Note that self._NUM_PARAMS_PER_LAYER is the number of parameter sets of
#   weight and bias per layer. It needs to be defined in subclasses.
#   """
#   # __doc__ += _cudnn_rnn_common_doc_string
#
#   # TODO(jamesqin): support float16 CuDNN RNN
#   def __init__(self,
#                rnn_mode,
#                num_layers,
#                num_units,
#                input_size,
#                input_mode=CUDNN_INPUT_LINEAR_MODE,
#                direction=CUDNN_RNN_UNIDIRECTION,
#                dtype=dtypes.float32,
#                dropout=0.,
#                seed=0):
#     """Creates a CudnnRNN model from model spec.
#
#     Args:
#       rnn_mode: a string specifies the mode, under which this RNN model runs.
#           Could be either 'lstm', 'gru', 'rnn_tanh' or 'rnn_relu'.
#       num_layers: the number of layers for the RNN model.
#       num_units: the number of units within the RNN model.
#       input_size: the size of the input, it could be different from the
#           num_units.
#       input_mode: indicate whether there is a linear projection between the
#           input and the actual computation before the first layer. It could be
#           'linear_input', 'skip_input' or 'auto_select'.
#           'linear_input' (default) always applies a linear projection of input
#           onto RNN hidden state. (standard RNN behavior).
#           'skip_input' is only allowed when input_size == num_units;
#           'auto_select' implies 'skip_input' when input_size == num_units;
#           otherwise, it implies 'linear_input'.
#       direction: the direction model that the model operates. Could be either
#           'unidirectional' or 'bidirectional'
#       dtype: dtype of params, tf.float32 or tf.float64.
#       dropout: whether to enable dropout. With it is 0, dropout is disabled.
#       seed: the op seed used for initializing dropout. See @{tf.set_random_seed}
#           for behavior.
#     Raises:
#       ValueError: if direction is invalid.
#     """
#     self._num_layers = num_layers
#     self._num_units = num_units
#     self._input_size = input_size
#     self._rnn_mode = rnn_mode
#     self._input_mode = input_mode
#     self._direction = direction
#     self._dtype = dtype
#     self._dropout = dropout
#     self._seed = seed
#
#   @property
#   def input_mode(self):
#     return self._input_mode
#
#   @property
#   def input_size(self):
#     return self._input_size
#
#   @property
#   def num_units(self):
#     return self._num_units
#
#   @property
#   def num_layers(self):
#     return self._num_layers
#
#   @property
#   def rnn_mode(self):
#     return self._rnn_mode
#
#   @property
#   def direction(self):
#     return self._direction
#
#   def params_size(self):
#     """Calculates the size of the opaque parameter buffer needed for this model.
#
#     Returns:
#       The calculated parameter buffer size.
#     """
#     return cudnn_rnn_opaque_params_size(
#         rnn_mode=self._rnn_mode,
#         num_layers=self._num_layers,
#         num_units=self._num_units,
#         input_size=self._input_size,
#         dtype=self._dtype,
#         dropout=self._dropout,
#         seed=self._seed,
#         input_mode=self._input_mode,
#         direction=self._direction)
#
#   def __call__(self, input_data, input_h, input_c, params, is_training=True):
#     """Runs the forward step for the RNN model.
#
#     Args:
#       input_data: the input sequence to the RNN model. A Tensor of shape [?,
#         batch_size, input_size].
#       input_h: the initial hidden state for h. A Tensor of shape [num_layers,
#         batch_size, num_units].
#       input_c: the initial hidden state for c. This is only relevant for LSTM.
#         A Tensor of the same shape as input_h.
#       params: the parameter buffer created for this model.
#       is_training: whether this operation will be used in training or inference.
#     Returns:
#       output: the output sequuence.
#       output_h: the final state for h.
#       output_c: the final state for c. This is only relevant for LSTM.
#     """
#     return _cudnn_rnn(
#         input_data,
#         input_h,
#         input_c,
#         params,
#         is_training,
#         self._rnn_mode,
#         input_mode=self._input_mode,
#         direction=self._direction,
#         dropout=self._dropout,
#         seed=self._seed)
#
#   def params_to_canonical(self, params):
#     """Converts params from a specific format of cuDNN to the canonical format.
#
#     Args:
#       params: a Variable for weight and bias parameters.
#
#     Returns:
#       A function for the specific-to-canonical conversion.
#     """
#     return cudnn_rnn_opaque_params_to_canonical(
#         rnn_mode=self._rnn_mode,
#         num_layers=self._num_layers,
#         num_units=self._num_units,
#         input_size=self._input_size,
#         params=params,
#         input_mode=self._input_mode,
#         direction=self._direction,
#         dropout=self._dropout,
#         seed=self._seed)
#
#   def canonical_to_params(self, weights, biases):
#     """Converts params from the canonical format to a specific format of cuDNN.
#
#     Args:
#       weights: a Tensor for weight parameters.
#       biases: a Tensor for bias parameters.
#
#     Returns:
#       A function for the canonical-to-params-to-specific conversion..
#     """
#
#     return gen_cudnn_rnn_ops.cudnn_rnn_canonical_to_params(
#       num_layers=self._num_layers,
#       num_units=self._num_units,
#       input_size=self._input_size,
#       weights=weights,
#       biases=biases,
#       rnn_mode=self._rnn_mode,
#       input_mode=self._input_mode,
#       direction=self._direction)
#
# def _check_rnn_mode(rnn_mode):
#   if rnn_mode not in (CUDNN_LSTM, CUDNN_GRU, CUDNN_RNN_TANH, CUDNN_RNN_RELU):
#     raise ValueError("Invalid rnn_mode: %s, expect one of (%s, %s, %s, %s)" %
#                      (rnn_mode, CUDNN_LSTM, CUDNN_GRU, CUDNN_RNN_TANH,
#                       CUDNN_RNN_RELU))
#
#
# def check_direction(direction):
#   """Check validity of direction."""
#   if direction not in (CUDNN_RNN_UNIDIRECTION, CUDNN_RNN_BIDIRECTION):
#     raise ValueError("Invalid direction: %s, expecting %s or %s" %
#                      (direction, CUDNN_RNN_UNIDIRECTION, CUDNN_RNN_BIDIRECTION))
#
#
# def check_input_mode(input_mode):
#   if input_mode not in (CUDNN_INPUT_LINEAR_MODE, CUDNN_INPUT_SKIP_MODE,
#                         CUDNN_INPUT_AUTO_MODE):
#     raise ValueError("Invalid input_mode: %s, expect one of (%s, %s, %s)" %
#                      (input_mode, CUDNN_INPUT_LINEAR_MODE,
#                       CUDNN_INPUT_SKIP_MODE, CUDNN_INPUT_AUTO_MODE))
#
# def _cudnn_rnn(inputs,
#                input_h,
#                input_c,
#                params,
#                is_training,
#                rnn_mode,
#                input_mode=CUDNN_INPUT_LINEAR_MODE,
#                direction=CUDNN_RNN_UNIDIRECTION,
#                dropout=0.,
#                seed=0,
#                name=None):
#   """Cudnn RNN.
#   Args:
#     inputs: the input sequence to the RNN model. A Tensor of shape [?,
#       batch_size, input_size].
#     input_h: the initial hidden state for h. A Tensor of shape [num_layers,
#       batch_size, num_units].
#     input_c: the initial hidden state for c. This is only relevant for LSTM.
#       A Tensor of the same shape as input_h.
#     params: the parameter buffer created for this model.
#     is_training: whether this operation will be used in training or inference
#     rnn_mode: one of ('lstm', 'gru', 'rnn_relu', 'rnn_tanh').
#     input_mode: indicate whether there is a linear projection between the
#       input and the actual computation before the first layer. It could be
#       'linear_input', 'skip_input' or 'auto_select'.
#       'linear_input' (default) always applies a linear projection of input
#       onto RNN hidden state. (standard RNN behavior).
#       'skip_input' is only allowed when input_size == num_units;
#       'auto_select' implies 'skip_input' when input_size == num_units;
#       otherwise, it implies 'linear_input'.
#     direction: the direction model that the model operates. Could be either
#         'unidirectional' or 'bidirectional'
#     dropout: whether to enable dropout. With it is 0, dropout is disabled.
#     seed: the op seed used for initializing dropout. See @{tf.set_random_seed}
#         for behavior.
#     name: name of the operation.
#   Returns:
#     outputs, output_h, output_c
#   """
#   _check_rnn_mode(rnn_mode)
#   check_direction(direction)
#   check_input_mode(input_mode)
#   seed, seed2 = random_seed.get_seed(seed)
#   outputs, output_h, output_c, _ = gen_cudnn_rnn_ops.cudnn_rnn(
#       input=inputs,
#       input_h=input_h,
#       input_c=input_c,
#       params=params,
#       is_training=is_training,
#       rnn_mode=rnn_mode,
#       input_mode=input_mode,
#       direction=direction,
#       dropout=dropout,
#       seed=seed,
#       seed2=seed2,
#       name=name)
#   return (outputs, output_h, output_c)
#
#
# class CudnnLSTM(_CudnnRNN):
#   """Cudnn implementation of the LSTM model."""
#   # __doc__ += _cudnn_rnn_common_doc_string
#   # 4 sets of weight and bias parameters for the recurrent input, and 4 for the
#   # previous layer input.
#   _NUM_PARAMS_PER_LAYER = CUDNN_LSTM_PARAMS_PER_LAYER
#
#   def __init__(self,
#                num_layers,
#                num_units,
#                input_size,
#                input_mode=CUDNN_INPUT_LINEAR_MODE,
#                direction=CUDNN_RNN_UNIDIRECTION,
#                dtype=dtypes.float32,
#                dropout=0.,
#                seed=0):
#     """Creates a Cudnn LSTM model from model spec.
#
#     Args:
#       num_layers: the number of layers for the RNN model.
#       num_units: the number of units within the RNN model.
#       input_size: the size of the input, it could be different from the
#           num_units.
#       input_mode: indicate whether there is a linear projection between the
#           input and The actual computation before the first layer. It could be
#           'skip_input', 'linear_input' or 'auto_select'.
#           'skip_input' is only allowed when input_size == num_units;
#           'auto_select' implies 'skip_input' when input_size == num_units;
#           otherwise, it implies 'linear_input'.
#       direction: the direction model that the model operates. Could be either
#           'unidirectional' or 'bidirectional'
#       dtype: dtype of params, tf.float32 or tf.float64.
#       dropout: whether to enable dropout. With it is 0, dropout is disabled.
#       seed: the seed used for initializing dropout.
#     """
#     super(CudnnLSTM, self).__init__(
#         CUDNN_LSTM,
#         num_layers,
#         num_units,
#         input_size,
#         input_mode=input_mode,
#         direction=direction,
#         dtype=dtype,
#         dropout=dropout,
#         seed=seed)
#
#   def __call__(self, input_data, input_h, input_c, params, is_training=True):
#     """Runs the forward step for the Cudnn LSTM model.
#
#     Args:
#       input_data: the input sequence to the LSTM model. A Tensor of shape [?,
#         batch_size, input_size].
#       input_h: the initial hidden state for h. A Tensor of shape [num_layers,
#         batch_size, num_units].
#       input_c: the initial hidden state for c. A Tensor of the same shape as
#         input_h.
#       params: the parameter buffer created for this model.
#       is_training: whether this operation will be used in training or inference.
#     Returns:
#       output: the output sequuence.
#       output_h: the final state for h.
#       output_c: the final state for c.
#     """
#     output, output_h, output_c = super(CudnnLSTM, self).__call__(
#         input_data, input_h, input_c, params, is_training=is_training)
#     return (output, output_h, output_c)
