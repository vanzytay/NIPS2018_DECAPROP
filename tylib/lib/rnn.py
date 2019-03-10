from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .att_op import *
from .seq_op import *


''' Implements RNN layers that are experimentally convenient.
'''

def build_rnn(embed, lengths, rnn_type="LSTM", reuse=False,
                rnn_dim=None, name="", num_layers=1, dropout=None,
                initializer=None, use_cudnn=0, is_train=False,
                train_init=1, var_drop=1, fuse_kernel=1):
    ''' Builds RNN model with support for experiments

    Update:Removed Support for Attention
    Attention layer should be added AFTER

    Base recurrent units supported - `LSTM`, `GRU`, `RNN`.
    Extra combinations:
        'BI': Adds Bidirectionality
        `LAST`: Outputs LAST pooling vector in 2nd arg
        'MEAN': Outputs MEAN pooling vector in 2nd arg

    Args:
        embed: `tensor`. bsz x time_steps x dim
        lengths: `tensor`. bsz x 1 (actual length of each sequence)
        rnn_type: `str`. Controls recurrent type (LSTM|GRU|RNN)
            Support for conjunction. (See conjunction notes)
        reuse: 'bool`. To reuse RNN parameters or not.
        rnn_dim: `int`. The dimension size of RNN params.
        name: `str`. Variable name.
        num_layers: `int`. Number of layers of RNN.
        dropout: `tensor`. placeholder
        initializer: initializer function

    Returns:
        hidden_states: `tensor`. bsz x time_steps x rnn_dim (hidden rep)
        vec: `tensor`. bsz x dim. hidden rep with pooling applied
            LAST or MEA

    '''
    vec = None

    if(initializer is None):
        rnn_initializer = tf.orthogonal_initializer()
    else:
        rnn_initializer = initializer

    if('RMC' in rnn_type or 'FMC' in rnn_type):
        use_cudnn = 0

    with tf.variable_scope('RNN_{}'.format(name),
                initializer=rnn_initializer, reuse=reuse) as scope:
        if(use_cudnn==1):
            # Use CUDNN model
            if('BI_' in rnn_type):
                direction = 'bidirectional'
            else:
                direction = 'unidirectional'
            print("Direction={}".format(direction))
            # embed = tf.transpose(embed, [1,0,2])
            bsz = tf.shape(embed)[0]
            outputs = embed
            for i in range(num_layers):
                with tf.variable_scope('cudnn{}'.format(i)) as scope:
                    rnn = cudnn_rnn(num_layers=1,
                                num_units=rnn_dim,
                                batch_size=bsz,
                                input_size=outputs.get_shape().as_list()[-1],
                                keep_prob=dropout,
                                is_train=is_train,
                                rnn_type=rnn_type,
                                init=rnn_initializer,
                                direction=direction
                                )
                    outputs = rnn(outputs, seq_len=lengths,
                                var_drop=var_drop, concat_layers=False,
                                train_init=train_init,
                                )

            print("RNN outputs={}".format(outputs))
            return outputs, None


        if('LSTM' in rnn_type):
            scell = tf.contrib.rnn.BasicLSTMCell(
                                    rnn_dim, forget_bias=1.0,
                                    state_is_tuple=True,
                                    reuse=reuse)

        elif('GRU' in rnn_type):
            scell = tf.contrib.rnn.GRUCell(rnn_dim)
        # elif('RMC' in rnn_type):
        #     num_heads = 1
        #     scell = snt.RelationalMemory(
        #                 mem_slots=1,
        #                 head_size=rnn_dim // num_heads,
        #                 num_heads=num_heads,
        #                 num_blocks=1,
        #                 gate_style='unit')
        # Dropout
        # if(dropout is not None):
        #     scell = tf.contrib.rnn.DropoutWrapper(
        #         scell, output_keep_prob=dropout)

        # Support Multi-layered RNN models
        if(num_layers > 1):
            stack_rnn = [scell]
            scell2 = tf.contrib.rnn.BasicLSTMCell(
                rnn_dim, forget_bias=1.0, state_is_tuple=True)
            for i in range(1, num_layers):
                stack_rnn.append(scell2)
            scell = tf.contrib.rnn.MultiRNNCell(
                                        stack_rnn,
                                        state_is_tuple=True)
        init_state = scell.zero_state(tf.shape(embed)[0], tf.float32)

        if('BI_' in rnn_type):
            # Support bidirectional RNN
            # rnn_dim *= 2
            if('LSTM' in rnn_type):
                scell2 = tf.contrib.rnn.BasicLSTMCell(
                                        rnn_dim, forget_bias=1.0,
                                        state_is_tuple=True,
                                        reuse=reuse)
            elif('GRU' in rnn_type):
                scell2 = tf.contrib.rnn.GRUCell(rnn_dim)

            init_state2 = scell2.zero_state(tf.shape(embed)[0], tf.float32)

            sentence_outputs, s_last_state = tf.nn.bidirectional_dynamic_rnn(
                                    cell_fw=scell,
                                    cell_bw=scell2,
                                    inputs=embed,
                                    dtype=tf.float32,
                                    initial_state_fw=init_state,
                                    initial_state_bw=init_state2,
                                    sequence_length=tf.cast(lengths, tf.int32)
                                )
            #print(s_last_state)
            sentence_outputs = tf.concat(sentence_outputs, 2)
        else:
            sentence_outputs, s_last_state = tf.nn.dynamic_rnn(
                    scell, embed, sequence_length=tf.cast(lengths, tf.int32),
                    initial_state=init_state, dtype=tf.float32)

        hidden_states = sentence_outputs

        if('MEAN' in rnn_type):
            vec = mean_over_time(sentence_outputs,
                                tf.expand_dims(lengths, 1))
        elif('LAST' in rnn_type):
            vec = last_relevant(sentence_outputs, lengths)
        elif('MAX' in rnn_type):
            vec = tf.reduce_max(sentence_outputs, 1)
        elif('ATT' in rnn_type):
            vec, att = attention(sentence_outputs,
                            context=None, reuse=reuse, name='',
                            kernel_initializer=initializer,
                            dropout=None)
            hidden_states = hidden_states * att
        else:
            vec = None

        if('FMC' in rnn_type):
            hidden_states = tf.split(hidden_states, chunk_size, 2)
            hidden_states = tf.concat(hidden_states, 1)

        return hidden_states, vec
