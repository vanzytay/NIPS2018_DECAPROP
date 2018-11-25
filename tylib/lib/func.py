import tensorflow as tf

from .seq_op import *
from .sim_op import *

INF = 1E30

class cudnn_rnn:
    """ Universal cudnn_rnn class
    Supports both LSTM and GRU

    Variational dropout is optional
    """

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0,
                    is_train=None, scope=None, init=None, rnn_type='',
                    direction='bidirectional'):
        if(init is None):
            rnn_init = tf.random_normal_initializer(stddev=0.1)
        else:
            rnn_init = init
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        self.num_units = num_units
        self.is_train = is_train
        self.keep_prob = keep_prob
        self.input_size = input_size
        self.rnn_type = rnn_type
        self.direction=direction
        self.num_params = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            if('LSTM' in rnn_type):
                gru_fw = tf.contrib.cudnn_rnn.CudnnLSTM(
                    1, num_units, kernel_initializer=rnn_init)
                if(self.direction=='bidirectional'):
                    gru_bw = tf.contrib.cudnn_rnn.CudnnLSTM(
                        1, num_units, kernel_initializer=rnn_init)
                else:
                    gru_bw = None
            else:
                gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(
                    1, num_units, kernel_initializer=rnn_init)
                if(self.direction=='bidirectional'):
                    gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(
                        1, num_units, kernel_initializer=rnn_init)
                else:
                    gru_bw = None

            self.grus.append((gru_fw, gru_bw, ))

    def __call__(self, inputs, seq_len, batch_size=None,
                is_train=None, concat_layers=True,
                var_drop=1, train_init=0):
        # batch_size = inputs.get_shape().as_list()[0]
        batch_size = tf.shape(inputs)[0]
        outputs = [tf.transpose(inputs, [1, 0, 2])]

        for layer in range(self.num_layers):
            if(train_init):
                init_fw = tf.tile(tf.Variable(
                    tf.zeros([1, 1, self.num_units])), [1, batch_size, 1])
                if(self.direction=='bidirectional'):
                    init_bw = tf.tile(tf.Variable(
                        tf.zeros([1, 1, self.num_units])), [1, batch_size, 1])
                else:
                    init_bw = None
            else:
                init_fw = tf.tile(tf.zeros([1, 1, self.num_units]),
                                [1, batch_size, 1])
                if(self.direction=='bidirectional'):
                    init_bw = tf.tile(tf.zeros([1, 1, self.num_units]),
                                    [1, batch_size, 1])
                else:
                    init_bw = None
            if(var_drop==1):
                mask_fw = dropout(tf.ones([1, batch_size, self.input_size],
                                    dtype=tf.float32),
                                  keep_prob=self.keep_prob, is_train=self.is_train)
                output_fw = outputs[-1] * mask_fw
                if(self.direction=='bidirectional'):
                    mask_bw = dropout(tf.ones([1, batch_size, self.input_size],
                                        dtype=tf.float32),
                                      keep_prob=self.keep_prob, is_train=self.is_train)
                    output_bw = outputs[-1] * mask_bw
            else:
                output_fw = outputs[-1]
                output_fw = dropout(output_fw,
                                keep_prob=self.keep_prob,
                                is_train=self.is_train)
                if(self.direction=='bidirectional'):
                    output_bw = outputs[-1]
                    output_bw = dropout(output_bw,
                                    keep_prob=self.keep_prob,
                                    is_train=self.is_train)
            gru_fw, gru_bw = self.grus[layer]
            if('LSTM' in self.rnn_type):
                init_state1 = (init_fw, init_fw)
                init_state2 = (init_bw, init_bw)
            else:
                init_state1 = (init_fw,)
                init_state2 = (init_bw,)

            with tf.variable_scope("fw_{}".format(layer)):
                out_fw, _ = gru_fw(
                    output_fw, initial_state=init_state1)
                self.num_params += gru_fw.canonical_weight_shapes

            out = out_fw

            if(self.direction=='bidirectional'):
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(
                        output_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                    out_bw, _ = gru_bw(inputs_bw, initial_state=init_state2)
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                out = tf.concat([out, out_bw], 2)
                self.num_params += gru_bw.canonical_weight_shapes
            outputs.append(out)
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])

        counter = 0
        for t in self.num_params:
            counter += t[0] * t[1]
        print('Cudnn Parameters={}'.format(counter))

        return res

class native_gru:
    """ Native GRU cell.
    This is functionallly similar to Cudnn version.
    However, this is less updated and short of some features.
    But I suggest to never use native GRU over the Cudnn version.
    """

    def __init__(self, num_layers, num_units, batch_size, input_size,
                        keep_prob=1.0, is_train=None, scope="native_gru"):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        self.scope = scope
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.rnn.GRUCell(
                num_units, kernel_initializer=tf.random_normal_initializer(stddev=0.1))
            gru_bw = tf.contrib.rnn.GRUCell(
                num_units, kernel_initializer=tf.random_normal_initializer(stddev=0.1))
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            mask_fw = dropout(tf.ones([batch_size, 1, input_size_],
                                dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([batch_size, 1, input_size_],
                                dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [inputs]
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                gru_fw, gru_bw = self.grus[layer]
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.dropout_mask[layer]
                with tf.variable_scope("fw_{}".format(layer)):
                    out_fw, _ = tf.nn.dynamic_rnn(
                        gru_fw, outputs[-1] * mask_fw, seq_len,
                                            initial_state=init_fw,
                                            dtype=tf.float32)
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(
                        outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    out_bw, _ = tf.nn.dynamic_rnn(
                        gru_bw, inputs_bw, seq_len, initial_state=init_bw, dtype=tf.float32)
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        return res

class ptr_net:
    def __init__(self, batch, hidden, keep_prob=1.0, is_train=None, scope="ptr_net"):
        self.gru = tf.contrib.rnn.GRUCell(hidden)
        self.batch = batch
        self.scope = scope
        self.keep_prob = keep_prob
        self.is_train = is_train
        self.dropout_mask = dropout(tf.ones(
            [batch, hidden], dtype=tf.float32), keep_prob=keep_prob, is_train=is_train)

    def __call__(self, init, match, d, mask):
        with tf.variable_scope(self.scope):
            d_match = dropout(match, keep_prob=self.keep_prob,
                              is_train=self.is_train)
            inp, logits1 = pointer(d_match, init * self.dropout_mask, d, mask)
            d_inp = dropout(inp, keep_prob=self.keep_prob,
                            is_train=self.is_train)
            _, state = self.gru(d_inp, init)
            tf.get_variable_scope().reuse_variables()
            _, logits2 = pointer(d_match, state * self.dropout_mask, d, mask)
            return logits1, logits2

class ptr_net_v2:
    def __init__(self, batch, hidden, keep_prob=1.0, is_train=None, scope="ptr_net"):
        # self.gru = tf.contrib.rnn.GRUCell(hidden)
        self.gru =  tf.contrib.cudnn_rnn.CudnnGRU(1, hidden)
        self.batch = batch
        self.scope = scope
        self.keep_prob = keep_prob
        self.is_train = is_train
        self.hidden = hidden

    def __call__(self, init, match, d, mask, lengths=None, get_repr=False):
        with tf.variable_scope(self.scope):
            batch = tf.shape(match)[0]
            self.dropout_mask = dropout(tf.ones(
                [batch, self.hidden], dtype=tf.float32),
                    keep_prob=self.keep_prob, is_train=self.is_train)

            d_match = dropout(match, keep_prob=self.keep_prob,
                              is_train=self.is_train)
            inp, logits1 = pointer_v2(d_match, init * self.dropout_mask,
                                d, mask)
            d_inp = dropout(inp, keep_prob=self.keep_prob,
                            is_train=self.is_train)
            d_inp = tf.transpose(d_inp, [1,0,2])
            init = tf.expand_dims(init,1)
            init = tf.transpose(init, [1,0,2])
            state, _ = self.gru(d_inp, initial_state=(init,))
            state = tf.transpose(state, [1,0,2])
            # state = tf.gather_nd(state, lengths-1)
            state = last_relevant(state, lengths)
            tf.get_variable_scope().reuse_variables()
            repr2, logits2 = pointer_v2(d_match,
                            state * self.dropout_mask,
                            d, mask)
            if(get_repr):
                return logits1, logits2, inp, repr2
            else:
                return logits1, logits2

def dropout(args, keep_prob, is_train, mode="recurrent"):
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

def softmax_mask(val, mask, inf=1E30):
    return -inf * (1 - tf.cast(mask, tf.float32)) + val

def pointer_v2(inputs, state, hidden, mask, scope="pointer", inf=1E30):
    with tf.variable_scope(scope):
        u = tf.concat([tf.tile(tf.expand_dims(state, axis=1), [
            1, tf.shape(inputs)[1], 1]), inputs], axis=2)
        s0 = tf.nn.tanh(dense(u, hidden, use_bias=False, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask, inf=INF)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = a * inputs
        # res = tf.reduce_sum(a * inputs, axis=1)
        return res, s1

def pointer(inputs, state, hidden, mask, scope="pointer", inf=1E30):
    with tf.variable_scope(scope):
        u = tf.concat([tf.tile(tf.expand_dims(state, axis=1), [
            1, tf.shape(inputs)[1], 1]), inputs], axis=2)
        s0 = tf.nn.tanh(dense(u, hidden, use_bias=False, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask, inf=INF)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * inputs, axis=1)
        return res, s1

def summ(memory, hidden, mask, keep_prob=1.0, is_train=None, scope="summ",
            inf=1E30):
    with tf.variable_scope(scope):
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        s0 = tf.nn.tanh(dense(d_memory, hidden, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask, inf=INF)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * memory, axis=1)
        return res

def dot_attention(inputs, memory, mask, hidden,
                keep_prob=1.0, is_train=None,
                scope="dot_attention", inf=1E30):
    """ This is the dot attention in Transformer.
    """
    with tf.variable_scope(scope):

        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        JX = tf.shape(inputs)[1]

        with tf.variable_scope("attention"):
            inputs_ = tf.nn.relu(
                dense(d_inputs, hidden, use_bias=False, scope="inputs"))
            memory_ = tf.nn.relu(
                dense(d_memory, hidden, use_bias=False, scope="memory"))
            outputs = tf.matmul(inputs_, tf.transpose(
                memory_, [0, 2, 1])) / (hidden ** 0.5)
            mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
            logits = tf.nn.softmax(softmax_mask(outputs, mask,
                                                inf=INF))
            outputs = tf.matmul(logits, memory)
            res = tf.concat([inputs, outputs], axis=2)

        with tf.variable_scope("gate"):
            dim = res.get_shape().as_list()[-1]
            d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
            gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False))
            return res * gate

def symmetric_dot_attention(inputs, memory, mask, hidden,
                keep_prob=1.0, is_train=None,
                scope="dot_attention", inf=1E30, init=None):
    """ Symmetric Dot Attention from FusionNet
    """
    with tf.variable_scope(scope):

        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        JX = tf.shape(inputs)[1]

        def bilinear(a, U, b):
            dim = a.get_shape().as_list()[2]
            a_len = tf.shape(a)[1]
            _a = tf.reshape(a, [-1, dim])
            z = tf.matmul(_a, U)
            z = tf.reshape(z, [-1, a_len, dim])
            y = tf.matmul(z, tf.transpose(b, [0, 2, 1]))
            return y

        with tf.variable_scope("attention"):
            dim = inputs.get_shape().as_list()[2]
            inputs_ = tf.nn.relu(
                dense(d_inputs, hidden, use_bias=False, scope="proj"))
            memory_ = tf.nn.relu(
                dense(d_memory, hidden, use_bias=False, scope="proj",
                                                reuse=True))
            weights_U = tf.get_variable("U", [dim, dim],
                                        initializer=init)
            outputs = bilinear(inputs, weights_U, memory)
            # outputs = tf.matmul(inputs_, tf.transpose(
            #     memory_, [0, 2, 1])) / (hidden ** 0.5)
            mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
            logits = tf.nn.softmax(softmax_mask(outputs, mask,
                                                inf=INF))
            outputs = tf.matmul(logits, memory)
            res = tf.concat([inputs, outputs], axis=2)

        with tf.variable_scope("gate"):
            dim = res.get_shape().as_list()[-1]
            d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
            gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False))
            return res * gate


def dot_attention_submul(inputs, memory, mask, hidden,
                keep_prob=1.0, is_train=None,
                scope="dot_attention", inf=1E30):
    """ This is the dot attention in Transformer.
    """
    with tf.variable_scope(scope):

        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        JX = tf.shape(inputs)[1]

        with tf.variable_scope("attention"):
            inputs_ = tf.nn.relu(
                dense(d_inputs, hidden, use_bias=False, scope="inputs"))
            memory_ = tf.nn.relu(
                dense(d_memory, hidden, use_bias=False, scope="memory"))
            outputs = tf.matmul(inputs_, tf.transpose(
                memory_, [0, 2, 1])) / (hidden ** 0.5)
            mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
            logits = tf.nn.softmax(softmax_mask(outputs, mask,
                                            inf=INF))
            outputs = tf.matmul(logits, memory)
        res = sub_mult_nn(outputs, inputs)
        return res

def dense(inputs, hidden, use_bias=True, scope="dense", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(
            len(inputs.get_shape().as_list()) - 1)] + [hidden]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden])
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res
