# coding=utf-8

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib import rnn


# CNN layer
# ===============================================
def cnn_layer(inputs, embedding_size, filter_sizes, num_filters):
    pooled_outputs = []
    sequence_length = inputs.shape[1]
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="conv_b")
            conv = tf.nn.conv2d(
                inputs,
                W,
                strides=[1,1,1,1],
                padding="VALID",
                name="conv"
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  #  tanh ---- Duyu Tang 2015
            pooled = tf.nn.max_pool(  # avg_pool  --- Duyu Tang 2015
                h,
                ksize = [1, sequence_length - filter_size + 1, 1, 1],
                strides = [1,1,1,1],
                padding = "VALID",
                name= "pool"
            )
            pooled_outputs.append(pooled)

    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    return h_pool_flat

# RNN layer
# ==================================================
# 返回一个序列中每个元素的长度
def get_length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    # abs 求绝对值,
    # reduce_max 求最大值, reduction_indices 在哪一个维度上求解
    # sign 返回符号-1 if x < 0 ; 0 if x == 0 ; 1 if x > 0
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    # 计算输入tensor元素的和,或者按照reduction_indices指定的轴进行
    return tf.cast(seq_len, tf.int32)
    # 将x的数据格式转化为int32

# bi-LSTM layer network
def biRNNLayer(inputs, hidden_size):

    # fw_cell = rnn_cell.LSTMCell(hidden_size)
    # bw_cell = rnn_cell.LSTMCell(hidden_size)
    fw_cell = rnn.GRUCell(hidden_size)  # 前向GRU, 输入的参数为隐藏层的个数
    bw_cell = rnn.GRUCell(hidden_size)  # 后向GRU, 输入的参数为隐藏层的个数
    ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=fw_cell,
        cell_bw=bw_cell,
        inputs=inputs,
        sequence_length=get_length(inputs),
        dtype=tf.float32
    )
    # outputs的size是[batch_size, max_time, hidden_size *2 ]
    outputs = tf.concat((fw_outputs, bw_outputs), 2)  # 按行拼接
    return outputs

# max pooling of rnn out layer
def max_pooling(lstm_out):

    # shape of lstm_out: [batch, sequence_length, rnn_size * 2 ]
    # do max-pooling to change the (sequence_length) tensor to 1-length tensor
    sequence_length, hidden_cell_size = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])
    lstm_out = tf.expand_dims(lstm_out, -1)
    output = tf.nn.max_pool(   # change to : tf.nn.average_pooling
        lstm_out,
        ksize=[1, sequence_length, 1, 1],
        strides=[1,1,1,1],
        padding='VALID'
    )
    output = tf.reshape(output, [-1, hidden_cell_size])
    return output

# attention mechanism layer
# ============================================================
def attention_layer(inputs, attention_size):

    # shape of inputs: [batch, sequence_length, hidden_size]
    sequence_length = inputs.get_shape()[1].value
    hidden_size = inputs.get_shape()[2].value

    # Attention mechanism
    W_omega = tf.get_variable("W_omega", initializer= tf.truncated_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.get_variable("b_omega", initializer= tf.truncated_normal([attention_size], stddev=0.1))
    u_omega = tf.get_variable("u_omega", initializer= tf.truncated_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
    return output

# auxiliary attention layer network
def auxAttention(input, aux, attention_size):

    # shape of input: [batch, sequence_length, embedding_size]
    # shape of aux: [batch, vector_length]

    len_aux = int(aux.get_shape()[1])
    seq_input = int(input.get_shape()[1])
    emb_len_input = int(input.get_shape()[2])

    Wm_input = tf.Variable(tf.truncated_normal([emb_len_input, attention_size], stddev=0.1), name="Wm_input")
    Wm_aux = tf.Variable(tf.truncated_normal([len_aux, attention_size], stddev=0.1), name="Wm_aux")
    W_u = tf.Variable(tf.truncated_normal([attention_size], stddev=0.1), name="W_u")
    W_b = tf.Variable(tf.truncated_normal([attention_size], stddev=0.1), name="W_b")

    # extend auxiliary vector to matrix
    extend_aux = tf.expand_dims(aux, 1)
    matrix_aux = tf.tile(extend_aux, [1, seq_input, 1])
    reshape_aux = tf.reshape(matrix_aux, [-1, len_aux])

    # attention
    v = tf.matmul(tf.reshape(input, [-1, emb_len_input]), Wm_input) + tf.matmul(reshape_aux, Wm_aux) + tf.reshape(W_b, [1, -1])
    v = tf.tanh(v)
    vu = tf.matmul(v, tf.reshape(W_u, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, seq_input])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
    output = input * tf.reshape(alphas, [-1, seq_input, 1])
    output = tf.reduce_sum(output, 1)
    return output





















































































































































































































































































































































































































































































































































































































































































































































































































































































