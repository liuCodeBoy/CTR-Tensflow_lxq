from collections import Iterable
import itertools

import tensorflow as tf
from tensorflow.python.keras import layers


class DNN(tf.keras.Model):
    """
        Deep Neural Network
    """

    def __init__(self,
                 units,
                 use_bias=True,
                 use_bn=False,
                 dropout=0,
                 activations=None,
                 kernel_initializers='glorot_uniform',
                 bias_initializers='zeros',
                 kernel_regularizers=tf.keras.regularizers.l2(1e-5),
                 bias_regularizers=None,
                 **kwargs):
        """
        :param units:
            An iterable of hidden layers' neural units' number, its length is the depth of the DNN.
        :param use_bias:
            Iterable/Boolean.
            If this is not iterable, every layer of the DNN will have the same param, the same below.
        :param activations:
            Iterable/String/TF activation class
        :param kernel_initializers:
            Iterable/String/TF initializer class
        :param bias_initializers:
            Iterable/String/TF initializer class
        :param kernel_regularizers:
            Iterable/String/TF regularizer class
        :param bias_regularizers:
            Iterable/String/TF regularizer class
        """

        super(DNN, self).__init__(**kwargs)

        self.units = units
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.dropout = dropout
        self.activations = activations
        self.kernel_initializers = kernel_initializers
        self.bias_initializers = bias_initializers
        self.kernel_regularizers = kernel_regularizers
        self.bias_regularizers = bias_regularizers

        if not isinstance(self.use_bias, Iterable):
            self.use_bias = [self.use_bias] * len(self.units)

        if not isinstance(self.use_bn, Iterable):
            self.use_bn = [self.use_bn] * len(self.units)

        if not isinstance(self.dropout, Iterable):
            self.dropout = [self.dropout] * len(self.units)

        if not isinstance(self.activations, Iterable):
            self.activations = [self.activations] * len(self.units)

        if isinstance(self.kernel_initializers, str) or not isinstance(self.kernel_initializers, Iterable):
            self.kernel_initializers = [self.kernel_initializers] * len(self.units)

        if isinstance(self.bias_initializers, str) or not isinstance(self.bias_initializers, Iterable):
            self.bias_initializers = [self.bias_initializers] * len(self.units)

        if isinstance(self.kernel_regularizers, str) or not isinstance(self.kernel_regularizers, Iterable):
            self.kernel_regularizers = [self.kernel_regularizers] * len(self.units)

        if isinstance(self.bias_regularizers, str) or not isinstance(self.bias_regularizers, Iterable):
            self.bias_regularizers = [self.bias_regularizers] * len(self.units)

        self.mlp = tf.keras.Sequential()
        for i in range(len(self.units)):
            self.mlp.add(layers.Dense(
                units=self.units[i],
                activation=self.activations[i],
                use_bias=self.use_bias[i],
                kernel_initializer=self.kernel_initializers[i],
                bias_initializer=self.bias_initializers[i],
                kernel_regularizer=self.kernel_regularizers[i],
                bias_regularizer=self.bias_regularizers[i]
            ))
            if self.dropout[i] > 0:
                self.mlp.add(layers.Dropout(self.dropout[i]))
            if self.use_bn[i]:
                self.mlp.add(layers.BatchNormalization())

    def call(self, inputs, **kwargs):

        output = self.mlp(inputs)

        return output


class FM(tf.keras.Model):
    """
        Factorization Machine Block
        compute cross features (order-2) and return their sum (without linear term)
    """
    def __init__(self, **kwargs):

        super(FM, self).__init__(**kwargs)

    def call(self, inputs, require_logit=True, **kwargs):
        """
        :param inputs:
            list of 2D tensor with shape [batch_size, embedding_size]
            all the features should be embedded and have the same embedding size
        :return:
            2D tensor with shape [batch_size, 1]
            sum of all cross features
        """

        # [b, n, m]
        inputs_3d = tf.stack(inputs, axis=1)

        # [b, m]
        # (a + b) ^ 2 - (a ^ 2 + b ^ 2) = 2 * ab, we need the cross feature "ab"
        square_of_sum = tf.square(tf.reduce_sum(inputs_3d, axis=1, keepdims=False))
        sum_of_square = tf.reduce_sum(tf.square(inputs_3d), axis=1, keepdims=False)
        if require_logit:
            outputs = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=1, keepdims=True)
        else:
            outputs = 0.5 * (square_of_sum - sum_of_square)

        return outputs


class InnerProduct(tf.keras.Model):

    def __init__(self, require_logit=True, **kwargs):

        super(InnerProduct, self).__init__(**kwargs)

        self.require_logit = require_logit

    def call(self, inputs, **kwargs):

        rows = list()
        cols = list()
        for i in range(len(inputs) - 1):
            for j in range(i, len(inputs)):
                rows.append(i)
                cols.append(j)

        # [batch_size, pairs_num, embedding_size]
        p = tf.stack([inputs[i] for i in rows], axis=1)
        q = tf.stack([inputs[j] for j in cols], axis=1)

        if self.require_logit:
            inner_product = tf.reduce_sum(p * q, axis=-1, keepdims=False)
        else:
            inner_product = tf.keras.layers.Flatten()(p * q)

        return inner_product


class OuterProduct(tf.keras.Model):

    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                 **kwargs):

        super(OuterProduct, self).__init__(**kwargs)

        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

    def call(self, inputs, **kwargs):

        outer_products_list = list()

        for i in range(len(inputs) - 1):
            for j in range(i + 1, len(inputs)):
                inp_i = tf.expand_dims(inputs[i], axis=1)
                inp_j = tf.expand_dims(inputs[j], axis=-1)
                kernel = self.add_weight(shape=(inp_i.shape[2], inp_j.shape[1]),
                                         initializer=self.kernel_initializer,
                                         regularizer=self.kernel_regularizer,
                                         trainable=True)
                product = tf.reduce_sum(tf.matmul(tf.matmul(inp_i, kernel), inp_j), axis=-1, keepdims=False)
                outer_products_list.append(product)

        outer_product_layer = tf.concat(outer_products_list, axis=1)

        return outer_product_layer


class CrossNetwork(tf.keras.Model):

    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 **kwargs):

        super(CrossNetwork, self).__init__(**kwargs)

        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer

    def call(self, inputs, layers_num=3, require_logit=True, **kwargs):

        x0 = tf.expand_dims(tf.concat(inputs, axis=1), axis=-1)
        x = tf.transpose(x0, [0, 2, 1])

        for i in range(layers_num):
            kernel = self.add_weight(shape=(x0.shape[1], 1),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     trainable=True)
            bias = self.add_weight(shape=(x0.shape[1], 1),
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   trainable=True)
            x = tf.matmul(tf.matmul(x0, x), kernel) + bias + tf.transpose(x, [0, 2, 1])
            x = tf.transpose(x, [0, 2, 1])

        x = tf.squeeze(x, axis=1)
        if require_logit:
            kernel = self.add_weight(shape=(x0.shape[1], 1),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     trainable=True)
            x = tf.matmul(x, kernel)

        return x


class CIN(tf.keras.Model):

    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                 **kwargs):

        super(CIN, self).__init__(**kwargs)

        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

    def call(self, inputs, hidden_width=(128, 64), require_logit=True, **kwargs):

        # [b, n, m]
        x0 = tf.stack(inputs, axis=1)
        x = tf.identity(x0)

        hidden_width = [x0.shape[1]] + list(hidden_width)

        finals = list()
        for h in hidden_width:
            rows = list()
            cols = list()
            for i in range(x0.shape[1]):
                for j in range(x.shape[1]):
                    rows.append(i)
                    cols.append(j)
            # [b, pair, m]
            x0_ = tf.gather(x0, rows, axis=1)
            x_ = tf.gather(x, cols, axis=1)
            # [b, m, pair]
            p = tf.transpose(tf.multiply(x0_, x_), [0, 2, 1])

            kernel = self.add_weight(shape=(p.shape[-1], h),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     trainable=True)
            # [b, h, m]
            x = tf.transpose(tf.matmul(p, kernel), [0, 2, 1])
            finals.append(tf.reduce_sum(x, axis=-1, keepdims=False))

        finals = tf.concat(finals, axis=-1)
        kernel = self.add_weight(shape=(finals.shape[-1], 1),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 trainable=True)
        logit = tf.matmul(finals, kernel)

        return logit


class AttentionBasedPoolingLayer(tf.keras.Model):
    def __init__(self,
                 attention_factor=4,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 **kwargs):

        super(AttentionBasedPoolingLayer, self).__init__(**kwargs)

        self.attention_factor = attention_factor
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer

        # units：输出的维度大小，改变inputs的最后一维
        # kernel_initializer = None,  ##卷积核的初始化器
        # kernel_regularizer = None,  ##卷积核的正则化，可选
        # bias_regularizer = None,  ##偏置项的正则化，可选

        self.att_layer = layers.Dense(
            units=self.attention_factor,
            activation='relu',
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer
        )


        self.att_proj_layer = layers.Dense(
            units=1,
            activation=None,
            use_bias=False,
            kernel_initializer=self.kernel_initializer
        )

    def call(self, inputs, **kwargs):

        interactions = list()

        for i in range(len(inputs) - 1):
            for j in range(i + 1, len(inputs)):
                interactions.append(tf.multiply(inputs[i], inputs[j]))
        # print(interactions)
        interactions = tf.stack(interactions, axis=1)
        print("interactions:",interactions)
        att_weight = self.att_layer(interactions)
        print("att_weight:",att_weight)
        # att_weight: Tensor("attention_based_pooling_layer/dense/Identity:0", shape=(None, 276, 4), dtype=float32)
        att_weight = self.att_proj_layer(att_weight)
        print("att_weight:",att_weight)

        att_weight = layers.Softmax(axis=1)(att_weight)
        print("att_weight:",att_weight)

        output = tf.reduce_sum(interactions * att_weight, axis=1)
        print("output:",output)

        return output


class AutoIntInteraction(tf.keras.Model):

    def __init__(self, att_embedding_size=8, heads=2, use_res=True, seed=2333, **kwargs):

        super(AutoIntInteraction, self).__init__(**kwargs)

        self.att_embedding_size = att_embedding_size
        self.heads = heads
        self.use_res = use_res
        self.seed = seed

    def call(self, inputs, **kwargs):

        m = inputs.shape[-1]

        W_Query = self.add_weight(shape=[m, self.att_embedding_size * self.heads],
                                  initializer=tf.keras.initializers.RandomNormal(seed=self.seed))
        W_key = self.add_weight(shape=[m, self.att_embedding_size * self.heads],
                                initializer=tf.keras.initializers.RandomNormal(seed=self.seed))
        W_Value = self.add_weight(shape=[m, self.att_embedding_size * self.heads],
                                  initializer=tf.keras.initializers.RandomNormal(seed=self.seed))

        queries = tf.matmul(inputs, W_Query)
        keys = tf.matmul(inputs, W_key)
        values = tf.matmul(inputs, W_Value)

        queries = tf.stack(tf.split(queries, self.heads, axis=2))
        keys = tf.stack(tf.split(keys, self.heads, axis=2))
        values = tf.stack(tf.split(values, self.heads, axis=2))

        att_score = tf.matmul(queries, keys, transpose_b=True)
        att_score = layers.Softmax(axis=-1)(att_score)

        result = tf.matmul(att_score, values)
        result = tf.concat(tf.split(result, self.heads), axis=-1)
        result = tf.squeeze(result, axis=0)

        if self.use_res:
            W_Res = self.add_weight(shape=[m, self.att_embedding_size * self.heads],
                                    initializer=tf.keras.initializers.RandomNormal(seed=self.seed))
            result = result + tf.matmul(inputs, W_Res)

        result = tf.keras.activations.relu(result)

        return result


class FGCNNlayer(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_width, new_feat_filters, pool_width, **kwargs):

        super(FGCNNlayer, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_width = kernel_width
        self.new_feat_filters = new_feat_filters
        self.pool_width = pool_width

    def call(self, inputs, **kwargs):

        output = inputs
        output = tf.keras.layers.Conv2D(
            filters=self.filters,
            strides=(1, 1),
            kernel_size=(self.kernel_width, 1),
            padding='same',
            activation='tanh',
            use_bias=True
        )(output)
        output = tf.keras.layers.MaxPooling2D(
            pool_size=(self.pool_width, 1)
        )(output)
        new_feat_output = tf.keras.layers.Flatten()(output)
        new_feat_output = tf.keras.layers.Dense(
            units=output.shape[1] * output.shape[2] * self.new_feat_filters,
            activation='tanh',
            use_bias=True
        )(new_feat_output)
        new_feat_output = tf.reshape(new_feat_output,
                                     shape=(-1, output.shape[1] * self.new_feat_filters, output.shape[2]))

        return output, new_feat_output


class BiInteraction(tf.keras.Model):

    def __init__(self, mode='all', **kwargs):

        super(BiInteraction, self).__init__(**kwargs)

        self.mode = mode

    def call(self, inputs, **kwargs):

        output = list()
        embedding_size = inputs[0].shape[-1]

        if self.mode == 'all':
            W = self.add_weight(
                shape=(embedding_size, embedding_size),
                initializer='glorot_uniform',
                regularizer=tf.keras.regularizers.l2(1e-5),
                trainable=True
            )
            for i in range(len(inputs) - 1):
                for j in range(i, len(inputs)):
                    inter = tf.tensordot(inputs[i], W, axes=(-1, 0)) * inputs[j]
                    output.append(inter)

        elif self.mode == 'each':
            for i in range(len(inputs) - 1):
                W = self.add_weight(
                    shape=(embedding_size, embedding_size),
                    initializer='glorot_uniform',
                    regularizer=tf.keras.regularizers.l2(1e-5),
                    trainable=True
                )
                for j in range(i, len(inputs)):
                    inter = tf.tensordot(inputs[i], W, axes=(-1, 0)) * inputs[j]
                    output.append(inter)

        elif self.mode == 'interaction':
            for i in range(len(inputs) - 1):
                for j in range(i, len(inputs)):
                    W = self.add_weight(
                        shape=(embedding_size, embedding_size),
                        initializer='glorot_uniform',
                        regularizer=tf.keras.regularizers.l2(1e-5),
                        trainable=True
                    )
                    inter = tf.tensordot(inputs[i], W, axes=(-1, 0)) * inputs[j]
                    output.append(inter)

        output = tf.concat(output, axis=1)
        return output


class SENet(tf.keras.Model):

    def __init__(self, axis=-1, reduction=4, **kwargs):

        super(SENet, self).__init__(**kwargs)

        self.axis = axis
        self.reduction = reduction

    def call(self, inputs, **kwargs):

        # inputs [batch_size, feats_num, embedding_size]
        feats_num = inputs.shape[1]

        weights = tf.reduce_mean(inputs, axis=self.axis, keepdims=False)     # [batch_size, feats_num]
        W1 = self.add_weight(
            shape=(feats_num, self.reduction),
            trainable=True,
            initializer='glorot_normal'
        )
        W2 = self.add_weight(
            shape=(self.reduction, feats_num),
            trainable=True,
            initializer='glorot_normal'
        )
        weights = tf.keras.activations.relu(tf.tensordot(weights, W1, axes=(-1, 0)))
        weights = tf.keras.activations.relu(tf.tensordot(weights, W2, axes=(-1, 0)))

        weights = tf.expand_dims(weights, axis=-1)
        output = tf.multiply(weights, inputs)  # [batch_size, feats_num, embedding_size]

        return output
