from keras import backend as K
from keras.engine.topology import Layer
from keras.activations import softmax
import numpy as np

from tensorflow.python.ops.linalg.linear_operator_circulant import LinearOperatorCirculant


class SelfAttention(Layer):
    def __init__(self, output_dim, add_abs_position=False,
                 add_rel_position=False, weight_normalization=False, **kwargs):
        self.output_dim = output_dim
        self.add_abs_position = add_abs_position
        self.add_rel_position = add_rel_position
        self.weight_normalization = weight_normalization
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_Q = self.add_weight(name='W_Q',
                                   shape=(input_shape[2], self.output_dim),
                                   initializer='glorot_normal',
                                   trainable=True)
        self.W_K = self.add_weight(name='W_K',
                                   shape=(input_shape[2], self.output_dim),
                                   initializer='glorot_normal',
                                   trainable=True)
        self.W_V = self.add_weight(name='W_V',
                                   shape=(input_shape[2], self.output_dim),
                                   initializer='glorot_normal',
                                   trainable=True)
        self.myN = input_shape[1]
        if self.add_abs_position:
            self.P = self.add_weight(name='P',
                                     shape=(input_shape[1], input_shape[1]),
                                     initializer='glorot_normal',
                                     trainable=True)
        if self.add_rel_position:
            self.R = self.add_weight(name='R',
                                     shape=(2 * input_shape[1],),
                                     initializer='glorot_normal',
                                     trainable=True)
            self.Rop = LinearOperatorCirculant(self.R, input_output_dtype=K.tf.float32)
        if self.weight_normalization:
            self.tau_Q = self.add_weight(name='tau_Q',
                                         shape=(1,),
                                         initializer='Ones',
                                         trainable=True)
            self.tau_K = self.add_weight(name='tau_K',
                                         shape=(1,),
                                         initializer='Ones',
                                         trainable=True)
            self.tau_V = self.add_weight(name='tau_V',
                                         shape=(1,),
                                         initializer='Ones',
                                         trainable=True)
        super(SelfAttention, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # (batch, maxlen, d_x)

        if self.weight_normalization:
            XW_Q = K.dot(x, self.W_Q)
            XW_Q = K.tf.multiply(self.tau_Q, XW_Q)  # + K.dot(x, self.c_Q) + self.b_Q
            XW_K = K.dot(x, self.W_K)
            XW_K = K.tf.multiply(self.tau_K, XW_K)  # + K.dot(x, self.c_K) + self.b_K
            XW_V = K.dot(x, self.W_V)
            XW_V = K.tf.multiply(self.tau_V, XW_V)  # + K.dot(x, self.c_K) + self.b_K

        else:
            XW_Q = K.dot(x, self.W_Q)
            XW_K = K.dot(x, self.W_K)
            XW_V = K.dot(x, self.W_V)
        E = K.batch_dot(XW_Q, K.permute_dimensions(XW_K, (0, 2, 1)))
        E = K.tf.scalar_mul(np.sqrt(1.0 / self.output_dim), E)
        if self.add_abs_position:
            E = E + self.P
        if self.add_rel_position:
            E = E + self.Rop.matmul(K.tf.eye(2 * self.myN))[:self.myN, :self.myN]
        A = softmax(E, axis=2)
        self.attention = A
        R = K.batch_dot(A, XW_V)
        return R

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        base_config = super(SelfAttention, self).get_config()
        base_config['output_dim'] = self.output_dim

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SelfAttentionPart1(Layer):
    def __init__(self, output_dim,
                 add_abs_position=False,
                 add_rel_position=False,
                 weight_normalization=False, **kwargs):
        self.output_dim = output_dim
        self.add_abs_position = add_abs_position
        self.add_rel_position = add_rel_position
        self.weight_normalization = weight_normalization
        super(SelfAttentionPart1, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_Q = self.add_weight(name='W_Q',
                                   shape=(input_shape[2], self.output_dim),
                                   initializer='glorot_normal',
                                   trainable=True)
        self.W_K = self.add_weight(name='W_K',
                                   shape=(input_shape[2], self.output_dim),
                                   initializer='glorot_normal',
                                   trainable=True)
        self.myN = input_shape[1]
        if self.add_abs_position:
            self.P = self.add_weight(name='P',
                                     shape=(input_shape[1], input_shape[1]),
                                     initializer='glorot_normal',
                                     trainable=True)
        if self.add_rel_position:
            self.R = self.add_weight(name='R',
                                     shape=(2 * input_shape[1],),
                                     initializer='glorot_normal',
                                     trainable=True)
            self.Rop = LinearOperatorCirculant(self.R, input_output_dtype=K.tf.float32)
        if self.weight_normalization:
            self.tau_Q = self.add_weight(name='tau_Q',
                                         shape=(1,),
                                         initializer='Ones',
                                         trainable=True)
            self.tau_K = self.add_weight(name='tau_K',
                                         shape=(1,),
                                         initializer='Ones',
                                         trainable=True)
        super(SelfAttentionPart1, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # (batch, maxlen, d_x)

        if self.weight_normalization:
            XW_Q = K.dot(x, self.W_Q)
            XW_Q = K.tf.multiply(self.tau_Q, XW_Q)  # + K.dot(x, self.c_Q) + self.b_Q
            XW_K = K.dot(x, self.W_K)
            XW_K = K.tf.multiply(self.tau_K, XW_K)  # + K.dot(x, self.c_K) + self.b_K

        else:
            XW_Q = K.dot(x, self.W_Q)
            XW_K = K.dot(x, self.W_K)
        E = K.batch_dot(XW_Q, K.permute_dimensions(XW_K, (0, 2, 1)))
        E = K.tf.scalar_mul(np.sqrt(1.0 / self.output_dim), E)

        if self.add_abs_position:
            E = E + self.P
        if self.add_rel_position:
            E = E + self.Rop.matmul(K.tf.eye(2 * self.myN))[:self.myN, :self.myN]
        A = softmax(E, axis=2)

        return A

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])

    def get_config(self):
        base_config = super(SelfAttentionPart1, self).get_config()
        base_config['output_dim'] = self.output_dim

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SelfAttentionPart2(Layer):
    def __init__(self, output_dim, weight_normalization=False, **kwargs):
        self.output_dim = output_dim
        self.weight_normalization = weight_normalization
        super(SelfAttentionPart2, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_V = self.add_weight(name='W_v',
                                   shape=(input_shape[0][2], self.output_dim),
                                   initializer='glorot_normal',
                                   trainable=True)
        if self.weight_normalization:
            self.tau_V = self.add_weight(name='tau_V',
                                         shape=(1,),
                                         initializer='Ones',
                                         trainable=True)
        super(SelfAttentionPart2, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # x has shape (batch, i [n], j[d_x])
        A = x[0]
        XW_V = K.dot(A, self.W_V)
        if self.weight_normalization:
            XW_V = K.tf.multiply(self.tau_V, XW_V)
        R = K.batch_dot(x[1], XW_V)
        return R

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)
