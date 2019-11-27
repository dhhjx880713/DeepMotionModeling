# encoding: UTF-8
import tensorflow as tf


class CausalConv1D(tf.compat.v1.layers.Conv1D):
    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 dilation_rate=1,
                 ##padding='valid', lock the padding method to be valid
                 data_format='channels_last',
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=tf.compat.v1.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(CausalConv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="valid",
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs
        )

    def call(self, inputs):
        ## left zero padding
        padding = tf.keras.layers.ZeroPadding1D(((self.kernel_size[0] - 1 ) *self.dilation_rate[0], 0))(inputs)
        conv = super(CausalConv1D, self).call(padding) 
        return conv

