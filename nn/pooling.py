# encoding: UTF-8

import tensorflow as tf


def average_pooling_1d(inputs, pool_size, stride, padding, data_format='channels_first', name='average_pooling_1d'):
    with tf.compat.v1.variable_scope(name):
        if data_format == 'channels_first':
            #### inputs shape is n_batches * n_channels * n_frames
            #### expand dimension on last axis
            pool_size = (1, 1, ) + (pool_size,) + (1,)
            strides = (1, 1, ) + (stride, ) + (1,)
            reshaped_inputs = tf.expand_dims(inputs, -1)
            res = tf.nn.avg_pool2d(input=reshaped_inputs, ksize=pool_size, strides=strides, padding=padding,
                                 data_format='NCHW', name=name)
            return tf.squeeze(res, axis=-1)
        else:
            #### inputs shape is n_batches * n_frames * n_channels
            reshaped_inputs = tf.expand_dims(inputs, 2)
            pool_size = (1, ) + (pool_size, ) + (1, 1, )
            strides = (1, ) + (stride,) + (1, 1, )
            res = tf.nn.avg_pool2d(input=reshaped_inputs, ksize=pool_size, strides=strides, padding=padding,
                                 data_format='NHWC', name=name)
            return tf.squeeze(res, axis=2)