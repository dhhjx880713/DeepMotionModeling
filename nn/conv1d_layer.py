# encoding: UTF-8
import tensorflow as tf
import numpy as np


class Conv1D(object):

    def __init__(self, filter_shape, stride, name='conv1d', rng=np.random):
        '''

        :param filter_shape: filter_width * in_channels * out_channels
        :param rng:
        '''
        self.filter_shape = filter_shape
        self.stride = stride
        self.output_shape = ()
        self.rng = rng
        self.name = name
        fan_in = np.prod(filter_shape[:-1])
        fan_out = filter_shape[0] * filter_shape[2]
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        W = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape))
        with tf.compat.v1.variable_scope(self.name):
            self.W = tf.Variable(W, name='W', dtype=tf.float32)
        self.params = [self.W]

    def __call__(self, input):
        output = tf.nn.conv1d(input=input, filters=self.W, stride=self.stride, padding='SAME', data_format="NCW", name=self.name)
        return output

    def cost(self, gamma=0.01):
        return gamma * tf.reduce_mean(input_tensor=abs(self.W))