import tensorflow as tf
import numpy as np
from .Layer import Layer


class BiasLayer(Layer):

    def __init__(self, shape, name=''):
        self.name = name
        # zero initialization
        b_value = np.zeros(shape, dtype=np.float32)
        with tf.compat.v1.variable_scope(self.name):
            self.b = tf.Variable(b_value, name='b', dtype=tf.float32)
        self.params = [self.b]

    def __call__(self, input):
        return input + self.b