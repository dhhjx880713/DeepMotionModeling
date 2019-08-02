import tensorflow as tf
import numpy as np
from .Layer import Layer


class HiddenLayer(Layer):

    def __init__(self, weights_shape, name='', rng=np.random):

        self.name = name
        W_bound = np.sqrt(6. / np.prod(weights_shape[-2:]))
        W = np.array(rng.uniform(low=-W_bound, high=W_bound, size=weights_shape),
                     dtype=np.float32)
        with tf.variable_scope(self.name):
            self.W = tf.Variable(W, name='W', dtype=tf.float32)

        self.params = [self.W]

    def __call__(self, input):
        return self.W.dot(input.T).T

    def cost(self, gamma=0.01):
        return gamma * tf.reduce_mean(abs(self.W))
