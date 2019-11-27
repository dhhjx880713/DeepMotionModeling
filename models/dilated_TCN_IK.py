import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute())+ r'/..')
import tensorflow as tf
import numpy as np
from nn.network import Network
from nn.causal_convolution import CausalConv1D
from datetime import datetime


class DilatedTCN_IK(Network):


    def __init__(self, name, sess=None):
        return super().__init__(name, sess=sess)
    
    def build(self, input_shape, output_shape, reuse=tf.compat.v1.AUTO_REUSE):
        
        with tf.compat.v1.variable_scope(self.name, reuse):
            self.input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, None, input_shape))
            self.output = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, None, output_shape))
            layer1_dropout = tf.compat.v1.layers.dropout(self.input, rate=0.5)
            layer1 = CausalConv1D(128, kernel_size=5, dilation_rate=1, activation=tf.nn.elu)(layer1_dropout)
            layer2_dropout = tf.compat.v1.layers.dropout(layer1, rate=0.5)
            layer2 = CausalConv1D(256, kernel_size=5, dilation_rate=2, activation=tf.nn.elu)(layer2_dropout)
            layer3_dropout = tf.compat.v1.layers.dropout(layer2, rate=0.5)
            layer3 = CausalConv1D(128, kernel_size=5, dilation_rate=4, activation=tf.nn.elu)(layer3_dropout)
            layer4_dropout = tf.compat.v1.layers.dropout(layer3, rate=0.5)
            self.output_op = CausalConv1D(output_shape, kernel_size=5, dilation_rate=1, activation=tf.nn.elu)(layer4_dropout)
            self.params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            self.saver = tf.compat.v1.train.Saver(self.params)
            self.cost = tf.reduce_mean(input_tensor=tf.pow(self.output_op - self.output, 2))
        
    def train(self, input_data, output_data, epochs, learning_rate, rng=np.random.RandomState(123456)):
        """one by one training for variable-length input
        
        Arguments:
            input_data {list} -- each item is a variable-length input
            output_data {list} -- the corresponding variable-length output
            epochs {int} -- number of epochs
            learning_rate {float} 
        
        """
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = self.optimizer.minimize(self.cost)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        last_mean = 0
        for epoch in range(epochs):
            # rng.shuffle(input_data)
            loss = []
            for i in range(len(input_data)):
                input_i = np.asarray(input_data[i])
                input_i = input_i[np.newaxis, :, :]
                output_i = np.asarray(output_data[i])
                output_i = output_i[np.newaxis, :, :]
                self.sess.run(train_op, feed_dict={self.input: input_i, self.output: output_i})
                loss.append(self.sess.run(self.cost, feed_dict={self.input: input_i, self.output: output_i}))
                if np.isnan(loss[-1]): return
                if i % 10 == 0:
                    sys.stdout.write('\r[Epoch %i]  %0.1f%% mean %.5f' % (epoch, 100 * float(i) / len(input_data),
                                                                          np.mean(loss)))
                    sys.stdout.flush()                      
            curr_mean = np.mean(loss)
            diff_mean, last_mean = curr_mean-last_mean, curr_mean
            print('\r[Epoch %i] 100.0%% mean %.5f diff %.5f %s' %
                (epoch, curr_mean, diff_mean, str(datetime.now())[11:19])) 
    
    def predict(self, input_data):
        """
        
        Arguments:
            input_data {numpy.ndarray} -- n_frames * n_dims
        
        Returns:
            [type] -- [description]
        """

        return self.sess.run(self.output_op, feed_dict={self.input: input_data[np.newaxis, :, :]})