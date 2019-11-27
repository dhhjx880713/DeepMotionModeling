# encoding: UTF-8

import tensorflow as tf
import numpy as np
from datetime import datetime
import sys
sys.path.append(r'../')
rng = np.random.RandomState(23456)
regularizer = tf.keras.regularizers.l2(l=0.5 * (0.01))
from nn.spectrum_pooling import spectrum_pooling_1d
from nn.unpooling import average_unpooling_1d, spectrum_unpooling_1d


class CNNFullyConnectedAutoencoder(object):
    
    def __init__(self, name, n_frames, n_dims, kernel_size, encode_activation, decode_activation, hidden_units,
                 n_epoches, batchsize, npc):
        self.name = name
        self.n_frames = n_frames
        self.n_dims = n_dims
        self.kernel_size = kernel_size
        self.encode_activation = encode_activation
        self.decode_activation = decode_activation
        self.hidden_units = hidden_units
        self.n_epoches = n_epoches
        self.batchsize = batchsize
        self.npc = npc
        self.sess = tf.compat.v1.InteractiveSession()

    @staticmethod
    def conv_1d_encode_layer(input, name, hidden_units, pooling, kernel_size, activation,
                             dropout_rate=0.25, reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope(name, reuse=reuse):
            dropout_res = tf.compat.v1.layers.dropout(input, rate=dropout_rate)
        if activation is None:
            conv_res = tf.compat.v1.layers.conv1d(dropout_res, hidden_units, kernel_size, padding='same', activation=tf.nn.relu,
                                        data_format='channels_first',
                                        kernel_regularizer=regularizer)
            conv_res = tf.contrib.layers.batch_norm(conv_res,
                                                    center=True,
                                                    scale=True,
                                                    is_training=True,
                                                    scope='bn')
        else:
            conv_res = tf.compat.v1.layers.conv1d(dropout_res, hidden_units, kernel_size, padding='same', activation=None,
                                        data_format='channels_first',
                                        kernel_regularizer=regularizer)
            conv_res = tf.contrib.layers.batch_norm(conv_res,
                                                    center=True,
                                                    scale=True,
                                                    is_training=True,
                                                    scope='bn')
            conv_res = activation(conv_res)
        if pooling is None:
            return conv_res
        elif pooling == 'average':
            pool_layer = tf.compat.v1.layers.average_pooling1d(conv_res, 2, strides=2, data_format='channels_first')
            return pool_layer
        elif pooling == 'max':
            pool_layer = tf.compat.v1.layers.max_pooling1d(conv_res, 2, strides=2, data_format='channels_first')
            return pool_layer
        elif pooling == 'spectrum':
            pool_layer = spectrum_pooling_1d(conv_res, 2, N=512, data_format='channels_first')
            return pool_layer
        else:
            raise KeyError('Unknown pooling!')

    @staticmethod
    def conv_1d_decode_layer(input, name, n_features, unpooling, kernel_size, activation, dropout_rate=0.25,
                             reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope(name, reuse=reuse):
            ## unpooling
            if unpooling == 'average':
                layer1 = average_unpooling_1d(input, pool_size=2, data_format='channels_first')
            elif unpooling == 'spectrum':
                layer1 = spectrum_unpooling_1d(input, pool_size=2, N=512, data_format='channels_first')
            else:
                layer1 = input
            layer2 = tf.compat.v1.layers.dropout(layer1, rate=dropout_rate)
            layer3 = tf.compat.v1.layers.conv1d(layer2, n_features, kernel_size, padding='same', activation=activation,
                                      data_format='channels_first', kernel_regularizer=regularizer)
            return layer3

    @staticmethod
    def fully_connected_layer(input, hidden_num, name='fully_connected_layer', activation=None,
                              reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope(name, reuse=reuse):
            layer_output = tf.compat.v1.layers.dense(input, hidden_num, reuse=reuse)
            if activation is None:
                layer_output = tf.contrib.layers.batch_norm(layer_output,
                                                            center=True,
                                                            scale=True,
                                                            is_training=True,
                                                            scope='bn')
                return layer_output
            else:
                layer_output = tf.contrib.layers.batch_norm(layer_output,
                                                            center=True,
                                                            scale=True,
                                                            is_training=True,
                                                            scope='bn')
                return activation(layer_output)

    def encode(self, input):
        layer1_conv = CNNFullyConnectedAutoencoder.conv_1d_encode_layer(input,
                                                                        name='conv_layer1',
                                                                        hidden_units=self.hidden_units,
                                                                        pooling='average',
                                                                        kernel_size=self.kernel_size,
                                                                        activation=self.encode_activation)
        layer2_input = tf.reshape(layer1_conv, (-1, layer1_conv.shape[1] * layer1_conv.shape[2]))
        layer2_fc = CNNFullyConnectedAutoencoder.fully_connected_layer(layer2_input,
                                                                       hidden_num=1024,
                                                                       name='fc_layer2',
                                                                       activation=self.encode_activation)
        layer3_fc = CNNFullyConnectedAutoencoder.fully_connected_layer(layer2_fc,
                                                                       hidden_num=512,
                                                                       name='fc_layer3',
                                                                       activation=self.encode_activation)
        layer4_fc = CNNFullyConnectedAutoencoder.fully_connected_layer(layer3_fc,
                                                                       hidden_num=256,
                                                                       name='fc_layer4',
                                                                       activation=self.encode_activation)
        layer5_fc = CNNFullyConnectedAutoencoder.fully_connected_layer(layer4_fc,
                                                                       hidden_num=128,
                                                                       name='fc_layer5',
                                                                       activation=self.encode_activation)
        layer6_fc = CNNFullyConnectedAutoencoder.fully_connected_layer(layer5_fc,
                                                                       hidden_num=32,
                                                                       name='fc_layer6',
                                                                       activation=self.encode_activation)
        layer7_fc = CNNFullyConnectedAutoencoder.fully_connected_layer(layer6_fc,
                                                                       hidden_num=self.npc,
                                                                       name='fc_layer7',
                                                                       activation=self.encode_activation)
        return layer7_fc

    def decode(self, input):
        layer8_fc = CNNFullyConnectedAutoencoder.fully_connected_layer(input,
                                                                       hidden_num=32,
                                                                       name='fc_layer8',
                                                                       activation=self.decode_activation)
        layer9_fc = CNNFullyConnectedAutoencoder.fully_connected_layer(layer8_fc,
                                                                       hidden_num=128,
                                                                       name='fc_layer9',
                                                                       activation=self.decode_activation)
        layer10_fc = CNNFullyConnectedAutoencoder.fully_connected_layer(layer9_fc,
                                                                        hidden_num=256,
                                                                        name='fc_layer10',
                                                                        activation=self.decode_activation)
        layer11_fc = CNNFullyConnectedAutoencoder.fully_connected_layer(layer10_fc,
                                                                        hidden_num=512,
                                                                        name='fc_layer11',
                                                                        activation=self.decode_activation)
        layer12_fc = CNNFullyConnectedAutoencoder.fully_connected_layer(layer11_fc,
                                                                        hidden_num=1024,
                                                                        name='fc_layer12',
                                                                        activation=self.decode_activation)
        layer13_fc = CNNFullyConnectedAutoencoder.fully_connected_layer(layer12_fc,
                                                                        hidden_num=self.hidden_units * self.n_frames // 2,
                                                                        name='fc_layer13',
                                                                        activation=self.decode_activation)
        deconv_input = tf.reshape(layer13_fc, (-1, self.hidden_units, self.n_frames // 2))

        layer14_deconv = CNNFullyConnectedAutoencoder.conv_1d_decode_layer(deconv_input,
                                                                           name='deconv_layer14',
                                                                           unpooling='average',
                                                                           n_features=self.n_dims,
                                                                           kernel_size=self.kernel_size,
                                                                           activation=None)
        return layer14_deconv

    def create_model(self, reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope(self.name, reuse):
            self.input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.n_dims, self.n_frames])
            self.latent_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.npc])
            self.encode_op = self.encode(self.input)
            decoder_output = self.decode(self.encode_op)
            self.loss = tf.reduce_mean(input_tensor=tf.pow(self.input - decoder_output, 2))
            self.decode_op = self.decode(self.latent_input)
            self.model_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            self.saver = tf.compat.v1.train.Saver(self.model_params)

    def get_params(self):
        return self.model_params

    def __call__(self, input_data):
        latent_value = self.sess.run(self.encode_op, feed_dict={self.input: input_data})
        return self.sess.run(self.decode_op, feed_dict={self.latent_input: latent_value})

    def train(self, training_data, learning_rate=0.01):
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = self.optimizer.minimize(self.loss)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        n_samples, n_features, n_frames = training_data.shape
        last_mean = 0
        for epoch in range(self.n_epoches):
            batchinds = np.arange(n_samples // self.batchsize)
            rng.shuffle(batchinds)
            c = []
            for bii, bi in enumerate(batchinds):

                self.sess.run(train_op, feed_dict={self.input: training_data[bi*self.batchsize: (bi+1)*self.batchsize]})
                c.append(self.sess.run(self.loss, feed_dict={self.input: training_data[bi*self.batchsize: (bi+1)*self.batchsize]}))
                if np.isnan(c[-1]): return
                if bii % (int(len(batchinds) / 10) + 1) == 0:
                    sys.stdout.write('\r[Epoch %i]  %0.1f%% mean %.5f' % (epoch, 100 * float(bii) / len(batchinds),
                                                                          np.mean(c)))
                    sys.stdout.flush()
            curr_mean = np.mean(c)
            diff_mean, last_mean = curr_mean-last_mean, curr_mean
            print('\r[Epoch %i] 100.0%% mean %.5f diff %.5f %s' %
                (epoch, curr_mean, diff_mean, str(datetime.now())[11:19]))

    def load_model(self, model_file):
        self.saver.restore(self.sess, model_file)

    def save_model(self, model_file):
        save_path = self.saver.save(self.sess, model_file)
        print("Model saved in file: %s " % save_path)