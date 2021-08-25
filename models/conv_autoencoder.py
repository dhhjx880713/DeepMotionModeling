# encoding: UTF-8
import tensorflow as tf
import numpy as np
from datetime import datetime
import sys

from tensorflow.python.keras.backend import dropout

sys.path.append(r'../')
rng = np.random.RandomState(23456)
# regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
regularizer = None
from nn.spectrum_pooling import spectrum_pooling_1d
from nn.unpooling import average_unpooling_1d, spectrum_unpooling_1d
from utilities.utils import gram_matrix, convert_anim_to_point_cloud_tf
# tf.compat.v1.disable_eager_execution()

class ConvAutoEncoder(object):
    def __init__(self, name, n_frames, n_dims, kernel_size, encode_activation, decode_activation, hidden_units,
                 pooling, dropout_rate=0.25, batch_norm=False, edge_padding=True, sess=None):

        self.name = name
        self.n_frames = n_frames
        self.n_dims = n_dims
        self.kernel_size = kernel_size
        self.encode_activation = encode_activation
        self.decode_activation = decode_activation
        self.hidden_units = hidden_units
        self.pooling = pooling
        self.edge_padding = edge_padding
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        if sess is not None:
            self.sess = sess
        else:
            self.sess = tf.compat.v1.InteractiveSession()

    @staticmethod
    def conv_1d_encode_layer(input, name, hidden_units, pooling, kernel_size, activation, dropout_rate=0.25, 
                             reuse=tf.compat.v1.AUTO_REUSE, batch_norm=False, edge_padding=True, train=True):
        with tf.compat.v1.variable_scope(name, reuse=reuse):
            if edge_padding:
                ### extend input for edge effects
                input = tf.concat([input,
                                   tf.tile(input[:, :, -2: -1], [1, 1, kernel_size - 1])
                                   ], axis=-1)
                padding = 'valid'
            else:
                padding = 'same'
            if train:
                dropout_res = tf.compat.v1.layers.dropout(input, rate=dropout_rate)
            else:
                dropout_res = input
            if activation is None:
                conv_res = tf.compat.v1.layers.conv1d(dropout_res, hidden_units, kernel_size, padding=padding,
                                            activation=activation,
                                            data_format='channels_first',
                                            kernel_regularizer=regularizer,
                                            reuse=reuse)
                # conv_res = tf.contrib.layers.batch_norm(conv_res,
                #                                         center=True,
                #                                         scale=True,
                #                                         is_training=True,
                #                                         scope='bn')
            else:
                conv_res = tf.compat.v1.layers.conv1d(dropout_res, hidden_units, kernel_size, padding=padding, activation=None,
                                            data_format='channels_first',
                                            kernel_regularizer=regularizer,
                                            reuse=reuse)
                if batch_norm:
                    conv_res = tf.contrib.layers.batch_norm(conv_res,
                                                            center=True,
                                                            scale=True,
                                                            is_training=True,
                                                            scope='bn')
                    conv_res = activation(conv_res)
                else:
                    conv_res = activation(conv_res)
            pool_size = 2
            if pooling is None:
                return conv_res
            elif pooling == 'average':
                pool_layer = tf.compat.v1.layers.average_pooling1d(conv_res, pool_size, strides=2, data_format='channels_first')
                return pool_layer
            elif pooling == 'max':
                pool_layer = tf.compat.v1.layers.max_pooling1d(conv_res, pool_size, strides=2, data_format='channels_first')
                return pool_layer
            elif pooling == 'spectrum':
                pool_layer = spectrum_pooling_1d(conv_res, pool_size, N=512, data_format='channels_first')
                return pool_layer
            else:
                raise KeyError('Unknown pooling!')

    @staticmethod
    def conv_1d_decode_layer(input, name, n_features, unpooling, kernel_size, activation, dropout_rate=0.25,
                             reuse=tf.compat.v1.AUTO_REUSE, edge_padding=True, train=True):

        with tf.compat.v1.variable_scope(name, reuse=reuse):
            ## unpooling
            if unpooling == 'average':
                unpool = average_unpooling_1d(input, pool_size=2, data_format='channels_first')
            elif unpooling == 'spectrum':
                unpool = spectrum_unpooling_1d(input, pool_size=2, N=512, data_format='channels_first')
            else:
                unpool = input
            if edge_padding:
                unpool = tf.concat([unpool,
                                    tf.tile(unpool[:, :, -2: -1], [1, 1, kernel_size - 1])
                                    ],
                                    axis=-1)
                padding = 'valid'
            else:
                padding = 'same'
            if train:
                dropout = tf.compat.v1.layers.dropout(unpool, rate=dropout_rate)
            else:
                dropout = unpool
            conv_res = tf.compat.v1.layers.conv1d(dropout, n_features, kernel_size, padding=padding, activation=activation,
                                      data_format='channels_first', kernel_regularizer=regularizer, reuse=reuse)

            return conv_res

    def encode(self, input, train):
        encoder_layer = ConvAutoEncoder.conv_1d_encode_layer(input, name='encoder_layer1',
                                                             hidden_units=self.hidden_units,
                                                             pooling=self.pooling,
                                                             kernel_size=self.kernel_size,
                                                             activation=self.encode_activation,
                                                             batch_norm=self.batch_norm,
                                                             edge_padding=self.edge_padding,
                                                             train=train,
                                                             dropout_rate=self.dropout_rate)
        return encoder_layer

    def decode(self, input, train):
        decoder_layer = ConvAutoEncoder.conv_1d_decode_layer(input, name='decoder_layer1',
                                                             n_features=self.n_dims,
                                                             unpooling=self.pooling,
                                                             kernel_size=self.kernel_size,
                                                             activation=self.decode_activation,
                                                             edge_padding=self.edge_padding,
                                                             train=train,
                                                             dropout_rate=self.dropout_rate)
        return decoder_layer
    
    def apply_FK(self):
        pass

    def create_model_with_weights(self, reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope(self.name, reuse=reuse):
            self.input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.n_dims, self.n_frames))
            encoder_layer = self.encode(self.input)
            decoder_layer = self.decode(encoder_layer)

            self.latent_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.hidden_units, self.n_frames//2))
            self.decoder_op = self.decode(self.latent_input)
            self.encoder_op = encoder_layer
            self.model_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            self.saver = tf.compat.v1.train.Saver(self.model_params)
            self.loss = tf.reduce_mean(input_tensor=tf.pow(self.input[:, :-3, :] - decoder_layer[:, :-3, :], 2)) + \
                        20000 * tf.reduce_mean(input_tensor=tf.pow(self.input[:, -3: -1, :] - decoder_layer[:, -3: -1, :], 2)) + \
                        30000 * tf.reduce_mean(input_tensor=tf.pow(self.input[:, -1, :] - decoder_layer[:, -1, :], 2))

    def create_model(self, batchsize, train=True, reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope(self.name, reuse=reuse):
            self.batchsize = batchsize
            self.input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.n_dims, self.n_frames))
            encoder_layer = self.encode(self.input, train)
            decoder_layer = self.decode(encoder_layer, train)

            self.latent_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.hidden_units, self.n_frames//2))
            self.decoder_op = self.decode(self.latent_input, train=False)
            self.encoder_op = encoder_layer
            self.model_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            self.saver = tf.compat.v1.train.Saver(self.model_params)
            self.loss = tf.reduce_mean(input_tensor=tf.pow(self.input - decoder_layer, 2))

    def create_model_joint_pos_loss(self, batchsize, Xmean, Xstd, reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope(self.name, reuse=reuse):
            self.batchsize = batchsize
            self.input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.n_dims, self.n_frames))
            encoder_layer = self.encode(self.input)
            decoder_layer = self.decode(encoder_layer)

            self.latent_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.hidden_units, self.n_frames//2))
            self.decoder_op = self.decode(self.latent_input)
            self.encoder_op = encoder_layer
            self.model_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            self.saver = tf.compat.v1.train.Saver(self.model_params)
            input_joint_pos = []
            output_joint_pos = []
            input_anim = tf.transpose(a=self.input, perm=[0, 2, 1]) * Xstd + Xmean
            output_anim = tf.transpose(a=decoder_layer, perm=[0, 2, 1]) * Xstd + Xmean 
            for i in range(self.batchsize):
                input_joint_pos.append(convert_anim_to_point_cloud_tf(input_anim[i]))
                output_joint_pos.append(convert_anim_to_point_cloud_tf(output_anim[i]))
            input_joint_pos = tf.stack(input_joint_pos, axis=0)
            output_joint_pos = tf.stack(output_joint_pos, axis=0)
            self.loss = tf.reduce_mean(input_tensor=tf.norm(tensor=input_joint_pos - output_joint_pos, axis=-1))

    def train(self, training_data, n_epochs, learning_rate=0.001):

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = self.optimizer.minimize(self.loss)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        n_samples, n_features, n_frames = training_data.shape
        last_mean = 0
        for epoch in range(n_epochs):
            batchinds = np.arange(n_samples // self.batchsize)
            rng.shuffle(batchinds)
            c = []
            for bii, bi in enumerate(batchinds):

                self.sess.run(train_op,
                              feed_dict={self.input: training_data[bi * self.batchsize: (bi + 1) * self.batchsize]})
                c.append(self.sess.run(self.loss, feed_dict={
                    self.input: training_data[bi * self.batchsize: (bi + 1) * self.batchsize]}))
                if np.isnan(c[-1]): return
                if bii % (int(len(batchinds) / 10) + 1) == 0:
                    sys.stdout.write('\r[Epoch %i]  %0.1f%% mean %.5f' % (epoch, 100 * float(bii) / len(batchinds),
                                                                          np.mean(c)))
                    sys.stdout.flush()
            curr_mean = np.mean(c)
            diff_mean, last_mean = curr_mean - last_mean, curr_mean
            print('\r[Epoch %i] 100.0%% mean %.5f diff %.5f %s' %
                  (epoch, curr_mean, diff_mean, str(datetime.now())[11:19]))

    def get_params(self):
        return self.model_params

    def __call__(self, input_data):
        latent_value = self.sess.run(self.encoder_op, feed_dict={self.input: input_data})
        return self.sess.run(self.decoder_op, feed_dict={self.latent_input: latent_value})

    def load_model(self, model_file):
        self.saver.restore(self.sess, model_file)

    def save_model(self, model_file):
        save_path = self.saver.save(self.sess, model_file)
        print("Model saved in file: %s " % save_path)

    def encode_data(self, input_data):
        return self.sess.run(self.encoder_op, feed_dict={self.input: input_data})

    def decode_data(self, input_data):
        return self.sess.run(self.decoder_op, feed_dict={self.latent_input: input_data})

    def transfer_style(self, content_motion, style_motion):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            epoches = 250
            encodered_content_motion = self.encode_data(content_motion)
            h_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
            content_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, self.n_dims, self.n_frames))
            style_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, self.n_dims, self.n_frames))
            content_encoder = self.encode(content_input)
            style_encoder = self.encode(style_input)
            encoded_style_motion = self.encode_data(style_motion)
            s = 100.0
            c = 1.0
            H = tf.Variable(initial_value=tf.random.normal(shape=[1, self.hidden_units, int(self.n_frames / 2)]),
                            dtype=tf.float32)
            H_decoder = self.decode(H)

            loss_op = c * tf.reduce_mean(input_tensor=tf.pow(H - content_encoder, 2)) + s * tf.reduce_sum(
                input_tensor=tf.pow(gram_matrix(H) - gram_matrix(style_encoder), 2))
            style_loss_op = tf.reduce_sum(input_tensor=tf.pow(gram_matrix(H) - gram_matrix(style_encoder), 2))
            content_loss_op = tf.reduce_mean(input_tensor=tf.pow(H - content_encoder, 2))
            train_op = h_optimizer.minimize(loss_op, var_list=[H])
            self.sess.run(tf.compat.v1.variables_initializer(h_optimizer.variables()))

            # assign_op = H.assign(encodered_content_motion)
            # self.sess.run(assign_op)
            for epoch in range(epoches):
                self.sess.run(train_op, feed_dict={content_input: content_motion,
                                                   style_input: style_motion})
                print('error is: ', self.sess.run(loss_op, feed_dict={content_input: content_motion,
                                                                      style_input: style_motion}))
                print('style loss is: ', self.sess.run(style_loss_op, feed_dict={content_input: content_motion,
                                                                                 style_input: style_motion}))
                print('content loss is: ', self.sess.run(content_loss_op, feed_dict={content_input: content_motion,
                                                                                     style_input: style_motion}))
            res = self.sess.run(H_decoder, feed_dict={self.latent_input: encodered_content_motion})
        return res