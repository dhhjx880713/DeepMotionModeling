# encoding: UTF-8
import tensorflow as tf
import numpy as np
from datetime import datetime
import sys

sys.path.append(r'../')
# regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
regularizer = None
rng = np.random.RandomState(23456)
from nn.spectrum_pooling import spectrum_pooling_1d
from nn.unpooling import average_unpooling_1d, spectrum_unpooling_1d
from utils import gram_matrix


class ConditionalMotionVaeEncoder(object):

    def __init__(self, npc, n_frames, n_dims, n_epoches, name, batchsize, encoder_activation=None, 
                 decoder_activation=None, n_random_samples=1, kernel_size=15, hidden_units=256, sess=None):
        self.npc = npc
        self.input_dims = n_frames * n_dims
        self.n_epoches = n_epoches
        self.name = name
        self.batchsize = batchsize
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.n_random_samples = n_random_samples
        self.n_dims = n_dims
        self.n_frames = n_frames
        self.kernel_size = kernel_size
        self.hidden_units = hidden_units
        if sess is not None:
            self.sess = sess
        else:
            self.sess = tf.compat.v1.InteractiveSession()

    @staticmethod
    def fully_connected_layer(input, hidden_num, name='fully_connected_layer', activation=None):
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            layer_output = tf.compat.v1.layers.dense(input, hidden_num)
            if activation is None:
                # layer_output = tf.contrib.layers.batch_norm(layer_output,
                #                                             center=True,
                #                                             scale=True,
                #                                             is_training=True,
                #                                             scope='bn')
                return layer_output
            else:
                layer_output = tf.contrib.layers.batch_norm(layer_output,
                                                            center=True,
                                                            scale=True,
                                                            is_training=True,
                                                            scope='bn')
                return activation(layer_output)

    @staticmethod
    def conv_1d_encode_layer(input, name, hidden_units, pooling, kernel_size, activation,
                             dropout_rate=0.25, reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope(name, reuse=reuse):
            dropout_res = tf.compat.v1.layers.dropout(input, rate=dropout_rate)
        if activation is None:
            conv_res = tf.compat.v1.layers.conv1d(dropout_res, hidden_units, kernel_size, padding='same', activation=tf.nn.relu,
                                        data_format='channels_first',
                                        kernel_regularizer=regularizer)
            # conv_res = tf.contrib.layers.batch_norm(conv_res,
            #                                         center=True,
            #                                         scale=True,
            #                                         is_training=True,
            #                                         scope='bn')
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

    def encode1(self, input):
        encoder_1 = ConditionalMotionVaeEncoder.fully_connected_layer(input,
                                                                      hidden_num=1024,
                                                                      name='encoder_1',
                                                                      activation=self.encoder_activation)
        encoder_2 = ConditionalMotionVaeEncoder.fully_connected_layer(encoder_1,
                                                           hidden_num=512,
                                                           name='encoder_2',
                                                           activation=self.encoder_activation)
        encoder_3 = ConditionalMotionVaeEncoder.fully_connected_layer(encoder_2,
                                                           hidden_num=256,
                                                           name='encoder_3',
                                                           activation=self.encoder_activation)
        mean_encoder_4 = ConditionalMotionVaeEncoder.fully_connected_layer(encoder_3,
                                                                hidden_num=128,
                                                                name='mean_encoder_4',
                                                                activation=None)
        log_sigma_encoder_4 = ConditionalMotionVaeEncoder.fully_connected_layer(encoder_3,
                                                                     hidden_num=128,
                                                                     name='log_sigma_encoder_4',
                                                                     activation=None)
        mean_encoder_5 = ConditionalMotionVaeEncoder.fully_connected_layer(mean_encoder_4,
                                                                hidden_num=32,
                                                                name='mean_encoder_5',
                                                                activation=None)
        log_sigma_encoder_5 = ConditionalMotionVaeEncoder.fully_connected_layer(log_sigma_encoder_4,
                                                                     hidden_num=32,
                                                                     name='log_sigma_encoder_5',
                                                                     activation=None)
        mean_encoder_6 = ConditionalMotionVaeEncoder.fully_connected_layer(mean_encoder_5,
                                                                hidden_num=self.npc,
                                                                name='mean_encoder_6',
                                                                activation=None)
        log_sigma_encoder_6 = ConditionalMotionVaeEncoder.fully_connected_layer(log_sigma_encoder_5,
                                                                     hidden_num=self.npc,
                                                                     name='log_sigma_encoder_6',
                                                                     activation=None)
        return mean_encoder_6, log_sigma_encoder_6

    def encode(self, input):
        encoder_1 = ConditionalMotionVaeEncoder.fully_connected_layer(input,
                                                                      hidden_num=1024,
                                                                      name='encoder_1',
                                                                      activation=self.encoder_activation)
        encoder_2 = ConditionalMotionVaeEncoder.fully_connected_layer(encoder_1,
                                                                      # hidden_num=256,
                                                                      hidden_num=512,
                                                                      name='encoder_2',
                                                                      activation=self.encoder_activation)
        encoder_3 = ConditionalMotionVaeEncoder.fully_connected_layer(encoder_2,
                                                                      # hidden_num=64,
                                                                      hidden_num=256,
                                                                      name='encoder_3',
                                                                      activation=self.encoder_activation)
        mean_encoder_4 = ConditionalMotionVaeEncoder.fully_connected_layer(encoder_3,
                                                                           hidden_num=self.npc,
                                                                           name='mean_encoder_4',
                                                                           activation=None)
        log_sigma_encoder_4 = ConditionalMotionVaeEncoder.fully_connected_layer(encoder_3,
                                                                                hidden_num=self.npc,
                                                                                name='log_sigma_encoder_4',
                                                                                activation=None)
        return mean_encoder_4, log_sigma_encoder_4

    def decode(self, input):
        decoder_4 = ConditionalMotionVaeEncoder.fully_connected_layer(input,
                                                                      # hidden_num=64,
                                                                      hidden_num=256,
                                                                      name='decoder_4',
                                                                      activation=self.decoder_activation)
        decoder_3 = ConditionalMotionVaeEncoder.fully_connected_layer(decoder_4,
                                                                      # hidden_num=256,
                                                                      hidden_num=512,
                                                                      name='decoder_3',
                                                                      activation=self.decoder_activation)
        decoder_2 = ConditionalMotionVaeEncoder.fully_connected_layer(decoder_3,
                                                                      hidden_num=1024,
                                                                      name='decoder_2',
                                                                      activation=self.decoder_activation)
        decoder_1 = ConditionalMotionVaeEncoder.fully_connected_layer(decoder_2,
                                                                      hidden_num=self.input_dims,
                                                                      name='decoder_1',
                                                                      activation=self.decoder_activation)
        return decoder_1

    def conv_encode(self, input):
        encode_layer = ConditionalMotionVaeEncoder.conv_1d_encode_layer(input,
                                                                        name='conv_encoder_1',
                                                                        hidden_units=self.hidden_units,
                                                                        pooling='average',
                                                                        kernel_size=self.kernel_size,
                                                                        activation=self.encoder_activation)
        return encode_layer

    def conv_decode(self, input):
        decode_layer = ConditionalMotionVaeEncoder.conv_1d_decode_layer(input,
                                                                        name='conv_decoder_1',
                                                                        n_features=self.n_dims,
                                                                        unpooling='average',
                                                                        kernel_size=self.kernel_size,
                                                                        activation=self.decoder_activation)
        return decode_layer

    def encode2(self, input):
        encoder_1 = ConditionalMotionVaeEncoder.fully_connected_layer(input,
                                                                      hidden_num=1024,
                                                                      name='encoder_1',
                                                                      activation=self.encoder_activation)
        encoder_2 = ConditionalMotionVaeEncoder.fully_connected_layer(encoder_1,
                                                           hidden_num=512,
                                                           name='encoder_2',
                                                           activation=self.encoder_activation)
        encoder_3 = ConditionalMotionVaeEncoder.fully_connected_layer(encoder_2,
                                                           hidden_num=256,
                                                           name='encoder_3',
                                                           activation=self.encoder_activation)
        encoder_4 = ConditionalMotionVaeEncoder.fully_connected_layer(encoder_3,
                                                           hidden_num=128,
                                                           name='encoder_4',
                                                           activation=self.encoder_activation)
        encoder_5 = ConditionalMotionVaeEncoder.fully_connected_layer(encoder_4,
                                                           hidden_num=64,
                                                           name='encoder_5',
                                                           activation=self.encoder_activation)
        mean_encoder_6 = ConditionalMotionVaeEncoder.fully_connected_layer(encoder_5,
                                                                hidden_num=self.npc,
                                                                name='mean_encoder_6',
                                                                activation=None)
        log_sigma_encoder_6 = ConditionalMotionVaeEncoder.fully_connected_layer(encoder_5,
                                                                     hidden_num=self.npc,
                                                                     name='log_sigma_encoder_6',
                                                                     activation=None)
        return mean_encoder_6, log_sigma_encoder_6

    def create_conv_model(self, reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope(self.name, reuse):
            self.input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.n_dims, self.n_frames))
            self.latent_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.npc))
            conv_output = self.conv_encode(self.input)
            reshape_conv_output = tf.reshape(conv_output, (-1, conv_output.shape[1] * conv_output.shape[2]))
            z_mu, z_log_sigma = self.encode(reshape_conv_output)
            self.z_op = self.sample(z_mu, z_log_sigma)

            decoder = self.decode(self.z_op)
            reshaped_decoder = tf.reshape(decoder, (-1, conv_output.shape[1], conv_output.shape[2]))
            recon_loss = tf.reduce_mean(input_tensor=tf.pow(reshaped_decoder - conv_output, 2))
            
            

    def sample(self, mu, log_sigma):
        eps = tf.random.normal(tf.shape(input=mu), dtype=tf.float32, mean=0., stddev=1.0,
                               name='epsilon')
        return mu + tf.exp(log_sigma / 2) * eps

    def decode2(self, input):
        decoder_6 = ConditionalMotionVaeEncoder.fully_connected_layer(input,
                                                           hidden_num=64,
                                                           name='decoder_6',
                                                           activation=self.decoder_activation)
        decoder_5 = ConditionalMotionVaeEncoder.fully_connected_layer(decoder_6,
                                                           hidden_num=128,
                                                           name='decoder_5',
                                                           activation=self.decoder_activation)
        decoder_4 = ConditionalMotionVaeEncoder.fully_connected_layer(decoder_5,
                                                           hidden_num=256,
                                                           name='decoder_4',
                                                           activation=self.decoder_activation)
        decoder_3 = ConditionalMotionVaeEncoder.fully_connected_layer(decoder_4,
                                                           hidden_num=512,
                                                           name='decoder_3',
                                                           activation=self.decoder_activation)
        decoder_2 = ConditionalMotionVaeEncoder.fully_connected_layer(decoder_3,
                                                           hidden_num=1024,
                                                           name='decoder_2',
                                                           activation=self.decoder_activation)
        decoder_1 = ConditionalMotionVaeEncoder.fully_connected_layer(decoder_2,
                                                           hidden_num=self.input_dims,
                                                           name='decoder_1',
                                                           activation=self.decoder_activation)
        return decoder_1

    def create_model(self, reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope(self.name, reuse):
            self.input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.n_dims, self.n_frames))
            self.latent_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.npc))
            self.conditional_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.n_dims, self.n_frames))
            ### initialize encoder
            reshaped_input = tf.reshape(self.input, (-1, self.n_dims * self.n_frames))
            z_mu, z_log_sigma = self.encode(reshaped_input)
            ### create new sample
            self.z_op = self.sample(z_mu, z_log_sigma)

            # self.z_op = z_mu
            ### initialize decoder
            # decoder_1 = self.decode_data(self.z_op, reuse)
            decoder_1 = self.decode(self.z_op)
            reshaped_decoder = tf.reshape(decoder_1, (-1, self.n_dims, self.n_frames))
            # mean_reshaped_decoder = tf.reduce_mean(reshaped_decoder, axis=0)
            # variational_error = 10 * -tf.reduce_mean(tf.pow(reshaped_decoder - mean_reshaped_decoder, 2))
            z_op_mean = tf.reduce_mean(input_tensor=self.z_op, axis=0)
            variational_error = 10 * tf.exp(-tf.reduce_mean(input_tensor=tf.pow(self.z_op - z_op_mean, 2)))
            
            ### loss is the sum or reconstruction error and KL distance between normal distribution and mapped distribution
            # reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=decoder_1, labels=reshaped_input), 1)
            reconstruction_loss = tf.reduce_sum(input_tensor=tf.pow(reshaped_decoder - self.input, 2))
            # kl_divergence_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + z_log_sigma - tf.square(mean_encoder_6) - tf.exp(z_log_sigma), 1))
            # reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=decoder_1, labels=self.input), 1)
            
            # style_loss = 1000 * tf.reduce_sum(tf.pow(tf.reduce_mean(gram_matrix(self.conditional_input), axis=0) - tf.reduce_mean(gram_matrix(reshaped_decoder), axis=0), 2))
            style_loss = 500 * tf.reduce_sum(input_tensor=tf.norm(tensor=tf.reduce_mean(input_tensor=gram_matrix(self.conditional_input), axis=0) - tf.reduce_mean(input_tensor=gram_matrix(reshaped_decoder), axis=0)))
            kl_divergence_loss = -0.5 * tf.reduce_sum(input_tensor=1 + z_log_sigma - tf.square(z_mu) - tf.exp(z_log_sigma), axis=1)
            self.mean_op = z_mu
            self.variance_op = z_log_sigma
            # self.cost = tf.reduce_mean(reconstruction_loss + style_loss)
            self.cost = tf.reduce_mean(input_tensor=reconstruction_loss + kl_divergence_loss + style_loss + variational_error)
            # self.cost = tf.reduce_mean(reconstruction_loss)
            # self.cost = tf.reduce_mean(reconstruction_loss + 0.01 * kl_divergence_loss + variational_error)
            self.reconstruction_loss_op = reconstruction_loss
            self.kl_divergence_loss_op = kl_divergence_loss
            self.style_loss_op = style_loss
            self.variational_loss = variational_error
            self.decoder_op = tf.reshape(self.decode(self.latent_input), (-1, self.n_dims, self.n_frames))

        self.model_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        self.saver = tf.compat.v1.train.Saver(self.model_params)
    
    def evaluate_loss(self, input_data, conditional_data):
        reconstruction_error, kl_error, style_error, variational_error = self.sess.run([self.reconstruction_loss_op, 
                                                                                        self.kl_divergence_loss_op, 
                                                                                        self.style_loss_op, 
                                                                                        self.variational_loss], 
                                                                                       feed_dict={self.input: input_data, self.conditional_input: conditional_data})
        return reconstruction_error, kl_error, style_error, variational_error

    def get_params(self):
        return self.model_params

    def get_mean(self, input_data):
        return self.sess.run(self.mean_op, feed_dict={self.input: input_data})

    def get_variance(self, input_data):
        return self.sess.run(self.variance_op, feed_dict={self.input: input_data})

    def __call__(self, input_data):
        latent_value = self.sess.run(self.z_op, feed_dict={self.input: input_data})
        print(latent_value)
        return self.sess.run(self.decoder_op, feed_dict={self.latent_input: latent_value})

    def pre_train(self, training_data):
        n_samples, input_dims = training_data.shape
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            layer1_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, input_dims))
            encoder_layer1_output = ConditionalMotionVaeEncoder.fully_connected_layer(layer1_input,
                                                                                      hidden_num=1024,
                                                                                      name='encoder_1',
                                                                                      activation=self.encoder_activation)
            layer1_output = ConditionalMotionVaeEncoder.fully_connected_layer(encoder_layer1_output,
                                                                              hidden_num=self.input_dims,
                                                                              name='decoder_1',
                                                                              activation=self.decoder_activation)
            layer1_loss = tf.reduce_mean(input_tensor=tf.pow(layer1_input - layer1_output, 2))
            layer1_trainer = self.optimizer.minimize(layer1_loss)
            
            layer2_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 1024))
            encoder_layer2_output = ConditionalMotionVaeEncoder.fully_connected_layer(layer2_input,
                                                                                      hidden_num=512,
                                                                                      name='encoder_2',
                                                                                      activation=self.encoder_activation)
            layer2_output = ConditionalMotionVaeEncoder.fully_connected_layer(encoder_layer2_output,
                                                                              hidden_num=1024,
                                                                              name='decoder_2',
                                                                              activation=self.decoder_activation)
            layer2_loss = tf.reduce_mean(input_tensor=tf.pow(layer2_output - layer2_input, 2))
            layer2_trainer = self.optimizer.minimize(layer2_loss)

            layer3_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 512))
            encoder_layer3_output = ConditionalMotionVaeEncoder.fully_connected_layer(layer3_input,
                                                                                      hidden_num=256,
                                                                                      name='encoder_3',
                                                                                      activation=self.encoder_activation)
            layer3_output = ConditionalMotionVaeEncoder.fully_connected_layer(encoder_layer3_output,
                                                                              hidden_num=512,
                                                                              name='decoder_3',
                                                                              activation=self.decoder_activation)
            layer3_loss = tf.reduce_mean(input_tensor=tf.pow(layer3_output - layer3_input, 2))
            layer3_trainer = self.optimizer.minimize(layer3_loss)

            layer4_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 256))
            encoder_layer4_output = ConditionalMotionVaeEncoder.fully_connected_layer(layer4_input,
                                                                                      hidden_num=128,
                                                                                      name='encoder_4',
                                                                                      activation=self.encoder_activation)
            layer4_output = ConditionalMotionVaeEncoder.fully_connected_layer(encoder_layer4_output,
                                                                              hidden_num=256,
                                                                              name='decoder_4',
                                                                              activation=self.decoder_activation)
            layer4_loss = tf.reduce_mean(input_tensor=tf.pow(layer4_output - layer4_input, 2))
            layer4_trainer = self.optimizer.minimize(layer4_loss)

            layer5_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 128))
            encoder_layer5_output = ConditionalMotionVaeEncoder.fully_connected_layer(layer5_input,
                                                                                      hidden_num=64,
                                                                                      name='encoder_5',
                                                                                      activation=self.encoder_activation)
            layer5_output = ConditionalMotionVaeEncoder.fully_connected_layer(encoder_layer5_output,
                                                                              hidden_num=128,
                                                                              name='decoder_5',
                                                                              activation=self.decoder_activation)
            layer5_loss = tf.reduce_mean(input_tensor=tf.pow(layer5_output - layer5_input, 2))
            layer5_trainer = self.optimizer.minimize(layer5_loss)


    def train(self, training_data, conditional_data, learning_rate=0.01, pre_train=True):
        '''

        :param training_data:
        :param conditional_input:
        :param learning_rate:
        :param pre_train:
        :return:
        '''
        print(training_data.shape)
        print(conditional_data.shape)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = self.optimizer.minimize(self.cost)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        n_samples = training_data.shape[0]
        last_mean = 0

        for epoch in range(self.n_epoches):
            batchinds = np.arange(n_samples // self.batchsize)
            rng.shuffle(batchinds)
            c = []
            c_reconstruction = []
            c_kl = []
            c_style_loss = []
            c_variational_loss = []
            for bii, bi in enumerate(batchinds):
                for i in range(self.n_random_samples):

                    self.sess.run(train_op,
                                  feed_dict={self.input: training_data[bi * self.batchsize: (bi + 1) * self.batchsize],
                                             self.conditional_input: conditional_data})
                    c.append(self.sess.run(self.cost, feed_dict={
                        self.input: training_data[bi * self.batchsize: (bi + 1) * self.batchsize],
                    self.conditional_input: conditional_data}))
                    c_reconstruction.append(self.sess.run(self.reconstruction_loss_op, feed_dict={
                        self.input: training_data[bi * self.batchsize: (bi + 1) * self.batchsize]
                    }))
                    c_kl.append(self.sess.run(self.kl_divergence_loss_op, feed_dict={
                        self.input: training_data[bi * self.batchsize: (bi + 1) * self.batchsize]
                    }))
                    c_style_loss.append(self.sess.run(self.style_loss_op, feed_dict={self.input: training_data[bi * self.batchsize: (bi + 1) * self.batchsize],
                                                                                     self.conditional_input: conditional_data}))
                    c_variational_loss.append(self.sess.run(self.variational_loss, feed_dict={self.input: training_data[bi * self.batchsize: (bi + 1) * self.batchsize],
                                                                                     self.conditional_input: conditional_data}))
                    # z_mean_value, z_sigma_value, z_value = self.sess.run([self.mean_op, self.variance_op, self.z_op],
                    #                                                      feed_dict={self.input: training_data[bi * self.batchsize: (bi + 1) * self.batchsize]})
                    # print('z mean value: ', z_mean_value)
                    # 
                    # print('z_sigma_value: ', z_sigma_value)
                    # 
                    # print('z value: ', z_value)

                if np.isnan(c[-1]): return
                if bii % (int(len(batchinds) / 10) + 1) == 0:
                    sys.stdout.write('\r[Epoch %i]  %0.1f%% mean %.5f' % (epoch, 100 * float(bii) / len(batchinds),
                                                                          np.mean(c)))
                    sys.stdout.flush()
            curr_mean = np.mean(c)
            diff_mean, last_mean = curr_mean - last_mean, curr_mean
            print('\r[Epoch %i] 100.0%% mean %.5f diff %.5f %s' %
                  (epoch, curr_mean, diff_mean, str(datetime.now())[11:19]))
            print()
            print('reconstruction error is: ', np.mean(c_reconstruction))

            print()
            print('KL divergenct error is: ', np.mean(c_kl))
            
            print('style error is: ', np.mean(c_style_loss))
            print()

            print('variational error is: ', np.mean(c_variational_loss))
            print()

    def load_model(self, model_file):
        self.saver.restore(self.sess, model_file)

    def save_model(self, model_file):
        save_path = self.saver.save(self.sess, model_file)
        print("Model saved in file: %s " % save_path)

    def generate_new_samples(self, n_samples):

        new_samples = np.random.randn(n_samples, self.npc) 

        res = self.sess.run(self.decoder_op, feed_dict={self.latent_input: new_samples})
        res = np.reshape(res, (-1, self.n_dims, self.n_frames))
        return res

    def export_z_value(self, n_samples):
        eps = np.random.randn(n_samples, self.npc)
        return
