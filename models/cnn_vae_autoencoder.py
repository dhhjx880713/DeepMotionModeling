# encoding: UTF-8
import tensorflow as tf
import numpy as np
from datetime import datetime
import sys

rng = np.random.RandomState(23456)


class MotionVaeEncoder(object):
    def __init__(self, npc, input_dims, n_epochs, name, batchsize, encoder_activation=None, decoder_activation=None,
                 n_random_samples=1):
        self.npc = npc
        self.input_dims = input_dims
        self.n_epochs = n_epochs
        self.name = name
        self.batchsize = batchsize
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.n_random_samples = n_random_samples
        self.sess = tf.InteractiveSession()

    @staticmethod
    def fully_connected_layer(input, hidden_num, name='fully_connected_layer', activation=None):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            layer_output = tf.layers.dense(input, hidden_num)
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
        encoder_1 = MotionVaeEncoder.fully_connected_layer(input,
                                                           hidden_num=1024,
                                                           name='encoder_1',
                                                           activation=self.encoder_activation)
        encoder_2 = MotionVaeEncoder.fully_connected_layer(encoder_1,
                                                           hidden_num=512,
                                                           name='encoder_2',
                                                           activation=self.encoder_activation)
        encoder_3 = MotionVaeEncoder.fully_connected_layer(encoder_2,
                                                           hidden_num=256,
                                                           name='encoder_3',
                                                           activation=self.encoder_activation)
        mean_encoder_4 = MotionVaeEncoder.fully_connected_layer(encoder_3,
                                                                hidden_num=128,
                                                                name='mean_encoder_4',
                                                                activation=self.encoder_activation)
        log_sigma_encoder_4 = MotionVaeEncoder.fully_connected_layer(encoder_3,
                                                                     hidden_num=128,
                                                                     name='log_sigma_encoder_4',
                                                                     activation=None)
        mean_encoder_5 = MotionVaeEncoder.fully_connected_layer(mean_encoder_4,
                                                                hidden_num=32,
                                                                name='mean_encoder_5',
                                                                activation=self.encoder_activation)
        log_sigma_encoder_5 = MotionVaeEncoder.fully_connected_layer(log_sigma_encoder_4,
                                                                     hidden_num=32,
                                                                     name='log_sigma_encoder_5',
                                                                     activation=None)
        mean_encoder_6 = MotionVaeEncoder.fully_connected_layer(mean_encoder_5,
                                                                hidden_num=self.npc,
                                                                name='mean_encoder_6',
                                                                activation=self.encoder_activation)
        log_sigma_encoder_6 = MotionVaeEncoder.fully_connected_layer(log_sigma_encoder_5,
                                                                     hidden_num=self.npc,
                                                                     name='log_sigma_encoder_6',
                                                                     activation=None)
        return mean_encoder_6, log_sigma_encoder_6

    # def encode_new(self, input):
    #     encoder_layer1_output = MotionVaeEncoder.fully_connected_layer(input,
    #                                                                         hidden_num=1024,
    #                                                                         name='encoder_layer1',
    #                                                                         activation=self.encoder_activation)
    #     encoder_layer2_output = MotionVaeEncoder.fully_connected_layer(encoder_layer1_output,
    #                                                                         hidden_num=512,
    #                                                                         name='encoder_layer2',
    #                                                                         activation=self.encoder_activation)
    #     encoder_layer3_output = MotionVaeEncoder.fully_connected_layer(encoder_layer2_output,
    #                                                                         hidden_num=256,
    #                                                                         name='encoder_layer3',
    #                                                                         activation=self.encoder_activation)
    #     encoder_layer4_output = MotionVaeEncoder.fully_connected_layer(encoder_layer3_output,
    #                                                                         hidden_num=128,
    #                                                                         name='encoder_layer4',
    #                                                                         activation=self.encoder_activation)
    #     encoder_layer5_output = MotionVaeEncoder.fully_connected_layer(encoder_layer4_output,
    #                                                                         hidden_num=32,
    #                                                                         name='encoder_layer5',
    #                                                                         activation=self.encoder_activation)
    #     encoder_layer6_output = MotionVaeEncoder.fully_connected_layer(encoder_layer5_output,
    #                                                                         hidden_num=self.npc,
    #                                                                         name='encoder_layer6',
    #                                                                         activation=self.encoder_activation)
    #     return encoder_layer6_output
    #
    # def decode_new(self, input):
    #     decoder_layer6_output = MotionVaeEncoder.fully_connected_layer(input,
    #                                                                         hidden_num=32,
    #                                                                         name='decoder_layer6',
    #                                                                         activation=self.decoder_activation)
    #     decoder_layer5_output = MotionVaeEncoder.fully_connected_layer(decoder_layer6_output,
    #                                                                         hidden_num=128,
    #                                                                         name='decoder_layer5',
    #                                                                         activation=self.decoder_activation)
    #     decoder_layer4_output = MotionVaeEncoder.fully_connected_layer(decoder_layer5_output,
    #                                                                         hidden_num=256,
    #                                                                         name='decoder_layer4',
    #                                                                         activation=self.decoder_activation)
    #     decoder_layer3_output = MotionVaeEncoder.fully_connected_layer(decoder_layer4_output,
    #                                                                         hidden_num=512,
    #                                                                         name='decoder_layer3',
    #                                                                         activation=self.decoder_activation)
    #     decoder_layer2_output = MotionVaeEncoder.fully_connected_layer(decoder_layer3_output,
    #                                                                         hidden_num=1024,
    #                                                                         name='decoder_layer2',
    #                                                                         activation=self.decoder_activation)
    #     decoder_layer1_output = MotionVaeEncoder.fully_connected_layer(decoder_layer2_output,
    #                                                                         hidden_num=self.input_dims,
    #                                                                         name='decoder_layer1',
    #                                                                         activation=None)
    #     return decoder_layer1_output

    def sample(self, mu, log_sigma):
        eps = tf.random_normal(tf.shape(mu), dtype=tf.float32, mean=0., stddev=1.0,
                               name='epsilon')
        return mu + tf.exp(log_sigma / 2) * eps

    def decode(self, input):
        decoder_6 = MotionVaeEncoder.fully_connected_layer(input,
                                                           hidden_num=32,
                                                           name='decoder_6',
                                                           activation=self.decoder_activation)
        decoder_5 = MotionVaeEncoder.fully_connected_layer(decoder_6,
                                                           hidden_num=128,
                                                           name='decoder_5',
                                                           activation=self.decoder_activation)
        decoder_4 = MotionVaeEncoder.fully_connected_layer(decoder_5,
                                                           hidden_num=256,
                                                           name='decoder_4',
                                                           activation=self.decoder_activation)
        decoder_3 = MotionVaeEncoder.fully_connected_layer(decoder_4,
                                                           hidden_num=512,
                                                           name='decoder_3',
                                                           activation=self.decoder_activation)
        decoder_2 = MotionVaeEncoder.fully_connected_layer(decoder_3,
                                                           hidden_num=1024,
                                                           name='decoder_2',
                                                           activation=self.decoder_activation)
        decoder_1 = MotionVaeEncoder.fully_connected_layer(decoder_2,
                                                           hidden_num=self.input_dims,
                                                           name='decoder_1',
                                                           activation=None)
        return decoder_1

    def create_model(self, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse):
            self.input = tf.placeholder(dtype=tf.float32, shape=(None, self.input_dims))
            self.latent_input = tf.placeholder(dtype=tf.float32, shape=(None, self.npc))
            ### initialize encoder
            z_mu, z_log_sigma = self.encode(self.input)
            ### create new sample
            # self.z_op = self.sample(z_mu, z_log_sigma)
            self.z_op = z_mu
            ### initialize decoder
            # decoder_1 = self.decode_data(self.z_op, reuse)
            decoder_1 = self.decode(self.z_op)

            ### loss is the sum or reconstruction error and KL distance between normal distribution and mapped distribution
            reconstruction_loss = self.input_dims * tf.reduce_mean(tf.pow(decoder_1 - self.input, 2))
            # kl_divergence_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + z_log_sigma - tf.square(mean_encoder_6) - tf.exp(z_log_sigma), 1))
            # reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=decoder_1, labels=self.input), 1)
            kl_divergence_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma - tf.square(z_mu) - tf.exp(z_log_sigma), 1)
            # self.mean_op = z_mu
            # self.variance_op = z_log_sigma

            self.cost = tf.reduce_mean(reconstruction_loss + kl_divergence_loss)
            # self.cost = reconstruction_loss
            self.reconstruction_loss_op = reconstruction_loss
            self.kl_divergence_loss_op = kl_divergence_loss
            self.decoder_op = self.decode(self.latent_input)

            self.model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            self.saver = tf.train.Saver(self.model_params)

    def get_params(self):
        return self.model_params

    # def print_recon_loss_and_kl_loss(self, input_data):
    #     reconstruction_loss = self.sess.run(self.reconstruction_loss_op, feed_dict={self.input: input_data})
    #     kl_divergence_loss = self.sess.run(self.kl_divergence_loss_op, feed_dict={self.input: input_data})
    #     return reconstruction_loss, kl_divergence_loss

    # def get_mean(self, input_data):
    #     return self.sess.run(self.mean_op, feed_dict={self.input: input_data})

    # def get_variance(self, input_data):
    #     return self.sess.run(self.variance_op, feed_dict={self.input: input_data})

    def __call__(self, input_data):
        latent_value = self.sess.run(self.z_op, feed_dict={self.input: input_data})
        return self.sess.run(self.decoder_op, feed_dict={self.latent_input: latent_value})

    def train(self, training_data, learning_rate=0.01, pre_train=True):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = self.optimizer.minimize(self.cost)
        self.sess.run(tf.global_variables_initializer())
        n_samples, input_dims = training_data.shape
        last_mean = 0

        for epoch in range(self.n_epochs):
            batchinds = np.arange(n_samples // self.batchsize)
            rng.shuffle(batchinds)
            c = []
            c_reconstruction = []
            c_kl = []
            for bii, bi in enumerate(batchinds):
                # for i in range(self.n_random_samples):

                self.sess.run(train_op,
                              feed_dict={self.input: training_data[bi * self.batchsize: (bi + 1) * self.batchsize]})
                c.append(self.sess.run(self.cost, feed_dict={
                    self.input: training_data[bi * self.batchsize: (bi + 1) * self.batchsize]}))
                c_reconstruction.append(self.sess.run(self.reconstruction_loss_op, feed_dict={
                    self.input: training_data[bi * self.batchsize: (bi + 1) * self.batchsize],
                }))
                c_kl.append(self.sess.run(self.kl_divergence_loss_op, feed_dict={
                    self.input: training_data[bi * self.batchsize: (bi + 1) * self.batchsize]
                }))
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

    def load_model(self, model_file):
        self.saver.restore(self.sess, model_file)

    def save_model(self, model_file):
        save_path = self.saver.save(self.sess, model_file)
        print("Model saved in file: %s " % save_path)

    def generate_new_samples(self, n_samples):

        new_samples = np.random.randn(n_samples, self.npc)

        res = self.sess.run(self.decoder_op, feed_dict={self.latent_input: new_samples})
        return res

    def export_z_value(self, n_samples):
        eps = np.random.randn(n_samples, self.npc)
        return