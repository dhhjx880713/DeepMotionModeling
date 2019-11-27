import tensorflow as tf 
import numpy as np 
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute())+ r'/..')
from nn.network import Network
from datetime import datetime
tf.compat.v1.disable_eager_execution()


class VAE(Network):

    def __init__(self, name, npc, sess=None):
        super().__init__(name=name, sess=sess)
        self.npc = npc

    def encode(self, input, train=True, reuse=tf.compat.v1.AUTO_REUSE):
        encode_1 = tf.compat.v1.layers.dense(input, 32, activation=tf.nn.elu, name='encode1', reuse=reuse)
        if train:
            encode_1 = tf.compat.v1.layers.dropout(encode_1, rate=0.5)
        encode_2 = tf.compat.v1.layers.dense(encode_1, 16, activation=tf.nn.elu, name='encode2', reuse=reuse)
        if train:
            encode_2 = tf.compat.v1.layers.dropout(encode_2, rate=0.5)
        mu = tf.compat.v1.layers.dense(encode_2, 1, name='mean', reuse=reuse)
        if train:
            mu = tf.compat.v1.layers.dropout(mu, rate=0.5)
        log_sigma = tf.compat.v1.layers.dense(encode_2, self.npc, name='variance', reuse=reuse)
        if train:
            log_sigma = tf.compat.v1.layers.dropout(log_sigma, rate=0.5)
        return mu, log_sigma

    def sample(self, mu, sigma):
        eps = tf.random.normal(tf.shape(input=mu), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
        z = mu + tf.exp(sigma / 2) * eps   
        return z

    def decode(self, input, output_shape, train=True, reuse=tf.compat.v1.AUTO_REUSE):
        decode_2 = tf.compat.v1.layers.dense(input, 16, activation=tf.nn.elu, name='decode2', reuse=reuse)
        if train:
            decode_2 = tf.compat.v1.layers.dropout(decode_2, rate=0.5)     
        decode_1 = tf.compat.v1.layers.dense(decode_2, 32, activation=tf.nn.elu, name='decode1', reuse=reuse)
        if train:
            decode_1 = tf.compat.v1.layers.dropout(decode_1, rate=0.5)
        output = tf.compat.v1.layers.dense(decode_1, output_shape, name='output', reuse=reuse)
        return output

    
    def build(self, input_shape, train=True, reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope(self.name, reuse):
            self.input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, input_shape))
            self.latent_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.npc))
            z_mu, z_log_sigma = self.encode(self.input, train=train)
            self.z_op = self.sample(z_mu, z_log_sigma)
            decoder_res = self.decode(self.z_op, input_shape, train)
            reconstruction_loss = tf.reduce_mean(input_tensor=tf.pow(decoder_res - self.input, 2)) 
            # kl_divergence_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma - tf.square(z_mu) - tf.exp(z_log_sigma), 1)
            kl_divergence_loss = -0.5 * tf.reduce_mean(input_tensor=1 + z_log_sigma - tf.square(z_mu) - tf.exp(z_log_sigma))

            # self.cost = reconstruction_loss + kl_divergence_loss
            self.cost = reconstruction_loss
            # self.output_op = self.decode(self.latent_input, input_shape, train=False)
            self.output_op = self.decode(self.z_op, input_shape, train)
            self.random_output_op = self.decode(self.latent_input, input_shape, train=False)
            self.params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            self.saver = tf.compat.v1.train.Saver(self.params)


    # def build(self, input_shape, reuse=tf.AUTO_REUSE):
    #     with tf.variable_scope(self.name, reuse):
    #         self.input = tf.placeholder(dtype=tf.float32, shape=(None, input_shape))
    #         h1_layer = tf.layers.dense(self.input, 32, activation=tf.nn.elu, name='layer1', reuse=reuse)
    #         h1_layer = tf.layers.dropout(h1_layer, rate=0.5)
    #         h2_layer = tf.layers.dense(h1_layer, 16, activation=tf.nn.elu, name='layer2', reuse=reuse)
    #         h2_layer = tf.layers.dropout(h2_layer, rate=0.5)
    #         mu = tf.layers.dense(h2_layer, 1, activation=tf.nn.elu, name='mean_layer', reuse=reuse)
    #         mu = tf.layers.dropout(mu, rate=0.5)
    #         log_sigma = tf.layers.dense(h2_layer, 1, activation=tf.nn.elu, name='variance_layer', reuse=reuse) 
    #         log_sigma = tf.layers.dropout(log_sigma, rate=0.5)
    #         eps = tf.random_normal(tf.shape(mu), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
    #         z = mu + tf.exp(log_sigma / 2) * eps
    #         h3_layer = tf.layers.dense(z, 16, activation=tf.nn.elu, name='layer3', reuse=reuse)
    #         h3_layer = tf.layers.dropout(h3_layer, rate=0.5)
    #         h4_layer = tf.layers.dense(h3_layer, 32, activation=tf.nn.elu, name='layer4', reuse=reuse)
    #         h4_layer = tf.layers.dropout(h4_layer, rate=0.5)
    #         self.output_op = tf.layers.dense(h4_layer, input_shape, name='output_layer', reuse=reuse)           
    #         self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    #         self.saver = tf.train.Saver(self.params)
    #         reconstruction_loss = tf.reduce_mean(tf.pow(self.output_op - self.input, 2))
    #         kl_divergence_loss = -0.5 * tf.reduce_sum(1 + log_sigma - tf.square(mu) - tf.exp(log_sigma), 1)
    #         self.cost = reconstruction_loss + kl_divergence_loss

    def generate_samples(self, X_values):
        # mu, sigma = self.sess.run(self.encode, )
        # new_samples = np.random.randn(n_samples, self.npc)
        # return self.sess.run(self.output_op, feed_dict={self.latent_input: new_samples})
        return self.sess.run(self.output_op, feed_dict={self.input: X_values})

    def generate_random_samples(self, n_samples):
        new_samples = np.random.randn(n_samples, self.npc)
        return self.sess.run(self.random_output_op, feed_dict={self.latent_input: new_samples})

    def train(self, input_data, n_random_samples, epochs, learning_rate, batchsize, rng=np.random.RandomState(123456)):
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = self.optimizer.minimize(self.cost)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        n_samples = len(input_data)
        last_mean = 0
        for epoch in range(epochs):
            batchinds = np.arange(n_samples // batchsize)
            c = []
            for bii, bi in enumerate(batchinds):
                for i in range(n_random_samples):
                    self.sess.run(train_op,
                                feed_dict={self.input: input_data[bi * batchsize: (bi + 1) * batchsize]})
                    c.append(self.sess.run(self.cost, feed_dict={
                        self.input: input_data[bi * batchsize: (bi + 1) * batchsize]}))
                if np.isnan(c[-1]): return
                if bii % (int(len(batchinds) / 1000) + 1) == 0:
                    sys.stdout.write('\r[Epoch %i]  %0.1f%% mean %.5f' % (epoch, 100 * float(bii) / len(batchinds),
                                                                        np.mean(c)))
                    sys.stdout.flush()
            curr_mean = np.mean(c)
            diff_mean, last_mean = curr_mean - last_mean, curr_mean
            print('\r[Epoch %i] 100.0%% mean %.5f diff %.5f %s' %
                  (epoch, curr_mean, diff_mean, str(datetime.now())[11:19]))