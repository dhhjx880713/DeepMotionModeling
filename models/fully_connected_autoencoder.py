# encoding: UTF-8
import tensorflow as tf
import numpy as np
from datetime import datetime
import sys
from mosi_utils_anim.utilities import write_to_json_file
rng = np.random.RandomState(23456)


class FullyConnectedEncoder(object):

    def __init__(self, npc, input_dim, name, batchsize=32, reuse=tf.compat.v1.AUTO_REUSE, encoder_activation=None, decoder_activation=None,
                 sess=None, logging=False):
        self.npc = npc
        self.input_dim = input_dim
        self.name = name
        self.batchsize = batchsize
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.logging = logging
        self.reuse = reuse
        if self.logging:
            self.log_info = {}  ## collect output information 
        if sess is not None:
            self.sess = sess
        else:
            self.sess = tf.compat.v1.InteractiveSession()

    @staticmethod
    def fully_connected_layer(input, hidden_num, name='fully_connected_layer', activation=None,
                              reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope(name, reuse=reuse):
            layer_output = tf.compat.v1.layers.dense(input, hidden_num, reuse=reuse)
            if activation is None:
                # layer_output = tf.contrib.layers.batch_norm(layer_output,
                #                                             center=True,
                #                                             scale=True,
                #                                             is_training=True,
                #                                             scope='bn')
                return layer_output
            else:
                # layer_output = tf.contrib.layers.batch_norm(layer_output,
                #                                             center=True,
                #                                             scale=True,
                #                                             is_training=True,
                #                                             scope='bn')
                return activation(layer_output)

    # def encode_2layer(self, input):
    #     encoder_layer1_output = FullyConnectedEncoder.fully_connected_layer(input,
    #                                                                         hidden_num=1024,
    #                                                                         name='encoder_layer1',
    #                                                                         activation=self.encoder_activation)
    #     encoder_layer2_output = FullyConnectedEncoder.fully_connected_layer(encoder_layer1_output,
    #                                                                         hidden_num=self.npc,
    #                                                                         name='encoder_layer2',
    #                                                                         activation=self.encoder_activation)                                                                        
    #     return encoder_layer2_output

    def encode(self, input):
        encoder_layer1_output = FullyConnectedEncoder.fully_connected_layer(input,
                                                                            hidden_num=1024,
                                                                            name='encoder_layer1',
                                                                            activation=self.encoder_activation,
                                                                            reuse=self.reuse)
        encoder_layer2_output = FullyConnectedEncoder.fully_connected_layer(encoder_layer1_output,
                                                                            hidden_num=512,
                                                                            name='encoder_layer2',
                                                                            activation=self.encoder_activation,
                                                                            reuse=self.reuse)
        encoder_layer3_output = FullyConnectedEncoder.fully_connected_layer(encoder_layer2_output,
                                                                            hidden_num=256,
                                                                            name='encoder_layer3',
                                                                            activation=self.encoder_activation,
                                                                            reuse=self.reuse)
        encoder_layer4_output = FullyConnectedEncoder.fully_connected_layer(encoder_layer3_output,
                                                                            hidden_num=128,
                                                                            name='encoder_layer4',
                                                                            activation=self.encoder_activation,
                                                                            reuse=self.reuse)
        encoder_layer5_output = FullyConnectedEncoder.fully_connected_layer(encoder_layer4_output,
                                                                            hidden_num=32,
                                                                            name='encoder_layer5',
                                                                            activation=self.encoder_activation,
                                                                            reuse=self.reuse)
        encoder_layer6_output = FullyConnectedEncoder.fully_connected_layer(encoder_layer5_output,
                                                                            hidden_num=self.npc,
                                                                            name='encoder_layer6',
                                                                            activation=self.encoder_activation,
                                                                            reuse=self.reuse)
        return encoder_layer6_output

    # def decode_2layer(self, input):
    #     decoder_layer2_output = FullyConnectedEncoder.fully_connected_layer(input,
    #                                                                         hidden_num=1024,
    #                                                                         name='decoder_layer2',
    #                                                                         activation=self.decoder_activation)
    #     decoder_layer1_output = FullyConnectedEncoder.fully_connected_layer(decoder_layer2_output,
    #                                                                         hidden_num=self.input_dim,
    #                                                                         name='decoder_layer1',
    #                                                                         activation=None) 
    #     return decoder_layer1_output                                                                                                                                             

    def decode(self, input):
        decoder_layer6_output = FullyConnectedEncoder.fully_connected_layer(input,
                                                                            hidden_num=32,
                                                                            name='decoder_layer6',
                                                                            activation=self.decoder_activation,
                                                                            reuse=self.reuse)
        decoder_layer5_output = FullyConnectedEncoder.fully_connected_layer(decoder_layer6_output,
                                                                            hidden_num=128,
                                                                            name='decoder_layer5',
                                                                            activation=self.decoder_activation,
                                                                            reuse=self.reuse)
        decoder_layer4_output = FullyConnectedEncoder.fully_connected_layer(decoder_layer5_output,
                                                                            hidden_num=256,
                                                                            name='decoder_layer4',
                                                                            activation=self.decoder_activation,
                                                                            reuse=self.reuse)
        decoder_layer3_output = FullyConnectedEncoder.fully_connected_layer(decoder_layer4_output,
                                                                            hidden_num=512,
                                                                            name='decoder_layer3',
                                                                            activation=self.decoder_activation,
                                                                            reuse=self.reuse)
        decoder_layer2_output = FullyConnectedEncoder.fully_connected_layer(decoder_layer3_output,
                                                                            hidden_num=1024,
                                                                            name='decoder_layer2',
                                                                            activation=self.decoder_activation,
                                                                            reuse=self.reuse)
        decoder_layer1_output = FullyConnectedEncoder.fully_connected_layer(decoder_layer2_output,
                                                                            hidden_num=self.input_dim,
                                                                            name='decoder_layer1',
                                                                            activation=None,
                                                                            reuse=self.reuse)
        return decoder_layer1_output

    # def create_model_2layer(self, reuse=tf.AUTO_REUSE):
    #     with tf.variable_scope(self.name, reuse):
    #         self.input = tf.placeholder(dtype=tf.float32, shape=(None, self.input_dim))
    #         self.latent_input = tf.placeholder(dtype=tf.float32, shape=(None, self.npc))
    #         self.encode_op = self.encode_2layer(self.input)
    #         decoder_layer1_output = self.decode_2layer(self.encode_op)

    #         self.loss_op = tf.reduce_mean(tf.pow(self.input - decoder_layer1_output, 2))

    #         self.decoder_op = self.decode_2layer(self.latent_input)

    #         self.model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    #         self.saver = tf.train.Saver(self.model_params)


    def create_model(self):
        with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
            self.input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.input_dim))
            self.latent_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.npc))
            self.encode_op = self.encode(self.input)
            decoder_layer1_output = self.decode(self.encode_op)

            self.loss_op = tf.reduce_mean(input_tensor=tf.pow(self.input - decoder_layer1_output, 2))

            self.decoder_op = self.decode(self.latent_input)

            self.model_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            self.saver = tf.compat.v1.train.Saver(self.model_params)

    def get_params(self):
        return self.model_params

    def __call__(self, input_data):
        # return self.sess.run(self.decoder_op, feed_dict={self.input: input_data})
        latent_value = self.sess.run(self.encode_op, feed_dict={self.input: input_data})
        return self.sess.run(self.decoder_op, feed_dict={self.latent_input: latent_value})

    def pre_train(self, training_data, n_epochs):
        n_samples, input_dims = training_data.shape
        if self.logging:
            self.log_info["pre_training"] = {} 
        with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
            layer1_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, input_dims))
            encoder_layer1_output = FullyConnectedEncoder.fully_connected_layer(layer1_input,
                                                                                hidden_num=1024,
                                                                                name='encoder_layer1',
                                                                                activation=self.encoder_activation,
                                                                                reuse=self.reuse)
            layer1_output = FullyConnectedEncoder.fully_connected_layer(encoder_layer1_output,
                                                                        hidden_num=input_dims,
                                                                        name='decoder_layer1',
                                                                        activation=None,
                                                                        reuse=self.reuse)
            layer1_loss = tf.reduce_mean(input_tensor=tf.pow(layer1_input - layer1_output, 2))
            layer1_trainer = self.optimizer.minimize(layer1_loss)

            layer2_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 1024))
            encoder_layer2_output = FullyConnectedEncoder.fully_connected_layer(layer2_input,
                                                                                hidden_num=512,
                                                                                name='encoder_layer2',
                                                                                activation=self.encoder_activation,
                                                                                reuse=self.reuse)
            layer2_output = FullyConnectedEncoder.fully_connected_layer(encoder_layer2_output,
                                                                        hidden_num=1024,
                                                                        name='decoder_layer2',
                                                                        activation=self.decoder_activation,
                                                                        reuse=self.reuse)
            layer2_loss = tf.reduce_mean(input_tensor=tf.pow(layer2_input - layer2_output, 2))
            layer2_trainer = self.optimizer.minimize(layer2_loss)

            layer3_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 512))
            encoder_layer3_output = FullyConnectedEncoder.fully_connected_layer(layer3_input,
                                                                                hidden_num=256,
                                                                                name='encoder_layer3',
                                                                                activation=self.encoder_activation,
                                                                                reuse=self.reuse)
            layer3_output = FullyConnectedEncoder.fully_connected_layer(encoder_layer3_output,
                                                                        hidden_num=512,
                                                                        name='decoder_layer3',
                                                                        activation=self.decoder_activation,
                                                                        reuse=self.reuse)
            layer3_loss = tf.reduce_mean(input_tensor=tf.pow(layer3_input - layer3_output, 2))
            layer3_trainer = self.optimizer.minimize(layer3_loss)

            layer4_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 256))
            encoder_layer4_output = FullyConnectedEncoder.fully_connected_layer(layer4_input,
                                                                                hidden_num=128,
                                                                                name='encoder_layer4',
                                                                                activation=self.encoder_activation,
                                                                                reuse=self.reuse)
            layer4_output = FullyConnectedEncoder.fully_connected_layer(encoder_layer4_output,
                                                                        hidden_num=256,
                                                                        name='decoder_layer4',
                                                                        activation=self.decoder_activation,
                                                                        reuse=self.reuse)
            layer4_loss = tf.reduce_mean(input_tensor=tf.pow(layer4_input - layer4_output, 2))
            layer4_trainer = self.optimizer.minimize(layer4_loss)

            layer5_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 128))
            encoder_layer5_output = FullyConnectedEncoder.fully_connected_layer(layer5_input,
                                                                                hidden_num=32,
                                                                                name='encoder_layer5',
                                                                                activation=self.encoder_activation,
                                                                                reuse=self.reuse)
            layer5_output = FullyConnectedEncoder.fully_connected_layer(encoder_layer5_output,
                                                                        hidden_num=128,
                                                                        name='decoder_layer5',
                                                                        activation=self.decoder_activation,
                                                                        reuse=self.reuse)
            layer5_loss = tf.reduce_mean(input_tensor=tf.pow(layer5_input - layer5_output, 2))
            layer5_trainer = self.optimizer.minimize(layer5_loss)

            layer6_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 32))
            encoder_layer6_output = FullyConnectedEncoder.fully_connected_layer(layer6_input,
                                                                                hidden_num=self.npc,
                                                                                name='encoder_layer6',
                                                                                activation=self.encoder_activation,
                                                                                reuse=self.reuse)
            layer6_output = FullyConnectedEncoder.fully_connected_layer(encoder_layer6_output,
                                                                        hidden_num=32,
                                                                        name='decoder_layer6',
                                                                        activation=self.decoder_activation,
                                                                        reuse=self.reuse)
            layer6_loss = tf.reduce_mean(input_tensor=tf.pow(layer6_input - layer6_output, 2))
            layer6_trainer = self.optimizer.minimize(layer6_loss)
            # 
            # layer7_input = tf.placeholder(dtype=tf.float32, shape=(None, 32))
            # encoder_layer7_output = FullyConnectedEncoder.fully_connected_layer(layer7_input,
            #                                                                     hidden_num=self.npc,
            #                                                                     name='encoder_layer7',
            #                                                                     activation=self.encoder_activation)
            # layer7_output = FullyConnectedEncoder.fully_connected_layer(encoder_layer7_output,
            #                                                             hidden_num=32,
            #                                                             name='decoder_layer7',
            #                                                             activation=self.decoder_activation)
            # layer7_loss = tf.reduce_mean(tf.pow(layer7_input - layer7_output, 2))
            # layer7_trainer = self.optimizer.minimize(layer7_loss)


        ### initilize parameters
        self.sess.run(tf.compat.v1.global_variables_initializer())
        print('pre-train layer1')
        for epoch in range(n_epochs):
            self.sess.run(layer1_trainer, feed_dict={layer1_input: training_data})
            err = self.sess.run(layer1_loss, feed_dict={layer1_input: training_data})
            sys.stdout.write('\r[Epoch %i] error %.5f' % (epoch, err))
        print("\n")
        if self.logging:
            self.log_info["pre_training"]["layer1"] = float(err)
        layer1_res = self.sess.run(encoder_layer1_output, feed_dict={layer1_input: training_data})
        print('pre-train layer2')
        for epoch in range(n_epochs):
            self.sess.run(layer2_trainer, feed_dict={layer2_input: layer1_res})
            err = self.sess.run(layer2_loss, feed_dict={layer2_input: layer1_res})
            sys.stdout.write('\r[Epoch %i] error %.5f' % (epoch, err))
        print("\n")
        if self.logging:
            self.log_info["pre_training"]["layer2"] = float(err)
        layer2_res = self.sess.run(encoder_layer2_output, feed_dict={layer2_input: layer1_res})
        print('pre-train layer3')
        for epoch in range(n_epochs):
            self.sess.run(layer3_trainer, feed_dict={layer3_input: layer2_res})
            err = self.sess.run(layer3_loss, feed_dict={layer3_input: layer2_res})
            sys.stdout.write('\r[Epoch %i] error %.5f' % (epoch, err))
        print("\n")
        if self.logging:
            self.log_info["pre_training"]["layer3"] = float(err)            
        layer3_res = self.sess.run(encoder_layer3_output, feed_dict={layer3_input: layer2_res})
        print('pre-train layer4')
        for epoch in range(n_epochs):
            self.sess.run(layer4_trainer, feed_dict={layer4_input: layer3_res})
            err = self.sess.run(layer4_loss, feed_dict={layer4_input: layer3_res})
            sys.stdout.write('\r[Epoch %i] error %.5f' % (epoch, err))
        print("\n")
        if self.logging:
            self.log_info["pre_training"]["layer4"] = float(err)        
        layer4_res = self.sess.run(encoder_layer4_output, feed_dict={layer4_input: layer3_res})
        print('pre-train layer5')
        for epoch in range(n_epochs):
            self.sess.run(layer5_trainer, feed_dict={layer5_input: layer4_res})
            err = self.sess.run(layer5_loss, feed_dict={layer5_input: layer4_res})
            sys.stdout.write('\r[Epoch %i] error %.5f' % (epoch, err))
        print("\n")
        if self.logging:
            self.log_info["pre_training"]["layer5"] = float(err)            
        layer5_res = self.sess.run(encoder_layer5_output, feed_dict={layer5_input: layer4_res})
        print('pre-train layer6')
        for epoch in range(n_epochs):
            self.sess.run(layer6_trainer, feed_dict={layer6_input: layer5_res})
            err = self.sess.run(layer6_loss, feed_dict={layer6_input: layer5_res})
            sys.stdout.write('\r[Epoch %i] error %.5f' % (epoch, err))
        print("\n")
        if self.logging:
            self.log_info["pre_training"]["layer6"] = float(err)            
        # layer6_res = self.sess.run(encoder_layer6_output, feed_dict={layer6_input: layer5_res})
        # print('pre-train layer7')
        # for epoch in range(self.n_epochs):
        #     self.sess.run(layer7_trainer, feed_dict={layer7_input: layer6_res})
        #     err = self.sess.run(layer7_loss, feed_dict={layer7_input: layer6_res})
        #     sys.stdout.write('\r[Epoch %i] error %.5f' % (epoch, err))

    def train(self, training_data, n_epochs, learning_rate=0.01, pre_train=False):
        with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = self.optimizer.minimize(self.loss_op)
            if pre_train:
                self.pre_train(training_data, n_epochs)
            else:

                self.sess.run(tf.compat.v1.global_variables_initializer())

            n_samples, input_dims = training_data.shape
            last_mean = 0
            for epoch in range(n_epochs):
                batchinds = np.arange(n_samples // self.batchsize)
                rng.shuffle(batchinds)
                c = []
                for bii, bi in enumerate(batchinds):

                    self.sess.run(train_op, feed_dict={self.input: training_data[bi*self.batchsize: (bi+1)*self.batchsize]})
                    c.append(self.sess.run(self.loss_op, feed_dict={self.input: training_data[bi*self.batchsize: (bi+1)*self.batchsize]}))
                    if np.isnan(c[-1]): 
                        print("loss value is non!\n")
                        return
                    if bii % (int(len(batchinds) / 10) + 1) == 0:
                        sys.stdout.write('\r[Epoch %i]  %0.1f%% mean %.5f' % (epoch, 100 * float(bii) / len(batchinds),
                                                                            np.mean(c)))
                        sys.stdout.flush()
                curr_mean = np.mean(c)
                diff_mean, last_mean = curr_mean-last_mean, curr_mean
                print('\r[Epoch %i] 100.0%% mean %.5f diff %.5f %s' %
                    (epoch, curr_mean, diff_mean, str(datetime.now())[11:19]))
                # err = sess.run(self.loss_op, feed_dict={self.input: training_data})
                # print('\r[Epoch %i]  error %.5f' % (epoch, err))
                # if epoch % 50 == 0:
                #     err = sess.run(self.loss_op, feed_dict={self.input: training_data})
                #     print('\r[Epoch %i]  error %.5f' % (epoch, err))
            
            if self.logging:
                self.log_info["training"] = {"final_error": float(curr_mean)}


    def load_model(self, model_file):
        self.saver.restore(self.sess, model_file)

    def save_model(self, model_file):
        save_path = self.saver.save(self.sess, model_file)
        print("Model saved in file: %s " % save_path)
        if self.logging:
            write_to_json_file(model_file + '.log', self.log_info)