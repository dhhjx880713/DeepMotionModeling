import tensorflow as tf
import numpy as np
rng = np.random.RandomState(23456)
from .PFNNParameter import PFNNParameter
import sys
from datetime import datetime
from .Layer import Layer
from .pfnn import PFNN


class ResPFNN(PFNN):
    """add a residual block in the middle layer to learn style
    
    Arguments:
        Layer {[type]} -- [description]
    """

    def __init__(self, nControlPoints, input_dim, output_dim, style_dim, dropout_rate, batchsize, name='ResPFNN', sess=None):
        super(ResPFNN, self).__init__(nControlPoints, input_dim, output_dim, dropout_rate, batchsize, name, sess)
        self.style_dim = style_dim
        self.hidden_dim = 512

        # self.style_W0 = tf.Variable(self.initial_style_weight((self.style_dim, self.hidden_dim), rng), name="s_W0")
        
        # self.style_W1 = tf.Variable(self.initial_style_weight((self.hidden_dim, self.style_dim), rng), name="s_W1")

        # self.style_b1 = tf.Variable(self.initial_style_bias(self.hidden_dim,), name="s_b1")

        self.style_W0 = tf.compat.v1.get_variable(name="s_W0", shape=(self.style_dim, self.hidden_dim), initializer=tf.compat.v1.zeros_initializer())
        self.style_W1 = tf.compat.v1.get_variable(name="s_W1", shape=(self.hidden_dim, self.style_dim), initializer=tf.compat.v1.zeros_initializer())
        self.style_b1 = tf.compat.v1.get_variable(name="s_b1", shape=(self.hidden_dim,), initializer=tf.compat.v1.zeros_initializer())
        self.style_diagonal = tf.compat.v1.get_variable(name="s_diag", shape=(self.style_dim), initializer=tf.compat.v1.ones_initializer())
        # self.style_diagonal = tf.Variable(np.ones(self.style_dim), name="s_diag", dtype=tf.float32)
        # self.style_diagonal = tf.get_variable(name="s_diag", shape=(self.style_dim, ), initializer=tf.ones_initializer())
        # self.style_diagonal = tf.Variable(np.random.rand(self.style_dim,), name="s_diag", dtype=tf.float32)

        # self.t0 = tf.Variable(self.initial_style_weight((input_dim, 10), rng), name="t0")
        # self.t1 = tf.Variable(np.random.rand(10), name="t1", dtype=tf.float32)
        # self.t2 = tf.Variable(self.initial_style_weight((10, output_dim), rng), name='t2')
        # self.tb0 = tf.Variable(self.initial_style_bias(10,), name="tb0")
        # self.tb1 = tf.Variable(self.initial_style_bias(10,), name="tb1")
        # self.tb2 = tf.Variable(self.initial_style_bias(output_dim,), name="tb2")

    def initial_style_weight(self, shape, rng):
        alpha_bound = np.sqrt(6. / self.style_dim * self.hidden_dim)
        alpha = np.asarray(
            rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape), dtype=np.float32)

        return tf.convert_to_tensor(value=alpha, dtype = tf.float32)        

    def initial_style_bias(self, shape):
        return tf.zeros(shape, dtype=tf.float32)
    

    # def create_test_model(self):
    #     H0 = tf.nn.dropout(self.input, keep_prob=self.dropout)
    #     H0 = tf.expand_dims(H0, -1)
    #     # H1 = tf.matmul(H0, self.t0) + self.tb0
    #     H1 = tf.matmul(tf.transpose(self.t0), H0) + self.tb0
    #     H1 = tf.nn.elu(H1)
    #     H1 = tf.nn.dropout(H1, keep_prob=self.dropout)

    #     # H2 = tf.matmul(H1, tf.diag(self.t1)) + self.tb1
    #     H2 = tf.matmul(tf.diag(self.t1), H1) + self.tb1
    #     H2 = tf.nn.elu(H2)
    #     H2 = tf.nn.dropout(H2, keep_prob=self.dropout)

    #     # H3 = tf.matmul(H2, self.t2) + self.tb2
    #     H3 = tf.matmul(tf.transpose(self.t2), H2) + self.tb2
    #     H3 = tf.squeeze(H3, -1)
    #     self.loss = tf.reduce_mean(tf.pow(H3 - self.Y, 2))
    #     self.output = H3
    #     self.train_params = [self.t0, self.tb0, self.t1, self.tb1, self.t2, self.tb2]

    # def train_model(self, training_data, output_data, n_epoches, learning_rate=0.0001):
    #     self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #     train_op = self.optimizer.minimize(self.loss, var_list=self.train_params)
    #     self.sess.run(tf.variables_initializer(self.optimizer.variables() + self.train_params))
    #     n_samples, n_features = training_data.shape
    #     last_mean = 0
    #     for epoch in range(n_epoches):
    #         batchinds = np.arange(n_samples // self.batchsize)
    #         rng.shuffle(batchinds)
    #         c = []
    #         for bii, bi in enumerate(batchinds):

    #             self.sess.run(train_op,
    #                           feed_dict={self.input: training_data[bi * self.batchsize: (bi + 1) * self.batchsize],
    #                                      self.Y: output_data[bi * self.batchsize: (bi + 1) * self.batchsize]})
    #             c.append(self.sess.run(self.loss, feed_dict={
    #                 self.input: training_data[bi * self.batchsize: (bi + 1) * self.batchsize],
    #                 self.Y: output_data[bi * self.batchsize: (bi + 1) * self.batchsize]}))
    #             if np.isnan(c[-1]):
    #                 raise ValueError
    #                 return
    #             if bii % (int(len(batchinds) / 1000) + 1) == 0:
    #                 sys.stdout.write('\r[Epoch %3i]  % 3.1f%% mean %03.5f' % (epoch, 100 * float(bii) / len(batchinds),
    #                                                                           np.mean(c)))
                                                                       
    #                 sys.stdout.flush()
    #         curr_mean = np.mean(c)
    #         diff_mean, last_mean = curr_mean - last_mean, curr_mean
    #         print('\r[Epoch %i] 100.0%% mean %.5f diff %.5f %s' %
    #               (epoch, curr_mean, diff_mean, str(datetime.now())[11:19]))
    #         print("diagonal parameters: ", self.sess.run(self.t1))      

    def create_model_diagonal(self):

        H0 = self.input[:, :-1]
        H0 = tf.expand_dims(H0, -1)
        H0 = tf.nn.dropout(H0, rate=1 - (self.dropout))
        b0 = tf.expand_dims(self.P0.bias, -1)

        H1 = tf.matmul(self.P0.weight, H0) + b0
        H1 = tf.nn.elu(H1)
        H1 = tf.nn.dropout(H1, rate=1 - (self.dropout))
        # print('H1 shape: ', H1.shape)
        # t1 = tf.matmul(self.style_W0, tf.squeeze(H1, -1))
        t1 = tf.matmul(tf.squeeze(H1, -1), tf.transpose(a=self.style_W0))
        t2 = tf.matmul(t1, tf.linalg.tensor_diag(self.style_diagonal))

        # print("style_w1 shape: ", self.style_W1.shape)
        # style_residual = tf.matmul(self.style_W1, t2) + self.style_b1
        style_residual = tf.matmul(t2, tf.transpose(a=self.style_W1)) + self.style_b1

        b1 = tf.expand_dims(self.P1.bias, -1)

        H2 = tf.matmul(self.P1.weight, H1) + b1 + tf.expand_dims(style_residual, -1)
        #H2 = tf.matmul(self.P1.weight, H1) + b1
        H2 = tf.nn.elu(H2)
        #H2 = tf.expand_dims(style_residual, -1)
        H2 = tf.nn.dropout(H2, rate=1 - (self.dropout))

        b2 = tf.expand_dims(self.P2.bias, -1)
        H3 = tf.matmul(self.P2.weight, H2) + b2
        H3 = tf.squeeze(H3, -1)
        self.loss = tf.reduce_mean(input_tensor=tf.pow(H3 - self.Y, 2))
        self.output = H3
        self.params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        self.saver = tf.compat.v1.train.Saver(self.params)        
        self.style_params = [self.style_W0, self.style_W1, self.style_b1, self.style_diagonal]     

    def create_model(self):

        H0 = self.input[:, :-1]
        H0 = tf.expand_dims(H0, -1)
        H0 = tf.nn.dropout(H0, rate=1 - (self.dropout))
        b0 = tf.expand_dims(self.P0.bias, -1)

        H1 = tf.matmul(self.P0.weight, H0) + b0
        H1 = tf.nn.elu(H1)
        H1 = tf.nn.dropout(H1, rate=1 - (self.dropout))


        style_residual = tf.matmul(tf.matmul(tf.squeeze(H1, -1), tf.transpose(a=self.style_W0)), tf.transpose(a=self.style_W1)) + self.style_b1

        b1 = tf.expand_dims(self.P1.bias, -1)
        H2 = tf.matmul(self.P1.weight, H1) + b1 + tf.expand_dims(style_residual, -1)
        H2 = tf.nn.elu(H2)
        H2 = tf.nn.dropout(H2, rate=1 - (self.dropout))

        b2 = tf.expand_dims(self.P2.bias, -1)
        H3 = tf.matmul(self.P2.weight, H2) + b2
        H3 = tf.squeeze(H3, -1)
        self.loss = tf.reduce_mean(input_tensor=tf.square(H3 - self.Y))
        self.output = H3
        self.params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        self.saver = tf.compat.v1.train.Saver(self.params)        
        self.style_params = [self.style_W0, self.style_W1, self.style_b1]
    
    def style_fine_turning(self, training_data, output_data, n_epoches, learning_rate=0.0001):
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = self.optimizer.minimize(self.loss, var_list=self.style_params)
        self.sess.run(tf.compat.v1.variables_initializer(self.optimizer.variables() + self.style_params))
        n_samples, n_features = training_data.shape
        last_mean = 0
        for epoch in range(n_epoches):
            batchinds = np.arange(n_samples // self.batchsize)
            rng.shuffle(batchinds)
            c = []
            for bii, bi in enumerate(batchinds):

                self.sess.run(train_op,
                              feed_dict={self.input: training_data[bi * self.batchsize: (bi + 1) * self.batchsize],
                                         self.Y: output_data[bi * self.batchsize: (bi + 1) * self.batchsize]})
                c.append(self.sess.run(self.loss, feed_dict={
                    self.input: training_data[bi * self.batchsize: (bi + 1) * self.batchsize],
                    self.Y: output_data[bi * self.batchsize: (bi + 1) * self.batchsize]}))
                if np.isnan(c[-1]):
                    raise ValueError
                    return
                if bii % (int(len(batchinds) / 1000) + 1) == 0:
                    sys.stdout.write('\r[Epoch %3i]  % 3.1f%% mean %03.5f' % (epoch, 100 * float(bii) / len(batchinds),
                                                                              np.mean(c)))
                                                                       
                    sys.stdout.flush()
            curr_mean = np.mean(c)
            diff_mean, last_mean = curr_mean - last_mean, curr_mean
            print('\r[Epoch %i] 100.0%% mean %.5f diff %.5f %s' %
                  (epoch, curr_mean, diff_mean, str(datetime.now())[11:19]))
            # print("bias vaule: ", self.sess.run(self.style_b1)[:10]) 
            # print("style diagonal: ", self.sess.run(self.style_diagonal))

    def save_style_params(self, savepath):
        style_W0 = self.sess.run(self.style_W0)
        style_W1 = self.sess.run(self.style_W1)
        style_b1 = self.sess.run(self.style_b1)
        style_W0.astype(np.float32).tofile(savepath+'/style_W0.bin')
        style_W1.astype(np.float32).tofile(savepath+'/style_W1.bin')
        style_b1.astype(np.float32).tofile(savepath+'/style_b1.bin')

    def save_style_diagonal_params(self, savepath):
        style_W0 = self.sess.run(self.style_W0)
        style_W1 = self.sess.run(self.style_W1)
        style_b1 = self.sess.run(self.style_b1)
        style_diagonal = self.sess.run(self.style_diagonal)
        style_W0.astype(np.float32).tofile(savepath+'/style_W0.bin')
        style_W1.astype(np.float32).tofile(savepath+'/style_W1.bin')
        style_b1.astype(np.float32).tofile(savepath+'/style_b1.bin')   
        style_diagonal.astype(np.float32).tofile(savepath+'/style_diagonal.bin')     

    def load_style_params(self, savepath):
        style_W0_value = np.fromfile(savepath + '/style_W0.bin', dtype=np.float32)
        style_W1_value = np.fromfile(savepath + '/style_W1.bin', dtype=np.float32)
        style_b1_value = np.fromfile(savepath + '/style_b1.bin', dtype=np.float32)
        style_W0_value = np.reshape(style_W0_value, (self.style_dim, self.hidden_dim))
        style_W1_value = np.reshape(style_W1_value, (self.hidden_dim, self.style_dim))
        self.sess.run(self.style_W0.assign(style_W0_value))
        self.sess.run(self.style_W1.assign(style_W1_value))
        self.sess.run(self.style_b1.assign(style_b1_value))