import tensorflow as tf
import numpy as np
rng = np.random.RandomState(23456)
from .PFNNParameter import PFNNParameter
import sys
from datetime import datetime
from .HiddenLayer import HiddenLayer
from .BiasLayer import BiasLayer
from .Layer import Layer


MODEL_PARAMS_MAPPING = {
    "pfnn/W0/W:0": 'L000_W',
    "pfnn/W1/W:0": 'L001_W',
    "pfnn/W2/W:0": 'L002_W',
    "pfnn/b0/b:0": 'L003_b',
    "pfnn/b1/b:0": 'L004_b',
    "pfnn/b2/b:0": 'L005_b'
}


class PFNN(Layer):

    def __init__(self, nControlPoints, input_dim, output_dim, dropout_rate, batchsize, name='pfnn', sess=None):

        self.nControlPoints = nControlPoints
        self.input = tf.placeholder(tf.float32, shape=[None, input_dim])
        self.Y = tf.placeholder(tf.float32, shape=[None, output_dim])
        self.phase = self.input[:, -1]
        self.P0 = PFNNParameter((nControlPoints, 512, input_dim - 1), rng, self.phase, 'wb0')
        self.P1 = PFNNParameter((nControlPoints, 512, 512), rng, self.phase, 'wb1')
        self.P2 = PFNNParameter((nControlPoints, output_dim, 512), rng, self.phase, 'wb2')
        self.dropout = dropout_rate
        self.batchsize = batchsize
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        if sess is not None:
            self.sess = sess
        else:
            self.sess = tf.Session()

    def load_params_from_theano(self, database):
        self.load(self.sess, database, mapping_dict=MODEL_PARAMS_MAPPING)

    @staticmethod
    def cubic(y0, y1, y2, y3, mu):
        return (
                (-0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3) * mu * mu * mu +
                (y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3) * mu * mu +
                (-0.5 * y0 + 0.5 * y2) * mu +
                (y1))

    def create_model1(self):

        with tf.variable_scope(self.name):
            self.W0 = HiddenLayer(weights_shape=(self.nControlPoints, 512, self.input_dim-1), name='W0', rng=rng)
            self.W1 = HiddenLayer(weights_shape=(self.nControlPoints, 512, 512), name='W1', rng=rng)
            self.W2 = HiddenLayer(weights_shape=(self.nControlPoints, self.output_dim, 512), name='W2', rng=rng)

            self.b0 = BiasLayer((self.nControlPoints, 512), name='b0')
            self.b1 = BiasLayer((self.nControlPoints, 512), name='b1')
            self.b2 = BiasLayer((self.nControlPoints, self.output_dim), name='b2')

            self.layers = [self.W0, self.W1, self.W2, self.b0, self.b1, self.b2]

            self.params = sum([layer.params for layer in self.layers], [])
            self.saver = tf.train.Saver(self.params)

        pscale = self.nControlPoints * self.input[:, -1]
        pamount = pscale % 1.0
        pindex_1 = tf.cast(pscale, tf.int32) % self.nControlPoints
        pindex_0 = (pindex_1 - 1) % self.nControlPoints
        pindex_2 = (pindex_1 + 1) % self.nControlPoints
        pindex_3 = (pindex_1 + 2) % self.nControlPoints

        Wamount = tf.expand_dims(pamount, -1)
        Wamount = tf.expand_dims(Wamount, -1)

        bamount = tf.expand_dims(pamount, -1)

        W0 = PFNN.cubic(tf.nn.embedding_lookup(self.W0.W, pindex_0),
                        tf.nn.embedding_lookup(self.W0.W, pindex_1),
                        tf.nn.embedding_lookup(self.W0.W, pindex_2),
                        tf.nn.embedding_lookup(self.W0.W, pindex_3), Wamount)
        W1 = PFNN.cubic(tf.nn.embedding_lookup(self.W1.W, pindex_0),
                        tf.nn.embedding_lookup(self.W1.W, pindex_1),
                        tf.nn.embedding_lookup(self.W1.W, pindex_2),
                        tf.nn.embedding_lookup(self.W1.W, pindex_3), Wamount)
        W2 = PFNN.cubic(tf.nn.embedding_lookup(self.W2.W, pindex_0),
                        tf.nn.embedding_lookup(self.W2.W, pindex_1),
                        tf.nn.embedding_lookup(self.W2.W, pindex_2),
                        tf.nn.embedding_lookup(self.W2.W, pindex_3), Wamount)
        b0 = PFNN.cubic(tf.nn.embedding_lookup(self.b0.b, pindex_0),
                        tf.nn.embedding_lookup(self.b0.b, pindex_1),
                        tf.nn.embedding_lookup(self.b0.b, pindex_2),
                        tf.nn.embedding_lookup(self.b0.b, pindex_3), bamount)
        b1 = PFNN.cubic(tf.nn.embedding_lookup(self.b1.b, pindex_0),
                        tf.nn.embedding_lookup(self.b1.b, pindex_1),
                        tf.nn.embedding_lookup(self.b1.b, pindex_2),
                        tf.nn.embedding_lookup(self.b1.b, pindex_3), bamount)
        b2 = PFNN.cubic(tf.nn.embedding_lookup(self.b2.b, pindex_0),
                        tf.nn.embedding_lookup(self.b2.b, pindex_1),
                        tf.nn.embedding_lookup(self.b2.b, pindex_2),
                        tf.nn.embedding_lookup(self.b2.b, pindex_3), bamount)

        H0 = self.input[:, :-1]
        H0 = tf.expand_dims(H0, -1)
        H1 = tf.matmul(W0, tf.nn.dropout(H0, keep_prob=self.dropout)) + tf.expand_dims(b0, -1)
        H1 = tf.nn.elu(H1)
        H2 = tf.matmul(W1, tf.nn.dropout(H1, keep_prob=self.dropout)) + tf.expand_dims(b1, -1)
        H2 = tf.nn.elu(H2)
        H3 = tf.matmul(W2, tf.nn.dropout(H2, keep_prob=self.dropout)) + tf.expand_dims(b2, -1)
        H3 = tf.squeeze(H3, -1)

        self.loss = tf.reduce_mean(tf.square(H3 - self.Y))
        self.output = H3

    def create_model(self):
        H0 = self.input[:, :-1]
        H0 = tf.expand_dims(H0, -1)
        H0 = tf.nn.dropout(H0, keep_prob=self.dropout)

        b0 = tf.expand_dims(self.P0.bias, -1)
        H1 = tf.matmul(self.P0.weight, H0) + b0
        H1 = tf.nn.elu(H1)
        H1 = tf.nn.dropout(H1, keep_prob=self.dropout)

        b1 = tf.expand_dims(self.P1.bias, -1)
        H2 = tf.matmul(self.P1.weight, H1) + b1
        H2 = tf.nn.elu(H2)
        H2 = tf.nn.dropout(H2, keep_prob=self.dropout)

        b2 = tf.expand_dims(self.P2.bias, -1)
        H3 = tf.matmul(self.P2.weight, H2) + b2
        H3 = tf.squeeze(H3, -1)
        self.loss = tf.reduce_mean(tf.square(H3 - self.Y))
        self.output = H3

    def train(self, training_data, output_data, n_epoches, learning_rate=0.0001):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = self.optimizer.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
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
            sys.stdout.write('\r[Epoch %3i] 100.0%% mean %03.5f diff %.5f %s' %
                            (epoch, curr_mean, diff_mean, str(datetime.now())[11:19]))

            sys.stdout.flush()

    def __call__(self, input_data):
        return self.sess.run(self.output, feed_dict={self.input: input_data})

    def load_model(self, model_file):
        self.saver.restore(self.sess, model_file)

    def save_model(self, model_file):
        save_path = self.saver.save(self.sess, model_file)
        print("Model saved in file: %s " % save_path)

    def save_params(self):
        return self.save(self.sess)

    def load_params(self, database):
        self.load(self.sess, database)
