import tensorflow as tf
import numpy as np
rng = np.random.RandomState(23456)
from .PFNNParameter import PFNNParameter
import sys
from datetime import datetime
from .Layer import Layer


# MODEL_PARAMS_MAPPING = {
#     "pfnn/W0/W:0": 'L000_W',
#     "pfnn/W1/W:0": 'L001_W',
#     "pfnn/W2/W:0": 'L002_W',
#     "pfnn/b0/b:0": 'L003_b',
#     "pfnn/b1/b:0": 'L004_b',
#     "pfnn/b2/b:0": 'L005_b'
# }
MODEL_PARAMS_MAPPING = {
    "wb0alpha:0": 'L000_W',
    "wb1alpha:0": 'L001_W',
    "wb2alpha:0": 'L002_W',
    "wb0beta:0": 'L003_b',
    "wb1beta:0": 'L004_b',
    "wb2beta:0": 'L005_b'
}


class PFNN(Layer):

    def __init__(self, nControlPoints, input_dim, output_dim, dropout_rate, batchsize, name='pfnn', sess=None):
        super(PFNN, self).__init__()
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
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.saver = tf.train.Saver(self.params)

    def train(self, training_data, output_data, n_epoches, learning_rate=0.0001, fine_turning=False):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = self.optimizer.minimize(self.loss)
        if not fine_turning:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess.run(tf.variables_initializer(self.optimizer.variables()))
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
            # sys.stdout.write('\r[Epoch %3i] 100.0%% mean %03.5f diff %.5f %s' %
            #                 (epoch, curr_mean, diff_mean, str(datetime.now())[11:19]))
            print('\r[Epoch %i] 100.0%% mean %.5f diff %.5f %s' %
                  (epoch, curr_mean, diff_mean, str(datetime.now())[11:19]))
            sys.stdout.flush()


    def __call__(self, input_data):
        return self.sess.run(self.output, feed_dict={self.input: input_data})

    def load_model(self, model_file):
        self.saver.restore(self.sess, model_file)

    def save_model(self, model_file):
        if self.saver is None:
            raise ValueError("No parameters to save. Add parameters to be saved in build")
        else:
            save_path = self.saver.save(self.sess, model_file)
            print("Model saved in file: %s " % save_path)

    def save_params(self, savepath, num_points):
        nslices = 4
        alpha = (self.sess.run(self.P0.alpha), self.sess.run(self.P1.alpha), self.sess.run(self.P2.alpha))
        beta = (self.sess.run(self.P0.beta), self.sess.run(self.P1.beta), self.sess.run(self.P2.beta))
        for i in range(num_points):
            """calculate the index and weights in phase function """
            pscale = nslices*(float(i)/num_points)
            #weight
            pamount = pscale % 1.0
            #index
            pindex_1 = int(pscale) % nslices
            pindex_0 = (pindex_1-1) % nslices
            pindex_2 = (pindex_1+1) % nslices
            pindex_3 = (pindex_1+2) % nslices
            
            for j in range(len(alpha)):
                a = alpha[j]
                b = beta[j]
                W = PFNN.cubic(a[pindex_0],a[pindex_1],a[pindex_2],a[pindex_3],pamount)
                B = PFNN.cubic(b[pindex_0],b[pindex_1],b[pindex_2],b[pindex_3],pamount)

                W.astype(np.float32).tofile(savepath+'/W%0i_%03i.bin' % (j,i))
                B.astype(np.float32).tofile(savepath+'/b%0i_%03i.bin' % (j,i))

    def load_params(self, database):
        self.load(self.sess, database)
