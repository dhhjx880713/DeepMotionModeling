import os
import tensorflow as tf 
import numpy as np
# from PFNNParameter import PFNNParameter
from tensorflow.keras import layers
rng = np.random.RandomState(23456)


class PFNNLayer(layers.Layer):

    def __init__(self, shape, rng, dropout_rate, activation, name=None, use_discrete_weights=False):
        super(PFNNLayer, self).__init__(name=name)
        self.n_knots = shape[0]
        self.weight_shape = shape
        self.bias_shape = shape[:-1]
        self.rng = rng
        self.dropout = dropout_rate
        self.activation = activation
        self.weight_knots = tf.Variable(self.initial_weight_knots, name="weight_knots")
        self.bias_knots = tf.Variable(np.zeros(self.bias_shape, dtype=np.float32), name='bias_knots')
        self.use_discrete_weights = use_discrete_weights

    def initial_weight_knots(self):
        bound = np.sqrt(6. / np.prod(self.weight_shape[-2:]))
        return np.asarray(self.rng.uniform(low=-bound, high=bound, size=self.weight_shape), dtype=np.float32)

    def call(self, inputs, training=False):
        phase = inputs[:, -1]     
        if training:
            inputs[:, :-1] = layers.Dropout(self.dropout)(inputs[:, :-1]) 
        if not self.use_discrete_weights:

            index_1, bamount, wamount = self.getIndexAmount(phase)
            index_0 = (index_1 - 1) % self.n_knots
            index_2 = (index_1 + 1) % self.n_knots
            index_3 = (index_1 + 2) % self.n_knots

            weight = PFNNLayer.cubic(tf.nn.embedding_lookup(params=self.weight_knots, ids=index_0),
                                    tf.nn.embedding_lookup(params=self.weight_knots, ids=index_1),
                                    tf.nn.embedding_lookup(params=self.weight_knots, ids=index_2),
                                    tf.nn.embedding_lookup(params=self.weight_knots, ids=index_3),
                                    wamount)
            ### weight shape: [len(input), self.weight_shape]
            bias = PFNNLayer.cubic(tf.nn.embedding_lookup(params=self.bias_knots, ids=index_0),
                                tf.nn.embedding_lookup(params=self.bias_knots, ids=index_1),
                                tf.nn.embedding_lookup(params=self.bias_knots, ids=index_2),
                                tf.nn.embedding_lookup(params=self.bias_knots, ids=index_3),
                                bamount) 
        else:
            p_index = tf.cast(phase % 1.0 * self.n_bins, 'int32')
            weight = tf.nn.embedding_lookup(params=self.discretized_weights, ids=p_index)
            bias = tf.nn.embedding_lookup(params=self.discretized_biases, ids=p_index)
        bias = tf.expand_dims(bias, -1)              
        res = tf.matmul(weight, inputs[:, :-1, np.newaxis]) + bias
        if self.activation is not None:
            res = self.activation(res)

        return tf.squeeze(res, -1)                                            

    def getIndexAmount(self, phase):
        pscale = self.n_knots * phase
        pamount = pscale % 1.0
        pindex_1 = tf.cast(pscale, 'int32') % self.n_knots
        bamount = tf.expand_dims(pamount, 1)
        wamount = tf.expand_dims(bamount, 1)          
        return  pindex_1, bamount, wamount

    @staticmethod
    def cubic(y0, y1, y2, y3, mu):
        return (
            (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + 
            (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + 
            (-0.5*y0+0.5*y2)*mu +
            (y1))  

    def export_weights(self, save_path, prefix='', suffix=''):
        self.weight_knots.numpy().tofile(os.path.join(save_path, '{}W{}.bin'.format(prefix, suffix)))
        self.bias_knots.numpy().tofile(os.path.join(save_path, '{}b{}.bin'.format(prefix, suffix)))

    def load_weights(self, path, prefix='', suffix=''):
        weight_value = np.fromfile(os.path.join(path, '{}W{}.bin'.format(prefix, suffix)), np.float32)
        bias_value = np.fromfile(os.path.join(path, '{}b{}.bin'.format(prefix, suffix)), np.float32)
        self.weight_knots.assign(np.reshape(weight_value, self.weight_shape))
        self.bias_knots.assign(np.reshape(bias_value, self.bias_shape))

    def export_discrete_weights(self, n_bins, save_path, prefix='', suffix=''):
        
        for i in range(n_bins):
            pscale = self.n_knots * (float(i) / n_bins)
            pamount = pscale % 1.0
            index_1 = int(pscale) % self.n_knots
            index_0 = (index_1 - 1) % self.n_knots
            index_2 = (index_1 + 1) % self.n_knots
            index_3 = (index_1 + 2) % self.n_knots   

            weight = PFNNLayer.cubic(self.weight_knots[index_0],
                                     self.weight_knots[index_1],
                                     self.weight_knots[index_2],
                                     self.weight_knots[index_3],
                                     pamount)
            bias = PFNNLayer.cubic(self.bias_knots[index_0],
                                   self.bias_knots[index_1],
                                   self.bias_knots[index_2],
                                   self.bias_knots[index_3],
                                   pamount)

            weight.numpy().tofile(os.path.join(save_path, '{}W{}_{:03d}.bin'.format(prefix, suffix, i)))
            bias.numpy().tofile(os.path.join(save_path, '{}b{}_{:03d}.bin'.format(prefix, suffix, i)))                                                       

    def load_discrete_weights(self, n_bins, path, prefix='', suffix=''):
        self.discretized_weights = []
        self.discretized_biases = []
        self.n_bins = n_bins
        try:
            for i in range(n_bins):
                weight_value = np.fromfile(os.path.join(path, '{}W{}_{:03d}.bin'.format(prefix, suffix, i)), dtype=np.float32)
                bias_value = np.fromfile(os.path.join(path, '{}b{}_{:03d}.bin'.format(prefix, suffix, i)), dtype=np.float32)
                self.discretized_weights.append(np.reshape(weight_value, self.weight_shape[1:]))
                self.discretized_biases.append(np.reshape(bias_value, self.bias_shape[1:]))
        except:
            raise IOError('cannot load model weights')
        self.use_discrete_weights = True
        self.discretized_weights = tf.convert_to_tensor(self.discretized_weights)
        self.discretized_biases = tf.convert_to_tensor(self.discretized_biases)



class PFNN(tf.keras.Model):

    def __init__(self, n_controls, input_dim, output_dim, dropout_rate, name='pfnn'):
        super(PFNN, self).__init__(name=name)
        self.n_controls = n_controls
        self.hidden_dim = 512
        self.layer1 = PFNNLayer(shape=(n_controls, self.hidden_dim, input_dim), rng=rng, dropout_rate=dropout_rate, activation=tf.nn.elu, name='L0')
        self.layer2 = PFNNLayer(shape=(n_controls, self.hidden_dim, self.hidden_dim), rng=rng, dropout_rate=dropout_rate, activation=tf.nn.elu, name='L1')
        self.layer3 = PFNNLayer(shape=(n_controls, output_dim, self.hidden_dim), rng=rng, dropout_rate=dropout_rate, activation=None, name='L2')
    
    def call(self, inputs, training=False):
        phase = tf.convert_to_tensor(inputs[:, -1])
        phase = tf.expand_dims(phase, -1)
        layer1_output = self.layer1(inputs, training)
        layer2_output = self.layer2(tf.concat([layer1_output, phase], axis=-1), training)
        output = self.layer3(tf.concat([layer2_output, phase], axis=-1), training)
        return output
    
    def export_weights(self, save_path):
        """save control knots: W0, b0, W1, b1, W2, b2

        """
        self.layer1.export_weights(save_path, suffix='0')
        self.layer2.export_weights(save_path, suffix='1')
        self.layer3.export_weights(save_path, suffix='2')

    def load_weights(self, path):
        self.layer1.load_weights(path, suffix='0')
        self.layer2.load_weights(path, suffix='1')
        self.layer3.load_weights(path, suffix='2')
    
    def export_discrete_weights(self, n_bins, save_path):
        """discretize control spline
        
        Arguments:
            save_path {[string]} -- folder to save discretized parameters
            n_bins {int} -- number of discete points to save
        """

        self.layer1.export_discrete_weights(n_bins, save_path, suffix='0')
        self.layer2.export_discrete_weights(n_bins, save_path, suffix='1')
        self.layer3.export_discrete_weights(n_bins, save_path, suffix='2')
    
    def load_discrete_weights(self, n_bins, path):
        self.layer1.load_discrete_weights(n_bins, path, suffix='0')
        self.layer2.load_discrete_weights(n_bins, path, suffix='1')
        self.layer3.load_discrete_weights(n_bins, path, suffix='2')