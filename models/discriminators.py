import tensorflow as tf 
import numpy as np
tf.compat.v1.disable_v2_behavior()


class NNDiscriminator_HPGAN(object):
    '''
    GAN discriminator network that discriminate between real future frames versus
    synthetic predicted frames by the generator. The discriminator uses feedforward
    neural network.
    '''
    def __init__(self, inputs, inputs_depth, sequence_length, reuse=False, scope=""):
        '''
        Initialize the discriminator network.

        Args:
            inputs(tf.placeholder): The input variable containing the data.
            inputs_depth(int): input embed size.
            sequence_length(int): the length of the input sequence.
            reuse(bool): True to reuse model parameters from a previously created model.
            scope(str): Prepend variable scope with `scope`.
        '''
        self._reuse = reuse
        self._num_neurons = 512
        self._num_layers = 3
        self._sequence_length = sequence_length
        self._element_shape = [np.prod(inputs.shape[2:].as_list()), inputs_depth]
        self._output_dims = 1
        self._output = None
        self._prob = None
        self._parameters = []
        self._weights = []
        self._stddev = 0.001
        if scope.strip() == "":
            self._scope = scope
        else:
            self._scope = scope + "/"

        self._build(inputs)

    @property
    def output(self):
        ''' Raw output of the network '''
        return self._output

    @property
    def prob(self):
        ''' Probalistic output of the network '''
        return self._prob

    @property
    def parameters(self):
        ''' All trainable parameters '''
        return self._parameters

    @property
    def weights(self):
        ''' Weights only parameters for regularization '''
        return self._weights

    def _build(self, inputs):
        '''
        Construct a discriminator model.

        Args:
            inputs(tf.placeholder): The input variable containing the data.
        '''
        with tf.compat.v1.variable_scope(self._scope + self.__class__.__name__, reuse=self._reuse) as vs:
            Wi = tf.compat.v1.get_variable("Wi", initializer=tf.random.truncated_normal(self._element_shape))
            # print("Wi shape: ", Wi.shape)
            inputs = tf.reshape(inputs, shape=[-1, inputs.shape[1]]+[np.prod(inputs.shape[2:].as_list())])

            d_inputs = tf.tensordot(inputs, Wi, axes=[[2], [0]])
            d_inputs.set_shape([None, inputs.shape[1], self._element_shape[-1]]) # https://github.com/tensorflow/tensorflow/issues/6682

            net = tf.reshape(d_inputs, [-1, np.prod(d_inputs.shape[1:].as_list())])

            layer_index = 0
            for _ in range(self._num_layers):
                net = tf.compat.v1.layers.dense(inputs=net, units=self._num_neurons, activation=tf.nn.relu, name="fc{}".format(layer_index+1), reuse=self._reuse)
                # net = tf.layers.dropout(inputs=net, rate=0.5)
                layer_index += 1
                # print("layer: {} shape: {}".format(layer_index, net.shape))

            self._output = tf.compat.v1.layers.dense(inputs=net, units=self._output_dims, name="fc{}".format(layer_index+1), reuse=self._reuse)
            self._prob = tf.nn.sigmoid(self._output)

            self._parameters = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name+'/')
            self._weights = [v for v in self._parameters if (v.name.endswith('Wi:0') or v.name.endswith('weights:0') or v.name.endswith('kernel:0'))]


def create_rnn_cell(cell_type, num_neurons, use_residual=False):
    ''' Create RNN cell depend on the provided type '''
    cell = None
    if cell_type == "lstm":
        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_neurons)
    elif cell_type == "lstmp":
        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_neurons, use_peepholes=True)        
    elif cell_type == "gru":
        cell = tf.compat.v1.nn.rnn_cell.GRUCell(num_neurons)
    # elif cell_type == "norm_lstm":
    #     cell = tf.compat.v1.nn.rnn_cell.LayerNormBasicLSTMCell(num_neurons)
    else:
        raise Exception("Unsupported cell type.")

    if use_residual:
        cell = tf.compat.v1.nn.rnn_cell.ResidualWrapper(cell)
    return cell


def create_rnn_model(num_layers, cell_type, num_neurons, use_residual=False):
    ''' Create RNN model '''
    if num_layers > 1:
        return tf.compat.v1.nn.rnn_cell.MultiRNNCell([create_rnn_cell(cell_type, num_neurons, use_residual) for _ in range(num_layers)])
    else:
        return create_rnn_cell(cell_type, num_neurons, use_residual)


def attention(inputs, 
              activation=None, 
              attention_len=None, 
              kernel_initializer=None,
              bias_initializer=None):
              
    num_neurons = inputs.shape[2].value
    if attention_len == None:
        attention_len = num_neurons

    # Trainable parameters
    Watt = tf.compat.v1.get_variable("Watt", shape=[num_neurons, attention_len], initializer=kernel_initializer)
    batt = tf.compat.v1.get_variable("batt", shape=[attention_len], initializer=bias_initializer)
    uatt = tf.compat.v1.get_variable("uatt", shape=[attention_len], initializer=kernel_initializer)

    # (Batch,Seq,Neuron) * (Neuron,Att) --> (Batch,Seq,Att)
    v = tf.compat.v1.einsum('bsn,na->bsa', inputs, Watt) + batt
    if activation != None:
        v = activation(v)

    # (Batch,Seq,Att) * (Att) --> (Batch, Seq)
    vu = tf.compat.v1.einsum('bsa,a->bs', v, uatt)
    alpha = tf.compat.v1.nn.softmax(vu, name='alpha')

    return tf.compat.v1.reduce_sum(inputs * tf.compat.v1.expand_dims(alpha, -1), 1)


class RNNDiscriminator(object):
    '''
    GAN discriminator network that discriminate between real future frames versus
    synthetic predicted frames by the generator. The discriminator uses recurrent
    network.
    '''
    def __init__(self, inputs, inputs_depth, sequence_length, 
                 use_attention=False, use_residual=False, cell_type='gru', output_category_dims=None,
                 bias_initializer=tf.constant_initializer(0.), kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.001),
                 reuse=False, scope=""):
        '''
        Initialize the discriminator network.

        Args:
            inputs(tf.placeholder): The input variable containing the data.
            inputs_depth(int): input embed size.
            sequence_length(int): the length of the input sequence.
            reuse(bool): True to reuse model parameters from a previously created model.
            scope(str): Prepend variable scope with `scope`.
        '''
        self._reuse = reuse
        self._use_attention = use_attention
        self._use_residual=use_residual
        self._bias_initializer=bias_initializer
        self._kernel_initializer=kernel_initializer
        self._cell_type = cell_type
        self._num_neurons = 1024
        self._num_layers = 2
        self._sequence_length = sequence_length
        self._element_shape = [np.prod(inputs.shape[2:].as_list()), inputs_depth]
        self._output_dims = 1
        self._output_category_dims = output_category_dims
        self._output = None
        self._output_category = None        
        self._prob = None
        self._parameters = []
        self._weights = []
        if scope.strip() == "":
            self._scope = scope
        else:
            self._scope = scope + "/"
        self._build(inputs)

    @property
    def output(self):
        ''' Raw output of the network '''
        return self._output

    @property
    def output_category(self):
        ''' Raw classification output of the network '''
        return self._output_category

    @property
    def prob(self):
        ''' Probalistic output of the network '''
        return self._prob

    @property
    def parameters(self):
        ''' All trainable parameters '''
        return self._parameters

    @property
    def weights(self):
        ''' Weights only parameters for regularization '''
        return self._weights

    def _build(self, inputs):
        '''
        Construct a discriminator model.

        Args:
            inputs(tf.placeholder): The input variable containing the data.
        '''
        with tf.compat.v1.variable_scope(self._scope + self.__class__.__name__, reuse=self._reuse) as vs:
            Wi = tf.compat.v1.get_variable("Wi", initializer=tf.compat.v1.truncated_normal(self._element_shape))
            inputs = tf.reshape(inputs, shape=[-1, inputs.shape[1].value]+[np.prod(inputs.shape[2:].as_list())])

            d_inputs = tf.tensordot(inputs, Wi, axes=[[2], [0]])
            d_inputs.set_shape([None, inputs.shape[1].value, self._element_shape[-1]]) # https://github.com/tensorflow/tensorflow/issues/6682

            cell = create_rnn_model(self._num_layers, self._cell_type, self._num_neurons)
            outputs, state = tf.compat.v1.nn.dynamic_rnn(cell, d_inputs, dtype=tf.float32)
            print("output shape {}".format(outputs.shape))
            if self._use_attention:
                last = attention(outputs,
                                    kernel_initializer=self._kernel_initializer,
                                    bias_initializer=self._bias_initializer)
            else:
                outputs = tf.transpose(outputs, [1, 0, 2])
                print("transposed shape: {}".format(outputs.shape))
                last = tf.gather(outputs, int(outputs.shape[0]) - 1)

            print("last shape {}.".format(last.shape))
            base = tf.compat.v1.layers.dense(inputs=last,
                                   units=self._num_neurons,
                                   activation=tf.nn.relu,
                                   name="fc1")
            print("base shape {}.".format(base.shape))
            self._output = tf.compat.v1.layers.dense(inputs=base,
                                           units=self._output_dims,
                                           activation=None,
                                           name="output")

            if self._output_category_dims != None:
                self._output_category = tf.compat.v1.layers.dense(inputs=base,
                                                        units=self._output_category_dims,
                                                        activation=None,
                                                        name="output_categories")

            # Wo = tf.get_variable("Wo", shape=[self._num_neurons, self._output_dims], initializer=self._kernel_initializer)
            # bo = tf.get_variable("bo", shape=[self._output_dims], initializer=self._bias_initializer)

            # self._output = tf.matmul(last, Wo) + bo
            self._prob = tf.nn.sigmoid(self._output)

            self._parameters = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name+'/')
            self._weights = [v for v in self._parameters if (v.name.endswith('Wi:0') or v.name.endswith('weights:0') or v.name.endswith('Wo:0'))]


def NNDiscriminator_HPGAN_test():
    element_size = 87
    input_sequence_length = 40
    inputs = tf.compat.v1.placeholder(tf.float32, shape=(None, input_sequence_length, element_size))
    embedding_size = 10
    d = NNDiscriminator_HPGAN(inputs, embedding_size, input_sequence_length)
    res = d.output
    print(res.shape)


def RNNDiscriminator_test():
    element_size = 87
    input_sequence_length = 40
    inputs = tf.compat.v1.placeholder(tf.float32, shape=(None, input_sequence_length, element_size))
    embedding_size = 10
    d = RNNDiscriminator(inputs, embedding_size, input_sequence_length)
    res = d.output
    print(res.shape)    


def test():
    data = tf.compat.v1.placeholder(tf.float32, shape=(40, None, 1024))
    last = tf.gather(data, int(data.shape[0])-1)
    print(last.shape)

if __name__ == "__main__":
    # NNDiscriminator_HPGAN_test()
    # RNNDiscriminator_test()
    test()