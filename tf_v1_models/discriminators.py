import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import utils

'''
Definition of different discriminators networks. 
Usage:
    d = xxxDiscriminator(inputs, *args) to create a discriminator object.
    outputs, trainable parameters and other possible outputs are defined as properties of the discriminator, 
    which can be referred to using d.xxx, e.g. d.outputs, d.parameters
'''

class RNNDiscriminator(object):
    '''
    Discriminator that discriminate between real frames versus generated frames using RNN.
    '''
    def __init__(self, inputs, hidden_size, embedding=None, embeded_size=None, cell_type='lstm', output_z_dims=None, bidirectional=False, 
                 output_category_dims=None, use_multi_states=False, input_length=None, output_length=None, 
                 bias_initializer=tf.constant_initializer(0.), kernel_initializer=tf.truncated_normal_initializer(stddev=0.001), 
                 reuse=tf.AUTO_REUSE, scope=""):
        '''
        Initialize the discriminator network.

        Args:
            inputs(tf.placeholder): The input variable containing current data in shape [batch_size, sequence_length, input_dims].
            hidden_size: number of hidden units in the cell.
            embeded_size: input embed size.
            output_z_dims: if not None, to regress the random vector used to generate the predicted frames.
            cell_type: "lstm" or "gru", the type of cell used in the model.
            bidirectional: True to use bidirectional RNN.
            output_category_dims: if not None, it is the number of action categories to classify.
            use_multi_states: True to use 4 selected hidden states to get the output 
            input_length: the length of given previous sequence.
            output_length: the length of predicted output sequence.
            bias_initializer: initializer for bias.
            kernel_initializer: initializer for weights.
            reuse: True, None or tf.AUTO_REUSE, whether to reuse parameters from a previously created model.
            scope: variable scope name.
        '''
        self._embedding = embedding
        if self._embedding is not None:
            self._input_dims = self._embedding.z_size
        else:
            self._input_dims = inputs.shape[2]
        self._hidden_size = hidden_size
        self._embeded_size = embeded_size
        self._cell_type = cell_type
        self._output_z_dims = output_z_dims
        self._bidirectional = bidirectional
        self._output_category_dims = output_category_dims
        self._use_multi_states = use_multi_states
        if self._use_multi_states:
            if input_length is None or output_length is None:
                raise ValueError("if using multiple states to get the output, you must give the length of input and output sequence")
        self._input_length = input_length
        self._output_length = output_length
        self._sequence_length = self._input_length + self._output_length
        self._bias_initializer = bias_initializer
        self._kernel_initializer = kernel_initializer
        self._reuse = reuse
        self._scope = scope
        self._parameters = []
        self._outputs = None #for WGAN
        self._prob = None #for GAN
        self._R = None
        self._output_category = None
        if scope.strip() == "":
            self._scope = scope
        else:
            self._scope = scope + "/"
        self._build(inputs)

    @property
    def outputs(self):
        return self._outputs

    @property
    def prob(self):
        return self._prob
    
    @property
    def R(self):
        return self._R
    
    @property
    def parameters(self):
        return self._parameters

    @property
    def output_category(self):
        return self._output_category
    
    def connect(self, inputs):
        self._build(inputs)
        return self._outputs, self._R

    def _get_a_cell(self):
        if self._cell_type == 'lstm':
            cell = tf.nn.rnn_cell.LSTMCell(self._hidden_size)
        elif self._cell_type == 'gru':
            cell = tf.nn.rnn_cell.GRUCell(self._hidden_size)
        return cell
    
    # def _input_projection(self, inputs):
    #     if self._embeded_size: 
    #         Wi = tf.get_variable("Wi", shape=[self._input_dims, self._embeded_size], initializer=self._kernel_initializer) 
    #         bi = tf.get_variable("bi", shape=[self._embeded_size], initializer=self._bias_initializer)
    #         encoder_inputs = tf.tensordot(inputs, Wi, axes=[[2], [0]]) + bi
    #     elif self._embedding is not None:
    #         inputs = tf.reshape(inputs, shape=[-1]+inputs.shape[2:].as_list())
    #         self._embedding.build_encoder(inputs)
    #         encoded = self._embedding.encoded
    #         encoder_inputs = tf.reshape(encoded, shape=[-1, self._input_length, self._input_dims])
    #     else:
    #         encoder_inputs = inputs
    #     return encoder_inputs

    def _build(self, inputs):
        if self._embedding is not None:
            inputs = tf.reshape(inputs, shape=[-1]+inputs.shape[2:].as_list())
            self._embedding.build_encoder(inputs)
            encoded = self._embedding.encoded
            inputs = tf.reshape(encoded, shape=[-1, self._sequence_length, self._input_dims])
        with tf.variable_scope(self._scope + self.__class__.__name__, reuse=self._reuse) as vs:
            if self._embeded_size: 
                Wi = tf.get_variable("Wi", shape=[self._input_dims, self._embeded_size], initializer=self._kernel_initializer) 
                bi = tf.get_variable("bi", shape=[self._embeded_size], initializer=self._bias_initializer)
                inputs = tf.tensordot(inputs, Wi, axes=[[2], [0]]) + bi
            if self._bidirectional:
                fw_cell = self._get_a_cell()
                bw_cell = self._get_a_cell()
                input_list = tf.unstack(inputs,self._sequence_length, axis=1)
                outputs = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell, input_list, dtype=tf.float32) 
            else:
                cell = self._get_a_cell()
                outputs = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            if self._use_multi_states:
                if self._bidirectional:
                    #output_fw = outputs[0]
                    #output_bw = outputs[1]]
                    output_state = [outputs[0][self._input_length][:,:self._hidden_size], outputs[0][-1][:,:self._hidden_size],
                                    outputs[0][self._output_length][:,self._hidden_size:], outputs[0][-1][:,self._hidden_size:]]
                    #fw_states = tf.gather(output_fw, [self._input_length,self._sequence_length-1], axis=1)
                    #bw_states = tf.gather(output_bw, [self._output_length,self._sequence_length-1], axis=1)
                    #final_state = tf.concat(values=[fw_states,bw_states], axis=1)
                    final_state = tf.concat(values=output_state, axis=1)
                    final_state = tf.reshape(final_state, shape=[-1,np.prod(final_state.shape[1:].as_list())])
                else:
                    final_state = tf.gather(outputs, [self._input_length-1, -1], axis=1)
                    final_state = tf.reshape(final_state, shape=[-1,np.prod(final_state.shape[1:].as_list())])
            else:
                final_state = outputs[:,-1,:]
            
            base = tf.layers.dense(inputs=final_state, units=self._hidden_size, activation=tf.nn.relu, name="fc1")
            self._outputs = tf.layers.dense(inputs=base, units=1, activation=None, name="output")
            if self._output_category_dims: #actions category
                self._output_category = tf.layers.dense(inputs=base, units=self._output_category_dims, 
                                                        activation=tf.nn.softmax, name="output_categories")
            if self._output_z_dims: # regress random vector
                self._R = tf.layers.dense(inputs=base, units=self._output_z_dims, activation=None, name="output_r")
            self._prob = tf.nn.sigmoid(self._outputs)
            self._parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name+'/')
            
class NNDiscriminator(object):
    '''
    The discriminator uses feedforward neural network.
    '''
    def __init__(self, inputs, hidden_size, reuse=tf.AUTO_REUSE, scope=""):
        '''
        Initialize the discriminator network.

        Args:
            inputs: a tf.placeholder in shape [batch_size, sequence_length, element_size], the input variable containing current data.
            hidden_size: an array of number of hidden units in each layer.
            reuse: True, None or tf.AUTO_REUSE, whether to reuse parameters from a previously created model.
            scope: variable scope name.
        '''
        self._hidden_size = hidden_size
        self._reuse = reuse
        self._outputs = None
        self._prob = None
        self._parameters = []
        if scope.strip() == "":
            self._scope = scope
        else:
            self._scope = scope + "/"

        self._build(inputs)
    
    @property
    def outputs(self):
        return self._outputs

    @property
    def prob(self):
        return self._prob

    @property
    def parameters(self):
        return self._parameters

    def _build(self, inputs):
        with tf.variable_scope(self._scope + self.__class__.__name__, reuse=self._reuse) as vs:
            net = tf.reshape(inputs, [-1, np.prod(inputs.shape[1:].as_list())])
            for i in range(len(self._hidden_size)):
                net = tf.layers.dense(inputs=net, units=self._hidden_size[i], activation=tf.nn.relu, name="fc{}".format(i), reuse=self._reuse)
            self._outputs = tf.layers.dense(inputs=net, units=1, name='fcoutput', reuse=self._reuse)
            self._prob = tf.nn.sigmoid(self._outputs)
            self._parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name+'/')

class NNResidualDiscriminator(object):
    '''
    The discriminator uses feedforward neural network with residual connection.
    '''
    def __init__(self, inputs, embeded_size, hidden_size, num_residual_blocks, 
                 activation=tf.nn.relu, reuse=tf.AUTO_REUSE, scope=""):
        
        self._element_shape = [np.prod(inputs.shape[2:].as_list()), embeded_size]
        self._hidden_size = hidden_size
        self._num_residual_blocks = _num_residual_blocks
        self._activation = activation
        self._reuse = reuse
        self._outputs = None
        self._prob = prob
        self._parameters = parameters
        if scope.strip() == "":
            self._scope = scope
        else:
            self._scope = scope + "/"

        self._build(inputs)

    @property
    def outputs(self):
        return self._outputs

    @property
    def prob(self):
        return self._prob

    @property
    def parameters(self):
        return self._parameters

    def _build_residual_unit(self, inputs, residual_index):
        '''
        Build a single residual unit.
        Args:
            inputs: input to the resnet unit.
        '''
        net1 = tf.layers.dense(inputs=inputs,
                               units=self._hidden_size,
                               activation=self._activation, 
                               name="resnet1_{}".format(residual_index+1), 
                               reuse=self._reuse)
        net2 = tf.layers.dense(inputs=net1,
                               units=inputs.shape[-1],
                               activation=None, 
                               name="resnet2_{}".format(residual_index+1), 
                               reuse=self._reuse)
        return  self._activation(net2 + inputs)

    def _build(self, inputs):
        with tf.variable_scope(self._scope + self.__class__.__name__, reuse=self._reuse) as vs:
            Wi = tf.get_variable("Wi", initializer=tf.truncated_normal(self._element_shape))
            d_inputs = tf.tensordot(inputs, Wi, axes=[[2], [0]])
            d_inputs.set_shape([None, inputs.shape[1].value, self._element_shape[-1]])
            net = tf.reshape(d_inputs, [-1, np.prod(d_inputs.shape[1:].as_list())])
            net = tf.layers.dense(inputs=net, 
                                  units=self._hidden_size, 
                                  activation=self._activation, 
                                  name="fc1", 
                                  reuse=self._reuse)
            residual_index = 0
            for _ in range(self._num_residual_blocks):
                net = self._build_residual_unit(net, residual_index)
                residual_index += 1
            self._outputs = tf.layers.dense(inputs=net, 
                                           units=1, 
                                           name="fc2", 
                                           reuse=self._reuse)
            self._prob = tf.nn.sigmoid(self._outputs)
            self._parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name+'/')

    
def test():
    batch_size = 32
    sequence_length = 30
    input_dims = 60
    input = tf.placeholder(shape = [batch_size, sequence_length, input_dims], dtype=tf.float32)
    d = NNDiscriminator(input, hidden_size=[32, 16])
    print(d.outputs.shape)
    print(d.parameters)


if __name__ == "__main__":
    test()    