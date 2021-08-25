import tensorflow as tf 
import numpy as np
tf.compat.v1.disable_v2_behavior()


class RNNGenerator(object):
    '''
    GAN generator network that predict a single future skeleton pose using RNN network.
    '''
    def __init__(self, inputs, inputs_depth, reuse=False):
        '''
        Initialize the generative network.

        Args:
            inputs(tf.placeholder): The input variable containing current data.
            inputs_depth(int): input embed size.
            reuse(bool): True to reuse model parameters from a previously created model.
        '''
        self._reuse = reuse
        self._batch_size = inputs.shape[0] # batch_size
        self._inputs_depth = inputs_depth
        self._inputs_shape = inputs.shape
        self._element_shape = [np.prod(inputs.shape[2:].as_list()), inputs_depth]
        self._output = None
        self._parameters = []
        self._weights = []
        self._stddev = 0.001
        self._build(inputs)

    @property
    def output(self):
        return self._output

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
        Construct a generative model.

        Args:
            inputs(tf.placeholder): The input variable containing current data.
        '''
        with tf.compat.v1.variable_scope(self.__class__.__name__, reuse=self._reuse) as vs:

            num_neurons = 256
            num_layers = 2

            Wi = tf.compat.v1.get_variable("Wi", initializer=tf.random.truncated_normal(self._element_shape, stddev=self._stddev))
            inputs = tf.reshape(inputs, shape=[-1, inputs.shape[1].value]+[np.prod(inputs.shape[2:].as_list())])

            d_inputs = tf.tensordot(inputs, Wi, axes=[[2], [0]])
            d_inputs.set_shape([None, inputs.shape[1].value, self._element_shape[-1]]) # https://github.com/tensorflow/tensorflow/issues/6682

            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_neurons)
            # cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=0.5)
            cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

            # Getting the last output
            outputs, state = tf.compat.v1.nn.dynamic_rnn(cell, d_inputs, dtype=tf.float32)
            outputs = tf.transpose(a=outputs, perm=[1, 0, 2])
            last = tf.gather(outputs, int(outputs.shape[0]) - 1)

            Wo = tf.compat.v1.get_variable("Wo", initializer=tf.random.truncated_normal([num_neurons, self._element_shape[0]], stddev=self._stddev)) #[256,96]
            bo = tf.compat.v1.get_variable("bo", initializer=tf.constant(0., shape=[self._element_shape[0]]))

            pred = tf.matmul(last, Wo) + bo #[batch_size, 96]?
            pred = tf.reshape(pred, pred.shape[:1].as_list() + [1] + self._inputs_shape[2:].as_list())#[batch_size,1,32,3]

            self._output = tf.tanh(pred)
            self._parameters = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name+'/')
            self._weights = [v for v in self._parameters if (v.name.endswith('Wi:0') or v.name.endswith('weights:0') or v.name.endswith('Wo:0'))]



class Seq2Seq_HPGAN(object):

    '''
    GAN generator network that predict future skeleton poses using sequernce to sequence network.
    '''
    def __init__(self, inputs, inputs_depth, z, input_sequence_length, output_sequence_length, 
                 cell_type='gru', project_to_rnn_output=False, reverse_input=False,
                 use_attention=False, use_residual=False,
                 bias_initializer=tf.compat.v1.constant_initializer(0.), kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.001),
                 reuse=False):
        '''
        Initialize the generative network.

        Args:
            inputs(tf.placeholder): The input variable containing current data.
            inputs_depth(int): input embed size.
            z(tf.placeholder, optional): A random generated input vector used as input.
            input_sequence_length(int): the length of the input sequence.
            output_sequence_length(int): the length of the resulted sequence.
            cell_type(str): The type of cell to use for the encode and decoder.
            project_to_rnn_output(bool): project the input to the number of hidden unit in the RNN.
            reverse_input(bool): reverse the input sequence before feeding it to the network.
            use_attention(bool): true to use attention instead of the last state of the encoder.
            use_residual(bool): use resent like structure for the recurrent.
            bias_initializer: initializer for the bias value.
            kernel_initializer: initializer for the `W` parameters.            
            reuse(bool): True to reuse model parameters from a previously created model.
        '''
        self._reuse = reuse
        self._batch_size = tf.shape(input=inputs)[0] # batch_size
        self._input_sequence_length = input_sequence_length
        self._output_sequence_length = output_sequence_length
        self._inputs_depth = inputs_depth
        self._inputs_shape = inputs.shape
        self._element_shape = inputs.shape[2:].as_list()
        self._output = None
        self._parameters = []
        self._weights = []
        self._num_neurons = 1024
        self._num_layers = 2
        self._num_nn_layers = 2
        self._cell_type = cell_type
        self._bias_initializer = bias_initializer
        self._kernel_initializer = kernel_initializer
        self._reccurent_bias_initializer = None
        self._reccurent_kernel_initializer = None
        self._project_to_rnn_output = project_to_rnn_output
        self._use_attention = use_attention
        self._use_residual = use_residual

        if self._use_residual:
            self._project_to_rnn_output = True

        # Similar to tf.zeros but support variable batch size.
        if self._project_to_rnn_output:
            self._zeros_input = tf.fill(tf.stack([tf.shape(input=inputs)[0], self._num_neurons]), 0.0)
        else:
            self._zeros_input = tf.fill(tf.stack([tf.shape(input=inputs)[0], self._inputs_depth]), 0.0)

        if reverse_input:
            inputs = tf.reverse(inputs, axis=[1])
        self._build(inputs, z)

    @property
    def output(self):
        return self._output

    @property
    def parameters(self):
        ''' All trainable parameters '''
        return self._parameters

    @property
    def weights(self):
        ''' Weights only parameters for regularization '''
        return self._weights

    def _build(self, inputs, z):
        '''
        Construct a generative model.

        Args:
            inputs(tf.placeholder): The input variable containing current data.
            z(tf.placeholder): A vector containss the randomly generated latent data.
        '''
        with tf.compat.v1.variable_scope(self.__class__.__name__, reuse=self._reuse) as vs:

            outputs, encoder_state = self._build_encoder(inputs, z)

            first_input = outputs[:, -1, :] # [batch, sequence, elements]
            if self._use_attention:
                encoder_state = attention(outputs, 
                                             kernel_initializer=self._kernel_initializer,
                                             bias_initializer=self._bias_initializer)
            self._output = self._build_decoder(first_input, z, encoder_state)

            self._parameters = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name+'/')
            self._weights = [v for v in self._parameters if (v.name.endswith('Wi:0') or \
                                                             v.name.endswith('weights:0') or \
                                                             v.name.endswith('Wo:0') or \
                                                             v.name.endswith('Wsi:0') or \
                                                             ('Wzi' in v.name) or \
                                                             ('Wzci' in v.name) or \
                                                             ('Wzhi' in v.name))]

    def _create_rnn_model(self):
        ''' Create RNN model '''
        return create_rnn_model(self._num_layers, 
                                   self._cell_type, 
                                   self._num_neurons, 
                                   use_residual=self._use_residual)

    def _input_projection(self, inputs):
        ''' Project each skeleton pose to the encoder. '''

        inputs = tf.reshape(inputs, shape=[-1, inputs.shape[1]]+[np.prod(inputs.shape[2:].as_list())]) #[batch,sequence,96]
        if self._project_to_rnn_output:
            net = inputs
            layer_index = 0
            num_neurons = self._num_neurons // (self._num_nn_layers+1) #num_neurons // 3???
            for i in range(self._num_nn_layers):
                net = tf.compat.v1.layers.dense(inputs=net, 
                                      units=(i+1)*num_neurons, #increasing hidden neurons
                                      kernel_initializer=self._kernel_initializer,
                                      bias_initializer=self._bias_initializer,
                                      activation=tf.nn.relu,
                                      name="fc{}".format(layer_index+1))
                # net = tf.layers.dropout(inputs=net, rate=0.5)
                layer_index += 1

            encoder_inputs = tf.compat.v1.layers.dense(inputs=net, 
                                             units=self._num_neurons,
                                             kernel_initializer=self._kernel_initializer,
                                             bias_initializer=self._bias_initializer,
                                             activation=None, 
                                             name="fc{}".format(layer_index+1))
            # encoder_inputs.set_shape([inputs.shape[0].value, inputs.shape[1].value, self._num_neurons])
            encoder_inputs.set_shape([inputs.shape[0], inputs.shape[1], self._num_neurons])
        else:
            Wi = tf.compat.v1.get_variable("Wi", shape=[np.prod(self._element_shape), self._inputs_depth], initializer=self._kernel_initializer)
            bi = tf.compat.v1.get_variable("bi", shape=[self._inputs_depth], initializer=self._bias_initializer)

            encoder_inputs = tf.tensordot(inputs, Wi, axes=[[2], [0]])
            # encoder_inputs.set_shape([inputs.shape[0].value, inputs.shape[1].value, self._inputs_depth]) # https://github.com/tensorflow/tensorflow/issues/6682
            encoder_inputs.set_shape([inputs.shape[0], inputs.shape[1], self._inputs_depth])
            encoder_inputs = encoder_inputs + bi

        return encoder_inputs

    def _output_projection(self, outputs):
        ''' Project each decoder output back to skeleton pose. '''

        if self._project_to_rnn_output:
            net = outputs
            layer_index = 0
            for i in range(self._num_nn_layers):
                net = tf.compat.v1.layers.dense(inputs=net, 
                                      units=int(self._num_neurons/(i+1)),
                                      kernel_initializer=self._kernel_initializer,
                                      bias_initializer=self._bias_initializer,
                                      activation=tf.nn.relu, 
                                      name="fc{}".format(layer_index+1))
                # net = tf.layers.dropout(inputs=net, rate=0.5)
                layer_index += 1

            pred = tf.compat.v1.layers.dense(inputs=net, 
                                   units=np.prod(self._element_shape),
                                   kernel_initializer=self._kernel_initializer,
                                   bias_initializer=self._bias_initializer,
                                   activation=None, 
                                   name="fc{}".format(layer_index+1))

            # pred.set_shape([outputs.shape[0].value, self._output_sequence_length, np.prod(self._element_shape)])
            pred.set_shape([outputs.shape[0], self._output_sequence_length, np.prod(self._element_shape)])
            # pred = tf.reshape(pred, shape=[-1, pred.shape[1].value] + self._inputs_shape[2:].as_list())
            pred = tf.reshape(pred, shape=[-1, pred.shape[1]] + self._inputs_shape[2:].as_list())
        else:
            Wo = tf.compat.v1.get_variable("Wo", shape=[self._num_neurons, np.prod(self._element_shape)], initializer=self._kernel_initializer)
            bo = tf.compat.v1.get_variable("bo", shape=[np.prod(self._element_shape)], initializer=self._bias_initializer)

            pred = tf.tensordot(outputs, Wo, axes=[[2], [0]])
            # pred.set_shape([outputs.shape[0].value, self._output_sequence_length, np.prod(self._element_shape)]) # https://github.com/tensorflow/tensorflow/issues/6682
            # pred = pred + bo
            # pred = tf.reshape(pred, shape=[-1, pred.shape[1].value] + self._inputs_shape[2:].as_list())
            pred.set_shape([outputs.shape[0], self._output_sequence_length, np.prod(self._element_shape)]) # https://github.com/tensorflow/tensorflow/issues/6682
            pred = pred + bo
            pred = tf.reshape(pred, shape=[-1, pred.shape[1]] + self._inputs_shape[2:].as_list())            

        return pred

    def _build_encoder(self, inputs, z):
        ''' Build the encoder part of the generative mode. '''
        with tf.compat.v1.variable_scope("encoder", reuse=self._reuse):
            encoder_inputs = self._input_projection(inputs)

            cell = self._create_rnn_model()
            outputs, state = tf.compat.v1.nn.dynamic_rnn(cell, encoder_inputs, dtype=tf.float32)  

            return outputs, state

    def _build_decoder(self, first_input, z, encoder_state):
        '''
        Build the decoder part of the generative mode. It can decode based on the initial state without
        the need of future_inputs.

        Args:
            first_input(tf.placeholder, optional): each cell takes input form the output of the previous cell,
                                                   except first cell. first_input is used for the first cell.
            z(tf.placeholder, optional): random vector in order to sample multiple predictions from the 
                                         same input.
            encoder_state(cell state): the last state of the encoder.

        Return:
            The output of the network.
        '''
        with tf.compat.v1.variable_scope("decoder", reuse=self._reuse):
            cell = self._create_rnn_model()
            outputs, _ = self._dynamic_rnn_decoder(cell, first_input, z, encoder_state, self._output_sequence_length)
            return self._output_projection(outputs)

    def _dynamic_rnn_decoder(self, cell, first_input, z, encoder_state, sequence_length, time_major=False, dtype=tf.float32):
        ''' Unroll the RNN decoder '''
        if not self._project_to_rnn_output:
            # From output state to input embed.
            Wsi = tf.compat.v1.get_variable("Wsi", 
                                  shape=[self._num_neurons, self._inputs_depth], 
                                  initializer=self._kernel_initializer)

        if first_input is None:
            first_input = self._zeros_input

        first_input = first_input if self._project_to_rnn_output else tf.matmul(first_input, Wsi)
        #if _project_to_rnn_output, the first input shape is [,num_neuron], else [,inputs_depth]

        if z is not None:
            is_tuple = isinstance(encoder_state[0], tf.compat.v1.nn.rnn_cell.LSTMStateTuple) if (self._num_layers > 1) else isinstance(encoder_state, tf.nn.rnn_cell.LSTMStateTuple)
            if is_tuple:
                if self._num_layers > 1:
                    states = []
                    for i in range(self._num_layers):
                        Wzhi = tf.compat.v1.get_variable("Wzhi{}".format(i), shape=[z.shape.as_list()[-1], encoder_state[i].h.shape.as_list()[-1]], initializer=self._kernel_initializer)
                        Wzci = tf.compat.v1.get_variable("Wzci{}".format(i), shape=[z.shape.as_list()[-1], encoder_state[i].c.shape.as_list()[-1]], initializer=self._kernel_initializer)                        
                        states.append(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(encoder_state[i].c + tf.matmul(z, Wzci), encoder_state[i].h + tf.matmul(z, Wzhi)))
                    encoder_state = tuple(states)
                else:
                    Wzhi = tf.compat.v1.get_variable("Wzhi", shape=[z.shape.as_list()[-1], encoder_state.h.shape.as_list()[-1]], initializer=self._kernel_initializer)
                    Wzci = tf.compat.v1.get_variable("Wzci", shape=[z.shape.as_list()[-1], encoder_state.c.shape.as_list()[-1]], initializer=self._kernel_initializer)      
                    encoder_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(encoder_state.c + tf.matmul(z, Wzci), encoder_state.h + tf.matmul(z, Wzhi))
            #not is_tuple, means not from lstmencoder?
            else:
                if self._num_layers > 1:
                    states = []
                    for i in range(self._num_layers):
                        Wzi = tf.compat.v1.get_variable("Wzi{}".format(i), shape=[z.shape.as_list()[-1], encoder_state[i].shape.as_list()[-1]], initializer=self._kernel_initializer)
                        states.append(encoder_state[i] + tf.matmul(z, Wzi))
                    encoder_state = tuple(states)
                else:
                    Wzi = tf.compat.v1.get_variable("Wzi", shape=[z.shape.as_list()[-1], encoder_state.shape.as_list()[-1]], initializer=self._kernel_initializer)
                    encoder_state = encoder_state + tf.matmul(z, Wzi)

        def loop_fn_init(time):
            elements_finished = (sequence_length <= 0)
            next_input = first_input
            next_cell_state = encoder_state
            emit_output = None
            next_loop_state = None

            return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

        def loop_fn_next(time, cell_output, cell_state, loop_state):
            emit_output = cell_output
            next_cell_state = cell_state

            elements_finished = (time >= sequence_length)
            finished = tf.reduce_all(input_tensor=elements_finished) #reduce all batches
            next_input = tf.cond(
                pred=finished,
                true_fn=lambda: self._zeros_input,
                false_fn=lambda: cell_output if self._project_to_rnn_output else tf.matmul(cell_output, Wsi))
            next_loop_state = None
            return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is None:
                return loop_fn_init(time)
            else:
                return loop_fn_next(time, cell_output, cell_state, loop_state)
        #details about loop function for rnn:https://www.tensorflow.org/api_docs/python/tf/nn/raw_rnn?hl=en 
        outputs_ta, final_state, _ = tf.compat.v1.nn.raw_rnn(cell, loop_fn)
        outputs = outputs_ta.stack()

        if not time_major:
            outputs = tf.transpose(a=outputs, perm=[1, 0, 2])

        return outputs, final_state
    

def attention(inputs, 
              activation=None, 
              attention_len=None, 
              kernel_initializer=None,
              bias_initializer=None):
              
    # num_neurons = inputs.shape[2].value
    num_neurons = inputs.shape[2]
    if attention_len == None:
        attention_len = num_neurons

    # Trainable parameters
    Watt = tf.compat.v1.get_variable("Watt", shape=[num_neurons, attention_len], initializer=kernel_initializer)
    batt = tf.compat.v1.get_variable("batt", shape=[attention_len], initializer=bias_initializer)
    uatt = tf.compat.v1.get_variable("uatt", shape=[attention_len], initializer=kernel_initializer)

    # (Batch,Seq,Neuron) * (Neuron,Att) --> (Batch,Seq,Att)
    v = tf.einsum('bsn,na->bsa', inputs, Watt) + batt
    if activation != None:
        v = activation(v)

    # (Batch,Seq,Att) * (Att) --> (Batch, Seq)
    vu = tf.einsum('bsa,a->bs', v, uatt)
    alpha = tf.nn.softmax(vu, name='alpha')

    return tf.reduce_sum(input_tensor=inputs * tf.expand_dims(alpha, -1), axis=1)


def create_rnn_cell(cell_type, num_neurons, use_residual=False):
    ''' Create RNN cell depend on the provided type '''
    cell = None
    if cell_type == "lstm":
        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_neurons)
    elif cell_type == "lstmp":
        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_neurons, use_peepholes=True)        
    elif cell_type == "gru":
        cell = tf.compat.v1.nn.rnn_cell.GRUCell(num_neurons)
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


class SequenceToSequenceGenerator_BiHMPGAN(object):
    '''
    GAN generator network that predict future skeleton poses using sequernce to sequence network.
    Implement the generator in paper BiHMP-GAN 
    '''
    def __init__(self, inputs, hidden_size, input_length, output_length, z=None, embeded_size=None, embedding=None,
                 cell_type='lstm', use_residual=False, bias_initializer=tf.constant_initializer(0.),
                 kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.001),reuse=tf.compat.v1.AUTO_REUSE,scope=""):
        '''
        Initialize the generator.

        Args:
            inputs(tf.placeholder): The input variable containing current data in shape [batch_size, sequence_length, input_dims]
                                    or [batch_size, sequence_length, num_joints, num_params_per_joint] is using skeleten autoencoder.
            z(tf.placeholder): A random generated input vector used as input in shape [batch_size, z_dims].
            hidden_size: number of hidden units in the cell.
            input_length: the length of the input sequence.
            output_length: the length of the output sequence.
            embeded_size: encoder/decoder input embeded size, if not None, the inputs will be projected to embeded_size before feed into rnn cell. 
            cell_type: "lstm" or "gru", the type of cell used in the model.
            use_residual: bool, whether to build decoder in a residual setup.
            bias_initializer: initializer for bias.
            kernel_initializer: initializer for weights.
            reuse: True, None or tf.AUTO_REUSE, whether to reuse parameters from a previously created model.
            scope: variable scope name.
        Possible extensions(TO DO):
            num_layers: the rnn cell can be multi-layers
            bidirectional: the rnn cell can be bidirectional
            variable_lengths: for input and output in variable lengths, at this moment, the sequences in one batch should have same lengths  
            z_None: z can be None to train a determined seq2seq model 
            reverse_input: for 2 way bidirectional training(like in the paper spatial temporal....)
        '''
        self._embedding = embedding
        if self._embedding is not None:
            self._input_dims = self._embedding.z_size
        else:
            self._input_dims = inputs.shape[2]
        if z is not None:
            self._z_dims = z.shape[1]
        self._embeded_size = embeded_size
        self._hidden_size = hidden_size
        self._input_length = input_length
        self._output_length = output_length
        self._cell_type = cell_type
        self._use_residual = use_residual
        self._bias_initializer = bias_initializer
        self._kernel_initializer = kernel_initializer
        self._reuse = reuse
        if scope.strip() == "":
            self._scope = scope
        else:
            self._scope = scope + "/"
        self._outputs = None
        self._parameters = []
        self._enc_parameters = []
        self._dec_parameters = []
        #build the network
        self._build(inputs,z)

    @property
    def parameters(self):
        return self._parameters
    
    @property
    def enc_parameters(self):
        return self._enc_parameters

    @property
    def dec_parameters(self):
        return self._dec_parameters
    
    @property
    def outputs(self):
        return self._outputs
    
    def connect(self, inputs, z):
        self._build(inputs, z)
        return self._outputs

    def _build(self, inputs, z):
        with tf.variable_scope(self._scope + self.__class__.__name__, reuse=self._reuse) as vs:
            encoder_inputs, outputs, encoder_state = self._build_rnn_encoder(inputs)
            Xt = encoder_inputs[:,-1,:] 
            if self._cell_type == 'lstm':
                final_state = encoder_state.h 
            elif self._cell_type == 'gru':
                final_state = encoder_state
            if z is not None:
                first_input = tf.concat(values=[Xt, final_state, z], axis=1)
                #first_input = Xt
            else:
                first_input = tf.concat(values=[Xt, final_state], axis=1)
                #first_input = Xt
            if self._use_residual:
                self._outputs = self._build_rnn_decoder(first_input, z, encoder_state, Xt)
            else:
                self._outputs = self._build_rnn_decoder(first_input, z, encoder_state)
            self._parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name+'/')
            self._enc_parameters = [v for v in self._parameters if 'encoder' in v.name]
            self._dec_parameters = [v for v in self._parameters if 'decoder' in v.name]

    def _input_projection(self, inputs):
        if self._embeded_size: 
            Wi1 = tf.get_variable("Wi1", shape=[self._input_dims, self._embeded_size], initializer=self._kernel_initializer) 
            bi1 = tf.get_variable("bi1", shape=[self._embeded_size], initializer=self._bias_initializer)
            encoder_inputs = tf.tensordot(inputs, Wi1, axes=[[2], [0]]) + bi1
            encoder_inputs = tf.nn.relu(encoder_inputs)
            Wi2 = tf.get_variable("Wi2", shape=[self._embeded_size, self._embeded_size*2], initializer=self._kernel_initializer) 
            bi2 = tf.get_variable("bi2", shape=[self._embeded_size*2], initializer=self._bias_initializer)
            encoder_inputs = tf.tensordot(encoder_inputs, Wi2, axes=[[2], [0]]) + bi2
        elif self._embedding is not None:
            inputs = tf.reshape(inputs, shape=[-1]+inputs.shape[2:].as_list())
            self._embedding.build_encoder(inputs)
            encoded = self._embedding.encoded
            encoder_inputs = tf.reshape(encoded, shape=[-1, self._input_length, self._input_dims])
        else:
            encoder_inputs = inputs
        return encoder_inputs
    
    def _get_a_cell(self, num_neuron):
        if self._cell_type == 'lstm':
            cell = tf.nn.rnn_cell.LSTMCell(num_neuron)
        elif self._cell_type == 'gru':
            cell = tf.nn.rnn_cell.GRUCell(num_neuron)
        return cell

    def _build_rnn_encoder(self, inputs):
        with tf.variable_scope("encoder", reuse=self._reuse):
            encoder_inputs = self._input_projection(inputs)
            cell = self._get_a_cell(self._hidden_size)
            # 'outputs' is a Tensor in shaped: [batch_size, time_steps, hidden_size]
            # 'state' is the final state, if LSTMcell, it will be StateTuple (c=[batch_size, c_size], h=[batch_size, h_size])
            # else it will be in shape of [batch_size, state_size]
            # the raw output's last valid element is the same value as the h state  
            outputs, state = tf.nn.dynamic_rnn(cell, encoder_inputs, dtype=tf.float32)
            return encoder_inputs, outputs, state

    def _output_projection(self, inputs):
        if self._embedding is not None:
            inputs = tf.reshape(inputs, shape=[-1]+inputs.shape[2:].as_list())
            self._embedding.build_decoder(inputs)
            decoded = self._embedding.decoded
            decoder_outputs = tf.reshape(decoded, shape=[-1, self._output_length]+decoded.shape[1:].as_list())
        elif self._embeded_size: 
            Wlast = tf.get_variable("Wlast", shape=[self._embeded_size*2, self._input_dims], initializer=self._kernel_initializer) 
            blast = tf.get_variable("blast", shape=[self._input_dims], initializer=self._bias_initializer)
            decoder_outputs = tf.tensordot(inputs, Wlast, axes=[[2], [0]]) + blast
        else:
            decoder_outputs = inputs
        return decoder_outputs

    def _build_rnn_decoder(self, first_input, z, encoder_state, last_X=None):
        with tf.variable_scope("decoder", reuse=self._reuse):
            if z is not None:
                cell = self._get_a_cell(self._hidden_size + self._z_dims)
                cell_size = self._hidden_size + self._z_dims
            else:
                cell = self._get_a_cell(self._hidden_size)
                cell_size = self._hidden_size
            outputs, _ = self._dynamic_rnn_decoder(cell, first_input, z, encoder_state, self._output_length, cell_size, last_X)
            decoder_outputs = self._output_projection(outputs)
            return decoder_outputs

    def _dynamic_rnn_decoder(self, cell, first_input, z, encoder_state, sequence_length, cell_size, last_X=None):
        if self._embeded_size:
            # From output state to input embed.
            Wsi = tf.get_variable("Wsi", shape=[first_input.shape[-1], self._embeded_size], initializer=self._kernel_initializer)
            first_input = tf.matmul(first_input, Wsi)
        input_depth = first_input.shape[-1]
        zeros_input = tf.fill(tf.stack([tf.shape(first_input)[0], input_depth]), 0.0)
        if self._embeded_size:
            embeded_dims = self._embeded_size*2
        else:
            embeded_dims = self._input_dims
        emit_structure = tf.zeros([embeded_dims], tf.float32)
        Woi = tf.get_variable("Woi",shape=[cell_size, embeded_dims], initializer=self._kernel_initializer)
        if z is not None:
            if self._cell_type == 'lstm':
                c_state = tf.concat(values=[encoder_state.c, z], axis=1)
                h_state = tf.concat(values=[encoder_state.h, z], axis=1)
                encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c_state, h_state)
            elif self._cell_type == 'gru':
                encoder_state = tf.concat(values=[encoder_state, z], axis=1)
        def get_next_input(cell_output):
            if self._cell_type == "lstm":
                next_input = tf.concat(values=[cell_output, encoder_state.h], axis=1)
            elif self._cell_type == 'gru':
                next_input = tf.concat(values=[cell_output, encoder_state], axis=1)
            if self._embeded_size:
                next_input = tf.matmul(next_input, Wsi)
            return next_input

        def loop_fn(time, cell_output, cell_state, loop_state):
            nonlocal last_X
            if cell_output is None: #when time=0
                elements_finished = (sequence_length <= 0)
                next_input = first_input
                next_cell_state = encoder_state
                emit_output = emit_structure
                next_loop_state = None
            else:
                #emit_output = cell_output #use residual: delta x
                if self._use_residual:
                    emit_output = tf.matmul(cell_output, Woi) + last_X
                else:
                    emit_output = tf.matmul(cell_output, Woi)
                next_loop_state = None
                next_cell_state = cell_state
                elements_finished = (time >= sequence_length)
                finished = tf.reduce_all(elements_finished)
                next_input = tf.cond(
                    finished,
                    lambda: zeros_input,
                    lambda: get_next_input(emit_output))
                if self._use_residual:
                    last_X = tf.matmul(cell_output, Woi) + last_X
            return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)
        outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
        outputs = outputs_ta.stack()
        outputs = tf.transpose(outputs, perm=[1,0,2])
        return outputs, final_state


class NNGenerator(object):
    '''
    GAN generator network that predict a single future skeleton pose using feedforward neural network.
    '''
    def __init__(self, inputs, inputs_depth, z, reuse=False):
        '''
        Initialize the generative network.

        Args:
            inputs(tf.placeholder): The input variable containing current data.
            inputs_depth(int): input embed size.
            reuse(bool): True to reuse model parameters from a previously created model.
        '''
        self._reuse = reuse
        self._batch_size = inputs.shape[0] # batch_size
        self._inputs_depth = inputs_depth
        self._inputs_shape = inputs.shape
        self._element_shape = [np.prod(inputs.shape[2:].as_list()), inputs_depth]
        self._output = None
        self._parameters = []
        self._weights = []
        self._stddev = 0.001
        self._build(inputs, z)

    @property
    def output(self):
        return self._output

    @property
    def parameters(self):
        ''' All trainable parameters '''
        return self._parameters

    @property
    def weights(self):
        ''' Weights only parameters for regularization '''
        return self._weights

    def _build(self, inputs, z):
        '''
        Construct a generative model.

        Args:
            inputs(tf.placeholder): The input variable containing current data.
        '''
        with tf.compat.v1.variable_scope(self.__class__.__name__, reuse=self._reuse) as vs:

            num_neurons = 1024
            num_layers = 3

            Wi = tf.compat.v1.get_variable("Wi", initializer=tf.random.truncated_normal(self._element_shape, stddev=self._stddev))
            inputs = tf.reshape(inputs, shape=[-1, inputs.shape[1].value]+[np.prod(inputs.shape[2:].as_list())])

            d_inputs = tf.tensordot(inputs, Wi, axes=[[2], [0]])
            d_inputs.set_shape([None, inputs.shape[1].value, self._element_shape[-1]]) # https://github.com/tensorflow/tensorflow/issues/6682

            net = tf.reshape(d_inputs, [-1, np.prod(d_inputs.shape[1:].as_list())])

            for _ in range(num_layers):
                net = tf.compat.v1.layers.dense(inputs=net, units=num_neurons, activation=tf.nn.relu, reuse=self._reuse)
                net = tf.compat.v1.layers.dropout(inputs=net, rate=0.5)

            pred = tf.compat.v1.layers.dense(inputs=net, units=self._element_shape[0], reuse=self._reuse)

            # pred = tf.reshape(pred, pred.shape[:1].as_list() + [1] + self._inputs_shape[2:].as_list())

            self._output = tf.tanh(pred)
            self._parameters = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name+'/')
            self._weights = [v for v in self._parameters if (v.name.endswith('Wi:0') or v.name.endswith('weights:0') or v.name.endswith('kernel:0'))]


class SequenceToSequenceGenerator(object):
    '''
    GAN generator network that predict future skeleton poses using sequernce to sequence network.
    '''
    def __init__(self, inputs, inputs_depth, z, input_sequence_length, output_sequence_length, 
                 cell_type='gru', project_to_rnn_output=False, reverse_input=False,
                 use_attention=False, use_residual=False,
                 bias_initializer=tf.constant_initializer(0.), kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.001),
                 reuse=False):
        '''
        Initialize the generative network.

        Args:
            inputs(tf.placeholder): The input variable containing current data.
            inputs_depth(int): input embed size.
            z(tf.placeholder, optional): A random generated input vector used as input.
            input_sequence_length(int): the length of the input sequence.
            output_sequence_length(int): the length of the resulted sequence.
            cell_type(str): The type of cell to use for the encode and decoder.
            project_to_rnn_output(bool): project the input to the number of hidden unit in the RNN.
            reverse_input(bool): reverse the input sequence before feeding it to the network.
            use_attention(bool): true to use attention instead of the last state of the encoder.
            use_residual(bool): use resent like structure for the recurrent.
            bias_initializer: initializer for the bias value.
            kernel_initializer: initializer for the `W` parameters.            
            reuse(bool): True to reuse model parameters from a previously created model.
        '''
        self._reuse = reuse
        self._batch_size = tf.shape(inputs)[0] # batch_size
        self._input_sequence_length = input_sequence_length
        self._output_sequence_length = output_sequence_length
        self._inputs_depth = inputs_depth
        self._inputs_shape = inputs.shape
        self._element_shape = inputs.shape[2:].as_list()
        self._output = None
        self._parameters = []
        self._weights = []
        self._num_neurons = 1024
        self._num_layers = 2
        self._num_nn_layers = 2
        self._cell_type = cell_type
        self._bias_initializer = bias_initializer
        self._kernel_initializer = kernel_initializer
        self._reccurent_bias_initializer = None
        self._reccurent_kernel_initializer = None
        self._project_to_rnn_output = project_to_rnn_output
        self._use_attention = use_attention
        self._use_residual = use_residual

        if self._use_residual:
            self._project_to_rnn_output = True

        # Similar to tf.zeros but support variable batch size.
        if self._project_to_rnn_output:
            self._zeros_input = tf.fill(tf.stack([tf.shape(inputs)[0], self._num_neurons]), 0.0)
        else:
            self._zeros_input = tf.fill(tf.stack([tf.shape(inputs)[0], self._inputs_depth]), 0.0)

        if reverse_input:
            inputs = tf.reverse(inputs, axis=[1])
        self._build(inputs, z)

    @property
    def output(self):
        return self._output

    @property
    def parameters(self):
        ''' All trainable parameters '''
        return self._parameters

    @property
    def weights(self):
        ''' Weights only parameters for regularization '''
        return self._weights

    def _build(self, inputs, z):
        '''
        Construct a generative model.

        Args:
            inputs(tf.placeholder): The input variable containing current data.
            z(tf.placeholder): A vector containss the randomly generated latent data.
        '''
        with tf.compat.v1.variable_scope(self.__class__.__name__, reuse=self._reuse) as vs:

            outputs, encoder_state = self._build_encoder(inputs)

            first_input = outputs[:, -1, :] # [batch, sequence, elements]
            # if self._use_attention:
            #     encoder_state = attention(outputs, 
            #                                  kernel_initializer=self._kernel_initializer,
            #                                  bias_initializer=self._bias_initializer)
            self._output = self._build_decoder(first_input, z, encoder_state)

            self._parameters = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name+'/')
            self._weights = [v for v in self._parameters if (v.name.endswith('Wi:0') or \
                                                             v.name.endswith('weights:0') or \
                                                             v.name.endswith('Wo:0') or \
                                                             v.name.endswith('Wsi:0') or \
                                                             ('Wzi' in v.name) or \
                                                             ('Wzci' in v.name) or \
                                                             ('Wzhi' in v.name))]

    def _create_rnn_model(self):
        ''' Create RNN model '''
        return create_rnn_model(self._num_layers, 
                                   self._cell_type, 
                                   self._num_neurons, 
                                   use_residual=self._use_residual)

    def _input_projection(self, inputs):
        ''' Project each skeleton pose to the encoder. '''

        inputs = tf.reshape(inputs, shape=[-1, inputs.shape[1].value]+[np.prod(inputs.shape[2:].as_list())])
        if self._project_to_rnn_output:
            net = inputs
            layer_index = 0
            num_neurons = self._num_neurons // (self._num_nn_layers+1)
            for i in range(self._num_nn_layers):
                net = tf.compat.v1.layers.dense(inputs=net, 
                                      units=(i+1)*num_neurons,
                                      kernel_initializer=self._kernel_initializer,
                                      bias_initializer=self._bias_initializer,
                                      activation=tf.nn.relu,
                                      name="fc{}".format(layer_index+1))
                # net = tf.layers.dropout(inputs=net, rate=0.5)
                layer_index += 1

            encoder_inputs = tf.compat.v1.layers.dense(inputs=net, 
                                             units=self._num_neurons,
                                             kernel_initializer=self._kernel_initializer,
                                             bias_initializer=self._bias_initializer,
                                             activation=None, 
                                             name="fc{}".format(layer_index+1))
            encoder_inputs.set_shape([inputs.shape[0].value, inputs.shape[1].value, self._num_neurons])
        else:
            Wi = tf.compat.v1.get_variable("Wi", shape=[np.prod(self._element_shape), self._inputs_depth], initializer=self._kernel_initializer)
            bi = tf.compat.v1.get_variable("bi", shape=[self._inputs_depth], initializer=self._bias_initializer)

            encoder_inputs = tf.tensordot(inputs, Wi, axes=[[2], [0]])
            encoder_inputs.set_shape([inputs.shape[0].value, inputs.shape[1].value, self._inputs_depth]) # https://github.com/tensorflow/tensorflow/issues/6682
            encoder_inputs = encoder_inputs + bi

        return encoder_inputs

    def _output_projection(self, outputs):
        ''' Project each decoder output back to skeleton pose. '''

        if self._project_to_rnn_output:
            net = outputs
            layer_index = 0
            for i in range(self._num_nn_layers):
                net = tf.compat.v1.layers.dense(inputs=net, 
                                      units=int(self._num_neurons/(i+1)),
                                      kernel_initializer=self._kernel_initializer,
                                      bias_initializer=self._bias_initializer,
                                      activation=tf.nn.relu, 
                                      name="fc{}".format(layer_index+1))
                # net = tf.layers.dropout(inputs=net, rate=0.5)
                layer_index += 1

            pred = tf.compat.v1.layers.dense(inputs=net, 
                                   units=np.prod(self._element_shape),
                                   kernel_initializer=self._kernel_initializer,
                                   bias_initializer=self._bias_initializer,
                                   activation=None, 
                                   name="fc{}".format(layer_index+1))

            pred.set_shape([outputs.shape[0].value, self._output_sequence_length, np.prod(self._element_shape)])
            pred = tf.reshape(pred, shape=[-1, pred.shape[1].value] + self._inputs_shape[2:].as_list())
        else:
            Wo = tf.compat.v1.get_variable("Wo", shape=[self._num_neurons, np.prod(self._element_shape)], initializer=self._kernel_initializer)
            bo = tf.compat.v1.get_variable("bo", shape=[np.prod(self._element_shape)], initializer=self._bias_initializer)

            pred = tf.tensordot(outputs, Wo, axes=[[2], [0]])
            pred.set_shape([outputs.shape[0].value, self._output_sequence_length, np.prod(self._element_shape)]) # https://github.com/tensorflow/tensorflow/issues/6682
            pred = pred + bo
            pred = tf.reshape(pred, shape=[-1, pred.shape[1].value] + self._inputs_shape[2:].as_list())

        return pred

    def _build_encoder(self, inputs):
        ''' Build the encoder part of the generative mode. '''
        with tf.compat.v1.variable_scope("encoder", reuse=self._reuse):
            encoder_inputs = self._input_projection(inputs)
            print("projected encoder shape: {} ".format(encoder_inputs.shape))
            cell = self._create_rnn_model()
            outputs, state = tf.compat.v1.nn.dynamic_rnn(cell, encoder_inputs, dtype=tf.float32)
            print("after dynamic rnn shape: {} ".format(outputs.shape))
            return outputs, state

    def _build_decoder(self, first_input, z, encoder_state):
        '''
        Build the decoder part of the generative mode. It can decode based on the initial state without
        the need of future_inputs.

        Args:
            first_input(tf.placeholder, optional): each cell takes input form the output of the previous cell,
                                                   except first cell. first_input is used for the first cell.
            z(tf.placeholder, optional): random vector in order to sample multiple predictions from the 
                                         same input.
            encoder_state(cell state): the last state of the encoder.

        Return:
            The output of the network.
        '''
        with tf.compat.v1.variable_scope("decoder", reuse=self._reuse):
            cell = self._create_rnn_model()
            print("the input for rnn decoder: {}".format(first_input.shape))
            outputs, _ = self._dynamic_rnn_decoder(cell, first_input, z, encoder_state, self._output_sequence_length)
            print("the output from rnn decoder: {}".format(outputs.shape))
            return self._output_projection(outputs)

    def _dynamic_rnn_decoder(self, cell, first_input, z, encoder_state, sequence_length, time_major=False, dtype=tf.float32):
        ''' Unroll the RNN decoder '''
        if not self._project_to_rnn_output:
            # From output state to input embed.
            Wsi = tf.compat.v1.get_variable("Wsi", 
                                  shape=[self._num_neurons, self._inputs_depth], 
                                  initializer=self._kernel_initializer)

        if first_input is None:
            first_input = self._zeros_input
        print("first input shape before {}".format(first_input.shape))
        first_input = first_input if self._project_to_rnn_output else tf.matmul(first_input, Wsi)
        print('first input shape after {}'.format(first_input.shape))
        if z is not None:

            is_tuple = isinstance(encoder_state[0], tf.compat.v1.nn.rnn_cell.LSTMStateTuple) if (self._num_layers > 1) else isinstance(encoder_state, tf.compat.v1.nn.rnn_cell.LSTMStateTuple)
            if is_tuple:
                if self._num_layers > 1:
                    states = []
                    for i in range(self._num_layers):
                        Wzhi = tf.compat.v1.get_variable("Wzhi{}".format(i), shape=[z.shape.as_list()[-1], encoder_state[i].h.shape.as_list()[-1]], initializer=self._kernel_initializer)
                        Wzci = tf.compat.v1.get_variable("Wzci{}".format(i), shape=[z.shape.as_list()[-1], encoder_state[i].c.shape.as_list()[-1]], initializer=self._kernel_initializer)                        
                        states.append(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(encoder_state[i].c + tf.matmul(z, Wzci), encoder_state[i].h + tf.matmul(z, Wzhi)))
                    encoder_state = tuple(states)
                else:
                    Wzhi = tf.compat.v1.get_variable("Wzhi", shape=[z.shape.as_list()[-1], encoder_state.h.shape.as_list()[-1]], initializer=self._kernel_initializer)
                    Wzci = tf.compat.v1.get_variable("Wzci", shape=[z.shape.as_list()[-1], encoder_state.c.shape.as_list()[-1]], initializer=self._kernel_initializer)      
                    encoder_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(encoder_state.c + tf.matmul(z, Wzci), encoder_state.h + tf.matmul(z, Wzhi))
            else:
                if self._num_layers > 1:
                    states = []
                    for i in range(self._num_layers):
                        Wzi = tf.compat.v1.get_variable("Wzi{}".format(i), shape=[z.shape.as_list()[-1], encoder_state[i].shape.as_list()[-1]], initializer=self._kernel_initializer)
                        states.append(encoder_state[i] + tf.matmul(z, Wzi))
                    encoder_state = tuple(states)
                else:
                    Wzi = tf.compat.v1.get_variable("Wzi", shape=[z.shape.as_list()[-1], encoder_state.shape.as_list()[-1]], initializer=self._kernel_initializer)
                    encoder_state = encoder_state + tf.matmul(z, Wzi)

        def loop_fn_init(time):
            elements_finished = (sequence_length <= 0)
            next_input = first_input
            next_cell_state = encoder_state
            emit_output = None
            next_loop_state = None

            return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

        def loop_fn_next(time, cell_output, cell_state, loop_state):
            emit_output = cell_output
            next_cell_state = cell_state

            elements_finished = (time >= sequence_length)
            finished = tf.reduce_all(elements_finished)
            next_input = tf.cond(
                finished,
                lambda: self._zeros_input,
                lambda: cell_output if self._project_to_rnn_output else tf.matmul(cell_output, Wsi))
            next_loop_state = None
            return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is None:
                return loop_fn_init(time)
            else:
                return loop_fn_next(time, cell_output, cell_state, loop_state)

        outputs_ta, final_state, _ = tf.compat.v1.nn.raw_rnn(cell, loop_fn)
        outputs = outputs_ta.stack()
        print("output from  raw_rnn {}".format(outputs.shape))

        tf.compat.v1.nn.dynamic_rnn()
        if not time_major:
            outputs = tf.transpose(outputs, perm=[1, 0, 2])

        return outputs, final_state


def test():
    element_size = 87
    input_sequence_length = 40
    inputs = tf.compat.v1.placeholder(tf.float32, shape=(None, input_sequence_length, element_size))
    embedding_size = 10
    g = NNGenerator(inputs, embedding_size, input_sequence_length)
    res = g.output
    print(res.shape)


def test_SequenceToSequenceGenerator():
    element_size = 87
    input_sequence_length = 40
    output_sequence_length = 20
    g_inputs = tf.compat.v1.placeholder(tf.float32, shape=(None, input_sequence_length, element_size))
    embedding_size = 10
    inputs_depth = 128
    z_size = 128
    g_z = tf.compat.v1.placeholder(tf.float32, shape=(None, z_size))
    g = SequenceToSequenceGenerator(g_inputs,
                                    inputs_depth,
                                    g_z, 
                                    input_sequence_length, 
                                    output_sequence_length,
                                    reverse_input=False)
    print(g.output.shape)                            



if __name__ == "__main__":
    # test()
    test_SequenceToSequenceGenerator()