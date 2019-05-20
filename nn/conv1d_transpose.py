# encoding: UTF-8

from tensorflow.python.layers.convolutional import Conv1D
from tensorflow.python.ops import init_ops
from tensorflow.python.layers import base
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
from tensorflow.python.ops import nn
import tensorflow as tf
from tensorflow.python.framework import tensor_shape


class Conv1DTranspose(Conv1D):

    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format='channels_last',
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 ** kwargs):
        super(Conv1DTranspose, self).__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            trainable=trainable,
            name=name,
            **kwargs
        )
        self.input_spec = base.InputSpec(ndim=3)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Inputs should have rank ' +
                             str(3) +
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        self.input_spec = base.InputSpec(ndim=3, axes={channel_axis: input_dim})
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel = self.add_variable(name='kernel',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        trainable=True,
                                        dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_variable(name='bias',
                                          shape=(self.filters,),
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          trainable=True,
                                          dtype=self.dtype)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        inputs_shape = array_ops.shape(inputs)
        batch_size = inputs_shape[0]

        if self.data_format == 'channels_first':
            c_axis, w_axis = 1, 2
        else:
            c_axis, w_axis = 2, 1

        width = inputs_shape[w_axis]
        kernel_w = self.kernel_size[0]
        stride_w = self.strides[0]

        # Infer the dynamic output shape:
        out_width = utils.deconv_output_length(width,
                                               kernel_w,
                                               self.padding,
                                               stride_w)
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_width)
        else:
            output_shape = (batch_size, out_width, self.filters)

        output_shape_tensor = array_ops.stack(output_shape)
        outputs = tf.contrib.nn.conv1d_transpose(
            inputs,
            self.kernel,
            output_shape_tensor,
            stride_w,
            padding=self.padding.upper(),
            data_format=utils.convert_data_format(self.data_format, ndim=3))

        # Infer the static output shape:
        out_shape = inputs.get_shape().as_list()
        out_shape[c_axis] = self.filters
        out_shape[w_axis] = utils.deconv_output_length(out_shape[w_axis],
                                                       kernel_w,
                                                       self.padding,
                                                       stride_w)
        outputs.set_shape(out_shape)

        if self.bias:
            if self.data_format == 'channels_first':
                bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                bias = array_ops.reshape(self.bias, (1, 1, self.filters))
                outputs += bias

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        output_shape = list(input_shape)
        if self.data_format == 'channels_first':
            c_axis, w_axis = 1, 2
        else:
            c_axis, w_axis = 2, 1

        output_shape[c_axis] = self.filters
        output_shape[w_axis] = utils.deconv_output_length(
            output_shape[w_axis], self.kernel_size, self.padding, self.strides)
        return tensor_shape.TensorShape(output_shape)
