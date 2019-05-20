# encoding: UTF-8
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python import framework
from tensorflow.python.layers.convolutional import Conv2D


class Conv2DTranspose(Conv2D):
  """Transposed 2D convolution layer (sometimes called 2D Deconvolution).

  The need for transposed convolutions generally arises
  from the desire to use a transformation going in the opposite direction
  of a normal convolution, i.e., from something that has the shape of the
  output of some convolution to something that has the shape of its input
  while maintaining a connectivity pattern that is compatible with
  said convolution.

  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: A tuple or list of 2 positive integers specifying the spatial
      dimensions of the filters. Can be a single integer to specify the same
      value for all spatial dimensions.
    strides: A tuple or list of 2 positive integers specifying the strides
      of the convolution. Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: one of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, no bias will
      be applied.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
  """

  def __init__(self, filters,
               kernel_size,
               strides=(1, 1),
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
               **kwargs):
    super(Conv2DTranspose, self).__init__(
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
        **kwargs)
    self.input_spec = base.InputSpec(ndim=4)

  def build(self, input_shape):
    if len(input_shape) != 4:
      raise ValueError('Inputs should have rank ' +
                       str(4) +
                       'Received input shape:', str(input_shape))
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis]
    self.input_spec = base.InputSpec(ndim=4, axes={channel_axis: input_dim})
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
      c_axis, h_axis, w_axis = 1, 2, 3
    else:
      c_axis, h_axis, w_axis = 3, 1, 2

    height, width = inputs_shape[h_axis], inputs_shape[w_axis]
    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides

    # Infer the dynamic output shape:
    print("##############################")
    print(width)
    print(kernel_w)
    print(stride_w)
    out_width = utils.deconv_output_length(width,
                                           kernel_w,
                                           self.padding,
                                           stride_w)

    if self.data_format == 'channels_first':
      output_shape = (batch_size, self.filters, out_width)
      # strides = (1, 1, stride_h, stride_w)
    else:
      output_shape = (batch_size, out_width, self.filters)
      # strides = (1, stride_h, stride_w, 1)
    print("#####################")
    print(batch_size.shape)
    print(out_width.shape)
    print(self.filters.shape)
    output_shape_tensor = array_ops.stack(output_shape)
    # outputs = nn.conv2d_transpose(
    #     inputs,
    #     self.kernel,
    #     output_shape_tensor,
    #     strides,
    #     padding=self.padding.upper(),
    #     data_format=utils.convert_data_format(self.data_format, ndim=4))
    #
    # # Infer the static output shape:
    # out_shape = inputs.get_shape().as_list()
    # out_shape[c_axis] = self.filters
    # out_shape[h_axis] = utils.deconv_output_length(out_shape[h_axis],
    #                                                kernel_h,
    #                                                self.padding,
    #                                                stride_h)
    # out_shape[w_axis] = utils.deconv_output_length(out_shape[w_axis],
    #                                                kernel_w,
    #                                                self.padding,
    #                                                stride_w)
    # outputs.set_shape(out_shape)
    #
    # if self.bias:
    #   outputs = nn.bias_add(
    #       outputs,
    #       self.bias,
    #       data_format=utils.convert_data_format(self.data_format, ndim=4))
    #
    # if self.activation is not None:
    #   return self.activation(outputs)
    # return outputs

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    output_shape = list(input_shape)
    if self.data_format == 'channels_first':
      c_axis, h_axis, w_axis = 1, 2, 3
    else:
      c_axis, h_axis, w_axis = 3, 1, 2

    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides

    output_shape[c_axis] = self.filters
    output_shape[h_axis] = utils.deconv_output_length(
        output_shape[h_axis], kernel_h, self.padding, stride_h)
    output_shape[w_axis] = utils.deconv_output_length(
        output_shape[w_axis], kernel_w, self.padding, stride_w)
    return tensor_shape.TensorShape(output_shape)