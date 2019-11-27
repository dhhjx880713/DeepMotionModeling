# encoding: UTF-8
import tensorflow as tf
from tensorflow.python.ops import array_ops


def average_unpooling_1d(inputs, pool_size, data_format='channels_first', name='average_unpooling_1d'):
    with tf.compat.v1.variable_scope(name):
        if data_format == 'channels_first':
            reshaped_inputs = tf.expand_dims(inputs, -1)
            #### convert data to channels last
            reshaped_inputs = tf.transpose(a=reshaped_inputs, perm=[0, 2, 3, 1])
            new_h = reshaped_inputs.shape[1] * pool_size
            output_shape_tensor = array_ops.stack((new_h, reshaped_inputs.shape[2]))
            outputs = tf.image.resize(reshaped_inputs, output_shape_tensor, name=name, method=tf.image.ResizeMethod.BILINEAR)
            outputs = tf.transpose(a=outputs, perm=[0, 3, 1, 2])
            outputs = tf.squeeze(outputs, -1)
            return outputs

        else:
            reshaped_inputs = tf.expand_dims(inputs, 2)
            new_h = reshaped_inputs.shape[1] * pool_size
            output_shape_tensor = array_ops.stack((new_h, reshaped_inputs.shape[2]))
            outputs = tf.image.resize(reshaped_inputs, output_shape_tensor, name=name, method=tf.image.ResizeMethod.BILINEAR)
            outputs = tf.squeeze(outputs, 2)
            return outputs


def spectrum_unpooling_1d(inputs, pool_size, N=512, data_format='channels_first', name='spectrum_unpooling_id'):
    with tf.compat.v1.variable_scope(name):
        assert pool_size % 2 == 0
        N = tf.convert_to_tensor(value=N)
        inputs = inputs * pool_size
        if data_format == 'channels_last':
            inputs = tf.transpose(a=inputs, perm=[0, 2, 1])
        ## extend inputs size to length of N/2
        if inputs.shape[2] % 2 == 0:
            left = inputs[:, :, :((N//2-inputs.shape[2])//2)]
            right = inputs[:, :, -(N//2 - inputs.shape[2])//2:]
        else:
            left = inputs[:, :, :((N//2-inputs.shape[2])//2)]
            right = inputs[:, :, -(N//2 - inputs.shape[2])//2-1:]
        left = tf.reverse(left, axis=[-1])
        right = tf.reverse(right, axis=[-1])
        extended_input = tf.concat([left, inputs, right], axis=-1)  ### lenght is N//2
        input_fft = tf.signal.fft(tf.complex(extended_input, extended_input*0.0))
        # input_fft = inputs
        outputs_fft = tf.concat([input_fft[:, :, :N//4],
                                 tf.zeros(shape=[inputs.shape[0], inputs.shape[1], N//2], dtype=tf.complex64),
                                 input_fft[:, :, -N//4:]],
                                axis=-1)
        # outputs_fft = input_fft
        outputs = tf.math.real(tf.signal.ifft(outputs_fft)[:, :, (N - inputs.shape[2] * pool_size)//2:
                                                     (N + inputs.shape[2] * pool_size)//2])
        outputs.set_shape((inputs.shape[0], inputs.shape[1], inputs.shape[2] * pool_size))
        return outputs


def spectrum_unpooling_1d_1(inputs, pool_size, data_format='channels_first', name='spectrum_unpooling_id'):
    with tf.compat.v1.variable_scope(name):
        assert pool_size % 2 == 0
        inputs = inputs * pool_size
        if data_format == 'channels_last':
            inputs = tf.transpose(a=inputs, perm=[0, 2, 1])
        input_shape = inputs.get_shape().as_list()
        input_fft = tf.signal.fft(tf.complex(inputs, inputs*0.0))

        outputs_fft = tf.concat([input_fft[:, :, :input_shape[2]//2],
                                 tf.zeros(shape=input_shape, dtype=tf.complex64),
                                 input_fft[:, :, -input_shape[2]//2:]],
                                axis=-1)

        outputs = tf.math.real(tf.signal.ifft(outputs_fft))

        return outputs