# encoding: UTF-8
import tensorflow as tf
import numpy as np


def spectrum_pooling_1d(inputs, pool_size, N=512, data_format='channels_first', name='spectrum_pooling'):
    '''
    
    :param inputs: batchsize * n_channels * n_length if data_format is channels_first, batchsize * n_length * n_channels
                   if data_format is channels_last
    :param pool_size: int, decides the ratio of spectrum to keep, the first 1/pooling_size will be kept
    :param N: 
    :return: 
    '''
    with tf.compat.v1.variable_scope(name):
        if data_format == 'channels_last':
            inputs = tf.transpose(a=inputs, perm=[0, 2, 1])
        assert pool_size % 2 ==0
        ## extend input size to length of N
        N = tf.convert_to_tensor(value=N)

        # left = tf.reverse(inputs[:, :, :((N-inputs.shape[2])//pooling_size)], axis=-1)
        # right = tf.reverse(inputs[:, :, -(N - inputs.shape[2] - left.shape[2]):], axis=-1)
        if inputs.shape[2] % 2 == 0:
            left = inputs[:, :, :((N-inputs.shape[2])//2)]
            right = inputs[:, :, -(N - inputs.shape[2])//2:] 
        else:
            left = inputs[:, :, :((N-inputs.shape[2])//2)] 
            right = inputs[:, :, -(N - inputs.shape[2])//2-1:] 
        left = tf.reverse(left, axis=[-1])
        right = tf.reverse(right, axis=[-1])
        extended_input = tf.concat([left, inputs, right], axis=-1)

        input_fft = tf.signal.fft(tf.complex(extended_input, extended_input * 0.0))
        ## spectrum pooling

        outputs_fft = tf.concat([input_fft[:, :, :N//(2*pool_size)],
                                 tf.zeros(shape=[inputs.shape[0], inputs.shape[1], 1], dtype=tf.complex64),
                                 input_fft[:, :, -N//(2*pool_size)+1:]],
                                 axis=-1)
        # outputs_fft = input_fft
        outputs_fft = outputs_fft / pool_size
        outputs = tf.math.real(tf.signal.ifft(outputs_fft))[:, :, (N - inputs.shape[2])//(2*pool_size):
                                                     (N + inputs.shape[2])//(2*pool_size)]
        outputs.set_shape((inputs.shape[0], inputs.shape[1], inputs.shape[2]//2))
        # outputs_fft.set_shape((inputs.shape[0], inputs.shape[1], 512))
        return outputs


def spectrum_pooling_1d_1(inputs, pool_size, data_format='channels_first', name='spectrum_pooling'):
    '''

    :param inputs: batchsize * n_channels * n_length if data_format is channels_first, batchsize * n_length * n_channels
                   if data_format is channels_last
    :param pool_size: int, decides the ratio of spectrum to keep, the first 1/pooling_size will be kept
    :param N:
    :return:
    '''
    with tf.compat.v1.variable_scope(name):
        if data_format == 'channels_last':
            inputs = tf.transpose(a=inputs, perm=[0, 2, 1])
        assert pool_size % 2 == 0
        input_shape = inputs.get_shape().as_list()
        input_fft = tf.signal.fft(tf.complex(inputs, inputs * 0.0))
        ## spectrum pooling
        outputs_fft = tf.concat([input_fft[:, :, :input_shape[2] // (2 * pool_size)+1 ],
                                 input_fft[:, :, -(input_shape[2] // (2 * pool_size))+1:]],
                                axis=-1)
        outputs_fft = outputs_fft / pool_size
        outputs = tf.math.real(tf.signal.ifft(outputs_fft))
        return outputs


def test_spectrum_pooling():
    stylized_data = np.load(r'../../theano/data/training_data/processed_stylized_data.npz')['clips']
    print(stylized_data.shape)
    X = np.swapaxes(stylized_data, 1, 2)
    input = tf.compat.v1.placeholder(shape=(1, 70, 240), dtype=tf.float32)
    pooling_op = spectrum_pooling_1d(input, pool_size=2)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        res = sess.run(pooling_op, feed_dict={input: X[0:1]})
        print(res.shape)


if __name__ == "__main__":
    test_spectrum_pooling()