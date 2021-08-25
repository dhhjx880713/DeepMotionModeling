# encoding: UTF-8
import tensorflow as tf
import numpy as np
from nn.conv1d_transpose import Conv1DTranspose
from nn.unpooling import average_unpooling_1d, spectrum_unpooling_1d
from nn.spectrum_pooling import spectrum_pooling_1d
regularizer = tf.keras.regularizers.l2(l=0.5 * (0.01))

"""
test models:

1. single layer conv1d, strides=2
2. multi-layer conv1d, strides=2
3. multi-layer conv1d, no strides
4. single layer conv1d, no strides
5 spectrum conv1d encoder
"""

def motion_encoder_stride(input, dropout_rate=0.25, name='encoder', reuse=tf.compat.v1.AUTO_REUSE):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        layer1_dropout = tf.compat.v1.layers.dropout(input, rate=dropout_rate)
        layer1_conv = tf.compat.v1.layers.conv1d(layer1_dropout, 256, 25, strides=2, padding='same', activation=tf.nn.relu,
                                       data_format='channels_first',
                                       kernel_regularizer=regularizer, bias_regularizer=regularizer)
    return layer1_conv


def motion_decoder_stride(input, n_input, dropout_rate=0.25, name='decoder', reuse=tf.compat.v1.AUTO_REUSE):

    with tf.compat.v1.variable_scope(name, reuse=reuse):
        layer1_dropout = tf.compat.v1.layers.dropout(input, rate=dropout_rate)
        layer1_deconv = Conv1DTranspose(n_input, 45, strides=2, padding='same', activation=tf.nn.relu,
                                        data_format='channels_first', dtype=input.dtype.base_dtype,
                                        kernel_regularizer=regularizer, bias_regularizer=regularizer)
    deconv = layer1_deconv.apply(layer1_dropout)
    return deconv


def motion_decoder_stride2d(input, n_input, dropout_rate=0.25, name='decoder', reuse=tf.compat.v1.AUTO_REUSE):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        input = tf.expand_dims(input, -1)
        input_dropout = tf.compat.v1.layers.dropout(input, rate=dropout_rate)
        deconv = tf.compat.v1.layers.conv2d_transpose(input_dropout, n_input, kernel_size=(25, 1), strides=(2, 1), padding='same',
                                            data_format='channels_first', kernel_regularizer=regularizer,
                                            bias_regularizer=regularizer)
        deconv = tf.squeeze(deconv, -1)
        return deconv


def motion_encoder_stride_multilayers(input, dropout_rate=0.25, name='encoder', reuse=tf.compat.v1.AUTO_REUSE):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        layer1_dropout = tf.compat.v1.layers.dropout(input, rate=dropout_rate)
        layer1_conv = tf.compat.v1.layers.conv1d(layer1_dropout, 256, 25, strides=2, padding='same', activation=tf.nn.relu,
                                       data_format='channels_first',
                                       kernel_regularizer=regularizer, bias_regularizer=regularizer)
        layer2_dropout = tf.compat.v1.layers.dropout(layer1_conv, rate=dropout_rate)
        layer2_conv = tf.compat.v1.layers.conv1d(layer2_dropout, 128, 25, strides=2, padding='same', activation=tf.nn.relu,
                                       data_format='channels_first', kernel_regularizer=regularizer,
                                       bias_regularizer=regularizer)
        layer3_dropout = tf.compat.v1.layers.dropout(layer2_conv, rate=dropout_rate)
        layer3_conv = tf.compat.v1.layers.conv1d(layer3_dropout, 64, 25, strides=2, padding='same', activation=tf.nn.relu,
                                       data_format='channels_first', kernel_regularizer=regularizer,
                                       bias_regularizer=regularizer)
        return layer3_conv


def motion_decoder_stride2d_multilayers(input, n_input, dropout_rate=0.25, name='decoder', reuse=tf.compat.v1.AUTO_REUSE):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        input = tf.expand_dims(input, -1)
        layer1_dropout = tf.compat.v1.layers.dropout(input, rate=dropout_rate)
        layer1_deconv = tf.compat.v1.layers.conv2d_transpose(layer1_dropout, 128, kernel_size=(25, 1), strides=(2, 1),
                                                   padding='same', data_format='channels_first',
                                                   kernel_regularizer=regularizer, bias_regularizer=regularizer)
        layer2_dropout = tf.compat.v1.layers.dropout(layer1_deconv, rate=dropout_rate)
        layer2_deconv = tf.compat.v1.layers.conv2d_transpose(layer2_dropout, 256, kernel_size=(25, 1), strides=(2, 1),
                                                   padding='same', data_format='channels_first',
                                                   kernel_regularizer=regularizer, bias_regularizer=regularizer)
        layer3_dropout = tf.compat.v1.layers.dropout(layer2_deconv, rate=dropout_rate)
        layer3_deconv = tf.compat.v1.layers.conv2d_transpose(layer3_dropout, n_input, kernel_size=(25, 1), strides=(2, 1),
                                                   padding='same', data_format='channels_first',
                                                   kernel_regularizer=regularizer, bias_regularizer=regularizer)
        deconv = tf.squeeze(layer3_deconv, -1)
        return deconv


def motion_decoder_multilayers(input, n_input, dropout_rate=0.25, name='decoder', reuse=tf.compat.v1.AUTO_REUSE,
                               use_regulizer=True):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        if use_regulizer:
            depool_layer1 = average_unpooling_1d(input, pool_size=2, data_format='channels_first')
            dropout_layer1 = tf.compat.v1.layers.dropout(depool_layer1, rate=dropout_rate)
            conv_layer1 = tf.compat.v1.layers.conv1d(dropout_layer1, 128, 25, padding='same', activation=None,
                                           data_format='channels_first', kernel_regularizer=regularizer,
                                           bias_regularizer=regularizer)
            depool_layer2 = average_unpooling_1d(conv_layer1, pool_size=2, data_format='channels_first')
            dropout_layer2 = tf.compat.v1.layers.dropout(depool_layer2, rate=dropout_rate)
            conv_layer2 = tf.compat.v1.layers.conv1d(dropout_layer2, n_input, 25, padding='same', activation=None,
                                           data_format='channels_first', kernel_regularizer=regularizer,
                                           bias_regularizer=regularizer)
            return conv_layer2
        else:
            depool_layer1 = average_unpooling_1d(input, pool_size=2, data_format='channels_first')
            dropout_layer1 = tf.compat.v1.layers.dropout(depool_layer1, rate=dropout_rate)
            conv_layer1 = tf.compat.v1.layers.conv1d(dropout_layer1, 128, 25, padding='same', activation=None,
                                           data_format='channels_first')
            depool_layer2 = average_unpooling_1d(conv_layer1, pool_size=2, data_format='channels_first')
            dropout_layer2 = tf.compat.v1.layers.dropout(depool_layer2, rate=dropout_rate)
            conv_layer2 = tf.compat.v1.layers.conv1d(dropout_layer2, n_input, 25, padding='same', activation=None,
                                           data_format='channels_first')
            return conv_layer2


def motion_encoder_multilayers(input, dropout_rate=0.25, name='encoder', reuse=tf.compat.v1.AUTO_REUSE, use_regulizer=True):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        if use_regulizer:

            dropout_layer1 = tf.compat.v1.layers.dropout(input, rate=dropout_rate)
            conv_layer1 = tf.compat.v1.layers.conv1d(dropout_layer1, 128, 25, padding='same', activation=tf.nn.relu,
                                           data_format='channels_first', kernel_regularizer=regularizer,
                                           bias_regularizer=regularizer)
            pool_layer1 = tf.compat.v1.layers.average_pooling1d(conv_layer1, 2, strides=2, data_format='channels_first')

            dropout_layer2 = tf.compat.v1.layers.dropout(pool_layer1, rate=dropout_rate)
            conv_layer2 = tf.compat.v1.layers.conv1d(dropout_layer2, 256, 25, padding='same', activation=tf.nn.relu,
                                           data_format='channels_first', kernel_regularizer=regularizer,
                                           bias_regularizer=regularizer)
            pool_layer2 = tf.compat.v1.layers.average_pooling1d(conv_layer2, 2, strides=2, data_format='channels_first')
            return pool_layer2
        else:

            dropout_layer1 = tf.compat.v1.layers.dropout(input, rate=dropout_rate)
            conv_layer1 = tf.compat.v1.layers.conv1d(dropout_layer1, 128, 25, padding='same', activation=tf.nn.relu,
                                           data_format='channels_first')
            pool_layer1 = tf.compat.v1.layers.average_pooling1d(conv_layer1, 2, strides=2, data_format='channels_first')

            dropout_layer2 = tf.compat.v1.layers.dropout(pool_layer1, rate=dropout_rate)
            conv_layer2 = tf.compat.v1.layers.conv1d(dropout_layer2, 256, 25, padding='same', activation=tf.nn.relu,
                                           data_format='channels_first')
            pool_layer2 = tf.compat.v1.layers.average_pooling1d(conv_layer2, 2, strides=2, data_format='channels_first')
            return pool_layer2


def motion_encoder_without_pooling(input, dropout_rate=0.25):
    layer1 = tf.compat.v1.layers.dropout(input, rate=dropout_rate)
    layer2 = tf.compat.v1.layers.conv1d(layer1, 256, 45, padding='same', activation=tf.nn.relu, data_format='channels_first',
                              kernel_regularizer=regularizer)
    # layer3 = tf.layers.average_pooling1d(layer2, 2, strides=2)
    return layer2


def motion_decoder_without_pooling(input, n_input, dropout_rate=0.25):
    # unpooling = tf.contrib.keras.layers.UpSampling1D()
    # layer1 = unpooling(input)
    layer2 = tf.compat.v1.layers.dropout(input, rate=dropout_rate)
    layer3 = tf.compat.v1.layers.conv1d(layer2, n_input, 45, padding='same', activation=None, data_format='channels_first',
                              kernel_regularizer=regularizer)
    return layer3


def ccreate_locomotion_regression(input, dropout_rate=0.25):
    layer1 = tf.compat.v1.layers.dropout(input, rate=dropout_rate)
    layer1 = tf.compat.v1.layers.conv1d(layer1, 64, 45, padding='same', activation=tf.nn.relu)

    layer2 = tf.compat.v1.layers.dropout(layer1)
    layer2 = tf.compat.v1.layers.conv1d(layer2, 128, 25, padding='same', activation=tf.nn.relu)

    layer3 = tf.compat.v1.layers.dropout(layer2)
    layer3 = tf.compat.v1.layers.conv1d(layer3, 256, 15, padding='same', activation=None)

    return layer3


def create_footstepper(input, n_output, dropout_rate=0.25):
    layer1 = tf.compat.v1.layers.dropout(input, rate=dropout_rate)
    layer1 = tf.compat.v1.layers.conv1d(layer1, 64, 65, padding='same', activation=tf.nn.relu)

    layer2 = tf.compat.v1.layers.dropout(layer1, rate=dropout_rate)
    layer2 = tf.compat.v1.layers.conv1d(layer2, n_output, 45, padding='same', activation=tf.nn.relu)

    return layer2


def motion_encoder_channel_first(input, name='motion_encoder', reuse=tf.compat.v1.AUTO_REUSE, dropout_rate=0.25, hidden_units=256,
                                 pooling='average', kernel_size=25, batch_normalization=False):
    '''
    
    :param input: n_samples * n_features * n_frames
    :param name: 
    :param reuse: 
    :param dropout_rate: 
    :param hidden_units: 
    :param pooling: 
    :param kernel_size: 
    :param batch_normalization: 
    :return: 
    '''
    with tf.compat.v1.variable_scope(name, reuse=reuse) as scope:
        layer1 = tf.compat.v1.layers.dropout(input, rate=dropout_rate)

        if not batch_normalization:
            layer2 = tf.compat.v1.layers.conv1d(layer1, hidden_units, kernel_size, padding='same', activation=tf.nn.relu, data_format='channels_first',
                                      kernel_regularizer=regularizer)
        else:
            layer2 = tf.compat.v1.layers.conv1d(layer1, hidden_units, kernel_size, padding='same', activation=None, data_format='channels_first',
                                      kernel_regularizer=regularizer)
            layer2 = tf.contrib.layers.batch_norm(layer2,
                                                  center=True,
                                                  scale=True,
                                                  is_training=True,
                                                  scope='bn')
            layer2 = tf.nn.relu(layer2, 'relu')
        if pooling is None:
            return layer2
        elif pooling == 'average':
            layer3 = tf.compat.v1.layers.average_pooling1d(layer2, 2, strides=2, data_format='channels_first')
            return layer3
        elif pooling == 'max':
            layer3 = tf.compat.v1.layers.max_pooling1d(layer2, 2, strides=2, data_format='channels_first')
            return layer3
        elif pooling == 'spectrum':
            layer3 = spectrum_pooling_1d(layer2, 2, N=512, data_format='channels_first')
            return layer3
        else:
            raise KeyError('Unknown pooling!')



def motion_decoder_channel_first(input, n_input, name='motion_decoder', reuse=tf.compat.v1.AUTO_REUSE, dropout_rate=0.25,
                                 unpool='average', kernel_size=25):
    '''
    
    :param input: n_samples * n_fratures * n_frames
    :param n_input: 
    :param name: 
    :param reuse: 
    :param dropout_rate: 
    :param unpool: 
    :param kernel_size: 
    :return: 
    '''
    with tf.compat.v1.variable_scope(name, reuse=reuse) as scope:
        ## unpooling
        if unpool == 'average':
            layer1 = average_unpooling_1d(input, pool_size=2, data_format='channels_first')
        elif unpool == 'spectrum':
            layer1 = spectrum_unpooling_1d(input, pool_size=2, N=512, data_format='channels_first')
        else:
            layer1 = input
        layer2 = tf.compat.v1.layers.dropout(layer1, rate=dropout_rate)
        layer3 = tf.compat.v1.layers.conv1d(layer2, n_input, kernel_size, padding='same', activation=None, data_format='channels_first',
                                  kernel_regularizer=regularizer)
        return layer3


def spectrum_style_transfer(input, name, reuse=tf.compat.v1.AUTO_REUSE, dropout_rate=0.25):
    with tf.compat.v1.variable_scope(name, reuse=reuse) as scope:
        with tf.compat.v1.variable_scope(name+'/layer1'):
            dropout_layer1 = tf.compat.v1.layers.dropout(input, rate=dropout_rate)
            conv_layer1 = tf.compat.v1.layers.conv1d(dropout_layer1, 128, 45, padding='same', activation=tf.nn.relu, data_format='channels_first',
                                           kernel_regularizer=regularizer)
        fft_layer2 = tf.signal.fft(tf.complex(conv_layer1, conv_layer1 * 0.0))
        mag_layer2 = tf.sqrt(tf.math.real(fft_layer2)**2 + tf.math.imag(fft_layer2)**2)
        max_mag_layer2 = tf.expand_dims(tf.reduce_max(input_tensor=mag_layer2, axis=-1), -1)
        # mag_layer2 = mag_layer2/max_mag_layer2
        phase_layer2 = tf.atan2(tf.math.imag(fft_layer2), tf.math.real(fft_layer2))
        # phase_layer2 = phase_layer2 / (np.pi / 2.0)
        ## adjust the phase and magnitude to
        dropout_mag_layer3 = tf.compat.v1.layers.dropout(mag_layer2, rate=dropout_rate)
        conv_mag_layer3 = tf.compat.v1.layers.conv1d(dropout_mag_layer3, 128, 45, padding='same', activation=tf.nn.relu,
                                           data_format='channels_first', kernel_regularizer=regularizer)
        dropout_phase_layer3 = tf.compat.v1.layers.dropout(phase_layer2, rate=dropout_rate)
        conv_phase_layer3 = tf.compat.v1.layers.conv1d(dropout_phase_layer3, 128, 45, padding='same', activation=tf.nn.relu,
                                             data_format='channels_first', kernel_regularizer=regularizer)

        dropout_mag_layer4 = tf.compat.v1.layers.dropout(conv_mag_layer3, rate=dropout_rate)
        conv_mag_layer4 = tf.compat.v1.layers.conv1d(dropout_mag_layer4, 128, 45, padding='same', activation=None,
                                           data_format='channels_first', kernel_regularizer=regularizer)

        dropout_phase_layer4 = tf.compat.v1.layers.dropout(conv_phase_layer3, rate=dropout_rate)
        conv_phase_layer4 = tf.compat.v1.layers.conv1d(dropout_phase_layer4, 128, 45, padding='same', activation=None,
                                             data_format='channels_first', kernel_regularizer=regularizer)

        # conv_mag_layer4 = conv_mag_layer4 * max_mag_layer2
        # conv_phase_layer4 = conv_phase_layer4 * (np.pi / 2.0)

        layer4_input = tf.complex(conv_mag_layer4*tf.cos(conv_phase_layer4),
                                  conv_mag_layer4*tf.sin(conv_phase_layer4))
        ifft_layer4 = tf.math.real(tf.signal.ifft(layer4_input))

        dropout_layer5 = tf.compat.v1.layers.dropout(ifft_layer4, rate=dropout_rate)
        conv_layer5 = tf.compat.v1.layers.conv1d(dropout_layer5, 128, 45, padding='same', data_format='channels_first',
                                       kernel_regularizer=regularizer)
        return conv_layer5
        # layer3_input = tf.complex(mag_layer2 * tf.cos(phase_layer2), mag_layer2 * tf.sin(phase_layer2))




def spectrum_motion_encoder(input, name='motion_encoder', reuse=tf.compat.v1.AUTO_REUSE, dropout_rate=0.25, hidden_units=256,
                            N=256):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        layer1 = tf.compat.v1.layers.dropout(input, rate=dropout_rate)
        layer1 = tf.compat.v1.layers.conv1d(layer1, hidden_units, 25, padding='same', activation=tf.nn.relu,
                                  data_format='channels_first',
                                  kernel_regularizer=regularizer, bias_regularizer=regularizer)
        ### zero padding to change the length of frames
        input_shape = layer1.get_shape().as_list()
        N_t = tf.convert_to_tensor(value=N)
        zero_padding = tf.zeros((input_shape[0], input_shape[1], N_t - input_shape[2]))
        extended_layer1 = tf.concat([layer1, zero_padding], axis=-1)
        # layer1 = tf.fft(tf.complex(layer1, layer1 * 0.0))
        layer1_fft = tf.signal.fft(tf.complex(extended_layer1, extended_layer1 * 0.0))
        layer1_fft.set_shape((input_shape[0], input_shape[1], N))
    return layer1_fft


def spectrum_motion_decoder(input, n_dims, name='motion_decoder', reuse=tf.compat.v1.AUTO_REUSE, dropout_rate=0.25):

    mag = tf.sqrt(tf.math.real(layer1)**2 + tf.math.imag(layer1)**2)
    phase = tf.atan2(tf.math.imag(layer1), tf.math.real(layer1))

    # layer2_a = tf.layers.dropout(mag)
    # layer2_a = tf.layers.dense(layer2_a, 32, activation=tf.nn.relu)
    # layer3_a = tf.layers.dense(layer2_a, 64)
    #
    # layer2_b = tf.layers.dropout(phase)
    # layer2_b = tf.layers.dense(layer2_b, 32, activation=tf.nn.relu)
    # layer3_b = tf.layers.dense(layer2_b, 64)

    layer2_a = tf.compat.v1.layers.dropout(mag, rate=dropout_rate)
    layer2_a = tf.compat.v1.layers.conv1d(layer2_a, 32, 45, padding='same', activation=tf.nn.relu, data_format='channels_first',
                                kernel_regularizer=regularizer)
    layer3_a = tf.compat.v1.layers.dropout(layer2_a, rate=dropout_rate)
    layer3_a = tf.compat.v1.layers.conv1d(layer3_a, 64, 45, padding='same', activation=None, data_format='channels_first',
                              kernel_regularizer=regularizer)

    layer2_b = tf.compat.v1.layers.dropout(phase, rate=dropout_rate)
    layer2_b = tf.compat.v1.layers.conv1d(layer2_b, 32, 45, padding='same', activation=tf.nn.relu, data_format='channels_first',
                                kernel_regularizer=regularizer)
    layer3_b = tf.compat.v1.layers.dropout(layer2_b, rate=dropout_rate)
    layer3_b = tf.compat.v1.layers.conv1d(layer3_b, 64, 45, padding='same', activation=None, data_format='channels_first',
                              kernel_regularizer=regularizer)


    layer4_input = tf.complex(layer3_a*tf.cos(layer3_b), layer3_a*tf.sin(layer3_b))
    layer4 = tf.math.real(tf.signal.ifft(layer4_input))

    layer5 = tf.compat.v1.layers.dropout(layer4, rate=dropout_rate)
    layer5 = tf.compat.v1.layers.conv1d(layer5, n_input, 45, padding='same', activation=None, data_format='channels_first',
                              kernel_regularizer=regularizer)
    return layer5


def create_style_transfer_model(input, n_inputs, name='style_transfer_model', reuse=tf.compat.v1.AUTO_REUSE, dropout_rate=0.25):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        layer1 = tf.compat.v1.layers.dropout(input, rate=dropout_rate)
        layer1 = tf.compat.v1.layers.conv1d(layer1, 128, 25, padding='same', activation=tf.nn.relu, data_format='channels_first',
                                  kernel_regularizer=regularizer, bias_regularizer=regularizer)
        layer2 = tf.compat.v1.layers.dropout(layer1, rate=dropout_rate)
        layer2 = tf.compat.v1.layers.conv1d(layer2, 128, 25, padding='same', activation=tf.nn.relu, data_format='channels_first',
                                  kernel_regularizer=regularizer, bias_regularizer=regularizer)
        layer3 = tf.compat.v1.layers.dropout(layer2, rate=dropout_rate)
        layer3 = tf.compat.v1.layers.conv1d(layer3, n_inputs, 25, padding='same', activation=None, data_format='channels_first',
                                  kernel_regularizer=regularizer, bias_regularizer=regularizer)
    return layer3



def create_style_transfer_hidden_model(input, name='style_transfer_hidden_model', reuse=tf.compat.v1.AUTO_REUSE,
                                       dropout_rate=0.25):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        with tf.compat.v1.variable_scope('layer1', reuse=reuse):
            layer1 = tf.compat.v1.layers.dropout(input, rate=dropout_rate)
            layer1 = tf.compat.v1.layers.conv1d(layer1, 256, 25, padding='same', activation=tf.nn.relu, data_format='channels_first',
                                      kernel_regularizer=regularizer, bias_regularizer=regularizer)

        with tf.compat.v1.variable_scope('layer2', reuse=reuse):
            layer2 = tf.compat.v1.layers.dropout(layer1, rate=dropout_rate)
            layer2 = tf.compat.v1.layers.conv1d(layer2, 256, 25, padding='same', activation=tf.nn.relu, data_format='channels_first',
                                      kernel_regularizer=regularizer, bias_regularizer=regularizer)

        with tf.compat.v1.variable_scope('layer3', reuse=reuse):
            layer3 = tf.compat.v1.layers.dropout(layer2, rate=dropout_rate)
            layer3 = tf.compat.v1.layers.conv1d(layer3, 256, 25, padding='same', activation=tf.nn.relu, data_format='channels_first',
                                      kernel_regularizer=regularizer, bias_regularizer=regularizer)
            layer3_pooling = tf.compat.v1.layers.average_pooling1d(layer3, 2, strides=2, data_format='channels_first')
    return layer3_pooling

