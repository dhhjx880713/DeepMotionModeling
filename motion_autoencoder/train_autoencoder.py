# encoding: UTF-8
import numpy as np
import tensorflow as tf
from models.simple_models import motion_decoder_channel_first, motion_encoder_channel_first, \
    motion_encoder_without_pooling, motion_decoder_without_pooling
import sys
import copy
from datetime import datetime
import os
from models.conv_autoencoder import ConvAutoEncoder


def train_autoencoder_channel_first(fine_tuning=False):
    """train a single convolutional layer network for motion encoding and decoding
    
    Keyword Arguments:
        fine_tuning {bool} -- [description] (default: {False})
    """

    accad_data = np.load(r'../theano/data/training_data/processed_accad_data.npz')['clips']
    cmu_data = np.load(r'../theano/data/training_data/processed_cmu_data.npz')['clips']
    stylized_data = np.load(r'../theano/data/training_data/processed_stylized_data.npz')['clips']
    ulm_data = np.load(r'../theano/data/training_data/processed_ulm_locomotion_data.npz')['clips']
    edin_data = np.load(r'../theano/data/training_data/processed_edin_data.npz')['clips']
    rng = np.random.RandomState(23456)
    X = np.concatenate([accad_data, stylized_data, edin_data, ulm_data], axis=0)
    # X = stylized_data
    X = np.swapaxes(X, 1, 2)
    Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
    Xmean[:,-7:-4] = 0.0
    Xmean[:,-4:]   = 0.5
    Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)

    Xstd[:,-7:-5] = 0.9 * X[:,-7:-5].std()
    Xstd[:,-5:-4] = 0.9 * X[:,-5:-4].std()
    Xstd[:,-4:] = 0.5
    # np.savez_compressed('ulm_preprocessed_core_channel_first.npz', Xmean=Xmean, Xstd=Xstd)
    X = (X - Xmean) / Xstd
    Y = copy.deepcopy(X)

    ### add random noise to X
    # X_tmp = np.zeros(X.shape)
    # for i in range(10):
    #     X_tmp += X * np.random.binomial(1, 0.9, X.shape)
    # X = X_tmp / 10.0
    print("training data size: ")
    print(X.shape)

    I = np.arange(len(X))
    rng.shuffle(I)
    X = X[I]
    Y = Y[I]
    batchsize = 10
    n_epochs = 10
    learning_rate = 0.00001
    n_samples, n_dims, n_frames = X.shape
    input = tf.compat.v1.placeholder(tf.float32, shape=[batchsize, n_dims, n_frames])
    # output  = tf.placeholder(tf.float32, shape=[batchsize, n_dims, n_frames])
    encoder_op = motion_encoder_channel_first(input, name='encoder', hidden_units=512, pooling='average',
                                              kernel_size=25, batch_normalization=True)

    decoder_op = motion_decoder_channel_first(encoder_op, n_dims, name='decoder', unpool='average', kernel_size=25)
    # pool_input = spectrum_pooling_1d(input, pool_size=2, N=512)
    # smoothed_input = spectrum_unpooling_1d(pool_input, pool_size=2, N=512)
    loss_op = tf.reduce_mean(input_tensor=tf.pow(input - decoder_op, 2))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss_op)
    encoder_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    decoder_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
    saver = tf.compat.v1.train.Saver(encoder_params + decoder_params)
    if fine_tuning:
        motion_encoder_origin_saver = tf.compat.v1.train.Saver(encoder_params + decoder_params)
        motion_encoder_origin_file = r'data\ulm_core_network_64_hidden_1e-05_5.ckpt'
    save_dir = 'data'
    # model_file = 'data/core_network_spectrum_pooling_250_0.00001_fine_tuning.ckpt'
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # if fine_tuning:
        #     motion_encoder_origin_saver.restore(sess, motion_encoder_origin_file)
        #     model_file_name = motion_encoder_origin_file
        #     model_file = os.path.join(save_dir, model_file_name)
        # else:
        #     model_file_name = '_'.join(['ulm_core_network_64_hidden', str(learning_rate), str(n_epochs)]) + '.ckpt'
        #     model_file = os.path.join(save_dir, model_file_name)
        last_mean = 0
        for epoch in range(n_epochs):
            batchinds = np.arange(n_samples // batchsize)
            rng.shuffle(batchinds)
            c = []
            model_filename = '_'.join(['ulm_core_network_64_hidden', str(learning_rate)]) + '.ckpt'
            model_filename = os.path.join(save_dir, model_filename)
            if fine_tuning and epoch > 0:
                saver.restore(sess, model_filename)
            # print(model_filename)

            for bii, bi in enumerate(batchinds):
                # sess.run(train_op, feed_dict={input: X[bi*batchsize: (bi+1)*batchsize],
                #                               output: Y[bi*batchsize: (bi+1)*batchsize]})
                # c.append(sess.run(loss_op, feed_dict={input: X[bi*batchsize: (bi+1)*batchsize],
                #                                       output: Y[bi * batchsize: (bi + 1) * batchsize]}))
                sess.run(train_op, feed_dict={input: X[bi*batchsize: (bi+1)*batchsize]})
                c.append(sess.run(loss_op, feed_dict={input: X[bi*batchsize: (bi+1)*batchsize]}))
                if np.isnan(c[-1]): return
                if bii % (int(len(batchinds) / 1000) + 1) == 0:
                    sys.stdout.write('\r[Epoch %i]  %0.1f%% mean %.5f' % (epoch, 100 * float(bii)/len(batchinds),
                                                                          np.mean(c)))
                    sys.stdout.flush()
            curr_mean = np.mean(c)
            diff_mean, last_mean = curr_mean-last_mean, curr_mean
            print('\r[Epoch %i] 100.0%% mean %.5f diff %.5f %s' %
                (epoch, curr_mean, diff_mean, str(datetime.now())[11:19]))
            saver.save(sess, model_filename)
            # sys.stdout.flush()
        # if model_file is not None:
        #     save_path = saver.save(sess, model_file)
        #     print("Model saved in file: %s" % save_path)


def train_autoencoder_stride():
    from models import motion_encoder_stride, motion_decoder_stride, motion_decoder_stride2d
    accad_data = np.load(r'../theano/data/training_data/processed_accad_data.npz')['clips']
    cmu_data = np.load(r'../theano/data/training_data/processed_cmu_data.npz')['clips']
    stylized_data = np.load(r'../theano/data/training_data/processed_stylized_data.npz')['clips']
    ulm_data = np.load(r'../theano/data/training_data/processed_ulm_locomotion_data.npz')['clips']
    edin_data = np.load(r'../theano/data/training_data/processed_edin_data.npz')['clips']
    rng = np.random.RandomState(23456)
    X = np.concatenate([accad_data, cmu_data, stylized_data, edin_data, ulm_data], axis=0)
    X = np.swapaxes(X, 1, 2)
    preprocess = np.load('preprocessed_core_channel_first.npz')
    X = (X - preprocess['Xmean']) / preprocess['Xstd']
    I = np.arange(len(X))
    rng.shuffle(I)
    X = X[I]
    batchsize = 1
    n_epochs = 200
    learning_rate = 0.0001
    n_samples, n_dims, n_frames = X.shape

    input = tf.compat.v1.placeholder(tf.float32, shape=[batchsize, n_dims, n_frames])
    encoder_op = motion_encoder_stride(input, name='encoder')
    # decoder_op = motion_decoder_stride(encoder_op, n_dims, name='decoder')

    decoder_op = motion_decoder_stride2d(encoder_op, n_dims, name='decoder')
    # print(encoder_op.shape)
    # print(decoder_op.shape)

    loss_op = tf.reduce_mean(input_tensor=tf.pow(input - decoder_op, 2))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss_op)
    init = tf.compat.v1.global_variables_initializer()
    encoder_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    decoder_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')


    saver = tf.compat.v1.train.Saver(encoder_params + decoder_params)
    model_file = 'data/core_network_stride2_2d_200.ckpt'
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(init)
        last_mean = 0
        for epoch in range(n_epochs):
            batchinds = np.arange(n_samples // batchsize)
            rng.shuffle(batchinds)
            c = []

            for bii, bi in enumerate(batchinds):
                sess.run(train_op, feed_dict={input: X[bi*batchsize: (bi+1)*batchsize]})
                c.append(sess.run(loss_op, feed_dict={input: X[bi*batchsize: (bi+1)*batchsize]}))
                if np.isnan(c[-1]): return
                if bii % (int(len(batchinds) / 1000) + 1) == 0:
                    sys.stdout.write('\r[Epoch %i]  %0.1f%% mean %.5f' % (epoch, 100 * float(bii)/len(batchinds),
                                                                          np.mean(c)))
                    sys.stdout.flush()
            curr_mean = np.mean(c)
            diff_mean, last_mean = curr_mean-last_mean, curr_mean
            print('\r[Epoch %i] 100.0%% mean %.5f diff %.5f %s' %
                (epoch, curr_mean, diff_mean, str(datetime.now())[11:19]))
            # sys.stdout.flush()
        if model_file is not None:
            save_path = saver.save(sess, model_file)
            print("Model saved in file: %s" % save_path)


def train_autoencoder():
    # accad_data = np.load(r'../theano/data/training_data/processed_accad_data.npz')['clips']
    # cmu_data = np.load(r'../theano/data/training_data/processed_cmu_data.npz')['clips']
    stylized_data = np.load(r'../theano/data/training_data/processed_stylized_data.npz')['clips']
    ulm_data = np.load(r'../theano/data/training_data/processed_ulm_locomotion_data.npz')['clips']
    edin_data = np.load(r'../theano/data/training_data/processed_edin_locomotion_data.npz')['clips']
    rng = np.random.RandomState(23456)
    X = np.concatenate([stylized_data, edin_data, ulm_data], axis=0)
    # X = np.swapaxes(X, 1, 2)
    # preprocess = np.load('preprocessed_core_channel_first.npz')
    # X = (X - preprocess['Xmean']) / preprocess['Xstd']
    # I = np.arange(len(X))
    # rng.shuffle(I)
    # X = X[I]

    """ Swap training data axis """
    X = np.swapaxes(X, 1, 2)
    Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
    Xmean[:,-7:-4] = 0.0
    Xmean[:,-4:]   = 0.5
    Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)

    Xstd[:,-7:-5] = 0.9 * X[:,-7:-5].std()
    Xstd[:,-5:-4] = 0.9 * X[:,-5:-4].std()
    Xstd[:,-4:] = 0.5
    meta_data_file = r'E:\workspace\tensorflow_results\data\processed_meta_locomotion_large_window.npz'
    np.savez_compressed(meta_data_file, Xmean=Xmean, Xstd=Xstd)

    normalized_X = (X - Xmean) / Xstd
    """ Create Model """
    batchsize = 10
    n_epochs = 250
    learning_rate = 1e-5
    n_samples, n_dims, n_frames = X.shape

    model_filename = r'E:\tensorflow\tags\tag_models/conv_autoencoder_small_window_' + str(learning_rate) + '_' + str(
        n_epoches) + '_' + str(batchsize) + '_' + str(hidden_units) + '.ckpt'
    m = ConvAutoEncoder(name='motion_autoencoder', n_frames=n_frames, n_dims=n_dims, kernel_size=kernel_size,
                        encode_activation=encode_activation, decode_activation=decode_activation,
                        hidden_units=hidden_units, n_epoches=n_epoches, batchsize=batchsize, pooling='average',
                        batch_norm=False)

    # ### create multilayers stride encoder/deconder
    # # encoder_op = motion_encoder_stride_multilayers(input, name='encoder')
    # # decoder_op = motion_decoder_stride2d_multilayers(encoder_op, n_dims, name='decoder')
    # # model_file = 'data/core_network_stride2_2d_multilayers.ckpt'
    #
    #
    # ### create multi-layers
    # encoder_op = motion_encoder_multilayers(input, name='encoder')
    # decoder_op = motion_decoder_multilayers(encoder_op, n_dims, name='decoder')
    # model_file = 'data/core_network_multilayers_10.ckpt'
    #
    # loss_op = tf.reduce_mean(tf.pow(input - decoder_op, 2))
    # optimizer = tf.train.AdamOptimizer(learning_rate)
    # train_op = optimizer.minimize(loss_op)
    # init = tf.global_variables_initializer()
    # encoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    # decoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
    #
    # saver = tf.train.Saver(encoder_params + decoder_params)
    #
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # with tf.Session(config=config) as sess:
    #     sess.run(init)
    #     last_mean = 0
    #     for epoch in range(n_epochs):
    #         batchinds = np.arange(n_samples // batchsize)
    #         rng.shuffle(batchinds)
    #         c = []
    #         for bii, bi in enumerate(batchinds):
    #             sess.run(train_op, feed_dict={input: X[bi*batchsize: (bi+1)*batchsize]})
    #             c.append(sess.run(loss_op, feed_dict={input: X[bi*batchsize: (bi+1)*batchsize]}))
    #             if np.isnan(c[-1]): return
    #             if bii % (int(len(batchinds) / 1000) + 1) == 0:
    #                 sys.stdout.write('\r[Epoch %i]  %0.1f%% mean %.5f' % (epoch, 100 * float(bii)/len(batchinds),
    #                                                                       np.mean(c)))
    #                 sys.stdout.flush()
    #         curr_mean = np.mean(c)
    #         diff_mean, last_mean = curr_mean-last_mean, curr_mean
    #         print('\r[Epoch %i] 100.0%% mean %.5f diff %.5f %s' %
    #             (epoch, curr_mean, diff_mean, str(datetime.now())[11:19]))
    #     if model_file is not None:
    #         save_path = saver.save(sess, model_file)
    #         print("Model saved in file: %s" % save_path)


def train_autoencoder_channel_first_without_pooling():
    accad_data = np.load(r'../theano/data/training_data/processed_accad_data.npz')['clips']
    cmu_data = np.load(r'../theano/data/training_data/processed_cmu_data.npz')['clips']
    stylized_data = np.load(r'../theano/data/training_data/processed_stylized_data.npz')['clips']
    ulm_data = np.load(r'../theano/data/training_data/processed_ulm_locomotion_data.npz')['clips']
    edin_data = np.load(r'../theano/data/training_data/processed_edin_data.npz')['clips']
    rng = np.random.RandomState(23456)
    X = np.concatenate([accad_data, cmu_data, stylized_data, edin_data, ulm_data], axis=0)
    X = np.swapaxes(X, 1, 2)
    Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis, :, np.newaxis]
    Xmean[:, -7:-4] = 0.0
    Xmean[:, -4:] = 0.5
    Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)

    Xstd[:, -7:-5] = 0.9 * X[:, -7:-5].std()
    Xstd[:, -5:-4] = 0.9 * X[:, -5:-4].std()
    Xstd[:, -4:] = 0.5
    np.savez_compressed('preprocessed_core_without_pooling.npz', Xmean=Xmean, Xstd=Xstd)
    X = (X - Xmean) / Xstd
    print("training data size: ")
    print(X.shape)
    I = np.arange(len(X))
    rng.shuffle(I)
    X = X[I]
    batchsize = 1
    n_epochs = 250
    learning_rate = 0.0001
    n_samples, n_dims, n_frames = X.shape
    input = tf.compat.v1.placeholder(tf.float32, shape=[None, n_dims, n_frames])
    encoder_op = motion_encoder_without_pooling(input)
    decoder_op = motion_decoder_without_pooling(encoder_op, n_dims)
    loss_op = tf.reduce_mean(input_tensor=tf.pow(input - decoder_op, 2))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss_op)
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()
    model_file = 'data/core_network_without_pooling'
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(init)
        last_mean = 0
        for epoch in range(n_epochs):
            batchinds = np.arange(n_samples // batchsize)
            rng.shuffle(batchinds)
            c = []
            for bii, bi in enumerate(batchinds):
                sess.run(train_op, feed_dict={input: X[bi * batchsize: (bi + 1) * batchsize]})
                c.append(sess.run(loss_op, feed_dict={input: X[bi * batchsize: (bi + 1) * batchsize]}))
                if np.isnan(c[-1]): return
                if bii % (int(len(batchinds) / 1000) + 1) == 0:
                    sys.stdout.write('\r[Epoch %i]  %0.1f%% mean %.5f' % (epoch, 100 * float(bii) / len(batchinds),
                                                                          np.mean(c)))
                    sys.stdout.flush()
            curr_mean = np.mean(c)
            diff_mean, last_mean = curr_mean - last_mean, curr_mean
            print('\r[Epoch %i] 100.0%% mean %.5f diff %.5f %s' %
                  (epoch, curr_mean, diff_mean, str(datetime.now())[11:19]))
            # sys.stdout.flush()
        if model_file is not None:
            save_path = saver.save(sess, model_file)
            print("Model saved in file: %s" % save_path)


def train_autoencoder_expmap():
    accad_data = np.load(r'data/training_data/processed_accad_data_expmap.npz')
    cmu_data = np.load(r'data/training_data/processed_cmu_data_expmap.npz')
    stylized_data = np.load(r'data/training_data/processed_stylized_data_expmap.npz')
    ulm_data = np.load(r'data/training_data/processed_ulm_data_expmap.npz')
    edin_data = np.load(r'data/training_data/processed_edin_data_expmap.npz')
    input_data = dict()
    # data_types = [stylized_data, edin_data, ulm_data]
    data_types = [stylized_data]
    for data_type in data_types:
        for filename in data_type.files:
            input_data[filename] = data_type[filename]
    X = None
    filelist = input_data.keys()
    for filename in filelist:
        input_data[filename] = np.swapaxes(input_data[filename], 0, 1)[np.newaxis, :, :]
        if not X is None:
            X = np.concatenate((X, input_data[filename]), axis=-1)
        else:
            X = input_data[filename]
    print(X.shape)
    Xmean = X.mean(axis=2)[:, :, np.newaxis]
    Xstd = X.std(axis=2)[:, :, np.newaxis]

    np.savez_compressed(r'data/preprocessed_core_expmap.npz', Xmean=Xmean, Xstd=Xstd)

    n_files = len(filelist)
    n_epochs = 200
    learning_rate = 0.0001
    input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, X.shape[1], None])
    encoder_op = motion_encoder_channel_first(input)
    decoder_op = motion_decoder_channel_first(encoder_op, X.shape[1])
    loss_op = tf.reduce_mean(input_tensor=tf.pow(input - decoder_op, 2))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss_op)
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()
    model_file = 'data/core_network_expmap'
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(init)
        last_mean = 0
        for epoch in range(n_epochs):
            c = []
            count = 1
            for filename in filelist:
                if input_data[filename].shape[2] > 100:  ## remove short motion files
                    training_data = (input_data[filename] - Xmean) / Xstd
                    sess.run(train_op, feed_dict={input: training_data})
                    c.append(sess.run(loss_op, feed_dict={input: training_data}))
                    if np.isnan(c[-1]): return
                    sys.stdout.write('\r[Epoch %i]  %0.1f%% mean %.5f' % (epoch, 100 * count/n_files,
                                                                          np.mean(c)))
                    sys.stdout.flush()
                count += 1
            curr_mean = np.mean(c)
            diff_mean, last_mean = curr_mean-last_mean, curr_mean
            print('\r[Epoch %i] 100.0%% mean %.5f diff %.5f %s' %
                (epoch, curr_mean, diff_mean, str(datetime.now())[11:19]))
            # sys.stdout.flush()
        if model_file is not None:
            save_path = saver.save(sess, model_file)
            print("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    # train_autoencoder()
    # train_autoencoder_expmap()
    train_autoencoder_channel_first(fine_tuning=True)
    # train_autoencoder_channel_first_without_pooling()
    # train_autoencoder_stride()
    # train_autoencoder_stride_multilayers()