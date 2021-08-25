# encoding: UTF-8
import tensorflow as tf
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import collections
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute())+ r'/..')
from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder
from mosi_utils_anim.utilities import write_to_json_file
from preprocessing.preprocessing import process_file
from nn.spectrum_pooling import spectrum_pooling_1d
from nn.unpooling import spectrum_unpooling_1d, average_unpooling_1d
from models.simple_models import motion_decoder_channel_first, motion_encoder_channel_first, \
    motion_encoder_without_pooling, motion_decoder_without_pooling, motion_encoder_stride, motion_decoder_stride, motion_decoder_stride2d
from utilities.utils import export_point_cloud_data_without_foot_contact, combine_motion_clips, export_point_cloud_data_without_foot_contact


GAME_ENGINE_SKELETON = collections.OrderedDict(
    [
        ('Root', {'parent': None, 'index': 0}),  #-1
        ('pelvis', {'parent': 'Root', 'index': 1}),  # 0
        ('spine_03', {'parent': 'pelvis', 'index': 2}),   # 1
        ('clavicle_l', {'parent': 'spine_03', 'index': 3}), # 2
        ('upperarm_l', {'parent': 'clavicle_l', 'index': 4}), # 3
        ('lowerarm_l', {'parent': 'upperarm_l', 'index': 5}), # 4
        ('hand_l', {'parent': 'lowerarm_l', 'index': 6}),  # 5
        ('clavicle_r', {'parent': 'spine_03', 'index': 7}), # 2
        ('upperarm_r', {'parent': 'clavicle_r', 'index': 8}), # 7
        ('lowerarm_r', {'parent': 'upperarm_r', 'index': 9}), # 8
        ('hand_r', {'parent': 'lowerarm_r', 'index': 10}),
        ('neck_01', {'parent': 'spine_03', 'index': 11}),
        ('head', {'parent': 'neck_01', 'index': 12}),
        ('thigh_l', {'parent': 'pelvis', 'index': 13}),
        ('calf_l', {'parent': 'thigh_l', 'index': 14}),
        ('foot_l', {'parent': 'calf_l', 'index': 15}),
        ('ball_l', {'parent': 'foot_l', 'index': 16}),
        ('thigh_r', {'parent': 'pelvis', 'index': 17}),
        ('calf_r', {'parent': 'thigh_r', 'index': 18}),
        ('foot_r', {'parent': 'calf_r', 'index': 19}),
        ('ball_r', {'parent': 'foot_r', 'index': 20})
    ]
)


def demo_motion_encoder_spectrum_pooling_single_file(bvhfile):
    """encode a bvhfile using trained model. The skeleton should be the same as the model
    
    Arguments:
        bvhfile {str} -- path to bvh file
    """
    meta_data = r'../data/models/preprocessed_core_channel_first.npz'
    if not os.path.exists(meta_data):
        print("run load_data to create data folder!")
        return
    else:
        preprocess = np.load(meta_data)
        bvhreader = BVHReader(bvhfile)
        frame_len = len(bvhreader.frames)
        print('number of frames: ', frame_len)
        filename = os.path.split(bvhfile)[-1]

        test_data = process_file(bvhfile, sliding_window=True, body_plane_indice=[2, 17, 13])
        assert test_data is not None
        test_data = np.swapaxes(test_data, 1, 2)
        print('input clips: ', test_data.shape)
        if test_data.shape[2] % 2 != 0:
            test_data = test_data[:, :, :-1]
        n_samples, n_dims, n_frames = test_data.shape
        normalized_data = (test_data - preprocess['Xmean']) / preprocess['Xstd']
        input = tf.compat.v1.placeholder(tf.float32, shape=[1, n_dims, n_frames])
        encoder_op = motion_encoder_channel_first(input, name='encoder', hidden_units=256, pooling='spectrum',
                                                kernel_size=25)
        decoder_op = motion_decoder_channel_first(encoder_op, n_dims, name='decoder', unpool='spectrum', kernel_size=25)

        pool_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, n_dims, n_frames])
        pooled_decoder = spectrum_pooling_1d(pool_input, pool_size=2, N=512)
        # print(decoder_op.shape)
        unpooled_decoder = spectrum_unpooling_1d(pooled_decoder, pool_size=2, N=512)

        # average_pooled_decoder = tf.layers.average_pooling1d(input, 2, strides=2, data_format='channels_first')
        # average_unpooled_decoder = average_unpooling_1d(average_pooled_decoder, 2, data_format='channels_first')
        saver = tf.compat.v1.train.Saver()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        reconstructed_clips = []
        out_dir = r'../data/results'
        with tf.compat.v1.Session(config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.restore(sess, 'data/core_network_spectrum_pooling_250_0.00001.ckpt')
            # saver.restore(sess, 'data/core_network_average_pooling_500.ckpt')
            for i in range(n_samples):
                # reconstructed_motion = sess.run(average_unpooled_decoder, feed_dict={input: normalized_data[i: i+1]})
                motion_decoder = sess.run(decoder_op, feed_dict={input: normalized_data[i:i+1]})
                reconstructed_motion = sess.run(unpooled_decoder, feed_dict={pool_input: motion_decoder})
                reconstructed_motion = (motion_decoder * preprocess['Xstd']) + preprocess['Xmean']
                reconstructed_clips.append(reconstructed_motion[0])
            reconstructed_clips = np.swapaxes(reconstructed_clips, 1, 2)
            print(reconstructed_clips.shape)
            combined_motion = combine_motion_clips(reconstructed_clips, frame_len-1, 120)
            combined_motion = np.swapaxes(combined_motion, 0, 1)[np.newaxis, :, :]
            export_point_cloud_data_without_foot_contact(combined_motion[:, :-4], os.path.join(out_dir, filename[:-4]+'.panim'))



def demo_motion_encoder_single_file(bvhfile, body_plane_indice=[2, 17, 13]):
    """encode a bvhfile using trained model. The skeleton should be the same as the model
    
    Arguments:
        bvhfile {str} -- path to bvh file
    """
    meta_data = r'../data/models/preprocessed_core_channel_first.npz'
    model_file = r'../data/models/core_network_average_pooling_300.ckpt'
    if not os.path.exists(meta_data) or not os.path.exists(model_file + '.index'):
        print("run load_data to create data folder!")
        return
    else:
        preprocess = np.load(meta_data)

        filename = os.path.split(bvhfile)[-1]
        test_data = process_file(bvhfile, sliding_window=False, body_plane_indice=body_plane_indice)

        test_data = np.swapaxes(test_data, 0, 1)[np.newaxis, :, :]

        if test_data.shape[2] % 2 != 0:
            test_data = test_data[:, :, :-1]

        n_samples, n_dims, n_frames = test_data.shape
        normalized_data = (test_data - preprocess['Xmean']) / preprocess['Xstd']
        input = tf.compat.v1.placeholder(tf.float32, shape=[1, n_dims, n_frames])
        encoder_op = motion_encoder_channel_first(input, name='encoder', hidden_units=256, pooling='average')
        decoder_op = motion_decoder_channel_first(encoder_op, n_dims, name='decoder', unpool='average')

        saver = tf.compat.v1.train.Saver()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        out_dir = r'../data/results'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with tf.compat.v1.Session(config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.restore(sess, model_file)
            encoded_motion = sess.run(encoder_op, feed_dict={input: normalized_data})
            print('encoded motion shape: ', encoded_motion.shape)
            reconstructed_motion = sess.run(decoder_op, feed_dict={input: normalized_data})
            # denoised_motion = sess.run(decoder_op, feed_dict={input: reconstructed_motion})
            reconstructed_motion = (reconstructed_motion * preprocess['Xstd']) + preprocess['Xmean']
            print(reconstructed_motion.shape)
            export_point_cloud_data_without_foot_contact(np.swapaxes(reconstructed_motion[0], 0, 1)[:, :-4], os.path.join(out_dir, filename[:-4]+'_average_pooling.panim'))


def demo_autoencoder_strides(bvhfile, body_plane_indice=[2, 17, 13]):
    """encode a bvhfile using trained model with strides. The skeleton should be the same as the model
    
    Arguments:
        bvhfile {str} -- path to bvh file  
    """
    meta_data = r'../data/models/preprocessed_core_channel_first.npz'
    model_file = r'../data/models/core_network_average_pooling_300.ckpt'
    preprocess = np.load('preprocessed_core_channel_first.npz')
    test_data = process_file(bvhfile, sliding_window=False, body_plane_indice=body_plane_indice)
    n_frames, n_dims = test_data.shape
    test_data = np.swapaxes(test_data, 0, 1)[np.newaxis, :, :]
    normalized_data = (test_data - preprocess['Xmean']) / preprocess['Xstd']
    input = tf.compat.v1.placeholder(tf.float32, [1, n_dims, n_frames])
    encoder_op = motion_encoder_stride(input, name='encoder')
    decoder_op = motion_decoder_stride2d(encoder_op, n_dims, name='decoder')
    encoder_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    decoder_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
    saver = tf.compat.v1.train.Saver(encoder_params + decoder_params)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, 'data/core_network_stride2_2d_200.ckpt')
        # reconstructed_motion = sess.run(decoder_op, feed_dict={input: normalized_data[13:14]})
        reconstructed_motion = sess.run(decoder_op, feed_dict={input: normalized_data})
        reconstructed_motion = (reconstructed_motion * preprocess['Xstd']) + preprocess['Xmean']
        # reconstructed_motion = np.swapaxes(reconstructed_motion, 1, 2)
        # export_point_cloud_data(reconstructed_motion, r'E:\tmp\reconstructed.panim')
        export_point_cloud_data(test_data, r'E:\tmp\origin.panim')
        export_point_cloud_data(reconstructed_motion, r'E:\tmp\locomotion_walk_001_000_stride2d.panim')


def demo_autoencoder_strides_multilayers():
    from models import motion_encoder_stride_multilayers, motion_decoder_stride2d_multilayers, \
        motion_encoder_multilayers, motion_decoder_multilayers
    preprocess = np.load('preprocessed_core_channel_first.npz')
    # test_file = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\edin\edin_locomotion\locomotion_walk_001_000.bvh'
    # test_data = process_file(test_file, sliding_window=False)
    test_file = r'E:\tmp\style_transfer_result.panim'
    test_data = process_panim_data(test_file, sliding_window=False)
    n_frames, n_dims = test_data.shape
    test_data = np.swapaxes(test_data, 0, 1)[np.newaxis, :, :]
    normalized_data = (test_data - preprocess['Xmean']) / preprocess['Xstd']
    input = tf.compat.v1.placeholder(tf.float32, [1, n_dims, n_frames])
    # encoder_op = motion_encoder_stride_multilayers(input, name='encoder')
    #
    # decoder_op = motion_decoder_stride2d_multilayers(encoder_op, n_dims, name='decoder')
    encoder_op = motion_encoder_multilayers(input, name='encoder')
    decoder_op = motion_decoder_multilayers(encoder_op, n_dims, name='decoder')
    encoder_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    decoder_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
    saver = tf.compat.v1.train.Saver(encoder_params + decoder_params)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, 'data/core_network_multilayers.ckpt')
        # reconstructed_motion = sess.run(decoder_op, feed_dict={input: normalized_data[13:14]})
        reconstructed_motion = sess.run(decoder_op, feed_dict={input: normalized_data})
        reconstructed_motion = (reconstructed_motion * preprocess['Xstd']) + preprocess['Xmean']
        # reconstructed_motion = np.swapaxes(reconstructed_motion, 1, 2)
        # export_point_cloud_data(reconstructed_motion, r'E:\tmp\reconstructed.panim')
        export_point_cloud_data(test_data, r'E:\tmp\origin.panim')
        export_point_cloud_data(reconstructed_motion, r'E:\tmp\denoised.panim')


def demo_motion_autoencoder_channel_first_combined():

    preprocess = np.load('preprocessed_core_channel_first.npz')
    # normalized_data = (data - preprocess['Xmean']) / preprocess['Xstd']
    test_file = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\edin\edin_locomotion\locomotion_walk_001_000.bvh'
    bvhreader = BVHReader(test_file)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    frame_len = len(bvhreader.frames)
    test_clips = process_file(test_file)
    test_clips = np.asarray(test_clips)
    n_clips, n_frames, n_dims = test_clips.shape
    test_clips = np.swapaxes(test_clips, 1, 2)
    normalized_clips = (test_clips - preprocess['Xmean']) / preprocess['Xstd']
    input = tf.compat.v1.placeholder(tf.float32, [None, n_dims, n_frames])
    encoder_op = motion_encoder_channel_first(input, name='encoder')
    decoder_op = motion_decoder_channel_first(encoder_op, n_dims, name='decoder')

    saver = tf.compat.v1.train.Saver()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, 'data/core_network_channel_first.ckpt')
        reconstructed_clips = np.zeros((n_clips, n_dims, n_frames))
        for i in range(n_clips):
            reconstructed_clips[i] = sess.run(decoder_op, feed_dict={input: normalized_clips[i:i+1]})[0]

        reconstructed_clips = (reconstructed_clips * preprocess['Xstd']) + preprocess['Xmean']
        reconstructed_clips = np.swapaxes(reconstructed_clips, 1, 2)
        print(frame_len)
        print(reconstructed_clips.shape)
        reconstructed_motion = combine_motion_clips(reconstructed_clips, frame_len-1, window_step=120)
        reconstructed_motion = np.swapaxes(reconstructed_motion, 0, 1)[np.newaxis, :, :]
        # reconstruct_motion_from_processed_data(reconstructed_motion, skeleton, r'E:\tmp\locomotion_walk_001_000_combined.bvh')
        export_point_cloud_data(reconstructed_motion, r'E:\tmp\locomotion_walk_001_000_combined.panim')


def demo_motion_spectrum_autoencoder():
    stylized_data_dic = np.load(r'data/training_data/processed_stylized_data.npz')
    filelist = stylized_data_dic.files
    preprocess = np.load(r'data/preprocessed_core_expmap.npz')
    skeleton_file = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\stylized_data\angry\angry_normalwalking_0.bvh'
    bvhreader = BVHReader(skeleton_file)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader, animated_joints=GAME_ENGINE_ANIMATED_JOINTS)
    test_file_index = 0
    test_data = stylized_data_dic[filelist[test_file_index]]
    n_frames, n_dims = test_data.shape
    test_data = np.swapaxes(test_data, 0, 1)[np.newaxis, :, :]
    normalized_test_data = (test_data - preprocess['Xmean']) / preprocess['Xstd']
    print(normalized_test_data.shape)
    input = tf.compat.v1.placeholder(tf.float32, [None, n_dims, n_frames])
    encoder_op = motion_encoder_channel_first(input)
    decoder_op = motion_decoder_channel_first(encoder_op, n_dims)
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    save_filename = r'E:\tmp\reconstructed_expmap.bvh'
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(init)
        saver.restore(sess, 'data/core_network_expmap')
        reconstructed_motion = sess.run(decoder_op, feed_dict={input: normalized_test_data})
        reconstructed_motion = (reconstructed_motion * preprocess['Xstd']) + preprocess['Xmean']
        reconstructed_motion = np.swapaxes(reconstructed_motion[0], 0, 1)
        origin_motion = np.swapaxes(test_data[0], 0, 1)
        print(origin_motion.shape)
        reconstruct_motion_from_processed_data(reconstructed_motion, skeleton, save_filename)
        reconstruct_motion_from_processed_data(origin_motion, skeleton, r'E:\tmp\original_motion.bvh')



def demo_motion_autoencoder_expmap():
    stylized_data_dic = np.load(r'data/training_data/processed_stylized_data_expmap.npz')
    filelist = stylized_data_dic.files
    preprocess = np.load(r'data/preprocessed_core_expmap.npz')
    skeleton_file = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\stylized_data\angry\angry_normalwalking_0.bvh'
    bvhreader = BVHReader(skeleton_file)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader, animated_joints=GAME_ENGINE_ANIMATED_JOINTS)
    test_file_index = 0
    test_data = stylized_data_dic[filelist[test_file_index]]
    n_frames, n_dims = test_data.shape
    test_data = np.swapaxes(test_data, 0, 1)[np.newaxis, :, :]
    normalized_test_data = (test_data - preprocess['Xmean']) / preprocess['Xstd']
    print(normalized_test_data.shape)
    input = tf.compat.v1.placeholder(tf.float32, [None, n_dims, n_frames])
    encoder_op = motion_encoder_channel_first(input)
    decoder_op = motion_decoder_channel_first(encoder_op, n_dims)
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    save_filename = r'E:\tmp\reconstructed_expmap.bvh'
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(init)
        saver.restore(sess, 'data/core_network_expmap')
        reconstructed_motion = sess.run(decoder_op, feed_dict={input: normalized_test_data})
        reconstructed_motion = (reconstructed_motion * preprocess['Xstd']) + preprocess['Xmean']
        reconstructed_motion = np.swapaxes(reconstructed_motion[0], 0, 1)
        origin_motion = np.swapaxes(test_data[0], 0, 1)
        print(origin_motion.shape)
        reconstruct_motion_from_processed_data(reconstructed_motion, skeleton, save_filename)
        reconstruct_motion_from_processed_data(origin_motion, skeleton, r'E:\tmp\original_motion.bvh')


def demo_motion_autoencoder_channel_first_without_pooling():
    # data = np.load(r'../theano/data/training_data/processed_edin_data.npz')['clips']
    # data = np.swapaxes(data, 1, 2)
    # n_samples, n_dims, n_frames = data.shape
    preprocess = np.load('preprocessed_core_without_pooling.npz')
    # normalized_data = (data - preprocess['Xmean']) / preprocess['Xstd']

    test_file = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\edin\edin_locomotion\locomotion_walk_001_000.bvh'
    test_data = process_file(test_file, sliding_window=False)
    n_frames, n_dims = test_data.shape
    test_data = np.swapaxes(test_data, 0, 1)[np.newaxis, :, :]
    normalized_data = (test_data - preprocess['Xmean']) / preprocess['Xstd']
    input = tf.compat.v1.placeholder(tf.float32, [None, n_dims, n_frames])
    encoder_op = motion_encoder_without_pooling(input)
    decoder_op = motion_decoder_without_pooling(encoder_op, n_dims)
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(init)
        saver.restore(sess, 'data/core_network_without_pooling')
        # reconstructed_motion = sess.run(decoder_op, feed_dict={input: normalized_data[13:14]})
        reconstructed_motion = sess.run(decoder_op, feed_dict={input: normalized_data})
        reconstructed_motion = (reconstructed_motion * preprocess['Xstd']) + preprocess['Xmean']

        # reconstructed_motion = np.swapaxes(reconstructed_motion, 1, 2)

        export_point_cloud_data(reconstructed_motion, r'E:\tmp\locomotion_walk_001_000_no_pooling.panim')


def test():
    # input_data = np.random.rand(10, 512, 90)
    # res = motion_encoder_without_pooling(input_data)
    # print(res.shape)
    accad_data = np.load(r'D:\workspace\my_git_repos\data\training_data\processed_accad_data.npz')['clips']

    print(accad_data.shape)


    ### load pretrained model
    input = tf.compat.v1.placeholder(tf.float32, [None, n_dims, n_frames])
    encoder_op = motion_encoder_without_pooling(input)
    decoder_op = motion_decoder_without_pooling(encoder_op, n_dims)
    motion_encoder = motion_encoder_without_pooling()



if __name__ == "__main__":
    # test_file = r'D:\workspace\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\edin\edin_locomotion\locomotion_jog_000_000.bvh'
    # demo_motion_encoder_single_file(test_file)
    # test()
    demo_motion_autoencoder_channel_first_without_pooling()