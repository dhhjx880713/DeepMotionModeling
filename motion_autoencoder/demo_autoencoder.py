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
from morphablegraphs.utilities import write_to_json_file
from mosi_utils_anim.animation_data import Quaternions


from preprocess_expmap import reconstruct_motion_from_processed_data, GAME_ENGINE_ANIMATED_JOINTS
from utils import export_point_cloud_data, GAME_ENGINE_SKELETON, combine_motion_clips
from preprocessing import process_file
from morphablegraphs.construction.retargeting.convert_panim_to_bvh import save_motion_data_to_bvh

from nn.spectrum_pooling import spectrum_pooling_1d
from nn.unpooling import spectrum_unpooling_1d, average_unpooling_1d



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


def demo_motion_encoder_spectrum_pooling_single_file():
    preprocess = np.load('preprocessed_core_channel_first.npz')
    # test_file = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\ulm\Take_carry\carry_041_3.bvh'
    test_file = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\edin\edin_locomotion\locomotion_walk_001_000.bvh'
    # test_file = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\stylized_data\angry\angry_normalwalking_2.bvh'
    # test_file = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\leftStance_game_engine_skeleton_new_grounded\walk_s_016_leftStance_564_623.bvh'
    # test_file = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_angryWalk\leftStance\walk_angry_normalwalking_2_leftStance_0_93.bvh'
    bvhreader = BVHReader(test_file)
    frame_len = len(bvhreader.frames)
    print('number of frames: ', frame_len)
    filename = os.path.split(test_file)[-1]

    test_data = process_file(test_file, sliding_window=True)
    assert test_data is not None
    test_data = np.swapaxes(test_data, 1, 2)
    print('input clips: ', test_data.shape)
    if test_data.shape[2] % 2 != 0:
        test_data = test_data[:, :, :-1]
    n_samples, n_dims, n_frames = test_data.shape
    normalized_data = (test_data - preprocess['Xmean']) / preprocess['Xstd']
    input = tf.placeholder(tf.float32, shape=[1, n_dims, n_frames])
    encoder_op = motion_encoder_channel_first(input, name='encoder', hidden_units=256, pooling='spectrum',
                                              kernel_size=25)
    # print(encoder_op.shape)
    decoder_op = motion_decoder_channel_first(encoder_op, n_dims, name='decoder', unpool='spectrum', kernel_size=25)

    # encoder_op = motion_encoder_channel_first(input, name='encoder', hidden_units=256, pooling='average')
    # decoder_op = motion_decoder_channel_first(encoder_op, n_dims, name='decoder', unpool='average')

    pool_input = tf.placeholder(dtype=tf.float32, shape=[1, n_dims, n_frames])
    pooled_decoder = spectrum_pooling_1d(pool_input, pool_size=2, N=512)
    # print(decoder_op.shape)
    unpooled_decoder = spectrum_unpooling_1d(pooled_decoder, pool_size=2, N=512)

    # average_pooled_decoder = tf.layers.average_pooling1d(input, 2, strides=2, data_format='channels_first')
    # average_unpooled_decoder = average_unpooling_1d(average_pooled_decoder, 2, data_format='channels_first')
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    reconstructed_clips = []
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'data/core_network_spectrum_pooling_250_0.00001.ckpt')
        # saver.restore(sess, 'data/core_network_average_pooling_500.ckpt')
        for i in range(n_samples):
            # reconstructed_motion = sess.run(average_unpooled_decoder, feed_dict={input: normalized_data[i: i+1]})
            motion_decoder = sess.run(decoder_op, feed_dict={input: normalized_data[i:i+1]})
            # reconstructed_motion = sess.run(unpooled_decoder, feed_dict={pool_input: normalized_data[i:i+1]})
            # mag = np.sqrt(np.real(spectrum)**2 + np.imag(spectrum)**2)
            # decodered_motion = sess.run(decoder_op, feed_dict={input: normalized_data[i: i + 1]})
            reconstructed_motion = sess.run(unpooled_decoder, feed_dict={pool_input: motion_decoder})
            # for j in range(n_dims):
            #     fig = plt.figure()
            #     plt.plot(range(n_frames), reconstructed_motion[i, j, :], label='after pooling')
            #     plt.plot(range(n_frames), normalized_data[i, j, :], label='before pooling')
            #     # plt.plot(range(512), mag[0, j, :], label='spectrum')
            #     plt.title(str(j))
            #     plt.legend()
            #     plt.show()

            reconstructed_motion = (motion_decoder * preprocess['Xstd']) + preprocess['Xmean']
            reconstructed_clips.append(reconstructed_motion[0])
        # reconstructed_clips = np.asarray(reconstructed_clips)
        reconstructed_clips = np.swapaxes(reconstructed_clips, 1, 2)
        print(reconstructed_clips.shape)
        combined_motion = combine_motion_clips(reconstructed_clips, frame_len-1, 120)
        combined_motion = np.swapaxes(combined_motion, 0, 1)[np.newaxis, :, :]
        # fig = plt.figure()
        # plt.plot(range(240), test_data[2, 24, :], label='before pooling')
        # # plt.plot(range(240), normalized_data[2, 24, :], label='normalized before pooling')
        # plt.plot(range(240), reconstructed_clips[2, 24, :], label='after pooling')
        # plt.legend()
        # plt.show()

        export_point_cloud_data(combined_motion, os.path.join(r'E:\tmp', filename[:-4]+'.panim'))



    # test_data = np.swapaxes(test_data, 1, 2)
    #
    # combined_motion = combine_motion_clips(test_data, frame_len-1, 120)
    # print(combined_motion.shape)
    # reconstructed_motion = np.swapaxes(combined_motion, 0, 1)[np.newaxis, :, :]
    # # reconstruct_motion_from_processed_data(reconstructed_motion, skeleton, r'E:\tmp\locomotion_walk_001_000_combined.bvh')
    # export_point_cloud_data(reconstructed_motion, r'E:\tmp\locomotion_walk_001_000_combined.panim')


def demo_motion_encoder_single_file():
    preprocess = np.load(r'E:\workspace\tensorflow_results\data\preprocessed_core_channel_first.npz')
    # test_file = r'C:\Users\hadu01\Downloads\fix-screws-by-hand\fix-screws-by-hand_007-snapPoseSkeleton.bvh'
    # test_file = r'E:\workspace\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\ACCAD\Female1_bvh\Female1_A08_CrouchToLie.bvh'
    test_file = r'E:\workspace\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\edin\edin_locomotion\locomotion_jog_000_000.bvh'
    # test_file = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\edin\edin_locomotion\locomotion_walk_001_000.bvh'
    # test_file = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\turnRightLeftStance_game_engine_skeleton_smoothed_grounded\walk_turnwalk_013_4_turnRight_204_347.bvh'
    # test_file = r'E:\tmp\style_transfer_motion.panim'
    filename = os.path.split(test_file)[-1]
    # test_data = process_file(test_file, sliding_window=False, with_game_engine=False, body_plane_indice=[2, 17, 13])
    test_data = process_file(test_file, sliding_window=False)
    # test_data = process_panim_data(test_file, sliding_window=False)
    print(test_data.shape)

    test_data = np.swapaxes(test_data, 0, 1)[np.newaxis, :, :]

    if test_data.shape[2] % 2 != 0:
        test_data = test_data[:, :, :-1]

    n_samples, n_dims, n_frames = test_data.shape
    normalized_data = (test_data - preprocess['Xmean']) / preprocess['Xstd']
    input = tf.placeholder(tf.float32, shape=[1, n_dims, n_frames])
    encoder_op = motion_encoder_channel_first(input, name='encoder', hidden_units=256, pooling='average')
    decoder_op = motion_decoder_channel_first(encoder_op, n_dims, name='decoder', unpool='average')
    # encoder_op = motion_encoder_channel_first(input, name='encoder', hidden_units=256, pooling='spectrum')
    # decoder_op = motion_decoder_channel_first(encoder_op, n_dims, name='decoder', unpool='spectrum')
    # print(encoder_op.shape)
    # print(decoder_op.shape)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, 'data/core_network_64_hidden_0.0001_250.ckpt')
        # saver.restore(sess, 'data/core_network_spectrum_pooling_250_0.00001.ckpt')
        saver.restore(sess, r'E:\workspace\tensorflow_results\data\core_network_average_pooling_300.ckpt')
        encoded_motion = sess.run(encoder_op, feed_dict={input: normalized_data})
        print('encoded motion shape: ', encoded_motion.shape)
        reconstructed_motion = sess.run(decoder_op, feed_dict={input: normalized_data})
        # denoised_motion = sess.run(decoder_op, feed_dict={input: reconstructed_motion})
        reconstructed_motion = (reconstructed_motion * preprocess['Xstd']) + preprocess['Xmean']
        # export_point_cloud_data(reconstructed_motion, os.path.join(r'E:\tmp', 'denoised_motion.panim'))
        # denoised_motion = (denoised_motion * preprocess['Xstd']) + preprocess['Xmean']

        print(reconstructed_motion.shape)
        export_point_cloud_data(np.swapaxes(reconstructed_motion[0], 0, 1), os.path.join(r'E:\workspace\tmp', filename[:-4]+'_average_pooling.panim'))
        # export_point_cloud_data(denoised_motion, os.path.join(r'E:\tmp', 'denoised_motion.panim'))


def demo_motion_autoencoder_channel_first():
    # data = np.load(r'../theano/data/training_data/processed_edin_data.npz')['clips']
    # data = np.swapaxes(data, 1, 2)
    # n_samples, n_dims, n_frames = data.shape
    preprocess = np.load('preprocessed_core_channel_first.npz')
    # normalized_data = (data - preprocess['Xmean']) / preprocess['Xstd']
    test_folder = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\cmu\01'
    save_folder = r'E:\tmp\01'
    for test_file in glob.glob(os.path.join(test_folder, '*.bvh')):
        test_data = process_file(test_file, sliding_window=False)
        test_filename = os.path.split(test_file)[-1]
        n_frames, n_dims = test_data.shape
        # test_file = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\edin\edin_locomotion\locomotion_walk_001_000.bvh'
        # test_file = r'E:\tmp\style_transfer_result.panim'
        #     bvhreader = BVHReader(test_file)
        #     skeleton = SkeletonBuilder().load_from_bvh(bvhreader, animated_joints=GAME_ENGINE_ANIMATED_JOINTS)
        # test_data = process_panim_data(test_file, sliding_window=False)
        test_data = np.swapaxes(test_data, 0, 1)[np.newaxis, :, :]
        normalized_data = (test_data - preprocess['Xmean']) / preprocess['Xstd']
        input = tf.placeholder(tf.float32, shape=[None, n_dims, n_frames])
        encoder_op = motion_encoder_channel_first(input, name='encoder', hidden_units=256, pooling='average')
        decoder_op = motion_decoder_channel_first(encoder_op, n_dims, name='decoder')
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            # saver.restore(sess, 'data/core_network_max_pooling.ckpt')
            saver.restore(sess, 'data/core_network_average_pooling_500.ckpt')


            # reconstructed_motion = sess.run(decoder_op, feed_dict={input: normalized_data[13:14]})
            reconstructed_motion = sess.run(decoder_op, feed_dict={input: normalized_data})
            reconstructed_motion = (reconstructed_motion * preprocess['Xstd']) + preprocess['Xmean']
            # reconstructed_motion = np.swapaxes(reconstructed_motion, 1, 2)

            export_point_cloud_data(reconstructed_motion, os.path.join(save_folder, test_filename[:-4]+'.panim'))
        save_motion_data_to_bvh(reconstructed_motion, GAME_ENGINE_SKELETON, test_file,
                                os.path.join(save_folder, test_filename))
        export_point_cloud_data(test_data, r'E:\tmp\origin.panim')
        export_point_cloud_data(reconstructed_motion, r'E:\tmp\denoised.panim')


def demo_autoencoder_strides():
    from models import motion_encoder_stride, motion_decoder_stride, motion_decoder_stride2d
    preprocess = np.load('preprocessed_core_channel_first.npz')
    test_file = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\edin\edin_locomotion\locomotion_walk_001_000.bvh'
    test_data = process_file(test_file, sliding_window=False)
    n_frames, n_dims = test_data.shape
    test_data = np.swapaxes(test_data, 0, 1)[np.newaxis, :, :]
    normalized_data = (test_data - preprocess['Xmean']) / preprocess['Xstd']
    input = tf.placeholder(tf.float32, [1, n_dims, n_frames])
    encoder_op = motion_encoder_stride(input, name='encoder')
    # decoder_op = motion_decoder_stride(encoder_op, n_dims, name='decoder')
    decoder_op = motion_decoder_stride2d(encoder_op, n_dims, name='decoder')
    encoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    decoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
    saver = tf.train.Saver(encoder_params + decoder_params)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
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
    input = tf.placeholder(tf.float32, [1, n_dims, n_frames])
    # encoder_op = motion_encoder_stride_multilayers(input, name='encoder')
    #
    # decoder_op = motion_decoder_stride2d_multilayers(encoder_op, n_dims, name='decoder')
    encoder_op = motion_encoder_multilayers(input, name='encoder')
    decoder_op = motion_decoder_multilayers(encoder_op, n_dims, name='decoder')
    encoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    decoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
    saver = tf.train.Saver(encoder_params + decoder_params)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
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
    input = tf.placeholder(tf.float32, [None, n_dims, n_frames])
    encoder_op = motion_encoder_channel_first(input, name='encoder')
    decoder_op = motion_decoder_channel_first(encoder_op, n_dims, name='decoder')

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
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
    input = tf.placeholder(tf.float32, [None, n_dims, n_frames])
    encoder_op = motion_encoder_channel_first(input)
    decoder_op = motion_decoder_channel_first(encoder_op, n_dims)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    save_filename = r'E:\tmp\reconstructed_expmap.bvh'
    with tf.Session(config=config) as sess:
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
    input = tf.placeholder(tf.float32, [None, n_dims, n_frames])
    encoder_op = motion_encoder_channel_first(input)
    decoder_op = motion_decoder_channel_first(encoder_op, n_dims)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    save_filename = r'E:\tmp\reconstructed_expmap.bvh'
    with tf.Session(config=config) as sess:
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

    test_file = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\edin\edin_locomotion\locomotion_walk_001_000.bvh'
    test_data = process_file(test_file, sliding_window=False)
    n_frames, n_dims = test_data.shape
    test_data = np.swapaxes(test_data, 0, 1)[np.newaxis, :, :]
    normalized_data = (test_data - preprocess['Xmean']) / preprocess['Xstd']
    input = tf.placeholder(tf.float32, [None, n_dims, n_frames])
    encoder_op = motion_encoder_without_pooling(input)
    decoder_op = motion_decoder_without_pooling(encoder_op, n_dims)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)
        saver.restore(sess, 'data/core_network_without_pooling')
        # reconstructed_motion = sess.run(decoder_op, feed_dict={input: normalized_data[13:14]})
        reconstructed_motion = sess.run(decoder_op, feed_dict={input: normalized_data})
        reconstructed_motion = (reconstructed_motion * preprocess['Xstd']) + preprocess['Xmean']

        # reconstructed_motion = np.swapaxes(reconstructed_motion, 1, 2)

        export_point_cloud_data(reconstructed_motion, r'E:\tmp\locomotion_walk_001_000_no_pooling.panim')



if __name__ == "__main__":
    # demo_motion_autoencoder()
    # demo_motion_autoencoder_expmap()
    # demo_motion_autoencoder_channel_first()
    # demo_motion_autoencoder_channel_first_without_pooling()
    # demo_motion_autoencoder_channel_first_combined()
    # demo_autoencoder_strides()
    # demo_autoencoder_strides_multilayers()
    demo_motion_encoder_single_file()
    # demo_motion_encoder_spectrum_pooling_single_file()