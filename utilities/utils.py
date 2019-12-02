import tensorflow as tf 
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from mosi_utils_anim.utilities import write_to_json_file
from .quaternions import Quaternions
import numpy as np
import collections
import glob
import os


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


def gram_matrix(X):
    '''
    X shape: n_batches * n_dims * n_frames
    :param X:
    :return: gram_matrix n_batches * n_dims * n_dims, sum over n_frames
    '''
    return tf.reduce_mean(input_tensor=tf.expand_dims(X, 2) * tf.expand_dims(X, 1), axis=3)


def convert_anim_to_point_cloud_tf(anim):
    '''

    :param anim:
    :return:
    '''
    joints, v_x, v_z, v_r = anim[:, :-3], anim[:, -3], anim[:, -2], anim[:, -1]
    joints = tf.reshape(joints, (joints.shape[0], -1, 3))
    n_frames, n_joints = joints.shape[0], joints.shape[1]
    v_r = tf.reshape(tf.cumsum(v_r), (n_frames, 1))
    """ Rotate motion about Y axis """
    v_x = tf.reshape(v_x, (n_frames, 1))
    v_z = tf.reshape(v_z, (n_frames, 1))
    #### create rotation matrix
    sin_theta = tf.sin(v_r)
    cos_theta = tf.cos(v_r)
    rotmat = tf.concat((cos_theta, tf.zeros((n_frames, 1)), sin_theta, tf.zeros((n_frames, 1)), tf.zeros((n_frames, 1)),
                        tf.ones((n_frames, 1)), tf.zeros((n_frames, 1)), tf.zeros((n_frames, 1)), -sin_theta, 
                        tf.zeros((n_frames, 1)), cos_theta, tf.zeros((n_frames, 1)), tf.zeros((n_frames, 1)), 
                        tf.zeros((n_frames, 1)), tf.zeros((n_frames, 1)), tf.ones((n_frames, 1))), axis=-1)
    rotmat = tf.reshape(rotmat, (n_frames, 4, 4))

    ones = tf.ones((n_frames, n_joints, 1))
    extended_joints = tf.concat((joints, ones), axis=-1)
    swapped_joints = tf.transpose(a=extended_joints, perm=(0, 2, 1))
    # print('swapped joints shape: ', swapped_joints.shape)
    rotated_joints = tf.matmul(rotmat, swapped_joints)
    rotated_joints = tf.transpose(a=rotated_joints, perm=(0, 2, 1))[:, :, :-1]
    """ Rotate Velocity"""
    trans = tf.concat((v_x, tf.zeros((n_frames, 1)), v_z, tf.ones((n_frames, 1))), axis=-1)
    trans = tf.expand_dims(trans, 1)
    swapped_trans = tf.transpose(a=trans, perm=(0, 2, 1))
    rotated_trans = tf.matmul(rotmat, swapped_trans)
    rotated_trans = tf.transpose(a=rotated_trans, perm=(0, 2, 1))
    v_x = rotated_trans[:, :, 0]
    v_z = rotated_trans[:, :, 2]
    v_x, v_z = tf.cumsum(v_x, axis=0), tf.cumsum(v_z, axis=0)
    rotated_trans = tf.concat((v_x, tf.zeros((n_frames, 1)), v_z), axis=-1)
    rotated_trans = tf.expand_dims(rotated_trans, 1)
    export_joints = rotated_joints + rotated_trans
    return export_joints


def export_point_cloud_data_without_foot_contact(motion_data, filename=None, skeleton=GAME_ENGINE_SKELETON):
    '''
    
    :param motion_data: n_frames * n_dims 
    :param filename: 
    :return: 
    '''
    motion_data = copy.deepcopy(motion_data)
    joints, root_x, root_z, root_r = motion_data[:, :-3], motion_data[:, -3], motion_data[:, -2], motion_data[:, -1]
    joints = joints.reshape((len(joints), -1, 3))  ### reshape 2D matrix to (n_frames, n_joints, n_dims)
    rotation = Quaternions.id(1)
    translation = np.array([[0, 0, 0]])
    # ref_dir = np.array([0, 0, 1])
    for i in range(len(joints)):
        joints[i, :, :] = rotation * joints[i]
        joints[i, :, 0] = joints[i, :, 0] + translation[0, 0]
        joints[i, :, 2] = joints[i, :, 2] + translation[0, 2]
        rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0, 1, 0])) * rotation
        # offsets.append(rotation * ref_dir)
        translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])
    if filename is None:
        return joints
    else:
        save_data = {'motion_data': joints.tolist(), 'has_skeleton': True, 'skeleton': skeleton}
        write_to_json_file(filename, save_data)


def combine_motion_clips(clips, motion_len, window_step):
    '''
    
    :param clips: 
    :param motion_len: 
    :param window_step: 
    :return: 
    '''
    clips = np.asarray(clips)
    n_clips, window_size, n_dims = clips.shape

    ## case 1: motion length is smaller than window_step
    if motion_len <= window_step:
        assert n_clips == 1
        left_index = (window_size - motion_len) // 2 + (window_size - motion_len) % 2
        right_index = window_size - (window_size - motion_len) // 2
        return clips[0][left_index: right_index]

    ## case 2: motion length is larger than window_step and smaller than window
    if motion_len > window_step and motion_len <= window_size:
        assert n_clips == 2
        left_index = (window_size - motion_len) // 2 + (window_size - motion_len) % 2
        right_index = window_size - (window_size - motion_len) // 2
        return clips[0][left_index: right_index]

    residue_frames = motion_len % window_step
    print('residue_frames: ', residue_frames)
    ## case 3: residue frames is smaller than window step
    if residue_frames <= window_step:
        residue_frames += window_step
        combined_frames = np.concatenate(clips[0:n_clips-2, :window_step], axis=0)
        left_index = (window_size - residue_frames) // 2 + (window_size - residue_frames) % 2
        right_index = window_size - (window_size - residue_frames) // 2
        combined_frames = np.concatenate((combined_frames, clips[-2, left_index:right_index]), axis=0)
        return combined_frames


def get_files(dir, suffix='.bvh', files=[]):
    files += glob.glob(os.path.join(dir, '*'+suffix))
    for subdir in next(os.walk(dir))[1]:
        get_files(os.path.join(dir, subdir), suffix, files)