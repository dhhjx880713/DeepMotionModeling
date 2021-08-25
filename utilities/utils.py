import tensorflow as tf 
import copy
import sys
from pathlib import Path

from tensorflow.python.ops.gen_array_ops import pad_v2
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from mosi_utils_anim.utilities import write_to_json_file
from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder
from mosi_utils_anim.animation_data.quaternion import Quaternion
from .quaternions import Quaternions
import numpy as np
import collections
import glob
import os
from .skeleton_def import MH_CMU_SKELETON, GAME_ENGINE_SKELETON
from transformations import rotation_matrix


def transform_point_cloud_3d(point_cloud_data, rotation_angle, rotation_axis, translation):
    """

    Args:
        point_cloud_data (numpy.array): n_frames * n_joints * 3
        rotation_angle (float): angle in degree
        rotation_axis (numpy.array): (x, y, z)
        translation (numpy.array): (x, y, z)
    """
    n_frames, n_joints, n_dims = point_cloud_data.shape
    #### get rotation matrix
    rotmat = rotation_matrix(np.deg2rad(rotation_angle), rotation_axis)
    #### extend point cloud matrix to 4x4
    ones = np.ones((n_frames, n_joints, 1))
    transmat = np.concatenate((point_cloud_data, ones), axis=-1)
    swapped_transmat = np.transpose(transmat, (0, 2, 1))
    transmat = np.matmul(rotmat, swapped_transmat)
    transmat = np.transpose(transmat, (0, 2, 1))
    return transmat[:, :, :3] + translation


def gram_matrix(X):
    '''
    X shape: n_batches * n_dims * n_frames
    :param X:
    :return: gram_matrix n_batches * n_dims * n_dims, sum over n_frames
    '''
    return tf.reduce_mean(input_tensor=tf.expand_dims(X, 2) * tf.expand_dims(X, 1), axis=3)


def convert_anim_to_point_cloud_tf(anim):
    ''' the preprocessing steps are different from Holden's code, the rotation difference is not computed by quaterion
        but by the difference between angles

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


def get_global_position_framewise(anim_data, up_axis=np.array([0, 1, 0])):
    """convert preprocess motion data (normalized joint positions T * 3 + global transformation and rotation)

    Args:
        anim_data (numpy.array): T * 3 + 3
    """
    relative_joint_positions, v_x, v_z, v_r = anim_data[:, :-3], anim_data[:, -3], anim_data[:, -2], anim_data[:, -1]
    v_r = np.cumsum(v_r)
    n_frames = len(relative_joint_positions)
    qs = Quaternions.from_angle_axis(v_r, up_axis)
    relative_joint_positions = relative_joint_positions.reshape((n_frames, -1, 3))
    global_joint_positions = []
    for i in range(n_frames):
        rotated_joints = qs[i] * relative_joint_positions[i]
        global_joint_positions.append(rotated_joints + qs[i] * np.array([v_x[i], 0, v_z[i]]))
    return np.asarray(global_joint_positions)


def get_global_position_framewise_tf(anim_data, up_axis=np.array([0, 1, 0])):
    relative_joint_positions, v_x, v_z, v_r = anim_data[:, :-3], anim_data[:, -3], anim_data[:, -2], anim_data[:, -1]
    n_frames = relative_joint_positions.shape[0]
    relative_joint_positions = tf.reshape(relative_joint_positions, (n_frames, -1, 3))
    n_joints = relative_joint_positions.shape[1]
    v_r = tf.reshape(tf.cumsum(v_r), (n_frames, 1))
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
    extended_joints = tf.concat((relative_joint_positions, ones), axis=-1)
    swapped_joints = tf.transpose(a=extended_joints, perm=(0, 2, 1))    
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

    rotated_trans = tf.concat((v_x, tf.zeros((n_frames, 1)), v_z), axis=-1)
    rotated_trans = tf.expand_dims(rotated_trans, 1)
    export_joints = rotated_joints + rotated_trans
    return export_joints


def get_global_position(relative_joint_position):
    """sequentially concatenate joint positions in relative coordinate into global coordinate

    Args:
        relative_joints (numpy.array): N_frames * 3
    """
    relative_joint_position = np.asarray(relative_joint_position)
    assert relative_joint_position.shape[-1] == 3
    n_frames = len(relative_joint_position)
    for i in range(n_frames-1):
        relative_joint_position[i+1][:, 0] += relative_joint_position[i][0, 0]
        relative_joint_position[i+1][:, 2] += relative_joint_position[i][0, 2]
    return relative_joint_position


def export_point_cloud_data_without_foot_contact(motion_data, filename=None, skeleton=MH_CMU_SKELETON, scale_factor=1):
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
    joints *= scale_factor
    if filename is None:
        return joints
    else:
        save_data = {'motion_data': joints.tolist(), 'has_skeleton': True, 'skeleton': skeleton}
        write_to_json_file(filename, save_data)


def reconstruct_global_position(motion_data):
    """
    
    Arguments:
        motion_data {numpy.array} -- n_frames * n_dims
    """
    relative_positions, root_x, root_z, root_r = motion_data[:, :-3], motion_data[:, -3], motion_data[:, -2], motion_data[:, -1]
    n_frames = len(relative_positions)
    relative_positions = relative_positions.reshape(n_frames, -1, 3)
    rotation = Quaternions.id(1)
    new_frames = []
    translation = np.zeros(3)
    for i in range(n_frames):
        rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0, 1, 0])) * rotation
        relative_positions[i] = rotation * relative_positions[i]
        translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])
        new_frames.append(relative_positions[i] + translation)
        # relative_positions[i] = rotation * relative_positions[i]
        # relative_positions[i] = relative_positions[i] + translation
        # rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0, 1, 0])) * rotation
        # translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])
        # new_frames.append(relative_positions[i])
    return np.asarray(new_frames)



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


def get_rotation_to_ref_direction(dir_vecs, ref_dir):
    
    rotations = []
    for dir_vec in dir_vecs:
        rotations.append(Quaternion.between(dir_vec, ref_dir))
    return rotations


def rotate_cartesian_frame(cartesian_frame, q):
    '''
    rotate a cartesian frame by given quaternion q
    :param cartesian_frame: ndarray (n_joints * 3)
    :param q: Quaternion
    :return:
    '''
    new_cartesian_frame = np.zeros(cartesian_frame.shape)
    for i in range(len(cartesian_frame)):
        new_cartesian_frame[i] = q * cartesian_frame[i]
    return new_cartesian_frame


def estimate_ground_height(cartesian_frames, fid_l, fid_r):
    foot_heights = np.minimum(cartesian_frames[:, fid_l, 1], cartesian_frames[:, fid_r, 1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)    
    return floor_height

def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    return -softmax(-x, **kw)


def average_euclidean_distance(pd_1, pd_2):
    """compute the average euclidean distance per joint between two point cloud

    Args:
        pd_1 (numpy.array): multidimensional array, the last dimension is (x, y, z)
        pd_2 (numpy.array): multidimensional array, the last dimension is (x, y, z)
    """
    assert pd_1.shape == pd_2.shape
    reshaped_pd_1 = np.reshape(pd_1, (np.prod(pd_1.shape[:-1]), pd_1.shape[-1]))
    reshaped_pd_2 = np.reshape(pd_2, (np.prod(pd_2.shape[:-1]), pd_2.shape[-1]))
    return np.linalg.norm(reshaped_pd_1 - reshaped_pd_2, axis=1).mean()