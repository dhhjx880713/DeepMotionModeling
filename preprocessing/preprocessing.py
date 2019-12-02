# encoding: UTF-8
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from mosi_utils_anim.animation_data import BVHReader, Skeleton, SkeletonBuilder
from mosi_utils_anim.animation_data.utils import convert_euler_frames_to_cartesian_frames, \
    convert_quat_frames_to_cartesian_frames, get_rotation_angles_for_vectors, \
    pose_orientation_euler, rotate_around_y_axis
from preprocessing.utils import rotate_cartesian_frames_to_ref_dir, cartesian_pose_orientation, rotation_cartesian_frames
from mosi_utils_anim.animation_data.quaternion import Quaternion
import numpy as np
import collections
import os
import glob
import argparse
import scipy.ndimage.filters as filters
from utilities.utils import get_files


GAME_ENGINE_ANIMATED_JOINTS = ['Game_engine', 'Root', 'pelvis', 'spine_03', 'clavicle_l', 'upperarm_l', 'lowerarm_l',
                               'hand_l', 'clavicle_r',
                               'upperarm_r', 'lowerarm_r', 'hand_r', 'neck_01', 'head', 'thigh_l', 'calf_l', 'foot_l',
                               'ball_l', 'thigh_r', 'calf_r', 'foot_r', 'ball_r']


GAME_ENGINE_ANIMATED_JOINTS_without_game_engine = ['Root', 'pelvis', 'spine_03', 'clavicle_l', 'upperarm_l', 'lowerarm_l',
                                                   'hand_l', 'clavicle_r', 'upperarm_r', 'lowerarm_r', 'hand_r',
                                                   'neck_01', 'head', 'thigh_l', 'calf_l', 'foot_l',
                                                   'ball_l', 'thigh_r', 'calf_r', 'foot_r', 'ball_r']


Edinburgh_animated_joints = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg',
                             'RightFoot', 'RightToeBase', 'Spine', 'Spine1', 'Neck1', 'Head', 'LeftArm', 'LeftForeArm',
                             'LeftHand', 'LeftHandIndex1', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandIndex1']


MH_CMU_ANIMATED_JOINTS = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LowerBack', 'Spine',
                          'Spine1', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LThumb', 'LeftFingerBase',
                          'LeftHandFinger1', 'Neck', 'Neck1', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm',
                          'RightHand', 'RThumb', 'RightFingerBase', 'RightHandFinger1', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase']

MH_CMU_SKELETON = collections.OrderedDict(
    [
        ('Hips', {'parent': None, 'index': 0}),
        ('LeftUpLeg', {'parent': 'Hips', 'index': 1}),
        ('LeftLeg', {'parent': 'LeftUpLeg', 'index': 2}),
        ('LeftFoot', {'parent': 'LeftLeg', 'index': 3}),
        ('LeftToeBase', {'parent': 'LeftFoot', 'index': 4}),
        ('LowerBack', {'parent': 'Hips', 'index': 5}),
        ('Spine', {'parent': 'LowerBack', 'index': 6}),
        ('Spine1', {'parent': 'Spine', 'index': 7}),
        ('LeftShoulder', {'parent': 'Spine1', 'index': 8}),
        ('LeftArm', {'parent': 'LeftShoulder', 'index': 9}),
        ('LeftForeArm', {'parent': 'LeftArm', 'index': 10}),
        ('LeftHand', {'parent': 'LeftForeArm', 'index': 11}),
        ('LThumb', {'parent': 'LeftHand', 'index': 12}),
        ('LeftFingerBase', {'parent': 'LeftHand', 'index': 13}),
        ('LeftHandFinger1', {'parent': 'LeftFingerBase', 'index': 14}),
        ('Neck', {'parent': 'Spine1', 'index': 15}),
        ('Neck1', {'parent': 'Neck', 'index': 16}),
        ('Head', {'parent': 'Neck1', 'index': 17}),
        ('RightShoulder', {'parent': 'Spine1', 'index': 18}),
        ('RightArm', {'parent': 'RightShoulder', 'index': 19}),
        ('RightForeArm', {'parent': 'RightArm', 'index': 20}),
        ('RightHand', {'parent': 'RightForeArm', 'index': 21}),
        ('RThumb', {'parent': 'RightHand', 'index': 22}),
        ('RightFingerBase', {'parent': 'RightHand', 'index': 23}),
        ('RightHandFinger1', {'parent': 'RightFingerBase', 'index': 24}),
        ('RightUpLeg', {'parent': 'Hips', 'index': 25}),
        ('RightLeg', {'parent': 'RightUpLeg', 'index': 26}),
        ('RightFoot', {'parent': 'RightLeg', 'index': 27}),
        ('RightToeBase', {'parent': 'RightFoot', 'index': 28})
    ]
)


GAME_ENGINE_SKELETON = collections.OrderedDict(
    [
        ('Root', {'parent': None, 'index': 0}),
        ('pelvis', {'parent': 'Root', 'index': 1}),
        ('spine_03', {'parent': 'pelvis', 'index': 2}),
        ('clavicle_l', {'parent': 'spine_03', 'index': 3}),
        ('upperarm_l', {'parent': 'clavicle_l', 'index': 4}),
        ('lowerarm_l', {'parent': 'upperarm_l', 'index': 5}),
        ('hand_l', {'parent': 'lowerarm_l', 'index': 6}),
        ('clavicle_r', {'parent': 'spine_03', 'index': 7}),
        ('upperarm_r', {'parent': 'clavicle_r', 'index': 8}),
        ('lowerarm_r', {'parent': 'upperarm_r', 'index': 9}),
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

Edinburgh_skeleton = collections.OrderedDict(
    [
        ('Root', {'parent': None, 'index': 0}),
        ('Hips', {'parent': 'Root', 'index': 1}),
        ('LeftUpLeg', {'parent': 'Hips', 'index': 2}),
        ('LeftLeg', {'parent': 'LeftUpLeg', 'index': 3}),
        ('LeftFoot', {'parent': 'LeftLeg', 'index': 4}),
        ('LeftToeBase', {'parent': 'LeftFoot', 'index': 5}),
        ('RightUpLeg', {'parent': 'Hips', 'index': 6}),
        ('RightLeg', {'parent': 'RightUpLeg', 'index': 7}),
        ('RightFoot', {'parent': 'RightLeg', 'index': 8}),
        ('RightToeBase', {'parent': 'RightFoot', 'index': 9}),
        ('Spine', {'parent': 'Hips', 'index': 10}),
        ('Spine1', {'parent': 'Spine', 'index': 11}),
        ('Neck1', {'parent': 'Spine1', 'index': 12}),
        ('Head', {'parent': 'Neck1', 'index': 13}),
        ('LeftArm', {'parent': 'Spine1', 'index': 14}),
        ('LeftForeArm', {'parent': 'LeftArm', 'index': 15}),
        ('LeftHand', {'parent': 'LeftForeArm', 'index': 16}),
        ('LeftHandIndex1', {'parent': 'LeftHand', 'index': 17}),
        ('RightArm', {'parent': 'Spine1', 'index': 18}),
        ('RightForeArm', {'parent': 'RightArm', 'index': 19}),
        ('RightHand', {'parent': 'RightForeArm', 'index': 20}),
        ('RightHandIndex1', {'parent': 'RightHand', 'index': 21})
    ]
)


def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    return -softmax(-x, **kw)


def max_smoothing(x, window_size):
    '''
    smooth a binary signal by majority vote
    :param x:
    :param window_size:
    :return:
    '''
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2
    extented_x = np.zeros(2 * half_window + len(x))
    extented_x[:half_window] = x[0]
    extented_x[-half_window:] = x[-1]
    extented_x[half_window: -half_window] = x
    new_x = np.zeros(len(x))
    for i in range(len(x)):
        vs = np.average(extented_x[i: i+window_size])
        if vs >= 0.5:
            new_x[i] = 1
        else:
            new_x[i] = 0
    return new_x


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


def process_bvhfile(filename, 
                    window=240, 
                    window_step=120, 
                    body_plane_indices=None, 
                    fid_l = np.array([15, 16]),
                    fid_r = np.array([19, 20]),
                    sliding_window=True,
                    animated_joints=MH_CMU_ANIMATED_JOINTS):

    '''
    extract normalized point cloud data from bvh file

    '''
    print(filename)
    bvhreader = BVHReader(filename)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    ref_dir = np.array([0, 0, 1])
    up_axis = np.array([0, 1, 0])
    cartesian_frames = convert_euler_frames_to_cartesian_frames(skeleton, bvhreader.frames,
                                                                animated_joints=animated_joints)
    cartesian_frames = rotate_cartesian_frames_to_ref_dir(cartesian_frames, ref_dir, body_plane_indices, up_axis)
    #### duplicate the first cartesian frame to make sure the output frame length is the same as input
    cartesian_frames = np.concatenate((cartesian_frames[0:1], cartesian_frames), axis=0)
    """ Put on Floor """
    # fid_l, fid_r = np.array([15, 16]), np.array([19, 20])

    foot_heights = np.minimum(cartesian_frames[:, fid_l, 1], cartesian_frames[:, fid_r, 1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    cartesian_frames = cartesian_frames - floor_height
    """ Compute forward direction for each frame """
    forward = []
    for i in range(len(cartesian_frames)):
        forward.append(cartesian_pose_orientation(cartesian_frames[i], body_plane_indices, up_axis))
    forward = np.asarray(forward)
    rotation_angles = get_rotation_angles_for_vectors(forward, ref_dir, up_axis)
    delta_angles = rotation_angles[1:] - rotation_angles[:-1]

    """ Get Root Velocity """
    velocity = (cartesian_frames[1:, 0:1] - cartesian_frames[:-1, 0:1]).copy()
    """ Remove Translation """
    cartesian_frames[:, :, 0] = cartesian_frames[:, :, 0] - cartesian_frames[:, 0:1, 0]
    cartesian_frames[:, :, 2] = cartesian_frames[:, :, 2] - cartesian_frames[:, 0:1, 2]
    """ Remove Y Rotation """
    cartesian_frames = rotation_cartesian_frames(cartesian_frames, -rotation_angles)
    """ Rotate speed vectory"""
    n_frames = len(rotation_angles) - 1
    # print(velocity.shape)
    velocity[:, :, 1] = 0
    ones = np.ones((velocity.shape[0], 1, 1))
    velocity = np.concatenate((velocity, ones), axis=-1)
    angles = -rotation_angles[1:]
    sin_theta = np.sin(angles)
    cos_theta = np.cos(angles)
    rotmat = np.array([cos_theta, np.zeros(n_frames), sin_theta, np.zeros(n_frames), np.zeros(n_frames),
                       np.ones(n_frames), np.zeros(n_frames), np.zeros(n_frames), -sin_theta, np.zeros(n_frames),
                       cos_theta, np.zeros(n_frames), np.zeros(n_frames), np.zeros(n_frames), np.zeros(n_frames),
                       np.ones(n_frames)]).T
    swapped_v_mat = np.transpose(velocity, (0, 2, 1))
    rotmat = np.reshape(rotmat, (n_frames, 4, 4))
    rotated_v = np.matmul(rotmat, swapped_v_mat)
    rotated_v = np.transpose(rotated_v, (0, 2, 1))
    """ Add Velocity, RVelocity """
    cartesian_frames = cartesian_frames[:-1]
    cartesian_frames = cartesian_frames.reshape(len(cartesian_frames), -1)
    cartesian_frames = np.concatenate([cartesian_frames, rotated_v[:, :, 0]], axis=-1)
    cartesian_frames = np.concatenate([cartesian_frames, rotated_v[:, :, 2]], axis=-1)
    cartesian_frames = np.concatenate([cartesian_frames, delta_angles[:, np.newaxis]], axis=-1)

    if sliding_window:
        """ Slide Over Windows """
        windows = []
        # windows_classes = []
        if len(cartesian_frames) % window_step == 0:
            n_clips = (len(cartesian_frames) - len(cartesian_frames) % window_step) // window_step
        else:
            n_clips = (len(cartesian_frames) - len(cartesian_frames) % window_step) // window_step + 1
        for j in range(0, n_clips):
            """ If slice too small pad out by repeating start and end poses """
            slice = cartesian_frames[j * window_step: j * window_step + window]
            if len(slice) < window:
                left = slice[:1].repeat((window - len(slice)) // 2 + (window - len(slice)) % 2, axis=0)
                right = slice[-1:].repeat((window - len(slice)) // 2, axis=0)
                slice = np.concatenate([left, slice, right], axis=0)
            if len(slice) != window: raise Exception()

            windows.append(slice)
        return np.asarray(windows)

    else:
        return cartesian_frames


def process_file(filename, window=240, window_step=120, sliding_window=True, animated_joints=MH_CMU_ANIMATED_JOINTS,
                 body_plane_indices=None, fid_l=[15, 16], fid_r=[19, 20]):
    """ Compute joint positions for animated joints """

    print(filename)
    bvhreader = BVHReader(filename)
    ref_dir = np.array([0, 0, 1])
    up_axis = np.array([0, 1, 0])

    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)

    cartesian_frames = convert_euler_frames_to_cartesian_frames(skeleton, bvhreader.frames,
                                                                animated_joints=animated_joints)                                                    
    cartesian_frames = np.concatenate((cartesian_frames[0:1], cartesian_frames), axis=0)  ## append the first frame for calculating velocity
    n_frames = len(cartesian_frames)
    forward = []
    for i in range(len(cartesian_frames)):
        forward.append(cartesian_pose_orientation(cartesian_frames[i], body_plane_indices, up_axis))
    forward = np.asarray(forward)
    rotations = get_rotation_to_ref_direction(forward, ref_dir=ref_dir)

    """ Put on Floor """

    foot_heights = np.minimum(cartesian_frames[:, fid_l, 1], cartesian_frames[:, fid_r, 1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)

    cartesian_frames = cartesian_frames - floor_height

    """ Get Root Velocity """
    velocity = (cartesian_frames[1:, 0:1] - cartesian_frames[:-1, 0:1]).copy()
    """ Remove Translation """
    cartesian_frames[:, :, 0] = cartesian_frames[:, :, 0] - cartesian_frames[:, 0:1, 0]
    cartesian_frames[:, :, 2] = cartesian_frames[:, :, 2] - cartesian_frames[:, 0:1, 2]
    """ Remove Y Rotation """
    for i in range(n_frames):
        cartesian_frames[i] = rotate_cartesian_frame(cartesian_frames[i], rotations[i])

    """ Rotate Velocity """
    for i in range(n_frames - 1):
        # print(rotations[i+1])

        velocity[i, 0] = rotations[i+1] * velocity[i, 0]
    """ Get Rotation Velocity """
    r_v = np.zeros(n_frames - 1)
    for i in range(n_frames - 1):
        q = rotations[i+1] * (-rotations[i])
        r_v[i] = Quaternion.get_angle_from_quaternion(q, ref_dir)

    """ Add Velocity, RVelocity, Foot Contacts to vector """
    cartesian_frames = cartesian_frames[1:]
    cartesian_frames = cartesian_frames.reshape(len(cartesian_frames), -1)
    cartesian_frames = np.concatenate([cartesian_frames, velocity[:, :, 0]], axis=-1)
    cartesian_frames = np.concatenate([cartesian_frames, velocity[:, :, 2]], axis=-1)
    cartesian_frames = np.concatenate([cartesian_frames, r_v[:, np.newaxis]], axis=-1)

    if sliding_window:
        """ Slide Over Windows """
        windows = []
        # windows_classes = []
        if len(cartesian_frames) % window_step == 0:
            n_clips = (len(cartesian_frames) - len(cartesian_frames) % window_step)//window_step 
        else:
            n_clips = (len(cartesian_frames) - len(cartesian_frames) % window_step) // window_step + 1
        for j in range(0, n_clips):
            """ If slice too small pad out by repeating start and end poses """
            slice = cartesian_frames[j * window_step : j * window_step + window]
            if len(slice) < window:
                left = slice[:1].repeat((window - len(slice)) // 2 + (window - len(slice)) % 2, axis=0)
                left[:, -7:-4] = 0.0
                right = slice[-1:].repeat((window - len(slice)) // 2, axis=0)
                right[:, -7:-4] = 0.0
                slice = np.concatenate([left, slice, right], axis=0)
            if len(slice) != window: raise Exception()

            windows.append(slice)
        return windows

    else:
        return cartesian_frames


def get_bvh_files(path):
    bvhfiles = []
    for root, dirs, files in os.walk(path):
        for filemane in [f for f in files if f.endswith(".bvh") and f != 'rest.bvh']:
            bvhfiles.append(os.path.join(root, filemane))
    return bvhfiles


def process_motion_vector(mv, window=240, window_step=120, sliding_window=True):
    cartesian_frames = convert_quat_frames_to_cartesian_frames(mv.skeleton, mv.frames,
                                                               animated_joints=GAME_ENGINE_ANIMATED_JOINTS)
    n_frames = len(cartesian_frames)
    forward = cartesian_frames[:, 0, :] - cartesian_frames[:, 1, :]
    forward[:, 1] = 0.0
    forward = forward/np.linalg.norm(forward, axis=-1)[:, np.newaxis]
    ref_dir = np.array([0, 0, 1])
    cartesian_frames = cartesian_frames[:, 1:, :]  ## remove 'Game_engine' joint
    rotations = get_rotation_to_ref_direction(forward, ref_dir=ref_dir)

    """ Put on Floor """
    fid_l, fid_r = np.array([15, 16]), np.array([19, 20])

    foot_heights = np.minimum(cartesian_frames[:, fid_l, 1], cartesian_frames[:, fid_r, 1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    cartesian_frames = cartesian_frames - floor_height

    """ Get Foot Contacts """
    velfactor, heightfactor = np.array([0.2, 0.2]), np.array([10.0, 5.0])

    feet_l_x = (cartesian_frames[1:, fid_l, 0] - cartesian_frames[:-1, fid_l, 0]) ** 2
    feet_l_y = (cartesian_frames[1:, fid_l, 1] - cartesian_frames[:-1, fid_l, 1]) ** 2
    feet_l_z = (cartesian_frames[1:, fid_l, 2] - cartesian_frames[:-1, fid_l, 2]) ** 2
    feet_l_h = cartesian_frames[:-1, fid_l, 1]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)

    feet_l[:, 0] = max_smoothing(feet_l[:, 0], 20)
    feet_l[:, 1] = max_smoothing(feet_l[:, 1], 20)


    feet_r_x = (cartesian_frames[1:, fid_r, 0] - cartesian_frames[:-1, fid_r, 0]) ** 2
    feet_r_y = (cartesian_frames[1:, fid_r, 1] - cartesian_frames[:-1, fid_r, 1]) ** 2
    feet_r_z = (cartesian_frames[1:, fid_r, 2] - cartesian_frames[:-1, fid_r, 2]) ** 2
    feet_r_h = cartesian_frames[:-1, fid_r, 1]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
    feet_r[:, 0] = max_smoothing(feet_r[:, 0], 20)
    feet_r[:, 1] = max_smoothing(feet_r[:, 1], 20)

    """ Get Root Velocity """
    velocity = (cartesian_frames[1:, 0:1] - cartesian_frames[:-1, 0:1]).copy()
    """ Remove Translation """
    cartesian_frames[:, :, 0] = cartesian_frames[:, :, 0] - cartesian_frames[:, 0:1, 0]
    cartesian_frames[:, :, 2] = cartesian_frames[:, :, 2] - cartesian_frames[:, 0:1, 2]
    """ Remove Y Rotation """
    for i in range(n_frames):
        cartesian_frames[i] = rotate_cartesian_frame(cartesian_frames[i], rotations[i])

    """ Rotate Velocity """
    for i in range(n_frames - 1):
        # print(rotations[i+1])

        velocity[i, 0] = rotations[i+1] * velocity[i, 0]
    """ Get Rotation Velocity """
    r_v = np.zeros(n_frames - 1)
    for i in range(n_frames - 1):
        q = rotations[i+1] * (-rotations[i])
        r_v[i] = Quaternion.get_angle_from_quaternion(q, np.array([0, 0, 1]))

    """ Add Velocity, RVelocity, Foot Contacts to vector """
    cartesian_frames = cartesian_frames[:-1]
    cartesian_frames = cartesian_frames.reshape(len(cartesian_frames), -1)
    cartesian_frames = np.concatenate([cartesian_frames, velocity[:, :, 0]], axis=-1)
    cartesian_frames = np.concatenate([cartesian_frames, velocity[:, :, 2]], axis=-1)
    cartesian_frames = np.concatenate([cartesian_frames, r_v[:, np.newaxis]], axis=-1)
    cartesian_frames = np.concatenate([cartesian_frames, feet_l, feet_r], axis=-1)

    if sliding_window:
        """ Slide Over Windows """
        windows = []
        # windows_classes = []
        if len(cartesian_frames) % window_step == 0:
            n_clips = (len(cartesian_frames) - len(cartesian_frames) % window_step) // window_step
        else:
            n_clips = (len(cartesian_frames) - len(cartesian_frames) % window_step) // window_step + 1
        for j in range(0, n_clips):
            """ If slice too small pad out by repeating start and end poses """
            slice = cartesian_frames[j * window_step: j * window_step + window]
            if len(slice) < window:
                left = slice[:1].repeat((window - len(slice)) // 2 + (window - len(slice)) % 2, axis=0)
                left[:, -7:-4] = 0.0
                right = slice[-1:].repeat((window - len(slice)) // 2, axis=0)
                right[:, -7:-4] = 0.0
                slice = np.concatenate([left, slice, right], axis=0)
            if len(slice) != window: raise Exception()

            windows.append(slice)
        return windows

    else:
        return cartesian_frames


def preprocess_cmu_data():
    clips = []
    accad_files = get_bvh_files(r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\cmu')

    for bvhfile in accad_files:
        new_clips = process_file(bvhfile, window=60, window_step=30)
        if new_clips is not None:
            clips += new_clips
    data_clips = np.array(clips)
    # np.savez_compressed(r'data\training_data\processed_cmu_data', clips=data_clips)
    np.savez_compressed(r'E:\tensorflow\data\training_data\small_window_size\processed_cmu_data', clips=data_clips)


def preprocess_edin_data():
    clips = []
    edin_files = get_bvh_files(r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\edin')

    for bvhfile in edin_files:
        new_clips = process_file(bvhfile, window=60, window_step=30)
        if new_clips is not None:
            clips += new_clips
    data_clips = np.array(clips)
    # np.savez_compressed(r'data\training_data\processed_edin_data', clips=data_clips)
    np.savez_compressed(r'E:\tensorflow\data\training_data\small_window_size\processed_edin_data', clips=data_clips)


def preprocess_edin_locomotion_data():
    clips = []
    edin_files = get_bvh_files(r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\edin\edin_locomotion')

    for bvhfile in edin_files:
        new_clips = process_file(bvhfile, window=60, window_step=30)
        if new_clips is not None:
            clips += new_clips
    data_clips = np.array(clips)
    # np.savez_compressed(r'data\training_data\processed_edin_locomotion_data', clips=data_clips)
    np.savez_compressed(r'E:\workspace\tensorflow_results\data\training_data\small_window_size\processed_edin_locomotion_data_new', clips=data_clips)


def preprocess_ulm_data():
    clips = []
    ulm_files = get_bvh_files(r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\ulm')

    for bvhfile in ulm_files:
        new_clips = process_file(bvhfile, window=60, window_step=30)
        if new_clips is not None:
            clips += new_clips
    data_clips = np.array(clips)
    # np.savez_compressed(r'data\training_data\processed_ulm_data', clips=data_clips)
    np.savez_compressed(r'E:\tensorflow\data\training_data\small_window_size\processed_ulm_data', clips=data_clips)


def preprocess_ulm_locomotion_data():
    clips = []
    root_dir = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\ulm_locomotion'
    locomotion_folders = ['Takle_turnwalk', 'Take_walk', 'Take_walk_s']
    bvhfiles = []
    for subfolder in locomotion_folders:
        bvhfiles += glob.glob(os.path.join(root_dir, subfolder, '*.bvh'))
        for bvhfile in bvhfiles:
            new_clips = process_file(bvhfile, window=60, window_step=30)
            if new_clips is not None:
                clips += new_clips
    data_clips = np.array(clips)
    # np.savez_compressed(r'data\training_data\processed_ulm_locomotion_data', clips=data_clips)
    np.savez_compressed(r'E:\workspace\tensorflow_results\data\training_data\small_window_size\processed_ulm_locomotion_data_new', clips=data_clips)


def preprocess_ACCAD_data():
    clips = []
    accad_files = get_bvh_files(r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\ACCAD')

    for bvhfile in accad_files:
        new_clips = process_file(bvhfile, window=60, window_step=30)
        if new_clips is not None:
            clips += new_clips
    data_clips = np.array(clips)
    # np.savez_compressed(r'data\training_data\processed_accad_data', clips=data_clips)
    np.savez_compressed(r'E:\tensorflow\data\training_data\small_window_size\processed_accad_data', clips=data_clips)


def preprocess_stylized_walking_data():
    clips = []
    accad_files = get_bvh_files(r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\stylized_data')

    for bvhfile in accad_files:
        new_clips = process_file(bvhfile, window=60, window_step=30)
        if new_clips is not None:
            clips += new_clips
    data_clips = np.array(clips)
    # np.savez_compressed(r'data\training_data\processed_stylized_data', clips=data_clips)
    np.savez_compressed(r'E:\workspace\tensorflow_results\data\training_data\small_window_size\processed_stylized_data_new', clips=data_clips)


def run_preprocessing(data_folder, save_folder, sliding_window=False, window=240, window_step=120, 
                      body_plane_indices=[], foot_left=[], foot_right=[], suffix='', animated_joints=MH_CMU_ANIMATED_JOINTS):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    bvhfiles = []

    get_files(data_folder, suffix='.bvh', files=bvhfiles)
    clips = []
    for bvhfile in bvhfiles:

        clip = process_file(bvhfile, sliding_window=sliding_window, window=window, window_step=window_step,
                            body_plane_indices=body_plane_indices, fid_l=foot_left, fid_r=foot_right, animated_joints=animated_joints)
        clips += clip.tolist()
    clips = np.array(clips)
    if suffix != '':
        save_filename = os.path.split(data_folder)[-1] + '_' + suffix
    else:
        save_filename = os.path.split(data_folder)[-1]

    np.savez_compressed(os.path.join(save_folder, save_filename), clips=clips)                    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder",  
                        type=str, 
                        default=r'data\training_data\mk_cmu_skeleton\h36m',
                        help="path to bvh files")
    parser.add_argument("--save_folder",
                        type=str,
                        default=r'data\training_data',
                        help="Directory where to save the results.")
    parser.add_argument("--slide_window",
                        type=bool,
                        default=False,
                        help="apply fixed-length window to slide over the input.")
    parser.add_argument('--window_length',
                        type=int,
                        default=120,
                        )
    parser.add_argument('--window_step',
                        type=int,
                        default=60)                                            
    parser.add_argument('--body_plane_indices',
                        type=list,
                        default=[7, 25, 1]) ## ['spine_03', 'thigh_r', 'thigh_l']
    parser.add_argument('--foot_left', 
                        type=list,
                        default=[3, 4])
    parser.add_argument('--foot_right', 
                        type=list,
                        default=[27, 28])
    parser.add_argument('--suffix',
                        type=str,
                        default='')
    parser.add_argument('--animated_joints',
                        type=list,
                        default=MH_CMU_ANIMATED_JOINTS)                

    args = parser.parse_args()
    run_preprocessing(args.data_folder, args.save_folder, args.slide_window, 
                      args.window_length, args.window_step, args.body_plane_indices,
                      args.foot_left, args.foot_right, args.suffix, args.animated_joints)


if __name__ == "__main__":
    main()