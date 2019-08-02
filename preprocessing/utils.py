import numpy as np 
import copy
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from pfnn.quaternions import Quaternions


def estimate_floor_height(foot_heights):
    """estimate offset from foot to floor
    
    Arguments:
        foot_heights {numpy.array} -- the heights of contact point in each frame
    """
    median_value = np.median(foot_heights)
    min_value = np.min(foot_heights)
    abs_diff = np.abs(median_value - min_value)
    return (2 * abs_diff) / (np.exp(abs_diff) + np.exp(-abs_diff)) + min_value


def sliding_window(data, window_size):
    """Slide Over Windows
    
    """
    windows = []
    window_step = int(window_size / 2.0)
    if len(data) % window_step == 0:
        n_clips = (len(data) - len(data) % window_step) // window_step
    else:
        n_clips = (len(data) - len(data) % window_step) // window_step + 1

    for j in range(0, n_clips):
        """ If slice too small pad out by repeating start and end poses """
        slice = data[j * window_step: j * window_step + window_size]
        if len(slice) < window_size:
            left = slice[:1].repeat((window_size - len(slice)) // 2 + (window_size - len(slice)) % 2, axis=0)
            right = slice[-1:].repeat((window_size - len(slice)) // 2, axis=0)
            slice = np.concatenate([left, slice, right], axis=0)
        if len(slice) != window_size: raise Exception()

        windows.append(slice)
    return windows


def combine_motion_clips(clips, motion_len, window_step):
    """combine motion clips to reconstruct original motion
    
    Arguments:
        clips {numpy.array} -- n_clips * n_frame * n_dims
        motion_len {int} -- number of original motion frames
        overlapping_len {int}
    
    """

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


def visualize_pfnn_X():
    pass


def visualize_pfnn_Y():
    pass


def covnert_pfnn_preprocessed_data_to_global_joint_positions(motion_data, use_speed=True):

    """reverse the process of preprocessing from pfnn 
    
    Arguments:
        motion_data {numpy.ndarray} -- n_frames * n_dims 
    
    Keyword Arguments:
        use_speed {bool} -- choose to use speed data or not
    """
    root_velocity = motion_data[:, :2]
    root_rvelocity = motion_data[:, 2:3]
    # input_joint_pos -> world space
    local_position = motion_data[:, 32:32+93] # out joint pos
    local_velocity = motion_data[:, 32+93:32+93*2]
    # out joint pos -> world space
    # out joint vel
    # ((input_joint_pos + joint vel) + out joint pos) / 2
    anim_data = np.concatenate([local_position, local_velocity, root_velocity, root_rvelocity], axis=-1)    
    if use_speed:
        joints, vel, root_x, root_z, root_r = anim_data[:, :93], anim_data[:,93:93*2], anim_data[:, -3], anim_data[:, -2], anim_data[:, -1]

        joints = joints.reshape((len(joints), -1, 3))  ### reshape 2D matrix to (n_frames, n_joints, n_dims)
        vel = vel.reshape((len(vel), -1, 3))
        rotation = Quaternions.id(1)
        translation = np.array([[0, 0, 0]])
        last_joints = np.array([0,0,0] * len(joints))
        # ref_dir = np.array([0, 0, 1])
        print("testing")
        for i in range(len(joints)):
            joints[i, :, :] = rotation * joints[i]
            joints[i, :, 0] = joints[i, :, 0] + translation[0, 0]
            joints[i, :, 2] = joints[i, :, 2] + translation[0, 2]
            if i == 0:
                last_joints = joints[i]
            else:
                vel[i, :, :] = rotation * vel[i]
                joints[i, :, :] = ((last_joints + vel[i, :, :]) + joints[i, :, :]) / 2
                last_joints = joints[i]
    else:
        joints, root_x, root_z, root_r = anim_data[:, :93], anim_data[:, -3], anim_data[:, -2], anim_data[:, -1]
        joints = joints.reshape((len(joints), -1, 3))  ### reshape 2D matrix to (n_frames, n_joints, n_dims)
        rotation = Quaternions.id(1)
        translation = np.array([[0, 0, 0]])
        for i in range(len(joints)):
            joints[i, :, :] = rotation * joints[i]
            joints[i, :, 0] = joints[i, :, 0] + translation[0, 0]
            joints[i, :, 2] = joints[i, :, 2] + translation[0, 2]
            rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0, 1, 0])) * rotation
            translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])    
    return joints