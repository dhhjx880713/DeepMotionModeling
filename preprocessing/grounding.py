"""test move motion data on floor
"""
import numpy as np
import os
import sys
import glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder
from mosi_utils_anim.animation_data.utils import convert_euler_frames_to_cartesian_frames, shift_euler_frames_to_ground, BVHWriter


def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    print('max: ', maxi)
    print('min: ', mini)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)


def move_motion_data_on_floor():
    # test_file = r'E:\workspace\mocap_data\raw_data\ACCAD\Female1_bvh\Female1_D6_CartWheel.bvh'
    # test_file = r'E:\workspace\mocap_data\mk_cmu_retargeting\pfnn_data\LocomotionFlat04_000.bvh'
    test_file = r'E:\gits\PFNN\data\animations\LocomotionFlat04_000.bvh'
    bvhreader = BVHReader(test_file)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    foot_l = ['LeftFoot', 'LeftToeBase']
    foot_r = ['RightFoot', 'RightToeBase']
    foot_l_indices = [skeleton.animated_joints.index(joint) for joint in foot_l]
    foot_r_indices = [skeleton.animated_joints.index(joint) for joint in foot_r]
    print(skeleton.animated_joints)
    print(len(skeleton.animated_joints))
    positions = convert_euler_frames_to_cartesian_frames(skeleton, bvhreader.frames)
    print(positions.shape)
    foot_heights = np.minimum(positions[:,foot_l_indices,1], positions[:,foot_r_indices,1]).min(axis=1)
    min_floor_height = min(foot_heights)
    print(foot_heights[0])
    # print("minimum floor height: ", min_floor_height)
    # print("mean: ", np.mean(foot_heights))
    # print("medium: ", np.median(foot_heights))
    # floor_height = softmin(foot_heights, softness=0.5, axis=0)
    # print(floor_height)
    # floor_height1 = offset_estimation(foot_heights)
    # print(floor_height1)


def test_command_line_arg():
    print("input file: ", sys.argv)


def estimate_floor_height(foot_heights):
    """estimate offset from foot to floor
    
    Arguments:
        foot_heights {[type]} -- [description]
    """
    median_value = np.median(foot_heights)
    min_value = np.min(foot_heights)
    abs_diff = np.abs(median_value - min_value)
    return (2 * abs_diff) / (np.exp(abs_diff) + np.exp(-abs_diff)) + min_value


def compute_floor_height():
    """take the median value of foot joint as reference, in order to avoid noise
       if 
    """


def shift_bvhfile_on_the_floor(bvhfile, foot_contact_joints, save_path):
    bvhreader = BVHReader(bvhfile)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    filename = os.path.split(bvhfile)[-1]
    # global_positions = convert_euler_frames_to_cartesian_frames(skeleton, bvhreader.frames)
    # foot_heights = np.minimum(global_positions[:, fid_l, 1], global_positions[:, fid_r, 1]).min(axis=1)
    # floor_height = softmin(foot_heights, softness=0.5, axis=0)
    new_frames = shift_euler_frames_to_ground(bvhreader.frames, foot_contact_joints, skeleton)
    BVHWriter(os.path.join(save_path, filename), skeleton, new_frames, skeleton.frame_time)


def shift_motion_data_on_the_floor():
    data_folder = r'D:\workspace\experiment data\style_transfer_using_gram_matrix\retarget\old'
    save_path = r'D:\workspace\experiment data\style_transfer_using_gram_matrix\retarget\old\ground'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    bvhfiles = glob.glob(os.path.join(data_folder, '*.bvh'))
    foot_contact_joints = ['LeftToeBase', 'RightToeBase']
    for bvhfile in bvhfiles:
        print(bvhfile)
        shift_bvhfile_on_the_floor(bvhfile, foot_contact_joints, save_path)


if __name__ == "__main__":
    # move_motion_data_on_floor()
    # test_command_line_arg()
    shift_motion_data_on_the_floor()