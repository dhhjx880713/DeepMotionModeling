# encoding: UTF-8
from morphablegraphs.animation_data import BVHReader
from morphablegraphs.motion_analysis import BVHAnalyzer
import os
import glob
from morphablegraphs.utilities import get_aligned_data_folder
import numpy as np
import collections


def get_joint_quaternion(aligned_files,
                         joint_name):
    joint_quat = []  # the expected return shape is n_files * n_frames * len_quat
    for bvhfile in aligned_files:
        bvhreader = BVHReader(bvhfile)
        motion = BVHAnalyzer(bvhreader)
        motion.to_quaternion()
        start_index, end_index = motion.get_filtered_joint_param_range(joint_name)
        joint_quat.append(motion.quat_frames[start_index: end_index])
    return np.asarray(joint_quat)


def cal_quaternion_for_all_joints(elementary_action,
                                  motion_primitive):
    aligned_data_folder = get_aligned_data_folder(elementary_action, motion_primitive)
    aligned_bvhfiles = glob.glob(os.path.join(aligned_data_folder, '*.bvh'))
    if aligned_bvhfiles != []:
        joints_quaternion_data = collections.OrderedDict()
        tmp_bvhreader = BVHReader(aligned_bvhfiles[0])
        motion = BVHAnalyzer(tmp_bvhreader)
        reduced_joints = [key for key in motion.node_name_frame_map.keys() if motion.node_name_frame_map[key] >= 0]
        for joint_name in reduced_joints:
            print(joint_name)
            joints_quaternion_data[joint_name] = get_joint_quaternion(aligned_bvhfiles, joint_name)
        return joints_quaternion_data
    else:
        return None


def get_quat_frames_motion_primitive(elementary_action,
                                     motion_primitive):
    quat_frames = []
    aligned_data_folder = get_aligned_data_folder(elementary_action, motion_primitive)
    aligned_bvhfiles = glob.glob(os.path.join(aligned_data_folder, '*.bvh'))
    assert aligned_bvhfiles != [], ("No motion file in the folder!")
    for bvhfile in aligned_bvhfiles:
        bvhreader = BVHReader(bvhfile)
        motion = BVHAnalyzer(bvhreader)
        motion.to_quaternion()
        quat_frames.append(motion.quat_frames)
    return np.asarray(quat_frames)