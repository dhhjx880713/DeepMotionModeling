# encoding: UTF-8
from morphablegraphs.utilities import get_aligned_data_folder, get_data_analysis_folder, write_to_json_file
import os
import glob
from morphablegraphs.animation_data import BVHReader
from morphablegraphs.motion_analysis.bvh_analyzer import BVHAnalyzer
import numpy as np
import json
import collections

def get_absolute_cartesian_joint_position(aligned_bvhfiles,
                                          joint_name):
    """

    :param aligned_bvhfiles: a list of bvh files
    :param joint_name (str): list of joints
    :return (list): n_samples * n_frames * len_cartesian_point
    """
    joint_pos = []
    for filename in aligned_bvhfiles:
        bvhreader = BVHReader(filename)
        motion = BVHAnalyzer(bvhreader)
        joint_pos.append(motion.get_global_pos_for_all_frames(joint_name).tolist())
    return joint_pos


def cal_absolute_cartesian_position(elementary_action,
                                    motion_primitive):
    """

    :param elementary_action: str
    :param motion_primitive: str
    :return (dic): each key is joint name, each value contains joint cartesian position data (n_samples * n_frames * len_cartesian_point)
    """
    # print(elementary_action + '_' + motion_primitive)
    aligned_data_folder = get_aligned_data_folder(elementary_action,
                                                  motion_primitive)
    aligned_bvhfiles = glob.glob(os.path.join(aligned_data_folder, '*.bvh'))
    if aligned_bvhfiles != []:
        joints_pos = collections.OrderedDict()
        # tmp_bvhreader = BVHReader(aligned_bvhfiles[0])
        # motion = BVHAnalyzer(tmp_bvhreader)
        # reduced_joints = [key for key in motion.node_name_frame_map.keys() if motion.node_name_frame_map[key] >= 0]
        joint_list = ['Hips', 'Spine', 'Spine_1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                      'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftUpLeg', 'LeftLeg', 'LeftFoot',
                      'RightUpLeg', 'RightLeg', 'RightFoot', 'Head_EndSite', 'Bip01_L_Finger0', 'Bip01_L_Finger1',
                      'Bip01_L_Finger2', 'Bip01_L_Finger3', 'Bip01_L_Finger4', 'Bip01_L_Toe0', 'Bip01_R_Finger0',
                      'Bip01_R_Finger1', 'Bip01_R_Finger2', 'Bip01_R_Finger3', 'Bip01_R_Finger4', 'Bip01_R_Toe0']
        for joint_name in joint_list:
            joints_pos[joint_name] = get_absolute_cartesian_joint_position(aligned_bvhfiles,
                                                                           joint_name)
        return joints_pos
    else:
        return None

def gen_absolute_cartesian_joint_pos_for_aligned_motion():
    aligned_data_folder = r'C:\repo\data\1 - MoCap\4 - Alignment'
    for cur_dir, subfolder, files in os.walk(aligned_data_folder):
        if subfolder == []:
            elementary_action_dir, motion_primitive = os.path.split(cur_dir)
            elementary_action_folder = os.path.split(elementary_action_dir)[-1]
            elementary_action = elementary_action_folder.split('_')[-1]
            data_analysis_folder = get_data_analysis_folder(elementary_action,
                                                            motion_primitive)
            print(elementary_action)
            print(motion_primitive)
            if not os.path.exists(os.path.join(data_analysis_folder, 'joint_absolute_cartesian_position.json')):
                joints_pos = cal_absolute_cartesian_position(elementary_action,
                                                             motion_primitive)
                write_to_json_file(os.path.join(data_analysis_folder,
                                                'joint_absolute_cartesian_position.json'),
                                   joints_pos)

def get_children_for_joints():
    skeleton_file = r'../../skeleton.bvh'
    bvhreader = BVHReader(skeleton_file)
    motion = BVHAnalyzer(bvhreader)
    reduced_joints = [key for key in motion.node_name_frame_map.keys() if motion.node_name_frame_map[key] >= 0]
    print(reduced_joints)
    reduced_joints_children = {}
    joint_list = []
    for joint in reduced_joints:
        node = motion.get_joint_by_joint_name(joint)
        print([child.node_name for child in node.children])
        reduced_joints_children[joint] = [child.node_name for child in node.children]
        joint_list += [child.node_name for child in node.children]
    return np.unique(joint_list)


if __name__ == "__main__":
    elementary_action = 'carryBoth'
    motion_primitive = 'endRightStance'
    # joint_pos = cal_absolute_cartesian_position(elementary_action,
    #                                             motion_primitive)
    # aligned_data_folder = get_aligned_data_folder(elementary_action,
    #                                               motion_primitive)
    # # aligned_bvhfiles = glob.glob(os.path.join(aligned_data_folder, '*.bvh'))
    # # for item in aligned_bvhfiles:
    # #     print(os.path.split(item)[-1])
    # #     bvhreader = BVHReader(item)
    # #     print(len(bvhreader.frames))
    # joints_pos = cal_absolute_cartesian_position(elementary_action,
    #                                              motion_primitive)

    # data_analysis_folder = get_data_analysis_folder(elementary_action,
    #                                                 motion_primitive)
    #
    # filename = os.path.join(data_analysis_folder, 'joint_absolute_cartesian_position.json')
    # write_to_json_file(filename, joints_pos)

    ##################################################
    ## calculate caresian position of target joints for the whole aligned data folder
    # gen_absolute_cartesian_joint_pos_for_aligned_motion()