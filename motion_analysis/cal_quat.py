# encoding: UTF-8
from morphablegraphs.animation_data import BVHReader
from morphablegraphs.motion_analysis.bvh_analyzer import BVHAnalyzer
from morphablegraphs.animation_data import LEN_QUAT
import os
import glob
from morphablegraphs.utilities import get_aligned_data_folder, get_data_analysis_folder, write_to_json_file, \
                                      load_json_file, areQuatClose
import numpy as np
import collections


def get_joint_quaternion(aligned_files,
                         joint_name):
    """
    Get target joint quaternion for a set of mocap files
    :param aligned_files: aligned bvh files
    :param joint_name: str
    :return: numpy.array3D(n_samples * n_frames * len_quat)
    """
    joint_quats = []  # the expected return shape is n_files * n_frames * len_quat
    # take the joint quaternion value of the first file as reference value
    ref_bvhreader = BVHReader(aligned_files[0])
    ref_motion = BVHAnalyzer(ref_bvhreader)
    ref_motion.to_quaternion()
    start_index, end_index = ref_motion.get_filtered_joint_param_range(joint_name)
    ref_quat = ref_motion.quat_frames[0, start_index: end_index]
    joint_quats.append(ref_motion.quat_frames[:, start_index: end_index].tolist())
    for i in range(1, len(aligned_files)):
        bvhreader = BVHReader(aligned_files[i])
        motion = BVHAnalyzer(bvhreader)
        motion.to_quaternion()
        tmp = []
        for j in range(len(motion.quat_frames)):
            if not areQuatClose(ref_quat, motion.quat_frames[j][start_index: end_index]):
                joint_quat = - motion.quat_frames[j][start_index: end_index]
            else:
                joint_quat = motion.quat_frames[j][start_index: end_index]
            tmp.append(joint_quat.tolist())
        joint_quats.append(tmp)
    return joint_quats


def get_joint_quaternion_from_frame_data(quat_frames,
                                         joint_name,
                                         bvh_analyzer):
    """
    :param quat_frames (Array<3d>): n_samples * n_frames * n_dims(79)
    :param joint_name (str): target joint name
    :return (Array<3d>): n_samples * n_frames * len_quat
    """
    start_index, end_index = bvh_analyzer.get_filtered_joint_param_range(joint_name)

    joint_quat = quat_frames[:, :, start_index : end_index]
    return joint_quat.tolist()

def cal_quaternion_for_all_joints_from_frame_data():
    test_file = r'C:\repo\data\1 - MoCap\7 - Mocap analysis\elementary_action_walk\beginRightStance\quat_frames.json'
    quat_frame_data = load_json_file(test_file)
    quat_frames = np.asarray(quat_frame_data.values())
    skeleton_file = r'../../skeleton.bvh'
    bvhreader = BVHReader(skeleton_file)
    motion = BVHAnalyzer(bvhreader)
    reduced_joints = [key for key in motion.node_name_frame_map.keys() if motion.node_name_frame_map[key] >= 0]
    joints_quaternion_data = collections.OrderedDict()
    for joint_name in reduced_joints:
        print(joint_name)
        joints_quaternion_data[joint_name] = get_joint_quaternion_from_frame_data(quat_frames,
                                                                                  joint_name,
                                                                                  motion)
    return joints_quaternion_data


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


def gen_quaternion_pos_for_aligned_motion():
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
            if not os.path.exists(os.path.join(data_analysis_folder, 'joint_quaternion.json')):
                joints_quat = cal_quaternion_for_all_joints(elementary_action,
                                                            motion_primitive)
                write_to_json_file(os.path.join(data_analysis_folder,
                                                'joint_quaternion.json'),
                                   joints_quat)


if __name__ == "__main__":
    elementary_action = 'pickBoth'
    motion_primitive = 'first'
    joint_quat_data = cal_quaternion_for_all_joints(elementary_action,
                                                    motion_primitive)

    target_folder = os.path.join(r'C:\repo\data\1 - MoCap\7 - Mocap analysis',
                                 'elementary_action_' + elementary_action,
                                 motion_primitive,
                                 'joint_quaternion.json')
    write_to_json_file(target_folder,
                       joint_quat_data)
    # gen_quaternion_pos_for_aligned_motion()