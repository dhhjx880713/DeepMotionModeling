import numpy as np
import glob
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from preprocessing.preprocessor import Preprocessor
from preprocessing.utils import estimate_floor_height, sliding_window
from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder
from mosi_utils_anim.animation_data.utils import convert_euler_frames_to_cartesian_frames, combine_motion_clips
from mosi_utils_anim.utilities import write_to_json_file, load_json_file
from mosi_utils_anim.animation_data.retargeting.directional_IK import DirectionalIK

"""simply extract joint poisiton from bvh files with fixed sized sliding window. All the clips are normalized in the way that starting position and orientation are the same
"""

animated_joint_list = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LowerBack', 'Spine', 'Spine1', 'LeftShoulder',
'LeftArm', 'LeftForeArm', 'LeftHand', 'LThumb', 'LeftFingerBase', 'LeftHandFinger1', 'Neck', 'Neck1', 'Head', 'RightShoulder',
'RightArm', 'RightForeArm', 'RightHand', 'RThumb', 'RightFingerBase', 'RightHandFinger1', 'RightUpLeg', 'RightLeg', 'RightFoot',
'RightToeBase']


def normalized_joint_positions():
    folder = r'E:\workspace\projects\cGAN\test_input'
    preprocessor = Preprocessor()
    
    preprocessor.load_bvh_files_from_directory(folder)
    
    preprocessor.rotate_euler_frames(np.array([0, 1]), ['RightUpLeg', 'Hips', 'LeftUpLeg'])
    preprocessor.translate_root_to_target(np.array([0, 0]))

    # preprocessor.shift_on_floor(foot_joints = ['LeftFoot', 'LeftToeBase', 'RightFoot', 'RightToeBase'])
    preprocessor.shift_on_floor(preprocessor.skeleton.animated_joints)
    preprocessor.save_files(r'E:\workspace\projects\cGAN\test_ouptut')
    animated_joint_list = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LowerBack', 'Spine', 'Spine1', 'LeftShoulder',
    'LeftArm', 'LeftForeArm', 'LeftHand', 'LThumb', 'LeftFingerBase', 'LeftHandFinger1', 'Neck', 'Neck1', 'Head', 'RightShoulder',
    'RightArm', 'RightForeArm', 'RightHand', 'RThumb', 'RightFingerBase', 'RightHandFinger1', 'RightUpLeg', 'RightLeg', 'RightFoot',
    'RightToeBase']
    # preprocessor.skeleton.animated_joints = animated_joint_list
    joint_des = preprocessor.skeleton.generate_bone_list_description(animated_joint_list)
    joint_positions = preprocessor.get_global_positions(animated_joint_list)
    # for key, value in joint_positions.items():
    #     print(key)
    #     print(value.shape)
    #### save as panim data
    # save_path = r'E:\workspace\projects\cGAN\test_ouptut\panim'
    # for key, value in joint_positions.items():
    #     save_data = {}
    #     save_data["has_skeleton"] = True
    #     save_data["skeleton"] = joint_des
    #     save_data['motion_data'] = value.tolist()
    #     write_to_json_file(os.path.join(save_path, key.replace('bvh', 'panim')), save_data)

    # #### create sliding window
    # joint_poss_clips = {}
    # for key, value in joint_positions.items():
    #     joint_poss_clips[key] = np.asarray(sliding_window(value, window_size=60))
    #     print(joint_poss_clips[key].shape)
    # tmp = [value for key, value in joint_poss_clips.items()]
    # combined_motion_data = np.concatenate(tmp, axis=0)
    # print(combined_motion_data.shape)
    # #### reassembly sliding window and save the panim file
    # save_path = r'E:\workspace\projects\cGAN\test_ouptut\panim\reconstruction'
    # reconstructed_motions = {}
    # for key, value in joint_poss_clips.items():
    #     motion_len = len(joint_positions[key])
    #     clip_data = np.reshape(joint_poss_clips[key], (value.shape[0], value.shape[1], value.shape[2] * value.shape[3]))

    #     combined_data = combine_motion_clips(clip_data, motion_len, 30)

    #     reconstructed_motions[key] = np.reshape(combined_data,
    #                                             (motion_len, value.shape[2], value.shape[3]))
    #     save_data = {}
    #     save_data["has_skeleton"] = True
    #     save_data["skeleton"] = joint_des
    #     save_data['motion_data'] =  reconstructed_motions[key].tolist()
    #     write_to_json_file(os.path.join(save_path, key.replace('bvh', 'panim')), save_data)        

    # #### convert painm to bvh to compare with original data 
    # ## stet 1: create constraints

    # # create_direction_constraints_panim_retargeting
    # target_skeleton_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_skeleton.bvh'
    # bvhreader = BVHReader(target_skeleton_file)
    # save_path = r'E:\workspace\projects\cGAN\test_ouptut\panim\reconstruction_bvh'
    # for key, value in reconstructed_motions.items():
    #     ik = DirectionalIK(bvhreader)
    #     ik(reconstructed_motions[key], joint_des, body_plane_joints=['RightUpLeg', 'Hips', 'LeftUpLeg'])
    #     ik.save_as_bvh(os.path.join(save_path, key))


def test_ik():
    panim_data = load_json_file(r'E:\workspace\projects\cGAN\test_ouptut\panim\reconstruction\Female1_B02_WalkToStandT2.panim')
    skeleton_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_skeleton.bvh'
    save_dir = r'E:\workspace\projects\cGAN\test_ouptut\panim\reconstruction\reconstructed_new.bvh'
    bvhreader = BVHReader(skeleton_file)
    ik = DirectionalIK(bvhreader)
    motion_data = np.asarray(panim_data['motion_data'])
    skeleton_des = panim_data['skeleton']
    ik(motion_data, skeleton_des, body_plane_joints=['RightUpLeg', 'Hips', 'LeftUpLeg'])
    save_dir = r'E:\workspace\projects\cGAN\test_ouptut\panim\reconstruction\reconstructed.bvh'
    ik.save_as_bvh(save_dir)


def get_normalized_joint_position():
    input_folder = r'E:\workspace\mocap_data\mk_cmu_retargeting\pfnn_data'
    preprocessor = Preprocessor()
    preprocessor.load_bvh_files_from_directory(input_folder)
    preprocessor.rotate_euler_frames(np.array([0, 1]), ['RightUpLeg', 'Hips', 'LeftUpLeg'])
    preprocessor.translate_root_to_target(np.array([0, 0]))
    preprocessor.shift_on_floor(foot_joints = animated_joint_list)    
    preprocessor.save_files(r'E:\workspace\projects\cGAN\processed_data\pfnn_data')
    ## remove zero length joints

    global_positions = preprocessor.get_global_positions(animated_joint_list)
    ## create clips

    ## concatenate motion clips
    # global_positions_sequence = [value for key, value in global_positions]
    motion_clips = []
    for key, value in global_positions.items():
        motion_clips += sliding_window(value, window_size=60)
    np.savez_compressed(r'E:\workspace\projects\cGAN\processed_data\joint_position\pfnn_data', clips=np.asarray(motion_clips))


def test():
    bvhfile = r'E:\workspace\projects\cGAN\test_ouptut\Female1_C12_RunTurnLeft45.bvh'
    bvhreader = BVHReader(bvhfile)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    print(skeleton.nodes[skeleton.root].rotation_order)

    

if __name__ == "__main__":
    # normalized_joint_positions()
    get_normalized_joint_position()
