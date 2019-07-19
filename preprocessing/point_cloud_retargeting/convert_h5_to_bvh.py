import h5py
import numpy as np
import sys
import os
import copy
import h5py
import glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/../..')
from mosi_utils_anim.utilities import load_json_file
from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder, BVHWriter
# from morphablegraphs.python_src.morphablegraphs.construction.retargeting.convert_panim_to_bvh import get_joint_parent_in_joint_mapping, \
#     create_direction_constraints_from_panim, create_direction_constraints_panim_retargeting, estimate_scale_factor_panim
from mosi_utils_anim.animation_data.retargeting.directional_constraints_retargeting import retarget_motion, estimate_scale_factor, retarget_single_motion, \
    create_direction_constraints, align_ref_frame, retarget_folder, create_direction_constraints
from transformations import rotation_matrix
from preprocessing.point_cloud_retargeting.convert_h5_to_panim import h6M_SKELETON, estimate_scale_factor_panim, create_direction_constraints_panim_retargeting
# from morphablegraphs.python_src.morphablegraphs.construction.retargeting.skeleton_definition import cmu_skeleton
#todo: figure out how to automatically avoid zero-length bone

JOINT_MAPPING_TO_CMU = {  ## joint mapping from h6m to cmu
    'Hips': 'Hips',
    'HipRight': 'RightUpLeg',
    'KneeRight': 'RightLeg',
    'FootRight': 'RightFoot',
    'ToeBaseRight': 'RightToeBase',
    'HipLeft': 'LeftUpLeg',
    'KneeLeft': 'LeftLeg',
    'FootLeft': 'LeftFoot',
    'ToeBaseLeft': 'LeftToeBase',
    # 'Spine1': '',
    'Spine2': 'Spine',
    'Neck': 'Neck',
    # 'Head': 'Head',
    'Site3': 'Head',
    # 'ShoulderLeft': '',
    'ElbowLeft': 'LeftArm',
    'WristLeft': 'LeftForeArm',
    'HandLeft': 'LeftHand',
    # 'ShouldRight': 'RightArm',
    'ElbowRight': 'RightArm',
    'WristRight': 'RightForeArm',
    'HandRight': 'RightHand'
}

JOINT_MAPPING_FROM_CMU_TO_H6M = {  ## joint mapping is defined in the way from target to source
    'Hips': 'Hips',
    'RightUpLeg': 'HipRight',
    'RightLeg': 'KneeRight',
    'RightFoot': 'FootRight',
    'RightToeBase': 'ToeBaseRight',
    'LeftUpLeg': 'HipLeft',
    'LeftLeg': 'KneeLeft',
    'LeftFoot': 'FootLeft',
    'LeftToeBase': 'ToeBaseLeft',
    'Spine': 'Spine2',
    'Neck': 'Neck',
    # 'Head': 'Head',
    'Head': 'Site3',
    'LeftArm': 'ElbowLeft',
    'LeftForeArm': 'WristLeft',
    'LeftHand': 'HandLeft',
    'RightArm': 'ElbowRight',
    'RightForeArm': 'WristRight',
    'RightHand': 'HandRight'
}



def convert_h5_to_bvh(input_file, save_file, target_bvh):
    with h5py.File(input_file, 'r') as h5f:
        poses = h5f['3D_positions'][:].T
        num_frames = poses.shape[0]
        poses = poses.reshape(num_frames,-1,3)
    target_skeleton = SkeletonBuilder().load_from_bvh(target_bvh)
    target_skeleton_description = target_skeleton.generate_bone_list_description()
    ## to do: remove hard coded correction
    rotation_angle = -90
    rotation_axis = np.array([1, 0, 0])
    rotmat = rotation_matrix(np.deg2rad(rotation_angle), rotation_axis)
    print(rotmat.shape)
    print(poses.shape)
    ones = np.ones((poses.shape[0], poses.shape[1], 1))
    extended_poses = np.concatenate((poses, ones), axis=-1)
    swapped_poses = np.transpose(extended_poses, (0, 2, 1))

    rotated_poses = np.matmul(rotmat, swapped_poses)
    rotated_poses = np.transpose(rotated_poses, (0, 2, 1))
    rotated_poses = rotated_poses[:, :, :3]
    src_torso_joints = ['HipRight', 'Spine2', 'HipLeft']
    target_torso_joints = ['RightUpLeg', 'Hips', 'LeftUpLeg']
    root_joint = 'Hips'
    root_index = 0

    skeleton_scaling_factor = estimate_scale_factor_panim(h6M_SKELETON, target_skeleton_description, 
                                                          src_torso_joints, target_torso_joints, rotated_poses[0:1],
                                                          target_bvh.frames[0], target_skeleton, 
                                                          JOINT_MAPPING_FROM_CMU_TO_H6M)
    print('skeleton factor is: ', skeleton_scaling_factor)                                                    
    directional_constraints = create_direction_constraints_panim_retargeting(h6M_SKELETON,
                                                                             target_skeleton_description,
                                                                             src_torso_joints,
                                                                             rotated_poses,
                                                                             JOINT_MAPPING_FROM_CMU_TO_H6M)
    n_frames = rotated_poses.shape[0]
    out_frames = []
    for i in range(n_frames):
        # print(i)
        pose_dir = directional_constraints[i]['pose_dir']
        if i == 0:
            new_frame = target_bvh.frames[0]
            ref_frame = align_ref_frame(new_frame, pose_dir, target_skeleton, target_torso_joints)
            ref_frame[0] = rotated_poses[0, root_index, 0] * skeleton_scaling_factor
            ref_frame[2] = rotated_poses[0, root_index, 2] * skeleton_scaling_factor
        else:
            # take the previous frame as initial guess
            new_frame = copy.deepcopy(out_frames[i-1])
            ref_frame = align_ref_frame(new_frame, pose_dir, target_skeleton, target_torso_joints)
            ref_frame[:3] = (rotated_poses[i, root_index] - rotated_poses[i-1, root_index]) * skeleton_scaling_factor + out_frames[i-1][:3]
        retarget_motion(root_joint, directional_constraints[i], target_skeleton, ref_frame)
        out_frames.append(ref_frame)
    BVHWriter(save_file, target_skeleton, out_frames, target_bvh.frame_time, is_quaternion=False)


def convert_h5_to_bvh_batch(input_folder, save_folder, target_bvhreader):
    h5files = glob.glob(os.path.join(input_folder, '*.h5'))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for bvhfile in h5files:
        print(bvhfile)
        filename = os.path.split(bvhfile)[-1]
        save_filename = os.path.join(save_folder, filename.replace('h5', 'bvh'))
        convert_h5_to_bvh(bvhfile, save_filename, target_bvhreader)


def batch_demo():
    data_set = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
    input_folder = r'E:\gits\HP_GAN\3d-pose-baseline\h36m\S1\MyPoses\3D_positions'
    output_folder = r'E:\workspace\projects\cGAN\h36m_cmu'
    target_bvhreader_file = r'E:\workspace\projects\cGAN\cmu_skeleton.bvh'
    target_bvh = BVHReader(target_bvhreader_file)
    input_path_segs = input_folder.split("\\")
    for id in data_set: 

        input_path_segs[-3] = id
        input_folder = "\\".join(input_path_segs)
        sub_output_folder = os.path.join(output_folder, id)
        if not os.path.exists(sub_output_folder):
            os.makedirs(sub_output_folder)
        convert_h5_to_bvh_batch(input_folder, sub_output_folder, target_bvh)


def demo():
    input_file = r'E:\gits\HP_GAN\3d-pose-baseline\h36m\S1\MyPoses\3D_positions\Directions 1.h5'
    filename = os.path.split(input_file)[-1]
    save_path = r'E:\workspace\projects\cGAN'
    save_file = os.path.join(save_path, filename.replace('h5', 'bvh'))
    target_bvhfile = r'E:\workspace\projects\cGAN\cmu_skeleton.bvh'
    target_bvh = BVHReader(target_bvhfile)

    convert_h5_to_bvh(input_file, save_file, target_bvh)

def test():
    CMU_Skeleton_file = r'E:\workspace\unity_workspace\MG\motion_in_json\cmu_skeleton.bvh'
    cmu_bvh = BVHReader(CMU_Skeleton_file)
    cmu_skeleton = SkeletonBuilder().load_from_bvh(cmu_bvh)
    skeleton_data = cmu_skeleton.generate_bone_list_description()
    print(skeleton_data)
    for joint in cmu_skeleton.animated_joints:
        print("joint name: ", joint)
        print(np.linalg.norm(cmu_skeleton.nodes[joint].offset))
        if np.linalg.norm(cmu_skeleton.nodes[joint].offset) < 1e-4:
            print("zero length bone.")

    h6m_panim = load_json_file(r'E:\gits\vae_motion_modeling\data\HPGAN_results\panim_output\1.panim')
    motion_data = np.asarray(h6m_panim['motion_data'])
    skeleton_des = h6m_panim['skeleton']
    # wristRightIndex = next(item["index"] for item in skeleton_des if item['name'] == "WristRight")
    # handRightIndex = next(item["index"] for item in skeleton_des if item['name'] == "HandRight")
    # print(motion_data[10, wristRightIndex])
    # print(motion_data[10, handRightIndex])
    for item in skeleton_des:
        print(item['name'])
        print(motion_data[0, item['index']])


def test1():
    # print(list(JOINT_MAPPING_TO_CMU.values()))
    mat1 = np.random.rand(10, 13, 3)
    rotmat = np.random.rand(4,4)
    ones = np.ones((mat1.shape[0], mat1.shape[1], 1))
    print(ones.shape)
    c_mat = np.concatenate((mat1, ones), axis=-1)
    print(c_mat.shape)
    swapped_mat = np.transpose(c_mat, (0, 2, 1))
    rotated_mat = np.matmul(rotmat, swapped_mat)
    rotated_mat = np.transpose(rotated_mat, (0, 2, 1))


def compute_scale_factor():
    h6m_panim = load_json_file(r'E:\workspace\projects\retargeting_experiments\panim_from_mk_retargeting\eating_long.panim')
    src_points = np.asarray(h6m_panim['motion_data'])
    src_point_des = h6m_panim['skeleton']
    bvh_skeleton_file = r'E:\workspace\projects\cGAN\cmu_skeleton.bvh'
    target_bvhreader = BVHReader(bvh_skeleton_file)
    target_skeleton = SkeletonBuilder().load_from_bvh(target_bvhreader)
    target_point_des = target_skeleton.generate_bone_list_description()
    src_body_plane = ['HipRight', 'Spine2', 'HipLeft']
    target_body_plane = ['RightUpLeg', 'Hips', 'LeftUpLeg']

    scaling = estimate_scale_factor_panim(src_point_des, target_point_des, src_body_plane,
                                target_body_plane, src_points[0:1], target_bvhreader.frames[0],
                                target_skeleton, JOINT_MAPPING_FROM_CMU_TO_H6M)
    print("scaling factor: ")                            
    print(scaling)


def create_direction_constraints_from_panim_test():
    ### test case1: sparse joint mapping 
    TEST_JOINTMAPPING = { 
    'Hips': 'Hips',  ## this must be included
    'ToeBaseRight': 'RightToeBase',
    'ToeBaseLeft': 'LeftToeBase',
    # 'Spine1': '',
    'Head': 'Head',
    # 'ShoulderLeft': '',
    'HandLeft': 'LeftHand',
    # 'ShouldRight': 'RightArm',
    'HandRight': 'RightHand'
    }
    bvh_skeleton_file = r'E:\workspace\projects\cGAN\cmu_skeleton.bvh'
    target_bvhreader = BVHReader(bvh_skeleton_file)
    target_skeleton = SkeletonBuilder().load_from_bvh(target_bvhreader)
    # h6m_panim = load_json_file(r'E:\gits\vae_motion_modeling\data\HPGAN_results\panim_output\1.panim')
    h6m_panim = load_json_file(r'E:\workspace\projects\retargeting_experiments\panim_from_mk_retargeting\eating_long.panim')
    motion_data = np.asarray(h6m_panim['motion_data'])
    src_skeleton_des = h6m_panim['skeleton']
    body_plane_list = ['HipRight', 'Spine2', 'HipLeft']
    target_body_plane_joints = ['RightUpLeg', 'Hips', 'LeftUpLeg']
    root_joint = 'Hips'
    root_index = next(item['index'] for item in src_skeleton_des if item['name'] == 'Hips')
    cmu_panim = load_json_file(r'E:\workspace\projects\cGAN\CMU_example_motion.panim')
    target_skeleton_des = cmu_panim['skeleton']

    skeleton_scale_factor = 0.09676443109526761
    # d_c = create_direction_constraints_from_panim(skeleton_des, motion_data, body_plane_list,
    #                                               JOINT_MAPPING_TO_CMU)
    d_c = create_direction_constraints_panim_retargeting(src_skeleton_des, target_skeleton_des,
                                                         body_plane_list, motion_data, JOINT_MAPPING_FROM_CMU_TO_H6M)
    # print(d_c[0])
    print("length of conditions: ", len(d_c))  
    n_frames = motion_data.shape[0]
    out_frames = []
    for i in range(n_frames):
        print(i)
        pose_dir = d_c[i]['pose_dir']
        if i == 0:
            new_frame = target_bvhreader.frames[0]
            ref_frame = align_ref_frame(new_frame, pose_dir, target_skeleton, target_body_plane_joints)
            ref_frame[0] = motion_data[0, root_index, 0] * skeleton_scale_factor
            ref_frame[2] = motion_data[0, root_index, 2] * skeleton_scale_factor
        else:
            # take the previous frame as initial guess
            new_frame = copy.deepcopy(out_frames[i-1])
            ref_frame = align_ref_frame(new_frame, pose_dir, target_skeleton, target_body_plane_joints)
            ref_frame[:3] = (motion_data[i, root_index] - motion_data[i-1, root_index]) * skeleton_scale_factor + out_frames[i-1][:3]
        retarget_motion(root_joint, d_c[i], target_skeleton, ref_frame)
        out_frames.append(ref_frame)
    save_filename = r'E:\workspace\projects\cGAN\eating_cmu.bvh'
    BVHWriter(save_filename, target_skeleton, out_frames, target_bvhreader.frame_time, is_quaternion=False)

EDIN_MAKEHUMAN_JOINT_SPARSE_MAPPING = {
    'Hips': 'Hips',
    # 'LHipJoint': 'LHipJoint',
    'LeftUpLeg': 'LeftUpLeg',
    'LeftLeg': 'LeftLeg',
    'LeftFoot': 'LeftFoot',
    'LeftToeBase': 'LeftToeBase',
    # 'RHipJoint': 'RHipJoint',
    'RightUpLeg': 'RightUpLeg',
    'RightLeg': 'RightLeg',
    'RightFoot': 'RightFoot',
    'RightToeBase': 'RightToeBase',
    # 'Neck': 'Neck',
    # 'Neck1': 'Neck',
    'Head': 'Head',

    # 'LowerBack': 'LowerBack',
    # 'Spine': 'Spine',
    # 'Spine': 'LowerBack',  ## take care if src joint has zero bone length
    # 'Spine1': 'Spine1',
    'LeftShoulder': 'LeftShoulder',
    # 'LeftArm': 'LeftShoulder',
    # 'LeftArm': 'LeftArm',
    # 'LeftForeArm': 'LeftForeArm',
    'LeftHand': 'LeftHand',

    'RightShoulder': 'RightShoulder',
    # 'RightArm': 'RightShoulder',
    # 'RightArm': 'RightArm',
    # 'RightForeArm': 'RightForeArm',
    'RightHand': 'RightHand',

    # 'LeftArm': 'clavicle_l',
    # 'LeftForeArm': 'upperarm_l',
    # 'LeftHand': 'lowerarm_l',
    # 'LeftHandIndex1': 'hand_l',
    # 'RightArm': 'clavicle_r',
    # 'RightForeArm': 'upperarm_r',
    # 'RightHand': 'lowerarm_r',
    # 'RightHandIndex1': 'hand_r'
}

def create_direction_constraints_from_bvh_test():
    content_motion_file = r'E:\workspace\projects\retargeting_experiments\test_data\LocomotionFlat01_000_short.bvh'
    dp_framework_motion_file = r'E:\gits\motionsynth_code\data\processed\edin\edin_locomotion\locomotion_jog_000_000.bvh'
    skeleton_file = r'E:\workspace\retargeting\makehuman_characters\fbx\cmu_skeleton\cmu_skeleton.bvh'
    # save_path = r'E:\workspace\projects\variational_style_simulation\retargeted_from_optimization'
    save_path = r'E:\workspace\projects\retargeting_experiments\retargeted_results'
    root_joint = "Hips"
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    # src_body_plane = ['thigh_r', 'Root', 'thigh_l']
    # retarget_single_motion(dp_framework_motion_file, skeleton_file, content_motion_file, save_path, root_joint,
    #                        target_body_plane, target_body_plane, EDIN_MAKEHUMAN_JOINT_MAPPING)
    bvhreader = BVHReader(content_motion_file)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    targets = create_direction_constraints(EDIN_MAKEHUMAN_JOINT_SPARSE_MAPPING, skeleton, bvhreader.frames[0],
                                           target_body_plane)
    print(targets)


if __name__ == "__main__":
    # test1()
    # create_direction_constraints_from_panim_test()
    # create_direction_constraints_from_bvh_test()
    # compute_scale_factor()
    # demo()
    batch_demo()