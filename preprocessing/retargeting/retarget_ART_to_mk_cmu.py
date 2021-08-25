import os
import collections
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/../..')
from mosi_utils_anim.retargeting.directional_constraints_retargeting import retarget_motion, \
    estimate_scale_factor, retarget_single_motion, create_direction_constraints, align_ref_frame, retarget_folder


### from source to target
ART_MAKEHUMAN_JOINT_MAPPING = {
    'pelvis': 'Hips',
    'leftUpperLeg': 'LeftUpLeg',
    'leftLowerLeg': 'LeftLeg',
    'leftFoot': 'LeftFoot',
    'leftFoot_EndSite': 'LeftToeBase',
    'rightUpperLeg': 'RightUpLeg',
    'rightLowerLeg': 'RightLeg',
    'rightFoot': 'RightFoot',
    'rightFoot_EndSite': 'RightToeBase',
    # 'Neck': 'Neck',
    'neck': 'Neck',
    'head': 'Head',

    'upperLumbarSpine': 'LowerBack',
    # 'Spine': 'Spine',
    # 'Spine': 'LowerBack',  ## take care if src joint has zero bone length
    'upperThoracicSpine': 'Spine1',
    'leftShoulder': 'LeftShoulder',
    'leftUpperArm': 'LeftArm',
    'leftLowerArm': 'LeftForeArm',
    'leftHand': 'LeftHand',

    'rightShoulder': 'RightShoulder',
    'rightUpperArm': 'RightArm',
    'rightLowerArm': 'RightForeArm',
    'rightHand': 'RightHand',

    # 'LeftArm': 'clavicle_l',
    # 'LeftForeArm': 'upperarm_l',
    # 'LeftHand': 'lowerarm_l',
    # 'LeftHandIndex1': 'hand_l',
    # 'RightArm': 'clavicle_r',
    # 'RightForeArm': 'upperarm_r',
    # 'RightHand': 'lowerarm_r',
    # 'RightHandIndex1': 'hand_r'
}



def retarget_single_file():
    input_file = r'D:\workspace\my_git_repos\capturesysevaluation\data\ART\4kmh.bvh'
    ref_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_T_pose.bvh'
    save_dir = os.path.join(os.path.split(input_file)[0], 'retarget')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    root_joint = "Hips"
    src_body_plane = ['lowerLumbarSpine', 'leftUpperLeg', 'rightUpperLeg']
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    retarget_single_motion(input_file, ref_file, rest_pose=input_file, save_dir=save_dir,
                           root_joint=root_joint, src_body_plane=src_body_plane,
                           target_body_plane=target_body_plane, 
                           joint_mapping=ART_MAKEHUMAN_JOINT_MAPPING,
                           n_frames=1500)


def retarget_captury_data():
    input_dir = r'D:\workspace\my_git_repos\capturesysevaluation\data\ART'
    save_dir = os.path.join(input_dir, 'retargeting')
    skeleton_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_T_pose.bvh'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    src_body_plane = ['lowerLumbarSpine', 'leftUpperLeg', 'rightUpperLeg']
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    root_joint = "Hips"
    retarget_folder(input_dir, skeleton_file, save_dir, ART_MAKEHUMAN_JOINT_MAPPING, root_joint=root_joint,
                    src_body_plane=src_body_plane, target_body_plane=target_body_plane)


if __name__ == "__main__":
    retarget_captury_data()
    # retarget_single_file()