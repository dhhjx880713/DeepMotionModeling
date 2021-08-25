import os
import collections
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/../..')
from mosi_utils_anim.retargeting.directional_constraints_retargeting import retarget_motion, \
    estimate_scale_factor, retarget_single_motion, create_direction_constraints, align_ref_frame, retarget_folder


### from source to target
CAPTURY_MAKEHUMAN_JOINT_MAPPING = {
    'Hips': 'Hips',
    'LeftUpLeg': 'LeftUpLeg',
    'LeftLeg': 'LeftLeg',
    'LeftFoot': 'LeftFoot',
    'LeftToeBase': 'LeftToeBase',
    'RightUpLeg': 'RightUpLeg',
    'RightLeg': 'RightLeg',
    'RightFoot': 'RightFoot',
    'RightToeBase': 'RightToeBase',
    # 'Neck': 'Neck',
    'Neck': 'Neck',
    'Head': 'Head',

    'Spine2': 'LowerBack',
    # 'Spine': 'Spine',
    # 'Spine': 'LowerBack',  ## take care if src joint has zero bone length
    'Spine3': 'Spine1',
    'LeftShoulder': 'LeftShoulder',
    'LeftArm': 'LeftArm',
    'LeftForeArm': 'LeftForeArm',
    'LeftHand': 'LeftHand',

    'RightShoulder': 'RightShoulder',

    'RightArm': 'RightArm',
    'RightForeArm': 'RightForeArm',
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



def retarget_captury_data():
    input_dir = r'D:\workspace\my_git_repos\capturesysevaluation\data\preprocessed\CapturyStudio'
    save_dir = os.path.join(input_dir, 'retargeting')
    skeleton_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_T_pose.bvh'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    src_body_plane = ['Spine1', 'LeftUpLeg', 'RightUpLeg']
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    root_joint = "Hips"
    retarget_folder(input_dir, skeleton_file, save_dir, CAPTURY_MAKEHUMAN_JOINT_MAPPING, root_joint=root_joint,
                    src_body_plane=src_body_plane, target_body_plane=target_body_plane)


if __name__ == "__main__":
    retarget_captury_data()