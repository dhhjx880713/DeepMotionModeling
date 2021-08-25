import os
import collections
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/../..')
from mosi_utils_anim.retargeting.directional_constraints_retargeting import retarget_motion, \
    estimate_scale_factor, retarget_single_motion, create_direction_constraints, align_ref_frame, retarget_folder


### from source to target
VICON_MAKEHUMAN_JOINT_MAPPING = {
    'Root': 'Hips',
    'L_Femur': 'LeftUpLeg',
    'L_Tibia': 'LeftLeg',
    'L_Foot': 'LeftFoot',
    'L_Toe': 'LeftToeBase',
    'R_Femur': 'RightUpLeg',
    'R_Tibia': 'RightLeg',
    'R_Foot': 'RightFoot',
    'R_Toe': 'RightToeBase',
    # 'Neck': 'Neck',
    'Neck': 'Neck',
    'Head': 'Head',

    'LowerBack': 'LowerBack',
    # 'Spine': 'Spine',
    # 'Spine': 'LowerBack',  ## take care if src joint has zero bone length
    'Thorax': 'Spine1',
    'L_Collar': 'LeftShoulder',
    'L_Humerus': 'LeftArm',
    'L_Elbow': 'LeftForeArm',
    'L_Wrist': 'LeftHand',

    'R_Collar': 'RightShoulder',

    'R_Humerus': 'RightArm',
    'R_Elbow': 'RightForeArm',
    'R_Wrist': 'RightHand',

    # 'LeftArm': 'clavicle_l',
    # 'LeftForeArm': 'upperarm_l',
    # 'LeftHand': 'lowerarm_l',
    # 'LeftHandIndex1': 'hand_l',
    # 'RightArm': 'clavicle_r',
    # 'RightForeArm': 'upperarm_r',
    # 'RightHand': 'lowerarm_r',
    # 'RightHandIndex1': 'hand_r'
}



def retarget_vicon_data():
    input_dir = r'D:\workspace\my_git_repos\capturesysevaluation\data\preprocessed\Vicon'
    save_dir = os.path.join(input_dir, 'retargeting')
    skeleton_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_T_pose.bvh'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    src_body_plane = ['LowerBack', 'L_Femur', 'R_Femur']
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    root_joint = "Hips"
    retarget_folder(input_dir, skeleton_file, save_dir, VICON_MAKEHUMAN_JOINT_MAPPING, root_joint=root_joint,
                    src_body_plane=src_body_plane, target_body_plane=target_body_plane)


if __name__ == "__main__":
    retarget_vicon_data()