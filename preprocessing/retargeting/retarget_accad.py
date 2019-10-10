# encoding: UTF-8
import os
import collections
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/../..')
from mosi_utils_anim.animation_data.retargeting.directional_constraints_retargeting import retarget_motion, \
    estimate_scale_factor, retarget_single_motion, create_direction_constraints, align_ref_frame, retarget_folder

ACCAD_JOINT_MAP = {
    'Hips': 'pelvis',
    'ToSpine': 'spine_01',
    'Spine': 'spine_02',
    'Spine1': 'spine_03',
    'LeftShoulder': 'clavicle_l',
    'LeftArm': 'upperarm_l',
    'LeftForeArm': 'lowerarm_l',
    'LeftHand': 'hand_l',
    'RightShoulder': 'clavicle_r',
    'RightArm': 'upperarm_r',
    'RightForeArm': 'lowerarm_r',
    'RightHand': 'hand_r',
    'Neck': 'neck_01',
    'Head': 'head',
    'LeftUpLeg': 'thigh_l',
    'LeftLeg': 'calf_l',
    'LeftFoot': 'foot_l',
    'LeftToeBase': 'ball_l',
    'RightUpLeg': 'thigh_r',
    'RightLeg': 'calf_r',
    'RightFoot': 'foot_r',
    'RightToeBase': 'ball_r'
}

JOINTS_DOFS = {
    "calf_l": ['X', 'Z'],
    "calf_r": ['X', 'Z'],
    "thigh_l": ['X', 'Z'],
    "thigh_r": ['X', 'Z'],
    "foot_l": ['X', 'Z'],
    "foot_r": ['X', 'Z'],
    "ball_l": ['X', 'Z'],
    "ball_r": ['X', 'Z'],
    "clavicle_l": ['X', 'Z'],
    "clavicle_r": ['X', 'Z'],
    "upperarm_l": ['X', 'Z'],
    "upperarm_r": ['X', 'Z'],
    "lowerarm_l": ['X', 'Z'],
    "lowerarm_r": ['X', 'Z'],
    "hand_l": ['X', 'Z'],
    "hand_r": ['X', 'Z'],
    "spine_03": ['X', 'Z']
}



def retarget_ACCAD_data():
    ACCAD_data_folder = r'E:\mocap_data\ACCAD'
    src_rest_pose = r'E:\mocap_data\ACCAD\Female1_bvh_XYZrotation\Female1_A01_Stand.bvh'
    save_dir = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\ACCAD'
    # output_folder = r'C:\Users\hadu01\Downloads\test_data\output'
    target_skeleton_file = r'C:\Users\hadu01\git-repos\ulm\morphablegraphs\python_src\game_engine_target.bvh'
    root_joint = 'pelvis'
    src_body_plane = ['Hips', 'RightUpLeg', 'LeftUpLeg']
    actions = ['Male2_bvh', 'Female1_bvh', 'Male1_bvh']
    for action in actions:
        print(action)
        src_folder = os.path.join(ACCAD_data_folder, action)
        save_folder = os.path.join(save_dir, action)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        retarget_folder(src_folder, target_skeleton_file, save_folder, ACCAD_JOINT_MAP, JOINTS_DOFS,
                        root_joint, src_body_plane, target_body_plane=None)


def run_retarget_single_motion():
    '''
    retarget single motion to game engine skeleton
    :return:
    '''
    motion_content_file = r'E:\mocap_data\ACCAD\Female1_bvh\Female1_C18_RunChangeDirection.bvh'
    target_skeleton_file = r'C:\Users\hadu01\git-repos\ulm\morphablegraphs\python_src\game_engine_target.bvh'
    src_rest_pose = r'E:\mocap_data\ACCAD\Female1_bvh_XYZrotation\Female1_A01_Stand.bvh'
    save_dir = r'E:\experiment data\tmp'
    root_joint = 'pelvis'
    src_body_plane = ['Hips', 'RightUpLeg', 'LeftUpLeg']
    target_body_plane = ['thigh_r', 'Root', 'thigh_l']
    retarget_single_motion(motion_content_file, target_skeleton_file, src_rest_pose, save_dir, root_joint,
                           src_body_plane, None, ACCAD_JOINT_MAP, JOINTS_DOFS)


# def test_pose_orientation():
#     test_file = r'E:\mocap_data\ACCAD\Female1_bvh_XYZrotation\Female1_B03_Walk1.bvh'
#     bvhreader = BVHReader(test_file)
#     skeleton = Skeleton()
#     skeleton.load_from_bvh(bvhreader)
#     pose_dir = pose_orientation_general(bvhreader.frames[0], ['Hips', 'RightUpLeg', 'LeftUpLeg'], skeleton)
#     print(pose_dir)
#
#     hip_pos = skeleton.nodes['Hips'].get_global_position_from_euler_frame(bvhreader.frames[0])
#     to_spine_pos = skeleton.nodes['ToSpine'].get_global_position_from_euler_frame(bvhreader.frames[0])
#     bone_dir = hip_pos - to_spine_pos
#     bone_dir_2d = np.array([bone_dir[0], bone_dir[2]])
#     bone_dir_2d = bone_dir_2d/np.linalg.norm(bone_dir_2d)
#     print(bone_dir_2d)


if __name__ == "__main__":
    # run_retarget_single_motion()
    retarget_ACCAD_data()