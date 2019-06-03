# encoding: UTF-8
from retarget_motion_using_direction import retarget_motion, estimate_scale_factor, retarget_single_motion, \
    create_direction_constraints, align_ref_frame, retarget_folder
import os


ULM_TO_GAME_ENGINE_MAP = dict()
ULM_TO_GAME_ENGINE_MAP["Hips"] = "pelvis"
ULM_TO_GAME_ENGINE_MAP["Spine"] = "spine_01"
ULM_TO_GAME_ENGINE_MAP["Spine1"] = "spine_03"
ULM_TO_GAME_ENGINE_MAP["LeftShoulder"] = "clavicle_l"
ULM_TO_GAME_ENGINE_MAP["RightShoulder"] = "clavicle_r"
ULM_TO_GAME_ENGINE_MAP["LeftArm"] = "upperarm_l"
ULM_TO_GAME_ENGINE_MAP["RightArm"] = "upperarm_r"
ULM_TO_GAME_ENGINE_MAP["LeftForeArm"] = "lowerarm_l"
ULM_TO_GAME_ENGINE_MAP["RightForeArm"] = "lowerarm_r"
ULM_TO_GAME_ENGINE_MAP["LeftHand"] = "hand_l"
ULM_TO_GAME_ENGINE_MAP["RightHand"] = "hand_r"
ULM_TO_GAME_ENGINE_MAP["LeftUpLeg"] = "thigh_l"
ULM_TO_GAME_ENGINE_MAP["RightUpLeg"] = "thigh_r"
ULM_TO_GAME_ENGINE_MAP["LeftLeg"] = "calf_l"
ULM_TO_GAME_ENGINE_MAP["RightLeg"] = "calf_r"
ULM_TO_GAME_ENGINE_MAP["LeftFoot"] = "foot_l"
ULM_TO_GAME_ENGINE_MAP["RightFoot"] = "foot_r"
ULM_TO_GAME_ENGINE_MAP["LeftToeBase"] = "ball_l"
ULM_TO_GAME_ENGINE_MAP["RightToeBase"] = "ball_r"
ULM_TO_GAME_ENGINE_MAP["Head"] = "head"

JOINTS_DOFS = {
    "calf_l": ['X', 'Z'],
    "calf_r": ['X', 'Z'],
    "thigh_l": ['X', 'Z'],
    "thigh_r": ['X', 'Z'],
    "LeftLeg": ['X', 'Z'],
    "RightLeg": ['X', 'Z'],
    "ball_l": ['X', 'Z'],
    "ball_r": ['X', 'Z'],
    "clavicle_l": ['X', 'Z'],
    "clavicle_r": ['X', 'Z'],
    "upperarm_l": ['X', 'Z'],
    "upperarm_r": ['X', 'Z'],
    "lowerarm_l": ['X', 'Z'],
    "lowerarm_r": ['X', 'Z'],
    "hand_l": ['X', 'Z'],
    "hand_r": ['X', 'Z']
}


def run_retarget_single_motion():
    # target_skeleton_file = r'C:\Program Files\Blender Foundation\Blender\2.78\scripts\addons\locomotion\T_pose.bvh'
    target_skeleton_file = r'C:\Users\hadu01\git-repos\ulm\morphablegraphs\python_src\game_engine_target.bvh'
    motion_content_file = r'C:\repo\data\1 - MoCap\1 - Rawdata\Take_walk_s\walk_s_001.bvh'
    # motion_content_file = r'C:\repo\data\1 - MoCap\1 - Rawdata\Take_walk\walk_002_2.bvh'
    rest_pose = motion_content_file
    save_dir = r'E:\experiment data\tmp'
    root_joint = 'pelvis'  ## root joint from target file
    src_body_plane = ["RightUpLeg", "LeftUpLeg", "Spine"]
    target_body_plane = ['thigh_r', 'thigh_l', 'spine_03']
    retarget_single_motion(motion_content_file, target_skeleton_file, rest_pose, save_dir, root_joint, src_body_plane,
                           None, ULM_TO_GAME_ENGINE_MAP, JOINTS_DOFS)


def retraget_ulm_data():
    ulm_data_folder = r'C:\repo\data\1 - MoCap\1 - Rawdata'
    save_dir = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\ulm'
    target_skeleton_file = r'../../../game_engine_target.bvh'
    root_joint = 'pelvis'
    src_body_plane = ["RightUpLeg", "LeftUpLeg", "Spine"]
    actions = ['Take_screw']
    # for action in next(os.walk(ulm_data_folder))[1]:
    for action in actions:
        print(action)
        src_folder = os.path.join(ulm_data_folder, action)
        save_folder = os.path.join(save_dir, action)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        retarget_folder(src_folder, target_skeleton_file, save_folder, ULM_TO_GAME_ENGINE_MAP, JOINTS_DOFS, root_joint,
                        src_body_plane, target_body_plane=None)


if __name__ == "__main__":
    # run_retarget_single_motion()
    retraget_ulm_data()
