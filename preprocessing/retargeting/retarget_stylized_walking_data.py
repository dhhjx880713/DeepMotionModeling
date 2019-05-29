# encoding: UTF-8

from retarget_motion_using_direction import *
from morphablegraphs.animation_data import BVHReader, BVHWriter, SkeletonBuilder
import glob
from retarget_motion_using_direction import retarget_motion, estimate_scale_factor, retarget_single_motion, \
    create_direction_constraints, align_ref_frame, retarget_folder
import os
import copy


## joint mapping from source skeleton to target skeleton
JOINT_MAPPING = dict()
JOINT_MAPPING["root"] = "pelvis"
JOINT_MAPPING["thorax"] = "spine_03"
JOINT_MAPPING["lclavicle"] = "clavicle_l"
JOINT_MAPPING["rclavicle"] = "clavicle_r"
JOINT_MAPPING["lhumerus"] = "upperarm_l"
JOINT_MAPPING["rhumerus"] = "upperarm_r"
JOINT_MAPPING["lradius"] = "lowerarm_l"
JOINT_MAPPING["rradius"] = "lowerarm_r"
JOINT_MAPPING["lhand"] = "hand_l"
JOINT_MAPPING["rhand"] = "hand_r"
JOINT_MAPPING["lfemur"] = "thigh_l"
JOINT_MAPPING["rfemur"] = "thigh_r"
JOINT_MAPPING["ltibia"] = "calf_l"
JOINT_MAPPING["rtibia"] = "calf_r"
JOINT_MAPPING["lfoot"] = "foot_l"
JOINT_MAPPING["rfoot"] = "foot_r"
JOINT_MAPPING["ltoes"] = "ball_l"
JOINT_MAPPING["rtoes"] = "ball_r"
JOINT_MAPPING["head"] = "head"


JOINT_MAPPING = dict()
JOINT_MAPPING["root"] = "pelvis"
JOINT_MAPPING["thorax"] = "spine_03"
JOINT_MAPPING["lclavicle"] = "clavicle_l"
JOINT_MAPPING["rclavicle"] = "clavicle_r"
JOINT_MAPPING["lhumerus"] = "upperarm_l"
JOINT_MAPPING["rhumerus"] = "upperarm_r"
JOINT_MAPPING["lradius"] = "lowerarm_l"
JOINT_MAPPING["rradius"] = "lowerarm_r"
JOINT_MAPPING["lhand"] = "hand_l"
JOINT_MAPPING["rhand"] = "hand_r"
JOINT_MAPPING["lfemur"] = "thigh_l"
JOINT_MAPPING["rfemur"] = "thigh_r"
JOINT_MAPPING["ltibia"] = "calf_l"
JOINT_MAPPING["rtibia"] = "calf_r"
JOINT_MAPPING["lfoot"] = "foot_l"
JOINT_MAPPING["rfoot"] = "foot_r"
JOINT_MAPPING["ltoes"] = "ball_l"
JOINT_MAPPING["rtoes"] = "ball_r"
JOINT_MAPPING["head"] = "head"

JOINT_DOFS_STYLE = {
    "lclavicle": ['X', 'Z'],
    "rclavicle": ['X', 'Z'],
    "thorax": ['X', 'Z'],
    "ltibia": ['X', 'Z'],
    "rtibia": ['X', 'Z'],
    "lfemur": ['X', 'Z'],
    "rfemur": ['X', 'Z'],
    "ltoes": ['X', 'Z'],
    "rtoes": ['X', 'Z'],
    "lradius": ['X', 'Z'],
    "rradius": ['X', 'Z'],
    "lhumerus": ['X', 'Z'],
    "rhumerus": ['X', 'Z'],
    "lhand": ['X', 'Z'],
    "rhand": ['X', 'Z']
}


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


def get_game_engine_to_style_mapping():

    JOINT_MAPPING_TO_STYLE_SKELETON = {}
    for key, value in JOINT_MAPPING.items():
        JOINT_MAPPING_TO_STYLE_SKELETON[value] = key
    return JOINT_MAPPING_TO_STYLE_SKELETON


def run_retarget_single_motion():
    input_file = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\leftStance_game_engine_skeleton_smoothed_grounded\walk_001_3_leftStance_398_446.bvh'
    # input_file = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_pickRight\reach_game_engine_skeleton_new\pickRight_056_4_reach_495_581.bvh'
    # input_file = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_insertBoth\retrieve_game_engine_skeleton_new\insertBoth_take_insert_001_retrieve_5854_5979.bvh'
    ref_file = r'C:\repo\data\1 - MoCap\3 - Cutting\elementary_action_walk\stylized_walking\leftStance\neutral\walk_neutral_normalwalking_29_leftStance_0_81.bvh'
    rest_pose = input_file
    output_folder = r'C:\Users\hadu01\Downloads\code and data (1)\test_motion'
    root_joint = 'root'
    src_body_plane = ['upperarm_r', 'upperarm_l', 'spine_03']
    target_body_plane = ["rfemur", "lfemur", "thorax"]
    joint_mapping = get_game_engine_to_style_mapping()
    retarget_single_motion(input_file, ref_file, rest_pose, output_folder, root_joint, src_body_plane,
                           target_body_plane, joint_mapping, JOINT_DOFS_STYLE)



def retarget_aligned_style_data():
    src_path = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\stylized_walking\rightStance'
    subfolders = ['angry', 'childlike', 'depressed', 'neutral', 'old', 'proud', 'sexy', 'strutting']
    ref_file = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\leftStance_game_engine_skeleton_smoothed\walk_001_2_leftStance_374_420.bvh'
    # save_folder = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\stylized_walking\angry\game_engine_skeleton'
    root_joint = 'pelvis'
    body_plane = ["lfemur", "thorax", "rfemur"]
    for style_type in subfolders:
        src_folder = os.path.join(src_path, style_type)
        save_folder = os.path.join(src_folder, 'game_engine_skeleton')
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        retarget_folder(src_folder, ref_file, save_folder, JOINT_MAPPING, JOINTS_DOFS, root_joint, body_plane)


def retarget_stylized_data():
    stylized_data_folder = r'C:\tmp\stylistic_motion'
    save_dir = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\stylized_data_raw'
    target_skeleton_file = r'C:\Users\hadu01\git-repos\ulm\mg-experiments\mg-tools\mg_analysis\morphablegraphs\python_src\game_engine_target.bvh'
    root_joint = 'pelvis'
    src_body_plane = ["lfemur", "thorax", "rfemur"]
    for action in next(os.walk(stylized_data_folder))[1]:
        print(action)
        src_folder = os.path.join(stylized_data_folder, action)
        save_folder = os.path.join(save_dir, action)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        retarget_folder(src_folder, target_skeleton_file, save_folder, JOINT_MAPPING, JOINTS_DOFS, root_joint,
                        src_body_plane, target_body_plane=None)


def retarget_stylized_data_folder():
    stylized_data_folder = r'C:\tmp\stylistic_motion_left'
    save_dir = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\stylized_data_raw'
    target_skeleton_file = r'../../../game_engine_target.bvh'
    root_joint = 'pelvis'
    src_body_plane = ["lfemur", "thorax", "rfemur"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    retarget_folder(stylized_data_folder, target_skeleton_file, save_dir, JOINT_MAPPING, JOINTS_DOFS, root_joint,
                    src_body_plane, target_body_plane=None)


if __name__ == "__main__":
    # run_retarget_single_motion()
    # retarget_stylized_data()
    retarget_stylized_data_folder()