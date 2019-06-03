import os
import collections
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/../..')
from mosi_utils_anim.animation_data.retargeting.directional_constraints_retargeting import retarget_motion, \
    estimate_scale_factor, retarget_single_motion, create_direction_constraints, align_ref_frame, retarget_folder


CMU_TO_GAME_ENGINE_MAPPING = {
    "Hips": "pelvis",
    "LeftUpLeg": "thigh_l",
    "LeftLeg": "calf_l",
    "LeftFoot": "foot_l",
    "LeftToeBase": "ball_l",
    "RightUpLeg": "thigh_r",
    "RightLeg": "calf_r",
    "RightFoot": "foot_r",
    "RightToeBase": "ball_r",
    "LowerBack": "spine_01",
    "Spine": "spine_02",
    "Spine1": "spine_03",
    "Neck1": "neck_01",
    "Head": "head",
    "LeftShoulder": "clavicle_l",
    "LeftArm": "upperarm_l",
    "LeftForeArm": "lowerarm_l",
    "LeftHand": "hand_l",
    "RightShoulder": "clavicle_r",
    "RightArm": "upperarm_r",
    "RightForeArm": "lowerarm_r",
    "RightHand": "hand_r"
}


GAME_ENGINE_TO_MH_CMU_MAPPING = {y:x for x, y in CMU_TO_GAME_ENGINE_MAPPING.items()}

GAME_ENGINE_JOINTS_DOFS = {
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
    # "spine_03": ['X', 'Z']
}

MH_CMU_JOINT_DOFS = {GAME_ENGINE_TO_MH_CMU_MAPPING[x]: y for x, y in GAME_ENGINE_JOINTS_DOFS.items()}


def run_retarget_single_motion():
    content_motion_file = r'E:\workspace\tensorflow_results\stylistic_path_following\neutral.bvh'
    skeleton_file = r'E:\workspace\unity_workspace\MG\motion_in_json\cmu_skeleton\angry_fast walking_147.bvh'
    save_path = r'E:\workspace\tensorflow_results\stylistic_path_following\tmp'
    root_joint = "Hips"
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    src_body_plane = ['thigh_r', 'Root', 'thigh_l']
    retarget_single_motion(content_motion_file, skeleton_file, content_motion_file, save_path, root_joint,
                           src_body_plane, target_body_plane, GAME_ENGINE_TO_MH_CMU_MAPPING, MH_CMU_JOINT_DOFS)


def run_retarget_motions():

    # input_folder = r'E:\gits\data\generated_motion\neutral_motions\angry'
    input_folder = r'E:\workspace\tensorflow_results\stylistic_path_following'
    skeleton_file = r'E:\workspace\unity_workspace\MG\motion_in_json\cmu_skeleton\angry_fast walking_147.bvh'
    spline_representation_for_motions(input_folder)
    bvhfiles = glob.glob(os.path.join(input_folder, '*.bvh'))
    print(bvhfiles)
    root_joint = "Hips"
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    src_body_plane = ['thigh_r', 'Root', 'thigh_l']
    save_path = os.path.join(input_folder, 'cmu_skeleton')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for bvhfile in bvhfiles:
        retarget_single_motion(bvhfile, skeleton_file, bvhfile, save_path, root_joint, src_body_plane,
                               target_body_plane, GAME_ENGINE_TO_MH_CMU_MAPPING, MH_CMU_JOINT_DOFS)
    # convert_to_json_unity_folder(save_path)


def retarget_single_motion_using_erik_solution():
    pass


def test():
    save_path = r'E:\gits\data\generated_motion\neutral_motions\angry\cmu_skeleton\corrected'
    convert_to_json_unity_folder(save_path)



if __name__ == "__main__":
    run_retarget_single_motion()
    # run_retarget_motions()
    # test()