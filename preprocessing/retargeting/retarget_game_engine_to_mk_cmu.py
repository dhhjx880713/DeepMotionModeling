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


def retarget_game_engine_to_mk_cmu_folder(input_path, output_path, skeleton_file, joint_mappling, root_joint, src_body_plane,
                                          target_body_plane):
    """[summary]
    
    Arguments:
        input_path {str} -- [description]
        output_path {str} -- [description]
        skeleton_file {str} -- [description]
        joint_mappling {dic} -- [description]
        root_joint {str} -- [description]
        src_body_plane {list} -- [description]
        target_body_plane {list} -- [description]
    """                                   
    subdirs = next(os.walk(input_path))[1]
    if subdirs == []:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        retarget_folder(input_path, skeleton_file, output_path, joint_mappling, joints_dofs=MH_CMU_JOINT_DOFS, root_joint=root_joint,
                        src_body_plane=src_body_plane, target_body_plane=target_body_plane)
    else:
        for subdir in subdirs:
            retarget_game_engine_to_mk_cmu_folder(os.path.join(input_path, subdir), os.path.join(output_path, subdir),
            skeleton_file, joint_mappling, root_joint, src_body_plane, target_body_plane)


def retarget_game_engine_to_mk_cmu_single_file():
    content_motion_file = r'E:\workspace\mocap_data\game_engine_retargeting\hdm05\HDM_bd_cartwheelLHandStart1Reps_001_120.bvh'
    skeleton_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_T_pose.bvh'
    save_path = r'E:\workspace\projects\retargeting_experiments\retargeted_results'
    root_joint = "Hips"
    src_body_plane = ['thigh_r', 'Root', 'thigh_l']
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']    
    retarget_single_motion(content_motion_file, skeleton_file, content_motion_file, save_path, root_joint, src_body_plane,
                           target_body_plane, GAME_ENGINE_TO_MH_CMU_MAPPING)


def retarget_accad():
    ACCAD_joint_mapping = {  ## left: target skeleton, right : source skeleton
    "Hips": "pelvis",
    "LeftUpLeg": "thigh_l",
    "LeftLeg": "calf_l",
    "LeftFoot": "foot_l",
    "LeftToeBase": "ball_l",
    "RightUpLeg": "thigh_r",
    "RightLeg": "calf_r",
    "RightFoot": "foot_r",
    "RightToeBase": "ball_r",
    # "LowerBack": "spine_01",
    # "Spine": "spine_02",
    "Spine1": "spine_03",
    # "Neck1": "neck_01",
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
    ACCAD_joint_mapping = {y:x for x, y in ACCAD_joint_mapping.items()}
    # skeleton_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_T_pose.bvh'
    skeleton_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_skeleton.bvh'
    input_path = r'E:\workspace\mocap_data\game_engine_retargeting\ACCAD'
    # output_path = r'E:\workspace\mocap_data\mk_cmu_retargeting\ACCAD'
    output_path = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\ACCAD'
    root_joint = "Hips"
    src_body_plane = ['thigh_r', 'Root', 'thigh_l']
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg'] 
    retarget_game_engine_to_mk_cmu_folder(input_path, output_path, skeleton_file, ACCAD_joint_mapping, root_joint, src_body_plane, target_body_plane)                          


def retarget_pfnn():
    pfnn_joint_mapping = {
    "Hips": "pelvis",
    "LeftUpLeg": "thigh_l",
    "LeftLeg": "calf_l",
    "LeftFoot": "foot_l",
    "LeftToeBase": "ball_l",
    "RightUpLeg": "thigh_r",
    "RightLeg": "calf_r",
    "RightFoot": "foot_r",
    "RightToeBase": "ball_r",
    # "LowerBack": "spine_01",
    # "Spine": "spine_02",
    # "Spine1": "spine_03",
    # "Neck1": "neck_01",
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
    ACCAD_joint_mapping = {y:x for x, y in ACCAD_joint_mapping.items()}
    skeleton_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_T_pose.bvh'
    input_path = r'E:\workspace\mocap_data\game_engine_retargeting\ACCAD'
    output_path = r'E:\workspace\mocap_data\mk_cmu_retargeting\ACCAD'
    root_joint = "Hips"
    src_body_plane = ['thigh_r', 'Root', 'thigh_l']
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg'] 
    retarget_game_engine_to_mk_cmu_folder(input_path, output_path, skeleton_file, ACCAD_joint_mapping, root_joint, src_body_plane, target_body_plane)                          

def retarget_edin():
    joint_mapping = {
    "Hips": "pelvis",
    "LeftUpLeg": "thigh_l",
    "LeftLeg": "calf_l",
    "LeftFoot": "foot_l",
    "LeftToeBase": "ball_l",
    "RightUpLeg": "thigh_r",
    "RightLeg": "calf_r",
    "RightFoot": "foot_r",
    "RightToeBase": "ball_r",
    # "LowerBack": "spine_01",
    # "Spine": "spine_02",
    "Spine1": "spine_03",
    # "Neck1": "neck_01",
    "Head": "head",
    # "LeftShoulder": "clavicle_l",
    "LeftArm": "upperarm_l",
    "LeftForeArm": "lowerarm_l",
    "LeftHand": "hand_l",
    # "RightShoulder": "clavicle_r",
    "RightArm": "upperarm_r",
    "RightForeArm": "lowerarm_r",
    "RightHand": "hand_r"
    }
    joint_mapping = {y:x for x, y in joint_mapping.items()}
    skeleton_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_skeleton.bvh'
    input_path = r'E:\workspace\mocap_data\game_engine_retargeting\edin'
    output_path = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\edin'
    root_joint = "Hips"
    src_body_plane = ['thigh_r', 'Root', 'thigh_l']
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg'] 
    #########################
    # content_motion_file = r'E:\workspace\mocap_data\game_engine_retargeting\edin\edin_locomotion\locomotion_run_000_000.bvh'
    # save_path = r'E:\workspace\projects\retargeting_experiments\retargeted_results'
 
    # retarget_single_motion(content_motion_file, skeleton_file, content_motion_file, save_path, root_joint, src_body_plane,
    #                        target_body_plane, joint_mapping)
    #########################
    retarget_game_engine_to_mk_cmu_folder(input_path, output_path, skeleton_file, joint_mapping, root_joint, src_body_plane, target_body_plane)                          


def retarget_hdm05():
    joint_mapping = {
    "Hips": "pelvis",
    "LeftUpLeg": "thigh_l",
    "LeftLeg": "calf_l",
    "LeftFoot": "foot_l",
    "LeftToeBase": "ball_l",
    "RightUpLeg": "thigh_r",
    "RightLeg": "calf_r",
    "RightFoot": "foot_r",
    "RightToeBase": "ball_r",
    # "LowerBack": "spine_01",
    # "Spine": "spine_02",
    # "Spine1": "spine_03",
    # "Neck1": "neck_01",
    "Head": "head",
    # "LeftShoulder": "clavicle_l",
    "LeftArm": "upperarm_l",
    "LeftForeArm": "lowerarm_l",
    "LeftHand": "hand_l",
    # "RightShoulder": "clavicle_r",
    "RightArm": "upperarm_r",
    "RightForeArm": "lowerarm_r",
    "RightHand": "hand_r"
    }
    joint_mapping = {y:x for x, y in joint_mapping.items()}
    skeleton_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_T_pose.bvh'
    input_path = r'E:\workspace\mocap_data\game_engine_retargeting\hdm05'
    output_path = r'E:\workspace\mocap_data\mk_cmu_retargeting\hdm05'
    root_joint = "Hips"
    src_body_plane = ['thigh_r', 'Root', 'thigh_l']
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg'] 
    retarget_game_engine_to_mk_cmu_folder(input_path, output_path, skeleton_file, joint_mapping, root_joint, src_body_plane, target_body_plane)                          


def retarget_cmu():
    joint_mapping = {
    "Hips": "pelvis",
    "LeftUpLeg": "thigh_l",
    "LeftLeg": "calf_l",
    "LeftFoot": "foot_l",
    "LeftToeBase": "ball_l",
    "RightUpLeg": "thigh_r",
    "RightLeg": "calf_r",
    "RightFoot": "foot_r",
    "RightToeBase": "ball_r",
    # "LowerBack": "spine_01",
    # "Spine": "spine_02",
    "Spine1": "spine_03",
    "Neck1": "neck_01",
    "Head": "head",
    # "LeftShoulder": "clavicle_l",
    "LeftArm": "upperarm_l",
    "LeftForeArm": "lowerarm_l",
    "LeftHand": "hand_l",
    # "RightShoulder": "clavicle_r",
    "RightArm": "upperarm_r",
    "RightForeArm": "lowerarm_r",
    "RightHand": "hand_r"
    }   
    joint_mapping = {y:x for x, y in joint_mapping.items()}
    skeleton_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_skeleton.bvh'
    input_path = r'E:\workspace\mocap_data\game_engine_retargeting\cmu'
    output_path = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\cmu'
    root_joint = "Hips"
    src_body_plane = ['thigh_r', 'Root', 'thigh_l']
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg'] 
    ########################
    content_motion_file = r'E:\workspace\mocap_data\game_engine_retargeting\cmu\32_09.bvh'
    save_path = r'E:\workspace\projects\retargeting_experiments\retargeted_results'
 
    retarget_single_motion(content_motion_file, skeleton_file, content_motion_file, save_path, root_joint, src_body_plane,
                           target_body_plane, joint_mapping)
    #########################
    # retarget_game_engine_to_mk_cmu_folder(input_path, output_path, skeleton_file, joint_mapping, root_joint, src_body_plane, target_body_plane)                          


def retarget_style():
    joint_mapping = {
    "Hips": "pelvis",
    "LeftUpLeg": "thigh_l",
    "LeftLeg": "calf_l",
    "LeftFoot": "foot_l",
    "LeftToeBase": "ball_l",
    "RightUpLeg": "thigh_r",
    "RightLeg": "calf_r",
    "RightFoot": "foot_r",
    "RightToeBase": "ball_r",
    # "LowerBack": "spine_01",
    # "Spine": "spine_02",
    # "Spine1": "spine_03",
    # "Neck1": "neck_01",
    # "Head": "head",
    "LeftShoulder": "clavicle_l",
    "LeftArm": "upperarm_l",
    "LeftForeArm": "lowerarm_l",
    "LeftHand": "hand_l",
    "RightShoulder": "clavicle_r",
    "RightArm": "upperarm_r",
    "RightForeArm": "lowerarm_r",
    "RightHand": "hand_r"
    }   
    joint_mapping = {y:x for x, y in joint_mapping.items()}
    skeleton_file = r'D:\workspace\mocap_data\skeleton_template\mk_cmu_skeleton.bvh'
    # input_path = r'E:\workspace\mocap_data\game_engine_retargeting\stylized_data_raw'
    # output_path = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\stylized_data_raw'
    input_path = r'D:\workspace\experiment data\cutted_holden_data_walking\game_engine_retargeting\childlike'
    output_path = os.path.join(input_path, 'mk_cmu')
    root_joint = "Hips"
    src_body_plane = ['thigh_r', 'Root', 'thigh_l']
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg'] 
    ########################
    # content_motion_file = r'E:\workspace\mocap_data\game_engine_retargeting\stylized_data_raw\sexy_normal walking_48.bvh'
    # save_path = r'E:\workspace\projects\retargeting_experiments\retargeted_results'
 
    # retarget_single_motion(content_motion_file, skeleton_file, content_motion_file, save_path, root_joint, src_body_plane,
    #                        target_body_plane, joint_mapping)
    #########################
    retarget_game_engine_to_mk_cmu_folder(input_path, output_path, skeleton_file, joint_mapping, root_joint, src_body_plane, target_body_plane)                          


def retarget_ulm():
    joint_mapping = {
    "Hips": "pelvis",
    "LeftUpLeg": "thigh_l",
    "LeftLeg": "calf_l",
    "LeftFoot": "foot_l",
    "LeftToeBase": "ball_l",
    "RightUpLeg": "thigh_r",
    "RightLeg": "calf_r",
    "RightFoot": "foot_r",
    "RightToeBase": "ball_r",
    # "LowerBack": "spine_01",
    # "Spine": "spine_02",
    # "Spine1": "spine_03",
    # "Neck1": "neck_01",
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
    joint_mapping = {y:x for x, y in joint_mapping.items()}
    skeleton_file = r'D:\workspace\mocap_data\skeleton_template\mk_cmu_skeleton.bvh'
    input_path = r'D:\workspace\mocap_data\game_engine_retargeting\ulm_locomotion'
    output_path = r'D:\workspace\mocap_data\mk_cmu_retargeting_default_pose\sulm_locomotion'

    root_joint = "Hips"
    src_body_plane = ['thigh_r', 'Root', 'thigh_l']
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg'] 
    ########################
    # content_motion_file = r'E:\workspace\mocap_data\game_engine_retargeting\ulm_locomotion\Take_walk\walk_001_1.bvh'
    # save_path = r'E:\workspace\projects\retargeting_experiments\retargeted_results'
 
    # retarget_single_motion(content_motion_file, skeleton_file, content_motion_file, save_path, root_joint, src_body_plane,
    #                        target_body_plane, joint_mapping)
    #########################
    retarget_game_engine_to_mk_cmu_folder(input_path, output_path, skeleton_file, joint_mapping, root_joint, src_body_plane, target_body_plane)                          



if __name__ == "__main__":
    # retarget_game_engine_to_mk_cmu_single_file()
    # input_path = r'E:\workspace\mocap_data\game_engine_retargeting\ACCAD'
    # input_path = r'E:\workspace\mocap_data\game_engine_retargeting\cmu'
    # input_path = r'E:\workspace\mocap_data\game_engine_retargeting\edin'
    # input_path = r'E:\workspace\mocap_data\game_engine_retargeting\hdm05'
    # input_path = r'E:\workspace\mocap_data\game_engine_retargeting\ulm'
    # input_path = r'E:\workspace\mocap_data\game_engine_retargeting\ulm_locomotion'
    # input_path = r'E:\workspace\mocap_data\test\game_engine'
    # output_path = r'E:\workspace\mocap_data\test\cmu'
    # skeleton_file = r'E:\workspace\retargeting\makehuman_characters\fbx\cmu_skeleton\cmu_skeleton.bvh'
    # root_joint = "Hips"
    # src_body_plane = ['thigh_r', 'Root', 'thigh_l']
    # target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg'] 
    # retarget_game_engine_to_mk_cmu_folder(input_path, output_path, skeleton_file, GAME_ENGINE_TO_MH_CMU_MAPPING, root_joint,
    #                                       src_body_plane, target_body_plane) 
    # retarget_accad()
    # retarget_hdm05()
    # retarget_cmu()
    # retarget_edin()
    retarget_style()
    # retarget_ulm()
