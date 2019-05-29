# encoding: UTF-8

from retarget_motion_using_direction import retarget_motion, estimate_scale_factor, retarget_single_motion, \
    create_direction_constraints, align_ref_frame, retarget_folder
import os


## map cmu skeleton to game engine skeleton
JOINT_MAP = dict()
JOINT_MAP["hip"] = "pelvis"
# JOINT_MAP["abdomen"] = "spine_01"
# JOINT_MAP["chest"] = "spine_03"
JOINT_MAP["neck"] = "neck_01"
JOINT_MAP["head"] = "head"
JOINT_MAP["rCollar"] = "clavicle_r"
JOINT_MAP["rShldr"] = "upperarm_r"
JOINT_MAP["rForeArm"] = "lowerarm_r"
JOINT_MAP["rHand"] = "hand_r"
# JOINT_MAP["rThumb1"] = "thumb_01_r"
# JOINT_MAP["rThumb2"] = "thumb_02_r"
# JOINT_MAP["rIndex1"] = "index_01_r"
# JOINT_MAP["rIndex2"] = "index_02_r"
# JOINT_MAP["rMid1"] = "middle_01_r"
# JOINT_MAP["rMid2"] = "middle_02_r"
# JOINT_MAP["rRing1"] = "ring_01_r"
# JOINT_MAP["rRing2"] = "ring_02_r"
# JOINT_MAP["rPinky1"] = "pinky_01_r"
# JOINT_MAP["rPinky2"] = "pinky_02_r"
JOINT_MAP["lCollar"] = "clavicle_l"
JOINT_MAP["lShldr"] = "upperarm_l"
JOINT_MAP["lForeArm"] = "lowerarm_l"
JOINT_MAP["lHand"] = "hand_l"
# JOINT_MAP["lThumb1"] = "thumb_01_l"
# JOINT_MAP["lThumb2"] = "thumb_02_l"
# JOINT_MAP["lIndex1"] = "index_01_l"
# JOINT_MAP["lIndex2"] = "index_02_l"
# JOINT_MAP["lMid1"] = "middle_01_l"
# JOINT_MAP["lMid2"] = "middle_02_l"
# JOINT_MAP["lRing1"] = "ring_01_l"
# JOINT_MAP["lRing2"] = "ring_02_l"
# JOINT_MAP["lPinky1"] = "pinky_01_l"
# JOINT_MAP["lPinky2"] = "pinky_02_l"
JOINT_MAP["rButtock"] = "thigh_r"
# JOINT_MAP["rThigh"] = "thigh_r"
JOINT_MAP["rShin"] = "calf_r"
JOINT_MAP["rFoot"] = "foot_r"
JOINT_MAP["rFoot_EndSite"] = "ball_r"
JOINT_MAP["lButtock"] = "thigh_l"
# JOINT_MAP["lThigh"] = "thigh_l"
JOINT_MAP["lShin"] = "calf_l"
JOINT_MAP["lFoot"] = "foot_l"
JOINT_MAP["lFoot_EndSite"] = "ball_l"


CMU_JOINT_MAPPING2 = {
    "Hips": "pelvis",
    "LeftUpLeg": "thigh_l",
    "LeftLeg": "calf_l",
    "LeftFoot": "foot_l",
    "LeftToeBase": "ball_l",
    "RightUpLeg": "thigh_r",
    "RightLeg": "calf_r",
    "RightFoot": "foot_r",
    "RightToeBase": "ball_r",
    "Spine": "spine_01",
    "Spine1": "spine_03",
    "Neck1": "neck_01",
    "Head": "head",
    "LeftArm": "upperarm_l",
    "LeftForeArm": "lowerarm_l",
    "LeftHand": "hand_l",
    "RightArm": "upperarm_r",
    "RightForeArm": "lowerarm_r",
    "RightHand": "hand_r"
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
    # "spine_03": ['X', 'Z']
}


def run_retarget_single_motion():
    motion_content_file = r'E:\BaiduNetdiskDownload\141_16.bvh'
    target_skeleton_file = r'C:\Users\hadu01\git-repos\ulm\mg-experiments\mg-tools\mg_analysis\morphablegraphs\python_src\game_engine_target_large.bvh'
    rest_pose = motion_content_file
    save_dir = r'E:\tmp'
    root_joint = 'pelvis'  ## root joint from target skeleton
    src_body_plane = ['rShldr', 'chest', 'lShldr']
    target_body_plane = ['thigh_r', 'Root', 'thigh_l']
    retarget_single_motion(motion_content_file, target_skeleton_file, rest_pose, save_dir, root_joint,
                           src_body_plane, None, JOINT_MAP, JOINTS_DOFS)



CMU_MAKEHUMAN_JOINT_MAPPING = {
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
    # 'LowerBack': 'LowerBack',
    'Spine': 'Spine',
    'Spine1': 'Spine1',
    # 'Neck': 'Neck',
    'Neck1':'Neck1',
    'Head': 'Head',
    # 'LeftShoulder': 'LeftShoulder',
    'LeftArm': 'LeftArm',
    'LeftForeArm': 'LeftForeArm',
    'LeftHand': 'LeftHand',
    # 'LeftFingerBase': 'LeftFingerBase',
    'LeftHandIndex1': 'LeftHandFinger1',
    # 'LThumb': 'LThumb',
    # 'RightShoulder': 'RightShoulder',
    'RightArm': 'RightArm',
    'RightForeArm': 'RightForeArm',
    'RightHand': 'RightHand',
    # 'RightFingerBase': 'RightFingerBase',
    'RightHandIndex1': 'RightHandFinger1',
    # 'RThumb': 'RThumb'
}


def retarget_between_different_cmu():
    motion_content_file = r'E:\gits\motionsynth_code\data\style_data_xyz\angry_fast walking_147.bvh'
    target_skeleton_file = r'E:\tmp\cmu_rig_skeleton.bvh'
    rest_pose = motion_content_file
    save_dir = r'E:\tmp'
    root_joint = 'Hips'  ## root joint from target skeleton
    src_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    retarget_single_motion(motion_content_file, target_skeleton_file, rest_pose, save_dir, root_joint,
                           src_body_plane, target_body_plane, CMU_MAKEHUMAN_JOINT_MAPPING)


def retarget_cmu_data():
    cmu_data_folder = r'E:\mocap_data\cmu mocap data'
    target_skeleton_file = r'C:\Users\hadu01\git-repos\ulm\morphablegraphs\python_src\game_engine_target.bvh'
    save_dir = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\cmu'
    root_joint = 'pelvis'
    src_body_plane = ['rShldr', 'chest', 'lShldr']
    target_body_plane = ['thigh_r', 'Root', 'thigh_l']
    for action in next(os.walk(cmu_data_folder))[1]:
        src_folder = os.path.join(cmu_data_folder, action)
        save_folder = os.path.join(save_dir, action)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        retarget_folder(src_folder, target_skeleton_file, save_folder, JOINT_MAP, JOINTS_DOFS, root_joint,
                        src_body_plane, target_body_plane=None)


def run_retarget_single_motion2():
    motion_content_file = r'E:\BaiduNetdiskDownload\141_16.bvh'
    target_skeleton_file = r'C:\Users\hadu01\git-repos\ulm\mg-experiments\mg-tools\mg_analysis\morphablegraphs\python_src\game_engine_target.bvh'
    rest_pose = motion_content_file
    save_dir = r'E:\tmp'
    root_joint = 'pelvis'  ## root joint from target skeleton
    src_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    target_body_plane = ['thigh_r', 'Root', 'thigh_l']
    retarget_single_motion(motion_content_file, target_skeleton_file, rest_pose, save_dir, root_joint,
                           src_body_plane, None, CMU_JOINT_MAPPING2, JOINTS_DOFS)


def retarget_cmu_data2():
    cmu_data_folder = r'E:\mocap_data\cmu mocap data'
    target_skeleton_file = r'C:\Users\hadu01\git-repos\ulm\morphablegraphs\python_src\game_engine_target.bvh'
    save_dir = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\cmu'
    root_joint = 'pelvis'
    src_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    target_body_plane = ['thigh_r', 'Root', 'thigh_l']
    actions = ['10', '11', '12', '13', '14']
    for action in actions:
        src_folder = os.path.join(cmu_data_folder, action)
        save_folder = os.path.join(save_dir, action)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        retarget_folder(src_folder, target_skeleton_file, save_folder, CMU_JOINT_MAPPING2, JOINTS_DOFS, root_joint,
                        src_body_plane, target_body_plane=None)


if __name__ == "__main__":
    # run_retarget_single_motion()
    # retarget_cmu_data()
    # run_retarget_single_motion2()
    # retarget_cmu_data2()
    retarget_between_different_cmu()