# encoding: UTF-8

from retarget_motion_using_direction import retarget_motion, estimate_scale_factor, retarget_single_motion, \
    create_direction_constraints, align_ref_frame, retarget_folder
import os


JOINT_MAPPING = {
    'hip': 'pelvis',
    'upperback': 'spine_01',
    'thorax': 'spine_03',
    'lfemur': 'thigh_l',
    'ltibia': 'calf_l',
    'lfoot': 'foot_l',
    'ltoes': 'ball_l',
    'rfemur': 'thigh_r',
    'rtibia': 'calf_r',
    'rfoot': 'foot_r',
    'rtoes': 'ball_r',
    'upperneck': 'neck_01',
    'head': 'head',
    'lhumerus': 'upperarm_l',
    'lradius': 'lowerarm_l',
    'lwrist': 'hand_l',

    'rhumerus': 'upperarm_r',
    'rradius': 'lowerarm_r',
    'rwrist': 'hand_r'
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



def run_retarget_single_motion():
    motion_content_file = r'C:\Users\hadu01\git-repos\motionsynth_code\data\processed\hdm05\HDM_bd_clap1Reps_001_120.bvh'
    target_skeleton_file = r'C:\Users\hadu01\git-repos\ulm\morphablegraphs\python_src\game_engine_target.bvh'
    rest_pose = motion_content_file
    save_dir = r'E:\experiment data\tmp'
    root_joint = 'pelvis'  ## root joint from target skeleton
    src_body_plane = ['lfemur', 'hip', 'rfemur']
    target_body_plane = ['thigh_r', 'Root', 'thigh_l']
    retarget_single_motion(motion_content_file, target_skeleton_file, rest_pose, save_dir, root_joint,
                           src_body_plane, None, JOINT_MAPPING, JOINTS_DOFS)


def retarget_HDM05_data():
    edin_data_folder = r'C:\Users\hadu01\git-repos\motionsynth_code\data\processed'
    target_skeleton_file = r'C:\Users\hadu01\git-repos\ulm\morphablegraphs\python_src\game_engine_target.bvh'
    save_dir = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting'
    root_joint = 'pelvis'
    src_body_plane = ['lfemur', 'hip', 'rfemur']
    target_body_plane = ['thigh_r', 'Root', 'thigh_l']
    actions = ['hdm05']
    # for action in next(os.walk(edin_data_folder))[1]:
    for action in actions:
        src_folder = os.path.join(edin_data_folder, action)
        save_folder = os.path.join(save_dir, action)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        retarget_folder(src_folder, target_skeleton_file, save_folder, JOINT_MAPPING, JOINTS_DOFS, root_joint,
                        src_body_plane, target_body_plane=None)


if __name__ == "__main__":
    # run_retarget_single_motion()
    retarget_HDM05_data()
