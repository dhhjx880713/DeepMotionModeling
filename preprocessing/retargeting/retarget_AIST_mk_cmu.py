import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/../..')
from mosi_utils_anim.retargeting.directional_constraints_retargeting import retarget_motion, \
    estimate_scale_factor, retarget_single_motion, create_direction_constraints, align_ref_frame, retarget_folder
import shutil
import glob


bad_samples = [
    'd04_mBR0_ch10'
    'd04_mBR4_ch07',
    'd05_mBR5_ch14',
    'd06_mBR4_ch20',
    'd20_mHO5_ch13',
    'd07_mJB2_ch10',
    'd07_mJB3_ch05',
    'd07_mJB3_ch10',
    'd08_mJB0_ch09',
    'd08_mJB1_ch09',
    'd09_mJB2_ch07',
    'd09_mJB4_ch09',
    'd09_mJB4_ch10',
    'd09_mJB2_ch17',
    'd09_mJS3_ch10',
    'd01_mJS0_ch01',
    'd01_mJS1_ch02',
    'd02_mJS0_ch08',
    'd26_mWA3_ch01',
    'd27_mWA4_ch08',
    'd27_mWA5_ch01',
    'd27_mWA5_ch08',
    'd26_mWA0_ch08'
]


### from source to target
AIST_MAKEHUMAN_JOINT_MAPPING = {
    '0': 'Hips',
    '1': 'LeftUpLeg',
    '4': 'LeftLeg',
    '7': 'LeftFoot',
    '10': 'LeftToeBase',
    '2': 'RightUpLeg',
    '5': 'RightLeg',
    '8': 'RightFoot',
    '11': 'RightToeBase',
    '12': 'Neck',
    '15': 'Head',

    '3': 'LowerBack',
    '6': 'Spine',
    '9': 'Spine1',
    '13': 'LeftShoulder',
    '16': 'LeftArm',
    '18': 'LeftForeArm',
    '20': 'LeftHand',

    '14': 'RightShoulder',
    '17': 'RightArm',
    '19': 'RightForeArm',
    '21': 'RightHand',
}



def retarget_single_file():
    input_file = r'D:\workspace\my_git_repos\capturesysevaluation\data\ART\4kmh.bvh'
    ref_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_T_pose.bvh'
    save_dir = os.path.join(os.path.split(input_file)[0], 'retarget')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    root_joint = "Hips"
    src_body_plane = ['1', '3', '2']
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    retarget_single_motion(input_file, ref_file, rest_pose=input_file, save_dir=save_dir,
                           root_joint=root_joint, src_body_plane=src_body_plane,
                           target_body_plane=target_body_plane, 
                           joint_mapping=AIST_MAKEHUMAN_JOINT_MAPPING,
                           n_frames=1500)


def check_filename(filename):
    is_bad_file = False
    for token in bad_samples:
        if token in filename:
            is_bad_file = True
    return is_bad_file


def filter_bad_samples():
    input_dir = r'E:\workspace\mocap_data\AIST'
    bvhfiles = glob.glob(os.path.join(input_dir, '*.bvh'))
    save_folder = os.path.join(input_dir, 'bad_samples')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for bvhfile in bvhfiles:
        if check_filename(bvhfile):
            filename = os.path.split(bvhfile)[-1]
            shutil.move(bvhfile, os.path.join(save_folder, filename))


def retarget_captury_data():
    input_dir = r'E:\workspace\mocap_data\AIST'
    save_dir = os.path.join(input_dir, 'retargeting')
    skeleton_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_T_pose.bvh'
    src_skeleton_file = r'E:\workspace\mocap_data\AIST\skeleton\skeleton.bvh'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    src_body_plane = ['1', '3', '2']
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    root_joint = "Hips"
    retarget_folder(input_dir, skeleton_file, save_dir, AIST_MAKEHUMAN_JOINT_MAPPING, root_joint=root_joint,
                    src_body_plane=src_body_plane, target_body_plane=target_body_plane, src_skeleton_file=src_skeleton_file)


if __name__ == "__main__":
    filter_bad_samples()
    retarget_captury_data()
    # retarget_single_file()