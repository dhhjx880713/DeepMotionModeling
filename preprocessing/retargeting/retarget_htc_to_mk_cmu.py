import os
import collections
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/../..')
from mosi_utils_anim.animation_data.retargeting.directional_constraints_retargeting import retarget_motion, \
    estimate_scale_factor, retarget_single_motion, create_direction_constraints, align_ref_frame, retarget_folder



### from source to target
HTC_MAKEHUMAN_JOINT_MAPPING = {
    'pelvis': 'Hips',
    'leftUpperLeg': 'LeftUpLeg',
    'leftLowerLeg': 'LeftLeg',
    'leftFoot': 'LeftFoot',
    'leftFoot_EndSite': 'LeftToeBase',
    'rightUpperLeg': 'RightUpLeg',
    'rightLowerLeg': 'RightLeg',
    'rightFoot': 'RightFoot',
    'rightFoot_EndSite': 'RightToeBase',
    # 'Neck': 'Neck',
    'neck': 'Neck',
    'head': 'Head',

    'upperLumbarSpine': 'LowerBack',
    # 'Spine': 'Spine',
    # 'Spine': 'LowerBack',  ## take care if src joint has zero bone length
    'upperThoracicSpine': 'Spine1',
    'leftShoulder': 'LeftShoulder',
    'leftUpperArm': 'LeftArm',
    'leftLowerArm': 'LeftForeArm',
    'leftHand': 'LeftHand',

    'rightShoulder': 'RightShoulder',
    'rightUpperArm': 'RightArm',
    'rightLowerArm': 'RightForeArm',
    'rightHand': 'RightHand',

    # 'LeftArm': 'clavicle_l',
    # 'LeftForeArm': 'upperarm_l',
    # 'LeftHand': 'lowerarm_l',
    # 'LeftHandIndex1': 'hand_l',
    # 'RightArm': 'clavicle_r',
    # 'RightForeArm': 'upperarm_r',
    # 'RightHand': 'lowerarm_r',
    # 'RightHandIndex1': 'hand_r'
}


def preprocessing():
    import glob
    from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder
    from mosi_utils_anim.animation_data.bvh import BVHWriter
    from mosi_utils_anim.animation_data.utils import transform_euler_frames

    data_folder = r'D:\workspace\projects\DHM2020\MotionCaptureData\HTC_Vive'
    export_folder = os.path.join(data_folder, 'processed')
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    bvhfiles = glob.glob(os.path.join(data_folder, '*.bvh'))
    for bvhfile in bvhfiles:
        filename = os.path.split(bvhfile)[-1]
        bvhreader = BVHReader(bvhfile)
        skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
        # bvhreader.scale()
        rotated_euler_frames = transform_euler_frames(bvhreader.frames, [180, 0, 0], offset=[0, 0, 0])
        BVHWriter(os.path.join(export_folder, filename), skeleton, rotated_euler_frames, skeleton.frame_time)


def retarget_captury_data():
    input_dir = r'D:\workspace\my_git_repos\capturesysevaluation\data\preprocessed\htc'
    save_dir = os.path.join(input_dir, 'retargeting')
    skeleton_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_T_pose.bvh'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    src_body_plane = ['lowerLumbarSpine', 'leftUpperLeg', 'rightUpperLeg']
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    root_joint = "Hips"
    retarget_folder(input_dir, skeleton_file, save_dir, HTC_MAKEHUMAN_JOINT_MAPPING, root_joint=root_joint,
                    src_body_plane=src_body_plane, target_body_plane=target_body_plane)


if __name__ == "__main__":
    # retarget_captury_data()
    preprocessing()