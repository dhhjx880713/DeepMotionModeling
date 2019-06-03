import os
import collections
import sys
sys.path.insert(0, os.path.abspath('.'))
from mosi_utils_anim.animation_data.retargeting.directional_constraints_retargeting import retarget_motion, \
    estimate_scale_factor, retarget_single_motion, create_direction_constraints, align_ref_frame, retarget_folder

'''
edin skeleton and makehuman cmu skeleton have the same joint list, however, the bone length and joint order are 
different. So the joint mapping can be one-by-one identical mapping 
'''

Edin_skeleton = collections.OrderedDict(
    [
        ('Hips', {'parent': None, 'index': 0}),  #-1
        # ('LHipJoint', {'parent': 'Hips', 'index': 1}),  # 0
        ('LeftUpLeg', {'parent': 'LHipJoint', 'index': 2}),   # 1
        ('LeftLeg', {'parent': 'LeftUpLeg', 'index': 3}), # 2
        ('LeftFoot', {'parent': 'LeftLeg', 'index': 4}), # 3
        ('LeftToeBase', {'parent': 'LeftFoot', 'index': 5}), # 4
        # ('RHipJoint', {'parent': 'Hips', 'index': 6}),  # 5
        ('RightUpLeg', {'parent': 'RHipJoint', 'index': 7}), # 2
        ('RightLeg', {'parent': 'RightUpLeg', 'index': 8}), # 7
        ('RightFoot', {'parent': 'RightLeg', 'index': 9}), # 8
        ('RightToeBase', {'parent': 'RightFoot', 'index': 10}),
        # ('LowerBack', {'parent': 'Hips', 'index': 11}),
        ('Spine', {'parent': 'LowerBack', 'index': 12}),
        ('Spine1', {'parent': 'Spine', 'index': 13}),
        # ('Neck', {'parent': 'Spine1', 'index': 14}),
        ('Neck1', {'parent': 'Neck', 'index': 15}),
        ('Head', {'parent': 'Neck1', 'index': 16}),
        # ('LeftShoulder', {'parent': 'Neck', 'index': 17}),
        ('LeftArm', {'parent': 'LeftShoulder', 'index': 18}),
        ('LeftForeArm', {'parent': 'LeftArm', 'index': 19}),
        ('LeftHand', {'parent': 'LeftForeArm', 'index': 20}),
        # ('LeftFingerBase', {'parent': 'LeftHand', 'index': 21}),
        # ('LeftHandIndex1', {'parent': 'LeftFingerBase', 'index': 22}),
        # ('LThumb', {'parent': 'LeftHand', 'index': 23}),
        # ('RightShoulder', {'parent': 'Neck', 'index': 24}),
        ('RightArm', {'parent': 'RightShoulder', 'index': 25}),
        ('RightForeArm', {'parent': 'RightArm', 'index': 26}),
        ('RightHand', {'parent': 'RightForeArm', 'index': 27}),
        # ('RightFingerBase', {'parent': 'RightHand', 'index': 28}),
        # ('RightHandIndex1', {'parent': 'RightFingerBase', 'index': 29}),
        # ('RThumb', {'parent': 'RightHand', 'index': 30})
    ]
)


EDIN_MAKEHUMAN_JOINT_MAPPING = {x: x for x in Edin_skeleton.keys()}

### zero length bone must be skipped
EDIN_MAKEHUMAN_JOINT_MAPPING = {
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
    # 'Neck': 'Neck',
    # 'Neck1': 'Neck',
    'Head': 'Head',

    # 'LowerBack': 'LowerBack',
    # 'Spine': 'Spine',
    # 'Spine': 'LowerBack',  ## take care if src joint has zero bone length
    'Spine1': 'Spine1',
    # 'LeftShoulder': 'LeftShoulder',
    # 'LeftArm': 'LeftShoulder',
    # 'LeftArm': 'LeftArm',
    'LeftForeArm': 'LeftForeArm',
    'LeftHand': 'LeftHand',

    # 'RightShoulder': 'RightShoulder',
    # 'RightArm': 'RightShoulder',
    # 'RightArm': 'RightArm',
    'RightForeArm': 'RightForeArm',
    'RightHand': 'RightHand',

    # 'LeftArm': 'clavicle_l',
    # 'LeftForeArm': 'upperarm_l',
    # 'LeftHand': 'lowerarm_l',
    # 'LeftHandIndex1': 'hand_l',
    # 'RightArm': 'clavicle_r',
    # 'RightForeArm': 'upperarm_r',
    # 'RightHand': 'lowerarm_r',
    # 'RightHandIndex1': 'hand_r'
}


def run_retarget_single_motion():
    # content_motion_file = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\stylistic_walking\sexy\sexy_normalwalking_16.bvh'
    # content_motion_file = r'E:\gits\motionsynth_code\data\processed\edin_locomotion\locomotion_jog_000_000.bvh'
    # content_motion_file = r'E:\gits\PFNN\data\animations\LocomotionFlat01_000.bvh'
    content_motion_file = r'E:\workspace\projects\retargeting_experiments\test_data\LocomotionFlat01_000_short.bvh'
    dp_framework_motion_file = r'E:\gits\motionsynth_code\data\processed\edin\edin_locomotion\locomotion_jog_000_000.bvh'
    skeleton_file = r'E:\workspace\retargeting\makehuman_characters\fbx\cmu_skeleton\cmu_skeleton.bvh'
    # save_path = r'E:\workspace\projects\variational_style_simulation\retargeted_from_optimization'
    save_path = r'E:\workspace\projects\retargeting_experiments\retargeted_results'
    root_joint = "Hips"
    target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    # src_body_plane = ['thigh_r', 'Root', 'thigh_l']
    retarget_single_motion(dp_framework_motion_file, skeleton_file, content_motion_file, save_path, root_joint,
                           target_body_plane, target_body_plane, EDIN_MAKEHUMAN_JOINT_MAPPING)


def retarget_pfnn_data():
    skeleton_file = r'E:\workspace\retargeting\makehuman_characters\fbx\cmu_skeleton\cmu_skeleton.bvh'
    save_folder = r'E:\workspace\projects\variational_style_simulation\retargeted_bvh_files_mk_cmu_skeleton\pfnn_data'
    src_folder = r'E:\gits\PFNN\data\animations'
    src_body_plane = target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    root_joint = "Hips"
    retarget_folder(src_folder, skeleton_file, save_folder, EDIN_MAKEHUMAN_JOINT_MAPPING, root_joint=root_joint,
                    src_body_plane=src_body_plane, target_body_plane=target_body_plane)



def scan_subfolders(dir):
    '''

    :return:
    '''
    subdirs = next(os.walk(dir))[1]
    if subdirs == []:
        print(dir)
    else:
        for subdir in next(os.walk(dir))[1]:
            scan_subfolders(os.path.join(dir, subdir))


def retarget_folder_iteratively(dir, save_dir, skeleton_file, joint_mappling, root_joint, src_body_plane,
                                target_body_plane):
    '''

    :param dir:
    :return:
    '''
    subdirs = next(os.walk(dir))[1]
    if subdirs == []:
        print(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # print(dir)
        retarget_folder(dir, skeleton_file, save_dir, joint_mappling, root_joint=root_joint,
                        src_body_plane=src_body_plane, target_body_plane=target_body_plane)
    else:
        for subdir in next(os.walk(dir))[1]:
            retarget_folder_iteratively(os.path.join(dir, subdir), os.path.join(save_dir, subdir), skeleton_file,
                                        joint_mappling, root_joint, src_body_plane, target_body_plane)


def retarget_deep_learning_framework_data():
    skeleton_file = r'E:\workspace\retargeting\makehuman_characters\fbx\cmu_skeleton\cmu_skeleton.bvh'
    save_folder = r'E:\workspace\projects\variational_style_simulation\retargeted_bvh_files_mk_cmu_skeleton\deep_learning_framework'
    src_folder = r'E:\gits\motionsynth_code\data\processed'
    src_body_plane = target_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    root_joint = "Hips"
    # subdirs = next(os.walk(src_folder))[1]
    # print(subdirs)
    # print(next(os.walk(src_folder))[1])
    # scan_subfolders(src_folder)
    # for subdir in next(os.walk(src_folder))[1]:
    #     print(type(subdir))
    # retarget_folder(src_folder, skeleton_file, save_folder, EDIN_MAKEHUMAN_JOINT_MAPPING, root_joint=root_joint,
    #                 src_body_plane=src_body_plane, target_body_plane=target_body_plane)
    retarget_folder_iteratively(src_folder, save_folder, skeleton_file, EDIN_MAKEHUMAN_JOINT_MAPPING, root_joint,
                                src_body_plane, target_body_plane)


if __name__ == "__main__":
    run_retarget_single_motion()
    # retarget_pfnn_data()
    # retarget_deep_learning_framework_data()