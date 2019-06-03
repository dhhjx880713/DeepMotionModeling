# encoding: UTF-8
from retarget_motion_using_direction import retarget_motion, estimate_scale_factor, retarget_single_motion, \
    create_direction_constraints, align_ref_frame, retarget_folder
import os


Edin_joint_mapping = {
    'Hips': 'pelvis',
    'Spine': 'spine_01',
    'Spine1': 'spine_03',
    'LeftUpLeg': 'thigh_l',
    'LeftLeg': 'calf_l',
    'LeftFoot': 'foot_l',
    'LeftToeBase': 'ball_l',
    'RightUpLeg': 'thigh_r',
    'RightLeg': 'calf_r',
    'RightFoot': 'foot_r',
    'RightToeBase': 'ball_r',
    'Neck1': 'neck_01',
    'Head': 'head',
    'LeftArm': 'upperarm_l',
    'LeftForeArm': 'lowerarm_l',
    'LeftHand': 'hand_l',
    'RightArm': 'upperarm_r',
    'RightForeArm': 'lowerarm_r',
    'RightHand': 'hand_r'
    # 'LeftArm': 'clavicle_l',
    # 'LeftForeArm': 'upperarm_l',
    # 'LeftHand': 'lowerarm_l',
    # 'LeftHandIndex1': 'hand_l',
    # 'RightArm': 'clavicle_r',
    # 'RightForeArm': 'upperarm_r',
    # 'RightHand': 'lowerarm_r',
    # 'RightHandIndex1': 'hand_r'
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
    motion_content_file = r'E:\gits\motionsynth_code\data\processed\edin\edin_locomotion\locomotion_jog_000_000.bvh'
    target_skeleton_file = r'C:\Users\hadu01\git-repos\ulm\mg-experiments\mg-tools\mg_analysis\morphablegraphs\python_src\game_engine_target.bvh'
    rest_pose = motion_content_file
    save_dir = r'E:\tmp'
    constrained_joints = ['LeftFoot', 'RightFoot']
    root_joint = 'pelvis'  ## root joint from target skeleton
    src_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    target_body_plane = ['thigh_r', 'Root', 'thigh_l']
    retarget_single_motion(motion_content_file, target_skeleton_file, rest_pose, save_dir, root_joint,
                           src_body_plane, None, Edin_joint_mapping, JOINTS_DOFS, constrained_joints)



def scale_motion():
    from morphablegraphs.animation_data import BVHReader, SkeletonBuilder, BVHWriter
    import numpy as np
    motion_content_file = r'E:\gits\motionsynth_code\data\processed\edin\edin_locomotion\locomotion_jog_001_003.bvh'
    bvhreader = BVHReader(motion_content_file)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    scale_factor =6.07743994723
    skeleton.scale(scale_factor)
    bvhreader.frames[:, :3] = bvhreader.frames[:, :3] * scale_factor
    # BVHWriter(r'E:\experiment data\tmp\scaled_motion.bvh', skeleton, bvhreader.frames, skeleton.frame_time,
    #           is_quaternion=False)
    target_joint = 'Hips'
    root_pos = np.zeros((len(bvhreader.frames), 3))
    for i in range(len(bvhreader.frames)):
        root_pos[i] = skeleton.nodes[target_joint].get_global_position_from_euler_frame(bvhreader.frames[i])
    # v1 = root_pos[1:] - root_pos[:-1]
    print(root_pos)

    test_file1 = r'E:\experiment data\tmp\locomotion_jog_000_001.bvh'
    bvhreader1 = BVHReader(test_file1)
    print(bvhreader1.frames[:, :3])
    # skeleton1 = SkeletonBuilder().load_from_bvh(bvhreader1)
    # root_joint = 'pelvis'
    # root_pos = np.zeros((len(bvhreader1.frames), 3))
    # for i in range(len(bvhreader1.frames)):
    #     root_pos[i] = skeleton1.nodes[root_joint].get_global_position_from_euler_frame(bvhreader1.frames[i])
    # v2 = root_pos[1:] - root_pos[:-1]
    # print(v2)



def scale_estimation_test():
    from morphablegraphs.animation_data import BVHReader, SkeletonBuilder
    from retarget_motion_using_direction import estimate_scale_factor
    # motion_content_file = r'C:\Users\hadu01\git-repos\motionsynth_code\data\processed\edin\edin_locomotion\locomotion_run_000_000.bvh'
    motion_content_file = r'C:\Users\hadu01\git-repos\motionsynth_code\data\processed\edin\edin_locomotion\rest.bvh'
    target_skeleton_file = r'C:\Users\hadu01\git-repos\ulm\morphablegraphs\python_src\game_engine_target.bvh'
    rest_pose = motion_content_file
    save_dir = r'E:\experiment data\tmp'
    root_joint = 'pelvis'  ## root joint from target skeleton
    src_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    scale = estimate_scale_factor(motion_content_file, root_joint, target_skeleton_file, Edin_joint_mapping)
    print(scale)

    # src_bvh = BVHReader(motion_content_file)
    # src_skeleton = SkeletonBuilder().load_from_bvh(src_bvh)
    # target_bvh = BVHReader(target_skeleton_file)
    # target_skeleton = SkeletonBuilder().load_from_bvh(target_bvh)
    # src_leg = ['LeftLeg', 'LeftFoot']
    # src_len = 0
    # target_len = 0
    # for joint in src_leg:
    #     joint_len = np.linalg.norm(src_skeleton.nodes[joint].offset)
    #     target_joint_len = np.linalg.norm(target_skeleton.nodes[Edin_joint_mapping[joint]].offset)
    #     src_len += joint_len
    #     target_len += target_joint_len
    # print(target_len/src_len)


def retarget_Edinburgh_data():
    edin_data_folder = r'E:\gits\motionsynth_code\data\processed'
    target_skeleton_file = r'C:\Users\hadu01\git-repos\ulm\mg-experiments\mg-tools\mg_analysis\morphablegraphs\python_src\game_engine_target.bvh'
    save_dir = r'E:\workspace\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting'
    root_joint = 'pelvis'
    src_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    target_body_plane = ['thigh_r', 'Root', 'thigh_l']
    actions = ['cmu']
    # for action in next(os.walk(edin_data_folder))[1]:
    for action in actions:
        src_folder = os.path.join(edin_data_folder, action)
        save_folder = os.path.join(save_dir, action)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        retarget_folder(src_folder, target_skeleton_file, save_folder, Edin_joint_mapping, JOINTS_DOFS, root_joint,
                        src_body_plane, target_body_plane=None)


def test():
    from morphablegraphs.utilities import load_json_file
    import numpy as np
    test_file = r'E:\tmp\path_data\spline16.panim'
    test_data = load_json_file(test_file)
    print(test_data.keys())
    print(test_data['skeleton'])
    motion_data = np.array(test_data['motion_data'])
    print(motion_data.shape)


if __name__ == "__main__":
    # run_retarget_single_motion()
    retarget_Edinburgh_data()
    # scale_estimation_test()
    # scale_motion()
    # test()