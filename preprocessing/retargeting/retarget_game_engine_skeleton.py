# encoding: UTF-8
import os
import collections
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/../..')
from mosi_utils_anim.animation_data.retargeting.directional_constraints_retargeting import retarget_motion, \
    estimate_scale_factor, retarget_single_motion, create_direction_constraints, align_ref_frame, retarget_folder
import glob
import os
import copy
import numpy as np


UPPERBODY_MAP = dict()
UPPERBODY_MAP["pelvis"] = "pelvis"
UPPERBODY_MAP["spine_01"] = "spine_01"
UPPERBODY_MAP["spine_02"] = "spine_02"
UPPERBODY_MAP["spine_03"] = "spine_03"
UPPERBODY_MAP["neck_01"] = "neck_01"
UPPERBODY_MAP["clavicle_l"] = "clavicle_l"
UPPERBODY_MAP["clavicle_r"] = "clavicle_r"
UPPERBODY_MAP["head"] = "head"

LOWERBODY_MAP = dict()
LOWERBODY_MAP['pelvis'] = 'pelvis'
LOWERBODY_MAP['thigh_l'] = 'thigh_l'
LOWERBODY_MAP['calf_l'] = 'calf_l'
LOWERBODY_MAP['foot_l'] = 'foot_l'
LOWERBODY_MAP['ball_l'] = 'ball_l'
LOWERBODY_MAP['thigh_r'] = 'thigh_r'
LOWERBODY_MAP['calf_r'] = 'calf_r'
LOWERBODY_MAP['foot_r'] = 'foot_r'
LOWERBODY_MAP['ball_r'] = 'ball_r'

JOINTS_DOFS = {
    "clavicle_l": ['X', 'Z'],
    "clavicle_r": ['X', 'Z'],
    "spine_03": ['X', 'Z'],
    "calf_l": ['X', 'Z'],
    "calf_r": ['X', 'Z'],
    "thigh_l": ['X', 'Z'],
    "thigh_r": ['X', 'Z'],
    "LeftLeg": ['X', 'Z'],
    "RightLeg": ['X', 'Z'],
    "ball_l": ['X', 'Z'],
    "ball_r": ['X', 'Z'],
    "upperarm_l": ['X', 'Z'],
    "upperarm_r": ['X', 'Z'],
    "lowerarm_l": ['X', 'Z'],
    "lowerarm_r": ['X', 'Z'],
    "hand_l": ['X', 'Z'],
    "hand_r": ['X', 'Z']
}


def create_direction_constraints(joint_map, src_skeleton, euler_frame):
    targets = {}
    for joint in joint_map.keys():
        create_direction_constraints_recursively(joint, targets, src_skeleton, euler_frame, joint_map)
    return targets


def create_direction_constraints_recursively(joint, targets, src_skeleton, euler_frame, JOINT_MAP):
    '''
    for given joint, first check it is in the JOINT_MAP or not.
    for all the joint's children, if the child is in JOINT_MAP table, create a direction target from

    recursively search the predecessors until one of the predecessor is in MAP_joint
    :param joint:
    :param targets:
    :param src_skeleton:
    :return:
    '''
    src_joint = src_skeleton.nodes[joint].parent
    ## only works for rocketbox skeleton
    pose_dir = pose_orientation_euler(euler_frame)
    targets['pose_dir'] = pose_dir
    while src_joint is not None:
        if src_joint.node_name in JOINT_MAP.keys():
            src_pos = src_joint.get_global_position_from_euler_frame(euler_frame)
            joint_pos = src_skeleton.nodes[joint].get_global_position_from_euler_frame(euler_frame)
            assert np.linalg.norm(joint_pos - src_pos), ('Bone direction should not be zero!')
            bone_dir = (joint_pos - src_pos)/np.linalg.norm(joint_pos - src_pos)
            #######
            # if src_joint.node_name == "LeftShoulder" or src_joint.node_name == "RightShoulder":
            #     bone_dir[1] = 0.0
            #     bone_dir = bone_dir/np.linalg.norm(bone_dir)

            if JOINT_MAP[src_joint.node_name] not in targets.keys():
                targets[JOINT_MAP[src_joint.node_name]] = {JOINT_MAP[joint]: bone_dir}
            else:
                targets[JOINT_MAP[src_joint.node_name]][JOINT_MAP[joint]] = bone_dir
            break
        src_joint = src_joint.parent


def retarget_folder():
    src_folder = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\rightStance\walk_001_3_rightStance_354_398.bvh'
    src_rest_pose = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\rightStance_female\walk_001_2_rightStance_420_460.bvh'
    output_folder = r'E:\processed data\biomotion\rightStance_female'

    ref_file = r'C:\Users\hadu01\git-repos\ulm\morphablegraphs\python_src\game_engine_target.bvh'
    ref_bvhreader = BVHReader(ref_file)
    ref_skeleton = Skeleton()
    ref_skeleton.load_from_bvh(ref_bvhreader)

    root_joint = 'pelvis'
    skeleton_scale_factor = estimate_scale_factor(src_rest_pose, root_joint, ref_file, UPPERBODY_MAP)

    bvhfiles = glob.glob(os.path.join(src_folder, '*.bvh'))
    for bvhfile in bvhfiles:
        bvhreader = BVHReader(bvhfile)
        skeleton = Skeleton()
        skeleton.load_from_bvh(bvhreader)

        filename = os.path.split(bvhfile)[-1]
        print(filename)
        out_frames = []
        targets = []
        n_frames = len(bvhreader.frames)
        # n_frames = 2
        # create constraints for each frame
        for i in range(n_frames):
            targets.append(create_direction_constraints(UPPERBODY_MAP, skeleton, bvhreader.frames[i]))

        # retarget motion
        for i in range(n_frames):
            # print(i)
            if i == 0:
                new_frame = ref_bvhreader.frames[0]
                # new_frame = np.zeros(len(ref_bvhreader.frames[0]))
            else:
                # take the previous frame as initial guess
                new_frame = copy.deepcopy(out_frames[i-1])
            new_frame[:3] = bvhreader.frames[i][:3] * skeleton_scale_factor
            retarget_motion(root_joint, targets[i], ref_skeleton, new_frame)
            out_frames.append(new_frame)

        BVHWriter(os.path.join(output_folder, filename), ref_skeleton, out_frames, ref_skeleton.frame_time,
                  is_quaternion=False)


def retarget_single_motion():
    ref_bvh = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\rightStance_game_engine_skeleton_smoothed\walk_001_3_rightStance_540_590.bvh'
    # ref_bvh = r'C:\Users\hadu01\git-repos\ulm\mg-unity-integration\morphablegraphs\python_src\game_engine_target.bvh'
    bvhfile = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\beginRightStance_game_engine_skeleton_new_grounded\walk_s_007_beginRightStance_124_214.bvh'
    ref_bvhreader = BVHReader(ref_bvh)
    ref_skeleton = Skeleton()
    ref_skeleton.load_from_bvh(ref_bvhreader)
    output_folder = r'E:\tmp'
    bvhreader = BVHReader(bvhfile)
    skeleton = Skeleton()
    skeleton.load_from_bvh(bvhreader)
    root_joint = 'pelvis'
    filename = os.path.split(ref_bvh)[-1]
    # skeleton_scale_factor = 1.0
    # print(filename)
    out_frames = []
    targets = []
    n_frames = len(ref_bvhreader.frames)

    ############ create constraints for each frame
    for i in range(n_frames):
        # targets.append(create_direction_constraints(UPPERBODY_MAP, skeleton, bvhreader.frames[0]))
        ## map to target pose
        upperbody_target = create_direction_constraints(UPPERBODY_MAP, skeleton, bvhreader.frames[0])

        ## keep the original lower body pose
        lowerbody_target = create_direction_constraints(LOWERBODY_MAP, skeleton, ref_bvhreader.frames[i])

        ## combine the constraints
        # targets.append(dict(list(upperbody_target.items()) + list(lowerbody_target.items())))
        combine_target = dict()
        for key, value in upperbody_target.iteritems():
            if key not in lowerbody_target.keys():
                combine_target[key] = value
            elif key == 'pose_dir':
                combine_target[key] = lowerbody_target[key]
            else:
                combine_target[key] = dict(list(upperbody_target[key].items()) + list(lowerbody_target[key].items()))
        for key, value in lowerbody_target.iteritems():
            if key not in upperbody_target.keys():
                combine_target[key] = value
        targets.append(combine_target)

    ###########retarget motion
    for i in range(n_frames):
    # for i in range(20):
        print(i)
        ## first rotate reference frame based on pose direction
        pose_dir = targets[i]['pose_dir']
        # print('target_dir: ', pose_dir)
        if i == 0:
            ref_frame = align_ref_frame(ref_bvhreader.frames[0], pose_dir, ref_skeleton)
            # new_frame = ref_bvhreader.frames[0]
            # new_frame = np.zeros(len(ref_bvhreader.frames[0]))
        else:
            # take the previous frame as initial guess
            new_frame = copy.deepcopy(ref_bvhreader.frames[i])
            # new_frame = copy.deepcopy(ref_bvhreader.frames[0])
            ref_frame = align_ref_frame(new_frame, pose_dir, ref_skeleton)
        # ref_frame[:3] = bvhreader.frames[i][:3] * skeleton_scale_factor
        retarget_motion(root_joint, targets[i], ref_skeleton, ref_frame, JOINTS_DOFS)
        out_frames.append(ref_frame)

    BVHWriter(os.path.join(output_folder, filename), ref_skeleton, out_frames, ref_skeleton.frame_time,
              is_quaternion=False)


def align_ref_frame(euler_frame, target_dir, skeleton):
    '''

    :param euler_frame:
    :param target_dir: 2D vector
    :return:
    '''
    pose_dir = get_game_engine_skeleton_pose_dir(euler_frame, skeleton)
    # print('pose_dir:', pose_dir)
    rotation_angle = get_rotation_angle(target_dir, pose_dir)
    rotated_frame = transform_euler_frame(euler_frame,
                                          [0, rotation_angle, 0],
                                          [0, 0, 0])
    return rotated_frame


def get_game_engine_skeleton_pose_dir(euler_frame, skeleton):
    game_eigine_pos = skeleton.nodes['Game_engine'].get_global_position_from_euler_frame(euler_frame)
    root_pos = skeleton.nodes['Root'].get_global_position_from_euler_frame(euler_frame)
    # print('game engine: ', game_eigine_pos)
    # print('root: ', root_pos)
    dir = game_eigine_pos - root_pos
    dir_2d = np.array([dir[0], dir[2]])
    return dir_2d/np.linalg.norm(dir_2d)


def retarget_motion_primitive(elementary_action,
                              motion_primitive):
    aligned_data_folder = get_aligned_data_folder(elementary_action, motion_primitive)
    output_folder = os.path.join(aligned_data_folder, 'upperbody_pose_fix')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    ref_file = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\beginRightStance_game_engine_skeleton_new_grounded\walk_s_007_beginRightStance_124_214.bvh'
    ref_bvhreader = BVHReader(ref_file)
    ref_skeleton = Skeleton()
    ref_skeleton.load_from_bvh(ref_bvhreader)
    root_joint = 'pelvis'

    bvhfiles = glob.glob(os.path.join(aligned_data_folder, '*.bvh'))
    for bvhfile in bvhfiles:
        bvhreader = BVHReader(bvhfile)
        skeleton = Skeleton()
        skeleton.load_from_bvh(bvhreader)
        filename = os.path.split(bvhfile)[-1]
        print(filename)
        out_frames = []
        targets = []
        n_frames = len(bvhreader.frames)
        # create constraints for each frame
        for i in range(n_frames):
            upperbody_target = create_direction_constraints(UPPERBODY_MAP, skeleton, ref_bvhreader.frames[0])
            lowerbody_target = create_direction_constraints(LOWERBODY_MAP, skeleton, bvhreader.frames[i])
            combine_target = dict()
            for key, value in upperbody_target.iteritems():
                if key not in lowerbody_target.keys():
                    combine_target[key] = value
                elif key == 'pose_dir':
                    combine_target[key] = lowerbody_target[key]
                else:
                    combine_target[key] = dict(list(upperbody_target[key].items()) + list(lowerbody_target[key].items()))
            for key, value in lowerbody_target.iteritems():
                if key not in upperbody_target.keys():
                    combine_target[key] = value
            targets.append(combine_target)

        # retarget motion
        for i in range(n_frames):
            pose_dir = targets[i]['pose_dir']

            new_frame = copy.deepcopy(bvhreader.frames[i])
            ref_frame = align_ref_frame(new_frame, pose_dir, ref_skeleton)

            retarget_motion(root_joint, targets[i], ref_skeleton, ref_frame, JOINTS_DOFS)
            out_frames.append(ref_frame)
        BVHWriter(os.path.join(output_folder, filename), ref_skeleton, out_frames, ref_skeleton.frame_time,
                  is_quaternion=False)

if __name__ == "__main__":
    # retarget_folder()
    # retarget_single_motion()
    elementary_action = 'walk'
    motion_primitives = [
                         # 'rightStance_game_engine_skeleton_smoothed',
                         # 'leftStance_game_engine_skeleton_smoothed',
                         # 'beginLeftStance_game_engine_skeleton_smoothed',
                         # 'beginRightStance_game_engine_skeleton_smoothed',
                         # 'endLeftStance_game_engine_skeleton_smoothed',
                         # 'endRightStance_game_engine_skeleton_smoothed',
                         # 'sidestepLeft_game_engine_skeleton_smoothed',
                         # 'sidestepRight_game_engine_skeleton_smoothed',
                         'turnLeftRightStance_game_engine_skeleton_smoothed',
                         'turnRightLeftStance_game_engine_skeleton_smoothed']
    # retarget_motion_primitive(elementary_action, motion_primitive)
    for motion_primitive in motion_primitives:
    #     # hand_orientation_correction(elementary_action, motion_primitive)
        retarget_motion_primitive(elementary_action, motion_primitive)
    # retarget_single_motion()