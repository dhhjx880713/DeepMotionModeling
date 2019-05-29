# encoding: UTF-8
import os
import glob
import numpy as np
import collections
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/../..')
from mosi_utils_anim.animation_data.utils import convert_euler_frames_to_cartesian_frames
from mosi_utils_anim.utilities import load_json_file, write_to_json_file
from mosi_utils_anim.animation_data import BVHReader, BVHWriter, SkeletonBuilder
from mosi_utils_anim.animation_data.skeleton_definition import Edinburgh_skeleton, \
    Edinburgh_ANIMATED_JOINTS, GAME_ENGINE_ANIMATED_JOINTS_without_game_engine
from mosi_utils_anim.animation_data.quaternion import Quaternion
to_meters = 1.0


def convert_bvh_to_panim_batch(data_folder, output_folder):
    for subdir in next(os.walk(data_folder))[1]:
        bvhfiles = glob.glob(os.path.join(data_folder, subdir, '*.bvh'))
        for bvhfile in bvhfiles:
            filename = os.path.split(bvhfile)[-1]
            bvhreader =BVHReader(bvhfile)
            skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
            cartesian_frames = convert_euler_frames_to_cartesian_frames(skeleton, bvhreader.frames,
                                                                        animated_joints=Edinburgh_ANIMATED_JOINTS)
            save_data = {}
            save_data["bones"] = Edinburgh_skeleton
            save_data["has_skeleton"] = True
            save_data["frames"] = cartesian_frames.tolist()
            output_dir = os.path.join(output_folder, subdir)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            export_filename = os.path.join(output_dir, filename[:-4] + '.panim')
            write_to_json_file(export_filename, save_data)


def convert_bvh_to_panim_data(input_file, save_file, animated_joints=None,
                              scale=1.0, unity_format=False):
    bvhreader = BVHReader(input_file)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader, animated_joints=animated_joints)
    bone_desc = skeleton.generate_bone_list_description()
    # print('bone description: ', bone_desc)
    cartesian_frames = convert_euler_frames_to_cartesian_frames(skeleton, bvhreader.frames,
                                                                animated_joints=animated_joints)
    print(cartesian_frames.shape)
    output_frames = []
    for frame in cartesian_frames:
        output_frame = []
        for point in frame:
            if unity_format:
                output_frame.append({'x': -point[0] * scale,  ## unity is left-hand coordinate system
                                     'y': point[1] * scale,
                                     'z': point[2] * scale})
            else:
                output_frame.append({'x': point[0] * scale,  ## unity is left-hand coordinate system
                                     'y': point[1] * scale,
                                     'z': point[2] * scale})
            output_frame_dic = {'WorldPos': output_frame}
        output_frames.append(output_frame_dic)

    save_data = {"skeleton": bone_desc,
                 "has_skeleton": True,
                 "motion_data": output_frames}
    write_to_json_file(save_file, save_data)



def test():
    # skeleton_bvhfile = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\stylistic_walking\sexy\sexy_normalwalking_16.bvh'
    # bvhreader = BVHReader(skeleton_bvhfile)
    # skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    # input_folder = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\stylistic_walking\sexy'
    # output_folder = r'E:\workspace\projects\variational_style_simulation\point_cloud_data'
    # convert_motions_to_point_cloud(skeleton, input_folder, output_folder)

    # skeleton_bvhfile = r'E:\workspace\unity_workspace\MG\motion_in_json\cmu_skeleton\angry_fast walking_147.bvh'
    # bvhreader = BVHReader(skeleton_bvhfile)
    # skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    # print(skeleton.animated_joints)
    # # input_folder = r'E:\workspace\unity_workspace\MG\motion_in_json\cmu_skeleton'
    # # output_folder = r'E:\workspace\unity_workspace\MG\motion_in_json\cmu_skeleton'
    # # convert_motions_to_point_cloud(skeleton, input_folder, output_folder, animated_joints=None)
    # MH_CMU_ANIMATED_JOINTS = ['Hips', 'LHipJoint', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LowerBack', 'Spine',
    #                           'Spine1', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LThumb', 'LeftFingerBase',
    #                           'LeftHandFinger1', 'Neck', 'Neck1', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm',
    #                           'RightHand', 'RThumb', 'RightFingerBase', 'RightHandFinger1', 'RHipJoint', 'RightUpLeg',
    #                           'RightLeg', 'RightFoot', 'RightToeBase']
    # print(len(MH_CMU_ANIMATED_JOINTS))

    # skeleton_bvhfile = r'E:\gits\motionsynth_code\data\processed\edin_locomotion\locomotion_jog_000_000.bvh'
    # convert_motions_to_point_cloud(skeleton, input_folder, output_folder, animated_joints=None)
    panim_data = load_json_file(r'E:\workspace\projects\retargeting_experiments\panim_from_mk_retargeting\LocomotionFlat04_000.panim')
    print(panim_data.keys())


MH_CMU_ANIMATED_JOINTS = ['Hips', 'LHipJoint', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LowerBack', 'Spine',
                          'Spine1', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LThumb', 'LeftFingerBase',
                          'LeftHandFinger1', 'Neck', 'Neck1', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm',
                          'RightHand', 'RThumb', 'RightFingerBase', 'RightHandFinger1', 'RHipJoint', 'RightUpLeg',
                          'RightLeg', 'RightFoot', 'RightToeBase']


def convert_motions_to_point_cloud(skeleton, data_folder, output_folder, animated_joints=GAME_ENGINE_ANIMATED_JOINTS_without_game_engine):
    '''
    convert bvhfile to cartesian position in the format n_frames * n_joints * 3
    :param skeleton:
    :param data_folder:
    :param output_folder:
    :return:
    '''
    bvhfiles = glob.glob(os.path.join(data_folder, '*.bvh'))
    for bvhfile in bvhfiles:
        bvhreader = BVHReader(bvhfile)
        filename = os.path.split(bvhfile)[-1]
        cartesian_frames = convert_euler_frames_to_cartesian_frames(skeleton, bvhreader.frames,
                                                                    animated_joints=animated_joints)
        np.save(os.path.join(output_folder, filename.replace('.bvh', '.npy')), cartesian_frames)


    # game engine skeleton
    # test_file = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\ulm_locomotion\Take_walk\walk_001_1.bvh'
    # save_path = r'E:\tmp\train_PFNN_ulm\Game_engine_skeleton.panim'
    # convert_bvh_to_panim_data(test_file, save_path)


def convert_motion_to_npy(bvhfile, output_folder):
    bvhreader = BVHReader(bvhfile)
    filename = os.path.split(bvhfile)[-1]
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    cartesian_frames = convert_euler_frames_to_cartesian_frames(skeleton, bvhreader.frames)
    np.save(os.path.join(output_folder, filename.replace('.bvh', '.npy')), cartesian_frames)


def load_panim_data():
    panim_data = load_json_file(r'E:\workspace\projects\variational_style_simulation\retargeted_cmu_files_test\angry_fast walking_147_mesh_retargeting.panim')
    frame_data = panim_data["frames"]
    # print(frame_data[0].keys())
    world_pos = frame_data[0]['WorldPos']
    print(len(world_pos))
    print(world_pos)
