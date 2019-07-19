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
