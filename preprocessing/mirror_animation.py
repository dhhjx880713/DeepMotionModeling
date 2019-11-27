import numpy as np
import os
import sys
from pathlib import Path
import copy
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from mosi_utils_anim.animation_data import BVHReader, BVHWriter, SkeletonBuilder
from mosi_utils_anim.animation_data.utils import convert_euler_frames_to_cartesian_frames
from mosi_utils_anim.animation_data.panim import Panim
from preprocessing.point_cloud_retargeting.point_cloud_IK import PointCouldIK
import glob


def point_cloud_mirroring_test():
    bvhfile = r'D:\workspace\mocap_data\mk_cmu_retargeting_default_pose\stylized_data_raw\angry_normal walking_2.bvh'
    bvhreader = BVHReader(bvhfile)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    cartesian_frames = convert_euler_frames_to_cartesian_frames(skeleton, bvhreader.frames)
    print(cartesian_frames.shape)
    # np.save(r'E:\workspace\projects\retargeting_experiments\retargeted_results\example.npy', cartesian_frames)
    ### mirroring about YOZ, all x -> -x
    mirrored_cartesian_frames = copy.deepcopy(cartesian_frames)
    mirrored_cartesian_frames[:, :, 0] = - mirrored_cartesian_frames[:, :, 0]
    np.save(r'D:\workspace\projects\retargeting_experiments\retargeted_results\example.npy', mirrored_cartesian_frames)


def point_cloud_mirror(point_cloud_data, skeleton_file, torso_plane):
    """mirror point cloud data about YOZ plane and retarget to bvh
    
    Arguments:
        point_cloud_data {numpy.ndarray} -- n_frames * n_joints * n_dims
    """
    skeleton_bvhreader = BVHReader(skeleton_file)
    skeleton = SkeletonBuilder().load_from_bvh(skeleton_bvhreader)
    skeleton_des = skeleton.generate_bone_list_description()
    panim = Panim()
    panim.setMotionData(point_cloud_data)
    panim.setSkeleton(skeleton_des)
    mirrored_data = panim.mirror()
    panim_ik_engine = PointCouldIK(skeleton, skeleton_bvhreader.frames[0], torso_plane=torso_plane, debug=False)
    output_frames = panim_ik_engine.run(mirrored_data)
    return output_frames


def mirror_bvhfile(input_file, skeleton_file, save_filename):
    input_bvhreader = BVHReader(input_file)
    skeleton = SkeletonBuilder().load_from_bvh(input_bvhreader)
    cartesian_frames = convert_euler_frames_to_cartesian_frames(skeleton, input_bvhreader.frames)
    ### for mk_cmu skeleton
    torso_plane = ['LeftUpLeg', 'LowerBack', 'RightUpLeg']
    output_frames = point_cloud_mirror(cartesian_frames, skeleton_file, torso_plane)

    BVHWriter(save_filename, skeleton, output_frames, skeleton.frame_time, is_quaternion=False)


def mirror_bvhfiles(input_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    bvhfiles = glob.glob(os.path.join(input_path, '*bvh'))
    skeleton_file = r'D:\workspace\mocap_data\skeleton_template\mk_cmu_skeleton.bvh'
    for bvhfile in bvhfiles:
        if 'mirror' not in bvhfile:
            filename = os.path.split(bvhfile)[-1]
            save_filename = filename.replace('.bvh', '_mirror.bvh')
            mirror_bvhfile(bvhfile, skeleton_file, os.path.join(save_path, save_filename))


def run_mirror_bvhfiles():
    types = ['angry', 'childlike', 'depressed', 'old', 'sexy']
    for type in types:

        input_path = os.path.join(r'D:\workspace\mocap_data\mk_cmu_retargeting_default_pose\style_locomotion', type)
        # save_path = os.path.join(input_path, "mirrored")

        mirror_bvhfiles(input_path, input_path)


def test():
    """before mirror the motion, make sure the heading direction is [0, 0, 1]. Then flipping x axis
    """
    save_folder = r'D:\workspace\projects\retargeting_experiments\retargeted_results'
    motion_data = np.load(r'D:\workspace\projects\retargeting_experiments\retargeted_results\example.npy')
    # output_data = motion_data * 10
    # np.save(r'D:\workspace\projects\retargeting_experiments\retargeted_results\example_rescale.npy', output_data)
    skeleton_file = r'D:\workspace\mocap_data\skeleton_template\mk_cmu_skeleton.bvh'
    skeleton_bvhreader = BVHReader(skeleton_file)
    skeleton = SkeletonBuilder().load_from_bvh(skeleton_bvhreader)
    torso_plane = ['LeftUpLeg', 'LowerBack', 'RightUpLeg']
    panim_ik_engine = PointCouldIK(skeleton, skeleton_bvhreader.frames[0], torso_plane=torso_plane, debug=False)
    output_frames = panim_ik_engine.run(motion_data)

    print(motion_data.shape)
    # BVHWriter(os.path.join(save_folder, 'retareget.bvh'), skeleton, output_frames, skeleton.frame_time)

if __name__ == "__main__":
    # point_cloud_mirroring_test()
    # bvhfile = r'D:\workspace\mocap_data\mk_cmu_retargeting_default_pose\style_locomotion\angry\angry_fast walking_145.bvh'
    # skeleton_file = r'D:\workspace\mocap_data\skeleton_template\mk_cmu_skeleton.bvh'
    # save_file = r'D:\tmp\tmp\mirrored.bvh'
    # mirror_bvhfile(bvhfile, skeleton_file, save_file)
    run_mirror_bvhfiles()
    # test()