import os
import collections
import sys
from pathlib import Path
import glob
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/../..')
from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder, MotionVector
from mosi_utils_anim.utilities import write_to_json_file
from mosi_utils_anim.animation_data.utils import convert_euler_frames_to_cartesian_frames


def convert_bvh_to_json_format(bvhfile, save_dir=None):
    '''

    :param bvhfile:
    :return:
    '''
    bvhreader = BVHReader(bvhfile)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)

    json_data = skeleton.to_json()
    if save_dir is None:
        save_dir, filename = os.path.split(bvhfile)
    else:
        filename = os.path.split(bvhfile)[-1]
    write_to_json_file(os.path.join(save_dir, filename.replace('.bvh', '.json')), json_data)


def convert_bvh_to_unity_json(bvhfile, save_dir=None, scale=1.0):
    bvhreader = BVHReader(bvhfile)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    skeleton.scale(scale)
    skeleton_json = skeleton.to_unity_format()
    mv = MotionVector(skeleton)
    mv.from_bvh_reader(bvhreader)
    # mv.scale_root(0.1)
    motion_json = mv.to_unity_format(scale=scale)
    output_json = {"skeletonDesc": skeleton_json,
                   "motion": motion_json}
    if save_dir is None:
        save_dir, filename = os.path.split(bvhfile)
    else:
        filename = os.path.split(bvhfile)[-1]
    write_to_json_file(os.path.join(save_dir, filename.replace('.bvh', '.json')), output_json)


def convert_bvh_to_unity_point_cloud(bvhfile, save_dir=None):
    bvhreader = BVHReader(bvhfile)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    print(skeleton.animated_joints)
    cartesian_frames = convert_euler_frames_to_cartesian_frames(skeleton, bvhreader.frames)
    print(cartesian_frames.shape)
    p_frames = []
    motion_data = {"frames": p_frames}
    
    cartesian_frames *= 0.1
    print(cartesian_frames[0][:10])
    for frame in cartesian_frames:
        new_frame = {"points": []}
        for point in frame:
            new_frame["points"].append({'x': point[0], 'y': point[1], 'z': point[2]})
        motion_data['frames'].append(new_frame)
    if save_dir is None:
        save_dir, filename = os.path.split(bvhfile)
    else:
        filename = os.path.split(bvhfile)[-1]
    write_to_json_file(os.path.join(save_dir, filename.replace('.bvh', '_point_cloud.json')), motion_data)


def convert_to_json_unity_folder(input_folder):
    ##input_folder = r'E:\gits\vae_motion_modeling\data\training_data\style_examples\cmu_skeleton'
    bvhfiles = glob.glob(os.path.join(input_folder, '*.bvh'))
    for bvhfile in bvhfiles:
        convert_bvh_to_unity_json(bvhfile, scale=0.1)


if __name__ == "__main__":
    # test_file = r'E:\gits\PFNN\data\animations\LocomotionFlat01_000.bvh'
    # test_file = r'C:\repo\data\1 - MoCap\2.1 - GameSkeleton retargeting\ulm_locomotion\Take_walk\walk_001_1.bvh'
    test_file = r'E:\workspace\unity_workspace\MG\motion_in_json\cmu_skeleton\angry_fast walking_147.bvh'
    # test_file = r'E:\workspace\unity_workspace\MG\motion_in_json\cmu_skeleton\dual_retargeting\LocomotionFlat02_001_100.bvh'
    # test_file = r'E:\gits\PFNN\data\animations\LocomotionFlat01_000.bvh'
    # test_file = r'E:\workspace\projects\retargeting_experiments\json_animation_data_from_mk_retargeting\LocomotionFlat04_000_mesh_retargeting_new.bvh'
    # test_file = r'E:\workspace\projects\variational_style_simulation\retargeted_bvh_files_mk_cmu_skeleton\pfnn_data\LocomotionFlat04_000.bvh'
    # convert_bvh_to_json_format(test_file, save_dir=r'E:\workspace\unity_workspace\MG\motion_in_json')
    # test_file = r'E:\workspace\retargeting\makehuman_characters\fbx\cmu_skeleton\retargeted_motion.bvh'
    # test_file = r'E:\workspace\retargeting\makehuman_characters\fbx\cmu_skeleton\cmu_skeleton.bvh'
    # test_file = r'E:\gits\motionsynth_code\data\processed\edin_locomotion\locomotion_jog_000_000.bvh'
    # convert_bvh_to_unity_json(test_file, save_dir=None, scale=0.1)
    # convert_bvh_to_unity_point_cloud(test_file)
    input_folder = r'E:\workspace\tensorflow_results\stylistic_path_following\cmu_skeleton'
    convert_to_json_unity_folder(input_folder)