import os
import collections
import glob
import sys
from pathlib import Path
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
    motion_json = mv.to_unity_format(scale=scale)
    output_json = collections.OrderedDict()
    output_json['skeletonDesc'] = skeleton_json
    output_json['motion'] = motion_json
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
    # print(cartesian_frames.shape)
    p_frames = []
    motion_data = {"frames": p_frames}
    
    cartesian_frames *= 0.1
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


def convert_bvh_to_json_unity_folder(input_folder):
    bvhfiles = glob.glob(os.path.join(input_folder, '*.bvh'))
    for bvhfile in bvhfiles:
        convert_bvh_to_unity_json(bvhfile, scale=0.1)
    subfolders = next(os.walk(input_folder))[1]
    if subfolders != []:
        for subfolder in subfolders:
            convert_bvh_to_json_unity_folder(os.path.join(input_folder, subfolder))


