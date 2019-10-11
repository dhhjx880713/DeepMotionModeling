import os
import sys
import glob
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder, BVHWriter
from mosi_utils_anim.animation_data.utils import pose_orientation_general, convert_euler_frame_to_cartesian_frame, transform_euler_frame, \
    get_rotation_angle
from mosi_utils_anim.animation_data.panim import Panim
from mosi_utils_anim.utilities import write_to_json_file


def generate_training_data():
    """set root joint to (0, 0, 0), align heading direction
    """
    # input_folder = r'E:\workspace\mocap_data\mk_cmu_retargeting\pfnn_data'
    input_folder = r'E:\workspace\mocap_data\mk_cmu_retargeting\stylized_data_raw'
    bvhfiles = glob.glob(os.path.join(input_folder, '*.bvh'))
    torso_joints = ['LeftUpLeg', 'LowerBack', 'RightUpLeg']
    X = []
    Y = []
    for bvhfile in bvhfiles:
        cartesian_frames, euler_frames = process_data(bvhfile, torso_joints)
        X.append(cartesian_frames)
        Y.append(euler_frames)
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    print(X.shape)
    print(Y.shape)
    np.savez_compressed(r'E:\workspace\projects\mesh_retargeting\training_data\point_cloud_plus_euler_frame\style_training_data', X=X, Y=Y)

def generate_training_data_as_json():
    input_folder = r'E:\workspace\mocap_data\mk_cmu_retargeting\stylized_data_raw'
    bvhfiles = glob.glob(os.path.join(input_folder, '*.bvh'))
    torso_joints = ['LeftUpLeg', 'LowerBack', 'RightUpLeg']
    X = []
    Y = []
    for bvhfile in bvhfiles:
        cartesian_frames, euler_frames = process_data(bvhfile, torso_joints)
        cartesian_frames = np.reshape(cartesian_frames, (cartesian_frames.shape[0], np.prod(cartesian_frames.shape[1:])))
        X.append(cartesian_frames.tolist())
        Y.append(euler_frames.tolist())   
    X_con = np.concatenate(X, axis=0) 
    Y_con = np.concatenate(Y, axis=0)
    X_mean = X_con.mean(axis=0)
    X_std = X_con.std(axis=0)
    Y_mean = Y_con.mean(axis=0)
    Y_std = Y_con.std(axis=0)

    save_data = {'X': X, 'Y': Y, 'Xmean': X_mean.tolist(), 'Xstd': X_std.tolist(), 'Ymean': Y_mean.tolist(), 'Ystd': Y_std.tolist()}
    write_to_json_file(r'E:\workspace\projects\mesh_retargeting\training_data\point_cloud_plus_euler_frame\style_training_data.json',
        save_data)

def process_data(bvhfile, torso_joints, ref_dir=np.array([0, 0, 1])):
    """
    1. make sure root position is (0, 0, 0)
    2. make sure each frame has the same heading direction
    3. after that, compute the global joint transform
    4. each transform is convert to expoential map
    
    Arguments:
        bvhfile {[type]} -- [description]
    
    Keyword Arguments:
        ref_dir {[type]} -- [description] (default: {np.array([0, 0, 1])})
    """
    bvhreader = BVHReader(bvhfile)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    ### set root to (0, 0, 0)
    X = []
    Y = []
    for euler_frame in bvhreader.frames:
        ## assume the first joint is the root joint
        global_positions = convert_euler_frame_to_cartesian_frame(skeleton, euler_frame)

        offset = -global_positions[0]

        forward = pose_orientation_general(euler_frame,
                                           torso_joints,
                                           skeleton)
        rot_angle = get_rotation_angle(ref_dir, forward)                                   
        new_frame = transform_euler_frame(euler_frame, angles=[0, rot_angle, 0], offset=offset, global_rotation=False)
        Y.append(new_frame)
        X.append(convert_euler_frame_to_cartesian_frame(skeleton, new_frame))
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y


def test():
    # test_file = r'E:\workspace\projects\cGAN\processed_data\ACCAD\Male1_bvh_Male1_A12_CrawlBackward.bvh'
    test_file = r'E:\workspace\projects\cGAN\processed_data\ACCAD\Female1_bvh_Female1_B12_WalkTurnRight90.bvh'
    bvhreader = BVHReader(test_file)
    torso_joints = ['LeftUpLeg', 'LowerBack', 'RightUpLeg']
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    cartesian_frames, euler_frames = process_data(test_file, torso_joints)
    print(cartesian_frames.shape)
    print(euler_frames.shape)
    panim = Panim()
    bone_desc = skeleton.generate_bone_list_description()
    panim.setSkeleton(bone_desc)
    panim.setMotionData(cartesian_frames)
    panim.save(r'E:\workspace\projects\mesh_retargeting\tmp\Male1_bvh_Male1_A12_CrawlBackward.panim')


if __name__ == "__main__":
    # test()
    # generate_training_data()
    generate_training_data_as_json()