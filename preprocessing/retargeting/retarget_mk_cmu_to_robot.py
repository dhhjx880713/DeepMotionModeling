import os
import collections
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/../..')
from mosi_utils_anim.retargeting.directional_constraints_retargeting import retarget_motion, \
    estimate_scale_factor, retarget_single_motion, create_direction_constraints, align_ref_frame, retarget_folder
from mosi_utils_anim.utilities import write_to_json_file, load_json_file
from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder, LEN_ROOT, LEN_EULER, BVHWriter
from transformations import euler_matrix, euler_from_matrix   
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter1d


### joint mapping from source to target
MAKEHUMAN_ROBOT_JOINT_MAPPTING = {
    'Hips': 'Hips',
    'Spine1': 'Chest',
    'LeftArm': 'LeftShoulder',
    'LeftForeArm': 'LeftElbow',
    'LeftHand': 'LeftHand',
    'RightArm': 'RightShoulder',
    'RightForeArm': 'RightElbow',
    'RightHand': 'RightHand',    
    'Neck': 'Neck',
    'Head': 'Head'
}


JOINTS_DOFS = {
    "LeftShoulder": ['X', 'Z'],
    "LeftElbow": ['X'],
    "RightShoulder": ['X', 'Z'],
    "RightElbow": ['X'],
}

def run_retarget_single_motion():

    source_motion_file = r'D:\workspace\my_git_repos\XAINES\output\output.bvh'
    skeleton_file = r'D:\workspace\my_git_repos\XAINES\output\robot_skeleton.bvh'
    save_dir = r'D:\workspace\my_git_repos\XAINES\output\retarget'
    root_joint = 'Hips'
    src_body_plane = ['LeftUpLeg', 'Hips', 'RightUpLeg']
    target_body_plane = ['LeftShoulder', 'RightShoulder', 'Hips']
    angle_speeds = []
    retarget_single_motion(source_motion_file, skeleton_file, save_dir, root_joint, src_body_plane, target_body_plane,
                           MAKEHUMAN_ROBOT_JOINT_MAPPTING, joints_dofs=JOINTS_DOFS, scale=False, angle_speeds=angle_speeds)
    print(len(angle_speeds))    
    write_to_json_file('angle_speeds.json', angle_speeds)
    ### smooth the data and save as numpy array

    frames = []
    for i in range(len(angle_speeds)):
        frames.append(np.ravel(list(angle_speeds[i].values())))
    frames = np.asarray(frames)
    n_dims = frames.shape[1]
    sigma = 2.5
    for i in range(n_dims):
        frames[:, i] = gaussian_filter1d(frames[:, i], sigma)
    np.save('smoothed_angle_speeds.npy', frames)
    smoothed_data = {'joints': list(angle_speeds[0].keys()), 'motion_data': frames.tolist()}
    write_to_json_file('smoothed_angle_speeds.json', smoothed_data)


def animate_motion_with_smoothed_data():
    bvhfile = r'D:\workspace\my_git_repos\XAINES\output\robot_skeleton.bvh'
    bvhreader = BVHReader(bvhfile)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    angular_speed = load_json_file('smoothed_angle_speeds.json')
    joint_list = angular_speed['joints']

    rotation_data = np.array(angular_speed['motion_data'])

    root_joint = 'Hips'

    new_frames = []
    for i in range(len(rotation_data)):
        if len(new_frames) != 0:
            new_frame = deepcopy(new_frames[-1])
        else:
            new_frame = bvhreader.frames[0]
        rotate_motion(root_joint, skeleton, new_frame, joint_list, rotation_data[i])
        new_frames.append(new_frame)

    BVHWriter('smoothed_animated_motion_with_angluar_speed.bvh', skeleton, new_frames, skeleton.frame_time)


def rotate_motion(joint_name, skeleton, frame, joint_list, rotation_data, rotation_order='rxyz'):

    if joint_name in joint_list and joint_name != "Chest" and joint_name != "Hips":
        joint_index = joint_list.index(joint_name)
        joint_index_skeleton = skeleton.nodes[joint_name].euler_frame_index
        local_rot = skeleton.nodes[joint_name].get_local_matrix_from_euler(frame)
        delta_angles = rotation_data[joint_index * LEN_EULER: (joint_index+1)*LEN_EULER]
        delta_rotmat = euler_matrix(*np.deg2rad(delta_angles), rotation_order)
        new_local_rotmat = np.dot(local_rot, delta_rotmat)
        new_euler_angles = euler_from_matrix(new_local_rotmat, axes=rotation_order)
        ### update frame value
        frame[LEN_ROOT + joint_index_skeleton * LEN_EULER : LEN_ROOT + (joint_index_skeleton + 1) * LEN_EULER] = np.rad2deg(new_euler_angles)
    
    for joint in skeleton.nodes[joint_name].children:
        rotate_motion(joint.node_name, skeleton, frame, joint_list, rotation_data, rotation_order)


def animate_motion_with_angular_speed():
    angular_speed = load_json_file('angle_speeds.json')

    keys = angular_speed[0].keys()
    smoothed_values = {}
    sigma = 5
    for key in keys:
        smoothed_values[key] = []
        for i in range(len(angular_speed)):
            smoothed_values[key].append(angular_speed[i][key])
        smoothed_values[key] = gaussian_filter1d(smoothed_values[key], sigma)
    
    #### update angular_speed data
    for i in range(len(angular_speed)):
        for key in angular_speed[i].keys():
            angular_speed[i][key] = smoothed_values[key][i]

    skeleton_file = r'D:\workspace\my_git_repos\XAINES\output\robot_skeleton.bvh'
    bvhreader = BVHReader(skeleton_file)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    root_joint = 'Hips'
    # print(skeleton.nodes['Hips'].node_name)
    # for child in skeleton.nodes['Hips'].children:
    #     print(child.node_name)
        # traverse_skeleton(skeleton.nodes['Hips'])
    new_frames = []
    for i in range(len(angular_speed)):
        if len(new_frames) != 0:
            new_frame = deepcopy(new_frames[-1])
        else:
            new_frame = bvhreader.frames[0]
        apply_speed_data(root_joint, skeleton, new_frame, angular_speed[i])
        new_frames.append(new_frame)

    BVHWriter('animated_motion_with_angluar_speed.bvh', skeleton, new_frames, skeleton.frame_time)

    write_to_json_file('angle_speeds.json', angular_speed)


def apply_speed_data(joint_name, skeleton, frame, angular_speed, rotation_order='rxyz'):
    """using angular speed data to animate skeleton

    Args:
        joint_name ([type]): [description]
        skeleton ([type]): [description]
        frame (numpy.array): bvh frame
        angular_speed (dictionary): rotation angle for each joint  
    """
    if joint_name in angular_speed.keys() and joint_name != "Chest" and joint_name != "Hips":
        euler_index = skeleton.nodes[joint_name].euler_frame_index
        local_rot = skeleton.nodes[joint_name].get_local_matrix_from_euler(frame)
        # prvious_angles = frame[euler_index * LEN_EULER + LEN_ROOT: (euler_index + 1) * LEN_EULER]
        delta_angles = angular_speed[joint_name]
        delta_rotmat = euler_matrix(*np.deg2rad(delta_angles), rotation_order)
        new_local_rotmat = np.dot(local_rot, delta_rotmat)
        new_euler_angles = euler_from_matrix(new_local_rotmat, axes=rotation_order)
        #### update frame value
        frame[euler_index * LEN_EULER + LEN_ROOT: (euler_index + 1) * LEN_EULER + LEN_ROOT] = np.rad2deg(new_euler_angles)
    for joint in skeleton.nodes[joint_name].children:

        apply_speed_data(joint.node_name, skeleton, frame, angular_speed, rotation_order=rotation_order)


def traverse_skeleton(joint):
    print(joint.node_name)
    print(joint.quaternion_frame_index)
    print(joint.euler_frame_index)
    for child in joint.children:
        traverse_skeleton(child)


def test():
    speed_data = load_json_file('angle_speeds.json')
    frames = []
    for i in range(len(speed_data)):
        frames.append(np.ravel(list(speed_data[i].values())))
    frames = np.asarray(frames)
    print(frames.shape)
    print(frames[0])


def extract_features():
    angular_velocities = load_json_file('smoothed_angle_speeds.json')


    motion_data = np.asarray(angular_velocities['motion_data'])
    n_frames = len(motion_data)
    joint_list = angular_velocities['joints']

    rightShoulder_index = joint_list.index('RightShoulder')
    target_indices = [rightShoulder_index * LEN_EULER, rightShoulder_index * LEN_EULER + 1, rightShoulder_index * LEN_EULER + 2]
    rightElbow_index = joint_list.index('RightElbow')
    target_indices.append(rightElbow_index * LEN_EULER)

    with open('motor_dofs.txt', 'w') as f:
        f.write(" ".join(['RightShoulder_Xrotation', 'RightShoulder_Yrotation', 'RightShoulder_Zrotation', 'RightElbow_Xrotation']))
        f.write("\n")
        for i in range(n_frames):
            f.write(" ".join(str(motion_data[i, index]) for index in target_indices))
            f.write("\n")

        



if __name__ == "__main__":
    # retarget_captury_data()
    # test()
    # run_retarget_single_motion()
    # animate_motion_with_angular_speed()
    extract_features()
    # animate_motion_with_smoothed_data()
