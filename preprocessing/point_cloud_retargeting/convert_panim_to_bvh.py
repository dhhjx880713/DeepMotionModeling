import copy
import numpy as np
import glob
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/../..')
from mosi_utils_anim.utilities import load_json_file
from mosi_utils_anim.animation_data.body_plane import BodyPlane
from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder, BVHWriter
from mosi_utils_anim.animation_data.retargeting.directional_constraints_retargeting import align_ref_frame, retarget_motion


JOINTMAPPING= {
    'pelvis': 'pelvis',
    'spine_03': 'spine_03',
    'clavicle_l': 'clavicle_l',
    'upperarm_l': 'upperarm_l',
    'lowerarm_l': 'lowerarm_l',
    'hand_l': 'hand_l',
    'clavicle_r': 'clavicle_r',
    'upperarm_r': 'upperarm_r',
    'lowerarm_r': 'lowerarm_r',
    'hand_r': 'hand_r',
    'neck_01': 'neck_01',
    'head': 'head'
}


def convert_panim_to_euler_frames(panim_data, bvh_skeleton_file, skeleton_type='game_engine_skeleton'):
    '''

    :param panim_data:
    :param bvh_skeleton_file:
    :param skeleton_type:
    :return:
    '''
    target_bvhreader = BVHReader(bvh_skeleton_file)
    target_skeleton = SkeletonBuilder().load_from_bvh(target_bvhreader)
    motion_data = panim_data['motion_data']
    skeleton_data = panim_data['skeleton']
    if skeleton_type == 'game_engine_skeleton':
        body_plane_joints = ['thigh_r', 'Root', 'thigh_l']
        root_joint = 'pelvis'
        root_index = skeleton_data['pelvis']['index']
    elif skeleton_type == 'cmu_skeleton':
        body_plane_joints = ['RightUpLeg', 'Hips', 'LeftUpLeg']
        root_joint = 'Hips'
        root_index = skeleton_data['Hips']['index']
    elif skeleton_type == "Edin_skeleton":
        body_plane_joints = ['RightUpLeg', 'Hips', 'LeftUpLeg']
        root_joint = 'Hips'
        root_index = skeleton_data['Hips']['index']
    else:
        raise ValueError('unknown skeleton type')
    motion_data = np.asarray(motion_data)
    targets = create_direction_constraints_from_panim(skeleton_data, motion_data, body_plane_joints)
    n_frames = motion_data.shape[0]
    out_frames = []

    # root_index = skeleton_data['Hips']['index']
    for i in range(n_frames):
        pose_dir = targets[i]['pose_dir']
        print(i)
        if i == 0:
            new_frame = target_bvhreader.frames[0]
            ref_frame = align_ref_frame(new_frame, pose_dir, target_skeleton, body_plane_joints)
            ref_frame[0] = motion_data[0, root_index, 0]
            ref_frame[2] = motion_data[0, root_index, 2]
        else:
            # take the previous frame as initial guess
            new_frame = copy.deepcopy(out_frames[i-1])
            ref_frame = align_ref_frame(new_frame, pose_dir, target_skeleton, body_plane_joints)
            ref_frame[:3] = (motion_data[i, root_index] - motion_data[i-1, root_index]) + out_frames[i-1][:3]
        retarget_motion(root_joint, targets[i], target_skeleton, ref_frame, GAME_ENGINE_JOINTS_DOFS)
        out_frames.append(ref_frame)

    return np.asarray(out_frames)


def convert_panim_to_bvh(panim_data, bvh_skeleton_file, save_filename, 
                         body_plane_joints=['thigh_r', 'Root', 'thigh_l'], root_joint='pelvis'):
    """convert motion from panim format to bvh format 
    
    Arguments:
        panim_data {json} -- json data contrains skeleton definition and point cloud data
        bvh_skeleton_file {str} -- path to skeleton bvh file
        save_filename {[str} -- saving path
    """
    target_bvhreader = BVHReader(bvh_skeleton_file)
    target_skeleton = SkeletonBuilder().load_from_bvh(target_bvhreader)
    motion_data = np.asarray(panim_data['motion_data'])
    print(motion_data.shape)
    skeleton_data = panim_data['skeleton']
    # body_plane_joints = ['thigh_r', 'Root', 'thigh_l']
    body_plane_joints = ['RightUpLeg', 'Hips', 'LeftUpLeg']
    motion_data = np.asarray(motion_data)
    # print(skeleton_data)
    targets = create_direction_constraints_from_panim(skeleton_data, motion_data, body_plane_joints)
    n_frames = motion_data.shape[0]
    out_frames = []
    # root_joint = 'pelvis'
    root_joint = 'Hips'
    # root_index = skeleton_data['pelvis']['index']
    # root_index = skeleton_data[root_joint]['index']
    root_index = next(item['index'] for item in skeleton_data if item['name'] == root_joint)
    for i in range(n_frames):
        pose_dir = targets[i]['pose_dir']
        print(i)
        if i == 0:
            new_frame = target_bvhreader.frames[0]
            ref_frame = align_ref_frame(new_frame, pose_dir, target_skeleton, body_plane_joints)
            ref_frame[0] = motion_data[0, root_index, 0]
            ref_frame[2] = motion_data[0, root_index, 2]
        else:
            # take the previous frame as initial guess
            new_frame = copy.deepcopy(out_frames[i-1])
            ref_frame = align_ref_frame(new_frame, pose_dir, target_skeleton, body_plane_joints)
            ref_frame[:3] = (motion_data[i, root_index] - motion_data[i-1, root_index]) + out_frames[i-1][:3]
        # retarget_motion(root_joint, targets[i], target_skeleton, ref_frame, JOINTS_DOFS)
        retarget_motion(root_joint, targets[i], target_skeleton, ref_frame)
        out_frames.append(ref_frame)
    BVHWriter(save_filename, target_skeleton, out_frames, target_bvhreader.frame_time, is_quaternion=False)


def save_motion_data_to_bvh(motion_data, skeleton_data, bvh_skeleton_file, save_filename):
    '''
    
    Args:
        motion_data:
        skeleton_data:
        bvh_skeleton_file:
        save_filename:

    Returns:

    '''
    body_plane_joints = ['thigh_r', 'Root', 'thigh_l']
    target_bvhreader = BVHReader(bvh_skeleton_file)
    target_skeleton = SkeletonBuilder().load_from_bvh(target_bvhreader)
    targets = create_direction_constraints_from_panim(skeleton_data, motion_data, body_plane_joints)
    n_frames = motion_data.shape[0]
    out_frames = []
    root_joint = 'Hips'
    # root_index = skeleton_data['pelvis']['index']
    root_index = skeleton_data['Hips']['index']
    for i in range(n_frames):
        pose_dir = targets[i]['pose_dir']
        print(i)
        if i == 0:
            new_frame = target_bvhreader.frames[0]
            ## align target pose frame to pose_dir
            ref_frame = align_ref_frame(new_frame, pose_dir, target_skeleton, body_plane_joints)
            ## set reference pose position
            ref_frame[0] = motion_data[0, root_index, 0]
            ref_frame[2] = motion_data[0, root_index, 2]
        else:
            # take the previous frame as initial guess
            new_frame = copy.deepcopy(out_frames[i-1])
            ref_frame = align_ref_frame(new_frame, pose_dir, target_skeleton, body_plane_joints)
            ref_frame[:3] = (motion_data[i, root_index] - motion_data[i-1, root_index]) + out_frames[i-1][:3]
        retarget_motion(root_joint, targets[i], target_skeleton, ref_frame, GAME_ENGINE_JOINTS_DOFS)
        out_frames.append(ref_frame)
        out_frames = np.array(out_frames)
    BVHWriter(save_filename, target_skeleton, out_frames, target_bvhreader.frame_time, is_quaternion=False)


def get_joint_parent_in_joint_mapping(joint, joint_mapping, skeleton):
    """find the up level joint in the kinematic chain, search a list of dictionary
    
    Arguments:
        joint {str} -- joint name
        joint_mapping {dict} -- joint mapping from src to target
        skeleton {list} 
    """
    joint_parent = next(item['parent'] for item in skeleton if item['name'] == joint)
    
    while joint_parent is not None:
        if joint_parent in joint_mapping.keys():
            return joint_parent
        else:
            joint_parent = next(item['parent'] for item in skeleton if item['name'] == joint_parent)    
    return None



def estimate_scale_factor_panim(src_panim_skeleton, target_panim_skeleton, src_body_plane, target_body_plane, 
                                src_motion_data, ref_frame, target_skeleton, joint_mapping):
    ## create single frame directional constraints
    d_c = create_direction_constraints_panim_retargeting(src_panim_skeleton, target_panim_skeleton, src_body_plane, 
                                                         src_motion_data, joint_mapping)
    ref_frame = align_ref_frame(ref_frame, d_c[0]['pose_dir'], target_skeleton, target_body_plane)
    root_joint = 'Hips'
    retarget_motion(root_joint, d_c[0], target_skeleton, ref_frame)
    # save_filename = r'E:\workspace\projects\cGAN\single_frame.bvh'
    # BVHWriter(save_filename, target_skeleton, [ref_frame], target_skeleton.frame_time, is_quaternion=False)
    # print(src_motion_data.shape)
    n_mapping_joints = len(joint_mapping)
    src_joint_mat = np.zeros((n_mapping_joints, 3))
    target_joint_mat = np.zeros((n_mapping_joints, 3))
    i = 0
    for key, value in joint_mapping.items():
        target_joint_mat[i] = target_skeleton.nodes[key].get_global_position_from_euler(ref_frame)
        src_joint_index = next(item['index'] for item in src_panim_skeleton if item['name'] == value)
        src_joint_mat[i] = src_motion_data[0, src_joint_index]
        i += 1

    src_maximum = np.amax(src_joint_mat, axis=0)  ## get the largest number of each axis
    src_minimum = np.amin(src_joint_mat, axis=0)  ## get the smallest number of each axis
    target_maximum = np.amax(target_joint_mat, axis=0)
    target_minimum = np.amin(target_joint_mat, axis=0)
    src_max_diff = np.max(src_maximum - src_minimum)
    target_max_diff = np.max(target_maximum - target_minimum)
    scale = target_max_diff/src_max_diff
    return scale



def create_direction_constraints_panim_retargeting(src_skeleton, target_skeleton, src_body_plane, src_motion_data, JOINT_MAPPING):
    """create bone direction constraints for retargeting src skeleton to target skeleton
    
    Arguments:
        src_skeleton {list} -- a list of dictionary, each item is a joint description
        target_skeleton {list} -- a list of dictionary, each item is a joint description
        src_body_plane {list} -- a list of joints to define the torso
        src_motion_data {numpy.array3d} -- n_frames * n_joints * 3
        JOINT_MAPPING {dict} -- joint mapping from target skeleton to src skeleton
    """
    n_frames, n_joints, _ = src_motion_data.shape
    targets = []
    if isinstance(target_skeleton, list) and isinstance(src_skeleton, list):
        for i in range(n_frames):
            frame_targets = {} ## initialize direcional constraints for one frame
            ## compute pose heading vector
            if src_body_plane is not None:
                torso_points = []
                for joint in src_body_plane:
                    joint_index = next(item['index'] for item in src_skeleton if item['name'] == joint)
                    torso_points.append(src_motion_data[i, joint_index])
                torso_plane = BodyPlane(torso_points)
                dir_vec = np.array([torso_plane.normal_vector[0], torso_plane.normal_vector[2]])
                frame_targets['pose_dir'] = dir_vec /np.linalg.norm(dir_vec)
            else:
                frame_targets['pose_dir'] = None
            ## compute bone directional constraints
            for joint in JOINT_MAPPING.keys(): ## this is differenct from no JOINT_MAPPING case
                parent = get_joint_parent_in_joint_mapping(joint, JOINT_MAPPING, target_skeleton)               
                if parent is not None:
                    ## get joint index in src_skeleotn
                    parent_index = next(item['index'] for item in src_skeleton if item['name'] == JOINT_MAPPING[parent])
                    joint_index = next(item['index'] for item in src_skeleton if item['name'] == JOINT_MAPPING[joint])
                    joint_dir = src_motion_data[i, joint_index] - src_motion_data[i, parent_index]
                    assert np.linalg.norm(joint_dir) != 0  ## avoid 0 zero directional constraint, if there are zero-length bone, skip the child bone, link to grandchild bone
                    if parent in frame_targets.keys():
                        frame_targets[parent][joint] = joint_dir / np.linalg.norm(joint_dir)
                    else:
                        frame_targets[parent] = {joint: joint_dir / np.linalg.norm(joint_dir)}
            targets.append(frame_targets)
    elif isinstance(target_skeleton, dict):
        pass

    else:
        raise KeyError("Unknown data type!")     
    return targets          

def create_direction_constraints_from_panim(target_skeleton_data, motion_data, body_plane_list=None, src_skeleton_data=None, JOINT_MAPPING=None):
    """create bone direction constraints from point cloud data. If JOINT_MAPPING is given, the direcional constraints are generated from 
       the joints inside of JOINT_MAPPING. A sparse joint mapping is possible. If not, the directional constraints are generated from the 
       joints inside of skeleton_data.
    
    Arguments:
        skeleton_data {list or dict} -- a list or dictionary contains the joints' information
        motion_data {numpy.array3d} -- n_frames * n_joints * 3
    
    Keyword Arguments:
        body_plane_list {list} -- a list of joints to define the torso  (default: {None})
        JOINT_MAPPING {dict} -- joint mapping from src skeleton to target skeleton (default: {None})
    
    Returns:
        [dict] -- normalized bone direction vectors
    """
    ## find directional vector from parent joint to child joint
    n_frames, n_joints, _ = motion_data.shape
    targets = []
    if isinstance(target_skeleton_data, list):
        if JOINT_MAPPING is not None:  ## the motion_data contains the source skeleton joint positions
            for i in range(n_frames):
                frame_targets = {} ## initialize direcional constraints for one frame
                ## compute pose heading vector
                if body_plane_list is not None:
                    points = []
                    for joint in body_plane_list:
                        joint_index = next(item['index'] for item in src_skeleton_data if item['name'] == joint)
                        points.append(motion_data[i, joint_index])
                    body_plane = BodyPlane(points)
                    dir_vec = np.array([body_plane.normal_vector[0], body_plane.normal_vector[2]])
                    frame_targets['pose_dir'] = dir_vec / np.linalg.norm(dir_vec)
                else:
                    frame_targets['pose_dir'] = None
                ## compute bone directional constraints
                for joint in JOINT_MAPPING.values(): ## this is differenct from no JOINT_MAPPING case
                    parent = get_joint_parent_in_joint_mapping(joint, JOINT_MAPPING,
                                                               target_skeleton_data)                                        
                    if parent is not None:
                        ##
                        parent_index = next(item['index'] for item in target_skeleton_data if item['name'] == parent)
                        joint_index = next(item['index'] for item in target_skeleton_data if item['name'] == joint)
                        joint_dir = motion_data[i, joint_index] - motion_data[i, parent_index]
                        assert np.linalg.norm(joint_dir) != 0  ## avoid 0 zero directional constraint, if there are zero-length bone, skip the child bone, link to grandchild bone
                        if JOINT_MAPPING[parent] in frame_targets.keys():
                            frame_targets[JOINT_MAPPING[parent]][JOINT_MAPPING[joint]] = joint_dir / np.linalg.norm(joint_dir)    
                        else:
                            frame_targets[JOINT_MAPPING[parent]] = {JOINT_MAPPING[joint]: joint_dir / np.linalg.norm(joint_dir)} 
                targets.append(frame_targets)   
        else:  ## no JOINT_MAPPING given, so the src_skeletion_data is the target skeleton
            for i in range(n_frames):
                frame_targets = {}
                if body_plane_list is not None:
                    points = []
                    for joint in body_plane_list:
                        joint_index = next(item['index'] for item in target_skeleton_data if item['name'] == joint)
                        points.append(motion_data[i, joint_index])
                    body_plane = BodyPlane(points)
                    dir_vec = np.array([body_plane.normal_vector[0], body_plane.normal_vector[2]])
                    frame_targets['pose_dir'] = dir_vec / np.linalg.norm(dir_vec)
                else:
                    frame_targets['pose_dir'] = None
                for joint_des in target_skeleton_data:  ## use all joints in skeleton defintion
                    if joint_des['parent'] is not None:
                        parent_index = next(item["index"] for item in target_skeleton_data if item["name"] == joint_des['parent'])
                        joint_dir = motion_data[i, joint_des["index"], :] - motion_data[i, parent_index, :]
                        assert np.linalg.norm(joint_dir) != 0 ## avoid 0 zero directional constraint, if there are zero-length bone, skip the child bone, link to grandchild bone
                        if joint_des['parent'] in frame_targets.keys():
                            frame_targets[joint_des['parent']][joint_des['name']] = joint_dir / np.linalg.norm(joint_dir)
                        else:
                            frame_targets[joint_des['parent']] = {joint_des['name']: joint_dir / np.linalg.norm(joint_dir)}
                targets.append(frame_targets)
    elif isinstance(target_skeleton_data, dict):  ## support for deprecated panim format
        if JOINT_MAPPING is not None:
            pass
    else:
        raise KeyError("Unknown data type!")
    # n_frames = len(motion_data)
    # targets = []
    # for i in range(n_frames):
    #     frame_targets = {}  ## generate direcional constraints for one frame
    #     if isinstance(src_skeleton_data, dict):
    #         if body_plane_list is None:
    #             frame_targets['pose_dir'] = None
    #         else:
    #             points = []
    #             for joint in body_plane_list:
    #                 joint_index = src_skeleton_data[joint]['index']
    #                 points.append(motion_data[i, joint_index])
    #             body_plane = Plane(points)
    #             normal_vec = body_plane.normal_vector
    #             dir_vec = np.array([normal_vec[0], normal_vec[2]])
    #             frame_targets['pose_dir'] = dir_vec/np.linalg.norm(dir_vec)
    #         for joint, value in src_skeleton_data.items():  ## pairing parent and child
    #             if value['parent'] is not None:
    #                 parent_index = src_skeleton_data[value['parent']]['index']
    #                 joint_dir = motion_data[i, src_skeleton_data[joint]['index'], :] - motion_data[i, parent_index, :]
    #                 assert np.linalg.norm(joint_dir) != 0  ## avoid 0 zero directional constraint, if there are zero-length bone, skip the child bone, link to grandchild bone
    #                 if value['parent'] in frame_targets.keys():
    #                     frame_targets[value['parent']][joint] = joint_dir / np.linalg.norm(joint_dir)
    #                 else:
    #                     frame_targets[value['parent']] = {joint: joint_dir / np.linalg.norm(joint_dir)}
    #     elif isinstance(src_skeleton_data, list):
    #         if body_plane_list is None:
    #             frame_targets['pose_dir'] = None
    #         else:
    #             points = []
    #             for joint in body_plane_list:
    #                 joint_index = next((item["index"] for item in src_skeleton_data if item["name"] == joint), None)
    #                 assert joint_index is not None, ("skeleton mismatch!")
    #                 points.append(motion_data[i, joint_index])
    #             body_plane = Plane(points)
    #             normal_vec = body_plane.normal_vector
    #             dir_vec = np.array([normal_vec[0], normal_vec[2]])
    #             frame_targets['pose_dir'] = dir_vec/np.linalg.norm(dir_vec)          
    #         for joint in src_skeleton_data:
    #             if joint['parent'] is not None:
    #                 parent_index = next(item["index"] for item in src_skeleton_data if item["name"] == joint['parent'])
    #                 joint_dir = motion_data[i, joint["index"], :] - motion_data[i, parent_index, :]
    #                 assert np.linalg.norm(joint_dir) != 0 ## avoid 0 zero directional constraint, if there are zero-length bone, skip the child bone, link to grandchild bone
    #                 if joint['parent'] in frame_targets.keys():
    #                     frame_targets[joint['parent']][joint['name']] = joint_dir / np.linalg.norm(joint_dir)
    #                 else:
    #                     frame_targets[joint['parent']] = {joint['name']: joint_dir / np.linalg.norm(joint_dir)}
    #     else:
    #         raise KeyError("Unknown data type!")
    
    return targets


def convert_style_transfer_data_to_bvh():
    data_dir = r'E:\workspace\mocap_data\original_processed\tmp'
    bvh_skeleton_file = r'E:\workspace\experiment data\cutted_holden_data_walking\tmp\ref.bvh'
    save_dir = r'E:\workspace\mocap_data\original_processed\sexy'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    panim_files = glob.glob(os.path.join(data_dir, '*.panim'))
    for panim_file in panim_files:
        if 'sexy' in panim_file:
            filename = os.path.split(panim_file)[-1]
            print(filename)
            panim_data = load_json_file(panim_file)
            output_filename = os.path.join(save_dir, filename.replace('panim', 'bvh'))
            convert_panim_to_bvh(panim_data, bvh_skeleton_file, output_filename)



if __name__ == "__main__":
    panim_data = load_json_file(r'E:\workspace\projects\cGAN\test_ouptut\panim\reconstruction\Female1_B02_WalkToStandT2.panim')
    target_skeleton_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_skeleton.bvh'
    save_dir = r'E:\workspace\projects\cGAN\test_ouptut\panim\reconstruction\reconstructed.bvh'
    convert_panim_to_bvh(panim_data, target_skeleton_file, save_dir)