import numpy as np
import copy
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/../..')
from mosi_utils_anim.animation_data.quaternion import Quaternion
from mosi_utils_anim.animation_data.utils import pose_orientation_from_point_cloud, rotate_euler_frame, \
    convert_euler_frames_to_cartesian_frames, compute_average_quaternions
from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder, BVHWriter

# from morphablegraphs.construction.retargeting.convert_bvh_to_json import convert_bvh_to_unity_json
'''
Automatically generate direction constriants from joint mapping. Start from root joint, recursively traverse its 
children, if the child is inside of dictionary keys, a direction contraints is generated. In the corresponding target 
skeleton, a corresponding target direction can be derivated from their corresponding value.

'''

# EDIN_MAKEHUMAN_JOINT_MAPPING = {
#     'Hips': 'Hips',
#     # 'LHipJoint': 'LHipJoint',
#     'LeftUpLeg': 'LeftUpLeg',
#     'LeftLeg': 'LeftLeg',
#     'LeftFoot': 'LeftFoot',
#     'LeftToeBase': 'LeftToeBase',
#     # 'RHipJoint': 'RHipJoint',
#     'RightUpLeg': 'RightUpLeg',
#     'RightLeg': 'RightLeg',
#     'RightFoot': 'RightFoot',
#     'RightToeBase': 'RightToeBase',

#     # 'Neck': 'Neck',
#     # 'Neck1': 'Neck',
#     'Head': 'Head',

#     # 'LowerBack': 'LowerBack',
#     # 'Spine': 'Spine',
#     # 'Spine': 'LowerBack',  ## take care if src joint has zero bone length
#     'Spine1': 'Spine1',
#     # 'LeftShoulder': 'LeftShoulder',
#     # 'LeftArm': 'LeftShoulder',
#     # 'LeftArm': 'LeftArm',
#     'LeftForeArm': 'LeftForeArm',
#     'LeftHand': 'LeftHand',

#     # 'RightShoulder': 'RightShoulder',
#     # 'RightArm': 'RightShoulder',
#     # 'RightArm': 'RightArm',
#     'RightForeArm': 'RightForeArm',
#     'RightHand': 'RightHand',

#     # 'LeftArm': 'clavicle_l',
#     # 'LeftForeArm': 'upperarm_l',
#     # 'LeftHand': 'lowerarm_l',
#     # 'LeftHandIndex1': 'hand_l',
#     # 'RightArm': 'clavicle_r',
#     # 'RightForeArm': 'upperarm_r',
#     # 'RightHand': 'lowerarm_r',
#     # 'RightHandIndex1': 'hand_r'
# }


# def traverse_joints(joint):
#     print(joint.node_name)
#     if joint.node_name in EDIN_MAKEHUMAN_JOINT_MAPPING.keys():
#         ''' 
#         its parent must also be in the dictionary (why)
#         '''
#         parent_joint = joint.parent
#     for child in joint.children:
#         traverse_joints(child)


# def create_partial_direction_constraints_from_point_cloud(point_cloud, skeleton,
#                                                           torso_plane=['LeftUpLeg', 'LowerBack', 'RightUpLeg']):
#     body_plane_indices = [skeleton.animated_joints.index(joint) for joint in torso_plane]
#     global_direction = pose_orientation_from_point_cloud(point_cloud[0], body_plane_indices)
#     print(global_direction)
#     traverse_joints(skeleton.nodes[skeleton.root])



class PatialPointCouldIK(object):
    '''
    joint mapping defines target bone directions e.g.: (src_parent -> src_child) -> (target_parent -> target_child)
    for the case of sparse mapping: (src_grand_parent)
    '''
    def __init__(self, skeleton, ref_frame, torso_plane, debug=False):
        self.skeleton = skeleton
        self.ref_frame = ref_frame
        self.torso_plane = torso_plane
        self.debug = debug

    def rotate_pose(self, target_dir):
        return rotate_euler_frame(self.ref_frame, target_dir, self.torso_plane, self.skeleton)

    def align_bone_orientation(self, bone_dir_constraints):
        '''

        :param bone_dir_constraints:
        :return:
        '''

    def run(self, bone_dir_constraints):
        '''
        traverse the skeleton hierarchical, find constrained bones, and align them
        :param bone_dir_constraints:
        :return:
        '''
        self.rotate_pose(bone_dir_constraints['pose_dir'])



class PointCouldIK(object):

    def __init__(self, skeleton, ref_frame, torso_plane=['LeftUpLeg', 'LowerBack', 'RightUpLeg'], debug=False):
        '''

        :param skeleton:
        :param ref_frame: euler frame
        :param torso_plane: use to find forward direction
        :param debug:
        '''
        self.skeleton = skeleton
        self.ref_frame = ref_frame
        self.torso_plane = torso_plane
        self.debug = debug

    def align_single_bone(self, joint_name, child_name, frame_data, points):
        '''
        each joint rotate its only rotation, not children
        :param joint_name: string
        :param frame_data: string
        :param points: n_joints * 3
        :return:
        '''
        if joint_name == self.skeleton.root:  ## set global position
            frame_data[:3] = points[0] - self.skeleton.nodes[joint_name].offset
        global_trans = self.skeleton.nodes[joint_name].get_global_matrix_from_euler_frame(frame_data)
        global_trans_inv = np.linalg.inv(global_trans)
        global_trans_inv[:3, 3] = np.zeros(3)  ### remove translation information
        ### find the bone vector in global coordinate system
        child_index = self.skeleton.animated_joints.index(child_name)
        joint_index = self.skeleton.animated_joints.index(joint_name)
        target_child_pos = points[child_index]
        target_parent_pos = points[joint_index]
        bone_vector_global = target_child_pos - target_parent_pos
        bone_offset = self.skeleton.nodes[child_name].offset
        bone_vector_local = PointCouldIK.get_local_position(global_trans_inv, bone_vector_global)
        ## rotate bone_offset to target bone_vector_local, whether the target joint position is reached or not doesn't matter
        q = Quaternion.between(bone_offset, bone_vector_local)
        return q

    def align_multiple_bones1(self, joint_name, frame_data, points):
        """align multiple non-zero length bone using average directional vector
        
        Arguments:
            joint_name {str} -- [the parent joint name]
            frame_data {numpy.array} -- [euler frame]
            points {numpy.array} -- [n_joint * 3]
        """
        children = [child.node_name for child in self.skeleton.nodes[joint_name].children if np.linalg.norm(child.offset) > 1e-5]
        if (len(children)) == 0:
            raise ValueError("cannot handle zero-length bones!")
        else:
            n_children = len(children)
            ## get global target positions
            target_points = [points[self.skeleton.animated_joints.index(child)] for child in children]
            ## convert global target positions into local space
            global_trans = self.skeleton.nodes[joint_name].get_global_matrix_from_euler_frame(frame_data)
            global_trans_inv = np.linalg.inv(global_trans)
            local_target_points = [PointCouldIK.get_local_position(global_trans_inv, point) for point in target_points]
            ## compute average target direcion
            average_target_direction = np.zeros(3)
            for i in range(n_children):
                # average_target_direction += local_target_points[i]
                average_target_direction += local_target_points[i] / np.linalg.norm(local_target_points[i])
            # average_target_direction = average_target_direction/len(local_target_points)
            average_target_direction = average_target_direction / np.linalg.norm(average_target_direction)
            ## compute average local offset
            average_offset = np.zeros(3)
            local_offsets = [np.array(self.skeleton.nodes[child].offset) for child in children]
            for i in range(n_children):
                # average_offset += local_offsets[i]
                average_offset += local_offsets[i] / np.linalg.norm(local_offsets[i])
            # average_offset = average_offset / len(average_offset)
            average_offset = average_offset / np.linalg.norm(average_offset)
            ## compute rotation
            return Quaternion.between(average_offset, average_target_direction)

    def align_multiple_bones(self, joint_name, frame_data, points):
        '''
        handle multiple children case, each case currently is handled seperately. Todo: try to get some general solution
        :return:
        '''

        target_points = [points[self.skeleton.animated_joints.index(child.node_name)] for child in
                         self.skeleton.nodes[joint_name].children]
        ## get global transition from updated frame
        global_trans = self.skeleton.nodes[joint_name].get_global_matrix_from_euler_frame(frame_data)
        global_trans_inv = np.linalg.inv(global_trans)


        # children_points_local = [get_local_position(global_trans_inv, point) for point in children_points]
        children_points_local = [np.asarray(child.offset) for child in self.skeleton.nodes[joint_name].children]
        # print("left shoulder local position: ", children_points_local[0])
        # print("neck local position: ", children_points_local[1])
        # print("right shoulder local position: ", children_points_local[2])
        target_points_local = [PointCouldIK.get_local_position(global_trans_inv, point) for point in target_points]

        # print("left shoulder target local position: ", target_points_local[0])
        # print("neck target local position: ", target_points_local[1])
        # print("right shoulder target local position: ", target_points_local[2])

        # print("start to align multiple joints:  ")
        # print("parent joint name: ", joint_name)
        # print("parent target position: ", )
        # for child in self.skeleton.nodes[joint_name].children:
        #     print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        #     print("child name: ", child.node_name, ",  target position: ", points[self.skeleton.animated_joints.index(child.node_name)])
        if len(children_points_local) == 3:
            ref_vec1 = children_points_local[0] - children_points_local[1]
            ref_vec2 = children_points_local[0] - children_points_local[2]

            target_vec1 = target_points_local[0] - target_points_local[1]
            target_vec2 = target_points_local[0] - target_points_local[2]
            # print("ref vec1: ", ref_vec1)
            # print("ref vec2: ", ref_vec2)
            ref_orthonormal = np.cross(ref_vec1, ref_vec2)
            ref_orthonormal = ref_orthonormal / np.linalg.norm(ref_orthonormal)
            # print("target_vec1: ", target_vec1)
            # print("target_vec2: ", target_vec2)

            target_orthonormal = np.cross(target_vec1, target_vec2)
            target_orthonormal = target_orthonormal / np.linalg.norm(target_orthonormal)

            q1 = Quaternion.between(ref_orthonormal, target_orthonormal)
            new_ref_vec1 = q1 * ref_vec1
            new_ref_vec2 = q1 * ref_vec2
            # new_ref_orthonormal = q1 * ref_orthonormal
            # print("new ref orthonormal: ", new_ref_orthonormal)
            # print("target orthonormal: ", target_orthonormal)
            # if np.allclose(new_ref_orthonormal, target_orthonormal):
            #     print("norm vector is aligned")
            q2 = Quaternion.between(new_ref_vec1, target_vec1)
            q3 = Quaternion.between(new_ref_vec2, target_vec2)
            average_q = Quaternion.slerp(q2, q3, 0.5)
            # normalized_target_vec1 = target_vec1 / np.linalg.norm(target_vec1)
            # normalized_target_vec2 = target_vec2 / np.linalg.norm(target_vec2)
            # ## debuging
            # print("######################### Debug for joint: ", joint_name)
            # # tmp = q1 * ref_orthonormal
            # # print("calculated normal", tmp)
            # # print("reference normal: ", ref_orthonormal)
            # # print("target normal: ", target_orthonormal)

            # new_ref_vec1 = q2 * new_ref_vec1
            # # print("global position after rotation: ", np.dot(global_trans, ))

            # new_ref_vec1 = new_ref_vec1 / np.linalg.norm(new_ref_vec1)
            # q_rot = q2 * q1
            # ref_vec1 = q_rot * ref_vec1
            # ref_vec1 = ref_vec1 / np.linalg.norm(ref_vec1)
            # target_vec1 = target_vec1 / np.linalg.norm(target_vec1)
            # print("target ref vec1: ", normalized_target_vec1)
            # print("calcualted reference vec1: ", new_ref_vec1)
            # print("another ref vec1: ", ref_vec1)
            # if np.allclose(new_ref_vec1, normalized_target_vec1):
            #     print("Referece vector 1 is aligned.")
            # else:
            #     print("Referece vector 1 is not aligned.")
            
            # leftshoulder_local_pos = q_rot * children_points_local[0]
            # neck_local_pos = q_rot * children_points_local[1]
            # RightShoulder_local_pos = q_rot * children_points_local[2]
            # print("calculated leftshoulder_local_pos: ", leftshoulder_local_pos)
            # print("calculated neck_local_pos: ", neck_local_pos)
            # print("calculated RightShoulder_local_pos: ", RightShoulder_local_pos)

            # print("UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU")
            # print("target vector 1: ", normalized_target_vec1)
            # print("target vector 2: ", normalized_target_vec2)
            # t1 = leftshoulder_local_pos - neck_local_pos
            # t2 = leftshoulder_local_pos - RightShoulder_local_pos
            # print(t1/np.linalg.norm(t1))
            # print(t2/np.linalg.norm(t2))

            # print("UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU")
            return average_q * q1
        else:
            raise NotImplementedError
    
    def align_two_bones(self, joint_name, frame_data, points):
        """handle two children case
        
        Arguments:
            joint_name {str} -- parent joint name 
            frame_data {numpy.array} -- reference frame
            points {numpy.array} -- point cloud data
        """
        # find worldToLocal transformation
        global_trans = self.skeleton.nodes[joint_name].get_global_matrix_from_euler_frame(frame_data)
        global_trans_inv = np.linalg.inv(global_trans)
        global_trans_inv[:3, 3] = np.zeros(3)  ### remove translation information    
        
        first_bone = self.skeleton.nodes[joint_name].children[0]
        second_bone = self.skeleton.nodes[joint_name].children[1]

        joint_index = self.skeleton.animated_joints.index(joint_name)
        first_bone_index = self.skeleton.animated_joints.index(first_bone.node_name)
        second_bone_index = self.skeleton.animated_joints.index(second_bone.node_name)

        joint_pos = points[joint_index]
        first_bone_pos = points[first_bone_index]
        second_bone_pos = points[second_bone_index]

        firstTargetBone = first_bone_pos - joint_pos
        secondTargetBone = second_bone_pos - joint_pos

        firstTargetBoneLocal = PointCouldIK.get_local_position(global_trans_inv, firstTargetBone)
        secondTargetBoneLocal = PointCouldIK.get_local_position(global_trans_inv, secondTargetBone)

        averageTargetBoneLocal = (firstTargetBoneLocal + secondTargetBoneLocal) / 2
        averageBoneOffset = (np.asarray(first_bone.offset) + np.asarray(second_bone.offset)) / 2

        return Quaternion.between(averageBoneOffset, averageTargetBoneLocal)

    def align_zero_length_bones(self, joint_name, frame_data, points):
        '''
        handle zero length bones, each case is also handled seperately.
        :param joint_name:
        :param frame_data:
        :param points:
        :return:
        '''
        children = self.skeleton.nodes[joint_name].children
        global_trans = self.skeleton.nodes[joint_name].get_global_matrix_from_euler_frame(frame_data)
        global_trans_inv = np.linalg.inv(global_trans)
        rotations = []
        ## get child for each children, assume each child only has one child
        for child in children:
            assert len(child.children) == 1
            grandchild = child.children[0]
            global_pos = grandchild.get_global_position_from_euler(frame_data)
            local_pos = PointCouldIK.get_local_position(global_trans_inv, global_pos)
            target_pos = points[grandchild.index]
            target_local_pos = PointCouldIK.get_local_position(global_trans_inv, target_pos)
            q = Quaternion.between(local_pos, target_local_pos)
            rotations.append(q)
        return compute_average_quaternions(rotations)

    def align_joint_orientation(self, joint_name, frame_data, points):
        children = self.skeleton.nodes[joint_name].children
        if children != []:
            if len(children) == 1:  ## most case
                if 'EndSite' not in children[0].node_name:
                    q = self.align_single_bone(joint_name, children[0].node_name, frame_data, points)
                    local_rotation = self.skeleton.nodes[joint_name].get_local_matrix_from_euler(frame_data)
                    local_rotation_q = Quaternion.fromMat(local_rotation[:3, :3])
                    new_local_rotation = local_rotation_q * q
                    frame_data[
                        self.skeleton.nodes[joint_name].rotation_channel_indices] = new_local_rotation.toEulerAnglesDegree()
                    if self.debug:
                        child_joint_name = children[0].node_name  ## only one child
                        child_joint_position = self.skeleton.nodes[child_joint_name].get_global_position_from_euler(
                            frame_data)
                        parent_joint_position = self.skeleton.nodes[joint_name].get_global_position_from_euler(frame_data)
                        bone_dir = child_joint_position - parent_joint_position
                        bone_dir = bone_dir / np.linalg.norm(bone_dir)
                        target_child_poition = points[self.skeleton.nodes[child_joint_name].index]
                        target_parent_position = points[self.skeleton.nodes[joint_name].index]
                        target_bone_dir = target_child_poition - target_parent_position
                        target_bone_dir = target_bone_dir / np.linalg.norm(target_bone_dir)
                        if np.allclose(bone_dir, target_bone_dir):
                            print(child_joint_name + " reaches the target.")
                        else:
                            print(child_joint_name + " misses the target.")
                            print("calculated bone direction: ", bone_dir)
                            print("target bone direction: ", target_bone_dir)
                else:
                    pass
            elif len(children) == 3:
                zero_bone_number = 0
                for child in children:
                    if np.linalg.norm(child.offset) < 1e-6:
                        zero_bone_number += 1
                if self.debug:
                    print('number of zero-length bone: ', zero_bone_number)
                if zero_bone_number == len(children):
                    q = self.align_zero_length_bones(joint_name, frame_data, points)
                    local_rotation = self.skeleton.nodes[joint_name].get_local_matrix_from_euler(frame_data)
                    local_rotation_q = Quaternion.fromMat(local_rotation[:3, :3])
                    new_local_rotation = local_rotation_q * q
                    frame_data[
                        self.skeleton.nodes[joint_name].rotation_channel_indices] = new_local_rotation.toEulerAnglesDegree()
                elif zero_bone_number == 2:
                    for child in children:
                        if np.linalg.norm(child.offset) > 1e-6:  ## find and rotate non zero length bone
                            q = self.align_single_bone(joint_name, child.node_name, frame_data, points)
                            local_rotation = self.skeleton.nodes[joint_name].get_local_matrix_from_euler(frame_data)
                            local_rotation_q = Quaternion.fromMat(local_rotation[:3, :3])
                            new_local_rotation = local_rotation_q * q
                            frame_data[self.skeleton.nodes[
                                joint_name].rotation_channel_indices] = new_local_rotation.toEulerAnglesDegree()
                            if self.debug:
                                child_joint_name = child.node_name
                                child_joint_position = self.skeleton.nodes[child_joint_name].get_global_position_from_euler(
                                    frame_data)
                                parent_joint_position = self.skeleton.nodes[joint_name].get_global_position_from_euler(
                                    frame_data)
                                bone_dir = child_joint_position - parent_joint_position
                                bone_dir = bone_dir / np.linalg.norm(bone_dir)
                                target_child_poition = points[self.skeleton.nodes[child_joint_name].index]
                                target_parent_position = points[self.skeleton.nodes[joint_name].index]
                                target_bone_dir = target_child_poition - target_parent_position
                                target_bone_dir = target_bone_dir / np.linalg.norm(target_bone_dir)
                                if np.allclose(bone_dir, target_bone_dir):
                                    print(child_joint_name + " reaches the target.")
                                else:
                                    print(child_joint_name + " misses the target.")
                                    print("calculated bone direction: ", bone_dir)
                                    print("target bone direction: ", target_bone_dir)
                    '''
                    just align the joint with non-zero bone length
                    '''
                elif zero_bone_number == 0:
                    # print("the name of no zero-length joint: ", joint_name)
                    # q = self.align_multiple_bones(joint_name, frame_data, points)
                    q = self.align_multiple_bones1(joint_name, frame_data, points)
                    # q = self.align_single_bone(joint_name, 'Neck', frame_data, points)
                    local_rotation = self.skeleton.nodes[joint_name].get_local_matrix_from_euler(frame_data)

                    local_rotation_q = Quaternion.fromMat(local_rotation[:3, :3])
                    # print("calcualted rotation: ", q)
                    new_local_rotation = local_rotation_q * q
                    # new_local_rotation = q * local_rotation_q
                    frame_data[self.skeleton.nodes[
                        joint_name].rotation_channel_indices] = new_local_rotation.toEulerAnglesDegree()
                    ## check each target child direction is reached or not
                    # for child in children:
                    #     # print("@@@@@@@@@@@@@@@@@@@@  child name is: ", child.node_name)
                    #     child_joint_name = child.node_name
                    #     child_joint_position = self.skeleton.nodes[child_joint_name].get_global_position_from_euler(frame_data)
                    #     parent_joint_position = self.skeleton.nodes[joint_name].get_global_position_from_euler(frame_data)
                    #     bone_dir = child_joint_position - parent_joint_position
                    #     bone_dir = bone_dir / np.linalg.norm(bone_dir)
                    #     # print("calculated bone direction: ", bone_dir)

                    #     target_child_poition = points[self.skeleton.nodes[child_joint_name].index]
                    #     target_parent_position = points[self.skeleton.nodes[joint_name].index]
                    #     target_bone_dir = target_child_poition - target_parent_position
                    #     target_bone_dir = target_bone_dir / np.linalg.norm(target_bone_dir)

                    #     # print("target bone direction: ", target_bone_dir)
                    #     # print("target bone position is: ", target_child_poition)
                    #     # print("calculated bone position is: ", child_joint_position)
                    #     if np.allclose(bone_dir, target_bone_dir):
                    #         print(child_joint_name + " reaches the target.")
                    #     else:
                    #         print(child_joint_name + " misses the target.")
                    #         # print("calculated bone direction: ", bone_dir)
                    #         # print("target bone direction: ", target_bone_dir)
                    #     # print("**********************")

                else:
                    raise NotImplementedError
            elif len(children) == 2:
                zero_bone_number = 0
                for child in children:
                    if np.linalg.norm(child.offset) < 1e-6:
                        zero_bone_number += 1
                if self.debug:
                    print('number of zero-length bone: ', zero_bone_number)
                if zero_bone_number == len(children):
                    q = self.align_zero_length_bones(joint_name, frame_data, points)
                    local_rotation = self.skeleton.nodes[joint_name].get_local_matrix_from_euler(frame_data)
                    local_rotation_q = Quaternion.fromMat(local_rotation[:3, :3])
                    new_local_rotation = local_rotation_q * q
                    frame_data[
                        self.skeleton.nodes[joint_name].rotation_channel_indices] = new_local_rotation.toEulerAnglesDegree() 
                elif zero_bone_number == 1:
                    for child in children:
                        if np.linalg.norm(child.offset) > 1e-6:  ## find and rotate non zero length bone
                            q = self.align_single_bone(joint_name, child.node_name, frame_data, points)
                            local_rotation = self.skeleton.nodes[joint_name].get_local_matrix_from_euler(frame_data)
                            local_rotation_q = Quaternion.fromMat(local_rotation[:3, :3])
                            new_local_rotation = local_rotation_q * q
                            frame_data[self.skeleton.nodes[
                                joint_name].rotation_channel_indices] = new_local_rotation.toEulerAnglesDegree()
                            if self.debug:
                                child_joint_name = child.node_name
                                child_joint_position = self.skeleton.nodes[child_joint_name].get_global_position_from_euler(
                                    frame_data)
                                parent_joint_position = self.skeleton.nodes[joint_name].get_global_position_from_euler(
                                    frame_data)
                                bone_dir = child_joint_position - parent_joint_position
                                bone_dir = bone_dir / np.linalg.norm(bone_dir)
                                target_child_poition = points[self.skeleton.nodes[child_joint_name].index]
                                target_parent_position = points[self.skeleton.nodes[joint_name].index]
                                target_bone_dir = target_child_poition - target_parent_position
                                target_bone_dir = target_bone_dir / np.linalg.norm(target_bone_dir)
                                if np.allclose(bone_dir, target_bone_dir):
                                    print(child_joint_name + " reaches the target.")
                                else:
                                    print(child_joint_name + " misses the target.")
                                    print("calculated bone direction: ", bone_dir)
                                    print("target bone direction: ", target_bone_dir)
                elif zero_bone_number == 0:
                    q = self.align_two_bones(joint_name, frame_data, points)

                    local_rotation = self.skeleton.nodes[joint_name].get_local_matrix_from_euler(frame_data)

                    local_rotation_q = Quaternion.fromMat(local_rotation[:3, :3])
                    new_local_rotation = local_rotation_q * q
                    frame_data[self.skeleton.nodes[
                        joint_name].rotation_channel_indices] = new_local_rotation.toEulerAnglesDegree()

                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError
                # pass
            for child in children:
                self.align_joint_orientation(child.node_name, frame_data, points)
        else:
            pass

    def run(self, point_cloud):
        '''

        :param point_cloud: n_frames * n_joints * 3
        :return:
        '''
        n_frames, n_joints, _ = point_cloud.shape
        output_frames = np.zeros((n_frames, len(self.ref_frame)))
        assert len(self.ref_frame) / 3 == n_joints + 1
        for i in range(n_frames):
            new_frame = self.align_pose_orientation(point_cloud[i])  ## rotate reference euler frame, no translation
            self.align_joint_orientation(self.skeleton.root, new_frame, point_cloud[i])
            output_frames[i] = new_frame
        return output_frames

    def align_pose_orientation(self, points):
        body_plane_indices = [self.skeleton.animated_joints.index(joint) for joint in self.torso_plane]
        target_dir = pose_orientation_from_point_cloud(points, body_plane_indices)
        return rotate_euler_frame(self.ref_frame, target_dir, self.torso_plane, self.skeleton)

    @staticmethod
    def get_local_position(trans_mat, pos):
        '''
        :param trans_mat: 4 x 4 matrix
        :param pos: numpy 3 array
        :return:
        '''
        return np.dot(trans_mat, np.append(pos, 1))[:-1]

    @staticmethod
    def get_global_position(trans_mat, pos):
        return np.dot(trans_mat, np.append(pos, 1))[:-1]


def IK_example():
    from morphablegraphs.animation_data import BVHReader, SkeletonBuilder, BVHWriter
    from morphablegraphs.animation_data.utils import convert_euler_frames_to_cartesian_frames
    import os
    from morphablegraphs.construction.retargeting.convert_bvh_to_json import convert_bvh_to_unity_json
    skeleton_bvhfile = r'E:\workspace\retargeting\makehuman_characters\fbx\cmu_skeleton\cmu_skeleton.bvh'
    # data_bvhfile = r'E:\workspace\projects\variational_style_simulation\retargeted_bvh_files_mk_cmu_skeleton\pfnn_data\LocomotionFlat01_000.bvh'
    data_bvhfile = r'E:\workspace\projects\variational_style_simulation\retargeted_bvh_files_mk_cmu_skeleton\pfnn_data\LocomotionFlat04_000.bvh'
    skeleton_bvhreader = BVHReader(skeleton_bvhfile)
    data_bvhreader = BVHReader(data_bvhfile)
    skeleton = SkeletonBuilder().load_from_bvh(skeleton_bvhreader)
    body_plane = ['LeftUpLeg', 'LowerBack', 'RightUpLeg']
    point_cloud = convert_euler_frames_to_cartesian_frames(skeleton, data_bvhreader.frames)
    print(point_cloud.shape)
    # ref_frame = skeleton_bvhreader.frames[0]
    # panim_ik_engine = PointCouldIK(skeleton, ref_frame, torso_plain=body_plane, debug=False)
    # output_frames = panim_ik_engine.run(point_cloud)
    # save_folder, filename = os.path.split(data_bvhfile)
    #
    # BVHWriter(os.path.join(save_folder, filename[:-4] + '_mesh_retargeting_new.bvh'),
    #           skeleton,
    #           output_frames,
    #           skeleton.frame_time,
    #           is_quaternion=False)
    # convert_bvh_to_unity_json(os.path.join(save_folder, filename[:-4] + '_mesh_retargeting_new.bvh'), scale=0.1)



def point_cloudIK_test():
    # from morphablegraphs.utilities import write_to_json_file
    '''
    test the scalablity of IK
    Edit data has different bone size from MK_CMU skeleton
    :return:
    '''
    # data_file = r'E:\gits\PFNN\data\animations\LocomotionFlat09_000.bvh'
    data_file = r'D:\workspace\projects\retargeting_experiments\retargeted_results\LocomotionFlat01_000_short.bvh'
    dataBVHReader = BVHReader(data_file)
    dataSkeleton = SkeletonBuilder().load_from_bvh(dataBVHReader)

    # skeleton_file = r'E:\workspace\projects\variational_style_simulation\retargeted_bvh_files_mk_cmu_skeleton\pfnn_data\LocomotionFlat04_000.bvh'
    skeleton_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_T_pose.bvh'
    bvhreader = BVHReader(skeleton_file)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)

    # print("data animated joints : ")
    # print(dataSkeleton.animated_joints)

    # print("animated joints: ")
    # print(skeleton.animated_joints)
    # animated_joints = []
    # for jointname in skeleton.animated_joints:
    #     if jointname == 'LeftHandFinger1':
    #         animated_joints.append('LeftHandIndex1')
    #     elif jointname == 'RightHandFinger1':
    #         animated_joints.append('RightHandIndex1')
    #     else:
    #         animated_joints.append(jointname)
    point_cloud = convert_euler_frames_to_cartesian_frames(dataSkeleton, dataBVHReader.frames, animated_joints=skeleton.animated_joints)
    print(point_cloud.shape)

    #######################
    ## visualize point cloud data
    # skeleton_desc = skeleton.generate_bone_list_description()
    # output_frames = []
    # for frame in point_cloud:
    #     output_frame = []
    #     for point in frame:
    #         output_frame.append({'x': point[0],
    #                              'y': point[1],
    #                              'z': point[2]})
    #         output_frame_dic = {'WorldPos': output_frame}
    #     output_frames.append(output_frame_dic)

    # save_data = {"skeleton": skeleton_desc,
    #              "has_skeleton": True,
    #              "motion_data": output_frames}
    # save_file = r'E:\workspace\projects\retargeting_experiments\test_data\LocomotionFlat09_000.panim'
    # write_to_json_file(save_file, save_data)

    ref_frame = bvhreader.frames[0]
    body_plane = ['LeftUpLeg', 'LowerBack', 'RightUpLeg']
    panim_ik_engine = PointCouldIK(skeleton, ref_frame, torso_plane=body_plane, debug=True)
    output_frames = panim_ik_engine.run(point_cloud)
    save_folder, filename = os.path.split(data_file)
    save_folder = r'E:\workspace\projects\retargeting_experiments\test_data'
    BVHWriter(os.path.join(save_folder, filename[:-4] + '_mesh_retargeting_new.bvh'),
              skeleton,
              output_frames,
              skeleton.frame_time,
              is_quaternion=False)
    # convert_bvh_to_unity_json(os.path.join(save_folder, filename[:-4] + '_mesh_retargeting_new.bvh'), scale=0.1)


def IK_for_different_size():

    skeleton_bvhfile = r'E:\workspace\retargeting\makehuman_characters\fbx\cmu_skeleton\cmu_skeleton.bvh'
    # data_bvhfile = r'E:\workspace\retargeting\makehuman_characters\fbx\cmu_skeleton\cmu_skeleton_test2.bvh'
    data_bvhfile = r'E:\workspace\projects\variational_style_simulation\retargeted_from_optimization\LocomotionFlat01_000.bvh'
    skeleton_bvhreader = BVHReader(skeleton_bvhfile)
    data_bvhreader = BVHReader(data_bvhfile)
    skeleton = SkeletonBuilder().load_from_bvh(skeleton_bvhreader)
    body_plane = ['LeftUpLeg', 'LowerBack', 'RightUpLeg']
    point_cloud = convert_euler_frames_to_cartesian_frames(skeleton, data_bvhreader.frames)
    print(point_cloud.shape)

    # test_panim_file = r'E:\workspace\projects\variational_style_simulation\tmp\result_rename.panim'
    # # test_panim_file = r'E:\workspace\unity_workspace\MG\motion_in_json\cmu_skeleton\angry_fast walking_147.panim'
    # motion_data = load_json_file(test_panim_file)
    # panim_data = np.asarray(motion_data['motion_data'])
    # joint_order_dict = motion_data['skeleton']
    # index_list = [skeleton.animated_joints.index(key) for key in joint_order_dict]
    # print(index_list)
    # # point_cloud = panim_data[:, index_list, :]
    # point_cloud = point_cloud[:, index_list, :]

    ref_frame = skeleton_bvhreader.frames[0]
    # ref_frame = scale_and_align_pose(ref_frame, skeleton, point_cloud, body_plane)
    # output_frames = pointCloudIKDirection(skeleton, ref_frame, point_cloud, body_plane, debug=False)
    panim_ik_engine = PointCouldIK(skeleton, ref_frame, torso_plain=body_plane, debug=True)
    output_frames = panim_ik_engine.run(point_cloud)
    save_folder, filename = os.path.split(data_bvhfile)

    BVHWriter(os.path.join(save_folder, filename[:-4] + '_mesh_retargeting_new.bvh'),
              skeleton,
              output_frames,
              skeleton.frame_time,
              is_quaternion=False)
    convert_bvh_to_unity_json(os.path.join(save_folder, filename[:-4] + '_mesh_retargeting_new.bvh'))


def test_panim_data_from_unity():
    from morphablegraphs.utilities import load_json_file
    json_file = r'E:\tmp\tmp\test1.json'
    jsonFrames = load_json_file(json_file)
    print(jsonFrames)
    pFrames = jsonFrames['Frames']
    print(len(pFrames))
    point_cloud_data = np.zeros((len(pFrames), 31, 3))

    for i in range(len(pFrames)):
        points = pFrames[i]['Points']
        for j in range(31):
            point_cloud_data[i, j] = np.array([points[j]['x'], points[j]['y'], points[j]['z']])
    point_cloud_data *= 10
    # starting_point = np.array([point_cloud_data[0, 0, 0], 0, point_cloud_data[0, 0, 2]])
    # point_cloud_data[:, :, :] = point_cloud_data[:, :, :] - starting_point

    # np.save(r'E:\tmp\tmp\frame_data.npy', point_cloud_data)
    skeleton_bvhfile = r'E:\workspace\retargeting\makehuman_characters\fbx\cmu_skeleton\cmu_skeleton.bvh'
    skeleton_bvhreader = BVHReader(skeleton_bvhfile)
    skeleton = SkeletonBuilder().load_from_bvh(skeleton_bvhreader)
    body_plane = ['LeftUpLeg', 'LowerBack', 'RightUpLeg']
    # body_plane = ['RightUpLeg', 'LowerBack', 'LeftUpLeg']
    ref_frame = skeleton_bvhreader.frames[0]
    panim_ik_engine = PointCouldIK(skeleton, ref_frame, torso_plain=body_plane, debug=True)
    output_frames = panim_ik_engine.run(point_cloud_data)
    BVHWriter(os.path.join(json_file.replace('json', 'bvh')),
              skeleton,
              output_frames,
              skeleton.frame_time,
              is_quaternion=False)
    convert_bvh_to_unity_json(json_file.replace('json', 'bvh'))


def test():
    # test_file = r'E:\workspace\retargeting\makehuman_characters\fbx\cmu_skeleton'
    # bvhreader = BVHReader(test_file)
    # skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    # point_cloud = convert_euler_frames_to_cartesian_frames(skeleton, bvhreader.frames)
    # create_partial_direction_constraints_from_point_cloud(point_cloud, skeleton)
    # print(skeleton.nodes['LeftFoot'].parent.node_name)
    bvhfile1 = r'E:\workspace\projects\retargeting_experiments\test_data\LocomotionFlat01_000_short_mesh_retargeting_new.bvh'


if __name__ == "__main__":
    # IK_for_different_size()
    # IK_example()
    # test()
    # test_panim_data_from_unity()
    # IK_example()
    point_cloudIK_test()