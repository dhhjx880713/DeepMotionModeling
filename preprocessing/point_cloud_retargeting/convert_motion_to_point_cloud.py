import os
import numpy as np
from morphablegraphs.animation_data import BVHReader, SkeletonBuilder, MotionVector, SKELETON_MODELS
from morphablegraphs.animation_data.retargeting.analytical import create_local_cos_map_from_skeleton_axes_with_map


def normalize(v):
    return v / np.linalg.norm(v)

def load_skeleton(file_path, joint_filter=None, scale=1.0):
    target_bvh = BVHReader(file_path)
    bvh_joints = list(target_bvh.get_animated_joints())
    if joint_filter is not None:
        animated_joints = [j for j in bvh_joints if j in joint_filter]
    else:
        print("set default joints")
        animated_joints = bvh_joints
    skeleton = SkeletonBuilder().load_from_bvh(target_bvh, animated_joints, add_tool_joints=False)
    skeleton.scale(scale)
    return skeleton

def convert_frame_to_point_cloud(skeleton, frame):
    n_dims = len(skeleton.animated_joints)
    point_cloud  = np.zeros((n_dims,3))
    skeleton.clear_cached_global_matrices()
    for idx, j in enumerate(skeleton.animated_joints):
        point_cloud[idx] = skeleton.nodes[j].get_global_position(frame, use_cache=True)
    return point_cloud


def convert_frame_to_point_cloud_with_cos(skeleton, frame):
    n_dims = len(skeleton.animated_joints)
    point_cloud  = np.zeros((n_dims*2,3))
    skeleton.clear_cached_global_matrices()
    o = 0
    for j in skeleton.animated_joints:
        m = skeleton.nodes[j].get_global_matrix(frame, use_cache=True)
        rot_m = m[:3, :3]
        point_cloud[o] = m[:3,3]#np.dot(rot_m, skeleton.nodes[j].offset)
        x_axis = np.array(skeleton.skeleton_model["cos_map"][j]["x"])
        rotated_x_axis = np.dot(rot_m, x_axis)
        rotated_x_axis = normalize(rotated_x_axis)
        point_cloud[o+1] = point_cloud[o] + rotated_x_axis*5
        o += 2
    return point_cloud


def convert_motion_to_point_cloud(skeleton, frames):
    point_cloud_list = []
    for f in frames:
        pc = convert_frame_to_point_cloud(skeleton, f)
        point_cloud_list.append(pc)
    return point_cloud_list

def convert_motion_to_point_cloud_with_cos(skeleton, frames):
    point_cloud_list = []
    for f in frames:
        pc = convert_frame_to_point_cloud_with_cos(skeleton, f)
        point_cloud_list.append(pc)
    return point_cloud_list


def convert_motion_data_to_point_clouds(skeleton, input_folder, output_folder, animated_joints=None):

    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith("bvh"):
                mv = load_motion_vector_from_bvh_file(input_folder + os.sep + file_name, animated_joints)
                if skeleton.skeleton_model is not None:
                    data = convert_motion_to_point_cloud_with_cos(skeleton, mv.frames)
                else:
                    data = convert_motion_to_point_cloud(skeleton, mv.frames)
                out_filename = output_folder + os.sep + file_name[:-3] + "npy"
                np.save(out_filename, data)




def load_motion_vector_from_bvh_file(bvh_file_path, animated_joints):
    bvh_data = BVHReader(bvh_file_path)
    mv = MotionVector(None)
    print("load",animated_joints)
    mv.from_bvh_reader(bvh_data, filter_joints=False, animated_joints=animated_joints)
    return mv


DEFAULT_ANIMATED_JOINTS = ["root", "pelvis", "spine", "spine_1", "spine_2", "neck", "left_clavicle", "left_shoulder",
                           "left_elbow", "left_wrist", "right_clavicle", "right_shoulder", "right_elbow", "right_wrist",
                           "left_hip", "left_knee", "right_elbow", "right_hip", "right_knee", "left_ankle", "right_ankle"]

def main():
    default_animated_joints = DEFAULT_ANIMATED_JOINTS
    skeleton_model = "game_engine_wrong_root"
    data_folder = r"E:\projects\model_data\game_engine\rightStance"
    output_folder = r"E:\projects\model_data\game_engine\rightStance\pc"
    skeleton_file = r"E:\projects\model_data\game_engine\rightStance\walk_001_1_rightStance_429_472_mirrored_from_leftStance.bvh"
    scale = 1.0
    skeleton = load_skeleton(skeleton_file,scale=scale)
    skeleton.skeleton_model = SKELETON_MODELS[skeleton_model]
    skeleton.animated_joints = []
    for j in default_animated_joints:
        skel_j = skeleton.skeleton_model["joints"][j]
        skeleton.animated_joints.append(skel_j)
    skeleton.skeleton_model["cos_map"] = create_local_cos_map_from_skeleton_axes_with_map(skeleton)
    if "cos_map" in skeleton.skeleton_model:
        skeleton.skeleton_model["cos_map"].update(skeleton.skeleton_model["cos_map"])
    convert_motion_data_to_point_clouds(skeleton, data_folder, output_folder)




if __name__ == "__main__":
    main()