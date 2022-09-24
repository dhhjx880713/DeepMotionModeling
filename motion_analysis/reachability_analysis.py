# encoding: UTF-8

import json
import matplotlib.pyplot as plt
import numpy as np
from morphablegraphs.utilities import load_json_file, get_data_analysis_folder, get_motion_primitive_path, \
                                    get_aligned_data_folder, get_semantic_motion_primitive_path
import os
import glob
from morphablegraphs.animation_data import BVHReader, BVHWriter
from morphablegraphs.motion_analysis import BVHAnalyzer
from morphablegraphs.motion_model import MotionPrimitive
from morphablegraphs.animation_data.utils import get_cartesian_coordinates_from_quaternion
from morphablegraphs.animation_data.skeleton import Skeleton
from morphablegraphs.animation_data.skeleton_builder import SkeletonBuilder
import mgrd as mgrd


def plot_pick_reachability():
    elementary_action = 'pickBoth'
    motion_primitive = 'first'
    data_file_path = get_data_analysis_folder(elementary_action,
                                              motion_primitive)
    json_data = load_json_file(os.path.join(data_file_path, 'joint_absolute_cartesian_position.json'))
    test_joints = ['LeftHand', 'RightHand']
    lefthand_last_points = np.asarray(json_data['LeftHand'])[:, -1, :]
    righthand_last_points = np.asarray(json_data['RightHand'])[:, -1, :]
    print(lefthand_last_points.shape)
    print(righthand_last_points.shape)
    fig = plt.figure()
    plt.plot(lefthand_last_points[:, 0], lefthand_last_points[:, 2], 'b.')
    plt.plot(righthand_last_points[:, 0], righthand_last_points[:, 2], 'b.')
    plt.gca().invert_yaxis()
    plt.title('XOZ plane')
    plt.show()


def plot_training_and_sample_data_2D(elementary_action,
                                     motion_primitive,
                                     target_joints,
                                     target_frameIdx,
                                     projection_plane,
                                     data_folder=None):
    aligned_data_folder = get_aligned_data_folder(elementary_action, motion_primitive)
    aligned_bvhfiles = glob.glob(os.path.join(aligned_data_folder, '*.bvh'))

    sampled_bvhfiles = glob.glob(os.path.join(data_folder, '*.bvh'))

    sampled_target_points = {}
    aligned_target_points = {}
    for joint_name in target_joints:
        aligned_target_points[joint_name] = []
        sampled_target_points[joint_name] = []
    for bvhfile in aligned_bvhfiles:
        bvhreader = BVHReader(bvhfile)
        motion = BVHAnalyzer(bvhreader)
        for joint_name in aligned_target_points:
            aligned_target_points[joint_name].append(motion.get_global_pos(joint_name, target_frameIdx))
    for bvhfile in sampled_bvhfiles:
        bvhreader = BVHReader(bvhfile)
        motion = BVHAnalyzer(bvhreader)
        for joint_name in sampled_target_points:
            sampled_target_points[joint_name].append(motion.get_global_pos(joint_name, target_frameIdx))
    for joint_name in target_joints:
        aligned_target_points[joint_name] = np.asarray(aligned_target_points[joint_name])
        sampled_target_points[joint_name] = np.asarray(sampled_target_points[joint_name])
    assert type(projection_plane) is str and len(projection_plane) == 3, ('The format of projection plane is not \
                                                                        correct')

    if projection_plane == 'XOY':
        projection_axes = (0, 1)
    elif projection_plane == 'XOZ':
        projection_axes = (0, 2)
    elif projection_plane == 'YOZ':
        projection_axes = (1, 2)
    elif projection_plane == 'YOX':
        projection_axes = (1, 0)
    elif projection_plane == 'ZOX':
        projection_axes = (2, 0)
    elif projection_plane == 'ZOY':
        projection_axes = (2, 1)
    else:
        raise KeyError('Unknown projection plane')
    fig = plt.figure()
    for joint_name in target_joints:
        plt.plot(aligned_target_points[joint_name][:, projection_axes[0]], aligned_target_points[joint_name][:, projection_axes[1]],
                 'ro')
        plt.plot(sampled_target_points[joint_name][:, projection_axes[0]],
                 sampled_target_points[joint_name][:, projection_axes[1]], 'b.')
    plt.gca().invert_yaxis()
    plt.xlabel(projection_plane[0])
    plt.ylabel(projection_plane[2])
    plt.title('_'.join([elementary_action,
                        motion_primitive,
                        projection_plane]))
    plt.show()


def plot_training_data_2D(elementary_action,
                          motion_primitive,
                          target_joints,
                          target_frameIdx,
                          projection_plane,
                          data_folder=None):
    if data_folder is None:
        aligned_data_folder = get_aligned_data_folder(elementary_action, motion_primitive)
    else:
        aligned_data_folder = data_folder
    bvhfiles = glob.glob(os.path.join(aligned_data_folder, '*.bvh'))
    target_points = {}
    for joint_name in target_joints:
        target_points[joint_name] = []
    for bvhfile in bvhfiles:
        bvhreader = BVHReader(bvhfile)
        motion = BVHAnalyzer(bvhreader)
        for joint_name in target_points:
            target_points[joint_name].append(motion.get_global_pos(joint_name, target_frameIdx))
    for joint_name in target_joints:
        target_points[joint_name] = np.asarray(target_points[joint_name])
    assert type(projection_plane) is str and len(projection_plane) == 3, ('The format of projection plane is not \
                                                                        correct')

    if projection_plane == 'XOY':
        projection_axes = (0, 1)
    elif projection_plane == 'XOZ':
        projection_axes = (0, 2)
    elif projection_plane == 'YOZ':
        projection_axes = (1, 2)
    elif projection_plane == 'YOX':
        projection_axes = (1, 0)
    elif projection_plane == 'ZOX':
        projection_axes = (2, 0)
    elif projection_plane == 'ZOY':
        projection_axes = (2, 1)
    else:
        raise KeyError('Unknown projection plane')
    fig = plt.figure()
    for joint_name in target_joints:
        plt.plot(target_points[joint_name][:, projection_axes[0]], target_points[joint_name][:, projection_axes[1]],
                 'b.')
    plt.gca().invert_yaxis()
    plt.xlabel(projection_plane[0])
    plt.ylabel(projection_plane[2])
    plt.title('_'.join([elementary_action,
                        motion_primitive,
                        projection_plane]))
    plt.show()


def test(elementary_action,
         motion_primitive,
         repo_dir):
    from morphablegraphs.motion_analysis.motion_primitive_evaluator import MotionPrimitiveEvaluator
    mm_evaluator = MotionPrimitiveEvaluator(elementary_action,
                                            motion_primitive,
                                            repo_dir)
    n_samples = 10000
    n_different_joints = 5
    n_constraints_per_joints = 5
    mm_evaluator.setup(n_samples, n_constraints_per_joints, n_different_joints)
    score = mm_evaluator.evaluate_cartesian_constraints(n_samples)
    print(score)


def motion_primitive_density_plot():
    elementary_action = 'pickBoth'
    motion_primitive = 'first'
    save_folder = r'C:\git-repo\tmp_samples\pickBoth_reach'
    N = 10000
    root_dir = r'C:\repo'
    skeleton_file = r'C:\git-repo\ulm\morphablegraphs\python_src\skeleton.bvh'
    bvhreader = BVHReader(skeleton_file)
    skeleton = Skeleton(bvhreader)
    motion_primitive_model_file = get_motion_primitive_path(root_dir,
                                                            elementary_action,
                                                            motion_primitive)
    test_model = MotionPrimitive(motion_primitive_model_file)
    samples = test_model.gaussian_mixture_model.sample(N)
    likelihood = test_model.gaussian_mixture_model.score(samples)
    left_hand = []
    right_hand = []
    count = 0
    for sample in samples:
        motion_spline = test_model.back_project(sample, use_time_parameters=False)
        frames = motion_spline.get_motion_vector()
        # left_hand.append(get_cartesian_coordinates_from_quaternion(skeleton,
        #                                                            'LeftHand',
        #                                                            frames[-1]))
        # right_hand.append(get_cartesian_coordinates_from_quaternion(skeleton,
        #                                                             'RightHand',
        #                                                             frames[-1]))
        left_pos = get_cartesian_coordinates_from_quaternion(skeleton,
                                                             'LeftHand',
                                                             frames[-1])
        right_pos = get_cartesian_coordinates_from_quaternion(skeleton,
                                                              'RightHand',
                                                              frames[-1])
        # if left_pos[2] > 20 or right_pos[2] > 20:
        #     BVHWriter(save_folder + os.sep + str(count) + '.bvh', skeleton, frames, skeleton.frame_time,
        #               is_quaternion=True)
        # count += 1

    # left_hand = np.asarray(left_hand)
    # right_hand = np.asarray(right_hand)
    # print(left_hand.shape)
    # print(right_hand.shape)
    # fig, ax = plt.subplots()
    # ax.scatter(left_hand[:, 0], left_hand[:, 2], c=likelihood, s=100, edgecolor='')
    # ax.scatter(right_hand[:, 0], right_hand[:, 2], c=likelihood, s=100, edgecolor='')
    # plt.colorbar(ax)
    # plt.show()
    # fig = plt.figure()
    # plt.scatter(left_hand[:, 0], left_hand[:, 2], c=likelihood, s=100, edgecolor='')
    # plt.scatter(right_hand[:, 0], right_hand[:, 2], c=likelihood, s=100, edgecolors='')
    # plt.colorbar()
    # plt.show()


def motion_primitive_sample_plot(elementary_action,
                                 motion_primitive,
                                 root_dir,
                                 joint_list,
                                 frame_index,
                                 projection_plane='XOZ'):
    print(elementary_action)
    print(motion_primitive)
    # save_folder = r'C:\git-repo\tmp_samples\pickBoth_reach'
    N = 1000
    bvhreader = BVHReader(r'..\..\game_engine_target_large.bvh')
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)

    motion_primitive_model_file = get_motion_primitive_path(root_dir,
                                                            elementary_action,
                                                            motion_primitive)
    print(motion_primitive_model_file)
    # test_model = MotionPrimitive(motion_primitive_model_file)
    # samples = test_model.gaussian_mixture_model.sample(N)
    # target_joint_position = {}
    # for joint in joint_list:
    #     target_joint_position[joint] = []
    #
    # for sample in samples:
    #     motion_spline = test_model.back_project(sample, use_time_parameters=False)
    #     frames = motion_spline.get_motion_vector()
    #     for joint in joint_list:
    #         target_joint_position[joint].append(get_cartesian_coordinates_from_quaternion(skeleton,
    #                                                                                       joint,
    #                                                                                       frames[frame_index]))
    # for joint in joint_list:
    #     target_joint_position[joint] = np.asarray(target_joint_position[joint])
    # if projection_plane == 'XOY':
    #     projection_axes = (0, 1)
    # elif projection_plane == 'XOZ':
    #     projection_axes = (0, 2)
    # elif projection_plane == 'YOZ':
    #     projection_axes = (1, 2)
    # elif projection_plane == 'YOX':
    #     projection_axes = (1, 0)
    # elif projection_plane == 'ZOX':
    #     projection_axes = (2, 0)
    # elif projection_plane == 'ZOY':
    #     projection_axes = (2, 1)
    # else:
    #     raise KeyError('Unknown projection plane')
    # fig = plt.figure()
    # for joint_name in joint_list:
    #     plt.plot(target_joint_position[joint_name][:, projection_axes[0]],
    #              target_joint_position[joint_name][:, projection_axes[1]],
    #              'b.')
    # plt.gca().invert_yaxis()
    # plt.xlabel(projection_plane[0])
    # plt.ylabel(projection_plane[2])
    # plt.title('_'.join([elementary_action,
    #                     motion_primitive,
    #                     projection_plane]))
    # plt.show()


def mgrd_reachability_test(elementary_action,
                           motion_primitive,
                           data_repo):
    # 1. create random constraints from randon sample
    # 2. create large amount of samples
    # 3. evaluate target joint spline (cartesian distance)
    test_model_file = get_semantic_motion_primitive_path(elementary_action,
                                                         motion_primitive,
                                                         data_repo)
    skeleton_file = r'C:\Users\hadu01\git-repos\ulm\morphablegraphs\python_src\mgrd\data\skeleton.json'
    skeletonLoader = mgrd.SkeletonJSONLoader(skeleton_file)
    skeleton = skeletonLoader.load()
    model = mgrd.MotionPrimitiveModel.load_from_file(skeleton, test_model_file)
    n_constraints = 1
    constrained_joint = 'RightHand'
    constrained_frames = [0, -1]
    constraint_samples = model.get_random_samples(n_constraints)
    # print(constraint_samples)
    cartesian_constraints = []
    joint_obj = model.skeleton.get_all_by_name([constrained_joint])
    output_folder = r'C:\experiment data\tmp'
    for sample in constraint_samples:
        # quat_spline = mgrd.create_quaternion_spline_from_sample(model, sample)
        quat_spline = model.create_spatial_spline(sample)
        time_spline = model.create_time_spline(sample)
        cartesian_spline = quat_spline.to_cartesian(joint_obj)
        for frame in constrained_frames:
            point = cartesian_spline.evaluate_cartesian_constraints(cartesian_spline.knots[frame])
        # print(point)
            cartesian_constraints.append(mgrd.CartesianConstraint(point, constrained_joint, 1.0))
        # bvhstr = mgrd.export_to_bvh_format(quat_spline, time_spline)
        # filename = os.path.join(output_folder, 'constrained_sample.bvh')
        # with open(filename, 'w') as outfile:
        #     outfile.write(bvhstr)
        # test_dist = mgrd.CartesianConstraint.score_spline(quat_spline, cartesian_constraints)
        # print(test_dist)
    n_samples = 10000
    test_samples = model.get_random_samples(n_samples)
    # print(test_samples[0])

    quat_splines = model.create_multiple_spatial_splines(test_samples)
    dists = mgrd.CartesianConstraint.score_splines(quat_splines, cartesian_constraints)
    min_dist = min(dists)
    print(min_dist)
    print("####################################")
    count = 1
    for sample in test_samples:
        quat_spline = mgrd.create_quaternion_spline_from_sample(model, sample)
        time_spline = mgrd.create_time_spline_from_sample(model, sample)
        motion_score = mgrd.score_motion_quality(model, quat_spline, time_spline)
        # print(motion_score)

        # bvhstr = mgrd.export_to_bvh_format(quat_spline, time_spline)
        # filename = os.path.join(output_folder, str(count) + '.bvh')
        # with open(filename, 'w') as outfile:
        #     outfile.write(bvhstr)
        # count += 1


def mgrd_motion_smoothness_test():
    pass





if __name__ == "__main__":
    # plot_pick_reachability()

    elementary_action = 'walk'
    motion_primitive = 'rightStance'
    repo_dir = r'C:\repo'
    # mgrd_reachability_test(elementary_action,
    #                        motion_primitive,
    #                        repo_dir)
    motion_primitive_sample_plot(elementary_action,
                                 motion_primitive,
                                 repo_dir,
                                 ['LeftFoot'],
                                 -1,
                                 'XOZ')
    data_folder = r'E:\experiment data\tmp'
    # plot_training_data_2D(elementary_action,
    #                       motion_primitive,
    #                       ['Hips'],
    #                       -1,
    #                       'XOZ',
    #                       data_folder)
    # plot_training_and_sample_data_2D(elementary_action,
    #                                  motion_primitive,
    #                                  ['Hips'],
    #                                  -1,
    #                                  'XOZ',
    #                                  data_folder)
    # motion_primitive_density_plot()
    # test(elementary_action,
    #      motion_primitive,
    #      repo_dir)