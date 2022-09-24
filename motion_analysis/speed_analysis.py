# encoding: UTF-8
from morphablegraphs.utilities import get_aligned_data_folder
import glob
import os
from bvh_analyzer import BVHAnalyzer
from morphablegraphs.animation_data import BVHReader
import matplotlib.pyplot as plt
import numpy as np


def plot_root_speed(elementary_action, motion_primitive):
    aligned_data = get_aligned_data_folder(elementary_action, motion_primitive)
    bvhfiles = glob.glob(os.path.join(aligned_data, '*.bvh'))
    motion_speeds = []
    for bvhfile in bvhfiles[:10]:
        filename = os.path.split(bvhfile)[-1]
        bvhreader = BVHReader(bvhfile)
        bvh_analyzer = BVHAnalyzer(bvhreader)
        root_speed = bvh_analyzer.get_joint_speed('pelvis')
        fig = plt.figure()
        plt.plot(range(len(root_speed)), root_speed)
        plt.title(filename)
        plt.show()


def compare_root_speed():
    '''
    compare the root speed before spline representation and after spline representation
    :return:
    '''
    origin_folder = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\leftStance_game_engine_skeleton_new_grounded'
    fixed_folder = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\leftStance_game_engine_skeleton_new_grounded\spine_fixed'
    origin_bvhfiles = glob.glob(os.path.join(origin_folder, '*.bvh'))
    fixed_bvhfiles = glob.glob(os.path.join(fixed_folder, '*.bvh'))
    N = 10
    for bvhfile in origin_bvhfiles[:N]:
        filename = os.path.split(bvhfile)[-1]
        fixed_bvhfile = os.path.join(fixed_folder, filename)
        assert os.path.exists(fixed_bvhfile)
        origin_bvhreader = BVHReader(bvhfile)
        fixed_bvhreader = BVHReader(fixed_bvhfile)
        origin_bvhanalyzer = BVHAnalyzer(origin_bvhreader)
        fixed_bvhanalyzer = BVHAnalyzer(fixed_bvhreader)
        origin_root_speed = origin_bvhanalyzer.get_joint_speed('pelvis')
        fixed_root_speed = fixed_bvhanalyzer.get_joint_speed('pelvis')
        fig = plt.figure()
        plt.plot(range(len(origin_root_speed)), origin_root_speed, label='before smoothing')
        plt.plot(range(len(fixed_root_speed)), fixed_root_speed, label='after smoothing')
        plt.title(filename)
        plt.legend()
        plt.show()


def plot_multimotions_speed():
    test_folder = r'E:\processed data\ulm_data\cut_motion\walk\leftStance'
    bvhfiles = glob.glob(os.path.join(test_folder, '*.bvh'))
    speeds = []
    fig = plt.figure()
    for bvhfile in bvhfiles[:50]:
        bvhreader = BVHReader(bvhfile)
        bvh_analyzer = BVHAnalyzer(bvhreader)
        speed = bvh_analyzer.get_joint_speed('Hips')
        x = range(len(speed))
        plt.plot(x, speed)
    plt.show()


def detect_static_poses(elementary_action, motion_primitive):
    aligned_data = get_aligned_data_folder(elementary_action, motion_primitive)
    bvhfiles = glob.glob(os.path.join(aligned_data, '*.bvh'))
    bad_motions = []
    for bvhfile in bvhfiles:
        filename = os.path.split(bvhfile)[-1]
        bvhreader = BVHReader(bvhfile)
        bvh_analyzer = BVHAnalyzer(bvhreader)
        speed = bvh_analyzer.get_joint_speed('pelvis')
        counter = 0
        for i in range(len(speed)):
            if speed[i] == 0:
                counter += 1
            else:
                counter = 0
            if counter > 2:
                bad_motions.append(filename)
                break
    print(bad_motions)


def detect_static_poses_for_folder(target_folder):
    bvhfiles = glob.glob(os.path.join(target_folder, '*.bvh'))
    bad_motions = []
    for bvhfile in bvhfiles:
        filename = os.path.split(bvhfile)[-1]
        bvhreader = BVHReader(bvhfile)
        bvh_analyzer = BVHAnalyzer(bvhreader)
        speed = bvh_analyzer.get_joint_speed('pelvis')
        counter = 0
        for i in range(len(speed)):
            if speed[i] == 0:
                counter += 1
            else:
                counter = 0
            if counter > 2:
                bad_motions.append(filename)
                break
    print(bad_motions)


def plot_single_motion_speed():
    test_file = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\rightStance_game_engine_skeleton_smoothed_grounded\walk_001_2_rightStance_509_559.bvh'
    # test_file = r'C:\Users\hadu01\git-repos\motionsynth_code\data\processed\edin_locomotion\locomotion_jog_001_000.bvh'
    bvhreader = BVHReader(test_file)
    bvh_analyzer = BVHAnalyzer(bvhreader)
    speed = bvh_analyzer.get_joint_speed('pelvis')
    # speed = bvh_analyzer.get_joint_speed('Hips')
    print('average speed: ', np.average(speed))
    # fig = plt.figure()
    # x = range(len(speed))
    # plt.plot(x, speed)
    # plt.show()


if __name__ == "__main__":
    elementary_action = 'walk'
    motion_primitive = 'rightStance_game_engine_skeleton_new'
    # plot_root_speed(elementary_action, motion_primitive)
    # detect_static_poses(elementary_action, motion_primitive)
    # target_folder = r'C:\Users\hadu01\Downloads\elementary_action_walk_grounded4\beginLeftStance_game_engine_skeleton_new_grounded'
    # detect_static_poses_for_folder(target_folder)
    # compare_root_speed()
    plot_single_motion_speed()