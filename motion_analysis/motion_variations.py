# encoding: UTF-8

from morphablegraphs.animation_data import BVHReader, Skeleton, BVHWriter
from morphablegraphs.motion_analysis.bvh_analyzer import BVHAnalyzer
import glob
import os
from morphablegraphs.utilities import get_aligned_data_folder, get_data_analysis_folder, load_json_file, \
                                      write_to_json_file
import numpy as np
from morphablegraphs.motion_analysis.cal_absolute_cartesian import cal_absolute_cartesian_position
from morphablegraphs.motion_analysis.cal_quat import cal_quaternion_for_all_joints, cal_quaternion_for_all_joints_from_frame_data
import matplotlib.pyplot as plt
import collections


def get_absolute_cartesian_joint_variance(aligned_bvhfiles,
                                          joint_name):
    """
    Calculate cartesion position variance for one joint from a set of bvh files
    :param aligned_bvhfiles:
    :param joint_name:
    :return:
    """
    joint_pos = []
    for filename in aligned_bvhfiles:
        bvhreader = BVHReader(filename)
        motion = BVHAnalyzer(bvhreader)
        joint_pos.append(motion.get_global_pos_for_all_frames(joint_name))
    joint_pos = np.array(joint_pos)
    var = 0
    n_frames = joint_pos.shape[1]
    for i in range(n_frames):
        cov_mat = np.cov(joint_pos[:, i, :].T)
        var += np.trace(cov_mat)
    return var/n_frames


def cal_joint_variation(value):
    """
    Calculate the variation of cartesian position of one joint over a sequence of frames
    :param value (array3D<float>): n_samples * n_frames * n_point_dimensions
    :return: scale
    """
    value = np.asarray(value)
    assert len(value.shape) == 3, ('The shape of input data is not correct!')
    joint_var = 0
    n_frames = value.shape[1]
    for i in range(n_frames):
        cov_mat = np.cov(value[:, i, :].T)
        joint_var += np.trace(cov_mat)
    return joint_var/n_frames


def cal_var_for_all_joints(joints_data):
    if joints_data is not None:
        joints_var = collections.OrderedDict()
        for key, value in joints_data.iteritems():
            joints_var[key] = cal_joint_variation(value)
    else:
        joints_var = None
    return joints_var


def get_absolute_cartesian_position_variance_for_all_joints(elementary_action,
                                                            motion_primitive,
                                                            data_repo=r'C:\repo'):
    data_analysis_folder = get_data_analysis_folder(elementary_action,
                                                    motion_primitive,
                                                    data_repo)
    if not os.path.exists(os.path.join(data_analysis_folder, 'joint_absolute_cartesian_position.json')):
        joints_pos_data = cal_absolute_cartesian_position(elementary_action,
                                                          motion_primitive)
    else:
        joints_pos_data = load_json_file(os.path.join(data_analysis_folder, 'joint_absolute_cartesian_position.json'),
                                         use_ordered_dict=True)
    joints_var = cal_var_for_all_joints(joints_pos_data)
    return joints_var


def get_relative_cartesian_position_variance_for_all_joints(elementary_action,
                                                            motion_primitive,
                                                            data_repo,
                                                            skeleton_file):
    bvhreadre = BVHReader(skeleton_file)
    motion = BVHAnalyzer(bvhreadre)
    data_analysis_folder = get_data_analysis_folder(elementary_action,
                                                    motion_primitive,
                                                    data_repo)
    if not os.path.exists(os.path.join(data_analysis_folder, 'joint_absolute_cartesian_position.json')):
        joints_pos_data = cal_absolute_cartesian_position(elementary_action,
                                                          motion_primitive)
    else:
        joints_pos_data = load_json_file(os.path.join(data_analysis_folder, 'joint_absolute_cartesian_position.json'),
                                         use_ordered_dict=True)
    relative_joint_position_var = collections.OrderedDict()
    if joints_pos_data is not None:
        for key in joints_pos_data.keys():
            if key != 'Hips':
                parent_joint = motion.get_parent_joint_name(key)

                relative_joint_position_data = cal_relative_joint_position(joints_pos_data[key],
                                                                           joints_pos_data[parent_joint])
                relative_joint_position_var[key] = cal_joint_variation(relative_joint_position_data)
            else:
                relative_joint_position_var[key] = cal_joint_variation(joints_pos_data[key])
    return relative_joint_position_var


def cal_relative_joint_position(joint_data, parent_joint_data):
    joint_data = np.asarray(joint_data)
    parent_joint_data = np.asarray(parent_joint_data)
    assert joint_data.shape == parent_joint_data.shape, ("The data should have the same shape! ")
    relative_joint_position = np.zeros(joint_data.shape)
    n_samples, n_frames, n_dims = joint_data.shape
    for i in range(n_samples):
        for j in range(n_frames):
            relative_joint_position[i, j] = joint_data[i, j] - parent_joint_data[i, j]
    return relative_joint_position


def get_relative_quaternion_variation_for_all_joints(elementary_action,
                                                     motion_primitive,
                                                     data_repo):
    data_analysis_folder = get_data_analysis_folder(elementary_action,
                                                    motion_primitive,
                                                    data_repo)
    if not os.path.exists(os.path.join(data_analysis_folder, 'joint_quaternion.json')):
        joints_quat_data = cal_quaternion_for_all_joints(elementary_action,
                                                         motion_primitive)
    else:
        joints_quat_data = load_json_file(os.path.join(data_analysis_folder, 'joint_quaternion.json'),
                                         use_ordered_dict=True)
    joints_var = collections.OrderedDict()
    if joints_quat_data is  not None:
        for key, value in joints_quat_data.iteritems():
            joints_var[key] = cal_joint_variation(value)
    else:
        joints_var = None
    return joints_var



def gen_relative_quaternion_variation_for_aligned_data():
    mocap_analysis_folder = r'C:\repo\data\1 - MoCap\7 - Mocap analysis'
    data_repo = r'C:\repo'
    relative_quaternion_var = {}
    for cur_dir, subfolders, files in os.walk(mocap_analysis_folder):
        if subfolders == []:
            path_segments = cur_dir.split(os.sep)
            motion_primitive = path_segments[-1]
            elementary_action = path_segments[-2].split('_')[-1]
            print('_'.join([elementary_action, motion_primitive]))
            relative_quaternion_var['_'.join([elementary_action, motion_primitive])] = \
            get_relative_quaternion_variation_for_all_joints(elementary_action, motion_primitive, data_repo)
    output_filename = os.path.join(mocap_analysis_folder, 'relative_quaternion_variation_for_mocap_data.json')
    write_to_json_file(output_filename, relative_quaternion_var)


def gen_absolute_cartesian_position_variation_for_aligned_data():
    mocap_analysis_folder = r'C:\repo\data\1 - MoCap\7 - Mocap analysis'
    data_repo = r'C:\repo'
    absolute_cartesian_pos_var = {}
    for cur_dir, subfolders, files in os.walk(mocap_analysis_folder):
        if subfolders == []:
            path_segments = cur_dir.split(os.sep)
            motion_primitive = path_segments[-1]
            print(motion_primitive)
            elementary_action = path_segments[-2].split('_')[-1]
            print(elementary_action)
            absolute_cartesian_pos_var['_'.join([elementary_action, motion_primitive])] = \
            get_absolute_cartesian_position_variance_for_all_joints(elementary_action, motion_primitive, data_repo)
    output_filename = os.path.join(mocap_analysis_folder, 'absolute_cartesian_position_variation_for_mocap_data.json')
    write_to_json_file(output_filename, absolute_cartesian_pos_var)


def gen_relative_cartesian_position_variation_for_aligned_data():
    mocap_analysis_folder = r'C:\repo\data\1 - MoCap\7 - Mocap analysis'
    skeleton_file = r'../../skeleton.bvh'
    data_repo = r'C:\repo'
    relative_cartesian_pos_var = {}
    for cur_dir, subfolders, files in os.walk(mocap_analysis_folder):
        if subfolders == []:
            path_segments = cur_dir.split(os.sep)
            motion_primitive = path_segments[-1]
            elementary_action = path_segments[-2].split('_')[-1]
            print(elementary_action)
            print(motion_primitive)
            relative_cartesian_pos_var['_'.join([elementary_action, motion_primitive])] = \
            get_relative_cartesian_position_variance_for_all_joints(elementary_action, motion_primitive,
                                                                    data_repo, skeleton_file)
    output_filename = os.path.join(mocap_analysis_folder, 'relative_cartesian_position_variation_for_mocap_data.json')
    write_to_json_file(output_filename, relative_cartesian_pos_var)


def plot_absolute_cartesian_pos_variation(elementary_action,
                                          motion_primitive):
    mocap_analysis_folder = r'C:\repo\data\1 - MoCap\7 - Mocap analysis'
    joint_vars_all = load_json_file(os.path.join(mocap_analysis_folder,
                                                 'absolute_cartesian_position_variation_for_mocap_data.json'),
                                    use_ordered_dict=True)
    joint_vars = joint_vars_all['_'.join([elementary_action,
                                          motion_primitive])]
    fig, ax = plt.subplots()
    ind = np.arange(len(joint_vars.keys()))
    width = 0.35

    rects = ax.bar(ind, joint_vars.values(), width)
    ax.set_xticks(ind)
    ax.set_xticklabels(joint_vars.keys(), rotation='vertical')
    plt.title('_'.join((elementary_action, motion_primitive)))
    fig.set_size_inches(18.5, 16.5)
    plt.show()
    # fig.savefig(os.path.join(mocap_analysis_folder, 'locomotion_synthesis_test.png'))
    # plt.close()

def plot_absolute_cartesian_pos_variance_for_reduced_joints(elementary_action,
                                                            motion_primitive):
    mocap_analysis_folder = r'C:\repo\data\1 - MoCap\7 - Mocap analysis'
    joint_vars_all = load_json_file(os.path.join(mocap_analysis_folder,
                                                 'absolute_cartesian_position_variation_for_mocap_data.json'),
                                    use_ordered_dict=True)
    joint_vars = joint_vars_all['_'.join([elementary_action,
                                          motion_primitive])]
    joint_names = []
    values = []
    for joint_name in joint_vars.keys():
        if 'Bip' not in joint_name and 'EndSite' not in joint_name:
            joint_names.append(joint_name)
            values.append(joint_vars[joint_name]/100)
    filename = r'C:\repo\data\1 - MoCap\7 - Mocap analysis\elementary_action_pickBoth\first\absolute cartesian position joint variance.png'
    fig, ax = plt.subplots()
    ind = np.arange(len(joint_names))
    width = 0.5

    rects = ax.bar(ind, values, width)
    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(joint_names, rotation='vertical', fontsize=15)
    plt.title('pick up', fontsize=20)
    plt.ylabel('Joint Variance', fontsize=20)
    fig.set_size_inches(18.5, 16.5)
    fig.patch.set_facecolor('white')
    fig.savefig(os.path.join(filename))
    plt.close()
    # plt.show()


def save_variation_plot(variation_data, filename):
    fig, ax = plt.subplots()
    ind = np.arange(len(variation_data.keys()))
    width = 0.5

    rects = ax.bar(ind, variation_data.values(), width)
    ax.set_xticks(ind)
    ax.set_xticklabels(variation_data.keys(), rotation='vertical')
    fig.set_size_inches(18.5, 16.5)
    fig.patch.set_facecolor('white')
    fig.savefig(os.path.join(filename))
    plt.close()



def exist_precomputed_variation(mocap_analysis_folder):
    return os.path.exists(os.path.join(mocap_analysis_folder,
                                       'absolute_cartesian_position_variation_for_mocap_data.json'))


def exist_precomputed_cartesian_joint_position(mocap_analysis_folder,
                                               elementary_action,
                                               motion_primitive):
    filepath = os.path.join(mocap_analysis_folder,
                            'elementary_action_' + elementary_action,
                            motion_primitive,
                            'joint_absolute_cartesian_position.json')
    return os.path.exists(filepath)


def plot_absolute_cartesian_variation_for_aligned_data():
    aligned_data_folder = r'C:\repo\data\1 - MoCap\4 - Alignment'
    mocap_analysis_folder = r'C:\repo\data\1 - MoCap\7 - Mocap analysis'
    if exist_precomputed_variation(mocap_analysis_folder):
        use_precomputed_data = True
        precomputed_variation = load_json_file(os.path.join(mocap_analysis_folder,
                                                            'absolute_cartesian_position_variation_for_mocap_data.json'),
                                               use_ordered_dict=True)
    else:
        use_precomputed_data = False
    for subdir in os.walk(mocap_analysis_folder).next()[1]:
        for motion_primitive_folder in os.walk(os.path.join(mocap_analysis_folder,
                                                            subdir)).next()[1]:
            elementary_action = subdir.split('_')[-1]
            if use_precomputed_data:
                if '_'.join([elementary_action, motion_primitive_folder]) in precomputed_variation.keys():
                    motion_primitive_var_data = precomputed_variation['_'.join([elementary_action,
                                                                                motion_primitive_folder])]

                    if motion_primitive_var_data is not None:
                        save_variation_plot(motion_primitive_var_data,
                                            os.path.join(mocap_analysis_folder,
                                                         subdir,
                                                         motion_primitive_folder,
                                                         'absolute_joint_position_variation.png'))
            elif os.path.exists(os.path.join(mocap_analysis_folder,
                                             subdir,
                                             motion_primitive_folder,
                                             'joint_absolute_cartesian_position.json')):
                motion_primitive_joint_pos_data = load_json_file(os.path.join(mocap_analysis_folder,
                                                                              subdir,
                                                                              motion_primitive_folder,
                                                                              'joint_absolute_cartesian_position.json'))
                if motion_primitive_joint_pos_data is not None:
                    motion_primitive_var_data = {}
                    for key, value in motion_primitive_joint_pos_data.iteritems():
                        motion_primitive_var_data[key] = cal_joint_variation(value)
                    save_variation_plot(motion_primitive_var_data,
                                        os.path.join(mocap_analysis_folder,
                                                     subdir,
                                                     motion_primitive_folder,
                                                     'absolute_joint_position_variation.png'))
            else:
                raise NotImplementedError


def plot_relative_cartesian_variation_for_aligned_data():
    aligned_data_folder = r'C:\repo\data\1 - MoCap\4 - Alignment'
    mocap_analysis_folder = r'C:\repo\data\1 - MoCap\7 - Mocap analysis'
    precomputed_variantion_file = os.path.join(mocap_analysis_folder,
                                               'relative_cartesian_position_variation_for_mocap_data.json')
    if os.path.exists(precomputed_variantion_file):
        use_precomputed_data = True
        precomputed_variation = load_json_file(precomputed_variantion_file)
    else:
        use_precomputed_data = False
    for subdir in os.walk(mocap_analysis_folder).next()[1]:

        for motion_primitive_folder in os.walk(os.path.join(mocap_analysis_folder,
                                                            subdir)).next()[1]:
            elementary_action = subdir.split('_')[-1]
            if use_precomputed_data:

                print('_'.join([elementary_action, motion_primitive_folder]))
                if '_'.join([elementary_action, motion_primitive_folder]) in precomputed_variation.keys():
                    motion_primitive_var_data = precomputed_variation['_'.join([elementary_action,
                                                                                motion_primitive_folder])]

                    if motion_primitive_var_data is not None:
                        save_variation_plot(motion_primitive_var_data,
                                            os.path.join(mocap_analysis_folder,
                                                         subdir,
                                                         motion_primitive_folder,
                                                         'relative_joint_position_variation.png'))


def plot_relative_quaternion_variation_for_aligned_data():
    aligned_data_folder = r'C:\repo\data\1 - MoCap\4 - Alignment'
    mocap_analysis_folder = r'C:\repo\data\1 - MoCap\7 - Mocap analysis'
    precomputed_variantion_file = os.path.join(mocap_analysis_folder,
                                               'relative_quaternion_variation_for_mocap_data.json')
    if os.path.exists(precomputed_variantion_file):
        use_precomputed_data = True
        precomputed_variation = load_json_file(precomputed_variantion_file)
    else:
        use_precomputed_data = False
    for subdir in os.walk(mocap_analysis_folder).next()[1]:

        for motion_primitive_folder in os.walk(os.path.join(mocap_analysis_folder,
                                                            subdir)).next()[1]:
            elementary_action = subdir.split('_')[-1]
            if use_precomputed_data:

                print('_'.join([elementary_action, motion_primitive_folder]))
                if '_'.join([elementary_action, motion_primitive_folder]) in precomputed_variation.keys():
                    motion_primitive_var_data = precomputed_variation['_'.join([elementary_action,
                                                                                motion_primitive_folder])]

                    if motion_primitive_var_data is not None:
                        print("save plot")
                        save_variation_plot(motion_primitive_var_data,
                                            os.path.join(mocap_analysis_folder,
                                                         subdir,
                                                         motion_primitive_folder,
                                                         'relative_quaternion_variation.png'))

def test():
    data_repo = r'C:\repo'
    joint_quat_vars = get_relative_quaternion_variation_for_all_joints('pickBoth',
                                                                       'first',
                                                                       data_repo)
    joint_cartesian_vars = get_absolute_cartesian_position_variance_for_all_joints('pickBoth',
                                                                                   'first',
                                                                                   data_repo)
    # print(joint_quat_vars)
    # labels = ['Hips\ntranslation'] + joint_quat_vars.keys()
    labels = joint_cartesian_vars.keys()
    # print(labels)
    # print(len(labels))
    # values = [joint_cartesian_vars['Hips']/100] + joint_quat_vars.values()
    values = np.asarray(joint_cartesian_vars.values())/100
    # print(values)
    # print(len(values))
    # print(joint_cartesian_vars)
    # filename = r'C:\repo\data\1 - MoCap\7 - Mocap analysis\elementary_action_pickBoth\first\relative quaternion joint variance.png'
    filename = r'C:\repo\data\1 - MoCap\7 - Mocap analysis\elementary_action_pickBoth\first\absolute cartesian position joint variance.png'
    fig, ax = plt.subplots()
    ind = np.arange(len(labels))
    width = 0.5

    rects = ax.bar(ind, values, width)
    ax.set_xticks(ind+width/2.0)
    ax.set_xticklabels(labels, rotation='vertical', fontsize=20)
    plt.title('pick up', fontsize=30)
    plt.ylabel('Joint Variance', fontsize=30)
    fig.set_size_inches(20, 25)
    fig.patch.set_facecolor('white')
    # fig.savefig(os.path.join(filename))
    # plt.close()
    plt.show()


def get_variance_for_each_dimension_scaled_motion_data():
    elementary_action = 'pickBoth'
    motion_primitive = 'first'
    data_analysis_folder = get_data_analysis_folder(elementary_action,
                                                    motion_primitive)
    scaled_quat_frames_dic = load_json_file(os.path.join(data_analysis_folder, 'scaled_smoothed_quat_frames.json'))
    scale_vector = scaled_quat_frames_dic['scale_vector']
    scaled_quat_frames_data = scaled_quat_frames_dic['data']
    scaled_quat_frame_data_mat = np.asarray(scaled_quat_frames_data.values())
    print(scaled_quat_frame_data_mat.shape)
    n_samples, n_frames, n_dims = scaled_quat_frame_data_mat.shape
    frame_variance = np.zeros(n_frames)


    for i in range(n_frames):
        frame_variance[i] = np.trace(np.cov(scaled_quat_frame_data_mat[:, i, :]))/np.float(n_dims)
    print(np.cumsum(frame_variance)/np.sum(frame_variance))
    # fig = plt.figure()
    # plt.plot(range(n_frames), frame_variance)
    # plt.show()


def plot_variance_position_space_vs_quaternion_space(elementary_action,
                                                     motion_primitive):
    data_analysis_folder = r'C:\repo\data\1 - MoCap\7 - Mocap analysis'
    relative_cartesian_pos_data = load_json_file(os.path.join(data_analysis_folder,
                                                              'relative_cartesian_position_variation_for_mocap_data.json'))
    relative_quaternion_data = load_json_file(os.path.join(data_analysis_folder,
                                                           'relative_quaternion_variation_for_mocap_data.json'))
    relative_cartesian_var_dic = relative_cartesian_pos_data['_'.join([elementary_action,
                                                                       motion_primitive])]
    relative_quat_var_dic = relative_quaternion_data['_'.join([elementary_action,
                                                               motion_primitive])]
    joint_namelist = []
    relative_quat_value = []
    relative_cartesian_value = []
    for key, value in relative_quat_var_dic.iteritems():
        joint_namelist.append(key)
        relative_quat_value.append(value)
        relative_cartesian_value.append(relative_cartesian_var_dic[key])
    normalized_quat_value = np.asarray(relative_quat_value)/np.max(relative_quat_value)
    normalized_cartesian_value = np.asarray(relative_cartesian_value)/np.max(relative_cartesian_value)
    print(normalized_quat_value)
    print(normalized_cartesian_value)
    # fig = plt.figure()
    # plt.plot(normalized_cartesian_value, normalized_quat_value, 'bo')
    # plt.xlabel('Variance ratio of joint position')
    # plt.ylabel('Variance ratio of joint orientation')
    # plt.show()
    fig, ax = plt.subplots()

    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    for i, txt in enumerate(joint_namelist):
        ax.annotate(txt, (normalized_cartesian_value[i],normalized_quat_value[i]), fontsize=15)
    ax.scatter(normalized_cartesian_value, normalized_quat_value)
    ax.set_xlabel('Variance ratio of joint position', fontsize=15)
    ax.set_ylabel('Variance ratio of joint orientation', fontsize=15)
    ax.set_title('Right-hand picking', fontsize=20)
    plt.show()
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # width = 0.5
    # ind = np.arange(len(joint_namelist))
    # rects = ax1.bar(ind, normalized_quat_value, width)
    # ax1.set_xticks(ind)
    # ax1.set_xticklabels(joint_namelist, rotation='vertical')
    # ax1.set_title('Variance of joint orientation')
    # ax1.set_ylabel('Variance ratio')
    # ax1.set_xlabel('( a )', fontsize=20)
    # ax2.bar(ind, normalized_cartesian_value, width)
    # ax2.set_xticks(ind)
    # ax2.set_xticklabels(joint_namelist, rotation='vertical')
    # ax2.set_title('Variance of joint position')
    # ax2.set_ylabel('Variance ratio')
    # ax2.set_xlabel('( b )', fontsize=20)
    # plt.subplots_adjust(bottom=0.25)
    # plt.suptitle('Two-hand picking', fontsize=20)
    # plt.show()

def get_game_engine_skeleton_pose_dir(euler_frame, skeleton):
    game_eigine_pos = skeleton.nodes['Game_engine'].get_global_position_from_euler_frame(euler_frame)
    root_pos = skeleton.nodes['Root'].get_global_position_from_euler_frame(euler_frame)
    # print('game engine: ', game_eigine_pos)
    # print('root: ', root_pos)
    dir = game_eigine_pos - root_pos
    dir_2d = np.array([dir[0], dir[2]])
    return dir_2d/np.linalg.norm(dir_2d)


def walking_variation_test(elementary_action, motion_primitive, skeleton_file):
    from morphablegraphs.animation_data import MotionVector
    from morphablegraphs.motion_model.motion_primitive_wrapper import MotionPrimitiveModelWrapper
    from morphablegraphs.utilities import get_semantic_motion_primitive_path
    from morphablegraphs.animation_data.motion_editing import pose_orientation_general
    N = 300
    repo_path = r'C:\repo'
    test_model_file = get_semantic_motion_primitive_path(elementary_action,
                                                         motion_primitive,
                                                         repo_path)
    mm = MotionPrimitiveModelWrapper()
    model_data = load_json_file(test_model_file)
    bvhreader = BVHReader(skeleton_file)
    animated_joints = model_data['sspm']['animated_joints']
    bvh_skeleton = Skeleton()
    # animated_joints = ["Hips", "Spine", "Spine_1", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    #                    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg", "LeftFoot",
    #                    "RightUpLeg", "RightLeg", "RightFoot"]
    bvh_skeleton.load_from_bvh(bvhreader, animated_joints)

    skeleton = bvh_skeleton.convert_to_mgrd_skeleton()
    mm._load_from_file(skeleton, test_model_file, animated_joints=bvh_skeleton.animated_joints)
    target_dir = [1, 0]
    errs = []
    motions = []
    for i in range(N):
        # print(i)
        quat_spline = mm.sample()
        quat_frames = quat_spline.get_motion_vector()
        # print(quat_frames[0])
        mv = MotionVector(bvh_skeleton)
        mv.set_frames(quat_frames)
        euler_frames = mv.get_complete_euler_frame()
        motions.append(euler_frames)
        pose_dir = get_game_engine_skeleton_pose_dir(euler_frames[-1], bvh_skeleton)
        err = np.linalg.norm(pose_dir - target_dir)
        # print(err)
        errs.append(err)
    min_index = min(xrange(len(errs)), key=errs.__getitem__)
    BVHWriter(r'E:\tmp\search_result0.bvh', bvh_skeleton, motions[min_index], bvh_skeleton.frame_time,
              is_quaternion=False)


if __name__ == '__main__':
    elementary_action = 'walk'
    motion_primitive = 'leftStance_game_engine_skeleton_new'
    skeleton_file = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\leftStance_game_engine_skeleton_new\normalized_on_ground\walk_001_3_leftStance_398_446.bvh'
    # plot_variance_position_space_vs_quaternion_space(elementary_action,
    #                                                  motion_primitive)
    # locomotion_synthesis_test()
    # plot_absolute_cartesian_pos_variation(elementary_action,
    #                                       motion_primitive)
    # plot_absolute_cartesian_pos_variance_for_reduced_joints(elementary_action,
    #                                                         motion_primitive)
    # plot_absolute_cartesian_variation_for_aligned_data()
    # joint_vars = get_absolute_cartesian_position_variance_for_all_joints(elementary_action,
    #                                                                      motion_primitive)
    # average_vars = np.average(joint_vars.values())
    # print(average_vars)
    # joint_var = get_relative_quaternion_variation_for_all_joints(elementary_action,
    #                                                              motion_primitive)
    # print(joint_var)
    # gen_absolute_cartesian_position_variation_for_motion_data()
    # gen_relative_cartesian_position_variation_for_motion_data()
    # plot_relative_cartesian_variation_for_aligned_data()
    # gen_relative_quaternion_variation_for_aligned_data()
    # plot_relative_quaternion_variation_for_aligned_data()
    # get_variance_for_each_dimension_scaled_motion_data()
    walking_variation_test(elementary_action, motion_primitive, skeleton_file)
