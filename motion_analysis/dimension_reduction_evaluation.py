# encoding: UTF-8
'''
Evaluate the performance using different dimension reduction methods on motion data
'''

from morphablegraphs.utilities.io_helper_functions import get_aligned_data_folder
from morphablegraphs.construction.construction_algorithm_configuration import ConstructionAlgorithmConfigurationBuilder
from morphablegraphs.construction.fpca.motion_dimension_reduction import MotionDimensionReduction
from morphablegraphs.animation_data import BVHReader, BVHWriter, Skeleton, MotionVector
from morphablegraphs.animation_data.utils import get_cartesian_coordinates_from_quaternion
import glob
import numpy as np
import os
from morphablegraphs.utilities import write_to_json_file, load_json_file
import matplotlib.pyplot as plt
import pylab as pb
from collections import OrderedDict
from morphablegraphs.external.PCA import *
DEFAULT_ANIMATED_JOINT_LIST = ["Hips", "Spine", "Spine_1", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"]


####### GET RECONSTRUCTION MOTIONS
def export_reconstructed_motion(elementary_action,
                                motion_primitive,
                                dimension_reduction_method='fpca',
                                save_folder='.',
                                fraction=0.95,
                                data_repo=r'C:\repo'):
    '''
    Save the reconstructed training data to specified folder
    :param elementary_action (str): elementary action name
    :param motion_primitive (str): motion primitive name
    :param save_folder (str): path to save folder
    :param data_repo (str): path to data folder
    :return:
    '''
    aligned_data_folder = get_aligned_data_folder(elementary_action,
                                                  motion_primitive,
                                                  repo_dir=data_repo)
    spatial_data = {}
    bvhfiles = glob.glob(os.path.join(aligned_data_folder, '*.bvh'))
    for i in range(len(bvhfiles)):
        bvhreader = BVHReader(bvhfiles[i])
        filename = os.path.split(bvhfiles[i])[-1]
        spatial_data[filename] = bvhreader.frames
    skeleton = Skeleton()
    skeleton.load_from_bvh(bvhreader, animated_joints=DEFAULT_ANIMATED_JOINT_LIST)
    params = ConstructionAlgorithmConfigurationBuilder(elementary_action,
                                                       motion_primitive)
    params.fraction = fraction
    dimension_reductor = MotionDimensionReduction(bvhreader,
                                                  params)
    dimension_reductor.load_spatial_data(spatial_data)
    if dimension_reduction_method == 'fpca':
        ## hard coded number of knots for different elementary actions
        if 'walk' in elementary_action.lower() or 'carry' in elementary_action.lower() or 'run' in elementary_action.lower():
            params.n_basis_functions_spatial = 7
        if 'turn' in motion_primitive.lower() or 'sidestep' in motion_primitive.lower():
            params.n_basis_functions_spatial = 47
        print('number of basis: ', params.n_basis_functions_spatial)
        dimension_reductor.use_fpca_on_spatial_data()
        reconstructed_data = dimension_reductor.get_backprojection_from_fpca()

        for file_index in range(len(dimension_reductor.fpca_spatial.fileorder)):
            BVHWriter(os.path.join(save_folder, dimension_reductor.fpca_spatial.fileorder[file_index]),
                      skeleton, reconstructed_data[file_index], bvhreader.frame_time,
                      is_quaternion=True, skipped_joints=True)
    elif dimension_reduction_method == 'pca':
        dimension_reductor.use_pca_on_spatial_data()
        reconstructed_data = dimension_reductor.get_backprojection_from_pca()
        for file_index in range(len(dimension_reductor.pca_spatial.fileorder)):
            BVHWriter(os.path.join(save_folder, dimension_reductor.fpca_spatial.fileorder[file_index]),
                      skeleton, reconstructed_data[file_index], bvhreader.frame_time,
                      is_quaternion=True, skipped_joints=True)
    else:
        raise NotImplementedError


def get_joints_position(elementary_action,
                        motion_primitive,
                        target_joints,
                        frame_index,
                        data_repo=r'C:\repo'):
    aligned_data_folder = get_aligned_data_folder(elementary_action,
                                                  motion_primitive,
                                                  repo_dir=data_repo)
    spatial_data = {}
    bvhfiles = glob.glob(os.path.join(aligned_data_folder, '*.bvh'))
    n_files = len(bvhfiles)
    for i in range(n_files):
        bvhreader = BVHReader(bvhfiles[i])
        filename = os.path.split(bvhfiles[i])[-1]
        spatial_data[filename] = bvhreader.frames
    filelist = spatial_data.keys()
    joints_pos = {}
    for joint in target_joints:
        cartesian_positions = []
        for i in range(n_files):
            bvhreader = BVHReader(os.path.join(aligned_data_folder, filelist[i]))
            skeleton = Skeleton()
            skeleton.load_from_bvh(bvhreader)
            cartesian_positions.append(list(skeleton.nodes[joint].get_global_position_from_euler_frame(bvhreader.frames[frame_index])))
        joints_pos[joint] = cartesian_positions
    return joints_pos, filelist


def compare_fpca_and_pca_on_motion_data():
    reserved_variance = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    elementary_action = 'pickBoth'
    motion_primitive = 'reach'
    target_joints = ['LeftHand', 'RightHand']
    frame_index = -1
    data_repo = r'C:\repo'
    pca_errors = []
    fpca_errors = []
    for var in reserved_variance:
        print(var)
        pca_errors.append(standPCA_on_motion_data(elementary_action,
                                                  motion_primitive,
                                                  target_joints,
                                                  frame_index,
                                                  var,
                                                  data_repo))
        fpca_errors.append(fpca_on_motion_data(elementary_action,
                                               motion_primitive,
                                               target_joints,
                                               frame_index,
                                               var,
                                               data_repo))
    fig = plt.figure()
    plt.plot(range(len(reserved_variance)), fpca_errors, label='FPCA reconstruction error')
    plt.plot(range(len(reserved_variance)), pca_errors, label='PCA reconstruction error')
    plt.xticks(range(len(reserved_variance)), reserved_variance)
    plt.xlabel('Variance of original data')
    plt.ylabel('Cartesian error of RightFoot [CM]')
    plt.legend()
    plt.show()


def export_cartesian_pos_for_raw_and_reconstructed_data(elementary_action,
                                                        motion_primitive,
                                                        interested_joints,
                                                        frame_index,
                                                        dimension_reduction_method='fpca',
                                                        fraction=0.95,
                                                        save_folder='.',
                                                        data_repo=r'C:\repo'):
    '''
    export position of target joints for original data and reconstructed data from FPCA
    :param elementary_action:
    :param motion_primitive:
    :param interested_joints:
    :param frame_index:
    :param save_folder:
    :param data_repo:
    :return:
    '''
    motion_data = {'elementary_action': elementary_action,
                   'motion_primitive': motion_primitive,
                   'data_type': 'raw data'}
    # joints_pos, filelist = get_joints_position(elementary_action,
    #                                            motion_primitive,
    #                                            interested_joints,
    #                                            frame_index,
    #                                            data_repo)
    aligned_data_folder = get_aligned_data_folder(elementary_action,
                                                  motion_primitive,
                                                  repo_dir=data_repo)
    spatial_data = {}
    bvhfiles = glob.glob(os.path.join(aligned_data_folder, '*.bvh'))
    n_files = len(bvhfiles)
    for i in range(n_files):
        bvhreader = BVHReader(bvhfiles[i])
        filename = os.path.split(bvhfiles[i])[-1]
        spatial_data[filename] = bvhreader.frames

    ### extract joints' position for raw data
    motion_data['file_list'] = spatial_data.keys()
    motion_data['number of motions'] = len(motion_data['file_list'])
    motion_data['frame_index'] = frame_index
    ref_skeleton = Skeleton()
    ref_skeleton.load_from_bvh(bvhreader, animated_joints=DEFAULT_ANIMATED_JOINT_LIST)
    for joint in interested_joints:
        cartesian_positions = []
        for i in range(n_files):
            bvhreader = BVHReader(os.path.join(aligned_data_folder, motion_data['file_list'][i]))
            cartesian_positions.append(list(ref_skeleton.nodes[joint].get_global_position_from_euler_frame(bvhreader.frames[frame_index])))
        motion_data[joint] = cartesian_positions
    export_filename = '_'.join([elementary_action, motion_primitive, 'raw', 'data.json'])
    write_to_json_file(os.path.join(save_folder, export_filename), motion_data)

    # apply dimension reduction and reconstruct motion data
    params = ConstructionAlgorithmConfigurationBuilder(elementary_action,
                                                       motion_primitive)
    params.fraction = fraction
    dimension_reductor = MotionDimensionReduction(bvhreader,
                                                  params)
    dimension_reductor.load_spatial_data(spatial_data)
    if dimension_reduction_method == 'fpca':
        if 'walk' in elementary_action.lower() or 'carry' in elementary_action.lower() or 'run' in elementary_action.lower():
            params.n_basis_functions_spatial = 7
        if 'turn' in motion_primitive.lower() or 'sidestep' in motion_primitive.lower():
            params.n_basis_functions_spatial = 47
        print('number of basis: ', params.n_basis_functions_spatial)


        dimension_reductor.use_fpca_on_spatial_data()
        reconstructed_data = dimension_reductor.get_backprojection_from_fpca()
    elif dimension_reduction_method == 'pca':
        dimension_reductor.use_pca_on_spatial_data()
        reconstructed_data = dimension_reductor.get_backprojection_from_pca()
    reconstructed_motion_data = {'frame_index': frame_index,
                                 'elementary_action': elementary_action,
                                 'motion_primitive': motion_primitive,
                                 'data_type': 'reconstructed data',
                                 'file_list': dimension_reductor.fpca_spatial.fileorder,
                                 'number of motions': len(dimension_reductor.fpca_spatial.fileorder)}

    for joint in interested_joints:
        cartesian_positions = []
        for i in range(len(bvhfiles)):
            cartesian_positions.append(list(get_cartesian_coordinates_from_quaternion(ref_skeleton, joint,
                                                                                      reconstructed_data[i, frame_index])))
        reconstructed_motion_data[joint] = cartesian_positions
    export_filename = '_'.join([elementary_action, motion_primitive, 'reconstructed', 'data.json'])
    # diffs = []
    # for filename in motion_data['file_list']:
    #     raw_index = motion_data['file_list'].index(filename)
    #     reconstructed_index = reconstructed_motion_data['file_list'].index(filename)
    #     raw_pos = np.asarray(motion_data['RightFoot'][raw_index])
    #     reconstructed_pos = np.asarray(reconstructed_motion_data['RightFoot'][reconstructed_index])
    #     diffs.append(np.linalg.norm(raw_pos - reconstructed_pos))
    # fig = plt.figure()
    # pb.hist(diffs, bins=20)
    # pb.xlabel('distance [cm]')
    # pb.ylabel('number of samples')
    # pb.show()
    # print('average reconstruction error is: ', np.average(diffs))
    write_to_json_file(os.path.join(save_folder, export_filename), reconstructed_motion_data)


def fpca_on_motion_data(elementary_action,
                        motion_primitive,
                        target_joints,
                        frame_index=-1,
                        fraction=0.95,
                        data_repo=r'C:\repo',
                        plot=False):
    aligned_data_folder = get_aligned_data_folder(elementary_action,
                                                  motion_primitive,
                                                  data_repo)
    spatial_data = {}
    bvhfiles = glob.glob(os.path.join(aligned_data_folder, '*.bvh'))
    for i in range(len(bvhfiles)):
        bvhreader = BVHReader(bvhfiles[i])
        filename = os.path.split(bvhfiles[i])[-1]
        spatial_data[filename] = bvhreader.frames

    params = ConstructionAlgorithmConfigurationBuilder(elementary_action,
                                                       motion_primitive)
    params.fraction = fraction
    if 'walk' in elementary_action.lower() or 'carry' in elementary_action.lower() or 'run' in elementary_action.lower():
        params.n_basis_functions_spatial = 7
    if 'turn' in motion_primitive.lower() or 'sidestep' in motion_primitive.lower():
        params.n_basis_functions_spatial = 47
    # params.n_basis_functions_spatial = 7
    print('number of basis: ', params.n_basis_functions_spatial)
    dimension_reductor = MotionDimensionReduction(bvhreader,
                                                  params)
    ref_skeleton = Skeleton()
    ref_skeleton.load_from_bvh(bvhreader, animated_joints=DEFAULT_ANIMATED_JOINT_LIST)
    dimension_reductor.load_spatial_data(spatial_data)
    dimension_reductor.use_fpca_on_spatial_data()
    reconstructed_data = dimension_reductor.get_backprojection_from_fpca()
    fileorder = dimension_reductor.fpca_spatial.fileorder
    #### evaluate Cartesian error
    full_body_errors = []
    for filename in fileorder:
        joint_errors = 0
        for joint in DEFAULT_ANIMATED_JOINT_LIST:
            raw_pos = ref_skeleton.nodes[joint].get_global_position_from_euler_frame(spatial_data[filename][frame_index, :])
            index = fileorder.index(filename)
            reconstructed_pos = ref_skeleton.nodes[joint].get_global_position(reconstructed_data[index, frame_index, :])
            joint_errors += np.linalg.norm(raw_pos - reconstructed_pos)
        full_body_errors.append(joint_errors/len(DEFAULT_ANIMATED_JOINT_LIST))
    print('average full body error is: ', np.average(full_body_errors))



    errors = []
    for filename in fileorder:
        joint_errors = 0
        for target_joint in target_joints:
            raw_pos = ref_skeleton.nodes[target_joint].get_global_position_from_euler_frame(spatial_data[filename][frame_index, :])
            index = fileorder.index(filename)
            reconstructed_pos = ref_skeleton.nodes[target_joint].get_global_position(reconstructed_data[index, frame_index, :])
            joint_errors += np.linalg.norm(raw_pos - reconstructed_pos)
        errors.append(joint_errors/len(target_joints))
    print('average error is: ', np.average(errors))
    if plot:
        fig = plt.figure()
        pb.hist(errors, bins=20)
        pb.show()
    return np.average(errors)

# def compare_joint_position():
#
#     elementary_action = 'walk'
#     motion_primitive = 'rightStance'
#     raw_data_folder = get_aligned_data_folder(elementary_action, motion_primitive)
#     reconstructed_data_folder = r'E:\experiment data\reconstructed_data_from_FPCA\rightStance'
#     raw_bvhfiles = glob.glob(os.path.join(raw_data_folder, '*.bvh'))
#     reconstructed_bvhfiles = glob.glob(os.path.join(reconstructed_data_folder, '*.bvh'))
#     diffs = OrderedDict()
#     test_joint = 'RightFoot'
#     for bvhfile in raw_bvhfiles:
#         filename = os.path.split(bvhfile)[-1]
#         reconstructed_bvhfile = os.path.join(reconstructed_data_folder, filename)
#         assert os.path.exists(reconstructed_bvhfile)
#         raw_bvh = BVHReader(bvhfile)
#         raw_skeleton = Skeleton()
#         raw_skeleton.load_from_bvh(raw_bvh)
#         raw_position = raw_skeleton.nodes[test_joint].get_global_position_from_euler_frame(raw_bvh.frames[-1])
#         reconstructed_bvh = BVHReader(reconstructed_bvhfile)
#         reconstructed_skeleton = Skeleton()
#         reconstructed_skeleton.load_from_bvh(reconstructed_bvh)
#         reconstructed_position = reconstructed_skeleton.nodes[test_joint].get_global_position_from_euler_frame(reconstructed_bvh.frames[-1])
#         diff = np.linalg.norm(raw_position - reconstructed_position)
#         diffs[filename] = diff
#     fig = plt.figure()
#     pb.hist(diffs.values())
#     pb.show()


def standPCA_on_motion_data(elementary_action,
                            motion_primitive,
                            target_joints,
                            frame_index=-1,
                            fraction=0.95,
                            data_repo=r'C:\repo',
                            plot=False):
    aligned_data_folder = get_aligned_data_folder(elementary_action,
                                                  motion_primitive,
                                                  data_repo)
    spatial_data = {}
    bvhfiles = glob.glob(os.path.join(aligned_data_folder, '*.bvh'))
    for i in range(len(bvhfiles)):
        bvhreader = BVHReader(bvhfiles[i])
        filename = os.path.split(bvhfiles[i])[-1]
        spatial_data[filename] = bvhreader.frames

    params = ConstructionAlgorithmConfigurationBuilder(elementary_action,
                                                       motion_primitive)
    params.fraction = fraction
    dimension_reductor = MotionDimensionReduction(bvhreader,
                                                  params)
    ref_skeleton = Skeleton()
    ref_skeleton.load_from_bvh(bvhreader, animated_joints=DEFAULT_ANIMATED_JOINT_LIST)
    dimension_reductor.load_spatial_data(spatial_data)

    dimension_reductor.use_pca_on_spatial_data()
    reconstructed_data = dimension_reductor.get_backprojection_from_pca()
    fileorder = dimension_reductor.pca_spatial.fileorder
    errors = []
    for filename in fileorder:
        joint_errors = 0
        for target_joint in target_joints:
            raw_pos = ref_skeleton.nodes[target_joint].get_global_position_from_euler_frame(spatial_data[filename][frame_index, :])
            index = fileorder.index(filename)
            reconstructed_pos = ref_skeleton.nodes[target_joint].get_global_position(reconstructed_data[index, frame_index, :])
            joint_errors += np.linalg.norm(raw_pos - reconstructed_pos)
        errors.append(joint_errors/len(target_joints))
    print('average error is: ', np.average(errors))
    if plot:
        fig = plt.figure()
        pb.hist(errors, bins=20)
        pb.show()
    return np.average(errors)


def SFPCA_on_motion_data(elementary_action,
                        motion_primitive,
                        target_joints,
                        npc,
                        frame_index=-1,
                        data_repo=r'C:\repo',
                        plot=False):
    from morphablegraphs.construction.fpca.scaled_fpca import ScaledFunctionalPCA
    from morphablegraphs.construction.fpca.functional_data import FunctionalData
    from morphablegraphs.utilities.custom_math import error_measure_3d_mat
    from sklearn.decomposition import PCA
    from morphablegraphs.motion_analysis.prepare_data import reshape_data_for_PCA, reshape_2D_data_to_motion_data
    aligned_data_folder = get_aligned_data_folder(elementary_action,
                                                  motion_primitive,
                                                  data_repo)
    spatial_data = {}
    bvhfiles = glob.glob(os.path.join(aligned_data_folder, '*.bvh'))
    for i in range(len(bvhfiles)):
        bvhreader = BVHReader(bvhfiles[i])
        filename = os.path.split(bvhfiles[i])[-1]
        spatial_data[filename] = bvhreader.frames

    params = ConstructionAlgorithmConfigurationBuilder(elementary_action,
                                                       motion_primitive)
    dimension_reductor = MotionDimensionReduction(bvhreader,
                                                  params)
    ref_skeleton = Skeleton()
    ref_skeleton.load_from_bvh(bvhreader, animated_joints=DEFAULT_ANIMATED_JOINT_LIST)
    dimension_reductor.load_spatial_data(spatial_data)
    dimension_reductor.convert_euler_to_quat()
    smoothed_quat_frames = dimension_reductor.smooth_quat_frames(dimension_reductor.quat_frames)
    fd_data, original_shape, knots = dimension_reductor.get_fd_data()
    # print(fd_data.shape)
    skeleton_json = r'C:\Users\hadu01\git-repos\ulm\morphablegraphs\python_src\mgrd\data\skeleton.json'
    # print(dimension_reductor.params.n_basis_functions_spatial)
    # # cartesian_coeffs = convert_quat_coeffs_to_cartesian_coeffs(elementary_action,
    # #                                                            motion_primitive,
    # #                                                            data_repo,
    # #                                                            skeleton_json,
    # #                                                            fd_data,
    # #                                                            knots)
    # # print(cartesian_coeffs.shape)

    sfpca = ScaledFunctionalPCA(elementary_action,
                                motion_primitive,
                                data_repo,
                                fd_data,
                                26,
                                skeleton_json,
                                knots,
                                19)
    sfpca.fit()
    # print(sfpca.weight_vec)
    projection = sfpca.transform()
    recontructed_functional_data = sfpca.inverse_transform(projection)
    '''
    my_pca = PCA(n_components=26)
    reshaped_functional_data = reshape_data_for_PCA(fd_data)
    my_pca.fit(reshaped_functional_data)
    projection = my_pca.transform(reshaped_functional_data)
    backprojection = my_pca.inverse_transform(projection)
    recontructed_functional_data = reshape_2D_data_to_motion_data(backprojection, fd_data.shape)
    '''

    # dimension_reductor.use_sfpca_on_spatial_data()
    # reconstructed_data = dimension_reductor.get_backprojection_from_pca()
    reconstructed_data = np.zeros(original_shape)

    for i in range(original_shape[0]):
        reconstructed_data[i] = FunctionalData.from_fd_to_data_withoutR(recontructed_functional_data[i],
                                                                        knots,
                                                                        original_shape[1])

    smoothed_quat_mat = np.asarray(smoothed_quat_frames.values())

    print(error_measure_3d_mat(reconstructed_data, smoothed_quat_mat))
    fileorder = dimension_reductor.quat_frames.keys()

    #### evaluate Cartesian error
    full_body_errors = []
    for filename in fileorder:
        joint_errors = 0
        for joint in DEFAULT_ANIMATED_JOINT_LIST:
            raw_pos = ref_skeleton.nodes[joint].get_global_position_from_euler_frame(spatial_data[filename][frame_index, :])
            index = fileorder.index(filename)
            reconstructed_pos = ref_skeleton.nodes[joint].get_global_position(reconstructed_data[index, frame_index, :])
            joint_errors += np.linalg.norm(raw_pos - reconstructed_pos)
        full_body_errors.append(joint_errors/len(DEFAULT_ANIMATED_JOINT_LIST))
    print('average full body error is: ', np.average(full_body_errors))


    errors = []
    for filename in fileorder:
        joint_errors = 0
        for target_joint in target_joints:
            raw_pos = ref_skeleton.nodes[target_joint].get_global_position_from_euler_frame(spatial_data[filename][frame_index, :])
            index = fileorder.index(filename)
            reconstructed_pos = ref_skeleton.nodes[target_joint].get_global_position(reconstructed_data[index, frame_index, :])
            joint_errors += np.linalg.norm(raw_pos - reconstructed_pos)
        errors.append(joint_errors/len(target_joints))
    print('average error is: ', np.average(errors))
    if plot:
        fig = plt.figure()
        pb.hist(errors, bins=40)
        pb.show()
    return np.average(errors)


def plot_functional_pca_result():
    variances = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    fpca_errs = [10.916271446889183, 8.0416437400866716, 5.9330822196455326, 4.7694988816754416, 3.1631186114019925,
                 2.4960535150614991, 1.6111550383678928, 0.43595459659382428]
    pca_errs = [8.3777002818566544, 7.9485941616519851, 5.4507466110645488, 4.8211464735560456, 3.1899368740894984,
                2.6087734102605271, 1.4484107357563676, 8.3647437071944143e-14]
    fig = plt.figure()
    plt.plot(range(len(variances)), fpca_errs, label='FPCA reconstruction error')
    plt.plot(range(len(variances)), pca_errs, label='PCA reconstruction error')
    plt.xticks(range(len(variances)), variances)
    plt.xlabel('Variance of original data')
    plt.ylabel('Cartesian error of RightFoot [CM]')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # compare_fpca_and_pca_on_motion_data()

    #### save joints' position
    # elementary_action = 'pick'
    # motion_primitive = 'rightStance'
    # interested_joints = ['Hips', 'RightFoot']
    # frame_index = -1
    # dimension_reduction_method = 'fpca'
    # fraction = 0.95
    # save_folder = r'E:\tmp'
    # export_cartesian_pos_for_raw_and_reconstructed_data(elementary_action,
    #                                                     motion_primitive,
    #                                                     interested_joints,
    #                                                     frame_index,
    #                                                     dimension_reduction_method,
    #                                                     fraction,
    #                                                     save_folder)

    ### fcuntional PCA
    elementary_action = 'walk'
    motion_primitive = 'leftStance'
    target_joints = ['LeftFoot']
    frame_index = -1
    data_repo = r'C:\repo'
    variance = 0.95
    # npc = 26  ## this could be calculate from FPCA using variance to be kept
    npc = 33 ## left step
    # err = fpca_on_motion_data(elementary_action,
    #                           motion_primitive,
    #                           target_joints,
    #                           frame_index,
    #                           variance,
    #                           data_repo,
    #                           plot=True)

    # err = standPCA_on_motion_data(elementary_action,
    #                               motion_primitive,
    #                               target_joints,
    #                               frame_index,
    #                               variance,
    #                               data_repo,
    #                               plot=True)
    #
    err = SFPCA_on_motion_data(elementary_action,
                               motion_primitive,
                               target_joints,
                               frame_index,
                               npc,
                               data_repo,
                               plot=True)