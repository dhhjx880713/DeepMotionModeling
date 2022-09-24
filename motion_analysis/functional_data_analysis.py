#encoding: UTF-8

'''
This script provides functions to evaluate and analyze the performance of cubic Bspline
1. functional_data_analysis(elementary_action, motion_primitive, n_basis, skeleton)
compare reconstruction error in Cartesian space and Quaternion space for one motion primitive

2. evalute_information_loss_for_functional_data(elementary_action, motion_primitive)
'''

from morphablegraphs.utilities import get_data_analysis_folder, load_json_file, write_to_json_file,\
                                      get_b_spline_knots
from morphablegraphs.utilities.custom_math import error_measure_3d_mat
import os
from morphablegraphs.construction.fpca import FPCASpatialData, PCAFunctionalData
import collections
import numpy as np
import copy
from morphablegraphs.animation_data import BVHReader, Skeleton
import matplotlib.pyplot as plt
import scipy.interpolate as si
from mpl_toolkits.mplot3d import Axes3D


def reshape_data(input_data):
    """
    reshape the 3d motion data matrix form n_frames (coeffs) * n_samples * n_dims to n_samples * n_frames (coeffs) * n_dims
    :param input_data (numpy.array <3d>): n_frames * n_samples * n_dims
    :return:  reshaped_data (numpy.array <3d>): n_samples * n_frames * n_dims
    """
    n_frames, n_samples, n_dims = input_data.shape
    reshaped_data = np.zeros((n_samples, n_frames, n_dims))
    for i in range(n_samples):
         reshaped_data[i, :, :] = input_data[:, i, :]
    return reshaped_data


def save_scaled_functional_data(target_folder,
                                functional_datamat,
                                n_basis,
                                n_frames,
                                scale_vector):
    knots = get_b_spline_knots(n_basis, n_frames)
    output_data = {'scale_vector': scale_vector,
                   'knots': knots.tolist(),
                   'functional_data': functional_datamat,
                   'n_canonical_frames': n_frames}
    output_filename = os.path.join(target_folder, 'functional_data_' + str(n_basis) + '_knots.json')
    write_to_json_file(output_filename, output_data)

def create_samples_from_fitting_spline(training_data,
                                       n_knots=7,
                                       granularity=100,
                                       degree=3):
    training_data = np.asarray(training_data)
    assert len(training_data.shape) == 2
    n_frames, n_dims = training_data.shape
    knots = get_b_spline_knots(n_knots, n_frames)
    x = range(n_frames)
    coeffs = [si.splrep(x, training_data[:, i], k=degree, t=knots[degree+1: -(degree+1)])[1][:-4] for i in range(n_dims)]
    x = np.linspace(0, n_frames-1, granularity)
    samples = [si.splev(x, (knots, coeffs[i], degree)) for i in range(n_dims)]
    return np.asarray(samples).T


def plot_functional_representation_for_one_joint(bvhfile):
    bvhreader = BVHReader(bvhfile)
    skeleton = Skeleton()
    skeleton.load_from_bvh(bvhreader)
    test_joint = 'Bip01_R_Toe0'
    n_frames = len(bvhreader.frames)
    joint_positions = []
    for i in range(n_frames):
        joint_positions.append(skeleton.nodes[test_joint].get_global_position_from_euler_frame(bvhreader.frames[i]))
    joint_positions = np.asarray(joint_positions)
    # print(joint_positions.shape)
    # tck = si.splrep(range(n_frames), joint_positions[:, 0])
    # print(tck)
    spline_samples = create_samples_from_fitting_spline(joint_positions)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(joint_positions[:, 0], joint_positions[:, 2], joint_positions[:, 1], 'ro', label='original motion')
    ax.plot(spline_samples[:, 0], spline_samples[:, 2], spline_samples[:, 1], 'b', label='spline fitting')
    plt.legend()
    plt.show()


def functional_data_analysis(elementary_action,
                             motion_primitive,
                             n_basis,
                             skeleton):
    repo_dir = r'C:\repo'
    data_analysis_folder = get_data_analysis_folder(elementary_action,
                                                    motion_primitive,
                                                    repo_dir)
    functional_data_file = os.path.join(data_analysis_folder, 'functional_data_' + str(n_basis) + '_knots.json')
    scaled_motion_datafile = os.path.join(data_analysis_folder, 'scaled_smoothed_quat_frames.json')

    # load scaled quaternion frame data
    if not os.path.exists(scaled_motion_datafile):
        scaled_smoothed_quat_frames, scale_vector = get_smoothed_quat_frames(elementary_action,
                                                                             motion_primitive,
                                                                             scale_root=True)
        output_data = {'data': scaled_smoothed_quat_frames,
                       'scale_vector': scale_vector}
        write_to_json_file(scaled_motion_datafile,
                           output_data)
    else:
        data = load_json_file(os.path.join(data_analysis_folder, 'scaled_smoothed_quat_frames.json'))
        scaled_smoothed_quat_frames = data['data']
        scaled_smoothed_quat_frames = collections.OrderedDict(sorted(scaled_smoothed_quat_frames.items()))
        scale_vector = data['scale_vector']
    raw_data = np.asarray(copy.deepcopy(scaled_smoothed_quat_frames.values()))
    if os.path.exists(functional_data_file):
        input_data = load_json_file(functional_data_file)
        n_canonical_frame = input_data['n_canonical_frames']
        functional_data = np.asarray(input_data['functional_data'])
        n_samples, n_frames, n_dims = functional_data.shape
        reshaped_functional_data = np.zeros((n_frames, n_samples, n_dims))
        for i in range(n_samples):
            reshaped_functional_data[:, i, :] = functional_data[i, :, :]
        resampled_motion_data = PCAFunctionalData.from_fd_to_data(reshaped_functional_data,
                                                                  n_canonical_frame)
    else:
        fpca_spatial = FPCASpatialData(scaled_smoothed_quat_frames, n_basis)
        fpca_spatial.fpca_on_spatial_data()
        n_frames, n_samples, n_dims = fpca_spatial.reshaped_data.shape
        reshaped_functional_data = reshape_data(np.asarray(fpca_spatial.fpcaobj.functional_data))
        save_scaled_functional_data(data_analysis_folder,
                                    reshaped_functional_data.tolist(),
                                    n_basis,
                                    n_frames,
                                    scale_vector)

        resampled_motion_data = fpca_spatial.fpcaobj.from_fd_to_data(np.asarray(fpca_spatial.fpcaobj.functional_data),
                                                                     n_frames)
    cartesian_error = cartesian_distance_measure(raw_data, resampled_motion_data, skeleton)
    quat_error = error_measure_3d_mat(raw_data, resampled_motion_data)
    return cartesian_error, quat_error


def evalute_information_loss_for_functional_data(elementary_action,
                                                 motion_primitive):
    print(elementary_action)
    print(motion_primitive)
    cartesian_errors = []
    quat_errors = []
    norder = 4
    n_basis = range(norder, 41)
    bvhreader = BVHReader(r'../../skeleton.bvh')
    skeleton = Skeleton()
    skeleton.load_from_bvh(bvhreader)
    data_repo = r'C:\repo'
    for i in n_basis:
        print(i)
        cartesian_err, quat_err = functional_data_analysis(elementary_action,
                                                           motion_primitive,
                                                           i,
                                                           skeleton)
        cartesian_errors.append(cartesian_err)
        quat_errors.append(quat_err)
    output_data = {'n_basis': n_basis,
                   'cartesian_errors': cartesian_errors,
                   'quat_errors': quat_errors}
    data_analysis_folder = get_data_analysis_folder(elementary_action,
                                                    motion_primitive,
                                                    data_repo)
    output_filename = os.path.join(data_analysis_folder, 'functional_representation_error.json')
    write_to_json_file(output_filename, output_data)


def plot_functional_representation_error_for_multimotions():
    motion_primitives = ['walk_leftStance', 'pickBoth_first', 'lookAt_lookAt', 'transfer_transfer', 'screw_reach',
                         'walk_rightStance', 'placeBoth_first', 'carryBoth_leftStance']
    labels = {'walk_leftStance': 'walking',
              'walk_rightStance': 'sidestep',
              'pickBoth_first': 'two-hand picking',
              'transfer_transfer': 'two-hand transfer',
              'screw_reach': 'screwing',
              'lookAt_lookAt': 'looking around',
              'placeBoth_first': 'two-hand placing',
              'carryBoth_leftStance': 'carrying'}
    data_repo = r'C:\repo'
    functional_reconstruction_errors = {}
    for motion_type in motion_primitives:
        elementary_action, motion_primitive = motion_type.split('_')
        data_analysis_folder = get_data_analysis_folder(elementary_action,
                                                        motion_primitive,
                                                        data_repo)
        functional_error_file = os.path.join(data_analysis_folder, 'functional_representation_error.json')
        data = load_json_file(functional_error_file)
        functional_reconstruction_errors[motion_type] = {'cartesian_errors': data['cartesian_errors'],
                                                         'n_basis': data['n_basis']}
    fig = plt.figure()
    for key, value in functional_reconstruction_errors.iteritems():
        plt.plot(value['n_basis'], value['cartesian_errors'], label=labels[key])
    plt.xlabel('Number of basis functions K', fontsize=15)
    plt.xlim((4, value['n_basis'][-1]))
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.ylabel('Root mean square error in joint space $[cm]$', fontsize=15)
    # plt.title('Functional Representation Evaluation', fontsize=25)
    plt.legend()
    plt.show()
    filename = 'functional_representation_evaluation.pdf'
    fig.savefig(filename, format='PDF')


if __name__ == "__main__":
    elementary_action = 'placeLeft'
    motion_primitive = 'first'
    test_file = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\rightStance\walk_001_3_rightStance_354_398.bvh'
    # evalute_information_loss_for_functional_data(elementary_action,
    #                                              motion_primitive)
    # plot_functional_representation_error_for_multimotions()
    plot_functional_representation_for_one_joint(test_file)
