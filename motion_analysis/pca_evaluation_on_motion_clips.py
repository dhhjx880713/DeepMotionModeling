# encoding: UTF-8
from morphablegraphs.utilities import write_to_json_file, get_data_analysis_folder, load_aligned_data, load_json_file
from morphablegraphs.utilities.custom_math import error_measure_3d_mat
from morphablegraphs.animation_data import BVHReader, Skeleton
from morphablegraphs.construction.fpca import MotionDimensionReduction
from morphablegraphs.construction.construction_algorithm_configuration import ConstructionAlgorithmConfigurationBuilder
import os
from helpfunctions import standardPCA, reshape_data_for_PCA, reshape_2D_data_to_motion_data, cartesian_error_measure
import matplotlib.pyplot as plt
import numpy as np
import copy
import collections
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_smoothed_quat_frames(elementary_action,
                             motion_primitive,
                             scale=False):
    aligned_motion_data = load_aligned_data(elementary_action, motion_primitive)
    params = ConstructionAlgorithmConfigurationBuilder(elementary_action, motion_primitive)
    skeleton_file = r'../skeleton.bvh'
    bvhreader = BVHReader(skeleton_file)
    dimension_reductor = MotionDimensionReduction(aligned_motion_data,
                                                  bvhreader,
                                                  params)
    dimension_reductor.convert_euler_to_quat()
    if scale:
        dimension_reductor.scale_rootchannels()
        smoothed_quat_frames = dimension_reductor.smooth_quat_frames(dimension_reductor.rescaled_quat_frames)
        for filename, value in smoothed_quat_frames.iteritems():
            smoothed_quat_frames[filename] = value.tolist()
        return smoothed_quat_frames, dimension_reductor.fdata['scale_vector']

    else:
        smoothed_quat_frames = dimension_reductor.smooth_quat_frames(dimension_reductor.quat_frames)

        for filename, value in smoothed_quat_frames.iteritems():
            smoothed_quat_frames[filename] = value.tolist()
        return smoothed_quat_frames


def load_skeleton():
    skeleton_file = r'../../skeleton.bvh'
    bvhreader = BVHReader(skeleton_file)
    skeleton = Skeleton()
    skeleton.load_from_bvh(bvhreader)
    return skeleton


def evaluate_standard_pca_on_motion_data(elementary_action,
                                         motion_primitive,
                                         npcs,
                                         save_result=True,
                                         weighted_error_func=True):
    data_analysis_folder = get_data_analysis_folder(elementary_action,
                                                    motion_primitive,
                                                    repo_dir=r'C:\repo')
    skeleton = load_skeleton()
    # load smoothed quaternion data
    print("load quaternion frames")
    optimized_weights_file = os.path.join(data_analysis_folder, '_'.join([elementary_action, motion_primitive,
                                                                          'optimization', 'result.json']))
    if not os.path.isfile(optimized_weights_file):
        raise IOError('Cannot find weight file for scaling features')
    else:
        optimal_weights_dic = load_json_file(optimized_weights_file)
    if not os.path.exists(os.path.join(data_analysis_folder, 'smoothed_quat_frames.json')):
        smoothed_quat_frames = get_smoothed_quat_frames(elementary_action,
                                                        motion_primitive)
        write_to_json_file(os.path.join(data_analysis_folder, 'smoothed_quat_frames.json'),
                           smoothed_quat_frames)
    else:
        smoothed_quat_frames = load_json_file(os.path.join(data_analysis_folder, 'smoothed_quat_frames.json'))
        smoothed_quat_frames = collections.OrderedDict(sorted(smoothed_quat_frames.items()))

    # load scaled smoothed quaternion data
    print("load scaled quaternion frames")
    if not os.path.exists(os.path.join(data_analysis_folder, 'scaled_smoothed_quat_frames.json')):
        scaled_smoothed_quat_frames, scale_vector = get_smoothed_quat_frames(elementary_action,
                                                                             motion_primitive,
                                                                             scale=True)
        output_data = {'data': scaled_smoothed_quat_frames,
                       'scale_vector': scale_vector}
        write_to_json_file(os.path.join(data_analysis_folder, 'scaled_smoothed_quat_frames.json'),
                           output_data)
    else:
        data = load_json_file(os.path.join(data_analysis_folder, 'scaled_smoothed_quat_frames.json'))
        scaled_smoothed_quat_frames = data['data']
        scaled_smoothed_quat_frames = collections.OrderedDict(sorted(scaled_smoothed_quat_frames.items()))
        scale_vector = data['scale_vector']

    unscaled_errors = {'quat_err': [],
                       'cartesian_err': []}
    scaled_errors = {'quat_err': [],
                     'cartesian_err': []}
    scaled_root_errors = {'quat_err': [],
                          'cartesian_err': []}
    feature_weighted_errors = {'quat_err': [],
                               'cartesian_err': []}
    raw_data = np.asarray(copy.deepcopy(smoothed_quat_frames.values()))
    for npc in npcs:
        print(npc)
        smoothed_quat_frames_data = np.asarray(copy.deepcopy(smoothed_quat_frames.values()))

        smoothed_quat_frames_2d = reshape_data_for_PCA(smoothed_quat_frames_data)

        # unscaled_pcaobj, unscaled_centerobj = standardPCA(smoothed_quat_frames_2d, fraction)
        # scaled_pcaobj, scaled_centerobj = standardPCA(scaled_smoothed_quat_frames_2d, fraction)
        pca_unscaled = PCA(n_components=npc)
        projection = pca_unscaled.fit_transform(smoothed_quat_frames_2d)
        backprojection = pca_unscaled.inverse_transform(projection)

        reshaped_backprojection = reshape_2D_data_to_motion_data(backprojection, smoothed_quat_frames_data.shape)
        unscaled_error = error_measure_3d_mat(raw_data, reshaped_backprojection)
        unscaled_errors['quat_err'].append(unscaled_error)
        print('unscaled quat frame error: ', unscaled_error)
        unscaled_cartesian_frame_error = cartesian_error_measure(raw_data,
                                                                 reshaped_backprojection,
                                                                 skeleton,
                                                                 weighted_error=weighted_error_func)
        print('unscaled cartesian frame error: ', unscaled_cartesian_frame_error)
        unscaled_errors['cartesian_err'].append(unscaled_cartesian_frame_error)

        ### apply Probabilistic PCA on raw data



        # evaluate PCA on normalized data
        pca_scaled = PCA(n_components=npc, whiten=True)
        # normalized_smoothed_quat_frames_2d = scale(smoothed_quat_frames_2d)
        scaler = StandardScaler().fit(smoothed_quat_frames_2d)
        normalized_smoothed_quat_frames_2d = scaler.transform(smoothed_quat_frames_2d)
        projection = pca_scaled.fit_transform(normalized_smoothed_quat_frames_2d)
        backprojection = pca_scaled.inverse_transform(projection)
        backprojection = scaler.inverse_transform(backprojection)
        reshaped_backprojection = reshape_2D_data_to_motion_data(backprojection, smoothed_quat_frames_data.shape)
        scaled_error = error_measure_3d_mat(raw_data, reshaped_backprojection)
        print('scaled quat frame error: ', scaled_error)
        scaled_cartesian_frame_error = cartesian_error_measure(raw_data,
                                                               reshaped_backprojection,
                                                               skeleton,
                                                               weighted_error=weighted_error_func)
        scaled_errors['quat_err'].append(scaled_error)
        print('scaled cartesian frame error: ', scaled_cartesian_frame_error)
        scaled_errors['cartesian_err'].append(scaled_cartesian_frame_error)

        #evaluate PCA on scaled root data
        scaled_smoothed_quat_frames_data = np.asarray(copy.deepcopy(scaled_smoothed_quat_frames.values()))
        scaled_smoothed_quat_frames_2d = reshape_data_for_PCA(scaled_smoothed_quat_frames_data)
        scaled_pcaobj, scaled_centerobj = standardPCA(scaled_smoothed_quat_frames_2d)

        scaled_reconstruction_data = reconstruct_pca_data(scaled_pcaobj,
                                                          scaled_centerobj,
                                                          scaled_smoothed_quat_frames_2d,
                                                          npc)
        reshaped_scaled_reconstruction_data = reshape_2D_data_to_motion_data(scaled_reconstruction_data,
                                                                             scaled_smoothed_quat_frames_data.shape)
        reshaped_scaled_reconstruction_data[:, :, 0] *= scale_vector[0]
        reshaped_scaled_reconstruction_data[:, :, 1] *= scale_vector[1]
        reshaped_scaled_reconstruction_data[:, :, 2] *= scale_vector[2]

        quat_scaled_root_error = error_measure_3d_mat(raw_data, reshaped_scaled_reconstruction_data)
        cartesian_scaled_root_error = cartesian_error_measure(raw_data,
                                                              reshaped_scaled_reconstruction_data,
                                                              skeleton,
                                                              weighted_error=weighted_error_func)
        print("scaled root quat frame error: ", quat_scaled_root_error)
        print('scaled root cartesian error: ', cartesian_scaled_root_error)
        scaled_root_errors['quat_err'].append(quat_scaled_root_error)
        scaled_root_errors['cartesian_err'].append(cartesian_scaled_root_error)


        # apply PCA on feature weighted data
        weights = optimal_weights_dic[str(npc)]['optimal weights']
        n_joints =19
        extended_weights = np.zeros(79)
        extended_weights[:3] = weights[:3]
        for i in range(n_joints):
            extended_weights[3 + i*4 : 3 + (i+1)*4] = weights[3 + i]
        weight_mat = np.diag(extended_weights)
        smoothed_quat_frames_data = np.asarray(copy.deepcopy(smoothed_quat_frames.values()))
        weighted_quat_sample_data = np.dot(smoothed_quat_frames_data, weight_mat)
        reshape_weighted_quat_sample_data = reshape_data_for_PCA(weighted_quat_sample_data)
        pcaobj, centerobj = standardPCA(reshape_weighted_quat_sample_data)
        eigenvectors = pcaobj.Vt[:npc]
        projected_data = np.dot(eigenvectors, reshape_weighted_quat_sample_data.T)
        backprojected_data = np.dot(np.transpose(eigenvectors), projected_data).T
        reconstructed_data = backprojected_data + centerobj.mean
        reshaped_reconstructed_data = reshape_2D_data_to_motion_data(reconstructed_data, smoothed_quat_frames_data.shape)
        inv_weight_mat = np.linalg.inv(weight_mat)
        unweighted_reconstructed_data = np.dot(reshaped_reconstructed_data, inv_weight_mat)
        quat_err = error_measure_3d_mat(raw_data, unweighted_reconstructed_data)
        cartesian_err = cartesian_error_measure(smoothed_quat_frames_data, unweighted_reconstructed_data, skeleton,
                                                weighted_error=weighted_error_func)
        print("feature weighted cartesian error: ", cartesian_err)
        print("feature weighted quaternion error: ", quat_err)
        feature_weighted_errors['cartesian_err'].append(cartesian_err)
        feature_weighted_errors['quat_err'].append(quat_err)

    if save_result:
        export_data = {'unscaled_errors': unscaled_errors,
                       'scaled_errors': scaled_errors,
                       'scaled_root_errors': scaled_root_errors,
                       'feature_weighted_errors': feature_weighted_errors}
        write_to_json_file(os.path.join(data_analysis_folder, 'standardPCA_analysis_npcs.json'), export_data)


def reconstruct_pca_data(pcaobj, centerobj, data, npc=None):
    data = np.asarray(data)
    assert len(data.shape) == 2
    if npc is None:
        eigenvectors = pcaobj.Vt[:pcaobj.npc]
    else:
        eigenvectors = pcaobj.Vt[:npc]
    projected_data = np.dot(eigenvectors, data.T)
    backprojection = np.dot(projected_data.T, eigenvectors)
    backprojection += centerobj.mean
    return backprojection


def data_to_plot():
    motion_data_result_file = r'C:\repo\data\1 - MoCap\7 - Mocap analysis\elementary_action_pickBoth\first\gplvm_evaluation_on_motion_data.json'
    scaled_motion_data_file = r'C:\repo\data\1 - MoCap\7 - Mocap analysis\elementary_action_pickBoth\first\gplvm_evaluation_on_scaled_motion_data.json'
    motion_data_evaluation_result = load_json_file(motion_data_result_file)
    scaled_motion_data_result = load_json_file(scaled_motion_data_file)
    npcs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    index = np.arange(len(npcs))
    fig, ax = plt.subplots()
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index, scaled_motion_data_result['quat_frame_errors'],
                     bar_width,
                     alpha=opacity,
                     color='b',
                     error_kw=error_config,
                     label='normalized motion data')
    rects2 = plt.bar(index + bar_width, motion_data_evaluation_result['quat_frame_errors'],
                     bar_width,
                     alpha=opacity,
                     color='r',
                     error_kw=error_config,
                     label='unnormalized motion data')
    plt.xlabel('number of principal components')
    plt.ylabel('reconstruction error')
    plt.xticks(index + bar_width, npcs)
    plt.title("end-effector error")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_results():
    data_file = r'C:\repo\data\1 - MoCap\7 - Mocap analysis\elementary_action_pickBoth\first\standardPCA_analysis_npcs.json'
    res_data = load_json_file(data_file)
    raw_data = res_data['unscaled_errors']
    normalized_data = res_data['scaled_errors']
    feature_weighted_errors = res_data['scaled_root_errors']
    fig = plt.figure()
    plt.plot(raw_data['quat_err'], label='raw data')
    plt.plot(normalized_data['quat_err'], label='normalized data')
    plt.plot(feature_weighted_errors['quat_err'], label='feature weighted data')
    plt.title('Error in original data space')
    plt.xlabel('number of principal components')
    plt.ylabel('reconstruction error')
    plt.legend()
    plt.show()

    fig = plt.figure()
    plt.plot(raw_data['cartesian_err'], label='raw data')
    plt.plot(normalized_data['cartesian_err'], label='normalized data')
    plt.plot(feature_weighted_errors['cartesian_err'], label='feature weighted data')
    plt.title('Cartesian error')
    plt.xlabel('number of principal components')
    plt.ylabel('reconstruction error $[cm^2]$')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    elementary_action = 'walk'
    motion_primitive = 'leftStance'
    print(elementary_action)
    print(motion_primitive)
    npcs = range(1, 21)
    evaluate_standard_pca_on_motion_data(elementary_action,
                                         motion_primitive,
                                         npcs,
                                         save_result=True,
                                         weighted_error_func=False)
