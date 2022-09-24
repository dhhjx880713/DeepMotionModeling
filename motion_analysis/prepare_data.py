# encoding: UTF-8
import collections
import os
from morphablegraphs.utilities.io_helper_functions import load_json_file, write_to_json_file, get_data_analysis_folder, load_aligned_data, \
                                      get_semantic_motion_primitive_path
import numpy as np
from morphablegraphs.construction.fpca import FunctionalData, MotionDimensionReduction
from morphablegraphs.construction.construction_algorithm_configuration import ConstructionAlgorithmConfigurationBuilder
from morphablegraphs.animation_data import BVHReader, Skeleton
from morphablegraphs.construction.preprocessing.quat_spline_constructor import QuatSplineConstructor
from morphablegraphs.external.transformations import quaternion_matrix
from itertools import izip
import copy


def create_quat_functional_data(elementary_action,
                                motion_primitive,
                                repo_dir,
                                n_basis):
    data_analysis_folder = get_data_analysis_folder(elementary_action,
                                                    motion_primitive,
                                                    repo_dir)
    if not os.path.exists(os.path.join(data_analysis_folder, 'smoothed_quat_frames.json')):
        smoothed_quat_frames = get_smoothed_quat_frames(elementary_action,
                                                        motion_primitive)
        smoothed_quat_frames = collections.OrderedDict(sorted(smoothed_quat_frames.items()))
        write_to_json_file(os.path.join(data_analysis_folder, 'smoothed_quat_frames.json'),
                           smoothed_quat_frames)
    else:
        smoothed_quat_frames = load_json_file(os.path.join(data_analysis_folder, 'smoothed_quat_frames.json'))
        smoothed_quat_frames = collections.OrderedDict(sorted(smoothed_quat_frames.items()))
    functional_data = collections.OrderedDict()
    fd = FunctionalData()
    n_canonical_frames = np.asarray(smoothed_quat_frames.values()).shape[1]
    fd.get_knots(n_basis, n_canonical_frames)
    for filename, quat_frames in smoothed_quat_frames.iteritems():
        functional_data[filename] = fd.convert_motion_to_functional_data(quat_frames, n_basis).tolist()
    return {'functional_data': functional_data,
            'knots': fd.knots.tolist(),
            'n_canonical_frames': n_canonical_frames}


def get_smoothed_quat_frames(elementary_action,
                             motion_primitive,
                             data_repo=None,
                             scale=False):
    aligned_motion_data = load_aligned_data(elementary_action, motion_primitive, data_repo)
    params = ConstructionAlgorithmConfigurationBuilder(elementary_action, motion_primitive)
    skeleton_file = r'../skeleton.bvh'
    bvhreader = BVHReader(skeleton_file)
    skeleton = Skeleton()
    skeleton.load_from_bvh(bvhreader)
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


def reshape_data_for_PCA(data_mat):
    data_mat = np.asarray(data_mat)
    assert len(data_mat.shape) == 3
    n_samples, n_frames, n_dims = data_mat.shape
    return np.reshape(data_mat, (n_samples, n_frames * n_dims))


def reshape_2D_data_to_motion_data(data_mat_2d, origin_shape):
    assert len(origin_shape) == 3
    data_mat_2d = np.asarray(data_mat_2d)
    n_samples, n_frames, n_dims = origin_shape
    assert n_samples * n_frames * n_dims == data_mat_2d.shape[0] * data_mat_2d.shape[1]
    return np.reshape(data_mat_2d, origin_shape)


def convert_quat_coeffs_to_cartesian_coeffs(elementary_action,
                                            motion_primitive,
                                            data_repo,
                                            skeleton_json,
                                            quat_coeffs_mat,
                                            knots):
    """
    convert a set of quaternion splines to cartesian splines
    :param quat_coeffs_mat (numpy.array<3d>): n_samples * n_basis * n_dims
    :return cartesian_coeffs_mat (numpy.array<3d>): n_samples * n_basis * n_dims
    """
    semantic_motion_primitive_file = get_semantic_motion_primitive_path(elementary_action,
                                                                        motion_primitive,
                                                                        data_repo)
    quat_spline_constructor = QuatSplineConstructor(semantic_motion_primitive_file,
                                                    skeleton_json)
    quat_coeffs_mat = np.asarray(quat_coeffs_mat)
    n_samples, n_basis, n_dims = quat_coeffs_mat.shape
    cartesian_coeffs_mat = []
    for i in range(n_samples):
        quat_spline = quat_spline_constructor.create_quat_spline_from_functional_data(quat_coeffs_mat[i],
                                                                                      knots)
        cartesian_spline = quat_spline.to_cartesian()
        cartesian_coeffs_mat.append(cartesian_spline.coeffs)
    return np.asarray(cartesian_coeffs_mat)


def create_cartesian_functional_data(elementary_action,
                                     motion_primitive,
                                     repo_dir,
                                     n_basis):
    functional_data_dic = create_quat_functional_data(elementary_action, motion_primitive, repo_dir, n_basis)
    knots = functional_data_dic['knots']
    n_canonical_frames = functional_data_dic['n_canonical_frames']
    functional_coeffs_dic = functional_data_dic['functional_data']
    semantic_motion_primitive_file = get_semantic_motion_primitive_path(elementary_action,
                                                                        motion_primitive,
                                                                        repo_dir)
    skeleton_json = r'../mgrd/data/skeleton.json'
    quat_spline_constructor = QuatSplineConstructor(semantic_motion_primitive_file,
                                                    skeleton_json)
    cartesian_functional_data = collections.OrderedDict()
    for filename, quat_coeffs in functional_coeffs_dic.iteritems():
        quat_spline = quat_spline_constructor.create_quat_spline_from_functional_data(quat_coeffs,
                                                                                      knots)
        cartesian_spline = quat_spline.to_cartesian()
        cartesian_functional_data[filename] = cartesian_spline.coeffs.tolist()
    return {'functional_data': cartesian_functional_data,
            'knots': knots,
            'n_canonical_frames': n_canonical_frames}


def convert_quaternion_frames_to_cartesian_frames(skeleton, quat_frames):
    """

    :param skeleton (Skeleton):
    :param quat_frames (numpy.array<2D>): n_frames * n_dims
    :return cartesian frames: (n_frames, n_joints, LEN_CARTESIAN_POINT)
    """

    cartesian_frames = []
    quat_frames = np.asarray(quat_frames)
    n_frames, n_dims = quat_frames.shape
    assert n_dims == 79, ("The length of quaternion frame is not correct!")
    for i in range(n_frames):
        cartesian_frames.append(convert_quaternion_frame_to_cartesian_frame(skeleton, quat_frames[i]))
    return cartesian_frames


def convert_quaternion_frame_to_cartesian_frame(skeleton, quat_frame):
    """
    Converts quaternion frame to cartesian frame by calling get_cartesian_coordinates_from_quaternion for each joint
    """
    cartesian_frame = []
    for node_name in skeleton.node_names:
        # ignore Bip joints and end sites
        if not node_name.startswith(
                "Bip") and "children" in skeleton.node_names[node_name].keys():
            cartesian_frame.append(
                get_cartesian_coordinates_from_quaternion(
                    skeleton,
                    node_name,
                    quat_frame))  # get_cartesian_coordinates2

    return cartesian_frame


def get_cartesian_coordinates_from_quaternion(skeleton, node_name, quaternion_frame, return_gloabl_matrix=False):
    """Returns cartesian coordinates for one node at one frame. Modified to
     handle frames with omitted values for joints starting with "Bip"

    Parameters
    ----------

    * node_name: String
    \tName of node
     * skeleton: Skeleton
    \tBVH data structure read from a file

    """
    if skeleton.node_names[node_name]["level"] == 0:
        root_frame_position = quaternion_frame[:3]
        root_node_offset = skeleton.node_names[node_name]["offset"]

        return [t + o for t, o in
                izip(root_frame_position, root_node_offset)]

    else:
        # Names are generated bottom to up --> reverse
        chain_names = list(skeleton.gen_all_parents(node_name))
        chain_names.reverse()
        chain_names += [node_name]  # Node is not in its parent list

        offsets = [skeleton.node_names[nodename]["offset"]
                   for nodename in chain_names]
        root_position = quaternion_frame[:3].flatten()
        offsets[0] = [r + o for r, o in izip(root_position, offsets[0])]

        j_matrices = []
        count = 0
        for node_name in chain_names:
            index = skeleton.node_name_frame_map[node_name] * 4 + 3
            j_matrix = quaternion_matrix(quaternion_frame[index: index + 4])
            j_matrix[:, 3] = offsets[count] + [1]
            j_matrices.append(j_matrix)
            count += 1

        global_matrix = np.identity(4)
        for j_matrix in j_matrices:
            global_matrix = np.dot(global_matrix, j_matrix)
        if return_gloabl_matrix:
            return global_matrix
        else:
            point = np.array([0, 0, 0, 1])
            point = np.dot(global_matrix, point)
            return point[:3].tolist()


def scale_root_channels(motion_data):
    '''
    normalized root channels for motion data matrix
    :param motion_data (numpy.narray<3d>): n_samples * n_frames * n_dims
    :return:
    '''
    motion_data = copy.deepcopy(motion_data)
    max_x = np.max(np.abs(motion_data[:, :, 0]))
    max_y = np.max(np.abs(motion_data[:, :, 1]))
    max_z = np.max(np.abs(motion_data[:, :, 2]))
    motion_data[:, :, 0] /= max_x
    motion_data[:, :, 1] /= max_y
    motion_data[:, :, 2] /= max_z
    scale_root_vector = [max_x, max_y, max_z]
    return motion_data, scale_root_vector


def convert_quat_functional_data_to_cartesian_functional_data(elementary_action,
                                                              motion_primitive,
                                                              data_repo,
                                                              skeleton_json,
                                                              quat_coeffs_mat,
                                                              knots):
    """
    convert a set of quaternion splines to cartesian splines
    :param quat_coeffs_mat (numpy.array<3d>): n_samples * n_basis * n_dims
    :return cartesian_coeffs_mat (numpy.array<3d>): n_samples * n_basis * n_dims
    """
    semantic_motion_primitive_file = get_semantic_motion_primitive_path(elementary_action,
                                                                        motion_primitive,
                                                                        data_repo)
    quat_spline_constructor = QuatSplineConstructor(semantic_motion_primitive_file,
                                                    skeleton_json)
    quat_coeffs_mat = np.asarray(quat_coeffs_mat)
    n_samples, n_basis, n_dims = quat_coeffs_mat.shape
    cartesian_coeffs_mat = []
    for i in range(n_samples):
        quat_spline = quat_spline_constructor.create_quat_spline_from_functional_data(quat_coeffs_mat[i],
                                                                                      knots)
        cartesian_spline = quat_spline.to_cartesian()
        cartesian_coeffs_mat.append(cartesian_spline.coeffs)
    return np.asarray(cartesian_coeffs_mat)