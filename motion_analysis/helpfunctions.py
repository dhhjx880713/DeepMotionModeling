# encoding: UTF-8
from morphablegraphs.utilities import load_aligned_data
from morphablegraphs.construction.construction_algorithm_configuration import ConstructionAlgorithmConfigurationBuilder
from morphablegraphs.animation_data import BVHReader, Skeleton
from morphablegraphs.animation_data.utils import convert_quaternion_frames_to_cartesian_frames
from morphablegraphs.construction.fpca import MotionDimensionReduction
import numpy as np
from morphablegraphs.external.PCA import *


def get_smoothed_quat_frames(elementary_action,
                             motion_primitive,
                             scale_root=False):
    aligned_motion_data = load_aligned_data(elementary_action, motion_primitive)
    params = ConstructionAlgorithmConfigurationBuilder(elementary_action, motion_primitive)
    skeleton_file = r'../../skeleton.bvh'
    bvhreader = BVHReader(skeleton_file)
    dimension_reductor = MotionDimensionReduction(aligned_motion_data,
                                                  bvhreader,
                                                  params)
    dimension_reductor.convert_euler_to_quat()
    if scale_root:
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


def standardPCA(input_data, fraction=0.9):
    '''
    Apply standard PCA on motion data

    Parameters
    ----------
    * data: 2d array
    \tThe data matrix contains motion data, which should be a matrix with
    shape n_sample * n_dims
    '''
    input_data = np.asarray(input_data)
    assert len(input_data.shape) == 2, ('Data matrix should be a 2d array')
    n_samples, n_dims = input_data.shape
    # print("dimension of data is: ", str(n_dims))
    centerobj = Center(input_data)
    pcaobj = PCA(input_data, fraction=fraction)
    print 'number of principal for standard PCA is:' + str(pcaobj.npc)
    return pcaobj, centerobj


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


def cartesian_error_measure(original_motions, reconstructed_motions, skeleton, weighted_error=True):
    """
    Measure reconstruction error after dimension reduction in Cartesian space
    :param original_motions (numpy.array<3d>): n_samples * n_frames * n_dims, quaternion representation of original
                                               data
    :param reconstructed_motions (numpy.array<3d>): n_samples * n_frames * n_dims, quaternion representation of
                                                    reconstructed data
    :return float: reconstruction error
    """
    assert original_motions.shape == reconstructed_motions.shape, \
        ('The shape of original data and reconstructed data should be the same!')
    err = 0
    n_samples = original_motions.shape[0]
    joint_weights = skeleton.joint_weights[:-2]
    for i in range(n_samples):
        original_cartesian_motion = convert_quaternion_frames_to_cartesian_frames(skeleton, original_motions[i])
        reconstructed_cartesian_motion = convert_quaternion_frames_to_cartesian_frames(skeleton,
                                                                                       reconstructed_motions[i])
        err += cartesian_motion_distance(original_cartesian_motion, reconstructed_cartesian_motion, joint_weights,
                                         weighted_error)
    return err/n_samples


def cartesian_motion_distance(cartesian_motion_a, cartesian_motion_b, joint_weights, weighted_error=True):
    """
    Measure distance between two motions which are represented by cartesian points
    :param cartesian_motion_a: n_frmaes * n_joints * LEN_CARTESIAN_POINT
    :param cartesian_motion_b: n_frmaes * n_joints * LEN_CARTESIAN_POINT
    :return float: dist
    """
    # assert len(joint_weights) == 19
    cartesian_motion_a = np.asarray(cartesian_motion_a)
    cartesian_motion_b = np.asarray(cartesian_motion_b)
    assert cartesian_motion_a.shape == cartesian_motion_b.shape
    dist = 0
    n_frames, n_joints, n_dims = cartesian_motion_a.shape
    for i in range(n_frames):
        for j in range(n_joints):
            if weighted_error:
                dist += np.linalg.norm(cartesian_motion_a[i, j] - cartesian_motion_b[i, j]) * joint_weights[j]
            else:

                dist += np.linalg.norm(cartesian_motion_a[i, j] - cartesian_motion_b[i, j])
    return dist/(n_frames * n_joints)


def load_skeleton():
    skeleton_bvhfile = r'../../skeleton.bvh'
    bvhreader = BVHReader(skeleton_bvhfile)
    skeleton = Skeleton()
    skeleton.load_from_bvh(bvhreader)
    return skeleton


def load_bvhreader():
    skeleton_bvhfile = r'../../skeleton.bvh'
    return BVHReader(skeleton_bvhfile)