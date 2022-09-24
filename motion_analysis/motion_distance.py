# encoding: UTF-8
import numpy as np
from . import LEN_JOINTS
from prepare_data import convert_quaternion_frames_to_cartesian_frames



def cartesian_error_measure(original_motions, reconstructed_motions, skeleton, weighted_error=True):
    """
    Measure reconstruction error after dimension reduction in Cartesian space
    :param original_motions (numpy.array<3d>): n_samples * n_frames * n_dims, quaternion representation of original
                                               data
    :param reconstructed_motions (numpy.array<3d>): n_samples * n_frames * n_dims, quaternion representation of
                                                    reconstructed data
    :return float: reconstruction error
    """
    assert original_motions.shape == reconstructed_motions.shape, ('The shape of original data and reconstructed data should be the same!')
    err = 0
    n_samples = original_motions.shape[0]
    joint_weights = skeleton.joint_weights[:-4]
    for i in range(n_samples):
        original_cartesian_motion = convert_quaternion_frames_to_cartesian_frames(skeleton, original_motions[i])
        reconstructed_cartesian_motion = convert_quaternion_frames_to_cartesian_frames(skeleton, reconstructed_motions[i])
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
    assert len(joint_weights) == LEN_JOINTS
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


def cartesian_distance_measure(original_motions, reconstructed_motions, skeleton):
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
        reconstructed_cartesian_motion = convert_quaternion_frames_to_cartesian_frames(skeleton, reconstructed_motions[i])
        err += cartesian_motion_distance(original_cartesian_motion, reconstructed_cartesian_motion, joint_weights)
    return err/n_samples