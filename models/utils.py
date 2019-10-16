import tensorflow as tf 


def gram_matrix(X):
    '''
    X shape: n_batches * n_dims * n_frames
    :param X:
    :return: gram_matrix n_batches * n_dims * n_dims, sum over n_frames
    '''
    return tf.reduce_mean(tf.expand_dims(X, 2) * tf.expand_dims(X, 1), axis=3)


def convert_anim_to_point_cloud_tf(anim):
    '''

    :param anim:
    :return:
    '''
    joints, v_x, v_z, v_r = anim[:, :-3], anim[:, -3], anim[:, -2], anim[:, -1]
    joints = tf.reshape(joints, (joints.shape[0], -1, 3))
    n_frames, n_joints = joints.shape[0], joints.shape[1]

    v_r = tf.reshape(tf.cumsum(v_r), (n_frames, 1))
    """ Rotate motion about Y axis """
    v_x = tf.reshape(v_x, (n_frames, 1))
    v_z = tf.reshape(v_z, (n_frames, 1))
    #### create rotation matrix
    sin_theta = tf.sin(v_r)
    cos_theta = tf.cos(v_r)
    rotmat = tf.concat((cos_theta, tf.zeros((n_frames, 1)), sin_theta, tf.zeros((n_frames, 1)), tf.zeros((n_frames, 1)),
                        tf.ones((n_frames, 1)), tf.zeros((n_frames, 1)), tf.zeros((n_frames, 1)), -sin_theta, 
                        tf.zeros((n_frames, 1)), cos_theta, tf.zeros((n_frames, 1)), tf.zeros((n_frames, 1)), 
                        tf.zeros((n_frames, 1)), tf.zeros((n_frames, 1)), tf.ones((n_frames, 1))), axis=-1)
    rotmat = tf.reshape(rotmat, (n_frames, 4, 4))

    ones = tf.ones((n_frames, n_joints, 1))
    extended_joints = tf.concat((joints, ones), axis=-1)
    swapped_joints = tf.transpose(extended_joints, (0, 2, 1))
    # print('swapped joints shape: ', swapped_joints.shape)
    rotated_joints = tf.matmul(rotmat, swapped_joints)
    rotated_joints = tf.transpose(rotated_joints, (0, 2, 1))[:, :, :-1]
    """ Rotate Velocity"""
    trans = tf.concat((v_x, tf.zeros((n_frames, 1)), v_z, tf.ones((n_frames, 1))), axis=-1)
    trans = tf.expand_dims(trans, 1)
    swapped_trans = tf.transpose(trans, (0, 2, 1))
    rotated_trans = tf.matmul(rotmat, swapped_trans)
    rotated_trans = tf.transpose(rotated_trans, (0, 2, 1))
    v_x = rotated_trans[:, :, 0]
    v_z = rotated_trans[:, :, 2]
    v_x, v_z = tf.cumsum(v_x, axis=0), tf.cumsum(v_z, axis=0)
    rotated_trans = tf.concat((v_x, tf.zeros((n_frames, 1)), v_z), axis=-1)
    rotated_trans = tf.expand_dims(rotated_trans, 1)
    export_joints = rotated_joints + rotated_trans
    return export_joints