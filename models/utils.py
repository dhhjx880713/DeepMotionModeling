import tensorflow as tf 


def gram_matrix(X):
    '''
    X shape: n_batches * n_dims * n_frames
    :param X:
    :return: gram_matrix n_batches * n_dims * n_dims, sum over n_frames
    '''
    return tf.reduce_mean(tf.expand_dims(X, 2) * tf.expand_dims(X, 1), axis=3)