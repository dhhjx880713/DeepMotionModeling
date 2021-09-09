import os
import sys

from numpy import linalg
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, '..'))
from models.conv_autoencoder import ConvAutoEncoder
from utilities.utils import export_point_cloud_data_without_foot_contact

import numpy as np
import tensorflow as tf
from mosi_utils_anim.animation_data.utils import align_point_clouds_2D
from utilities.utils import average_euclidean_distance
tf.compat.v1.disable_eager_execution()
EPS = 1e-6


def run_ConvAutoEncoder():
    ## load training data
    data_path = os.path.join(dirname, '../..', r'data/training_data/dancing/dancing_data.npy')
    training_data = np.load(data_path)
    # training_data = np.reshape(training_data, (training_data.shape[0], training_data.shape[1] * training_data.shape[2]))
    # n_frames, output_dim = training_data.shape
    n_samples, n_frames, n_dims = training_data.shape
    training_data = np.swapaxes(training_data, 1, 2)

    ### normalize data

    mean = training_data.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
    std = np.array([[[training_data.std()]]]).repeat(training_data.shape[1], axis=1)
    normalized_data = (training_data - mean) / std
    ### meta parameters
    batchsize = 32

    dropout_rate = 0.1
    kernel_size = 25
    encode_activation = tf.nn.elu
    decode_activation = tf.nn.elu
    name = "dancing_ClipConvEnc"
    # filename = name + "_{epoch:04d}.ckpt"
    filename = name + '.ckpt'
    checkpoint_path = os.path.join(dirname, "../..", r'data/models', name)  
    save_filename = os.path.join(checkpoint_path, filename)
    hidden_units = 128
    ### initial model
    model = ConvAutoEncoder(name=name, 
                            n_frames=n_frames,
                            n_dims=n_dims,
                            kernel_size=kernel_size,
                            encode_activation=encode_activation,
                            decode_activation=decode_activation,
                            hidden_units=hidden_units,
                            pooling='average',
                            dropout_rate=dropout_rate,
                            batch_norm=False)
    model.create_model(batchsize=batchsize)                        
    model.load_model(save_filename)

    ### load test data
    N = 100
    res = model(normalized_data)
    reconstructed_motions = res * std + mean
    reconstructed_motions = np.swapaxes(reconstructed_motions, 1, 2)
    print(reconstructed_motions.shape)
    # output_path = os.path.join(dirname, '../..', r'data/results/generated_motions', name)
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    # for i in range(N):
    #     export_point_cloud_data_without_foot_contact(reconstructed_motions[i], filename=os.path.join(output_path, str(i) + '.panim'), scale_factor=5)
    
    motion_3d = []
    training_data_3d = []
    training_data = np.swapaxes(training_data, 1, 2)
    for i in range(len(reconstructed_motions)):
        motion_3d.append(export_point_cloud_data_without_foot_contact(reconstructed_motions[i]))
        training_data_3d.append(export_point_cloud_data_without_foot_contact(training_data[i]))
    motion_3d = np.asarray(motion_3d)
    training_data_3d = np.asarray(training_data_3d)
    print(motion_3d.shape)
    err = average_euclidean_distance(motion_3d, training_data_3d)
    print("mean squared error is: {}".format(err))




if __name__ == "__main__":
    run_ConvAutoEncoder()