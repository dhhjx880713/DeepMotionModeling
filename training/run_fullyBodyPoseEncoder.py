import os
import sys

import numpy as np 
import tensorflow as tf 
import logging
tf.get_logger().setLevel(logging.ERROR)
dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dirname, ".."))
from models.fullBody_pose_encoder import FullBodyPoseEncoder
from tensorflow.keras import optimizers
from utilities.utils import get_global_position_framewise, get_global_position, export_point_cloud_data_without_foot_contact
from utilities.skeleton_def import MH_CMU_SKELETON
from mosi_utils_anim.utilities import write_to_json_file
from preprocessing.preprocessing import process_bvhfile, process_file
MSE = tf.keras.losses.MeanSquaredError()
EPS = 1e-6
BUFFER_SIZE = 60000


def get_training_data(name='h36m', data_type='angle'):
    data_path = os.path.join(dirname, r'../..', r'data\training_data\processed_mocap_data', name)
    filename = '_'.join([name, data_type]) + '.npy'
    if not os.path.isfile(os.path.join(data_path, filename)):
        print("Cannot find " + os.path.join(data_path, filename))
    else:

        motion_data = np.load(os.path.join(data_path, filename))
        return motion_data


def run_fullyBodyPoseEncoder():
    ### angular representation
    h36m_data = get_training_data()
    h36m_data = np.reshape(h36m_data, (h36m_data.shape[0] * h36m_data.shape[1], h36m_data.shape[2]))
    ###
    mean_value = h36m_data.mean(axis=0)[np.newaxis, :]
    std_value = h36m_data.std(axis=0)[np.newaxis, :]
    std_value[std_value<EPS] = EPS

    normalized_data = (h36m_data - mean_value) / std_value    
    # n_frames = len(normalized_data)
    n_frames, output_dim = normalized_data.shape
    batchsize = 256
    n_batches = n_frames // batchsize

    ### reconstruct training data
    starting_frame = 1000
    frame_length = 1000
    scale_factor = 5

    ### reconstruct bvh file
    test_file = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\h36m\S1\Walking.bvh'
    input_data = process_bvhfile(test_file, sliding_window=False)
    normalized_input_data = (input_data - mean_value) / std_value

    ### meta parameters
    batchsize = 256
    dropout_rate = 0.1
    name = "h36m_fullyBodyPoseEncoder_customized_loss"
    
    ### load model
    encoder = FullBodyPoseEncoder(output_dim=output_dim, dropout_rate=dropout_rate)
    model_file = r'E:\results\h36m_fullyBodyPoseEncoder_customized_loss\h36m_fullyBodyPoseEncoder_customized_loss_1000.ckpt'
    encoder.load_weights(model_file)

    # res = encoder(normalized_data[starting_frame : starting_frame + frame_length])
    res = encoder(normalized_input_data).numpy()

    ### reconstruct motion
    reconstructed_motion = res * std_value + mean_value
    # reconstructed_motion = reconstructed_motion.numpy()
    reconstructed_global_position = get_global_position_framewise(reconstructed_motion)
    reconstructed_global_position *= scale_factor
    reconstructed_motion_data = {'motion_data': reconstructed_global_position.tolist(), 'has_skeleton': True,
                                 'skeleton': MH_CMU_SKELETON}    
    write_to_json_file(r'E:\tmp\recon_fullyBodyPoseEncoder_walk.panim', reconstructed_motion_data)

    ### reconstruct orginal motion
    reconstructed_origin_motion = normalized_input_data * std_value + mean_value

    reconstructed_global_position = get_global_position_framewise(reconstructed_origin_motion)
    reconstructed_global_position *= scale_factor
    reconstructed_motion_data = {'motion_data': reconstructed_global_position.tolist(), 'has_skeleton': True,
                                 'skeleton': MH_CMU_SKELETON}    
    write_to_json_file(r'E:\tmp\recon_fullyBodyPoseEncoder_origin.panim', reconstructed_motion_data)


if __name__ == "__main__":
    run_fullyBodyPoseEncoder()