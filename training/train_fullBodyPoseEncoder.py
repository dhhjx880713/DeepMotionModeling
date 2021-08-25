import os
import sys
import numpy as np 
import tensorflow as tf 
dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dirname, ".."))
from models.fullBody_pose_encoder import FullBodyPoseEncoder
from tensorflow.keras import optimizers
from utilities.utils import get_global_position_framewise_tf
MSE = tf.keras.losses.MeanSquaredError()
EPS = 1e-6
BUFFER_SIZE = 60000

def custom_loss(y_actual, y_pred):
    """measure MSE in global joint position space

    Args:
        y_actual : original motion
        y_pred : predicted motion
    """
    target_motion = get_global_position_framewise_tf(y_actual)
    predicted_motion = get_global_position_framewise_tf(y_pred)
    loss = MSE(target_motion, predicted_motion)
    return loss


def get_training_data(name='h36m', data_type='angle'):
    data_path = os.path.join(dirname, r'../..', r'data\training_data\processed_mocap_data', name)
    filename = '_'.join([name, data_type]) + '.npy'
    if not os.path.isfile(os.path.join(data_path, filename)):
        print("Cannot find " + os.path.join(data_path, filename))
    else:

        motion_data = np.load(os.path.join(data_path, filename))
        return motion_data


def train_fullBodyPoseEncoder():
    ### load training data
    h36m_data = get_training_data(name='h36m', data_type='quaternion')
    h36m_data = np.reshape(h36m_data, (h36m_data.shape[0] * h36m_data.shape[1], h36m_data.shape[2]))
    ### normalize data
    n_frames, output_dim = h36m_data.shape
    mean_value = h36m_data.mean(axis=0)[np.newaxis, :]
    std_value = h36m_data.std(axis=0)[np.newaxis, :]
    std_value[std_value<EPS] = EPS    
    normalized_data = (h36m_data - mean_value) / std_value
    ### meta parameters
    batchsize = 256
    epochs = 100
    dropout_rate = 0.1
    name = "h36m_PoseEnc_quat"
    filename = name + "_{epoch:04d}.ckpt"
    checkpoint_path = os.path.join(dirname, "../..", r'data/models', name, filename)
    training_dataset = tf.data.Dataset.from_tensor_slices((normalized_data, normalized_data)).shuffle(BUFFER_SIZE).batch(batchsize)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=10)
    encoder = FullBodyPoseEncoder(output_dim=output_dim, dropout_rate=dropout_rate)
    encoder.compile('adam', 'mse')
    encoder.fit(training_dataset, epochs=100, callbacks=[cp_callback])



def train_dancing_clips():
    data_path = os.path.join(dirname, '../..', r'data/training_data/dancing/dancing_data.npy')
    training_data = np.load(data_path)
    training_data = np.reshape(training_data, (training_data.shape[0], training_data.shape[1] * training_data.shape[2]))
    n_frames, output_dim = training_data.shape
    ### normalize data
    mean_value = training_data.mean(axis=0)[np.newaxis, :]
    std_value = training_data.std(axis=0)[np.newaxis, :]
    std_value[std_value<EPS] = EPS    
    normalized_data = (training_data - mean_value) / std_value  
    ### meta parameters
    batchsize = 16
    epochs = 500
    dropout_rate = 0.1
    npc = 256
    name = "dancing_ClipEnc"
    filename = name +  '_' + str(npc) + "_{epoch:04d}.ckpt"
    checkpoint_path = os.path.join(dirname, "../..", r'data/models', name, filename)
    training_dataset = tf.data.Dataset.from_tensor_slices((normalized_data, normalized_data)).shuffle(BUFFER_SIZE).batch(batchsize)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=50)
    encoder = FullBodyPoseEncoder(output_dim=output_dim, dropout_rate=dropout_rate, npc=npc)
    encoder.compile('adam', 'mse')
    encoder.fit(training_dataset, epochs=epochs, callbacks=[cp_callback])


def train_fullBodyPoseEncoder_custom_loss():
    ### load training data
    h36m_data = get_training_data() ## custom loss only works for angular representation
    h36m_data = np.reshape(h36m_data, (h36m_data.shape[0] * h36m_data.shape[1], h36m_data.shape[2]))
    ### normalize data
    n_frames, output_dim = h36m_data.shape
    mean_value = h36m_data.mean(axis=0)[np.newaxis, :]
    std_value = h36m_data.std(axis=0)[np.newaxis, :]
    std_value[std_value<EPS] = EPS    
    normalized_data = (h36m_data - mean_value) / std_value
    ### meta parameters
    batchsize = 256
    epochs = 1000
    dropout_rate = 0.1
    learning_rate = 1e-5
    name = "h36m_fullyBodyPoseEncoder_customized_loss"        
    filename = name + "_{epoch:04d}.ckpt"
    checkpoint_path = os.path.join(r'E:\results', name, filename)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=100)
    ### initialize model    
    encoder = FullBodyPoseEncoder(output_dim=output_dim, dropout_rate=dropout_rate)
    encoder.compile(optimizer=optimizers.Adam(learning_rate), loss=custom_loss)
    # encoder.load_weights(r'E:\results\h36m_frameEncoder_customized_loss\h36m_frameEncoder_customized_loss_0100.ckpt')
    encoder.fit(normalized_data, normalized_data, batch_size=batchsize, epochs=epochs, callbacks=[cp_callback])


if __name__ == "__main__":
    # train_fullBodyPoseEncoder()
    # train_fullBodyPoseEncoder_custom_loss()
    train_dancing_clips()