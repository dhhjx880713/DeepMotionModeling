import os
import sys
import numpy as np 
import tensorflow as tf 
dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dirname, ".."))
from models.frame_encoder import FrameEncoder, FrameEncoderDropoutFirst, FrameEncoderNoGlobal
from tensorflow.keras import optimizers
from utilities.utils import get_global_position_framewise_tf, convert_anim_to_point_cloud_tf
MSE = tf.keras.losses.MeanSquaredError()
EPS = 1e-6


def get_training_data(name='h36m', data_type='angle'):
    data_path = os.path.join(dirname, r'../..', r'data\training_data\processed_mocap_data', name)
    filename = '_'.join([name, data_type]) + '.npy'
    if not os.path.isfile(os.path.join(data_path, filename)):
        print("Cannot find " + os.path.join(data_path, filename))
    else:

        motion_data = np.load(os.path.join(data_path, filename))
        return motion_data


def train_FrameEncoder_dropout_first():
    ### load training data
    h36m_data = get_training_data(name='h36m', data_type='quaternion')
    h36m_data = np.reshape(h36m_data, (h36m_data.shape[0] * h36m_data.shape[1], h36m_data.shape[2]))
    ### normalize data
    mean_value = h36m_data.mean(axis=0)[np.newaxis, :]
    std_value = h36m_data.std(axis=0)[np.newaxis, :]
    std_value[std_value<EPS] = EPS    
    normalized_data = (h36m_data - mean_value) / std_value
    ### meta parameters
    batchsize = 64
    epochs = 100
    dropout_rate = 0.1
    name = "h36m_frameEncDropoutFirst_quat"
    filename = name + "_{epoch:04d}.ckpt"
    # checkpoint_path = os.path.join(dirname, '../..', r'data/models', name, filename)
    checkpoint_path = os.path.join(r'E:\results', name, filename)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=10)
    ### initialize model
    encoder = FrameEncoderDropoutFirst(dropout_rate=dropout_rate)
    encoder.compile('adam', 'mse')
    encoder.fit(normalized_data, normalized_data, batch_size=batchsize, epochs=epochs, callbacks=[cp_callback])


def train_FrameEncoder():
    ### load training data
    h36m_data = get_training_data(name='h36m', data_type='quaternion')
    # h36m_data = get_training_data(name='h36m', data_type='angle')
    print(h36m_data.shape)
    h36m_data = np.reshape(h36m_data, (h36m_data.shape[0] * h36m_data.shape[1], h36m_data.shape[2]))
    ### normalize data
    mean_value = h36m_data.mean(axis=0)[np.newaxis, :]
    std_value = h36m_data.std(axis=0)[np.newaxis, :]
    std_value[std_value<EPS] = EPS

    normalized_data = (h36m_data - mean_value) / std_value
    ### meta parameters
    batchsize = 256
    epochs = 100
    dropout_rate = 0.1
    learning_rate = 1e-4
    name = "h36m_frameEnc_quat"
    filename = name + "_{epoch:04d}.ckpt"
    checkpoint_path = os.path.join(dirname, '../..', r'data/models', name, filename)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=10)
    ### initialize model
    encoder = FrameEncoder(dropout_rate=dropout_rate)
    encoder.compile(optimizer=optimizers.Adam(learning_rate), loss='mse')
    encoder.fit(normalized_data, normalized_data, batch_size=batchsize, epochs=epochs, callbacks=[cp_callback])


def train_frameEncoder_no_global_trans():
    """ train a frame encoder only including global position of each joint
    """
    h36m_data = get_training_data(name='h36m', data_type='quaternion')
    h36m_data = np.reshape(h36m_data, (h36m_data.shape[0] * h36m_data.shape[1], h36m_data.shape[2]))
  
    ### normalize data
    mean_value = h36m_data.mean(axis=0)[np.newaxis, :]
    std_value = h36m_data.std(axis=0)[np.newaxis, :]
    std_value[std_value<EPS] = EPS    
    normalized_data = (h36m_data - mean_value) / std_value 
    training_data = normalized_data[:, :-3]
    frame_encoder = FrameEncoderNoGlobal(dropout_rate=0.1)
    learning_rate = 1e-4
    epochs = 100
    batchsize = 64
    name = "h36m_FrameEncNoGlobal"
    frame_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                          loss='mse',
                          metrics=['accuracy'])
    # checkpoint_path = r'../../data/models/frame_encoder_no_global_trans/frame_encoder' + '-' + str(epochs) + '-' + str(learning_rate) + '-{epoch:04d}.ckpt'
    # checkpoint_path = os.path.join(dirname, '../..', 'data/models', name, name + '-{epoch:04d}.ckpt')
    checkpoint_path = os.path.join(dirname, '../..', 'data/models', name, '{epoch:04d}.ckpt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=10)
    # frame_encoder.save_weights(checkpoint_path.format(epoch=0))
    frame_encoder.fit(training_data, training_data, epochs=epochs, batch_size=batchsize, callbacks=[cp_callback])



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


def custom_loss_global(y_actual, y_pred):
    target_motion = convert_anim_to_point_cloud_tf(y_actual)
    predicted_motion = convert_anim_to_point_cloud_tf(y_pred)
    loss = MSE(target_motion, predicted_motion)
    return loss


def train_FrameEncoder_customized_loss():
    ### load training data
    h36m_data = get_training_data()
    print(h36m_data.shape)
    h36m_data = np.reshape(h36m_data, (h36m_data.shape[0] * h36m_data.shape[1], h36m_data.shape[2]))
    ### normalize data
    mean_value = h36m_data.mean(axis=0)[np.newaxis, :]
    std_value = h36m_data.std(axis=0)[np.newaxis, :]
    std_value[std_value<EPS] = EPS

    normalized_data = (h36m_data - mean_value) / std_value
    ### meta parameters
    batchsize = 64
    epochs = 1000
    dropout_rate = 0.1
    learning_rate = 1e-5
    name = "h36m_frameEncoder_customized_loss_global"
    # name = "h36m_frameEncoder_customized_loss_finetuning"
    filename = name + "_{epoch:04d}.ckpt"
    # checkpoint_path = os.path.join(dirname, '../..', r'data/models', name, filename)
    checkpoint_path = os.path.join(r'E:\results', name, filename)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=100)
    ### initialize model
    encoder = FrameEncoder(dropout_rate=dropout_rate)
    # encoder.compile(optimizer=optimizers.Adam(learning_rate), loss=custom_loss)
    encoder.compile(optimizer=optimizers.Adam(learning_rate), loss=custom_loss_global)
    # encoder.load_weights(r'E:\results\h36m_frameEncoder_customized_loss\h36m_frameEncoder_customized_loss_0100.ckpt')
    encoder.fit(normalized_data, normalized_data, batch_size=batchsize, epochs=epochs, callbacks=[cp_callback])


def fine_tuning_FrameEncoder_customrized_loss():
    ### load training data
    h36m_data = get_training_data()
    print(h36m_data.shape)
    h36m_data = np.reshape(h36m_data, (h36m_data.shape[0] * h36m_data.shape[1], h36m_data.shape[2]))
    ### normalize data
    mean_value = h36m_data.mean(axis=0)[np.newaxis, :]
    std_value = h36m_data.std(axis=0)[np.newaxis, :]
    std_value[std_value<EPS] = EPS    
    normalized_data = (h36m_data - mean_value) / std_value
    ### meta parameters
    batchsize = 256
    epochs = 2000
    dropout_rate = 0.1
    learning_rate = 1e-5

    name = "h36m_frameEncoder_customized_loss"
    filename = name + "_{epoch:04d}.ckpt"
    checkpoint_path = os.path.join(r'E:\results', name, filename)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=100)
    ### initialize model
    save_path = os.path.join(r'E:\results', name)
    encoder = FrameEncoder(dropout_rate=dropout_rate, name=name)
    # encoder.compile(optimizer=optimizers.Adam(learning_rate), loss=custom_loss)
    # encoder.fit(normalized_data, normalized_data, batch_size=batchsize, epochs=epochs, callbacks=[cp_callback], initial_epoch=1000)
    fine_tuning_epochs = 100
    model_file = r'E:\results\h36m_frameEncoder_customized_loss\h36m_frameEncoder_customized_loss_1000.ckpt'
    encoder.fine_tuning(model_file, normalized_data, fine_tuning_epochs, batchsize, learning_rate=learning_rate,
            save_path=save_path, save_every_epochs=10)


def test():

    encoder = FrameEncoder(dropout_rate=0.3)
    input_data = np.random.rand(100, 90)
    res = encoder._encoder(input_data)
    print(res.shape)

    decode = encoder._decoder(res)
    print(decode.shape)




if __name__ == "__main__":
    # train_FrameEncoder()
    # train_FrameEncoder_customized_loss()
    # train_FrameEncoder_dropout_first()
    # fine_tuning_FrameEncoder_customrized_loss()
    train_frameEncoder_no_global_trans()
