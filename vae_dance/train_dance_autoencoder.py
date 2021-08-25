import os
import sys
import numpy as np 
import tensorflow as tf 
dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dirname, ".."))
from models.fullBody_pose_encoder import FullBodyPoseEncoder
from mosi_dev_deepmotionmodeling.preprocessing.preprocessing import process_file
import glob
EPS = 1e-6
BUFFER_SIZE = 60000


bad_samples = [
    'd04_mBR0_ch10'
    'd04_mBR4_ch07',
    'd05_mBR5_ch14',
    'd06_mBR4_ch20',
    'd20_mHO5_ch13',
    'd07_mJB2_ch10',
    'd07_mJB3_ch05',
    'd07_mJB3_ch10',
    'd08_mJB0_ch09',
    'd08_mJB1_ch09',
    'd09_mJB2_ch07',
    'd09_mJB4_ch09',
    'd09_mJB4_ch10',
    'd09_mJB2_ch17',
    'd09_mJS3_ch10',
    'd01_mJS0_ch01',
    'd01_mJS1_ch02',
    'd02_mJS0_ch08',
    'd26_mWA3_ch01',
    'd27_mWA4_ch08',
    'd27_mWA5_ch01',
    'd27_mWA5_ch08',
    'd26_mWA0_ch08'
]

def check_filename(filename):
    is_bad_file = False
    for token in bad_samples:
        if token in filename:
            is_bad_file = True
    return is_bad_file


def create_training_data():
    data_folder = r'E:\workspace\mocap_data\AIST\retargeting'
    bvhfiles = glob.glob(os.path.join(data_folder, '*.bvh'))
    ### get joint positions
    window_size = 240
    motion_clips = []
    for bvhfile in bvhfiles:
        motion_clips.extend(process_file(bvhfile, window=window_size, window_step=window_size // 2))
    motion_clips = np.stack(motion_clips, axis=0)
    np.save(os.path.join(dirname, '../..', r'data/training_data/dancing/dancing_data.npy'), motion_clips)
    return motion_clips


def train_dancing_autoencoder():
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


if __name__ == "__main__":
    train_dancing_autoencoder()