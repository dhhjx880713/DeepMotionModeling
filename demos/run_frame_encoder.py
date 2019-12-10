import os
import sys
from pathlib import Path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from models.frame_encoder import FrameEncoder
import numpy as np 
from tensorflow.keras import Sequential, layers, Model, optimizers, losses
import tensorflow as tf 
from utilities.utils import export_point_cloud_data_without_foot_contact
from utilities.skeleton_def import MH_CMU_SKELETON
import copy


def train_frameEncoder():
    input_data = np.load(r'../../data/training_data/h36m.npz')['clips']
    frame_encoder = FrameEncoder()
    learning_rate = 1e-3
    epochs = 100
    batchsize = 256
    frame_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                          loss='mse',
                          metrics=['accuracy'])
    checkpoint_path = r'../../data/models/frame_encoder-{epoch:04d}.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=10)
    frame_encoder.save_weights(checkpoint_path.format(epoch=0))
    frame_encoder.fit(input_data, input_data[:, :-3], epochs=epochs, batch_size=batchsize, callbacks=[cp_callback])


def run_frameEncoder():
    frame_encoder = FrameEncoder()

    input_data = np.load(r'../../data/training_data/h36m.npz')['clips']
    learning_rate = 1e-3
    epochs = 100
    batchsize = 256
    frame_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                          loss='mse',
                          metrics=['accuracy'])
    frame_encoder.build(input_shape=input_data.shape)

    checkpoint_path = r'../../data/models/frame_encoder-{epoch:04d}.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    frame_encoder.load_weights(latest)

    
    frame_num = 1000  

    ref = input_data[:frame_num, :]
    res = copy.deepcopy(ref)
    
    res[:frame_num, :87] = frame_encoder(input_data[:frame_num, :])
    export_point_cloud_data_without_foot_contact(ref, filename=r'../../data/test_data/ref.panim', skeleton=MH_CMU_SKELETON)
    export_point_cloud_data_without_foot_contact(res, filename=r'../../data/test_data/reconstructed.panim', skeleton=MH_CMU_SKELETON)


if __name__ == "__main__":
    run_frameEncoder()
    # train_frameEncoder())
