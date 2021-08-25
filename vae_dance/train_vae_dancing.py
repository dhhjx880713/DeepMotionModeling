import os
import sys
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, '../..'))
import numpy as np
import tensorflow as tf
from mosi_dev_deepmotionmodeling.models.motion_vae_encoder import MotionVaeEncoder
tf.compat.v1.disable_eager_execution()


def train_vae():
    data_path = os.path.join(dirname, '../..', 'data/training_data/dancing/encoded_data.npy')
    if not os.path.exists(data_path):
        training_data = create_training_data()
        np.save(data_path, training_data)
    else:
        training_data = np.load(data_path)

    EPS = 1e-6
    npc = 32

    reshaped_motion_data = training_data

    input_dims = training_data.shape[1]

    model_name = 'dance_vae'
    n_epochs = 300
    batchsize = 32
    learning_rate = 1e-4
    model = MotionVaeEncoder(npc=npc, input_dims=input_dims, name=model_name,
                             encoder_activation=tf.nn.elu,
                             decoder_activation=tf.nn.elu,
                             n_random_samples=1)

    save_path = os.path.join(dirname, '../..', 'data/models', model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_filename = os.path.join(save_path, model_name + '.ckpt')
    model.create_model()
    model.train(reshaped_motion_data, learning_rate=learning_rate, n_epochs=n_epochs, batchsize=batchsize)
    model.save_model(save_filename)


if __name__ == "__main__":
    train_vae()