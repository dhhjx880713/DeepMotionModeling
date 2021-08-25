
import os
import sys
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(dirname, ".."))
from models.conv_autoencoder import ConvAutoEncoder
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
EPS = 1e-6


def train_ConvAutoEncoder():
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
    epochs = 100
    dropout_rate = 0.1
    kernel_size = 25
    encode_activation = tf.nn.elu
    decode_activation = tf.nn.elu
    name = "dancing_ClipConvEnc"
    # filename = name + "_{epoch:04d}.ckpt"
    filename = name + '.ckpt'
    checkpoint_path = os.path.join(dirname, "../..", r'data/models', name)  
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    save_filename = os.path.join(checkpoint_path, filename)
    hidden_units = 128
    learning_rate = 1e-4
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
    model.train(normalized_data, n_epochs=epochs, learning_rate=learning_rate)
    model.save_model(save_filename)


if __name__ == "__main__":
    train_ConvAutoEncoder()