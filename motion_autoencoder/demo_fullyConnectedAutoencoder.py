import os
import sys
import numpy as np
from pathlib import Path
import tensorflow as tf
import sklearn.decomposition.pca as sk_pca
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from models.fully_connected_autoencoder import FullyConnectedEncoder
from models.cnn_autoencoder import CNNAutoEncoder
from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder
from mosi_utils_anim.animation_data.utils import convert_euler_frames_to_cartesian_frames
from preprocessing.utils import sliding_window, combine_motion_clips



def demo_fullyConnectedEncoder():
    input_data = np.load(r'./data/training_data/cmu_skeleton/ulm.npz')['clips']
    reshaped_input_data = np.reshape(input_data, (input_data.shape[0], input_data.shape[1], np.prod(input_data.shape[2:])))
    n_samples, n_frames, n_dims = reshaped_input_data.shape
    Xmean = reshaped_input_data.mean(axis=1).mean(axis=0)[np.newaxis, np.newaxis, :]
    Xstd = np.array([[[reshaped_input_data.std()]]]).repeat(n_dims, axis=2)
    normalized_data = (reshaped_input_data - Xmean) / Xstd

    input_2d = np.reshape(normalized_data, (normalized_data.shape[0], np.prod(normalized_data.shape[1:])))
    
    learning_rate = 0.01
    n_epochs = 1000
    print(input_2d.shape)
    encoder = FullyConnectedEncoder(npc=10, input_dim=input_2d.shape[1], name='fullyConnectedAutoencoder',
    encoder_activation=tf.nn.tanh, decoder_activation=None, logging=True)
    encoder.create_model()
    # encoder.create_model_2layer()

    # encoder.train(input_2d, n_epochs=n_epochs, learning_rate=learning_rate, pre_train=True)
    save_folder = r'./data/experiment_results'
    filename = '_'.join(['fullyConnectedEncoder_pretrain', str(learning_rate), str(n_epochs)])
    # encoder.save_model(os.path.join(save_folder, filename))
    encoder.load_model(os.path.join(save_folder, filename))
    backprojection = encoder(input_2d)
    print(backprojection.shape)
    pca_error = np.mean((input_2d - backprojection)**2)
    print('pca reconstruction error for normalized data is: ', pca_error)  




def pca_evaluation():
    ## load data
    str(Path(__file__).parent.absolute())
    data_path = r'./data/training_data/cmu_skeleton'
    # datasets = ['h36m', 'pfnn', 'stylistic_raw', 'ulm']
    datasets = ['h36m', 'ACCAD']
    training_data = None
    save_folder = r'./data/experiment_results'
    for dataset in datasets:
        if training_data is None:
            training_data = np.load(os.path.join(data_path, dataset + '.npz'))['clips']
        else:
            training_data = np.concatenate([training_data, np.load(os.path.join(data_path, dataset + '.npz'))['clips']], axis=0)
    reshaped_input_data = np.reshape(training_data, (training_data.shape[0], training_data.shape[1], np.prod(training_data.shape[2:])))
    n_samples, n_frames, n_dims = reshaped_input_data.shape
    Xmean = reshaped_input_data.mean(axis=1).mean(axis=0)[np.newaxis, np.newaxis, :]
    Xstd = np.array([[[reshaped_input_data.std()]]]).repeat(n_dims, axis=2)

    normalized_data = (reshaped_input_data - Xmean) / Xstd

    input_2d = np.reshape(normalized_data, (normalized_data.shape[0], np.prod(normalized_data.shape[1:])))

    max_value = np.max(input_2d)
    min_value = np.min(input_2d)
    print('data range is min: %.5f  max: %.5f. ' % (min_value, max_value))
    npc = 10
    pca = sk_pca.PCA(n_components=npc)
    projection = pca.fit_transform(input_2d)
    print(projection.shape)
    backprojection = pca.inverse_transform(projection)
    pca_error = np.mean((input_2d - backprojection)**2)
    print('pca reconstruction error for normalized data is: ', pca_error)   
    animated_joint_list = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LowerBack', 'Spine', 'Spine1', 'LeftShoulder',
    'LeftArm', 'LeftForeArm', 'LeftHand', 'LThumb', 'LeftFingerBase', 'LeftHandFinger1', 'Neck', 'Neck1', 'Head', 'RightShoulder',
    'RightArm', 'RightForeArm', 'RightHand', 'RThumb', 'RightFingerBase', 'RightHandFinger1', 'RightUpLeg', 'RightLeg', 'RightFoot',
    'RightToeBase']
    ### export motions 
    test_file = r'E:\workspace\projects\cGAN\processed_data\ACCAD\Male1_bvh_Male1_A10_LieToCrouch.bvh'
    bvhreader = BVHReader(test_file)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    cartesian_frames = convert_euler_frames_to_cartesian_frames(skeleton, bvhreader.frames, animated_joint_list)
    print(cartesian_frames.shape)
    motion_clips = np.asarray(sliding_window(cartesian_frames, window_size=60))
    print(motion_clips.shape)
    motion_clips = np.reshape(motion_clips, (motion_clips.shape[0], motion_clips.shape[1], np.prod(motion_clips.shape[2:])))
    normalized_motion_clips = (motion_clips - Xmean) / Xstd
    print(normalized_motion_clips.shape)
    motion_clips_2d = np.reshape(normalized_motion_clips, (normalized_motion_clips.shape[0], np.prod(normalized_motion_clips.shape[1:])))
    motion_clips_projection = pca.fit_transform(motion_clips_2d)

    reconstruction = pca.inverse_transform(motion_clips_projection)
    print(reconstruction.shape)
    reconstructed_motion_clips = np.reshape(reconstruction, normalized_motion_clips.shape)
    reconstructed_motion = combine_motion_clips(reconstructed_motion_clips, len(cartesian_frames), 30)


if __name__ == "__main__":
    demo_fullyConnectedEncoder()
    # pca_evaluation()