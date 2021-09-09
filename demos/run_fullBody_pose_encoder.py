import os
import sys
from pathlib import Path
abspath = os.path.abspath(__file__)
dirname = os.path.dirname(abspath)
os.chdir(dirname)
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from models.fullBody_pose_encoder import FullBodyPoseEncoder
import numpy as np 
from tensorflow.keras import Sequential, layers, Model, optimizers, losses
import tensorflow as tf 
from utilities.utils import export_point_cloud_data_without_foot_contact, average_euclidean_distance
from utilities.skeleton_def import MH_CMU_SKELETON
import time
import copy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA
EPS = 1e-6



def train_fullBody_pose_encoder():
    training_data_dir = r'../../data/training_data/h36m.npz'
    input_data = np.load(training_data_dir)
    assert 'clips' in input_data.keys(), "cannot find motion data in " + training_data_dir
    motion_data = input_data['clips']
    ## normalize input data
    if 'mean' not in input_data.keys():
        mean_value = motion_data.mean(axis=0)[np.newaxis, :]
    else:
        mean_value = input_data['mean']
    if 'std' not in input_data.keys():
        std_value = motion_data.std(axis=0)[np.newaxis, :]
        std_value[std_value < EPS] = EPS
    else:
        std_value = input_data['std']
    normalized_data = (motion_data - mean_value) / std_value
    learning_rate = 1e-3
    epochs = 100
    batchsize = 256

    pose_encoder = FullBodyPoseEncoder(output_dim=normalized_data.shape[1])
    learning_rate = 1e-3
    epochs = 100
    batchsize = 256
    pose_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                         loss='mse',
                         metrics=['accuracy'])     
    checkpoint_path = r'../../data/models/fb_pose_encoder/pose_encoder' + '-' + str(epochs) + '-' + str(learning_rate) + '-{epoch:04d}.ckpt'

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=10)
    pose_encoder.save_weights(checkpoint_path.format(epoch=0))
    pose_encoder.fit(normalized_data, normalized_data, epochs=epochs, batch_size=batchsize, callbacks=[cp_callback])                              


def run_fullBody_pose_encoder():
    training_data_dir = r'../../data/training_data/h36m.npz'
    input_data = np.load(training_data_dir)
    assert 'clips' in input_data.keys(), "cannot find motion data in " + training_data_dir
    motion_data = input_data['clips']
    ## normalize input data
    if 'mean' not in input_data.keys():
        mean_value = motion_data.mean(axis=0)[np.newaxis, :]
    else:
        mean_value = input_data['mean']
    if 'std' not in input_data.keys():
        std_value = motion_data.std(axis=0)[np.newaxis, :]
        std_value[std_value < EPS] = EPS
    else:
        std_value = input_data['std']
    normalized_data = (motion_data - mean_value) / std_value
    learning_rate = 1e-3
    epochs = 100
    batchsize = 256
    scale_factor = 5
    pose_encoder = FullBodyPoseEncoder(output_dim=normalized_data.shape[1])    
    pose_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                         loss='mse',
                         metrics=['accuracy'])
    pose_encoder.build(input_shape=motion_data.shape)              

    ### load model

    checkpoint_path = r'../../data/models/fb_pose_encoder/pose_encoder' + '-' + str(epochs) + '-' + str(learning_rate) + '-{epoch:04d}.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(latest)
    pose_encoder.load_weights(latest)

    frame_num = 1000
    ref = motion_data[:frame_num, :]
    encodered_data = np.asarray(pose_encoder.encode(normalized_data[:frame_num, :]))
    print(encodered_data.shape)
    res = np.asarray(pose_encoder(normalized_data[:frame_num, :]))
    res = res * std_value + mean_value   
    output_folder = r'../../data/test_data/fullBody_pose_encoder'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder) 
    timestr = time.strftime("%Y%m%d-%H%M%S")
    export_point_cloud_data_without_foot_contact(ref, filename= os.path.join(output_folder, 'ref_' + timestr + '.panim'), skeleton=MH_CMU_SKELETON, scale_factor=scale_factor)
    export_point_cloud_data_without_foot_contact(res, filename= os.path.join(output_folder, 'reconstructed_' + timestr + '.panim'), skeleton=MH_CMU_SKELETON, scale_factor=scale_factor)

    # ## visualize motion sequence in manifold
    # # manifold_embedded = TSNE(n_components=3).fit_transform(encodered_data)
    # manifold_embedded = PCA(n_components=3).fit_transform(encodered_data)
    # print(manifold_embedded.shape)

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(manifold_embedded[:, 0], manifold_embedded[:, 1], manifold_embedded[:, 2])

    # plt.show()


    #### quantative evaluation 
    frames = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    ref_frames = export_point_cloud_data_without_foot_contact(ref)
    res_frames = export_point_cloud_data_without_foot_contact(res)
    errors = []
    for frame_index in frames:
        mse = np.sum((ref_frames[:frame_index] - res_frames[:frame_index]) ** 2) / frame_index
        errors.append(mse)
    fig = plt.figure()
    plt.plot(range(len(errors)), errors)
    plt.show()



def run_fullyBodyEncoder_dancing_clips():
    ### load training data to calculate meta information
    training_data_path = os.path.join(dirname, '../..', r'data/training_data/dancing/dancing_data.npy')
    training_data = np.load(training_data_path)
    n_samples, n_frames, n_dims = training_data.shape
    training_data = np.reshape(training_data, (training_data.shape[0], training_data.shape[1] * training_data.shape[2]))
    output_dim = training_data.shape[1]
    dropout_rate = 0.1
    npc = 128
    mean_value = training_data.mean(axis=0)[np.newaxis, :]
    std_value = training_data.std(axis=0)[np.newaxis, :]
    std_value[std_value<EPS] = EPS    
    normalized_data = (training_data - mean_value) / std_value  

    ### load model
    model_name = r'D:\workspace\my_git_repos\vae_motion_modeling\data\models\dancing_ClipEnc\dancing_ClipEnc_128_0500.ckpt'
    model = FullBodyPoseEncoder(output_dim=output_dim, dropout_rate=dropout_rate, npc=npc)
    model.load_weights(model_name)

    encoded_motion = model.encode(normalized_data).numpy()
    print(encoded_motion.shape)
    
    ### save encoded data

    # np.save(os.path.join(os.path.dirname(training_data_path), 'encoded_data.npy'), encoded_motion)

    ### load test data
    # res = model(normalized_data[:100])
    # reconstructed_motion = res * std_value + mean_value
    # reconstructed_motion = np.reshape(reconstructed_motion, (100, 120, 90))
    # save_path = r'E:\tmp'
    # for i in range(len(reconstructed_motion)):
    #     save_filename = os.path.join(save_path, 'dancing_' + str(i) + '.panim')
    #     export_point_cloud_data_without_foot_contact(reconstructed_motion[i], save_filename, scale_factor=5)

    # ### quantitative evaluation
    # res = model(normalized_data)
    # reconstructed_motion = res * std_value + mean_value
    # print(reconstructed_motion.shape)
    # reconstructed_motion = np.reshape(reconstructed_motion, (n_samples, n_frames, n_dims))
    # training_data = np.reshape(training_data, (n_samples, n_frames, n_dims))
    # motion_3d = []
    # training_data_3d = []
    # for i in range(len(reconstructed_motion)):
    #     motion_3d.append(export_point_cloud_data_without_foot_contact(reconstructed_motion[i]))
    #     training_data_3d.append(export_point_cloud_data_without_foot_contact(training_data[i]))
    # motion_3d = np.asarray(motion_3d)
    # training_data_3d = np.asarray(training_data_3d)
    # err = average_euclidean_distance(motion_3d, training_data_3d)
    # print("average euclidean error per joint is {}".format(err))



def decode_data():
    ## load data 
    data_file = r'D:\workspace\my_git_repos\vae_motion_modeling\data\training_data\dancing\sampled_motions.npy'
    encoded_data = np.load(data_file)
    print(encoded_data.shape)

    ## load model
    ### load training data to calculate meta information
    training_data_path = r'D:\workspace\my_git_repos\vae_motion_modeling\data\training_data\dancing\dancing_data.npy'
    training_data = np.load(training_data_path)
    training_data = np.reshape(training_data, (training_data.shape[0], training_data.shape[1] * training_data.shape[2]))
    output_dim = training_data.shape[1]
    dropout_rate = 0.1

    mean_value = training_data.mean(axis=0)[np.newaxis, :]
    std_value = training_data.std(axis=0)[np.newaxis, :]
    std_value[std_value<EPS] = EPS    


    ### load model
    model_name = r'D:\workspace\my_git_repos\vae_motion_modeling\data\models\dancing_ClipEnc\dancing_ClipEnc_0150.ckpt'
    model = FullBodyPoseEncoder(output_dim=output_dim, dropout_rate=dropout_rate, npc=256)
    model.load_weights(model_name)    

    res = model.decode(encoded_data)
    reconstructed_motion = res * std_value + mean_value
    reconstructed_motion = np.reshape(reconstructed_motion, (100, 120, 90))

    save_path = r'D:\workspace\my_git_repos\vae_motion_modeling\data\tmp\dancing'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(len(reconstructed_motion)):
        save_filename = os.path.join(save_path, 'dancing_' + str(i) + '.panim')
        export_point_cloud_data_without_foot_contact(reconstructed_motion[i], save_filename, scale_factor=5)


if __name__ == "__main__":
    # train_fullBody_pose_encoder()
    # run_fullBody_pose_encoder()
    run_fullyBodyEncoder_dancing_clips()
    # decode_data()