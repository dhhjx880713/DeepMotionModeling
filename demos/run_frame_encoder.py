import os
import sys
from pathlib import Path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from models.frame_encoder import FrameEncoder
from models.frame_encoder import FrameEncoderNoGlobal
import numpy as np 
from tensorflow.keras import Sequential, layers, Model, optimizers, losses
import tensorflow as tf 
from utilities.utils import export_point_cloud_data_without_foot_contact
from utilities.skeleton_def import MH_CMU_SKELETON
import copy
import time
from mosi_utils_anim.utilities import write_to_json_file
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
EPS = 1e-6


def train_frameEncoder_no_global_trans():
    """ train a frame encoder only including global position of each joint
    """
    input_data_dir = r'../../data/training_data/h36m.npz'
    input_data = np.load(input_data_dir)
    assert 'clips' in input_data.keys(), "cannot find motion data in " + input_data_dir
    motion_data = input_data['clips'][:, :-3]  ## global transformation is not included
    print(motion_data.shape)
    assert motion_data.shape[1] == 87
    ## normalize input data

    mean_value = motion_data.mean(axis=0)[np.newaxis, :]
    std_value = motion_data.std(axis=0)[np.newaxis, :]
    std_value[std_value < EPS] = EPS

    # np.savez_compressed(input_data_dir, clips=motion_data, mean=mean_value, std=std_value)
    normalized_data = (motion_data - mean_value) / std_value

    frame_encoder = FrameEncoderNoGlobal()
    learning_rate = 1e-3
    epochs = 100
    batchsize = 256
    frame_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                          loss='mse',
                          metrics=['accuracy'])
    # checkpoint_path = r'../../data/models/frame_encoder_no_global_trans/frame_encoder' + '-' + str(epochs) + '-' + str(learning_rate) + '-{epoch:04d}.ckpt'
    checkpoint_path = r'../../data/models/Fenc_no_global_trans/Fenc' + '-' + str(epochs) + '-' + str(learning_rate) + '-{epoch:04d}.ckpt'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=10)
    frame_encoder.save_weights(checkpoint_path.format(epoch=0))
    frame_encoder.fit(normalized_data, normalized_data, epochs=epochs, batch_size=batchsize, callbacks=[cp_callback])


def train_frameEncoder():
    input_data_dir = r'../../data/training_data/h36m.npz'
    input_data = np.load(input_data_dir)
    assert 'clips' in input_data.keys(), "cannot find motion data in " + input_data_dir
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
    # np.savez_compressed(input_data_dir, clips=motion_data, mean=mean_value, std=std_value)
    normalized_data = (motion_data - mean_value) / std_value

    dropout_rate = 0.3
    frame_encoder = FrameEncoder(dropout_rate=dropout_rate)
    learning_rate = 1e-3
    epochs = 100
    batchsize = 256
    frame_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                          loss='mse',
                          metrics=['accuracy'])
    checkpoint_path = r'../../data/models/frame_encoder1/frame_encoder' + '-' + str(epochs) + '-' + str(learning_rate) + '-' + str(dropout_rate) + '-{epoch:04d}.ckpt'

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=10)
    frame_encoder.save_weights(checkpoint_path.format(epoch=0))
    frame_encoder.fit(normalized_data, normalized_data, epochs=epochs, batch_size=batchsize, callbacks=[cp_callback])


def run_frameEncoderNoGlobalTrans():
    """[summary]
    """
    starting_frame = 0
    frame_length = 1000
    scale_factor = 5
    frame_encoder = FrameEncoderNoGlobal()
    input_data_dir = r'../../data/training_data/h36m.npz'
    input_data = np.load(input_data_dir)
    assert 'clips' in input_data.keys(), "cannot find motion data in " + input_data_dir
    motion_data = input_data['clips'][:, :-3]    

    ## normalize input data
    mean_value = motion_data.mean(axis=0)[np.newaxis, :]
    std_value = motion_data.std(axis=0)[np.newaxis, :]
    std_value[std_value < EPS] = EPS
    normalized_data = (motion_data - mean_value) / std_value

    # ### visualize input data
    input_motion = normalized_data[starting_frame: starting_frame+frame_length] * std_value + mean_value
    input_motion *= scale_factor
    input_motion = np.reshape(input_motion, (input_motion.shape[0], 29, 3))
    export_panim_data = {'motion_data': input_motion.tolist(), 'skeleton': MH_CMU_SKELETON, 'has_skeleton': True}
    write_to_json_file(r'D:\workspace\my_git_repos\vae_motion_modeling\data\test_data\ref_motion_no_global.panim', export_panim_data)

    learning_rate = 1e-3
    epochs = 100
    batchsize = 256
    frame_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                          loss='mse',
                          metrics=['accuracy'])
    frame_encoder.build(input_shape=motion_data.shape)
    checkpoint_path = r'../../data/models/Fenc_no_global_trans/Fenc' + '-' + str(epochs) + '-' + str(learning_rate) + '-{epoch:04d}.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(latest)
    frame_encoder.load_weights(latest)   

    ### qualitative evaluation
    res = np.asarray(frame_encoder(normalized_data[starting_frame : starting_frame + frame_length]))
    res_motion = res * std_value + mean_value
    res_motion *= scale_factor
    print(res_motion.shape)
    res_motion = np.reshape(res_motion, (res_motion.shape[0], 29, 3))
    export_res_motion = {'motion_data': res_motion.tolist(), 'has_skeleton': True, 'skeleton': MH_CMU_SKELETON}
    write_to_json_file(r'D:\workspace\my_git_repos\vae_motion_modeling\data\test_data\res_motion_no_global.panim', export_res_motion)
    # ### quantitive evaluation

    # print(motion_data.shape)
    # print(normalized_data.shape)
    # batchsize = 1000
    # print(motion_data.shape)

    # motion_data = motion_data[1000:10000]
    # normalized_data = normalized_data[1000:10000]
    # n_frames = len(motion_data)
    # n_batches = n_frames // batchsize + 1
    # output_frames = []
    # for i in range(n_batches):
    #     output_frames.append(frame_encoder(normalized_data[i*batchsize: (i+1)*batchsize]))

    # res = np.concatenate(output_frames, axis=0)
    # res = res * std_value + mean_value

    # frame_error = np.sum((motion_data[:, :-3] - res[:, :-3]) ** 2) / n_frames
    # # print(res.shape)
    # # ref_frames = export_point_cloud_data_without_foot_contact(motion_data) 
    # # reconstructed_frames = export_point_cloud_data_without_foot_contact(res)
    # # frame_error = np.sum((ref_frames - reconstructed_frames) ** 2) / n_frames
    # # print(ref_frames.shape)
    # print("average frame error is: ", frame_error)


def run_frameEncoder():
    starting_frame = 0
    frame_length = 1000
    scale_factor = 5
    dropout_rate = 0.25
    frame_encoder = FrameEncoder(dropout_rate=dropout_rate)

    input_data_dir = r'../../data/training_data/h36m.npz'
    input_data = np.load(input_data_dir)
    assert 'clips' in input_data.keys(), "cannot find motion data in " + input_data_dir
    motion_data = input_data['clips']
    ## normalize input data
    # if 'mean' not in input_data.keys():
    #     mean_value = motion_data.mean(axis=0)[np.newaxis, :]
    # else:
    #     mean_value = input_data['mean']
    # if 'std' not in input_data.keys():
    #     std_value = motion_data.std(axis=0)[np.newaxis, :]
    #     std_value[std_value < EPS] = EPS
    # else:
    #     std_value = input_data['std']
    mean_value = motion_data.mean(axis=0)[np.newaxis, :]
    std_value = motion_data.std(axis=0)[np.newaxis, :]
    std_value[std_value < EPS] = EPS
    normalized_data = (motion_data - mean_value) / std_value
    learning_rate = 1e-3
    epochs = 100
    batchsize = 256
    frame_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                          loss='mse',
                          metrics=['accuracy'])
    frame_encoder.build(input_shape=motion_data.shape)

    # frame_encoder.fit(normalized_data, normalized_data, epochs=1, batch_size=batchsize)
    # frame_encoder.summary()
    # checkpoint_path = r'../../data/models/frame_encoder1/frame_encoder' + '-' + str(epochs) + '-' + str(learning_rate) + '-' + str(dropout_rate) + '-{epoch:04d}.ckpt'
    # checkpoint_path = r'D:\workspace\my_git_repos\vae_motion_modeling\data\models\frame_encoder\frame_encoder-100-1e-05-0100.ckpt'
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    # print(latest)
    # frame_encoder.load_weights(latest)
    frame_encoder.load_weights(r'D:\workspace\my_git_repos\vae_motion_modeling\data\models\frame_encoder1\frame_encoder-100-0.001-0-0100.ckpt')



    ### generate reconstruction motions
    ref = motion_data[starting_frame : starting_frame + frame_length] 
    res = np.asarray(frame_encoder(normalized_data[starting_frame : starting_frame + frame_length]))
    # encoded_data = np.asarray(frame_encoder._encoder(normalized_data[starting_frame : starting_frame + frame_length]))
    # print(encoded_data.shape)
    ### generate random noise

    # z = np.random.rand(100, 10)
    # z = np.linspace(0, 1, 100)

    # z = np.tile(z, (10, 1)).T

    # new_encoded_data = np.zeros((200, 10))
    # new_encoded_data[:100] = encoded_data[:100]
    # # print(encoded_data[-10:])
    # # print("###########################")
    # new_encoded_data[100:] = encoded_data[100:] + z * -10
    # # print(new_encoded_data[-10:])
    # decoded_data = np.asarray(frame_encoder._decoder(new_encoded_data))
    res = res * std_value + mean_value
    # res = decoded_data * std_value + mean_value

    # timestr = time.strftime("%Y%m%d-%H%M%S")
    # # export_point_cloud_data_without_foot_contact(ref, filename=r'../../data/test_data/ref_' + timestr + '.panim', skeleton=MH_CMU_SKELETON, scale_factor=scale_factor)
    # export_point_cloud_data_without_foot_contact(res, filename=r'../../data/test_data/reconstructed_' + timestr + '.panim', skeleton=MH_CMU_SKELETON, scale_factor=scale_factor)

    ref_output_motion = {'motion_data': np.reshape(ref[:, :-3] * scale_factor, (ref.shape[0], 29, 3)).tolist(), 
                         'skeleton': MH_CMU_SKELETON,
                         'has_skeleton': True}
    reconstructed_output_motion = {'motion_data': np.reshape(res[:, :-3] * scale_factor, (res.shape[0], 29, 3)).tolist(),
                                   'skeleton': MH_CMU_SKELETON,
                                   'has_skeleton':True}
    
    write_to_json_file(r'D:\workspace\my_git_repos\vae_motion_modeling\data\test_data\ref_example.panim', ref_output_motion)                     
    write_to_json_file(r'D:\workspace\my_git_repos\vae_motion_modeling\data\test_data\res_example.panim', reconstructed_output_motion)


    # ### quantitive evaluation

    # print(motion_data.shape)
    # print(normalized_data.shape)
    # batchsize = 1000
    # print(motion_data.shape)

    # motion_data = motion_data[1000:10000]
    # normalized_data = normalized_data[1000:10000]
    # n_frames = len(motion_data)
    # n_batches = n_frames // batchsize + 1
    # output_frames = []
    # for i in range(n_batches):
    #     output_frames.append(frame_encoder(normalized_data[i*batchsize: (i+1)*batchsize]))

    # res = np.concatenate(output_frames, axis=0)
    # res = res * std_value + mean_value

    # frame_error = np.sum((motion_data[:, :-3] - res[:, :-3]) ** 2) / n_frames
    # # print(res.shape)
    # # ref_frames = export_point_cloud_data_without_foot_contact(motion_data) 
    # # reconstructed_frames = export_point_cloud_data_without_foot_contact(res)
    # # frame_error = np.sum((ref_frames - reconstructed_frames) ** 2) / n_frames
    # # print(ref_frames.shape)
    # print("average frame error is: ", frame_error)

    # ## visualize motion sequence in manifold
    # manifold_embedded = TSNE(n_components=3).fit_transform(encoded_data)
    # # manifold_embedded = PCA(n_components=3).fit_transform(encoded_data)
    # print(manifold_embedded.shape)

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(manifold_embedded[:, 0], manifold_embedded[:, 1], manifold_embedded[:, 2])
    # plt.show()


def test():
    input_data_dir = r'../../data/training_data/h36m.npz'
    input_data = np.load(input_data_dir)
    assert 'clips' in input_data.keys(), "cannot find motion data in " + input_data_dir
    motion_data = input_data['clips']
    print(motion_data.shape) 
    joint_data = motion_data[:1000, :-3].reshape((-1, 29, 3))
    export_data = {"motion_data": joint_data.tolist(), "has_skeleton": True, "skeleton": MH_CMU_SKELETON}
    write_to_json_file(r"D:\workspace\my_git_repos\vae_motion_modeling\data\ref.panim", export_data)   


if __name__ == "__main__":
    # run_frameEncoder()
    train_frameEncoder()
    # train_frameEncoder_no_global_trans()
    # run_frameEncoderNoGlobalTrans()
    # test()