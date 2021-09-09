import os
import sys
import numpy as np
import glob
from pathlib import Path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
import tensorflow as tf 
from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder
from mosi_utils_anim.utilities import load_json_file, write_to_json_file
from preprocessing.preprocessing import process_bvhfile, process_file
from utilities.utils import export_point_cloud_data_without_foot_contact
# from models.frame_encoder import FrameEncoder
from models.fullBody_pose_encoder import FullBodyPoseEncoder
from tensorflow.keras import Sequential, layers, Model, optimizers, losses
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
EPS = 1e-6



def get_index():
    fid_l = ['LeftFoot', 'LeftToeBase']
    fid_r = ['RightFoot', 'RightToeBase']
    body_plane = ['Spine', 'RightUpLeg', 'LeftUpLeg']
    bvhreader = BVHReader(r'C:\Users\hadu01\Downloads\data\OptiTrack\6kmh.bvh')
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    # print(skeleton.animated_joints)
    animated_joints = skeleton.animated_joints
    fid_l_indices = [animated_joints.index(joint) for joint in fid_l]
    fid_r_indices = [animated_joints.index(joint) for joint in fid_r]
    body_plane_indices = [animated_joints.index(joint) for joint in body_plane]

    print(fid_l_indices)
    print(fid_r_indices)
    print(body_plane_indices)



def test():
    json_data = load_json_file(r'D:\workspace\my_git_repos\capturesysevaluation\2 - Preprocessing\walk_art_l_featureVector.json')
    training_data = []
    for filename, data in json_data.items():

        training_data.append(np.ravel([item for sublist in data for subsublist in sublist for item in subsublist]))
    training_data = np.array(training_data)
    print(training_data.shape)
    pca = PCA(n_components=10)
    pca.fit(training_data)
    print(pca.explained_variance_ratio_)
    print(np.sum(pca.explained_variance_ratio_))
    # first_motion = json_data['registered_2696_2762_04_6kmh.bvh']
    # frames = []
    # for frame in first_motion:
    #     values = []
    #     for value in frame:
    #         values += value
    #     print(len(values))
    #     frames += values
    # frames = np.array(frames)
    # print(frames.shape)


def autoencoder_test():
    npc = 10
    json_data = load_json_file(r'D:\workspace\my_git_repos\capturesysevaluation\data\2 - Preprocessing\walk_art_l_featureVector.json')
    training_data = []
    for filename, data in json_data.items():

        training_data.append(np.ravel([item for sublist in data for subsublist in sublist for item in subsublist]))
    training_data = np.array(training_data)
    mean = training_data.mean(axis=0)[np.newaxis, :]
    std = training_data.std(axis=0)[np.newaxis, :]
    std[std < EPS] = EPS
    normalized_data = (training_data - mean) / std
    #### initialize model
    dropout_rate = 0.05
    epochs = 1000
    batchsize = 1
    learning_rate = 1e-4    
    pose_encoder = FullBodyPoseEncoder(output_dim=normalized_data.shape[1], dropout_rate=dropout_rate, npc=npc)
    pose_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                        loss='mse')  
    checkpoint_path = r'../../data/models/OptiTrack/autoencoder' + '-' + str(npc) + '-' + str(epochs) + '-' + str(learning_rate) + '-{epoch:04d}.ckpt'

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=100) 
    pose_encoder.fit(normalized_data, normalized_data, epochs=epochs, batch_size=batchsize, callbacks=[cp_callback])         

    ###### load and evaluation

    dropout_rate = 0.05
    epochs = 1000
    batchsize = 1
    learning_rate = 1e-4  
    model_path = r'../../data/models/OptiTrack/autoencoder' + '-' + str(npc) + '-' + str(epochs) + '-' + str(learning_rate) + '-1000.ckpt'  
    pose_encoder = FullBodyPoseEncoder(output_dim=normalized_data.shape[1], dropout_rate=dropout_rate, npc=npc)   
    pose_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                        loss='mse')
    pose_encoder.build(input_shape=normalized_data.shape) 
    pose_encoder.load_weights(model_path)

    encoded_data = np.asarray(pose_encoder.encode(normalized_data))
    print(encoded_data.shape)
    reconstructed_motion = np.asarray(pose_encoder(normalized_data))
    reconstructed_motion = reconstructed_motion * std + mean 
    r_var_origin = np.mean(training_data.var(axis=0))
    r_var = np.mean(reconstructed_motion.var(axis=0))
    print(r_var)    
    print(r_var / r_var_origin)
    mse = np.sum((training_data - reconstructed_motion) ** 2) / 65
    print(mse)


def PCA_on_preprocessed_data():
    data_folder = r'D:\workspace\my_git_repos\capturesysevaluation\data\2 - Preprocessing'
    capturing_systems = ['art', 'capturystudio', 'optitrack', 'vicon']
    motion_types = ['l', 'r']
    npc = 10
    results = {}
    n_frames = 65

    for sys in capturing_systems:
        for type in motion_types:
            training_data = []
            json_data = load_json_file(os.path.join(data_folder, '_'.join(['walk', sys, type, 'featureVector.json'])))
            for filename, data in json_data.items():
                training_data.append(np.ravel([item for sublist in data for subsublist in sublist for item in subsublist]))
            training_data = np.asarray(training_data)
            pca = PCA(n_components=npc)
            pca.fit(training_data)
            # explained_variance = np.sum(pca.explained_variance_ratio_)
            projection = pca.transform(training_data)
            backprojection = pca.inverse_transform(projection)
            origin_var = np.sum(training_data.var(axis=0))
            recon_var = np.sum(backprojection.var(axis=0))
            explained_variance = recon_var / origin_var
            mse = np.sum((training_data - backprojection)**2) / n_frames
            print("mean square error: ", mse)
            results['_'.join([sys, type])] = {'explained_variance': explained_variance,
                                              'mse': mse}
    print(results)
    write_to_json_file('pca_preprocessed_data.json', results)


def train_autoencoder_on_preprocessed_data():
    data_folder = r'C:\Users\hadu01\Downloads\data\2 - Preprocessing'
    capturing_systems = ['art', 'capturystudio', 'optitrack', 'vicon']
    motion_types = ['l', 'r']
    npc = 10
    n_frames = 65

    ##### MODEL TRAINING
    for sys in capturing_systems:
        for type in motion_types:
            training_data = []
            json_data = load_json_file(os.path.join(data_folder, '_'.join(['walk', sys, type, 'featureVector.json'])))
            for filename, data in json_data.items():
                training_data.append(np.ravel([item for sublist in data for subsublist in sublist for item in subsublist]))
            training_data = np.asarray(training_data)
            mean = training_data.mean(axis=0)[np.newaxis, :]
            std = training_data.std(axis=0)[np.newaxis, :]
            std[std < EPS] = EPS
            normalized_data = (training_data - mean) / std

            dropout_rate = 0.05
            epochs = 1000
            batchsize = 1
            learning_rate = 1e-4
            pose_encoder = FullBodyPoseEncoder(output_dim=normalized_data.shape[1], dropout_rate=dropout_rate, npc=npc)
            pose_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                                loss='mse')            

            pose_encoder.fit(normalized_data, normalized_data, epochs=epochs, batch_size=batchsize)
            save_path = os.path.join(r'../../data/models', sys, type)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            pose_encoder.save_weights(os.path.join(save_path, '_'.join([str(dropout_rate), str(epochs), str(learning_rate), '.ckpt'])))


def evaluate_autoencoder_on_preprocessed_data():
    data_folder = r'C:\Users\hadu01\Downloads\data\2 - Preprocessing'
    capturing_systems = ['art', 'capturystudio', 'optitrack', 'vicon']
    motion_types = ['l', 'r']
    npc = 10
    n_frames = 65
    res = {}    
    for sys in capturing_systems:
        for type in motion_types:
            training_data = []
            json_data = load_json_file(os.path.join(data_folder, '_'.join(['walk', sys, type, 'featureVector.json'])))
            for filename, data in json_data.items():
                training_data.append(np.ravel([item for sublist in data for subsublist in sublist for item in subsublist]))
            training_data = np.asarray(training_data)
            mean = training_data.mean(axis=0)[np.newaxis, :]
            std = training_data.std(axis=0)[np.newaxis, :]
            std[std < EPS] = EPS
            normalized_data = (training_data - mean) / std
            #### load model
            dropout_rate = 0.05
            epochs = 1000
            batchsize = 1
            learning_rate = 1e-4
            pose_encoder = FullBodyPoseEncoder(output_dim=normalized_data.shape[1], dropout_rate=dropout_rate, npc=npc)            
            model_path = os.path.join(r'../../data/models', sys, type, '_'.join([str(dropout_rate), str(epochs), str(learning_rate), '.ckpt']))
            pose_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                                loss='mse')
            pose_encoder.build(input_shape=normalized_data.shape)             
            pose_encoder.load_weights(model_path)

            reconstructed_motions = np.asarray(pose_encoder(normalized_data))
            reconstructed_motions = reconstructed_motions * std + mean
            origin_var = np.sum(training_data.var(axis=0))
            recon_var = np.sum(reconstructed_motions.var(axis=0))
            ratio = recon_var / origin_var

            mse = np.sum((training_data - reconstructed_motions) ** 2) / n_frames
            print("mean square error: ", mse)
            res['_'.join([sys, type])] = {'explained_variance': ratio,
                                          'mse': mse}
    print(res)
    write_to_json_file('autoencoder_preprocessed_data.json', res)


def bar_plot_results():
    result_data = load_json_file('pca_preprocessed_data.json')
    print(result_data.keys())
    labels = ['capturystudio', 'optitrack', 'vicon', 'art']
    left_var = []
    right_var = []
    left_mse = []
    right_mse = []
    for sys in labels:
        left_var.append(result_data['_'.join([sys, 'l'])]['explained_variance'])
        left_mse.append(result_data['_'.join([sys, 'l'])]['mse'])
        right_var.append(result_data['_'.join([sys, 'r'])]['explained_variance'])
        right_mse.append(result_data['_'.join([sys, 'r'])]['mse'])

    width = 0.35
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    # rects_var_l = ax.bar(x - width/2, left_var, width, label='left step')
    # rects_var_r = ax.bar(x + width/2, right_var, width, label='right step')
    rects_var_l = ax.bar(x - width/2, left_mse, width, label='left step')
    rects_var_r = ax.bar(x + width/2, right_mse, width, label='right step')
    # ax.set_ylabel("explained variance")
    ax.set_ylabel('mse')
    ax.set_title("PCA on quaternion frames")
    ax.set_xticks(x)
    # plt.ylim(0.85, 1.0)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')   

    autolabel(rects_var_l)
    autolabel(rects_var_r)
    fig.tight_layout()

    plt.show()    


def bar_plot_results_vae():
    result_data = load_json_file('autoencoder_preprocessed_data.json')
    print(result_data.keys())
    labels = ['capturystudio', 'optitrack', 'vicon', 'art']
    left_var = []
    right_var = []
    left_mse = []
    right_mse = []
    for sys in labels:
        left_var.append(result_data['_'.join([sys, 'l'])]['explained_variance'])
        left_mse.append(result_data['_'.join([sys, 'l'])]['mse'])
        right_var.append(result_data['_'.join([sys, 'r'])]['explained_variance'])
        right_mse.append(result_data['_'.join([sys, 'r'])]['mse'])

    width = 0.35
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    # rects_var_l = ax.bar(x - width/2, left_var, width, label='left step')
    # rects_var_r = ax.bar(x + width/2, right_var, width, label='right step')
    rects_var_l = ax.bar(x - width/2, left_mse, width, label='left step')
    rects_var_r = ax.bar(x + width/2, right_mse, width, label='right step')
    # ax.set_ylabel("explained variance")
    ax.set_ylabel("mse")
    ax.set_title("VAE on quaternion frames")
    ax.set_xticks(x)
    # plt.ylim(0.85, 1.2)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')   

    autolabel(rects_var_l)
    autolabel(rects_var_r)
    fig.tight_layout()

    plt.show()


def modeling_training():
    # npc = 3
    # ### OptiTrack 
    # data_folder = r'C:\Users\hadu01\Downloads\data\OptiTrack'
    # bvhfiles = glob.glob(os.path.join(data_folder, '*.bvh'))

    # motion_clips = []
    # for bvhfile in bvhfiles:
    #     motion_clip = process_bvhfile(bvhfile, body_plane_indices=np.array([1, 21, 17]), 
    #                                   fid_l=np.array([19, 20]), 
    #                                   fid_r=np.array([23, 24]), 
    #                                   animated_joints=None, 
    #                                   sliding_window=False)
    #     # print(motion_clip.shape)
    #     motion_clips.append(motion_clip)
    # training_data = np.concatenate(motion_clips, axis=0)
    # print(training_data.shape)
    # np.savez_compressed('OptiTrack_training_data.npz', X=training_data)

    ########## load training data
    training_data_path = r'OptiTrack_training_data.npz'
    training_data = np.load(training_data_path)
    motion_data = training_data['X']
    if 'mean' not in training_data.keys():
        mean= motion_data.mean(axis=0)[np.newaxis, :]
    else:
        mean = training_data['mean']
    if 'std' not in training_data.keys():
        
        std = motion_data.std(axis=0)[np.newaxis, :]
        std[std < EPS] = EPS
    else:
        std = training_data['std']
    # np.savez_compressed(training_data_path, X=motion_data, mean=mean, std=std)

    normalized_data = (motion_data - mean) / std
    #### initialize model
    dropout_rate = 0.3
    epochs = 100
    batchsize = 64
    learning_rate = 1e-4
    npcs = range(3, 11)
    for npc in npcs:
        pose_encoder = FullBodyPoseEncoder(output_dim=normalized_data.shape[1], dropout_rate=dropout_rate, npc=npc)
        pose_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                            loss='mse',
                            metrics=['accuracy'])     
        checkpoint_path = r'../../data/models/OptiTrack/pose_encoder' + '-' + str(npc) + '-' + str(epochs) + '-' + str(learning_rate) + '-{epoch:04d}.ckpt'

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            verbose=1, 
            save_weights_only=True,
            period=100)    
        # pose_encoder.save_weights(checkpoint_path.format(epoch=0))
        pose_encoder.fit(normalized_data, normalized_data, epochs=epochs, batch_size=batchsize, callbacks=[cp_callback])                              



def pca_on_cartesian_frames_optitrack():

    training_data_path = r'OptiTrack_training_data.npz'
    training_data = np.load(training_data_path)
    motion_data = training_data['X']  
    npcs = range(3, 11)
    MSEs = []
    for npc in npcs:
        pca = PCA(n_components=npc)
        pca.fit(motion_data)
        projection = pca.transform(motion_data)
        backprojection = pca.inverse_transform(projection)

        #### reconstruct cartesian frames
        ref_cartesian_frames = export_point_cloud_data_without_foot_contact(motion_data)
        recon_cartesian_frames = export_point_cloud_data_without_foot_contact(backprojection)
        n_frames, n_joints, _ = ref_cartesian_frames.shape
        n_frames = 1000
        mse = np.sum((ref_cartesian_frames[:1000] - recon_cartesian_frames[:1000]) ** 2) / (n_frames * n_joints)
        MSEs.append(mse)

    fig = plt.figure()
    plt.plot(range(3, 11), MSEs)
    plt.show()


def compression_evaluation():

    # npc = 10
    bvhreader = BVHReader(r'C:\Users\hadu01\Downloads\data\OptiTrack\6kmh.bvh')
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    skeleton_def = skeleton.generate_bone_list_description()


    training_data_path = r'OptiTrack_training_data.npz'
    training_data = np.load(training_data_path)
    motion_data = training_data['X']
    if 'mean' not in training_data.keys():
        mean= motion_data.mean(axis=0)[np.newaxis, :]
    else:
        mean = training_data['mean']
    if 'std' not in training_data.keys():
        
        std = motion_data.std(axis=0)[np.newaxis, :]
        std[std < EPS] = EPS
    else:
        std = training_data['std']
    normalized_data = (motion_data - mean) / std
    #### initialize model
    dropout_rate = 0.3
    epochs = 100
    batchsize = 64
    learning_rate = 1e-4
    var = np.mean(motion_data.var(axis=0))
    print(var)
    # npcs = range(3, 11)
    npcs = [10]
    MSEs = []
    for npc in npcs:
        pose_encoder = FullBodyPoseEncoder(output_dim=normalized_data.shape[1], dropout_rate=dropout_rate, npc=npc)
        pose_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                            loss='mse')
        pose_encoder.build(input_shape=motion_data.shape)   

        #### load model
        checkpoint_path = r'../../data/models/OptiTrack/pose_encoder' + '-' + str(npc) + '-' + str(epochs) + '-' + str(learning_rate) + '-0100.ckpt'
        # checkpoint_dir = os.path.dirname(checkpoint_path)
        # print(checkpoint_dir)

        pose_encoder.load_weights(checkpoint_path)

        encoded_data = np.asarray(pose_encoder.encode(normalized_data))
        print(encoded_data.shape)
        reconstructed_motion = np.asarray(pose_encoder(normalized_data))
        reconstructed_motion = reconstructed_motion * std + mean 
        r_var = np.mean(reconstructed_motion.var(axis=0))
        print(r_var)
    #     ref_cartesian_frames = export_point_cloud_data_without_foot_contact(motion_data)
    #     recon_cartesian_frames = export_point_cloud_data_without_foot_contact(reconstructed_motion)
    #     n_frames, n_joints, _ = ref_cartesian_frames.shape
    #     # print(n_frames)
    #     print(recon_cartesian_frames.shape)
    #     n_frames = 1000
    #     mse = np.sum((ref_cartesian_frames[:1000] - recon_cartesian_frames[:1000]) ** 2) / (n_frames * n_joints)
    #     print(mse)
    #     MSEs.append(mse)
    #     ##### qualitive evaluation 

    #     export_point_cloud_data_without_foot_contact(reconstructed_motion[:1000], 'reconstructed_motion_' + str(npc) + '.panim', skeleton_def)
    # fig = plt.figure()
    # plt.plot(range(3, 11), MSEs)
    # plt.show()



def visual_training_data():
    traing_data = np.load('OptiTrack_training_data.npz')['X']
    print(traing_data.shape)
    bvhreader = BVHReader(r'C:\Users\hadu01\Downloads\data\OptiTrack\6kmh.bvh')
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    skeleton_def = skeleton.generate_bone_list_description()
    export_point_cloud_data_without_foot_contact(traing_data[:1000, :], 'motion.panim', skeleton_def)
    

def load_preprocessing_data():
    data = load_json_file(r'C:\Users\hadu01\Downloads\data\2 - Preprocessing\walk_art_l_featureVector.json')
    # for key in data.keys():
    #     print(key)
    motion_vector = data['registered_2696_2762_04_6kmh.bvh']
    motion_vector = np.ravel(motion_vector)
    print(motion_vector.shape)
    # print(motion_vector.shape)
    # first_frame = motion_vector[0]
    # print(len(first_frame))

    # max_data = load_json_file(r'C:\Users\hadu01\Downloads\data\2 - Preprocessing\walk_art_l_maxVector.json')



if __name__ == "__main__":
    # compression_evaluation()
    # get_index()
    # visual_training_data()
    # modeling_training()
    # load_preprocessing_data()
    # test()
    # PCA_on_preprocessed_data()
    # autoencoder_test()
    # train_autoencoder_on_preprocessed_data()
    # bar_plot_results_vae()
    evaluate_autoencoder_on_preprocessed_data()
    # pca_on_cartesian_frames_optitrack()
