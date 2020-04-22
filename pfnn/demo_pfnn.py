import numpy as np
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from pfnn.pfnn import PFNN
import tensorflow as tf 
tf.compat.v1.disable_eager_execution()
from utilities.utils import export_point_cloud_data_without_foot_contact
from utilities.skeleton_def import MH_CMU_SKELETON_FULL


def discrete_export_pfnn():
    model_file = r'D:\workspace\my_git_repos\deepMotionSunthesis_tf2.0\trained_models/pfnn_no_rot.ckpt'
    save_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\tensorflow\no_local_rot'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    n_controls = 4
    batchsize = 32
    dropout = 0.7
    Xshape = 342
    Yshape = 218
    model = PFNN(n_controls, Xshape + 1, Yshape, dropout, batchsize)
    model.create_model()
    model.load_model(model_file)
    model.save_params(save_path, 50)


def test():
    n_controls = 4
    batchsize = 32
    dropout = 0.7
    Xshape = 342
    Yshape = 311    
    model = PFNN(n_controls, Xshape+1, Yshape, dropout, batchsize)    
    model.create_model()    


def demo_pfnn():
    # model_file = r'D:\workspace\my_git_repos\deepMotionSynthesis\trained_models\pfnn_no_rot.ckpt'
    # meta_data_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\no_local_rot'
    model_file = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground\network.npz'
    meta_data_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground'
    n_controls = 4
    batchsize = 32
    dropout = 0.7
    Xshape = 342
    Yshape = 311
    model = PFNN(n_controls, Xshape+1, Yshape, dropout, batchsize)    
    model.create_model()
    # model.load_model(model_file)
    model_parameters = np.load(model_file)
    model.load_params_from_theano(model_parameters)
    ### load test data
    Xmean = np.fromfile(os.path.join(meta_data_path, 'Xmean.bin'), dtype=np.float32)
    Xstd = np.fromfile(os.path.join(meta_data_path, 'Xstd.bin'), dtype=np.float32)
    Ymean = np.fromfile(os.path.join(meta_data_path, 'Ymean.bin'), dtype=np.float32)
    Ystd = np.fromfile(os.path.join(meta_data_path, 'Ystd.bin'), dtype=np.float32)
    print(Ymean.shape)
    test_data = np.load(r'D:\gits\PFNN\mk_cmu_database.npz')
    # np.savez_compressed(r'D:\workspace\my_git_repos\deepMotionSynthesis\test_mk_cmu_database.npz',
    #                     Xun=test_data['Xun'][:1000], Yun=test_data['Yun'][:1000], Pun=test_data['Pun'][:1000])
    
    # test_data = np.load(r'D:\workspace\my_git_repos\deepMotionSynthesis\test_mk_cmu_database.npz')
    X = test_data['Xun']
    Y = test_data['Yun']
    P = test_data['Pun']
    print(Y.shape)
    # #### change run to walk

    # normalized_X_input = (X - Xmean) / Xstd
    # normalized_X_input[:, 48 : 48+12] = 0
    # normalized_X_input[:, 48+12 : 48+24] = 10
    # normalized_X_input[:, 48+24 : 48+36] = 0
    # normalized_X_input[:, 48+36 : 48+48] = 0
    # normalized_X_input[:, 48+48 : 48+60] = 0
    # normalized_X_input[:, 48+60 : 48+72] = 0
    # batchsize = 100

    # input = np.concatenate([normalized_X_input[-batchsize:], P[-batchsize:][...,np.newaxis]], axis=-1)
    # print(input.shape)

    # Y_prediction = model(input)

    # print(Y_prediction.shape)
    # Y_prediction = Y_prediction * Ystd + Ymean
    # visualize_output_data(Y_prediction, r'D:\tmp\tmp\motion_mk_cmu_random.panim')

    ######### compare mean pose output
    # mean_pose = np.zeros(Xmean.shape)

    # mean_pose = np.append(mean_pose, 0)

    # mean_prediction = model(mean_pose[np.newaxis, :])
    # print(mean_prediction.shape)
    # ### unnormalize output
    # output = mean_prediction * Ystd + Ymean
    # print(output.shape)
    # visualize_output_data(output, r'D:\tmp\tmp\mean_prediction.panim')

def path_following():
    pass


def gait_test():
    gait = np.loadtxt(r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\pfnn_data\LocomotionFlat01_000.gait')[::2]
    # print(gait.shape)
    gait = np.concatenate([
        gait[:,0:1],
        gait[:,1:2],
        gait[:,2:3] + gait[:,3:4],
        gait[:,4:5] + gait[:,6:7],
        gait[:,5:6],
        gait[:,7:8]
    ], axis=-1)
    n_frames = len(gait)
    window = 60
    rootgait = gait[0:2*window:10]
    print(rootgait.shape)
    print(rootgait)
    res = np.hstack([rootgait[:, 0].ravel(),
                     rootgait[:, 1].ravel(),
                     rootgait[:, 2].ravel(),
                     rootgait[:, 3].ravel(),
                     rootgait[:, 4].ravel(),
                     rootgait[:, 5].ravel()])
    print(res)


def visualize_output_data(output_data, filename):
    '''

    :param output_data: numpy array, n_frames * 311  (no local rotation case: 218 dimensions)

    :return:
    '''
    root_velocity = output_data[:, :2]
    root_rvelocity = output_data[:, 2:3]
    # input_joint_pos -> world space
    local_position = output_data[:, 32:32+93] # out joint pos
    local_velocity = output_data[:, 32+93:32+93*2]
    # local_rot = output_data[:, 32+93*2:]  ### not used
    # out joint pos -> world space
    # out joint vel
    # ((input_joint_pos + joint vel) + out joint pos) / 2
    anim_data = np.concatenate([local_position, root_velocity, root_rvelocity], axis=-1)
    export_point_cloud_data_without_foot_contact(anim_data, filename, skeleton=MH_CMU_SKELETON_FULL)





if __name__ == "__main__":
    # discrete_export_pfnn()
    demo_pfnn()
    # gait_test()
    # test()