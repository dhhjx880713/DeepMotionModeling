import os
import sys
import numpy as np
from numpy.core.fromnumeric import std
from tensorflow.python.ops.gen_math_ops import exp 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf 
import logging
tf.get_logger().setLevel(logging.ERROR)
dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dirname, ".."))
from models.frame_encoder import FrameEncoder, FrameEncoderDropoutFirst
from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder
from preprocessing.preprocessing import process_bvhfile, process_file
from mosi_utils_anim.utilities import write_to_json_file
from utilities.utils import get_global_position_framewise, get_global_position, export_point_cloud_data_without_foot_contact
from utilities.skeleton_def import MH_CMU_SKELETON
EPS = 1e-6
from sklearn.metrics import mean_squared_error
import argparse


def get_training_data(name='h36m', data_type='angle'):
    data_path = os.path.join(dirname, r'../..', r'data\training_data\processed_mocap_data', name)
    filename = '_'.join([name, data_type]) + '.npy'
    if not os.path.isfile(os.path.join(data_path, filename)):
        print("Cannot find " + os.path.join(data_path, filename))
    else:

        motion_data = np.load(os.path.join(data_path, filename))
        return motion_data


def run_frameEncoder1():
    starting_frame = 500
    frame_length = 1000
    dropout_rate = 0.1
    
    scale_factor = 5
    input_data_dir = r'D:\workspace\my_git_repos\vae_motion_modeling\data\training_data/h36m.npz'
    input_data = np.load(input_data_dir)
    assert 'clips' in input_data.keys(), "cannot find motion data in " + input_data_dir
    motion_data = input_data['clips']
    mean_value = motion_data.mean(axis=0)[np.newaxis, :]
    std_value = motion_data.std(axis=0)[np.newaxis, :]
    std_value[std_value < EPS] = EPS
    normalized_data = (motion_data - mean_value) / std_value
    frame_encoder = FrameEncoder(dropout_rate=dropout_rate)
    # frame_encoder.load_weights(r'D:\workspace\my_git_repos\vae_motion_modeling\data\models\frame_encoder1\frame_encoder-100-0.0001-0.1-0100.ckpt')
    frame_encoder.load_weights(r'D:\workspace\my_git_repos\vae_motion_modeling\data\models\h36m_frameEnc_quat\h36m_frameEnc_quat_0100.ckpt')
    # ref = motion_data[starting_frame : starting_frame + frame_length]
    # res = np.asarray(frame_encoder(ref))
    ### load test motion 
    test_file = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\h36m\S1\Walking.bvh'
    input_data = process_bvhfile(test_file, sliding_window=False)

    normalized_input_data = (input_data - mean_value) / std_value
    res = np.asarray(frame_encoder(normalized_input_data))



def run_frameEncoderDropuoutFirst_quat():
    ### load training data
    h36m_data = get_training_data(name='h36m', data_type='quaternion')
    h36m_data = np.reshape(h36m_data, (h36m_data.shape[0] * h36m_data.shape[1], h36m_data.shape[2]))
    ### normalize data
    mean_value = h36m_data.mean(axis=0)[np.newaxis, :]
    std_value = h36m_data.std(axis=0)[np.newaxis, :]
    std_value[std_value<EPS] = EPS    

    ### load the model 
    model_file = r'E:\results\h36m_frameEncDropoutFirst_quat\h36m_frameEncDropoutFirst_quat_0020.ckpt'
    encoder = FrameEncoderDropoutFirst(dropout_rate=0.1)
    encoder.load_weights(model_file)
    
    ### load test motion 
    test_file = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\h36m\S1\Walking.bvh'
    input_data = process_file(test_file, sliding_window=False)    

    ### use training data
    starting_frame = 0
    frame_length = 1000
    input_data = h36m_data[starting_frame : starting_frame + frame_length]

    normalized_input_data = (input_data - mean_value) / std_value
    print(normalized_input_data.shape)
    res = np.asarray(encoder(normalized_input_data))

    scale_factor = 5
    reconstructed_motion = res * std_value + mean_value

    original_motion = normalized_input_data * std_value + mean_value

    export_point_cloud_data_without_foot_contact(reconstructed_motion, r'E:\tmp\recon_frameEncoderDropoutFirst_quat.panim', scale_factor=scale_factor)
    export_point_cloud_data_without_foot_contact(original_motion, r'E:\tmp\recon_frameEncoderDropoutFirst_origin.panim', scale_factor=scale_factor)


def run_frameEncoder_quat(input_file, save_path='.'):

    ### load training data  (to get statistical data (mean and variance))
    h36m_data = get_training_data(name='h36m', data_type='quaternion')

    h36m_data = np.reshape(h36m_data, (h36m_data.shape[0] * h36m_data.shape[1], h36m_data.shape[2]))
    ### normalize data
    mean_value = h36m_data.mean(axis=0)[np.newaxis, :]
    std_value = h36m_data.std(axis=0)[np.newaxis, :]
    std_value[std_value<EPS] = EPS

    ### load the model 
    # model_name = r'D:\workspace\my_git_repos\vae_motion_modeling\data\models\h36m_frameEnc_quat\h36m_frameEnc_quat_0100.ckpt'
    model_name = r'D:\workspace\my_git_repos\vae_motion_modeling\data\models\frame_encoder1\frame_encoder-100-0.0001-0.1-0100.ckpt'
    encoder = FrameEncoder(dropout_rate=0.1)
    encoder.load_weights(model_name)   
    ### load test motion 
    # test_file = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\h36m\S1\Walking.bvh'
    input_data = process_file(input_file, sliding_window=False)

    normalized_input_data = (input_data - mean_value) / std_value
    print(normalized_input_data.shape)
    res = np.asarray(encoder(normalized_input_data))

    scale_factor = 5
    reconstructed_motion = res * std_value + mean_value

    original_motion = normalized_input_data * std_value + mean_value
    filename = os.path.split(input_file)[-1]

    model_name = os.path.split(model_name)[-1]
    save_filename = filename[:-4] + '_' + model_name[:-5] + '.panim'
    save_filename = os.path.join(save_path, save_filename)
    export_point_cloud_data_without_foot_contact(reconstructed_motion, save_filename, scale_factor=scale_factor)
    # export_point_cloud_data_without_foot_contact(original_motion, r'E:\tmp\recon_frameEncoder_origin.panim', scale_factor=scale_factor)
 
    # original_motion[:, -1] = - original_motion[:, -1]
    # reconstructed_motion[:, -1] = - reconstructed_motion[:, -1]

    # relative_joint_positions = get_global_position_framewise(reconstructed_motion)
    # global_joint_position = get_global_position(relative_joint_positions)
    # global_joint_position *= scale_factor

    original_relative_joint_positions = get_global_position_framewise(original_motion)
    original_global_joint_position = get_global_position(original_relative_joint_positions)
    original_global_joint_position *= scale_factor    

    reconstructed_motion_data_origin = {'motion_data': original_global_joint_position.tolist(), 'has_skeleton': True,
                                 'skeleton': MH_CMU_SKELETON}
    save_filename = os.path.join(save_path, filename[:-4] + '_' + 'origin.panim')
    # write_to_json_file(save_filename, reconstructed_motion_data_origin)
    export_point_cloud_data_without_foot_contact(original_motion, save_filename, scale_factor=scale_factor)


def run_frameEncoder_euler():

    h36m_data = get_training_data()
    h36m_data = np.reshape(h36m_data, (h36m_data.shape[0] * h36m_data.shape[1], h36m_data.shape[2]))
    ### normalize data
    mean_value = h36m_data.mean(axis=0)[np.newaxis, :]
    std_value = h36m_data.std(axis=0)[np.newaxis, :]
    std_value[std_value<EPS] = EPS
    test_file = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\h36m\S1\Walking.bvh'
    input_data = process_bvhfile(test_file, sliding_window=False)
    scale_factor = 5
    normalized_input_data = (input_data - mean_value) / std_value

    model_file = r'D:\workspace\my_git_repos\vae_motion_modeling\data\models\h36m_frameEnc_angle\h36m_frameEnc_angle_0100.ckpt'
    encoder = FrameEncoder(dropout_rate=0.1)
    encoder.load_weights(model_file)
    res = encoder(normalized_input_data)

    reconstructed_motion = res * std_value + mean_value
    export_point_cloud_data_without_foot_contact(reconstructed_motion.numpy(), 
                                                 r'E:\tmp\recon_frameEncoder_angle.panim',
                                                 scale_factor=scale_factor)


def run_frameEncoder():
    """test frame encoder for h36m data

    """
    ### load model
    model_name = os.path.join(dirname, '../..', r'data/models', "h36m_frameEnc_quat", "h36m_frameEnc_quat_0100.ckpt")
    encoder = FrameEncoder(dropout_rate=0.1)
    encoder.load_weights(model_name)

    ### load test file
    test_file = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\h36m\S1\Walking.bvh'
    input_data = process_file(test_file, sliding_window=False)
    
    ### get statistical values
    ### task to do: save the value
    h36m_data = get_training_data(data_type='quaternion')
    # ### get Mean and Var
    # h36m_data = get_training_data()
    # print(h36m_data.shape)
    h36m_data = np.reshape(h36m_data, (h36m_data.shape[0] * h36m_data.shape[1], h36m_data.shape[2]))
    ### normalize data
    scale_factor = 5
    mean_value = h36m_data.mean(axis=0)[np.newaxis, :]
    std_value = h36m_data.std(axis=0)[np.newaxis, :]
    std_value[std_value<EPS] = EPS
    normalized_input_data = (input_data - mean_value) / std_value

    output = encoder(normalized_input_data)
    reconstructed_motion = output * std_value + mean_value

    export_point_cloud_data_without_foot_contact(reconstructed_motion.numpy(), 
                                                 r'E:\tmp\recon_frameEncoder.panim', 
                                                 scale_factor=scale_factor)

    # ### load the model 
    # # model_name = r'D:\workspace\my_git_repos\vae_motion_modeling\data\models\h36m_frameEnc_angle\h36m_frameEnc_angle_0100.ckpt'
    # model_name = r'E:\results\h36m_frameEncoder_customized_loss\h36m_frameEncoder_customized_loss_0100.ckpt'
    # encoder = FrameEncoder(dropout_rate=0.1)
    # # encoder.build(input_shape=h36m_data.shape)
    # encoder.load_weights(model_name)

    # ### load test motion 
    # test_file = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\h36m\S1\Walking.bvh'
    # input_data = process_bvhfile(test_file, sliding_window=False)

    # normalized_input_data = (input_data - mean_value) / std_value
    # print(normalized_input_data.shape)
    # res = np.asarray(encoder(normalized_input_data))
    # # print(res.shape)
    # scale_factor = 5
    # reconstructed_motion = res * std_value + mean_value

    # original_motion = normalized_input_data * std_value + mean_value
    # reconstructed_motion[:, -1] = -reconstructed_motion[:, -1]
    # export_point_cloud_data_without_foot_contact(reconstructed_motion, r'E:\tmp\recon_frameEncoder_customized_loss.panim', scale_factor=scale_factor)

    # relative_joint_positions = get_global_position_framewise(reconstructed_motion)
    # global_joint_position = get_global_position(relative_joint_positions)
    # global_joint_position *= scale_factor

    # original_relative_joint_positions = get_global_position_framewise(original_motion)
    # original_global_joint_position = get_global_position(original_relative_joint_positions)
    # original_global_joint_position *= scale_factor

    # reconstructed_motion_data = {'motion_data': global_joint_position.tolist(), 'has_skeleton': True,
    #                              'skeleton': MH_CMU_SKELETON}
    # write_to_json_file(r'E:\tmp\recon_frameEncoder2.panim', reconstructed_motion_data)

    # reconstructed_motion_data_origin = {'motion_data': original_global_joint_position.tolist(), 'has_skeleton': True,
    #                              'skeleton': MH_CMU_SKELETON}
    # write_to_json_file(r'E:\tmp\recon_frameEncoder_origin.panim', reconstructed_motion_data_origin)



def test():
    h36m_data = get_training_data()
    print(h36m_data.shape)
    mean_pose = h36m_data.mean(axis=(0, 1))
    std_pose = h36m_data.std(axis=(0, 1))
    std_pose[std_pose<EPS] = EPS

    model_name = r'D:\workspace\my_git_repos\vae_motion_modeling\data\models\h36m_frameEnc_angle\h36m_frameEnc_angle_0100.ckpt'
    encoder = FrameEncoder(dropout_rate=0.1)
    encoder.load_weights(model_name)

    normalized_data = (h36m_data - mean_pose) / std_pose
    frame_leng = 1000
    normalized_data = np.reshape(normalized_data, (normalized_data.shape[0]*normalized_data.shape[1], normalized_data.shape[2]))
    test_data = normalized_data[:frame_leng]
    first_half = test_data[:frame_leng//2]
    second_half = test_data[frame_leng//2:]
    res = encoder(test_data).numpy()
    res_first_half = encoder(first_half).numpy()
    res_second_half = encoder(second_half).numpy()

    recon_motion = res * std_pose + mean_pose
    recon_first_half = res_first_half * std_pose + mean_pose
    recon_second_half = res_second_half * std_pose + mean_pose

    ### create global joint position
    recon_motion = get_global_position_framewise(recon_motion)
    print(recon_motion.shape)

    recon_motion_first_half = get_global_position_framewise(recon_first_half)
    recon_motion_second_half = get_global_position_framewise(recon_second_half)
    print(recon_motion_second_half.shape)

    print(np.allclose(recon_motion[500:], recon_motion_second_half, rtol=1e-4))

def quantitive_evaluation():
    """measure reconstruction error in global joint space
    """
    ### assumptions: quaternion and angular representations should achieve similar error level

    ### angular representation
    h36m_data = get_training_data()
    h36m_data = np.reshape(h36m_data, (h36m_data.shape[0] * h36m_data.shape[1], h36m_data.shape[2]))
    ###
    mean_value = h36m_data.mean(axis=0)[np.newaxis, :]
    std_value = h36m_data.std(axis=0)[np.newaxis, :]
    std_value[std_value<EPS] = EPS
    normalized_data = (h36m_data - mean_value) / std_value

    ### load test motion 

    # res = encoder(normalized_data[:1000])
    # reconstructed_data = res * std_value + mean_value
    
    # loss = mean_squared_error(h36m_data[:1000], reconstructed_data[:1000])
    # print(loss)
    n_frames = len(normalized_data)
    batchsize = 256
    n_batches = n_frames // batchsize
    ####
    print("#########################")
    print("model with costomized loss: ")
    ### load model
    # model_name = r'D:\workspace\my_git_repos\vae_motion_modeling\data\models\h36m_frameEnc_angle\h36m_frameEnc_angle_0100.ckpt'
    model_name = r'E:\results\h36m_frameEncoder_customized_loss\h36m_frameEncoder_customized_loss_0100.ckpt'
    encoder = FrameEncoder(dropout_rate=0.1)
    encoder.load_weights(model_name)
    losses_original_space = []
    losses_global_space = []
    for i in range(n_batches):
        input_batch = normalized_data[batchsize * i: batchsize * (i+1)]
        res = encoder(input_batch)
        reconstructed_data = res * std_value + mean_value
        reconstructed_data = reconstructed_data.numpy()
        losses_original_space.append(mean_squared_error(h36m_data[batchsize * i : batchsize * (i+1)], reconstructed_data))

        reconstructed_global_position = get_global_position_framewise(reconstructed_data)
        reconstructed_global_position = np.reshape(reconstructed_global_position, (reconstructed_global_position.shape[0], np.prod(reconstructed_global_position.shape[1:])))
        original_global_position = get_global_position_framewise(h36m_data[batchsize * i : batchsize * (i+1)])
        original_global_position = np.reshape(original_global_position, (original_global_position.shape[0], np.prod(original_global_position.shape[1:])))
        losses_global_space.append(mean_squared_error(original_global_position, reconstructed_global_position))
    average_loss_original_space = np.sum(losses_original_space) / n_batches
    print("reconstruction loss in original space: {}".format(average_loss_original_space))
    average_loss_global_space = np.sum(losses_global_space) / n_batches
    print("reconstruction loss in global space: {}".format(average_loss_global_space))

    ####
    print("##########################")
    print("model with fine_tuning custom model: ")
    model_file = r'E:\results\h36m_frameEncoder_customized_loss\h36m_frameEncoder_customized_loss_fine_tuning_0100.ckpt'
    model_name = "h36m_frameEncoder_customized_loss"
    fine_tuned_encoder = FrameEncoder(dropout_rate=0.1, name=model_name)
    fine_tuned_encoder.load_weights(model_file)
    losses_original_space = []
    losses_global_space = []    
    for i in range(n_batches):
        input_batch = normalized_data[batchsize * i : batchsize * (i+1)]
        res = fine_tuned_encoder(input_batch)
        reconstructed_data = res * std_value + mean_value
        reconstructed_data = reconstructed_data.numpy()
        losses_original_space.append(mean_squared_error(h36m_data[batchsize * i : batchsize * (i+1)], reconstructed_data))

        reconstructed_global_position = get_global_position_framewise(reconstructed_data)
        reconstructed_global_position = np.reshape(reconstructed_global_position, (reconstructed_global_position.shape[0], np.prod(reconstructed_global_position.shape[1:])))
        original_global_position = get_global_position_framewise(h36m_data[batchsize * i : batchsize * (i+1)])
        original_global_position = np.reshape(original_global_position, (original_global_position.shape[0], np.prod(original_global_position.shape[1:])))
        losses_global_space.append(mean_squared_error(original_global_position, reconstructed_global_position))  

    average_loss_original_space = np.sum(losses_original_space) / n_batches
    print("reconstruction loss in original space: {}".format(average_loss_original_space))
    average_loss_global_space = np.sum(losses_global_space) / n_batches
    print("reconstruction loss in global space: {}".format(average_loss_global_space))

    ####
    print("##########################")
    print("model with standard loss: ")
    model_name = r'D:\workspace\my_git_repos\vae_motion_modeling\data\models\h36m_frameEnc_angle\h36m_frameEnc_angle_0100.ckpt'
    normal_encoder = FrameEncoder(dropout_rate=0.1)
    normal_encoder.load_weights(model_name)
    losses_original_space = []
    losses_global_space = []
    for i in range(n_batches):
        input_batch = normalized_data[batchsize * i: batchsize * (i+1)]
        res = normal_encoder(input_batch)
        reconstructed_data = res * std_value + mean_value

        reconstructed_data = reconstructed_data.numpy()

        losses_original_space.append(mean_squared_error(h36m_data[batchsize * i : batchsize * (i+1)], reconstructed_data))

        reconstructed_global_position = get_global_position_framewise(reconstructed_data)
        reconstructed_global_position = np.reshape(reconstructed_global_position, (reconstructed_global_position.shape[0], np.prod(reconstructed_global_position.shape[1:])))
        original_global_position = get_global_position_framewise(h36m_data[batchsize * i : batchsize * (i+1)])
        original_global_position = np.reshape(original_global_position, (original_global_position.shape[0], np.prod(original_global_position.shape[1:])))
        losses_global_space.append(mean_squared_error(original_global_position, reconstructed_global_position))

    average_loss_original_space = np.sum(losses_original_space) / n_batches
    print("reconstruction loss in original space: {}".format(average_loss_original_space))
    average_loss_global_space = np.sum(losses_global_space) / n_batches
    print("reconstruction loss in global space: {}".format(average_loss_global_space))



def qualitive_evaluation():
    ### angular representation
    h36m_data = get_training_data()
    h36m_data = np.reshape(h36m_data, (h36m_data.shape[0] * h36m_data.shape[1], h36m_data.shape[2]))
    ###
    mean_value = h36m_data.mean(axis=0)[np.newaxis, :]
    std_value = h36m_data.std(axis=0)[np.newaxis, :]
    std_value[std_value<EPS] = EPS

    normalized_data = (h36m_data - mean_value) / std_value    
    n_frames = len(normalized_data)
    batchsize = 256
    n_batches = n_frames // batchsize

    ### reconstruct training data
    starting_frame = 1000
    frame_length = 1000
    scale_factor = 5

    ### reconstruct bvh file
    test_file = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\h36m\S1\Walking.bvh'
    input_data = process_bvhfile(test_file, sliding_window=False)
    normalized_input_data = (input_data - mean_value) / std_value    


    #### reconstruct with fine-tuned custom-loss model
    ### load model
    model_name = "h36m_frameEncoder_customized_loss"
    encoder = FrameEncoder(dropout_rate=0.1, name=model_name)
    ### original model
    # model_name = r'D:\workspace\my_git_repos\vae_motion_modeling\data\models\h36m_frameEnc_angle\h36m_frameEnc_angle_0100.ckpt'
    # encoder.load_weights(model_name)
    ### use fine-tuned model
    pretrained_model = r'E:\results\h36m_frameEncoder_customized_loss\h36m_frameEncoder_customized_loss_fine_tuning_0100.ckpt'
    encoder.load_weights(pretrained_model)

    # res = encoder(normalized_data[starting_frame: starting_frame + frame_length]).numpy()
    res = encoder(normalized_input_data).numpy()

    ### reconstruct motion
    reconstructed_motion = res * std_value + mean_value
    # reconstructed_motion = reconstructed_motion.numpy()
    reconstructed_global_position = get_global_position_framewise(reconstructed_motion)
    reconstructed_global_position *= scale_factor
    reconstructed_motion_data = {'motion_data': reconstructed_global_position.tolist(), 'has_skeleton': True,
                                 'skeleton': MH_CMU_SKELETON}
    # write_to_json_file(r'E:\tmp\recon_frameEncoder_standard_model.panim', reconstructed_motion_data)
    write_to_json_file(r'E:\tmp\recon_frameEncoder_fine_tuned_model_walk.panim', reconstructed_motion_data)
    
    # ### customized model
    # model_name = r'E:\results\h36m_frameEncoder_customized_loss\h36m_frameEncoder_customized_loss_0100.ckpt'
    # encoder.load_weights(model_name)
    # res = encoder(normalized_data[starting_frame: starting_frame + frame_length])

    # ### reconstruct motion
    # reconstructed_motion = res * std_value + mean_value
    # reconstructed_motion = reconstructed_motion.numpy() 
    # reconstructed_global_position = get_global_position_framewise(reconstructed_motion)
    # reconstructed_global_position *= scale_factor
    # # reconstructed_global_position = np.reshape(reconstructed_global_position, (reconstructed_global_position.shape[0], np.prod(reconstructed_global_position.shape[1:])))
    # reconstructed_motion_data = {'motion_data': reconstructed_global_position.tolist(), 'has_skeleton': True,
    #                              'skeleton': MH_CMU_SKELETON}
    # write_to_json_file(r'E:\tmp\recon_frameEncoder_customized_model.panim', reconstructed_motion_data)

    # ### reconstruct oroginal motion
    # original_motion = normalized_data[starting_frame : starting_frame + frame_length] * std_value + mean_value
    # reconstructed_global_position = get_global_position_framewise(original_motion)
    # reconstructed_global_position *= scale_factor
    # reconstructed_motion_data = {'motion_data': reconstructed_global_position.tolist(), 'has_skeleton': True,
    #                              'skeleton': MH_CMU_SKELETON}
    # write_to_json_file(r'E:\tmp\original_motion.panim', reconstructed_motion_data)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        type=str,
                        default=r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\h36m\S1\Walking.bvh')
    parser.add_argument("--save_path",
                        type=str,
                        default=r'E:\tmp')
    args = parser.parse_args()

    # run_frameEncoder()
    run_frameEncoder_quat(args.input_file, args.save_path)
    # run_frameEncoder_euler()
    # test()
    # run_frameEncoderDropuoutFirst_quat()
    # quantitive_evaluation()
    # qualitive_evaluation()
    # run_frameEncoder1()