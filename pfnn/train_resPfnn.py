import numpy as np
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from pfnn.ResPFNN import ResPFNN
from pfnn.train_vanilla_pfnn import get_training_data
import tensorflow as tf
import argparse


def train_diagonal_resPfnn():
    target_style = "angry"
    model_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground\network.npz'
    training_data_path = r'D:\workspace\projects\variational_style_simulation\training_data\pfnn_preprocessing\mk_cmu_database' + '_' + target_style + '.npz'

    meta_data_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground'
    save_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\mk_cmu_ground_residual\diagonal'
    input_data, output_data = get_training_data(training_data_path, meta_data_path)
    n_controls = 4
    batchsize = 32
    dropout = 0.7
    n_epoches = 300
    style_dims = 30
    model = ResPFNN(n_controls, input_data.shape[1], output_data.shape[1], style_dims, dropout, batchsize)
    # model.create_model()
    model.create_model_diagonal()

    database = np.load(model_path)
    model.load_params_from_theano(database)
    # model.train(input_data, output_data, n_epoches=100)
    model.style_fine_turning(input_data, output_data, n_epoches=n_epoches, learning_rate=1e-3)
    model.save_model(r'trained_models/finetuned_diagonal' + target_style + '_' + str(n_epoches) +  '.ckpt')
    save_folder = os.path.join(save_path, target_style, str(n_epoches))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model.save_params(save_folder, num_points=50)
    # model.save_style_params(save_folder)
    model.save_style_diagonal_params(save_folder)


    # model.create_test_model()
    # model.train_model(input_data, output_data, n_epoches=100)


def train_resPfnn(style, model_file, meta_path, input_data, save_path):
    """train a residual PFNN based on pre-trained model
    
    Arguments:
        style {str} -- target style
        model_file {str} -- path to pretrained model
        meta_path {str} -- path to Mean and Variance data
        input_data {str} -- path to input data
        save_path {str} -- save path
    """
    input_data, output_data = get_training_data(input_data, meta_path)
    n_controls = 4
    batchsize = 32
    dropout = 0.7
    n_epoches = 300
    style_dims = 30
    ## load pretrained model
    model = ResPFNN(n_controls, input_data.shape[1], output_data.shape[1], style_dims, dropout, batchsize)
    model.create_model()

    database = np.load(model_file)
    model.load_params_from_theano(database) 
    model.style_fine_turning(input_data, output_data, n_epoches=n_epoches, learning_rate=1e-3)
    save_folder = os.path.join(save_path, style, str(n_epoches))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model.save_params(save_folder, num_points=50)
    model.save_style_params(save_folder)


def train_resPfnn_local():
    target_style = "angry"
    model_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground\network.npz'
    training_data_path = r'D:\workspace\projects\variational_style_simulation\training_data\pfnn_preprocessing\mk_cmu_database' + '_' + target_style + '.npz'

    meta_data_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground'
    save_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\mk_cmu_ground_residual'
    train_resPfnn(target_style, model_path, meta_data_path, training_data_path, save_path)


def train_resPfnn_cmd():
    """train a residual PFNN based on pre-trained model
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-style", 
                        "--target_style", 
                        type=str,
                        required=True)
    parser.add_argument("-model_file", 
                        "--model_file", 
                        type=str,
                        required=True)   
    parser.add_argument("-meta_path", 
                        "--meta_path", 
                        type=str,
                        required=True)   
    parser.add_argument("-input_data", 
                        "--input_data", 
                        type=str,
                        required=True)                                                                  
    parser.add_argument("-save_path", 
                        "--save_path", 
                        type=str,
                        required=True)
    args = parser.parse_args()
    train_resPfnn(args.target_style,
                  args.model_file,
                  args.meta_path,
                  args.input_data,
                  args.save_path)


if __name__ == "__main__":
    train_resPfnn_cmd()
    # train_diagonal_resPfnn()