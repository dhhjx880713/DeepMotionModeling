import numpy as np
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from pfnn.ResPFNN import ResPFNN
from pfnn.pfnn import PFNN
from pfnn.train_vanilla_pfnn import get_training_data
import tensorflow as tf


def evaluate_model_error():
    n_controls = 4
    batchsize = 32
    dropout = 0.7  
    ### load pfnn model
    model_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground\network.npz'
    model = PFNN(n_controls, 343, 311, dropout, batchsize)
    model.create_model()
    database = np.load(model_path)
    model.load_params_from_theano(database)

    ## load training data
    motion_data = r'D:\gits\PFNN\mk_cmu_database_ground.npz'
    meta_data_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground'
    input_data, output_data = get_training_data(motion_data, meta_data_path=meta_data_path)
    print(input_data.shape)
    print(output_data.shape)

    print(model.sess.run(model.loss, 
    feed_dict={model.input: input_data[:256], model.Y: output_data[:256]}))


    ## load style data
    target_style = "childlike"
    training_data_path = r'D:\workspace\projects\variational_style_simulation\training_data\pfnn_preprocessing\mk_cmu_database' + '_' + target_style + '.npz'
    style_input, style_output = get_training_data(training_data_path, meta_data_path)
    print(style_input.shape)
    print(style_output.shape)
    print(model.sess.run(model.loss, 
    feed_dict={model.input: style_input[:256], model.Y: style_output[:256]}))


def evaluate_res_pfnn_error():
    n_controls = 4
    batchsize = 32
    dropout = 0.7  
    vanilla_model_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground\network.npz'
    ### load ResPFNN
    #  
    model = ResPFNN(n_controls, 343, 311, 30, dropout, batchsize)
    model.create_model()
    # database = np.load(model_path)
    # model.load_params_from_theano(database)
    target_style = "angry"
    n_epoches = 300
    model.load_model(r'trained_models/finetuned_' + target_style + '_' + str(n_epoches) +  '.ckpt')
    # model.sess.run(tf.variables_initializer(model.style_params))

    ### load vanilla model
    # model = PFNN(n_controls, 343, 311, dropout, batchsize)
    # model.create_model()
    # database = np.load(model_path)
    # model.load_params(database)

    target_style = "angry"
    model_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground\network.npz'
    training_data_path = r'D:\workspace\projects\variational_style_simulation\training_data\pfnn_preprocessing\mk_cmu_database' + '_' + target_style + '.npz'
    meta_data_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground'
    style_input, style_output = get_training_data(training_data_path, meta_data_path)
    print(model.sess.run(model.loss, feed_dict={model.input: style_input[:256], model.Y: style_output[:256]}))

    ## load training data
    motion_data = r'D:\gits\PFNN\mk_cmu_database_ground.npz'
    meta_data_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground'
    input_data, output_data = get_training_data(motion_data, meta_data_path=meta_data_path)
    print(input_data.shape)
    print(output_data.shape)
    print("original data loss: ")
    print(model.sess.run(model.loss, 
    feed_dict={model.input: input_data[:256], model.Y: output_data[:256]}))    


def evaluate_diag_pfnn_error():
    target_style = "angry"
    training_data_path = r'D:\workspace\projects\variational_style_simulation\training_data\pfnn_preprocessing\mk_cmu_database' + '_' + target_style + '.npz'
    meta_data_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground'
    style_input, style_output = get_training_data(training_data_path, meta_data_path)   


    motion_data = r'D:\gits\PFNN\mk_cmu_database_ground.npz' 
    style_input, style_output = get_training_data(motion_data, meta_data_path=meta_data_path)


    n_controls = 4
    batchsize = 32
    dropout = 0.7    
    ### load vanilla model and evaluate error on stylistic training data
    g1 = tf.Graph()
    with g1.as_default() as g:
        vanilla_model_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground\network.npz'
        
        vanilla_model = PFNN(n_controls, 343, 311, dropout, batchsize, name="pfnn")
        vanilla_model.create_model()
        database = np.load(vanilla_model_path)
        vanilla_model.load_params_from_theano(database)
        print("vanilla model loss: ")
        print(vanilla_model.sess.run(vanilla_model.loss,  feed_dict={vanilla_model.input: style_input[:256], vanilla_model.Y: style_output[:256]}))       

    ### using vanilla parameters in residual model and evaluate error on stylistic training data
    g2 = tf.Graph()
    with g2.as_default() as g:
        target_style = "angry"
        n_epoches = 300
        style_dims = 30
        style_model = ResPFNN(n_controls, style_input.shape[1], style_output.shape[1], style_dims, dropout, batchsize, name="respfnn")
        style_model.create_model_diagonal()    
        # print(database.keys())
        style_model.load_params_from_theano(database)
        style_model.sess.run(tf.variables_initializer(style_model.style_params))
        print(style_model.sess.run(style_model.loss,  feed_dict={style_model.input: style_input[:256], style_model.Y: style_output[:256]}))  

    
    
    ### load non-diagonal residual model and evaluate error on stylistic training data
    style_path = r''
    g3 = tf.Graph()
    with g3.as_default() as g:
        target_style = "angry"
        n_epoches = 300
        style_dims = 30
        style_model = ResPFNN(n_controls, style_input.shape[1], style_output.shape[1], style_dims, dropout, batchsize)
        style_model.create_model()
        style_model.load_params_from_theano(database)
        style_model.load_style_params(r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\mk_cmu_ground_finetuning\angry\300')
        print("residual model loss: ")
        print(style_model.sess.run(style_model.loss,  feed_dict={style_model.input: style_input[:256], style_model.Y: style_output[:256]}))


def load_model():
    target_style = "angry"
    training_data_path = r'D:\workspace\projects\variational_style_simulation\training_data\pfnn_preprocessing\mk_cmu_database' + '_' + target_style + '.npz'
    meta_data_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground'
    style_input, style_output = get_training_data(training_data_path, meta_data_path)
    n_controls = 4
    batchsize = 32
    dropout = 0.7 

    vanilla_model_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground\network.npz'
  
    # vanilla_model = PFNN(n_controls, 343, 311, dropout, batchsize)
    # vanilla_model.create_model()
    # database = np.load(vanilla_model_path)
    # vanilla_model.load_params_from_theano(database)

    n_epoches = 300
    style_dims = 30
    model = ResPFNN(n_controls, style_input.shape[1], style_output.shape[1], style_dims, dropout, batchsize)
    model.create_model_diagonal()
    # model.load_model(r'trained_models/finetuned_diagonal' + target_style + '_' + str(n_epoches) +  '.ckpt')
    # print("diagonal residual model loss: ")
    # print(model.sess.run(model.loss,  feed_dict={model.input: style_input[:256], model.Y: style_output[:256]}))


if __name__ == "__main__":
    # evaluate_model_error()
    # evaluate_res_pfnn_error()
    evaluate_diag_pfnn_error()
    # load_model()