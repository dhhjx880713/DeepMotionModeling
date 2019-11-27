import numpy as np
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from pfnn.pfnn import PFNN



def get_training_data(training_data_path, meta_data_path=None, save_path=None):
    """prepare training data for pfnn
    
    Arguments:
        training_data_path {string} -- path to .npz file
        meta_data_path {[type]} -- [description]
        save_path {[type]} -- [description]
    """

    training_data = np.load(training_data_path)
    if not save_path is None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    X = training_data['Xun']
    Y = training_data['Yun']
    P = training_data['Pun']
    if meta_data_path is None:
        """ Calculate Mean and Std """

        Xmean, Xstd = X.mean(axis=0), X.std(axis=0)
        Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)

        j = 31
        w = ((60*2)//10)

        Xstd[w*0:w* 1] = Xstd[w*0:w* 1].mean() # Trajectory Past Positions
        Xstd[w*1:w* 2] = Xstd[w*1:w* 2].mean() # Trajectory Future Positions
        Xstd[w*2:w* 3] = Xstd[w*2:w* 3].mean() # Trajectory Past Directions
        Xstd[w*3:w* 4] = Xstd[w*3:w* 4].mean() # Trajectory Future Directions
        Xstd[w*4:w*10] = Xstd[w*4:w*10].mean() # Trajectory Gait

        """ Mask Out Unused Joints in Input """
        ## this is modified because the joint order is changed
        joint_weights = np.array([
            1,
            1e-10, 1, 1, 1, 1,  ## left leg, LHipJoint has zero length
            1, 1, 1,
            1, 1, 1, 1, 1e-10, 1e-10, 1e-10,
            1, 1, 1,
            1, 1, 1, 1, 1e-10, 1e-10, 1e-10,
            1e-10, 1, 1, 1, 1]).repeat(3)

        Xstd[w*10+j*3*0:w*10+j*3*1] = Xstd[w*10+j*3*0:w*10+j*3*1].mean() / (joint_weights * 0.1) # Pos
        Xstd[w*10+j*3*1:w*10+j*3*2] = Xstd[w*10+j*3*1:w*10+j*3*2].mean() / (joint_weights * 0.1) # Vel
        Xstd[w*10+j*3*2:          ] = Xstd[w*10+j*3*2:          ].mean() # Terrain

        Ystd[0:2] = Ystd[0:2].mean() # Translational Velocity
        Ystd[2:3] = Ystd[2:3].mean() # Rotational Velocity
        Ystd[3:4] = Ystd[3:4].mean() # Change in Phase
        Ystd[4:8] = Ystd[4:8].mean() # Contacts

        Ystd[8+w*0:8+w*1] = Ystd[8+w*0:8+w*1].mean() # Trajectory Future Positions
        Ystd[8+w*1:8+w*2] = Ystd[8+w*1:8+w*2].mean() # Trajectory Future Directions        

        Ystd[8+w*2+j*3*0:8+w*2+j*3*1] = Ystd[8+w*2+j*3*0:8+w*2+j*3*1].mean() # Pos
        Ystd[8+w*2+j*3*1:8+w*2+j*3*2] = Ystd[8+w*2+j*3*1:8+w*2+j*3*2].mean() # Vel
        Ystd[8+w*2+j*3*2:8+w*2+j*3*3] = Ystd[8+w*2+j*3*2:8+w*2+j*3*3].mean() # Rot

        """ Save Mean / Std / Min / Max """

        Xmean.astype(np.float32).tofile(os.path.join(save_path, 'Xmean.bin'))
        Ymean.astype(np.float32).tofile(os.path.join(save_path, 'Ymean.bin'))
        Xstd.astype(np.float32).tofile(os.path.join(save_path, 'Xstd.bin'))
        Ystd.astype(np.float32).tofile(os.path.join(save_path, 'Ystd.bin'))

    else:
        """load precomputed mean and std
        """
        Xmean = np.fromfile(os.path.join(meta_data_path, 'Xmean.bin'), dtype=np.float32)
        Xstd = np.fromfile(os.path.join(meta_data_path, 'Xstd.bin'), dtype=np.float32)
        Ymean = np.fromfile(os.path.join(meta_data_path, 'Ymean.bin'), dtype=np.float32)
        Ystd = np.fromfile(os.path.join(meta_data_path, 'Ystd.bin'), dtype=np.float32)

    X = (X - Xmean) / Xstd
    Y = (Y - Ymean) / Ystd

    input_data = np.concatenate((X, P[...,np.newaxis]), axis=-1)
    output_data = Y
    return input_data, output_data

    
def train_pfnn():
    training_data_path = r'./mk_cmu_database.npz'
    meta_data_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\no_local_rot'
    save_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn'

    input_data, output_data = get_training_data(training_data_path, meta_data_path, save_path)
    n_controls = 4
    batchsize = 256
    dropout = 0.7
    model = PFNN(n_controls, input_data.shape[1], output_data.shape[1], dropout, batchsize)
    model.create_model()
    print("training start")
    model.train(input_data, output_data, n_epoches=10)
    model.save_model(r'trained_models/pfnn_no_rot_256.ckpt')
    # model.save_params()


def finetune_pfnn():
    # training_data_path = r'D:\gits\PFNN\mk_cmu_database_ground.npz'
    # target_style = "proud"
    # target_style = "childlike"
    target_style = "old"
    # target_style = 'sexy' ## to be continued
    n_epoches = 10
    # training_data_path = r'D:\workspace\projects\variational_style_simulation\training_data\pfnn_preprocessing\mk_cmu_database' + '_' + target_style + '.npz'
    training_data_path = r'D:\gits\PFNN\mk_cmu_' + target_style + '.npz'
    meta_data_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground'
    save_path = os.path.join(r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\mk_cmu_ground_finetuning', target_style+'_'+str(n_epoches))
    model_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground\network.npz'

    input_data, output_data = get_training_data(training_data_path, meta_data_path, save_path)
    n_controls = 4
    batchsize = 32
    dropout = 0.7  
    model = PFNN(n_controls, input_data.shape[1], output_data.shape[1], dropout, batchsize)
    # model = PFNN(n_controls, 343, 311, dropout, batchsize)
    model.create_model()
    # model.load_model(model_path)

    database = np.load(model_path)
    model.load_params_from_theano(database)
    print("The existing model is loaded!")
    for i in range(n_epoches):
        model.train(input_data, output_data, n_epoches=1, learning_rate=1e-4, fine_turning=True)
        model.save_model(r'trained_models/finetuned.ckpt')
    model.save_params(save_path, num_points=50)


if __name__ == "__main__":
    # train_pfnn()
    finetune_pfnn()
