import numpy as np
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from pfnn.pfnn import PFNN



def discrete_export_pfnn():
    model_file = r'trained_models/pfnn_no_rot.ckpt'
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


if __name__ == "__main__":
    discrete_export_pfnn()