import numpy as np
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from pfnn.pfnn import PFNN


def train_pfnn():
    training_data = np.load(r'E:\workspace\projects\variational_style_simulation\pfnn\origin_data\training_data\vanilla\database.npz')
    X = training_data['Xun']
    Y = training_data['Yun']
    P = training_data['Pun']

    # load mean and standard derivation
    param_folder = r'E:\workspace\projects\variational_style_simulation\pfnn\origin_data\training_data\vanilla'
    Xmean = np.fromfile(os.path.join(param_folder, 'Xmean.bin'), dtype=np.float32)
    Xstd = np.fromfile(os.path.join(param_folder, 'Xstd.bin'), dtype=np.float32)
    Ymean = np.fromfile(os.path.join(param_folder, 'Ymean.bin'), dtype=np.float32)
    Ystd = np.fromfile(os.path.join(param_folder, 'Ystd.bin'), dtype=np.float32)

    X = (X - Xmean) / Xstd
    Y = (Y - Ymean) / Ystd

    input_data = np.concatenate((X, P[...,np.newaxis]), axis=-1)
    print(input_data.shape)
    print(Y.shape)
    model = PFNN(4, 343, 311, 0.7, 32)
    # model.create_model()
    model.create_model1()
    model.train(input_data, Y, n_epoches=10)


if __name__ == "__main__":
    train_pfnn()