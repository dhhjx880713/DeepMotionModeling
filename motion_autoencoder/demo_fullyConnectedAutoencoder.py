import os
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from models.fully_connected_autoencoder import FullyConnectedEncoder



def demo_fullyConnectedEncoder():
    input_data = np.load(r'./data/training_data/h36m_cmu.npz')['clips']
    reshaped_input_data = np.reshape(input_data, (input_data.shape[0], input_data.shape[1], np.prod(input_data.shape[2:])))
    print(reshaped_input_data.shape)
    Xmean = 0
    Xstd = 0
    # encoder = FullyConnectedEncoder()
    # encoder.train()
    # encoder.pre_train()


if __name__ == "__main__":
    demo_fullyConnectedEncoder()