
import os
import sys
import numpy as np
import tensorflow as tf
dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dirname, '..'))
from models.fully_connected_autoencoder import FullyConnectedEncoder



def train_FullyConnectedEncoder():
    npc = 10
    name = 'fullyConnectedEncoder'
    input_dim = 90
    data = np.random.rand(100, 90)
    model = FullyConnectedEncoder(npc=npc, input_dim=input_dim, name=name)
    model.create_model()
    # res = model(data)
    # print(res.shape)
    # model.train(data, n_epochs=10)
    res = model(data)
    print(res.shape)

if __name__ == "__main__":
    train_FullyConnectedEncoder()    