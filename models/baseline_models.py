import tensorflow as tf 
import numpy as np 
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from mosi_utils_anim.utilities import write_to_json_file


def load_data():
    data = np.load(r'D:\workspace\my_git_repos\deepMotionSynthesis\data\training_data\cmu_skeleton\h36m.npz')['clips']
    return data


class FCNRegressor(tf.keras.Model):

    def __init__(self):
        super(FCNRegressor, self).__init__()



class RNNRegressor(tf.keras.Model):

    def __init__(self):
        super(RNNRegressor, self).__init__()
        self.lstm = tf.keras.layers.LSTM(128)
    
    def call(self, inputs, training=False):
        res = self.lstm(inputs)
        return res


def test():
    data = load_data()
    print(data.shape)


if __name__ == "__main__":
    test()