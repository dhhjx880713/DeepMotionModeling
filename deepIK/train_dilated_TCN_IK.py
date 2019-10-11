import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from models.dilated_TCN_IK import DilatedTCN_IK
import numpy as np
from mosi_utils_anim.utilities import load_json_file
from mosi_utils_anim.animation_data import BVHReader, BVHWriter, SkeletonBuilder
from mosi_utils_anim.animation_data.utils import pose_orientation_general, convert_euler_frame_to_cartesian_frame, shift_euler_frames_to_ground


def train_model():
    training_data = load_json_file(r'D:\workspace\projects\mesh_retargeting\training_data\point_cloud_plus_euler_frame\style_training_data.json')
    X = training_data['X']
    Y = training_data['Y']
    Xmean = np.asarray(training_data['Xmean'])
    Xstd = np.asarray(training_data['Xstd'])

    Ymean = np.asarray(training_data['Ymean'])
    Ystd = np.asarray(training_data['Ystd'])
    Xstd[:] = Xstd.mean()
    Ystd[:] = Ystd.mean()

    X_normalized = []
    Y_normalized = []

    for i in range(len(X)):
        X_normalized.append((np.asarray(X[i]) - Xmean) / Xstd)
        Y_normalized.append((np.asarray(Y[i]) - Ymean) / Ystd)
    dilated_tcn_ik = DilatedTCN_IK(name='tcn_ik')
    dilated_tcn_ik.build(input_shape=93, output_shape=96)    
    epochs = 1000
    training_rate = 1e-05
    dilated_tcn_ik.train(X_normalized, Y_normalized, epochs, training_rate)
    dilated_tcn_ik.save(r'trained_models/forward_ik.ckpt')


    # input = np.array(X[0])
    # input = np.reshape(input, [1, input.shape[0], np.prod(input.shape[1:])])
    # res = dilated_tcn_ik.predict(input)
    # print(res.shape)

if __name__ == "__main__":
    train_model()
