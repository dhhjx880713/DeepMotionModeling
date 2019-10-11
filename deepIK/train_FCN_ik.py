import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from models.fcn_ik import FCN_IK
import numpy as np
from mosi_utils_anim.utilities import load_json_file
from mosi_utils_anim.animation_data import BVHReader, BVHWriter, SkeletonBuilder
from mosi_utils_anim.animation_data.utils import pose_orientation_general, convert_euler_frame_to_cartesian_frame, shift_euler_frames_to_ground


def train_4_layers_fcn():
    """a four-layer fcn mapping joint point over multiple previous frames to relative euler angles of current frame
    """
    ### load training data
    training_data = load_json_file(r'E:\workspace\projects\mesh_retargeting\training_data\point_cloud_plus_euler_frame\style_training_data.json')
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
    ### construct model
    fcn_ik_4_layer = FCN_IK(name='fcn_ik_4_layer')
    fcn_ik_4_layer.build(input_shape=93, output_shape=96)
    ### training
    epochs = 1000
    training_rate = 1e-05
    fcn_ik_4_layer.train(X_normalized, Y_normalized, epochs, training_rate)

    fcn_ik_4_layer.save(r'trained_models/fcn_4_layers_ik.ckpt')



def train_4_layers_fcn_multiframes():
    
    window_len = 10

if __name__ == "__main__":
    train_4_layers_fcn()
    