import os
import sys
import glob
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder, BVHWriter
from mosi_utils_anim.animation_data.utils import pose_orientation_general, convert_euler_frame_to_cartesian_frame, shift_euler_frames_to_ground
from deepIK.create_training_data import process_data
from models.network import Network

rng = np.random.RandomState(123456)
"""try to predict global tranform from global position

1. the data should be normalized
"""


class FullyConnectedIKNetwork(Network):

    def __init__(self, name, sess=None):
        super(FullyConnectedIKNetwork, self).__init__(name, sess)
    
    def build(self, input_shape, output_shape, reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope(self.name, reuse):
            self.input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,)+input_shape)
            self.output = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, output_shape))
            flatten = tf.compat.v1.layers.flatten(self.input, name="flatten")
            layer1_out = tf.compat.v1.layers.dense(flatten, 128, activation=tf.nn.elu, name='layer1', reuse=reuse)
            layer2_dropout = tf.compat.v1.layers.dropout(layer1_out, rate=0.5)
            layer2_out = tf.compat.v1.layers.dense(layer2_dropout, 256, activation=tf.nn.elu, name='layer2', reuse=reuse)
            layer3_dropout = tf.compat.v1.layers.dropout(layer2_out, rate=0.5)
            layer3_out = tf.compat.v1.layers.dense(layer3_dropout, 128, activation=tf.nn.elu, name='layer3', reuse=reuse)
            layer4_dropout = tf.compat.v1.layers.dropout(layer3_out, rate=0.5)
            self.out_op = tf.compat.v1.layers.dense(layer4_dropout, output_shape, name='layer4', reuse=reuse)
            self.params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            self.saver = tf.compat.v1.train.Saver(self.params)
            self.cost = tf.reduce_mean(input_tensor=tf.pow(self.out_op - self.output, 2))
                                     
def create_network(input_shape, output_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation=tf.nn.elu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation=tf.nn.elu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation=tf.nn.elu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_shape)
    ])
    return model    

def evaluate_model():
    model_file = r'E:\workspace\projects\mesh_retargeting\tmp\sequential_1_0.0001_300'
    input_shape = (31, 3)
    output_shape = 96
    network = create_network(input_shape, output_shape)
    network.load_weights(model_file)
    
    ### evaluate 
    training_data_file = r'E:\workspace\projects\mesh_retargeting\training_data\point_cloud_plus_euler_frame\style_training_data.npz'
    training_data = np.load(training_data_file)
    X = training_data['X']
    Y = training_data['Y']
    print(X.shape)
    print(Y.shape)

    output = network.predict(X[0:100])
    # print(output.shape)
    test_file = r'E:\workspace\mocap_data\mk_cmu_retargeting\ACCAD\Female1_bvh\Female1_B03_Walk1.bvh'
    torso_joints = ['LeftUpLeg', 'LowerBack', 'RightUpLeg']
    # cartesian_frames, euler_frames = process_data(test_file, torso_joints=torso_joints)
    # output = network.predict(cartesian_frames)

    ### export frames
    bvhskeleotn_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_skeleton.bvh'
    bvhreader = BVHReader(bvhskeleotn_file)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    foot_contact_joints = ['LeftFoot', 'RightFoot']
    # euler_frames = shift_euler_frames_to_ground(euler_frames, foot_contact_joints, skeleton)
    output = shift_euler_frames_to_ground(output, foot_contact_joints, skeleton)
    
    # BVHWriter(r'E:\workspace\projects\mesh_retargeting\tmp\Female1_B03_Walk1.bvh', skeleton, euler_frames,
    #     skeleton.frame_time, is_quaternion=False)

    BVHWriter(r'E:\workspace\projects\mesh_retargeting\tmp\example.bvh', skeleton, output,
        skeleton.frame_time, is_quaternion=False)


def train_model():
    training_data_file = r'E:\workspace\projects\mesh_retargeting\training_data\point_cloud_plus_euler_frame\style_training_data.npz'
    training_data = np.load(training_data_file)

    input_shape = (31, 3)
    output_dims = 96
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation=tf.nn.elu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation=tf.nn.elu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation=tf.nn.elu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(96)
    ], name='4_layer_mlp_euler_pose')

    learning_rate = 1e-4
    epochs = 1000
    adam_opt = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam_opt, loss='mean_squared_error', metrics=['accuracy'])
    model.fit(training_data['X'], training_data['Y'], epochs=epochs)
    # model.weights
    model_name = '_'.join([model.name, str(learning_rate), str(epochs)])
    save_path = r'E:\workspace\projects\mesh_retargeting\tmp'
    model.save(os.path.join(save_path, model_name))
    # model.load_weights()

def test():
    file1 = r'E:\workspace\repo\data\1 - MoCap\4 - Alignment\elementary_action_neutralWalk\leftStance\walk_001_1_leftStance_558_617_mirrored_from_rightStance.bvh'

    file2 = r'E:\workspace\repo\data\1 - MoCap\4 - Alignment\elementary_action_neutralWalk\leftStance\walk_001_2_leftStance_420_460_mirrored_from_rightStance.bvh'
    bvhreader1 = BVHReader(file1)
    bvhreader2 = BVHReader(file2)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader1)

    root_trans1 = skeleton.nodes['FK_back1_jnt'].get_global_matrix_from_euler_frame(bvhreader1.frames[0])
    root_trans2 = skeleton.nodes['FK_back1_jnt'].get_global_matrix_from_euler_frame(bvhreader2.frames[0])
    print(root_trans1)
    print(root_trans2)


def data_normalization():
    """find out a way to represent joint position in global translation and rotation invariant way
    idea 1: the most straightforward way is to remove global rotaiton matrix of root joint
    """
    test_file = r'E:\workspace\projects\cGAN\processed_data\ACCAD\Male1_bvh_Male1_A12_CrawlBackward.bvh'
    bvhreader = BVHReader(test_file)
    filename = os.path.split(test_file)[-1]
    torso_joints = ['LeftUpLeg', 'LowerBack', 'RightUpLeg']
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    new_frames = []
    for frame in bvhreader.frames:
        frame[:6] = 0.0
        new_frames.append(frame)
    BVHWriter(os.path.join(r'E:\workspace\projects\cGAN\tmp', filename), skeleton, new_frames, skeleton.frame_time,
        is_quaternion=False)



def test_forward_direction():
    test_file = r'E:\workspace\projects\cGAN\processed_data\ACCAD\Male1_bvh_Male1_A12_CrawlBackward.bvh'
    bvhreader = BVHReader(test_file)
    torso_joints = ['LeftUpLeg', 'LowerBack', 'RightUpLeg']
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    forward1 = pose_orientation_general(bvhreader.frames[0], torso_joints, skeleton)
    print(forward1)
    """holden
    """
    global_positions = convert_euler_frame_to_cartesian_frame(skeleton, bvhreader.frames[0])

    sdr_l, sdr_r, hip_l, hip_r = 10, 20, 2, 27
    across = (
        (global_positions[sdr_l] - global_positions[sdr_r]) + 
        (global_positions[hip_l] - global_positions[hip_r]))
    across = across / np.linalg.norm(across)
    forward2 = np.cross(across, np.array([0, 1, 0]))
    print(forward2)

def another_train():
    training_data_file = r'E:\workspace\projects\mesh_retargeting\training_data\point_cloud_plus_euler_frame\style_training_data.npz'
    training_data = np.load(training_data_file)
    X = training_data['X']
    Y = training_data['Y']
    input_shape = (31, 3)
    output_dims = 96
    sess = tf.compat.v1.InteractiveSession()
    model = FullyConnectedIKNetwork("IK", sess=sess)
    model.build(input_shape, output_dims)
    model.train(X, Y, epochs=1, learning_rate=1e-4, batchsize=32)
    model.save(r'trained_models\mymodel.ckpt')
    params = model.get_params()
    print(model.sess.run("IK/layer1/bias:0"))

    new_model = FullyConnectedIKNetwork("IK", sess=sess)
    new_model.build(input_shape, output_dims)
    new_model.load(r'trained_models\mymodel.ckpt')
    loaded_parameters = model.get_params()
    print(new_model.sess.run("IK/layer1/bias:0"))

def another_model_evaluate():
    input_shape = (31, 3)
    output_dims = 96
    model = FullyConnectedIKNetwork("IK")
    model.build(input_shape, output_dims)
    model.load(r'trained_models\mymodel.ckpt')
    params = model.get_params()
    # print(params[0])
    for item in params:
        print(item.name)
        print(model.sess.run(item.name))

if __name__ == "__main__":
    # get_training_data()
    # test_forward_direction()
    # data_normalization()
    train_model()
    # evaluate_model()
    # another_train()
    # another_model_evaluate()