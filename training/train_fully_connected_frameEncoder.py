import os
from posixpath import dirname
import sys
import numpy as np 
from pathlib import Path

from tensorflow.python.ops.linalg_ops import norm
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
dirname = os.path.dirname(os.path.abspath(__file__))
from utilities.utils import get_files, get_rotation_to_ref_direction, rotate_cartesian_frame, estimate_ground_height, reconstruct_global_position
from utilities.skeleton_def import MH_CMU_SKELETON
from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder
from mosi_utils_anim.animation_data.utils import convert_euler_frames_to_cartesian_frames
from mosi_utils_anim.utilities import write_to_json_file
from preprocessing.utils import rotate_cartesian_frames_to_ref_dir, cartesian_pose_orientation
from mosi_utils_anim.animation_data.quaternion import Quaternion
from utilities.quaternions import Quaternions
from models.frame_encoder import FrameEncoder
from models.fullBody_pose_encoder import FullBodyPoseEncoder
import tensorflow as tf
from tensorflow.keras import Sequential, callbacks, layers, Model, optimizers, losses
from tensorflow_graphics.geometry.transformation.quaternion import between_two_vectors_3d, from_axis_angle, rotate
import time
from train_frameEncoder import get_training_data

EPS = 1e-6
BATCHSIZE = 128


def process_file(bvhfile, animated_joints):
    bvhreader = BVHReader(bvhfile)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    cartesian_frames = convert_euler_frames_to_cartesian_frames(skeleton, bvhreader.frames, animated_joints=animated_joints)
    print(cartesian_frames.shape)
    ref_dir = np.array([0, 0, 1])
    body_plane_indices=[7, 25, 1]
    up_axis = np.array([0, 1, 0])
    forward = []
    for i in range(len(cartesian_frames)):
        forward.append(cartesian_pose_orientation(cartesian_frames[i], body_plane_indices, up_axis))
    forward = np.asarray(forward)
    rotations = get_rotation_to_ref_direction(forward, ref_dir=ref_dir)

    #### put character on floor
    floor_height = estimate_ground_height(cartesian_frames, fid_l=[3, 4], fid_r=[27, 28])
    cartesian_frames = cartesian_frames - floor_height
    velocities = []
    cartesian_frames = np.concatenate((cartesian_frames[0:1], cartesian_frames), axis=0)
    velocities = (cartesian_frames[1:, 0:1] - cartesian_frames[:-1, 0:1]).copy()

    ### remove translation
    cartesian_frames = cartesian_frames[1:]
    cartesian_frames[:, :, 0] = cartesian_frames[:, :, 0] - cartesian_frames[:, 0:1, 0]
    cartesian_frames[:, :, 2] = cartesian_frames[:, :, 2] - cartesian_frames[:, 0:1, 2]
    n_frames = len(rotations)
    ### rotate cartesian frames
    for i in range(n_frames):
        cartesian_frames[i] = rotate_cartesian_frame(cartesian_frames[i], rotations[i])
    for i in range(len(cartesian_frames)):
        velocities[i, 0] = rotations[i] * velocities[i, 0]

    r_v = np.zeros(n_frames)
    ### compute rotation angle
    for i in range(len(rotations)):
        r_v[i] = Quaternion.get_angle_from_quaternion(rotations[i], ref_dir)
    cartesian_frames = cartesian_frames.reshape(len(cartesian_frames), -1) 
    cartesian_frames = np.concatenate([cartesian_frames, velocities[:, :, 0]], axis=-1)
    cartesian_frames = np.concatenate([cartesian_frames, velocities[:, :, 2]], axis=-1)
    cartesian_frames = np.concatenate([cartesian_frames, r_v[:, np.newaxis]], axis=-1)
    return cartesian_frames    



def process_and_visualize():
    animated_joints = list(MH_CMU_SKELETON.keys())
    bvhfile = r'D:\workspace\my_git_repos\vae_motion_modeling\data\training_data\mk_cmu_skeleton\h36m\S1\Walking 1.bvh'

    ####  compute the ground truth
    bvhreader = BVHReader(bvhfile)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)

    cartesian_frames = convert_euler_frames_to_cartesian_frames(skeleton, bvhreader.frames, animated_joints=animated_joints)

    floor_height = estimate_ground_height(cartesian_frames, fid_l=[3, 4], fid_r=[27, 28])
    cartesian_frames = cartesian_frames - floor_height
    global_root = np.array([cartesian_frames[0, 0, 0], 0, cartesian_frames[0, 0, 2]])

    cartesian_frames = cartesian_frames - global_root

    ####

    processed_frames = process_file(bvhfile, animated_joints)
    relative_joint_positions = processed_frames[:, :-3].reshape(len(processed_frames), -1, 3)
    export_data = {'motion_data': relative_joint_positions.tolist(), 'has_skeleton': True, 'skeleton': MH_CMU_SKELETON}
    write_to_json_file(r'D:\tmp\customized_preprocessed.panim', export_data)

    ### reconstruct global motion
    # reconstructed_motion = compute_global_positions(processed_frames)
    reconstructed_motion = compute_global_positions_sequentially_tf(tf.convert_to_tensor(processed_frames, dtype=tf.float32))
    reconstructed_motion = reconstructed_motion.numpy()
    diff = np.sum((cartesian_frames - reconstructed_motion) ** 2)
    print("reconstruction error: ", diff)

    export_data = {'motion_data': reconstructed_motion.tolist(), 'has_skeleton': True, 'skeleton': MH_CMU_SKELETON}
    write_to_json_file(r'D:\tmp\reconstructed_motion.panim', export_data)
    export_data = {'motion_data': cartesian_frames.tolist(), 'has_skeleton': True, 'skeleton': MH_CMU_SKELETON}
    write_to_json_file(r'D:\tmp\origin_motion.panim', export_data)


def preprocessing_data(data_folder):
    """preprocess training data

    Arguments:
        data_path {[type]} -- [description]
    """
    bvhfiles = []
    frames = []
    animated_joints = list(MH_CMU_SKELETON.keys())
    get_files(data_folder, suffix='.bvh', files=bvhfiles)
    for filename in bvhfiles:
        processed_frames = process_file(filename, animated_joints)
        frames.append(processed_frames)
    frames = np.concatenate(frames, axis=0)
    print(frames.shape)
    np.savez_compressed('h36m_frame_data.npz', frames=frames)


def custom_loss(y_actural, y_pred):
    print(y_actural.shape)
    return tf.math.reduce_mean(tf.math.square(y_actural - y_pred))


def custom_loss1(y_actural, y_pred):
    return tf.math.square(y_actural - y_pred)


def cutom_loss2(y_actual, y_pred):
    pass


@tf.function()
def compute_relative_positions_wrapper(input):
    relative_positions = tf.numpy_function(compute_relative_position, [input], tf.float32)
    return relative_positions

@tf.function()
def joint_pos_loss(y_actual, y_pred):
    y_actual = tf.cast(y_actual, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # y_pos_actual = compute_relative_positions_tf(y_actual)
    # y_pred_actual = compute_relative_positions_tf(y_pred)
    y_pos_actual = compute_relative_positions_map(y_actual)
    y_pred_actual = compute_relative_positions_map(y_pred)
    return tf.math.reduce_mean(tf.math.square(y_pos_actual - y_pred_actual))


def test():
    training_data = np.load(r'h36m_frame_data.npz')
    motion_data = training_data['frames']
    # motion_vecotr = tf.constant(motion_data[100], dtype=tf.float32)
    # res = compute_global_position_tf(motion_vecotr)
    # print(res)

    # res = compute_global_position(motion_data[100])
    # print(res)
    motion_data = tf.convert_to_tensor(motion_data, dtype=tf.float32)
    print(joint_pos_loss(motion_data[0:BATCHSIZE], motion_data[BATCHSIZE:2*BATCHSIZE]))



def train_frame_encoder_mse():
    # data_folder = r'data\training_data\mk_cmu_skeleton\h36m'
    # preprocessing_data(data_folder)
    training_data = np.load('h36m_frame_data.npz')
    motion_data = training_data['frames']
    if 'mean' not in training_data.keys():
        mean = motion_data.mean(axis=0)[np.newaxis, :]
    else:
        mean = training_data['mean']
    if 'std' not in training_data.keys():
        std = motion_data.std(axis=0)[np.newaxis, :]
        std[std < EPS] = EPS
    else:
        std = training_data['std']
    
    # normalized_data = (motion_data - mean) / std
    normalized_data = motion_data
    #### initialize model
    dropout_rate = 0.3
    epochs = 300
    batchsize = BATCHSIZE
    learning_rate = 1e-4    
    npc = 10
    pose_encoder = FullBodyPoseEncoder(output_dim=normalized_data.shape[1], dropout_rate=dropout_rate, npc=npc)
    pose_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                         loss='mse')
    pose_encoder.fit(normalized_data, normalized_data, epochs=epochs, batch_size = batchsize)
    save_path = r'frame_encoder_models_mse'
    pose_encoder.save_weights(os.path.join(save_path, '_'.join([str(dropout_rate), str(epochs), str(learning_rate)]) + '.ckpt'))


def train_fullyConnected_frameEncoder():
    # data_folder = r'data\training_data\mk_cmu_skeleton\h36m'
    # preprocessing_data(data_folder)
    # training_data = np.load(r'D:\workspace\my_git_repos\vae_motion_modeling\data\training_data\framewise\h36m_frame_data.npz')
    # motion_data = training_data['frames']
    # print(motion_data.shape)

    h36m_data = get_training_data(name='h36m', data_type='quaternion')
    h36m_data = np.reshape(h36m_data, (h36m_data.shape[0] * h36m_data.shape[1], h36m_data.shape[2]))
    ### normalize data
    mean_value = h36m_data.mean(axis=0)[np.newaxis, :]
    std_value = h36m_data.std(axis=0)[np.newaxis, :]
    std_value[std_value<EPS] = EPS    
    normalized_data = (h36m_data - mean_value) / std_value 

    #### initialize model
    dropout_rate = 0.1
    epochs = 100
    batchsize = 64
    learning_rate = 1e-4    
    npc = 10
    name = "h36m_fullyConnected"
    filename = name + "_{epoch:04d}.ckpt"
    checkpoint_path = os.path.join(dirname, '../..', 'data/models', name, filename)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=10
    )
    pose_encoder = FullBodyPoseEncoder(output_dim=normalized_data.shape[1], dropout_rate=dropout_rate, npc=npc)
    pose_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                         loss='mse')
    pose_encoder.fit(normalized_data, normalized_data, epochs=epochs, batch_size = batchsize, callbacks=[cp_callback])
    # save_path = r'frame_encoder_models'
    # pose_encoder.save_weights(os.path.join(save_path, '_'.join([str(npc), str(dropout_rate), str(epochs), str(learning_rate)]) + '.ckpt'))


def evaluate_frame_encoder():
    training_data = np.load(r'D:\workspace\my_git_repos\vae_motion_modeling\h36m_frame_data.npz')
    motion_data = training_data['frames']

    #### initialize model
    dropout_rate = 0.0
    epochs = 300
    batchsize = BATCHSIZE
    learning_rate = 1e-4    
    npc = 10
    save_path = r'frame_encoder_models'
    pose_encoder = FullBodyPoseEncoder(output_dim=motion_data.shape[1], dropout_rate=dropout_rate, npc=npc)
    pose_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                         loss=joint_pos_loss)
    pose_encoder.build(input_shape=motion_data.shape)
    res = pose_encoder(motion_data[:100])
    print(res.shape)
    # pose_encoder.load_weights(os.path.join(save_path, '_'.join([str(dropout_rate), str(epochs), str(learning_rate)]) + '.ckpt'))                     

    # reconstructed_motion = np.array(pose_encoder(motion_data[:1000]))
    # print(reconstructed_motion.shape)
    # reconstructed_motion_global = compute_global_positions_sequentially(reconstructed_motion)
    # reconstructed_motion_local = compute_relative_position(reconstructed_motion)
    # print(reconstructed_motion.shape)
    # origin_motion_global = compute_global_positions_sequentially(motion_data[:1000])
    
    # origin_motion_local = compute_relative_position(motion_data[:1000])

    # mse_local = np.linalg.norm(reconstructed_motion_local - origin_motion_local)
    # print("MSE for relative positions: ", mse_local)

    # mse_global = np.linalg.norm(reconstructed_motion_global - origin_motion_global)
    # print("MSE for global positions: ", mse_global)

    # origin_data = {'motion_data': origin_motion_global.tolist(), 'has_skeleton': True, 'skeleton': MH_CMU_SKELETON}
    # export_data = {'motion_data': reconstructed_motion_global.tolist(), 'has_skeleton': True, 'skeleton': MH_CMU_SKELETON}
    # write_to_json_file(r'D:\tmp\origin.panim', origin_data)
    # write_to_json_file(r'D:\tmp\reconstructed.panim', export_data)



def compute_global_positions_sequentially(motion_data):
    n_frames = len(motion_data)
    
    new_frames = []
    for i in range(n_frames):
        new_frames.append(compute_global_position(motion_data[i]))
    new_frames = np.array(new_frames)
    for i in range(1, n_frames):
        prev_trans = np.array([new_frames[i-1][0, 0], 0, new_frames[i-1][0, 2]])
        new_frames[i] += prev_trans
    return np.array(new_frames)



@tf.function
def compute_relative_positions_tf(motion_data):
    new_frames = []
    for motion_vector in motion_data:
        new_frames.append(compute_global_position_tf(motion_vector))
    new_frames = tf.stack(new_frames, axis=0)        
    return new_frames


def compute_relative_positions_map(motion_data):
    # new_frames = tf.map_fn(compute_global_position_tf, motion_data)
    new_frames = tf.vectorized_map(compute_global_position_tf, motion_data)
    return new_frames


def compute_global_position(motion_vector):
    relative_joints, root_x, root_z, root_r = motion_vector[:-3], motion_vector[-3], motion_vector[-2], motion_vector[-1]
    relative_joints = relative_joints.reshape(-1, 3)
    # q = Quaternion.fromAngleAxis(-root_r, np.array([0, 1, 0]))
    q = Quaternions.from_angle_axis(-root_r, np.array([0, 1, 0]))
    v = np.array([root_x, 0, root_z])
    rotated_v = q * v
    return q * relative_joints +rotated_v


def compute_relative_position(motion_data):
    new_frames = []
    for motion_vector in motion_data:
        new_frames.append(compute_global_position(motion_vector))
    return np.array(new_frames, dtype=np.float32)


@tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
def tf_function(input):
    new_frames = tf.numpy_function(compute_relative_position, [input], tf.float32)
    return new_frames

def test_tf_numpy_function():
    training_data = np.load(r'D:\workspace\my_git_repos\vae_motion_modeling\h36m_frame_data.npz')
    motion_data = tf.convert_to_tensor(training_data['frames'][:2], dtype=tf.float32)    
    new_frames = tf_function(motion_data)
    print(new_frames.shape)

def compute_global_positions_sequentially_tf(motion_data):
    print(motion_data.shape)
    n_frames = motion_data.shape[0]
    new_frames = []
    for i in range(n_frames):
        new_frames.append(compute_global_position_tf(motion_data[i]))

    new_frames = tf.stack(new_frames, axis=0)
    print(new_frames.shape)
    output_frames = [new_frames[0]]
    for i in range(1, n_frames):
        prev_trans = tf.stack([output_frames[i-1][0, 0], tf.constant(0, dtype=tf.float32), output_frames[i-1][0, 2]], axis=0)
        output_frames.append(new_frames[i] + prev_trans)
    output_frames = tf.stack(output_frames, axis=0)
    return output_frames


def compute_global_positions(motion_data):
    '''
    reconstruct global position from motion data
    '''
    relative_frames, root_x, root_z, root_r = motion_data[:, :-3], motion_data[:, -3], motion_data[:, -2], motion_data[:, -1]
    n_frames = len(relative_frames)
    relative_positions = relative_frames.reshape(n_frames, -1, 3)
    translation  = np.array([0, 0, 0])
    for i in range(len(relative_frames)):
        rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0, 1, 0]))
        relative_positions[i] = rotation * relative_positions[i]
        translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])
        relative_positions[i] += translation
    return relative_positions



def batch_quaternion():
    angles = np.random.rand(4)
    angles = tf.convert_to_tensor(angles, dtype=tf.float32)
    axis = tf.constant((0, 1, 0), dtype=tf.float32)
    qs = from_axis_angle(axis, angles)
    print(qs)


@tf.function
def compute_global_position_tf(motion_vector):
    relative_joints, root_x, root_z, root_r = motion_vector[:-3], motion_vector[-3], motion_vector[-2], motion_vector[-1]
    relative_joints = tf.reshape(relative_joints, (-1, 3))
    axis = tf.constant((0, 1, 0), dtype=tf.float32)
    root_r = tf.reshape(root_r, (1, )) 
    q = from_axis_angle(axis, -root_r)
    v = tf.stack((root_x, tf.constant(0, dtype=tf.float32), root_z), axis=0)
    rotated_v = rotate(v, q)
    return rotate(relative_joints, q) + rotated_v
  

def compute_global_position_batch(motion_data):
    relative_joints, root_x, root_z, root_r = motion_data[:, :-3], motion_data[:, -3], motion_data[:, -2], motion_data[:, -1]
    relative_joints = tf.reshape(relative_joints, (-1, -1, 3))
    axis = tf.constant((0, 1, 0), dtype=tf.float32)
    qs = Quaternions.from_angle_axis()  


def test_compute_global_position():
    training_data = np.load('h36m_frame_data.npz')
    motion_data = training_data['frames']
    angles = motion_data[:100, -1]
    qs = Quaternions.from_angle_axis(angles, np.array([0, 1, 0]))
    print(qs.shape)


def test_compute_global_position_tf():
    training_data = np.load('h36m_frame_data.npz')
    motion_data = tf.convert_to_tensor(training_data['frames'], dtype=tf.float32)

    res = compute_global_positions_sequentially_tf(motion_data[:2])
    print(res.shape)


def test_compute_relative_position_tf():
    training_data = np.load(r'D:\workspace\my_git_repos\vae_motion_modeling\h36m_frame_data.npz')
    motion_data = tf.convert_to_tensor(training_data['frames'], dtype=tf.float32)
    res = compute_relative_positions_map(motion_data[:100])
    print(res.shape)    
    res = res.numpy()    
    export_data = {'motion_data': res.tolist(), 'has_skeleton': True, 'skeleton': MH_CMU_SKELETON}
    write_to_json_file(r'D:\tmp\tf_map.panim', export_data)


def quaternion_operation_compare():
    from tensorflow_graphics.geometry.transformation.quaternion import between_two_vectors_3d, from_axis_angle, rotate

    # v1 = np.random.rand(10)
    # v2 = np.random.rand(10)
    # print(custom_loss1(v1, v2))
    from tensorflow_graphics import version
    from tensorflow_graphics.geometry.transformation.quaternion import between_two_vectors_3d
    a = tf.constant((1, 0, 0), dtype=tf.float32)
    b = tf.constant((0, 1, 0), dtype=tf.float32)
    c = tf.constant((-.999, .001, 0), dtype=tf.float32)
    d = tf.constant((-1, 0, 0), dtype=tf.float32)

    print("no rotation:")
    print(between_two_vectors_3d(a, a))
    print("90 deg rotation between x and y axis: ")
    print(between_two_vectors_3d(a, b))
    print("almost 180 degree rotation from x-axis to negative x-axis:")
    print(between_two_vectors_3d(a, c))
    print("180 degree rotation from x-axis to negative x-axis: BROKEN")
    print(between_two_vectors_3d(a, d))

    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    c = np.array([-.999, .001, 0])
    d = np.array([-1, 0, 0])
    print(Quaternion.between(a, a))
    print(Quaternion.between(a, b))
    print(Quaternion.between(a, c))
    print(Quaternion.between(a, d))


    bone_vector = np.random.rand(3)
    angle = np.deg2rad(60)
    # r1 = Quaternion.from_axis_angle()
    ref_axis = np.array([0, 1, 0])
    q = Quaternion.fromAngleAxis(angle, ref_axis)
    print(q)
    axis = tf.constant((0, 1, 0), dtype=tf.float32)
    rotated_bone = q * bone_vector
    print("rotated bone is: ", rotated_bone)
    angles = np.random.rand(4)
    angle = tf.constant(angles, shape=(4, ), dtype=tf.float32)

    q1 = from_axis_angle(axis, angle)
    print(q1)
    bone_vector = tf.constant(bone_vector, dtype=tf.float32)
    rotated_bone = rotate(bone_vector, q1)
    print(rotated_bone)
    w = q1[3].numpy()
    print(w)


if __name__ == "__main__":
    train_fullyConnected_frameEncoder()
    # evaluate_frame_encoder()
    # test()
    # quaternion_operation_compare()
    # test_compute_global_position()
    # process_and_visualize()
    # test_compute_global_position_tf()
    # batch_quaternion()
    # test_tf_numpy_function()
    # test_compute_relative_position_tf()
    # train_frame_encoder_mse()