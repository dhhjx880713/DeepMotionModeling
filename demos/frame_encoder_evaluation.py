import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + r'/../')
from models.frame_encoder import FrameEncoder
import numpy as np
from tensorflow.keras import Sequential, layers, Model, optimizers, losses
from utilities.utils import export_point_cloud_data_without_foot_contact
from utilities.skeleton_def import MH_CMU_SKELETON, MH_CMU_ANIMATED_JOINTS
from preprocessing.preprocessing import process_file
import copy

EPS = 1e-6


def evaluate_frame_encoder():
    n_frames = 1000
    walking = r'D:\workspace\my_git_repos\vae_motion_modeling\data\training_data\mk_cmu_skeleton\h36m\S1\Walking.bvh'
    waving = r'D:\workspace\my_git_repos\vae_motion_modeling\data\training_data\mk_cmu_skeleton\h36m\S1\Greeting.bvh'
    walking_motion = process_file(walking, sliding_window=False)[:n_frames]
    waving_motion = process_file(waving, sliding_window=False)[:n_frames]
    scale_factor = 1
    body_parts = {'torso': ['Hips', 'LowerBack', 'Spine', 'Spine1', 'Neck', 'Neck1', 'Head'],
                  'leftArm': ['LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LThumb', 'LeftFingerBase', 'LeftHandFinger1'],
                  'rightArm': ['RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RThumb', 'RightFingerBase', 'RightHandFinger1'],
                  'leftLeg': ['LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'],
                  'rightLeg': ['RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase'],
                  'global_trans': ['velocity_x', 'velocity_z', 'velocity_r']}
    torso_indices = [np.arange(MH_CMU_ANIMATED_JOINTS.index(joint) * 3, (MH_CMU_ANIMATED_JOINTS.index(joint)+1)*3) for joint in body_parts['torso']]
    leftArm_indices = [np.arange(MH_CMU_ANIMATED_JOINTS.index(joint) * 3, (MH_CMU_ANIMATED_JOINTS.index(joint)+1)*3) for joint in body_parts['leftArm']]
    rightArm_indices = [np.arange(MH_CMU_ANIMATED_JOINTS.index(joint) * 3, (MH_CMU_ANIMATED_JOINTS.index(joint)+1)*3) for joint in body_parts['rightArm']]
    leftLeg_indices = [np.arange(MH_CMU_ANIMATED_JOINTS.index(joint) * 3, (MH_CMU_ANIMATED_JOINTS.index(joint)+1)*3) for joint in body_parts['leftLeg']]
    rightLeg_indices = [np.arange(MH_CMU_ANIMATED_JOINTS.index(joint) * 3, (MH_CMU_ANIMATED_JOINTS.index(joint)+1)*3) for joint in body_parts['rightLeg']]

    ## replace two arms
    new_motion = copy.deepcopy(walking_motion)
    new_motion[:, leftArm_indices] = waving_motion[:, leftArm_indices]
    new_motion[:, rightArm_indices] = waving_motion[:, rightArm_indices]
    export_point_cloud_data_without_foot_contact(new_motion, filename=r'D:\tmp\ref_combined_motion_1.panim', scale_factor=scale_factor)
    export_point_cloud_data_without_foot_contact(walking_motion, filename=r'D:\tmp\ref_walking_1.panim', scale_factor=scale_factor)
    export_point_cloud_data_without_foot_contact(waving_motion, filename=r'D:\tmp\ref_waving_motion_1.panim', scale_factor=scale_factor)
    ## load frame encoder 
    dropout_rate = 0.1
    learning_rate = 1e-4
    epochs = 100
    batchsize = 256
    frame_encoder = FrameEncoder(dropout_rate=dropout_rate)
    input_data_dir = r'../../data/training_data/h36m.npz'
    input_data = np.load(input_data_dir)

    assert 'clips' in input_data.keys(), "cannot find motion data in " + input_data_dir
    motion_data = input_data['clips']
    frame_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                          loss='mse',
                          metrics=['accuracy'])
    mean_value = motion_data.mean(axis=0)[np.newaxis, :]
    std_value = motion_data.std(axis=0)[np.newaxis, :]
    std_value[std_value < EPS] = EPS
    # frame_encoder.build(input_shape=motion_data.shape)
    frame_encoder.load_weights(r'D:\workspace\my_git_repos\vae_motion_modeling\data\models\frame_encoder1\frame_encoder-100-0.0001-0.1-0100.ckpt')
    
    normalized_walking = (walking_motion - mean_value) / std_value
    normalized_waving = (waving_motion - mean_value) / std_value
    normalized_combined_motion = (new_motion - mean_value) / std_value
    recon_walking = frame_encoder(normalized_walking).numpy()
    recon_waving = frame_encoder(normalized_waving).numpy()
    recon_combined_motion = frame_encoder(normalized_combined_motion).numpy()
    recon_walking = recon_walking * std_value + mean_value
    recon_waving = recon_waving * std_value +mean_value
    recon_combined_motion = recon_combined_motion * std_value + mean_value
    export_point_cloud_data_without_foot_contact(recon_walking, filename=r'D:\tmp\recon_walking_motion_1.panim', scale_factor=scale_factor)
    export_point_cloud_data_without_foot_contact(recon_waving, filename=r'D:\tmp\recon_waving_motion_1.panim', scale_factor=scale_factor)
    export_point_cloud_data_without_foot_contact(recon_combined_motion, filename=r'D:\tmp\recon_combined_motion_1.panim', scale_factor=scale_factor)



if __name__ == "__main__":
    evaluate_frame_encoder()

