import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute())+ r'/..')
from models.dilated_TCN_IK import DilatedTCN_IK
import numpy as np
from mosi_utils_anim.utilities import load_json_file
from mosi_utils_anim.animation_data import BVHReader, BVHWriter, SkeletonBuilder
from mosi_utils_anim.animation_data.utils import pose_orientation_general, convert_euler_frame_to_cartesian_frame, shift_euler_frames_to_ground



def evaluate_model():
    dilated_tcn_ik = DilatedTCN_IK(name='tcn_ik')
    dilated_tcn_ik.build(input_shape=93, output_shape=96)   
    dilated_tcn_ik.load(r'trained_models/forward_ik.ckpt') 

    training_data = load_json_file(r'D:\workspace\projects\mesh_retargeting\training_data\point_cloud_plus_euler_frame\style_training_data.json')
    X = training_data['X']
    Y = training_data['Y']
    Xmean = np.asarray(training_data['Xmean'])
    Xstd = np.asarray(training_data['Xstd'])

    Ymean = np.asarray(training_data['Ymean'])
    Ystd = np.asarray(training_data['Ystd'])
    Xstd[:] = Xstd.mean()
    Ystd[:] = Ystd.mean()

    test_input = np.asarray(X[0])
    test_input = (test_input - Xmean) / Xstd

    normalized_euler_frames = dilated_tcn_ik.predict(test_input)
    print(normalized_euler_frames.shape)
    output = normalized_euler_frames * Ystd + Ymean
    ### export euler frames
    bvhskeleton_file = r'D:\workspace\mocap_data\skeleton_template\mk_cmu_skeleton.bvh'
    bvhreader = BVHReader(bvhskeleton_file)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    foot_contact_joints = ['LeftFoot', 'RightFoot']
    output = shift_euler_frames_to_ground(output[0], foot_contact_joints, skeleton)    
    BVHWriter(r'D:\workspace\projects\mesh_retargeting\tmp\example1.bvh', skeleton, output,
        skeleton.frame_time, is_quaternion=False)


if __name__ == "__main__":
    evaluate_model()