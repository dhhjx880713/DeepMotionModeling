import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from models.fcn_ik import FCN_IK
import numpy as np
from mosi_utils_anim.utilities import load_json_file
from mosi_utils_anim.animation_data import BVHReader, BVHWriter, SkeletonBuilder
from mosi_utils_anim.animation_data.utils import pose_orientation_general, convert_euler_frame_to_cartesian_frame, shift_euler_frames_to_ground



def evaluate_fcn_ik_4_layers():
    ### load data
    training_data = load_json_file(r'E:\workspace\projects\mesh_retargeting\training_data\point_cloud_plus_euler_frame\style_training_data.json')
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

    ### load model
    fcn_ik_4_layers = FCN_IK(name='fcn_ik_4_layer')   
    fcn_ik_4_layers.build(input_shape=93, output_shape=96)
    fcn_ik_4_layers.load(r'trained_models/fcn_4_layers_ik.ckpt') 
    normalized_euler_frames = fcn_ik_4_layers.predict(test_input)
    output = normalized_euler_frames * Ystd + Ymean
    print(output.shape)
    ### export euler frames
    bvhskeleton_file = r'E:\workspace\mocap_data\skeleton_template\mk_cmu_skeleton.bvh'
    bvhreader = BVHReader(bvhskeleton_file)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    foot_contact_joints = ['LeftFoot', 'RightFoot']
    output = shift_euler_frames_to_ground(output, foot_contact_joints, skeleton)    
    BVHWriter(r'E:\workspace\projects\mesh_retargeting\tmp\example2.bvh', skeleton, output,
        skeleton.frame_time, is_quaternion=False)


if __name__ == "__main__":
    evaluate_fcn_ik_4_layers()