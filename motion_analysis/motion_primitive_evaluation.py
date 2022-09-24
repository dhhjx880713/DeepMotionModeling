# encoding: UTF-8
from morphablegraphs.motion_model import MotionPrimitive
import seaborn as sns
from morphablegraphs.utilities import get_motion_primitive_path, get_semantic_motion_primitive_path, \
                                      get_aligned_data_folder, load_json_file, write_to_json_file
import numpy as np
import matplotlib.pyplot as plt
import mgrd as mgrd
import os
import copy
from morphablegraphs.animation_data import BVHReader, BVHWriter, Skeleton, SkeletonBuilder
from morphablegraphs.motion_model.motion_primitive_wrapper import MotionPrimitiveModelWrapper
from morphablegraphs.animation_data.utils import convert_quat_frames_to_cartesian_frames
script_dir = os.path.dirname(__file__)


game_engine_animated_joints = [u'Game_engine', u'Root', u'pelvis', u'spine_01', u'spine_02', u'spine_03',
                               u'clavicle_l', u'upperarm_l', u'lowerarm_l', u'hand_l', u'clavicle_r', u'upperarm_r',
                               u'lowerarm_r', u'hand_r', u'neck_01', u'head', u'thigh_l', u'calf_l', u'foot_l',
                               u'ball_l', u'thigh_r', u'calf_r', u'foot_r', u'ball_r']

def frame_range_evalutation(elementary_action,
                            motion_primitive):
    N = 1000
    repo_path = r'C:\repo'
    frame_numbers = []
    motion_primitive_file = get_motion_primitive_path(repo_path, elementary_action, motion_primitive)
    model = MotionPrimitive(motion_primitive_file)
    for i in range(N):
        motion_spline = model.sample()
        frames = motion_spline.get_motion_vector()
        frame_numbers.append(len(frames))
    sns.distplot(frame_numbers)
    plt.show()


def export_motion_for_old_model(elementary_action,
                                motion_primitive,
                                skeleton_file):
    N = 100
    repo_path = r'C:\repo'
    frame_numbers = []
    motion_primitive_file = get_motion_primitive_path(repo_path, elementary_action, motion_primitive)
    # motion_primitive_file = r'C:\repo\data\3 - Motion primitives\motion_primitives_quaternion_PCA95\motion_primitives_quaternion_PCA95 m24-integration-6.0\elementary_action_models\elementary_action_pickBoth\pickBoth_first_quaternion_mm.json'
    model = MotionPrimitive(motion_primitive_file)
    bvhreader = BVHReader(skeleton_file)
    skeleton = Skeleton()
    skeleton.load_from_bvh(bvhreader)
    output_folder = r'E:\experiment data\tmp'
    for i in range(N):
        motion_spline = model.sample()
        frames = motion_spline.get_motion_vector()
        filename = output_folder + os.sep + str(i) + '.bvh'
        BVHWriter(filename, skeleton, frames, bvhreader.frame_time,
                  is_quaternion=True, skipped_joints=False)


def frame_range_evaluation_semantic_model(elementary_action,
                                          motion_primitive):
    N = 10000
    repo_path = r'C:\repo'
    test_model_file = get_semantic_motion_primitive_path(repo_path,
                                                         elementary_action,
                                                         motion_primitive)
    # test_model_file = r'C:\repo\data\3 - Motion primitives\motion_primitives_quaternion_PCA95\elementary_action_temporal_semantic_models\elementary_action_walk\temp.json'
    skeleton_file = r'C:\git-repo\ulm\morphablegraphs\python_src\mgrd\data\skeleton.json'
    skeletonLoader = mgrd.SkeletonJSONLoader(skeleton_file)
    skeleton = skeletonLoader.load()
    model = mgrd.MotionPrimitiveModel.load_from_file(skeleton, test_model_file)
    samples = model.get_random_samples(N)
    output_folder = r'C:\git-repo\tmp_samples\walk_samples'
    frame_numbers = []
    for i in range(len(samples)):
        time_spline = mgrd.create_time_spline_from_sample(model, samples[i])
        # print(time_spline.coeffs[:, 0])
        # raw_input('')
        warped_time_spline = time_spline.warp_self()
        # print(warped_time_spline.knots[-1])
        frame_numbers.append(warped_time_spline.knots[-1])
    sns.distplot(frame_numbers)
    plt.show()


def export_motions_for_semantic_model():
    N = 30
    repo_path = r'C:\repo'
    test_model_file = get_semantic_motion_primitive_path(repo_path,
                                                         elementary_action,
                                                         motion_primitive)
    # test_model_file = r'C:\repo\data\3 - Motion primitives\motion_primitives_quaternion_PCA95\elementary_action_temporal_semantic_models\elementary_action_walk\temp.json'
    skeleton_file = r'C:\git-repo\ulm\morphablegraphs\python_src\mgrd\data\skeleton.json'
    skeletonLoader = mgrd.SkeletonJSONLoader(skeleton_file)
    print(test_model_file)
    test_model_file = r'C:\repo\data\3 - Motion primitives\motion_primitives_quaternion_PCA95_temporal_semantic\elementary_action_walk\walk_leftStance_quaternion_mm.json'
    test_json_file = load_json_file(test_model_file)
    print(test_json_file.keys())
    skeleton = skeletonLoader.load()
    model = mgrd.MotionPrimitiveModel.load_from_file(skeleton, test_model_file)
    samples = model.get_random_samples(N)
    output_folder = r'C:\git-repo\tmp_samples\placeBoth\reach'
    for i in range(len(samples)):
        print(i)
        quat_spline = mgrd.create_quaternion_spline_from_sample(model, samples[i])
        time_spline = mgrd.create_time_spline_from_sample(model, samples[i])
        bvhstr = mgrd.export_to_bvh_format(quat_spline, time_spline)
        filename = output_folder + os.sep + str(i) + '.bvh'
        with open(filename, 'w') as outfile:
            outfile.write(bvhstr)


def motion_speed_evaluation():
    from morphablegraphs.motion_model.motion_primitive_wrapper import MotionPrimitiveModelWrapper
    elementary_action = 'oldWalk'
    motion_primitive = 'leftStance'
    bvhreader = BVHReader(r'C:\Users\hadu01\git-repos\ulm\morphablegraphs\python_src\game_engine_target.bvh')

    repo_path = r'C:\repo'
    test_model_file = get_semantic_motion_primitive_path(elementary_action,
                                                         motion_primitive,
                                                         repo_path)
    print(test_model_file)
    mm = MotionPrimitiveModelWrapper()
    model_data = load_json_file(test_model_file)
    animated_joints = model_data['sspm']['animated_joints']
    bvh_skeleton = SkeletonBuilder().load_from_bvh(bvhreader, animated_joints=animated_joints)
    skeleton = bvh_skeleton.convert_to_mgrd_skeleton()
    mm._load_from_file(skeleton, test_model_file, animated_joints=animated_joints)
    output_folder = r'E:\experiment data\tmp1'
    quat_spline = mm.sample()

    # quat_frames = quat_spline.get_motion_vector(step_size=2)
    # print(quat_frames.shape)
    # filename = os.path.join(output_folder, 'normal_speed_2.bvh')
    # complete_quat_frames = bvh_skeleton.add_fixed_joint_parameters_to_motion(quat_frames)
    # BVHWriter(filename, bvh_skeleton, complete_quat_frames, bvh_skeleton.frame_time, is_quaternion=True)


def sample_motion_primitive(elementary_action, motion_primitive, N):
    repo_path = r'C:\repo'
    skeleton_file = r'..\..\game_engine_target_large.bvh'
    skeleton_file = os.path.join(script_dir, skeleton_file)
    test_model_file = get_semantic_motion_primitive_path(elementary_action,
                                                         motion_primitive,
                                                         repo_path)
    mm = MotionPrimitiveModelWrapper()
    model_data = load_json_file(test_model_file)
    bvhreader = BVHReader(skeleton_file)
    animated_joints = model_data['sspm']['animated_joints']
    bvh_skeleton = SkeletonBuilder().load_from_bvh(bvhreader, animated_joints)

    skeleton = bvh_skeleton.convert_to_mgrd_skeleton()

    mm._load_from_file(skeleton, test_model_file, animated_joints=animated_joints)
    cartesian_motions = []
    GAME_ENGINE_ANIMATED_JOINTS = ['Game_engine', 'Root', 'pelvis', 'spine_03', 'clavicle_l', 'upperarm_l',
                                   'lowerarm_l',
                                   'hand_l', 'clavicle_r',
                                   'upperarm_r', 'lowerarm_r', 'hand_r', 'neck_01', 'head', 'thigh_l', 'calf_l',
                                   'foot_l',
                                   'ball_l', 'thigh_r', 'calf_r', 'foot_r', 'ball_r']
    for i in range(N):
        quat_spline = mm.sample()
        quat_frames = quat_spline.get_motion_vector(max_frames=60)
        cartesian_motions.append(convert_quat_frames_to_cartesian_frames(bvh_skeleton, quat_frames, GAME_ENGINE_ANIMATED_JOINTS))
    return np.asarray(cartesian_motions)


def motion_primitive_evaluation(elementary_action,
                                motion_primitive,
                                skeleton_file):

    N = 500
    repo_path = r'E:\workspace\repo'
    test_model_file = get_semantic_motion_primitive_path(elementary_action,
                                                         motion_primitive,
                                                         repo_path)
    mm = MotionPrimitiveModelWrapper()
    model_data = load_json_file(test_model_file)
    bvhreader = BVHReader(skeleton_file)
    animated_joints = model_data['sspm']['animated_joints']
    bvh_skeleton = SkeletonBuilder().load_from_bvh(bvhreader, animated_joints)

    skeleton = bvh_skeleton.convert_to_mgrd_skeleton()

    mm._load_from_file(skeleton, test_model_file, animated_joints=animated_joints)
    output_folder = r'E:\workspace\experiment data\tmp1'
    elementary_action_folder = os.path.join(output_folder, elementary_action)
    if not os.path.exists(elementary_action_folder):
        os.mkdir(elementary_action_folder)
    motion_primitive_folder = os.path.join(elementary_action_folder, motion_primitive)
    if not os.path.exists(motion_primitive_folder):
        os.mkdir(motion_primitive_folder)

    for i in range(N):
        print(i)
        quat_spline = mm.sample()
        quat_frames = quat_spline.get_motion_vector()
        filename = os.path.join(motion_primitive_folder, str(i) + '.bvh')
        # mv = MotionVector(bvh_skeleton)
        # mv.set_frames(quat_frames)
        complete_quat_frames = bvh_skeleton.add_fixed_joint_parameters_to_motion(quat_frames)
        # BVHWriter(filename, bvh_skeleton, euler_frames, bvhreader.frame_time, is_quaternion=False)
        BVHWriter(filename, bvh_skeleton, complete_quat_frames, bvh_skeleton.frame_time, is_quaternion=True)


def motion_primitive_evaluation_using_mgrd(elementary_action,
                                           motion_primitive,
                                           skeleton_file):
    from morphablegraphs.animation_data import MotionVector
    N = 100
    repo_path = r'C:\repo'
    test_model_file = get_semantic_motion_primitive_path(elementary_action,
                                                         motion_primitive,
                                                         repo_path)
    if skeleton_file.endswith('bvh'):
        bvhreader = BVHReader(skeleton_file)
        # bvhreader.set_animated_joints()
        animated_joints = [joint for joint in bvhreader.get_animated_joints()]
        bvh_skeleton = Skeleton()
        bvh_skeleton.load_from_bvh(bvhreader, animated_joints)
        skeleton = bvh_skeleton.convert_to_mgrd_skeleton()
    elif skeleton_file.endswith('json'):
        skeletonLoader = mgrd.SkeletonJSONLoader(skeleton_file)
        skeleton = skeletonLoader.load()
    else:
        raise NotImplementedError


    model = mgrd.MotionPrimitiveModel.load_from_file(skeleton, test_model_file)
    output_folder = r'E:\experiment data\tmp1'
    elementary_action_folder = os.path.join(output_folder, elementary_action)
    if not os.path.exists(elementary_action_folder):
        os.mkdir(elementary_action_folder)
    motion_primitive_folder = os.path.join(elementary_action_folder, motion_primitive)
    if not os.path.exists(motion_primitive_folder):
        os.mkdir(motion_primitive_folder)
    samples = model.get_random_samples(N)

    for i in range(N):
        print(i)
        quat_spline = mgrd.create_quaternion_spline_from_sample(model, samples[i])
        # time_spline = mgrd.create_time_spline_from_sample(model, samples[i])
        # bvhstr = mgrd.export_to_bvh_format(quat_spline, time_spline)
        filename = os.path.join(motion_primitive_folder, str(i) + '.bvh')
        # with open(filename, 'w') as outfile:
        #     outfile.write(bvhstr)

        quat_frames = quat_spline.get_motion_vector()
        # print(quat_frames[0])
        mv = MotionVector(bvh_skeleton)
        mv.set_frames(quat_frames)
        euler_frames = mv.get_complete_euler_frame()
        pos = skeleton.nodes['pelvis'].get_global_position_from_euler_frame(euler_frames[-1])
        print(pos)
        BVHWriter(filename, bvh_skeleton, euler_frames, bvhreader.frame_time, is_quaternion=False)


def plot_frame_range_of_training_data(elementary_action,
                                      motion_primitive):
    aligned_data_folder = get_aligned_data_folder(elementary_action,
                                                  motion_primitive)
    timewarping_data = load_json_file(os.path.join(aligned_data_folder, 'timewarping.json'))
    frame_numbers = []
    for filename, warping_index in timewarping_data.iteritems():
        frame_numbers.append(warping_index[-1])
    sns.distplot(frame_numbers)
    plt.show()


def get_sample_using_motion_primitive_wrapper(elementary_action, motion_primitive):
    from morphablegraphs.motion_model.motion_primitive_wrapper import MotionPrimitiveModelWrapper
    from mgrd import SkeletonMotionBVHExporter
    from morphablegraphs.animation_data.motion_editing import get_step_length, convert_quat_spline_to_euler_frames
    model_file = get_semantic_motion_primitive_path(elementary_action, motion_primitive)
    model_data = load_json_file(model_file)
    skeleton_file = r'../../game_engine_4.bvh'
    bvhreader = BVHReader(skeleton_file)
    skeleton = Skeleton()
    skeleton.load_from_bvh(bvhreader)
    mgrd_skeleton = skeleton.convert_to_mgrd_skeleton()
    mm = MotionPrimitiveModelWrapper()
    mm._load_from_file(mgrd_skeleton, model_file)
    new_sample = mm.sample()
    # motion_duration = max(new_sample.knots)
    #
    # # print(motion_duration)
    # frame_time =0.013889
    bvh_exporter = SkeletonMotionBVHExporter(new_sample)
    # bvh_exporter.set_frame_time(frame_time)
    # # motion_str = SkeletonMotionBVHExporter(new_sample).set_frame_time(frame_time).export_to_string()
    # # with open('test.bvh', 'w') as outfile:
    # #     outfile.write(motion_str)
    # euler_frames = bvh_exporter.convert_frame_data()
    # step_length = get_step_length(euler_frames)
    # print("step length is: ", step_length)

    ## create sample with different frames
    euler_frames_1 = convert_quat_spline_to_euler_frames(new_sample, 50)
    step_length_1 = get_step_length(euler_frames_1)
    print("step length is: ", step_length_1)

    euler_frames_2 = convert_quat_spline_to_euler_frames(new_sample, 100)
    step_length_2 = get_step_length(euler_frames_2)
    print("step length is: ", step_length_2)


def test():
    test_data_file = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_pickBoth\first\temporal_semantic_low_dimensional_data_old.json'
    json_data = load_json_file(test_data_file)
    print(np.array(json_data['eigen_vector']).shape)
    print(np.array(json_data['low_dimensional_data']).shape)
    new_data = copy.deepcopy(json_data)
    eigen_vector = np.array(new_data['eigen_vector']) * 10
    new_data['eigen_vector'] = eigen_vector.tolist()
    low_dimensional_data = np.array(new_data['low_dimensional_data']) * 0.1
    new_data['low_dimensional_data'] = low_dimensional_data.tolist()
    new_filename = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_pickBoth\first\temporal_semantic_low_dimensional_data.json'
    write_to_json_file(new_filename, new_data)


if __name__ == '__main__':
    # elementary_action = 'hybrPick'
    # motion_primitive = 'reach'
    elementary_action = 'angryWalk'
    # motion_primitive = 'reach_game_engine_skeleton_new'
    motion_primitive = 'leftStance'
    # skeleton_file = r'../../game_engine_target.bvh'
    # elementary_action = 'grasp'
    # motion_primitive = 'grasp'
    skeleton_file = r'C:\Users\hadu01\git-repos\ulm\mg-experiments\mg-tools\mg_analysis\morphablegraphs\python_src\game_engine_target_large.bvh'
    # skeleton_file = r'C:\repo\data\1 - MoCap\3 - Cutting\elementary_action_hybrPick\reach\17-11-20-Hybrit-VW_pickup-part_009_snapPoseSkeleton_reach_0_58.bvh'
    # skeleton_file = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\beginLeftStance_raw_skeleton\walk_001_1_beginRightStance_beginLeftStance_674_753_mirrored.bvh'
    # skeleton_file = r'C:\Users\hadu01\git-repos\ulm\morphablegraphs\python_src\game_engine_target.bvh'
    # get_sample_using_motion_primitive_wrapper(elementary_action, motion_primitive)
    # frame_range_evalutation(elementary_action,
    #                         motion_primitive)
    # export_motion_for_old_model(elementary_action,
    #                             motion_primitive,
    #                             skeleton_file)
    # frame_range_evaluation_semantic_model(elementary_action,
    #                                       motion_primitive)
    # plot_frame_range_of_training_data(elementary_action,
    #                                   motion_primitive)
    # test()
    elementary_actions = ['oldWalk', 'proudWalk', 'childlikeWalk', 'depressedWalk']
    motion_primitives = ['leftStance', 'rightStance']
    for elementary_action in elementary_actions:
        for motion_primitive in motion_primitives:
            motion_primitive_evaluation(elementary_action,
                                        motion_primitive,
                                        skeleton_file)
    # motion_primitive_evaluation_using_mgrd(elementary_action,
    #                                        motion_primitive,
    #                                        skeleton_file)
    # export_motions_for_semantic_model()
    # motion_speed_evaluation()