import os
import sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.insert(0, r'../')


# print(sys.path)
from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder
import numpy as np
import tensorflow as tf
import yaml

EPS = 1e-6





def run_MotionEnDecoder():
    training_data_path = r'../../data/training_data/clipwise/h36m.npz' 
    training_data = np.load(training_data_path)['clips']
    print(training_data.shape)
    mean_pose = training_data.mean(axis=(0, 1))
    std_pose = training_data.std(axis=(0, 1))
    std_pose[std_pose<EPS] = EPS
    training_data = (training_data - mean_pose) / std_pose
    encoder = MotionEncoder(output_dims=10, hidden_dims=128)
    decoder = MotionDecoder(output_dims=90, hidden_dims=128)
    print(encoder.trainable_variables)
    epochs = 1
    batchsize = 32
    learning_rate = 1e-4
    model = MotionEnDecoder(encoder, decoder)
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')
    # model.fit(training_data, training_data, epochs=epochs, batch_size=batchsize)


def test():
    # filename = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\stylized_data_raw\angry_fast punching_434.bvh'
    # bvhreader = BVHReader(filename)
    # skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    # print(skeleton.animated_joints)
    # print(len(skeleton.animated_joints))
    filename = r'D:\gits\deep-motion-editing\style_transfer\data\mocap_xia\angry_01_000.bvh'
    bvhreader = BVHReader(filename)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    # print(skeleton.animated_joints)
    # print(len(skeleton.animated_joints))
    # for joint in skeleton.animated_joints:
    #     print(joint)
    #     print(skeleton.nodes[joint].offset)

    skeleton_filename = r'D:\gits\deep-motion-editing\style_transfer\global_info\skeleton_CMU.yml'
    f = open(skeleton_filename, "r")
    skel = yaml.load(f, Loader=yaml.Loader)
    print(len(skel['parents']))
    print(len(skel['chosen_joints']))
    offsets = skel['offsets']
    for i in range(len(offsets)):
        if (np.asarray(offsets[i]) == np.asarray(skeleton.nodes[skeleton.animated_joints[i]].offset)).all():
            print(True)


def create_mk_skeleton():
    import copy 

    ref_bvhfile = r'D:\gits\deep-motion-editing\style_transfer\data\mocap_xia\angry_01_000.bvh'
    target_bvhfile = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\stylized_data_raw\angry_fast punching_434.bvh'
    ref_skeleton_def = r'D:\gits\deep-motion-editing\style_transfer\global_info\skeleton_CMU.yml'

    ref_bvhreader = BVHReader(ref_bvhfile)
    ref_skeleton = SkeletonBuilder().load_from_bvh(ref_bvhreader)

    target_bvhfile = BVHReader(target_bvhfile)
    target_skeleton = SkeletonBuilder().load_from_bvh(target_bvhfile)
    f = open(ref_skeleton_def, "r")
    ref_skel = yaml.load(f, Loader=yaml.Loader)
    target_skel = copy.deepcopy(ref_skel)
    target_skel['offsets'] = [target_skeleton.nodes[joint].offset for joint in target_skeleton.animated_joints]
    ### create parent index
    ref_parent_index = ref_skel['parents']


    target_parent_index = []
    for joint in target_skeleton.animated_joints:
        if target_skeleton.nodes[joint].parent is not None:
            parent_name = target_skeleton.nodes[joint].parent.node_name
            target_parent_index.append(target_skeleton.animated_joints.index(parent_name))
        else:
            target_parent_index.append(-1)

    target_skel['parents'] = target_parent_index

    ### create chosen joints
    ref_chosen_joints = ref_skel['chosen_joints']
    target_chosen_joints = []
    for i in ref_chosen_joints:
        joint_name = ref_skeleton.animated_joints[i]

        ### replace Index to Finger
        if 'Index' in joint_name:
            joint_name = joint_name.replace("Index", "Finger")

        target_chosen_joints.append(target_skeleton.animated_joints.index(joint_name))
    
    target_skel['chosen_joints'] = target_chosen_joints
    print(target_skel['chosen_joints'])
    ### create chosen_parents
    target_chosen_parents = []
    for i in range(len(target_chosen_joints)):
        joint_name = target_skeleton.animated_joints[target_chosen_joints[i]]
        if not target_skeleton.nodes[joint_name].parent is None:
            parent_name = target_skeleton.nodes[joint_name].parent.node_name
            target_chosen_parents.append(target_skeleton.animated_joints.index(parent_name))
        else:
            target_chosen_parents.append(-1)
    target_skel['chosen_parents'] = target_chosen_parents


    #### for meta_infomation like left_foot, right_foot, hips, shoulders...  The indices are all for chosen index
    # print(target_skel)
    output_filename = 'skeleton_MK_CMU.yml'
    with open(output_filename, 'w') as outfile:
        tmp = yaml.dump(target_skel, outfile)


    # for i in ref_parent_index:
    #     if i == -1:
    #         target_parent_index.append(-1)
    #     else:
    #         ref_parent_joint = ref_skeleton.animated_joints[i]
    #         target_parent_index.append(target_skeleton.animated_joints.index(ref_parent_joint))
    # print(target_parent_index)

    # for joint in target_skeleton.nodes:
    #     if joint.parent is None:
    #         target_parent_index.append(-1)
    #     else:
    #         parent_joint = joint.parent.node_name
    #         parent_index = target_skeleton.animated_joints.index(parent_joint)
    #         target_parent_index.append(parent_index)
    # print(target_parent_index)



    # for i in range(1, len(target_parent_index)):
    #     print(i)
    #     print(ref_skeleton.animated_joints[i] + ' parent is: ')
    #     print(ref_skeleton.animated_joints[ref_parent_index[i]])
    #     print(target_skeleton.animated_joints[i] + ' parent is: ')
        # print(target_skeleton.animated_joints[target_parent_index[i]])

    ### create chosen_joints

def test1():
    ##load_yaml()
    target_filename = r'D:\gits\deep-motion-editing\style_transfer\global_info\skeleton_CMU.yml'
    with open(target_filename, 'r') as input:
        data = yaml.load(input, Loader=yaml.Loader)
    print(data)
    save_path = r'D:\gits\deep-motion-editing\style_transfer\global_info'
    # with open(os.path.join(save_path, 'test.yml'), 'w') as f:
    #     tmp = yaml.dump(data, f, default_flow_style=False)
        # tmp = yaml.dump(data, indent=4)
        # f.write(tmp)
        # f.close()
    with open(os.path.join(save_path, 'test.yml'), 'r') as f:
        tmp = yaml.load(f, Loader=yaml.Loader)
    print(data == tmp)


if __name__ == "__main__":
    # run_MotionEnDecoder()

    # create_mk_skeleton()

    test1()