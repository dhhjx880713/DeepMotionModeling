# encoding: UTF-8

from morphablegraphs.motion_analysis import BVHAnalyzer
from morphablegraphs.animation_data import BVHReader
import matplotlib.pyplot as plt


def cal_and_plot_interested_joint_speed():
    test_bvhfile = r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_demotakes\screw\PT_Schraube_1.bvh'
    test_joint = 'RightHand'
    bvhreader = BVHReader(test_bvhfile)
    motion = BVHAnalyzer(bvhreader)
    right_hand_speed = motion.get_joint_speed_each_dim(test_joint)
    # print(right_hand_speed.shape)
    left_hand_speed = motion.get_joint_speed_each_dim('LeftHand')
    fig = plt.figure()
    plt.subplot(311)
    plt.plot(left_hand_speed[:, 0], 'b', label='left hand speed')
    plt.plot(right_hand_speed[:, 0], 'r', label='right hand speed')
    plt.title('X axis')

    plt.subplot(312)
    plt.plot(left_hand_speed[:, 1], 'b', label='left hand speed')
    plt.plot(right_hand_speed[:, 1], 'r', label='right hand speed')
    plt.title('Y axis')

    plt.subplot(313)
    plt.plot(left_hand_speed[:, 2], 'b', label='left hand speed')
    plt.plot(right_hand_speed[:, 2], 'r', label='right hand speed')
    plt.title('Z axis')

    plt.show()


def cal_and_plot_interested_joint_acc():
    test_bvhfile = r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_demotakes\screw\PT_Schraube_1.bvh'
    test_joint = 'RightHand'
    bvhreader = BVHReader(test_bvhfile)
    motion = BVHAnalyzer(bvhreader)
    right_hand_acc = motion.get_joint_acceleration(test_joint)
    # print(right_hand_speed.shape)
    left_hand_acc = motion.get_joint_acceleration('LeftHand')
    fig = plt.figure()
    plt.subplot(311)
    plt.plot(left_hand_acc[:, 0], 'b', label='left hand acc')
    plt.plot(right_hand_acc[:, 0], 'r', label='right hand acc')
    plt.title('X axis')

    plt.subplot(312)
    plt.plot(left_hand_acc[:, 1], 'b', label='left hand acc')
    plt.plot(right_hand_acc[:, 1], 'r', label='right hand acc')
    plt.title('Y axis')

    plt.subplot(313)
    plt.plot(left_hand_acc[:, 2], 'b', label='left hand acc')
    plt.plot(right_hand_acc[:, 2], 'r', label='right hand acc')
    plt.title('Z axis')

    plt.show()


if __name__ == '__main__':
    # cal_and_plot_interested_joint_speed()
    cal_and_plot_interested_joint_acc()