import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/../..')
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from mosi_dev_deepmotionmodeling.mosi_utils_anim.utilities import load_json_file
import mpl_toolkits.mplot3d.axes3d as p3


class PointCloudVisualizer(object):

    def __init__(self):
        super().__init__()
        self.fig = plt.figure()
        self.ax = p3.Axes3D(self.fig)

    def load_panim_data(self, panim_data):
        self.motion_data = panim_data['motion_data']
        self.skeleton = panim_data['skeleton']
        self.joint_list = list(self.skeleton.keys())
    
    def plot_one_frame(self, frame_index):
        fig = plt.figure()
        self.ax.cla()
        ax = p3.Axes3D(fig)
        for joint, joint_dict in self.skeleton.items():
            if joint_dict['parent'] is not None:
                joint_pos = self.motion_data[frame_index, joint_dict['index']]
                parent_index = self.joint_list.index(joint_dict['parent'])
                parent_pos = self.motion_data[frame_index, parent_index]
                tmp = np.array([[joint_pos[0], joint_pos[2], joint_pos[1]], 
                                [parent_pos[0], parent_pos[2], parent_pos[1]]])
                tmp = np.transpose(tmp)
                ax.plot(*tmp, color='b')            
        # ax.set_aspect('equal')
        ax.autoscale(tight=True)
        plt.show()

    def plot_frames(self, stride=1, root_offset=[0, 0, 0]):
        self.ax.cla()
        root_offset = np.array(root_offset)
        count = 0
        for i in range(0, len(self.motion_data), stride):
            self._plot_one_frame(self.motion_data[i], root_offset * count)
            count += 1

        plt.show()
    
    def _plot_one_frame(self, frame_data, root_offset):
        frame_data += root_offset
        for joint, joint_dict in self.skeleton.items():
            if joint_dict['parent'] is not None:
                joint_pos = frame_data[joint_dict['index']]
                parent_index = self.joint_list.index(joint_dict['parent'])
                parent_pos = frame_data[parent_index]
                tmp = np.array([[joint_pos[0], joint_pos[2], joint_pos[1]], 
                                [parent_pos[0], parent_pos[2], parent_pos[1]]])
                tmp = np.transpose(tmp) 
                self.ax.plot(*tmp, color='b') 
                                    

def test_visualizer():
    panim_data = load_json_file(r'C:\Users\hadu01\Downloads\bihmpgan_0624\epoch80.panim')
    v = PointCloudVisualizer()
    v.load_panim_data(panim_data)
    v.plot_frames(stride=10, root_offset=[0, 0, -30])



if __name__ == "__main__":
    test_visualizer()