# encoding: UTF-8

from morphablegraphs.animation_data import BVHReader, BVHWriter
import numpy as np
import os

def export_skeleton(bvhfile, save_path=None):
    bvhreader = BVHReader(bvhfile)
    # default_frame = np.zeros([1, bvhreader.frames.shape[1]])
    # if save_path is not None:
    #     BVHWriter(os.path.join(save_path, 'skeleton.bvh'), bvhreader, default_frame, bvhreader.frame_time,
    #               is_quaternion=False)
    print(bvhreader.node_names['Head_EndSite'])


if __name__ == "__main__":
    test_file = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_pickLeft\first\pickLeft_002_3_reach_621_777.bvh'
    save_path = r'C:\Users\hadu01\temp'
    export_skeleton(test_file, save_path)