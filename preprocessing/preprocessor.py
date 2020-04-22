import numpy as np
import glob
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder, BVHWriter
from mosi_utils_anim.animation_data.utils import convert_euler_frames_to_cartesian_frames, rotate_euler_frames
from preprocessing.utils import estimate_floor_height

"""simply extract joint poisiton from bvh files with fixed sized sliding window. All the clips are normalized in the way that starting position and orientation are the same
"""


class Preprocessor(object):
    """a general preprocessing class to support different proprocessing methods
    
    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self):
        self.bvhreaders = []
        self.skeleton = None
        self.euler_frames = {}
    
    def load_bvh_files_from_directory(self, dir):
        """
        
        Arguments:
            dir {str} -- path to the folder
        """
        print("Data Loading...")
        bvhfiles = []
        Preprocessor.get_files(dir, bvhfiles)
        input_path_segs = dir.split(os.sep)
        if bvhfiles != []:
            for bvhfile in bvhfiles:
                bvhreader = BVHReader(bvhfile)
                path_segs = bvhfile.split(os.sep)
                res = [i for i in path_segs if i not in input_path_segs]

                self.bvhreaders.append(bvhreader)
                self.euler_frames['_'.join(res)] = bvhreader.frames
        self.skeleton = SkeletonBuilder().load_from_bvh(self.bvhreaders[0])

    @staticmethod
    def get_files(path, files=[], suffix='.bvh'):
        files += glob.glob(os.path.join(path, '*' + suffix))
        subdirs = next(os.walk(path))[1]
        if subdirs != []:
            for subdir in subdirs:
                Preprocessor.get_files(os.path.join(path, subdir), files)
    
    def translate_root_to_target(self, target_point):
        """translate root position of the first frame of bvh motions to the target position. The translation is only applied on the floor
        
        Arguments:
            target_point {numpy.array} -- e.g.: numpy.array([0, 0])
        """
        print('translate motion data...')
        for key, value in self.euler_frames.items():
            root_pos = self.skeleton.nodes[self.skeleton.root].get_global_position_from_euler(value[0])
            offset = np.array([target_point[0] - root_pos[0], 0.0, target_point[1] - root_pos[2]])
            self.euler_frames[key][:, :3] = value[:, :3] + offset
    
    def rotate_euler_frames(self, target_direction, body_joints, global_rotation=False):
        """rotate euler frames about y axis to face target direction
        
        Arguments:
            target_direction {2d numpy.array}
        """
        print("align motion data...")
        for key, value in self.euler_frames.items():
            self.euler_frames[key] = rotate_euler_frames(value, 0, target_direction, body_joints, self.skeleton, 
                                                         rotation_order=self.skeleton.nodes[self.skeleton.root].rotation_order)

    def shift_on_floor(self, foot_joints):
        """translate motion to the floor

        """
        for key, value in self.euler_frames.items():
            foot_positions = convert_euler_frames_to_cartesian_frames(self.skeleton, value, animated_joints=foot_joints)
            foot_heights = foot_positions.min(axis=1)[:, 1]
            floor_height = estimate_floor_height(foot_heights)
            self.euler_frames[key][:, :3] = value[:, :3] - floor_height

    def get_global_positions(self, joint_list=[]):
        global_poss = {}
        if joint_list != []:
            for key, value in self.euler_frames.items():
                print(key)
                global_poss[key] = convert_euler_frames_to_cartesian_frames(self.skeleton, value, animated_joints=joint_list)
        else:
            for key, value in self.euler_frames.items():
                print(key)
                global_poss[key] = (convert_euler_frames_to_cartesian_frames(self.skeleton, value))
        return global_poss
    
    def save_files(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for key, value in self.euler_frames.items():
            BVHWriter(os.path.join(save_path, key), self.skeleton, value, self.skeleton.frame_time, is_quaternion=False)