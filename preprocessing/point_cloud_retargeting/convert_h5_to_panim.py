""" Process Human3.6M datdaset
http://vision.imar.ro/human3.6m/description.php
"""

import h5py
import numpy as np
from transformations import rotation_matrix
import os
import glob
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/../..')
from mosi_utils_anim.animation_data.panim import Panim


h6M_SKELETON = [
    {'name': 'Hips', 'parent': None, 'index': 0},
    {'name': 'HipRight', 'parent': 'Hips', 'index': 1},
    {'name': 'KneeRight', 'parent': 'HipRight', 'index': 2},
    {'name': 'FootRight', 'parent': 'KneeRight', 'index': 3},
    {'name': 'ToeBaseRight', 'parent': 'FootRight', 'index': 4},
    {'name': 'Site1', 'parent': 'ToeBaseRight', 'index': 5},
    {'name': 'HipLeft', 'parent': 'Hips', 'index': 6},
    {'name': 'KneeLeft', 'parent': 'HipLeft', 'index': 7},
    {'name': 'FootLeft', 'parent': 'KneeLeft', 'index': 8},
    {'name': 'ToeBaseLeft', 'parent': 'FootLeft', 'index': 9},
    {'name': 'Site2', 'parent': 'ToeBaseLeft', 'index': 10},
    {'name': 'Spine1', 'parent': 'Hips', 'index': 11},
    {'name': 'Spine2', 'parent': 'Spine1', 'index': 12},
    {'name': 'Neck', 'parent': 'Spine2', 'index': 13},
    {'name': 'Head', 'parent': 'Neck', 'index': 14},
    {'name': 'Site3', 'parent': 'Head', 'index': 15},
    {'name': 'ShoulderLeft', 'parent': 'Neck', 'index': 16},
    {'name': 'ElbowLeft', 'parent': 'ShoulderLeft', 'index': 17},
    {'name': 'WristLeft', 'parent': 'ElbowLeft', 'index': 18},
    {'name': 'HandLeft', 'parent': 'WristLeft', 'index': 19},
    {'name': 'HandThumbLeft', 'parent': 'HandLeft', 'index': 20},
    {'name': 'Site4', 'parent': 'HandThumbLeft', 'index': 21},
    {'name': 'WristEndLeft', 'parent': 'HandLeft', 'index': 22},
    {'name': 'Site5', 'parent': 'WristEndLeft', 'index': 23},
    {'name': 'ShoulderRight', 'parent': 'Neck', 'index': 24},
    {'name': 'ElbowRight', 'parent': 'ShoulderRight', 'index': 25},
    {'name': 'WristRight', 'parent': 'ElbowRight', 'index': 26},
    {'name': 'HandRight', 'parent': 'WristRight', 'index': 27},
    {'name': 'HandThumbRight', 'parent': 'HandRight', 'index': 28},
    {'name': 'Site6', 'parent': 'HandThumbRight', 'index': 29},
    {'name': 'WristEndRight', 'parent': 'HandRight', 'index': 30},
    {'name': 'Site7', 'parent': 'WristEndRight', 'index': 31},
]


def convert_h5_to_panim(input_file, save_file):
    """convert .h5 to customized .panim file
    
    Arguments:
        input_file {str} -- .h5 file
        save_file {str} -- .panim (json) file 
    """
    with h5py.File(input_file, 'r') as h5f:
        poses = h5f['3D_positions'][:].T
        num_frames = poses.shape[0]
        poses = poses.reshape(num_frames,-1,3)    
    ## to do: remove hard coded correction
    rotation_angle = -90
    rotation_axis = np.array([1, 0, 0])
    rotmat = rotation_matrix(np.deg2rad(rotation_angle), rotation_axis) 
    ones = np.ones((poses.shape[0], poses.shape[1], 1))
    extended_poses = np.concatenate((poses, ones), axis=-1)
    swapped_poses = np.transpose(extended_poses, (0, 2, 1))

    rotated_poses = np.matmul(rotmat, swapped_poses)
    rotated_poses = np.transpose(rotated_poses, (0, 2, 1))
    rotated_poses = rotated_poses[:, :, :3]    
    panim = Panim()
    panim.setSkeleton(h6M_SKELETON)
    panim.setMotionData(rotated_poses)
    panim.save(save_file)


def convert_h5_to_panim_folder(input_dir, output_dir):
    """convert all .h5 in input directory to .panim and save them in output directory
    
    Arguments:
        input_dir {str} -- path to input directory
        output_dir {str} -- path to output directory
    """
    h5files = glob.glob(os.path.join(input_dir, '*.h5'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for bvhfile in h5files:
        print(bvhfile)
        filename = os.path.split(bvhfile)[-1]
        save_filename = os.path.join(output_dir, filename.replace('h5', 'bvh'))
        convert_h5_to_panim(bvhfile, save_filename)    