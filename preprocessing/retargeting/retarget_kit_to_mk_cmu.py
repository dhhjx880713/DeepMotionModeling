import os
import collections
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/../..')
from mosi_utils_anim.retargeting.directional_constraints_retargeting import retarget_motion, \
    estimate_scale_factor, retarget_single_motion, create_direction_constraints, align_ref_frame, retarget_folder
import pandas as pd
from mosi_utils_anim.utilities import write_to_json_file
import numpy as np


def convert_fke_to_panim():
    ### fke is csv format
    csv_file = r'D:\gits\text2motion-thesis_project\dataset\kit-mocap\00002_quat.fke'
    fke_data = pd.read_csv(csv_file)
    motion_data = fke_data.values[:, 1:]
    n_frames, n_dims = motion_data.shape
    motion_data = motion_data.reshape(n_frames, n_dims//3, 3)
    origin_point = np.array([motion_data[0, 0, 0], motion_data[0, 0, 1], motion_data[0, 0, 2]])
    motion_data = motion_data - origin_point
    scale_factor = 0.1
    motion_data = motion_data * scale_factor
    panim_data = {'motion_data': motion_data.tolist()}
    write_to_json_file(r'E:\tmp\fke.panim', panim_data)


def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    return -softmax(-x, **kw)


def Xnect_skeleton():
    '''
    * 3D joints - { 0:'head TOP', 1:'neck',  2:'Rsho',  3:'Relb',  4:'Rwri',  5:'Lsho',  6:'Lelb', 7:'Lwri', 8:'Rhip', 9:'Rkne', 10:'Rank', 11:'Lhip', 12:'Lkne', 13:'Lank', 14: Root, 15: Spine, 16:'Head', 17: 'Rhand', 18: 'LHand', 19: 'Rfoot', 20: 'Lfoot' }
    '''
    motion_file3d = r'E:\XNECT_PROJECT\XNECT\x64\Release\raw3D.txt'
    motion_file2d = r'E:\XNECT_PROJECT\XNECT\x64\Release\raw2D.txt'
    input_data = pd.read_csv(motion_file3d, sep=' ', header=None)
    input_values = input_data.values

    export_folder = os.path.split(motion_file3d)[0]
    # motion_data = input_values[:, 2:-1]
    character_ids = np.unique(input_values[:, 1])
    print(character_ids)
    motion_data = {}
    scale = 0.1

    """ Put on the floor """
    foot_heights = np.minimum(cartesian_frames[:, fid_l, 1], cartesian_frames[:, fid_r, 1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    cartesian_frames = cartesian_frames - floor_height

    for id in character_ids:
        motion_data[id] = []
    for line in input_values:
        motion_data[line[1]].append(line[2: -1])
    for id in character_ids:
        n_frames = len(motion_data[id])
        joint_position = np.reshape(motion_data[id], (n_frames, 21, 3))
        joint_position = joint_position - np.array([joint_position[0, 0, 0], 0.0, joint_position[0, 0, 2]])
        joint_position *= scale
        panim_data = {'motion_data': joint_position.tolist()}
        write_to_json_file(os.path.join(export_folder, str(int(id)) + '.panim'), panim_data)
    



def retareget_kit_tomk_cmu():
    from pathlib import Path
    ## load c3d file
    file_src = r'D:\gits\text2motion-thesis_project\dataset\kit-mocap\00001_mmm.xml'
    p = Path(file_src)
    kit_skeleton_joint_list = ['BLNx_joint', 'BLNy_joint', 'BLNz_joint', 'BPx_joint', 'BPy_joint', 'BPz_joint', 'BTx_joint', 'BTy_joint', 'BTz_joint', 'BUNx_joint', 'BUNy_joint', 'BUNz_joint', 'LAx_joint', 'LAy_joint', 'LAz_joint', 'LEx_joint', 'LEz_joint', 'LHx_joint', 'LHy_joint', 'LHz_joint', 'LKx_joint', 'LSx_joint', 'LSy_joint', 'LSz_joint', 'LWx_joint', 'LWy_joint', 'LFx_joint', 'LMrot_joint', 'RAx_joint', 'RAy_joint', 'RAz_joint', 'REx_joint', 'REz_joint', 'RHx_joint', 'RHy_joint', 'RHz_joint', 'RKx_joint', 'RSx_joint', 'RSy_joint', 'RSz_joint', 'RWx_joint', 'RWy_joint', 'RFx_joint', 'RMrot_joint']
    n_joints = (len(kit_skeleton_joint_list) - 1) / 3
    print(n_joints)
    


if __name__ == "__main__":
    # convert_fke_to_panim()
    # Xnect_skeleton()
    retareget_kit_tomk_cmu()