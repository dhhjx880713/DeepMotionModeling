"""
create phase annotation and gait from customized foot contact file
"""
import numpy as np
import glob
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from mosi_utils_anim.animation_data import BVHReader


def create_gait_for_style_file(input_file):
    """create dummy gait annotation from filename
    if walking in the filename, the gait annotation for all frames is [0, 1, 0, 0, 0, 0, 0, 0]
    if running in the filename, the gait annotaiton for all frames is [0, 0, 1, 0, 0, 0, 0, 0]
    
    Arguments:
        input_file {[string} -- file path
    """
    bvhreader = BVHReader(input_file)
    n_frames = len(bvhreader.frames)
    if "walking" in input_file:
        gait = np.tile([0, 1, 0, 0, 0, 0, 0, 0], [n_frames, 1])
    elif "running" in input_file:
        gait = np.tile([0, 0, 1, 0, 0, 0, 0, 0], [n_frames, 1])
    else:
        raise ValueError("Unknown gait")
    np.savetxt(input_file.replace('bvh', 'gait'), gait, fmt='%1.6f')


def create_gait_files():
    motion_data_folder = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\style_locomotion\sexy_tmp'
    bvhfiles = glob.glob(os.path.join(motion_data_folder, '*.bvh'))
    for bvhfile in bvhfiles:
        print(bvhfile)
        create_gait_for_style_file(bvhfile)


def create_phase_from_foot_contact(input_file,save_filename):
    """Right contact is 0, the next left contact is 0.5, the next right contact is 0 (1 in the phase space, 0 == 1)
    So left contact is always marked as 0.5
    if the file is start with right contact, then it is marked as 0
    
    Arguments:
        input_file {string} -- file path
        example: each line is a contact annotation
            left 0 
            right 69
            left 136
            right 200
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    # current_foot_contact = None
    # prev_foot_contact = None
    annotation = []
    for l in lines:
        segs = l.strip().split(' ')
        contact_anno = {'foot_contact': segs[0],
                        'frame_index': int(segs[1])}
        annotation.append(contact_anno)
    phase = np.zeros(annotation[-1]['frame_index'] + 1) 
    for i in range(len(annotation) - 1):

        if annotation[i]['foot_contact'] == "left" and annotation[i+1]['foot_contact'] == 'right':
            phase[annotation[i]['frame_index']] = 0.5
            phase[annotation[i+1]['frame_index']] = 0.0
            phase[annotation[i]['frame_index']: annotation[i+1]['frame_index']] = np.linspace(0.5, 1.0, annotation[i+1]['frame_index'] - annotation[i]['frame_index'] + 1)[:-1]
        elif annotation[i]['foot_contact'] == 'right' and annotation[i+1]['foot_contact'] == 'left':
            phase[annotation[i]['frame_index']] = 0.0
            phase[annotation[i+1]['frame_index']] = 0.5
            phase[annotation[i]['frame_index']: annotation[i+1]['frame_index']] = np.linspace(0.0, 0.5, annotation[i+1]['frame_index'] - annotation[i]['frame_index'] + 1)[:-1]
        else:

            raise ValueError("Unknown annotation!")
     
    np.savetxt(save_filename, phase, fmt='%1.5f')


def create_phase_files():
    motion_data_folder = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\style_locomotion\sexy_tmp'
    bvhfiles = glob.glob(os.path.join(motion_data_folder, '*.bvh'))

    for bvhfile in bvhfiles:
        print(bvhfile)
        footsteps_file = bvhfile.replace('.bvh', '_footsteps.txt')
        save_filename = bvhfile.replace('.bvh', '.phase')
        create_phase_from_foot_contact(footsteps_file, save_filename)


def mirror_footstep_annotation(input_file):
    """change left to right, right to left
    
    Arguments:
        input_file {string} -- file path
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    for l in lines:
        if "left" in l:
            l.replace("left", "right")
        if "right" in l:
            l.replace("right", "left")
    save_filename = input_file.replace("footsteps.txt", "mirror_footsteps.txt")
    with open(save_filename, 'w') as out:
        out.writelines(lines)


def mirror_footstep_files():
    # input_folder = r'D:\workspace\mocap_data\mk_cmu_retargeting_default_pose\style_locomotion\childlike'
    input_folder = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\style_locomotion\sexy_tmp'
    footstep_files = glob.glob(os.path.join(input_folder, '*_footsteps.txt'))
    for f in footstep_files:
        print(f)
        mirror_footstep_annotation(f)


def test():
    input_file = r'D:\workspace\mocap_data\mk_cmu_retargeting_default_pose\style_locomotion\angry\angry_fast walking_145.txt'
    create_phase_from_foot_contact(input_file)



def change_filename_batch():
    input_folder = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\style_locomotion\sexy_tmp'
    target_files = glob.glob(os.path.join(input_folder, '*.txt'))
    for f in target_files:
        os.rename(f, f.replace('.txt', '_footsteps.txt'))


def rename_mirror_files():
    input_folder = r'D:\workspace\mocap_data\mk_cmu_retargeting_default_pose\style_locomotion\angry\mirrored'
    target_files = glob.glob(os.path.join(input_folder, '*.bvh'))
    for f in target_files:
        os.rename(f, f.replace('.bvh', '_mirror.bvh'))   


def test1():
    input_file = r'D:\workspace\mocap_data\mk_cmu_retargeting_default_pose\style_locomotion\angry\angry_fast walking_145_footsteps.txt'
    mirror_footstep_annotation(input_file)


if __name__ == "__main__":
    # test()
    # change_filename_batch()
    # create_phase_files()
    # test1()
    create_gait_files()
    # mirror_footstep_files()
    # rename_mirror_files()
