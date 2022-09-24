# encoding: UTF-8
from morphablegraphs.construction.preprocessing.motion_dtw import MotionDynamicTimeWarping
from morphablegraphs.animation_data import BVHReader


def DTWDist(bvh_file1, bvh_file2):
    # calculate dynamic time wraping distance between two motion clips
    dtw = MotionDynamicTimeWarping()
    bvhreader1 = BVHReader(bvh_file1)
    bvhreader2 = BVHReader(bvh_file2)
    dtw.ref_bvhreader = bvhreader1
    test_motion = {'filename': bvhreader1.filename, 'frames': bvhreader1.frames}
    ref_motion = {'filename': bvhreader2.filename, 'frames': bvhreader2.frames}
    distgrid = dtw.get_distgrid(ref_motion, test_motion)
    res = dtw.calculate_path(distgrid)
    return res[2]



if __name__ == "__main__":
    bvhfile1 = r'C:\repo\data\1 - MoCap\3 - Cutting\elementary_action_walk\leftStance\walk_001_3_leftStance_310_354.bvh'
    bvhfile2 = r'C:\repo\data\1 - MoCap\3 - Cutting\elementary_action_walk\leftStance\walk_001_3_leftStance_310_354.bvh'
    dist = DTWDist(bvhfile1, bvhfile2)
    print(dist)