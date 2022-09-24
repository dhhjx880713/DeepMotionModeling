# encoding: UTF-8

from morphablegraphs.animation_data import BVHReader
import os
import glob


class FrameCounter(object):
    def __init__(self):
        pass

    def set_folders(self, folders):
        self.folder_list = folders

    def count_frames(self):
        n_frames = 0
        for folder in self.folder_list:
            bvhfiles = glob.glob(os.path.join(folder, '*.bvh'))
            for filename in bvhfiles:
                bvhreader = BVHReader(filename)
                n_frames += len(bvhreader.frames)
        return n_frames



if __name__ == "__main__":
    # test_folders = [r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_walk',
    #                 r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_walk\Take_walk_s',
    #                 r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\ext\walk']
    # lookAt = [r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_look_around']
    # carry = [r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_carry',
    #          r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_carry_extended',
    #          r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\ext\carry']
    # screw = [r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\Takes_screw',
    #          r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\demotakes\screw']
    # transfer = [r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_transfer']
    sidestep = [r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_sidecarry',
                r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_sidestep',
                r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\ext\sidesteps']
    # one_hand_pick = [r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\Takes_ext\pickplace']
    # two_hand_pick = [r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_pick',
    #                  r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_pick_extended']
    # one_hand_place = [r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\Place_one_hand']
    # two_hand_place = [r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_place']
    frame_counter = FrameCounter()
    frame_counter.set_folders(sidestep)
    print(frame_counter.count_frames())