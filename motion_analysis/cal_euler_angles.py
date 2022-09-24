# encoding: UTF-8

from morphablegraphs.utilities import load_json_file
import numpy as np


def quat_variance_analysis():
    test_data_file = r'C:\repo\data\1 - MoCap\7 - Mocap analysis\elementary_action_walk\beginRightStance\quat_frames.json'
    test_data = load_json_file(test_data_file)
    quat_frames = np.asarray(test_data.values())
    print(quat_frames.shape)

if __name__ == "__main__":
    quat_variance_analysis()