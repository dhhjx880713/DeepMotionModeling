import os 
import sys
import numpy as np 
dirname = os.path.dirname(__file__)
sys.path.append(os.path.join(dirname, '../..'))
from mosi_dev_deepmotionmodeling.preprocessing.preprocessing import process_file


def process_single_file():
    ## h36m file
    input_file = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\h36m\S1\Purchases 1.bvh'
    res = process_file(input_file)
    res = np.asarray(res)
    print(res.shape)





if __name__ == "__main__":
    process_single_file()