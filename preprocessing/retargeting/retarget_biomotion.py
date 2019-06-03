# encoding: UTF-8
import scipy.io


def create_targets_from_mat_file():
    mat_file = r'E:\mocap_data\mocap_matlab\walking_matlab\walk_80.mat'
    joint_list = []
    data = scipy.io.loadmat(mat_file)
    data['motion_data'] = data.pop('md')
    print(data['motion_data'].shape)


if __name__ == "__main__":
    create_targets_from_mat_file()