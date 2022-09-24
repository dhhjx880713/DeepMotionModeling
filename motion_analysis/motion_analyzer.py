# encoding: UTF-8
from morphablegraphs.utilities import get_data_analysis_folder, write_to_json_file, load_json_file, \
                                      get_aligned_data_folder, get_cut_data_folder, load_bvh_files_from_folder
import os
from helpfunctions import get_smoothed_quat_frames, cartesian_error_measure, load_skeleton, load_bvhreader
import collections
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from morphablegraphs.animation_data import BVHWriter, get_cartesian_coordinates_from_euler_full_skeleton
from morphablegraphs.construction.fpca import FPCASpatialData
import matplotlib.pyplot as plt


class MotionAnalyzer(object):

    def __init__(self, elementary_action, motion_primitive):
        self.elementary_action = elementary_action
        self.motion_primitive = motion_primitive
        self.skeleton = load_skeleton()
        self.bvhreader = load_bvhreader()

    def set_data_dir(self, repo_dir):
        self.repo_dir = repo_dir

    def get_smoothed_quat_frames(self):
        data_analysis_folder = get_data_analysis_folder(self.elementary_action,
                                                        self.motion_primitive,
                                                        self.repo_dir)
        data_filename = os.path.join(data_analysis_folder, 'smoothed_quat_frames.json')

        if not os.path.exists(data_filename):
            smoothed_quat_frames = get_smoothed_quat_frames(self.elementary_action,
                                                            self.motion_primitive)
            write_to_json_file(data_filename,
                               smoothed_quat_frames)
        else:
            smoothed_quat_frames = load_json_file(data_filename)
        smoothed_quat_frames = collections.OrderedDict(sorted(smoothed_quat_frames.items()))
        self.smoothed_quat_frames = np.asarray(smoothed_quat_frames.values())
        self.filenames = smoothed_quat_frames.keys()

    def get_scaled_smoothed_quat_frames(self):
        data_analysis_folder = get_data_analysis_folder(self.elementary_action,
                                                        self.motion_primitive,
                                                        self.repo_dir)
        data_filename = os.path.join(data_analysis_folder, 'scaled_smoothed_quat_frames.json')
        if not os.path.exists(data_filename):
            scaled_smoothed_quat_frames, scale_vecotr = get_smoothed_quat_frames(self.elementary_action,
                                                                   self.motion_primitive,
                                                                   scale_root=True)
            output_data = {'scale_vector': scale_vecotr,
                           'data': scaled_smoothed_quat_frames}
            write_to_json_file(data_filename,
                               output_data)
        else:
            data = load_json_file(data_filename)
            scaled_smoothed_quat_frames = data['data']
            scale_vecotr = data['scale_vector']
        self.scale_vector = scale_vecotr
        self.scaled_smoothed_quat_frames_dic = collections.OrderedDict(sorted(scaled_smoothed_quat_frames.items()))
        self.scaled_smoothed_quat_frames = np.asarray(self.scaled_smoothed_quat_frames_dic.values())
        self.filenames = self.scaled_smoothed_quat_frames_dic.keys()

    @staticmethod
    def reshape_motion_data_for_PCA(data_mat):
        data_mat = np.asarray(data_mat)
        assert len(data_mat.shape) == 3
        n_samples, n_frames, n_dims = data_mat.shape
        return np.reshape(data_mat, (n_samples, n_frames * n_dims))

    @staticmethod
    def reshape_2D_mat_to_motion_data(data_mat_2d, origin_shape):
        assert len(origin_shape) == 3
        data_mat_2d = np.asarray(data_mat_2d)
        n_samples, n_frames, n_dims = origin_shape
        assert n_samples * n_frames * n_dims == data_mat_2d.shape[0] * data_mat_2d.shape[1]
        return np.reshape(data_mat_2d, origin_shape)

    def run_pca_on_smoothed_quat_frames(self, n_pc):
        self.reshaped_smooth_quat_frames = MotionAnalyzer.reshape_motion_data_for_PCA(self.smoothed_quat_frames)
        self.pca_unscaled = PCA(n_components=n_pc)
        self.pca_unscaled.fit(self.reshaped_smooth_quat_frames)

    def run_pca_on_scaled_root_translation_quat_frames(self, n_pc):
        self.pca_scaled = PCA(n_components=n_pc)
        self.reshaped_scaled_data = MotionAnalyzer.reshape_motion_data_for_PCA(self.scaled_smoothed_quat_frames)
        self.pca_scaled.fit(self.reshaped_scaled_data)

    def export_reconstructed_data_smoothed_quat_frames(self, export_folder):
        projection = self.pca_unscaled.transform(self.reshaped_smooth_quat_frames)
        backprojection = self.pca_unscaled.inverse_transform(projection)
        reshaped_backprojection = MotionAnalyzer.reshape_2D_mat_to_motion_data(backprojection,
                                                                               self.smoothed_quat_frames.shape)
        print('Cartesian reconstruction error is: ', cartesian_error_measure(self.smoothed_quat_frames,
                                                                             reshaped_backprojection,
                                                                             self.skeleton))
        for i in range(len(self.filenames)):
            output_filename = os.path.join(export_folder, self.filenames[i])
            BVHWriter(output_filename, self.skeleton, reshaped_backprojection[i], self.skeleton.frame_time,
                      is_quaternion=True)

    def export_reconstructed_data_scaled_smoothed_quat_frames(self, export_folder):
        projection = self.pca_scaled.transform(self.reshaped_scaled_data)
        backprojection = self.pca_scaled.inverse_transform(projection)
        reshaped_backprojection = MotionAnalyzer.reshape_2D_mat_to_motion_data(backprojection,
                                                                               self.scaled_smoothed_quat_frames.shape)
        reshaped_backprojection[:, :, 0] *= self.scale_vector[0]
        reshaped_backprojection[:, :, 1] *= self.scale_vector[1]
        reshaped_backprojection[:, :, 2] *= self.scale_vector[2]
        print('Cartesian reconstruction error is: ', cartesian_error_measure(self.smoothed_quat_frames,
                                                                             reshaped_backprojection,
                                                                             self.skeleton))
        for i in range(len(self.filenames)):
            output_filename = os.path.join(export_folder, self.filenames[i])
            BVHWriter(output_filename, self.skeleton, reshaped_backprojection[i], self.skeleton.frame_time,
                      is_quaternion=True)

    def run_fpca_on_scaled_root_quat_frames(self, n_basis, n_pc):
        self.fpca = FPCASpatialData(self.scaled_smoothed_quat_frames_dic, n_basis=n_basis, n_pc=n_pc)
        self.fpca.fpca_on_spatial_data()

    def export_reconstructed_data_fpca(self, export_folder):
        backprojected_functional_data = self.fpca.fpcaobj.backproject_data(self.fpca.fpcaobj.low_vecs)
        # print(backprojected_functional_data.shape)
        reshaped_backprojection = self.fpca.fpcaobj.reshape_fd_back(backprojected_functional_data,
                                                               self.fpca.fpcaobj.origin_shape)
        backprojection = self.fpca.fpcaobj.from_fd_to_data(reshaped_backprojection,
                                                      self.scaled_smoothed_quat_frames.shape[1])
        backprojection[:, :, 0] *= self.scale_vector[0]
        backprojection[:, :, 1] *= self.scale_vector[1]
        backprojection[:, :, 2] *= self.scale_vector[2]
        print('Cartesian reconstruction error is: ', cartesian_error_measure(self.smoothed_quat_frames,
                                                                             backprojection,
                                                                             self.skeleton))
        for i in range(len(self.filenames)):
            output_filename = os.path.join(export_folder, self.filenames[i])
            BVHWriter(output_filename, self.skeleton, backprojection[i], self.skeleton.frame_time,
                      is_quaternion=True)

    def plot_absolute_joint_trajectory(self, joint_name):
        data_analysis_folder = get_data_analysis_folder(self.elementary_action,
                                                        self.motion_primitive,
                                                        self.repo_dir)
        cartesian_position_data = load_json_file(os.path.join(data_analysis_folder,
                                                              'joint_absolute_cartesian_position.json'))
        # print(np.asarray(cartesian_position_data[cartesian_position_data.keys()[0]]).shape)
        joint_pos_data = np.asarray(cartesian_position_data[joint_name])
        n_samples, n_frames, n_dims = joint_pos_data.shape
        x = range(n_frames)
        N = 300
        fig = plt.figure()
        for i in range(N):
            plt.plot(x, joint_pos_data[i, :, 1])
        plt.show()

    def plot_absolute_joint_trajectory_after_aligned(self, joint_name, N=100):
        aligned_data_folder = get_aligned_data_folder(self.elementary_action,
                                                      self.motion_primitive,
                                                      self.repo_dir)
        aligned_motion_data_dic = load_bvh_files_from_folder(aligned_data_folder)
        euler_frames = np.asarray(aligned_motion_data_dic.values())
        n_samples, n_frames, n_dims = euler_frames.shape
        print(euler_frames.shape)
        if N > n_samples:
            N = n_samples
        cartesian_data = np.zeros((n_samples, n_frames, 3))
        for i in range(N):
            for j in range(n_frames):
                cartesian_data[i, j] = get_cartesian_coordinates_from_euler_full_skeleton(self.bvhreader,
                                                                                          self.skeleton,
                                                                                          joint_name,
                                                                                          euler_frames[i, j])
        x = range(n_frames)
        fig = plt.figure()
        for i in range(N):
            plt.plot(x, cartesian_data[i, :, 1])
        plt.title('Walk leftStance', fontsize=15)
        plt.xlabel('Number of frames', fontsize=15)
        plt.ylabel('Height of Left Toe $[cm]$', fontsize=15)
        plt.show()

    def plot_absolute_joint_trajectory_before_aligned(self, joint_name, N=100):
        cut_data_folder = get_cut_data_folder(self.elementary_action,
                                              self.motion_primitive,
                                              self.repo_dir)
        cut_motion_data_dic = load_bvh_files_from_folder(cut_data_folder)
        plot_data = []
        n_samples = len(cut_motion_data_dic)
        if N > n_samples:
            N = n_samples
        flielist = cut_motion_data_dic.keys()
        for i in range(N):
            euler_frames = cut_motion_data_dic[flielist[i]]
            cartesian_data = np.zeros((len(euler_frames), 3))
            for j in range(len(euler_frames)):
                cartesian_data[j] = get_cartesian_coordinates_from_euler_full_skeleton(self.bvhreader,
                                                                                       self.skeleton,
                                                                                       joint_name,
                                                                                       euler_frames[j])
            plot_data.append(cartesian_data)
        fig = plt.figure()
        for i in range(N):
            x = range(len(plot_data[i]))
            plt.plot(x, plot_data[i][:, 1])
        plt.title('Walk leftStance', fontsize=15)
        plt.xlabel('Number of frames', fontsize=15)
        plt.ylabel('Height of Left Toe $[cm]$', fontsize=15)
        plt.show()


if __name__ == "__main__":
    elementary_action = 'walk'
    motion_primitive = 'leftStance'
    motion_analyzer = MotionAnalyzer(elementary_action,
                                     motion_primitive)
    motion_analyzer.set_data_dir(r'C:\repo')
    # motion_analyzer.plot_absolute_joint_trajectory('LeftFoot')
    motion_analyzer.plot_absolute_joint_trajectory_after_aligned('LeftFoot', N=300)
    motion_analyzer.plot_absolute_joint_trajectory_before_aligned('LeftFoot', N=300)
    # motion_analyzer.get_smoothed_quat_frames()
    # motion_analyzer.get_scaled_smoothed_quat_frames()
    # motion_analyzer.run_pca_on_smoothed_quat_frames(5)
    # motion_analyzer.export_reconstructed_data_smoothed_quat_frames(r'C:\experiment data\elementary_action_walk\leftStance_reconstruction')
    # motion_analyzer.run_pca_on_scaled_root_translation_quat_frames(5)
    # motion_analyzer.export_reconstructed_data_scaled_smoothed_quat_frames(r'C:\experiment data\elementary_action_walk\leftStance')
    # motion_analyzer.run_fpca_on_scaled_root_quat_frames(47, 10)
    # motion_analyzer.export_reconstructed_data_fpca(r'C:\experiment data\scaled_root_fpca_reconstruction')
