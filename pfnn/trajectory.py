import numpy as np
import os
import sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from mosi_utils_anim.animation_data.quaternion import Quaternion
from pfnn_new import PFNN
from mosi_utils_anim.utilities import load_json_file, write_to_json_file
import copy



class Style(object):

    def __init__(self, name=None, bias=1, transition=0.1):
        self.name = name
        self.bias = bias
        self.transition = transition


class Point(object):

    def __init__(self, n_styles, index=0):
        self.index = index
        self.transformation = np.identity(4)
        self.left_sample = np.zeros(3)
        self.right_sample = np.zeros(3)
        self.slope = 0
        self.phase = 0
        self.styles = np.zeros(n_styles)
        self.set_direction(np.array([0, 0, 1]))
        self.set_position(np.array([0, 0, 0]))

    def get_index(self):
        return self.index
    
    def set_index(self, index):
        self.index = index
    
    def set_transformation(self, trans):
        self.transformation = copy.deepcopy(trans) 
    
    def get_transformation(self):
        return copy.deepcopy(self.transformation)
    

    def set_position(self, position):
        position = copy.deepcopy(position)
        self.transformation[0, 3] = position[0]
        self.transformation[1, 3] = position[1]
        self.transformation[2, 3] = position[2]
    
    def get_position(self):
        return self.transformation[:-1, 3]
    
    def set_rotation(self, q):
        trans = np.dot(np.identity(4), q.toMat4())
        self.transformation[:3, :3] = trans[:3, :3]

    def get_rotation(self):
        return Quaternion.fromMat(self.transformation)
    
    def get_direction(self):
        return self.transformation[:-1, 2] / np.linalg.norm(self.transformation[:-1, 2])
    
    def set_direction(self, direction):
        if np.all(direction == 0):
            direction = np.array([0, 0, 1])
        q = Quaternion.between(np.array([0, 0, 1]), direction)
        self.set_rotation(q)

    def get_relative_position_to(self, position):
        """convert the given position vector into relative coodinate
        
        Arguments:
            position {numpy.array} -- (x, y, z)
        """
        assert len(position) == 3
        relative_position = np.dot(np.linalg.inv(self.transformation), np.append(position, 1))
        return relative_position[:-1]
    
    def get_relative_direction_to(self, direction):
        """convert given direction into relative coodinate system
        
        Arguments:
            direction {numpy.array} -- (x, y, z)
        """
        assert len(direction) == 3
        relative_direction = np.dot(np.linalg.inv(self.transformation), np.append(direction, 1))
        return relative_direction[:-1]
    
    def get_left_sample(self):
        return self.left_sample
    
    def set_left_sample(self, left_sample):
        """
        
        Arguments:
            left_sample {numpy.array} -- left sample position
        """
        self.left_sample = left_sample

    def get_right_sample(self):
        return self.right_sample
    
    def set_right_sample(self, right_sample):
        """
        
        Arguments:
            right_sample {numpy.array} -- right sample position
        """
        self.right_sample = right_sample
    
    def get_slope(self):
        return self.slope
    
    def set_slope(self, slope):
        self.slope = slope

class Actor(object):

    def __init__(self, n_joints, positions=None, velocities=None):
        self.n_joints = n_joints
        self.positions = positions
        self.velocities = velocities
    
    def get_positions(self):
        return copy.deepcopy(self.positions)
    
    def set_positions(self, positions):
        self.positions = copy.deepcopy(positions)
    
    def get_velocities(self):
        return copy.deepcopy(self.velocities)
    
    def set_velocities(self, velocities):
        self.velocities = copy.deepcopy(velocities)
    
    def get_root_position(self):
        root_pos = copy.deepcopy(self.positions[0])
        root_pos[1] = 0
        return root_pos
    
    def get_forward_drection(self, body_joint_indices = [10, 20, 2, 27]):
        sdr_l, sdr_r, hip_l, hip_r = body_joint_indices
        across = (self.positions[sdr_l] - self.positions[sdr_r]) + (self.positions[hip_l] - self.positions[hip_r])
        across = across / np.linalg.norm(across)
        forward = np.cross(across, np.array([0, 1, 0]))
        return forward


def get_default_pose():
    defaule_pose_file = r'D:\tmp\tmp\mean_prediction.panim'
    
    pose_data = load_json_file(defaule_pose_file)
    motion_data = pose_data['motion_data']
    skeleton_def = pose_data['skeleton']
    return np.asarray(motion_data[0]), skeleton_def


def get_global_position(pos, trans):
    """convert input position vector into global space
    
    Arguments:
        pos {numpy.array} -- (x, y, z)
        trans {Matrix4x4} -- transformation matrix
    """
    g_pos = np.dot(trans, np.append(pos, 1))
    return g_pos[:-1]


def get_global_direction(dir, trans):
    """convert input direction vector into global space
    
    Arguments:
        dir {numpy.array} -- (x, y, z)
        trans {Matrix4x4} -- transformation matrix
    """

    g_dir = np.dot(trans[:3, :3], dir)
    return g_dir


def get_local_position(pos, trans):
    """convert input position v into local space
    
    Arguments:
        pos {numpy.array} -- (x, y, z))
        trans {Matrix4x4} -- transformation matrix
    """
    l_pos = np.dot(np.linalg.inv(trans), np.append(pos, 1))
    return l_pos[:-1]

def get_local_direction(dir, trans):
    """convert input direction vector into local space
    
    Arguments:
        dir {numpy.array} -- (x, y, z)
        trans {Matrixx4x4} -- transformation matrix
    """

    l_dir = np.dot(np.linalg.inv(trans[:3, :3]), dir)
    return l_dir


class Controller(object):

    def __init__(self, debug=False):
        self.debug = debug
        if self.debug:
            self.frame_index = 0
            self.log_data = {}
        default_pose_data, skeleton_def = get_default_pose()
        self.n_joints = len(skeleton_def)
        self.skeleton_def = skeleton_def
        positions = np.zeros([self.n_joints, 3])
        velocities = np.zeros([self.n_joints, 3])
        positions = default_pose_data
        self.actor = Actor(self.n_joints)
        self.actor.set_positions(positions)
        self.actor.set_velocities(velocities)
        # weight_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\tensorflow\mk_cmu_ground_control_parameters'
        meta_data_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground'
        style_list = create_style_list()
        initial_pos = self.actor.get_root_position()
        initial_dir = self.actor.get_forward_drection()
        self.traj = Trajectory(111, style_list, initial_pos=initial_pos, initial_dir=initial_dir)
        self.input_dims = 342
        self.output_dims = 311
        self.point_sample = 12
        self.n_knots = 4
        self.network = PFNN(4, self.input_dims, self.output_dims, 0.3, name='pfnn')
        # self.network.load_weights(weight_path)
        weight_path = r'D:\workspace\my_git_repos\deepMotionSynthesis\data\pfnn\network_parameters\cmu_ground'
        self.network.load_discrete_weights(n_bins=50, path=weight_path)
        self.load_meta_data(meta_data_path)
        self.unit_scale = 1.0
        self.phase = 0.0
        self.phase_index = 3
        self.gait_transition = 0.25
    
    def load_meta_data(self, meta_data_path):
        self.Xmean = np.fromfile(os.path.join(meta_data_path, 'Xmean.bin'), dtype=np.float32)
        self.Xstd = np.fromfile(os.path.join(meta_data_path, 'Xstd.bin'), dtype=np.float32)
        self.Ymean = np.fromfile(os.path.join(meta_data_path, 'Ymean.bin'), dtype=np.float32)
        self.Ystd = np.fromfile(os.path.join(meta_data_path, 'Ystd.bin'), dtype=np.float32)

    def update(self):
        target_direction = np.array([0, 0, 100])
        # target_vel = 200 * target_direction
        ### update trajectory gait
        for i in range(self.traj.n_styles):

            if i == 1:
                self.traj.points[self.traj.root_index].styles[i] =  (1 - self.gait_transition) * self.traj.points[self.traj.root_index].styles[i] + self.gait_transition * 1.0
                # self.traj.points[self.traj.root_index].styles[i] = 1
            else:
                self.traj.points[self.traj.root_index].styles[i] =  (1 - self.gait_transition) * self.traj.points[self.traj.root_index].styles[i] + self.gait_transition * 0.0
        # print("before prediction: ")
        # evaluate_traj(self.traj)
        self.traj.predict_future_trajectory(target_direction)

        # print("after prediction: ")
        # evaluate_traj(self.traj)
        input_vector = np.zeros(self.input_dims)

        current_root_trans = self.traj.points[self.traj.root_index].get_transformation()
        previous_root_trans = self.traj.points[self.traj.root_index - 1].get_transformation()

        if self.debug:
            
            self.log_data[self.frame_index] = {"positions": [], "directions": [], "local positions": [], "local directions": []}
            for i in range(self.point_sample):
                pos = self.traj.points[i*self.traj.point_density].get_position()
                dir = self.traj.points[i*self.traj.point_density].get_direction()
                local_pos = get_local_position(self.traj.points[i*self.traj.point_density].get_position(), current_root_trans)
                local_dir = get_local_direction(self.traj.points[i*self.traj.point_density].get_direction(), current_root_trans)
                self.log_data[self.frame_index]["positions"].append([pos[0], pos[2]])
                self.log_data[self.frame_index]["directions"].append([dir[0], dir[2]])
                self.log_data[self.frame_index]["local positions"].append([local_pos[0], local_pos[2]])
                self.log_data[self.frame_index]["local directions"].append([local_dir[0], local_dir[2]])
            self.frame_index += 1
        # print("input root position: ", self.traj.points[self.traj.root_index].get_position())
        for i in range(self.point_sample):

            pos = get_local_position(self.traj.points[i*self.traj.point_density].get_position(), current_root_trans)
            dir = get_local_direction(self.traj.points[i*self.traj.point_density].get_direction(), current_root_trans)
            
            input_vector[self.point_sample*0 + i] = self.unit_scale * pos[0]
            input_vector[self.point_sample*1 + i] = self.unit_scale * pos[2]
            input_vector[self.point_sample*2 + i] = dir[0]
            input_vector[self.point_sample*3 + i] = dir[2]
        
        # #### evaluate path input
        # input_path = np.zeros((12, 2))
        # input_direction = np.zeros((12, 2))
        # input_path[:, 0] = input_vector[:12]
        # input_path[:, 1] = input_vector[12:24]
        # input_direction[:, 0] = input_vector[24: 36]
        # input_direction[:, 1] = input_vector[36: 48]

        # print("frame: ", self.frame_index)
        # print("#############################")
        # print("input positions: ", input_path)
        # print("input directions: ", input_direction)


        ### set point style
        for i in range(self.point_sample):
            for j in range(self.traj.n_styles):
                input_vector[self.point_sample * (4 + j) + i] = self.traj.points[i * self.traj.point_density].styles[j]

        # print(input_vector[self.point_sample*4 : self.point_sample * 10])
        ### set body position and velocity
        global_positions = copy.deepcopy(self.actor.get_positions())
        global_velocities = copy.deepcopy(self.actor.get_velocities())
        for i in range(self.n_joints):
            input_vector[10 * self.point_sample + i * 3: 10 * self.point_sample + (i+1) * 3] = self.unit_scale * get_local_position(global_positions[i], previous_root_trans) 
            input_vector[10 * self.point_sample + self.n_joints * 3 * 1 + i * 3 : 10 * self.point_sample + self.n_joints * 3 * 1 + (i+1)*3] = self.unit_scale * get_local_direction(global_velocities[i], previous_root_trans)
        # input_vector[10 * self.point_sample + 3 * 1 * self.n_joints: 10 * self.point_sample + 3 * 2 * self.n_joints] = np.ravel(self.actor.get_velocities())

        ### set all heights to zero
        input_vector[10 * self.point_sample + 3 * 2 * self.n_joints:] = 0

        ### normalize input vector
        input_vector = (input_vector - self.Xmean) / self.Xstd 
        ### predict

        network_input = np.append(input_vector, self.phase)
        output_vector = self.network(network_input[np.newaxis, :])

        ### denormalize output vector
        output_vector = output_vector * self.Ystd + self.Ymean
        output_vector = output_vector[0].numpy()

        # print("predicted x vel: ", output_vector[0])
        # print("predicted z vel: ", output_vector[1])
  
        # print("phase value: ", output_vector[4])
        ### update past trajectory
        non_idle_amount = self.traj.update_trajectory(output_vector, self.unit_scale, current_root_trans)

        # print("stand amount: ", non_idle_amount)
        ### update phase
        ### set phase damping
        damping = 1 - (0.9 * non_idle_amount + 0.1)
        self.phase = (self.phase + (1 - damping) * output_vector[self.phase_index])
        # self.phase = self.phase + 0.1 / (2 * np.pi)
        # if self.phase < 0:
        #     self.phase = self.phase - np.floor(self.phase)
        self.phase = self.phase % 1.0
        # print("non_idle_amount: ", non_idle_amount)
        # print(self.phase)

        ### update posture
        new_positions = np.zeros([self.n_joints, 3])
        new_velocities = np.zeros([self.n_joints, 3])
        pos_index = 32 + self.n_joints * 3 * 0
        vel_index = 32 + self.n_joints * 3 * 1
        # current_root_trans = self.traj.points[self.traj.root_index].get_transformation()
        for i in range(self.n_joints):
            new_pos = output_vector[pos_index + i * 3 : pos_index + (i + 1) * 3] / self.unit_scale
            new_velocity = output_vector[vel_index + i * 3 : vel_index + (i + 1) * 3] / self.unit_scale
            prev_local_pos = get_local_position(global_positions[i], current_root_trans)
            pos = vector_interpolation(prev_local_pos + new_velocity, new_pos, 0.5)

            global_pos = get_global_position(pos, current_root_trans)
            global_vel = get_global_direction(new_velocity, current_root_trans)
            new_positions[i] = global_pos
            new_velocities[i] = global_vel
        self.actor.set_positions(new_positions)
        self.actor.set_velocities(new_velocities)


class Trajectory(object):
    
    def __init__(self, n_points, style_list, initial_pos=None, initial_dir=None):
        self.style_list = style_list
        self.n_styles = len(self.style_list)
        self.n_points = n_points
        self.points = [Point(self.n_styles, index=i) for i in range(self.n_points)]
        self.up_vector = np.array([0, 1, 0])
        self.root_index = 60
        self.bias_pos = 0.75
        self.bias_dir = 1.25
        self.trajectory_correction = 0.75
        self.point_density = 10
        for i in range(n_points):   #### all motions start from standing
            self.points[i].styles[0] = 1
        if initial_pos is not None:
            for point in self.points:
                point.set_position(initial_pos)
        if initial_dir is not None:
            for point in self.points:
                point.set_direction(initial_dir)
    
    def update_trajectory(self, output, unit_scale, current_root_trans):
        self.non_idle_amount = (1 - self.points[self.root_index].styles[0]) ** 0.25
        self.update_past_trajectory()

        ### update current root
        current_root_trans = self.points[self.root_index].get_transformation()
        new_root_position = self.points[self.root_index].get_position()
        # print("current root position: ", new_root_position)

        new_velocity = self.non_idle_amount * np.array([output[0] / unit_scale, 0.0, output[1] / unit_scale])
        # print("new velocity: ", new_velocity)
        current_root_position = get_global_position(new_velocity, current_root_trans)
        # print("new velocity in global: ", current_root_position)
        self.points[self.root_index].set_position(current_root_position)
        self.points[self.root_index].set_direction(Quaternion.fromAngleAxis(self.non_idle_amount * (-output[2]), self.up_vector) * self.points[self.root_index].get_direction())
        # print("the updated new root position: ", current_root_position)
        ### update future trajectory
        self.update_future_trajectory(output, new_velocity, unit_scale)
        return self.non_idle_amount

    def update_future_trajectory(self, output, new_velocity, unit_scale):
        next_root_trans = self.points[self.root_index].get_transformation()
        next_root_pos = self.points[self.root_index].get_position()
        next_root_dir = self.points[self.root_index].get_direction()
        global_velocity = get_global_direction(new_velocity, next_root_trans)
        for i in range(self.root_index+1, self.n_points):
            self.points[i].set_position(self.points[i].get_position() + global_velocity)
        w = self.root_index // self.point_density
        # w = 6
        for i in range(self.root_index+1, self.n_points):
            m = ((i - self.root_index) / self.point_density) % 1.0

            posX = (1 - m) * output[8 + w * 0 + i // self.point_density - w] + m * output[8 + w * 0 + i // self.point_density - w + 1]     
            posZ = (1 - m) * output[8 + w * 1 + i // self.point_density - w] + m * output[8 + w * 1 + i // self.point_density - w + 1]
            dirX = (1 - m) * output[8 + w * 2 + i // self.point_density - w] + m * output[8 + w * 2 + i // self.point_density - w + 1]
            dirZ = (1 - m) * output[8 + w * 3 + i // self.point_density - w] + m * output[8 + w * 3 + i // self.point_density - w + 1]

            self.points[i].set_position(vector_interpolation(self.points[i].get_position(), 
                                                             get_global_position(np.array([posX / unit_scale, 0.0, posZ / unit_scale]), next_root_trans),
                                                             self.trajectory_correction))
            dir = np.array([dirX, 0, dirZ]) 
            normalized_dir = dir / np.linalg.norm(dir)
            self.points[i].set_direction(vector_interpolation(self.points[i].get_direction(),
                                                              get_global_direction(normalized_dir, next_root_trans),
                                                              self.trajectory_correction))  
        ### smooth trajectory
        for i in range(self.root_index + 1, self.n_points):
            prev_point = self.get_previous_sample(i)
            next_point = self.get_next_sample(i)
            factor = (i % self.point_density) / self.point_density
            ### interpolate the intermediate points
            self.points[i].set_position((1 - factor) * prev_point.get_position() + factor * next_point.get_position())
            self.points[i].set_direction((1 - factor) * prev_point.get_direction() + factor * next_point.get_direction())
            self.points[i].set_left_sample((1 - factor) * prev_point.get_left_sample() + factor * next_point.get_left_sample())
            self.points[i].set_right_sample((1 - factor) * prev_point.get_right_sample() + factor * next_point.get_right_sample())
            self.points[i].set_slope((1 - factor) * prev_point.get_slope() + factor * next_point.get_slope())
            
               
    def get_previous_sample(self, index):
        return self.points[(index // self.point_density) * self.point_density]

    def get_next_sample(self, index):
        if index % self.point_density == 0:
            return self.points[(index // self.point_density) * self.point_density]
        else:
            # return self.points[(index // self.point_density + 1) * self.point_density - 1] 
            if (index // self.point_density + 1) * self.point_density == 120:
                return self.points[119]
            else:
                return self.points[(index // self.point_density + 1) * self.point_density]


    def predict_future_trajectory(self, target_direction):
        trajectory_positions_blend = np.zeros([self.n_points, 3])
        trajectory_positions_blend[self.root_index] = self.points[self.root_index].get_position()
        rescale = 1 / (self.n_points - (self.root_index + 1))
        for i in range(self.root_index+1, len(self.points)):
            scale_pos = 1 - pow(1 - (i - self.root_index) / self.root_index, self.bias_pos)
            scale_dir = 1 - pow(1 - (i - self.root_index) / self.root_index, self.bias_dir)
            # self.points[i].update_direction()
            trajectory_positions_blend[i] = trajectory_positions_blend[i-1] + vector_interpolation(self.points[i].get_position() - self.points[i-1].get_position(),
                                                                                                   self.pool_bias() * rescale * target_direction, 
                                                                                                   scale_pos)
            self.points[i].set_direction(vector_interpolation(self.points[i].get_direction(), target_direction, scale_dir))

            ### update gait information
            for j in range(self.n_styles):
                self.points[i].styles[j] = self.points[self.root_index].styles[j]

        ### set trajectory point position
        for i in range(self.root_index+1, len(self.points)):
            self.points[i].set_position(trajectory_positions_blend[i])
        
        ### smooth trajectory
        for i in range(self.root_index + 1, self.n_points):
            prev_point = self.get_previous_sample(i)
            next_point = self.get_next_sample(i)

            factor = (i % self.point_density) / self.point_density
            ### interpolate the intermediate points
            self.points[i].set_position((1 - factor) * prev_point.get_position() + factor * next_point.get_position())
            self.points[i].set_direction((1 - factor) * prev_point.get_direction() + factor * next_point.get_direction())
            self.points[i].set_left_sample((1 - factor) * prev_point.get_left_sample() + factor * next_point.get_left_sample())
            self.points[i].set_right_sample((1 - factor) * prev_point.get_right_sample() + factor * next_point.get_right_sample())
            self.points[i].set_slope((1 - factor) * prev_point.get_slope() + factor * next_point.get_slope())

    def update_past_trajectory(self):
        for i in range(self.root_index):
            self.points[i].set_position(self.points[i+1].get_position())
            self.points[i].set_direction(self.points[i+1].get_direction())
            self.points[i].set_left_sample(self.points[i+1].get_left_sample())
            self.points[i].set_right_sample(self.points[i+1].get_right_sample())
            self.points[i].set_slope(self.points[i+1].get_slope())
            for j in range(self.n_styles):
                self.points[i].styles[j] = self.points[i+1].styles[j]

    def pool_bias(self):
        styles = self.points[self.root_index].styles
        pool_bias = 0
        for i in range(len(styles)):
            pool_bias += styles[i] * self.style_list[i].bias
        return pool_bias



def vector_interpolation(v1, v2, t):
    return ((1 - t) * v1 + t * v2)



def create_style_list():
    styles = []
    styles.append(Style('idle', 1.0, 0.1))
    styles.append(Style('walk', 1.0, 0.1))
    styles.append(Style('run', 2.5, 0.1))
    styles.append(Style('crouch', 1.0, 0.1))
    styles.append(Style('jump', 1.0, 0.1))
    styles.append(Style('bump', 1.0, 0.1))
    return styles


def evaluate_traj(traj):
    """print the sampled 12 points by 10 samples
    
    Arguments:
        traj {Trajectory} -- [description]
    """
    points = []
    directions = []
    for i in range(12):
        points.append(traj.points[i*10].get_position())
        directions.append(traj.points[i*10].get_position())
    print("positions: ", points)
    print("##################")
    print("directions: ", directions)

def test_controller():
    debug = False
    c = Controller(debug=debug)
    frames = []
    for i in range(1000):
        frames.append(c.actor.get_positions())
        c.update()
        if debug is True:
            write_to_json_file(r'D:\tmp\tmp\path.json', c.log_data)  
    # print(np.asarray(frames).shape)
    frames = np.asarray(frames)
    motion = {'motion_data': frames.tolist(), 'has_skeleton': True, 'skeleton': c.skeleton_def}
    write_to_json_file(r'D:\tmp\tmp\python_controller.panim', motion)


def test_trajectory():
    default_pose_data, skeleton_def = get_default_pose()
    n_joints = len(skeleton_def)

    actor = Actor(n_joints)
    actor.set_positions(default_pose_data)
    initial_pos = actor.get_root_position()
    initial_dir = actor.get_forward_drection()
    styles = create_style_list()
    t = Trajectory(n_points=120, style_list=styles, initial_pos=initial_pos, initial_dir=initial_dir)
    for i in range(12):
        print(t.points[i*10].get_direction())


def test():
    # print(pow(2 ,3))
    # v1 = np.array([0, 0, 0])
    # v2 = np.array([0, 0.01, 0])
    # # v = np.interp(0.5, v1, v2)
    # print(np.all(v1 == 0))
    # print(np.all(v2 == 0))
    # new_array = np.array([2, 3, 5])
    # new_array = np.append(new_array, 0)
    # print(new_array)
    default_pose_data, skeleton_def = get_default_pose()
    print(default_pose_data)
    print("#####################")
    print(np.ravel(default_pose_data))

def rotation_mat_and_vector():
    from mosi_utils_anim.animation_data.quaternion import Quaternion
    angles = np.array([0, 50, 0])
    q = Quaternion.fromEulerAngles(np.deg2rad(angles))
    print(q)
    p = Point(6)
    p.set_rotation(q)
    dir = p.get_direction()
    print(dir)

    # forward = np.array([0, 0, 1])
    # ref_dir = q * forward
    # print(ref_dir)
    target_direction = np.array([50, 0, 200])
    p.set_direction(target_direction)
    print(target_direction/ np.linalg.norm(target_direction))
    print(p.get_direction())


def test_point():
    p = Point(n_styles = 6)
    target_direction = np.array([0.4, 0, 0.7])
    targe_position = np.array([11, 32, 90])
    target_direction = target_direction / np.linalg.norm(target_direction)
    # print(target_direction)

    target_direction = np.random.rand(3)


    p.set_position(targe_position)
    p.set_direction(target_direction)
    
    d = p.get_direction()
    print(d)

    target_direction = target_direction / np.linalg.norm(target_direction)
    print(target_direction)
    # print(p.get_position())

    # p1 = Point(n_styles=6)
    # p1.set_direction(np.array([0, 0, 1]))
    # p1.set_position(np.array([0, 0, 10]))
    # trans = p1.get_transformation()
    # p_new = get_local_transformation(p.get_position(), trans)
    # print(p_new)


def test_actor():
    default_pose_data, skeleton_def = get_default_pose()
    n_joints = len(skeleton_def)
    actor = Actor(n_joints)
    actor.set_positions(default_pose_data)
    print(actor.get_root_position())
    print(actor.get_forward_drection())

if __name__ == "__main__":
    # traj = Trajectory(120)
    # print(len(traj.points))
    # test()
    # rotation_mat_and_vector()
    # test_trajectory()
    # get_default_pose()
    test_controller()
    # test_point()
    # test_actor()