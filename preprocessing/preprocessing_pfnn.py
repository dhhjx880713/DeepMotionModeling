import numpy as np 
import glob
import sys
import scipy.interpolate as interpolate
import scipy.ndimage.filters as filters
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()) + r'/..')
from mosi_utils_anim.animation_data import BVHReader, BVHWriter, SkeletonBuilder
from mosi_utils_anim.animation_data.utils import convert_euler_frames_to_cartesian_frames, \
    convert_quat_frames_to_cartesian_frames, rotate_cartesian_frames_to_ref_dir, get_rotation_angles_for_vectors, \
    rotation_cartesian_frames, cartesian_pose_orientation, pose_orientation_euler, rotate_around_y_axis
from preprocessing.utils import covnert_pfnn_preprocessed_data_to_global_joint_positions
from mosi_utils_anim.animation_data.quaternion import Quaternion
from pfnn.Learning import RBF


rng = np.random.RandomState(1234)
to_meters = 5.6444
window = 60
njoints = 31
Edin_to_cmu_scale = 0.6034419925985102
""" Load Terrain Patches """

patches_database = np.load(r'D:\gits\PFNN\patches.npz')
patches = patches_database['X'].astype(np.float32)
patches_coord = patches_database['C'].astype(np.float32)




def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)


def generate_database(data_folder, scale=False, save_filename=None):
    """ Phases, Inputs, Outputs """

    P, X, Y = [], [], []
    bvhfiles = glob.glob(os.path.join(data_folder, '*.bvh'))

    for data in bvhfiles:
        if not 'rest' in data:
            print('Processing Clip %s' % data)

            """ Data Types """

            if   'LocomotionFlat12_000' in data: type = 'jumpy'
            elif 'NewCaptures01_000'    in data: type = 'flat'
            elif 'NewCaptures02_000'    in data: type = 'flat'
            elif 'NewCaptures03_000'    in data: type = 'jumpy'
            elif 'NewCaptures03_001'    in data: type = 'jumpy'
            elif 'NewCaptures03_002'    in data: type = 'jumpy'
            elif 'NewCaptures04_000'    in data: type = 'jumpy'
            elif 'WalkingUpSteps06_000' in data: type = 'beam'
            elif 'WalkingUpSteps09_000' in data: type = 'flat'
            elif 'WalkingUpSteps10_000' in data: type = 'flat'
            elif 'WalkingUpSteps11_000' in data: type = 'flat'
            elif 'Flat' in data: type = 'flat'
            else: type = 'rocky'

            """ Load Data """
            bvhreader = BVHReader(data)
            skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
            cartesian_frames = convert_euler_frames_to_cartesian_frames(skeleton, bvhreader.frames)
            global_positions = cartesian_frames * (to_meters * (1.0 / Edin_to_cmu_scale))

            global_positions = global_positions[::2]
            """ Load Phase / Gait """

            phase = np.loadtxt(data.replace('.bvh', '.phase'))[::2]
            gait = np.loadtxt(data.replace('.bvh', '.gait'))[::2]

            """ Merge Jog / Run and Crouch / Crawl """

            gait = np.concatenate([
                gait[:,0:1],
                gait[:,1:2],
                gait[:,2:3] + gait[:,3:4],
                gait[:,4:5] + gait[:,6:7],
                gait[:,5:6],
                gait[:,7:8]
            ], axis=-1)

            """ Preprocess Data """

            Pc, Xc, Yc = process_data(global_positions, phase, gait, type=type)

            with open(data.replace('.bvh', '_footsteps.txt'), 'r') as f:
                footsteps = f.readlines()

            """ For each Locomotion Cycle fit Terrains """

            for li in range(len(footsteps)-1):

                curr, next = footsteps[li+0].split(' '), footsteps[li+1].split(' ')

                """ Ignore Cycles marked with '*' or not in range """

                if len(curr) == 3 and curr[2].strip().endswith('*'): continue
                if len(next) == 3 and next[2].strip().endswith('*'): continue
                if len(next) <  2: continue
                if int(curr[0])//2-window < 0: continue
                if int(next[0])//2-window >= len(Xc): continue

                """ Fit Heightmaps """

                slc = slice(int(curr[0])//2-window, int(next[0])//2-window+1)
                H, Hmean = process_heights(global_positions[
                    int(curr[0])//2-window:
                    int(next[0])//2+window+1], type=type, scale=scale)
                for h, hmean in zip(H, Hmean):

                    Xh, Yh = Xc[slc].copy(), Yc[slc].copy()

                    """ Reduce Heights in Input/Output to Match"""

                    xo_s, xo_e = ((window*2)//10)*10+1, ((window*2)//10)*10+njoints*3+1
                    yo_s, yo_e = 8+(window//10)*4+1, 8+(window//10)*4+njoints*3+1
                    Xh[:,xo_s:xo_e:3] -= hmean[...,np.newaxis]
                    Yh[:,yo_s:yo_e:3] -= hmean[...,np.newaxis]
                    Xh = np.concatenate([Xh, h - hmean[...,np.newaxis]], axis=-1)

                    """ Append to Data """

                    P.append(np.hstack([0.0, Pc[slc][1:-1], 1.0]).astype(np.float32))
                    X.append(Xh.astype(np.float32))
                    Y.append(Yh.astype(np.float32))

    """ Clip Statistics """

    # print('Total Clips: %i' % len(X))
    # print('Shortest Clip: %i' % min(map(len,X)))
    # print('Longest Clip: %i' % max(map(len,X)))
    # print('Average Clip: %i' % np.mean(list(map(len,X))))

    """ Merge Clips """

    print('Merging Clips...')

    Xun = np.concatenate(X, axis=0)
    Yun = np.concatenate(Y, axis=0)
    Pun = np.concatenate(P, axis=0)

    # print(Xun.shape, Yun.shape, Pun.shape)

    print('Saving Database...')
    if save_filename is not None:
        np.savez_compressed(save_filename, Xun=Xun, Yun=Yun, Pun=Pun)
    else:
        np.savez_compressed('mk_cmu_database.npz', Xun=Xun, Yun=Yun, Pun=Pun)

""" Sampling Patch Heightmap """    

def patchfunc(P, Xp, hscale=3.937007874, vscale=3.0, scale=False):  ##todo: figure out hscale
    if scale:
        hscale = hscale / to_meters
        vscale = vscale / to_meters
    Xp = Xp / hscale + np.array([P.shape[1]//2, P.shape[2]//2])
    
    A = np.fmod(Xp, 1.0)
    X0 = np.clip(np.floor(Xp).astype(np.int), 0, np.array([P.shape[1]-1, P.shape[2]-1]))
    X1 = np.clip(np.ceil (Xp).astype(np.int), 0, np.array([P.shape[1]-1, P.shape[2]-1]))
    
    H0 = P[:,X0[:,0],X0[:,1]]
    H1 = P[:,X0[:,0],X1[:,1]]
    H2 = P[:,X1[:,0],X0[:,1]]
    H3 = P[:,X1[:,0],X1[:,1]]
    
    HL = (1-A[:,0]) * H0 + (A[:,0]) * H2
    HR = (1-A[:,0]) * H1 + (A[:,0]) * H3
    
    return (vscale * ((1-A[:,1]) * HL + (A[:,1]) * HR))[...,np.newaxis]


def get_rotation_to_ref_direction(dir_vecs, ref_dir):
    rotations = []
    for dir_vec in dir_vecs:
        rotations.append(Quaternion.between(dir_vec, ref_dir))
    return rotations


def process_heights(global_positions, nsamples=10, type='flat', scale=False):
    
    """ Extract Forward Direction """
    ref_dir = np.array([0, 0, 1])
    # sdr_l, sdr_r, hip_l, hip_r = 18, 25, 2, 7
    sdr_l, sdr_r, hip_l, hip_r = 10, 20, 2, 27
    across = (
        (global_positions[:,sdr_l] - global_positions[:,sdr_r]) + 
        (global_positions[:,hip_l] - global_positions[:,hip_r]))
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
    
    """ Smooth Forward Direction """
    
    direction_filterwidth = 20
    forward = filters.gaussian_filter1d(
        np.cross(across, np.array([[0,1,0]])), direction_filterwidth, axis=0, mode='nearest')    
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

    root_rotation = get_rotation_to_ref_direction(forward, ref_dir=ref_dir)

    """ Foot Contacts """
    
    # fid_l, fid_r = np.array([4,5]), np.array([9,10])
    fid_l, fid_r = np.array([4, 5]), np.array([29, 30])
    if not scale:
        # velfactor = np.array([0.02, 0.02])
        velfactor = np.array([0.05, 0.05])
    else:
        velfactor = np.array([0.02, 0.02]) / to_meters
    
    feet_l_x = (global_positions[1:,fid_l,0] - global_positions[:-1,fid_l,0])**2
    feet_l_y = (global_positions[1:,fid_l,1] - global_positions[:-1,fid_l,1])**2
    feet_l_z = (global_positions[1:,fid_l,2] - global_positions[:-1,fid_l,2])**2
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor))
    
    feet_r_x = (global_positions[1:,fid_r,0] - global_positions[:-1,fid_r,0])**2
    feet_r_y = (global_positions[1:,fid_r,1] - global_positions[:-1,fid_r,1])**2
    feet_r_z = (global_positions[1:,fid_r,2] - global_positions[:-1,fid_r,2])**2
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor))
    
    feet_l = np.concatenate([feet_l, feet_l[-1:]], axis=0)
    feet_r = np.concatenate([feet_r, feet_r[-1:]], axis=0)
    
    """ Toe and Heel Heights """
    if not scale:
        toe_h, heel_h = 4.0, 5.0
    else:
        toe_h, heel_h = 4.0 / to_meters, 5.0 / to_meters
    
    """ Foot Down Positions """
    
    feet_down = np.concatenate([
        global_positions[feet_l[:,0],fid_l[0]] - np.array([0, heel_h, 0]),
        global_positions[feet_l[:,1],fid_l[1]] - np.array([0,  toe_h, 0]),
        global_positions[feet_r[:,0],fid_r[0]] - np.array([0, heel_h, 0]),
        global_positions[feet_r[:,1],fid_r[1]] - np.array([0,  toe_h, 0])
    ], axis=0)
    
    """ Foot Up Positions """
    
    feet_up = np.concatenate([
        global_positions[~feet_l[:,0],fid_l[0]] - np.array([0, heel_h, 0]),
        global_positions[~feet_l[:,1],fid_l[1]] - np.array([0,  toe_h, 0]),
        global_positions[~feet_r[:,0],fid_r[0]] - np.array([0, heel_h, 0]),
        global_positions[~feet_r[:,1],fid_r[1]] - np.array([0,  toe_h, 0])
    ], axis=0)
    
    """ Down Locations """
    
    feet_down_xz = np.concatenate([feet_down[:,0:1], feet_down[:,2:3]], axis=-1)
    feet_down_xz_mean = feet_down_xz.mean(axis=0)
    feet_down_y = feet_down[:,1:2]
    feet_down_y_mean = feet_down_y.mean(axis=0)
    feet_down_y_std  = feet_down_y.std(axis=0)
        
    """ Up Locations """
        
    feet_up_xz = np.concatenate([feet_up[:,0:1], feet_up[:,2:3]], axis=-1)
    feet_up_y = feet_up[:,1:2]
    
    if len(feet_down_xz) == 0:
    
        """ No Contacts """
    
        terr_func = lambda Xp: np.zeros_like(Xp)[:,:1][np.newaxis].repeat(nsamples, axis=0)
        
    elif type == 'flat':
        
        """ Flat """
        
        terr_func = lambda Xp: np.zeros_like(Xp)[:,:1][np.newaxis].repeat(nsamples, axis=0) + feet_down_y_mean
    
    else:
        
        """ Terrain Heights """
        
        terr_down_y = patchfunc(patches, feet_down_xz - feet_down_xz_mean)
        terr_down_y_mean = terr_down_y.mean(axis=1)
        terr_down_y_std  = terr_down_y.std(axis=1)
        terr_up_y = patchfunc(patches, feet_up_xz - feet_down_xz_mean)
        
        """ Fitting Error """
        
        terr_down_err = 0.1 * ((
            (terr_down_y - terr_down_y_mean[:,np.newaxis]) -
            (feet_down_y - feet_down_y_mean)[np.newaxis])**2)[...,0].mean(axis=1)
        
        terr_up_err = (np.maximum(
            (terr_up_y - terr_down_y_mean[:,np.newaxis]) -
            (feet_up_y - feet_down_y_mean)[np.newaxis], 0.0)**2)[...,0].mean(axis=1)
        
        """ Jumping Error """
        
        if type == 'jumpy':
            terr_over_minh = 5.0
            if scale:
                terr_over_minh = terr_over_minh / to_meters
            terr_over_err = (np.maximum(
                ((feet_up_y - feet_down_y_mean)[np.newaxis] - terr_over_minh) -
                (terr_up_y - terr_down_y_mean[:,np.newaxis]), 0.0)**2)[...,0].mean(axis=1)
        else:
            terr_over_err = 0.0
        
        """ Fitting Terrain to Walking on Beam """
        
        if type == 'beam':

            beam_samples = 1
            beam_min_height = 40.0
            if scale:
                beam_min_height = beam_min_height / to_meters

            beam_c = global_positions[:,0]
            beam_c_xz = np.concatenate([beam_c[:,0:1], beam_c[:,2:3]], axis=-1)
            beam_c_y = patchfunc(patches, beam_c_xz - feet_down_xz_mean)

            if not scale:
                beam_o = (
                    beam_c.repeat(beam_samples, axis=0) + np.array([50, 0, 50]) *
                    rng.normal(size=(len(beam_c)*beam_samples, 3)))
            else:
                beam_o = (
                    beam_c.repeat(beam_samples, axis=0) + np.array([50, 0, 50]) / to_meters *
                    rng.normal(size=(len(beam_c)*beam_samples, 3)))
            beam_o_xz = np.concatenate([beam_o[:,0:1], beam_o[:,2:3]], axis=-1)
            beam_o_y = patchfunc(patches, beam_o_xz - feet_down_xz_mean)

            beam_pdist = np.sqrt(((beam_o[:,np.newaxis] - beam_c[np.newaxis,:])**2).sum(axis=-1))
            if not scale:
                beam_far = (beam_pdist > 15).all(axis=1)
            else:
                beam_far = (beam_pdist > 15 / to_meters).all(axis=1)
            terr_beam_err = (np.maximum(beam_o_y[:,beam_far] - 
                (beam_c_y.repeat(beam_samples, axis=1)[:,beam_far] - 
                 beam_min_height), 0.0)**2)[...,0].mean(axis=1)

        else:
            terr_beam_err = 0.0
        
        """ Final Fitting Error """
        
        terr = terr_down_err + terr_up_err + terr_over_err + terr_beam_err
        
        """ Best Fitting Terrains """
        
        terr_ids = np.argsort(terr)[:nsamples]
        terr_patches = patches[terr_ids]
        terr_basic_func = lambda Xp: (
            (patchfunc(terr_patches, Xp - feet_down_xz_mean) - 
            terr_down_y_mean[terr_ids][:,np.newaxis]) + feet_down_y_mean)
        
        """ Terrain Fit Editing """
        
        terr_residuals = feet_down_y - terr_basic_func(feet_down_xz)
        terr_fine_func = [RBF(smooth=0.1, function='linear') for _ in range(nsamples)]
        for i in range(nsamples): terr_fine_func[i].fit(feet_down_xz, terr_residuals[i])
        terr_func = lambda Xp: (terr_basic_func(Xp) + np.array([ff(Xp) for ff in terr_fine_func]))
        
        
    """ Get Trajectory Terrain Heights """
    
    root_offsets_c = global_positions[:,0]
    root_offsets_r = np.zeros(root_offsets_c.shape)
    root_offsets_l = np.zeros(root_offsets_c.shape)
    for i in range(len(root_rotation)):
        root_offsets_r[i] = -root_rotation[i] * np.array([+25, 0, 0]) + root_offsets_c[i]
        root_offsets_l[i] = -root_rotation[i] * np.array([-25, 0, 0]) + root_offsets_c[i]


    root_heights_c = terr_func(root_offsets_c[:,np.array([0,2])])[...,0]
    root_heights_r = terr_func(root_offsets_r[:,np.array([0,2])])[...,0]
    root_heights_l = terr_func(root_offsets_l[:,np.array([0,2])])[...,0]
    
    """ Find Trajectory Heights at each Window """
    
    root_terrains = []
    root_averages = []
    for i in range(window, len(global_positions)-window, 1): 
        root_terrains.append(
            np.concatenate([
                root_heights_r[:,i-window:i+window:10],
                root_heights_c[:,i-window:i+window:10],
                root_heights_l[:,i-window:i+window:10]], axis=1))
        root_averages.append(root_heights_c[:,i-window:i+window:10].mean(axis=1))
     
    root_terrains = np.swapaxes(np.array(root_terrains), 0, 1)
    root_averages = np.swapaxes(np.array(root_averages), 0, 1)
    
    return root_terrains, root_averages


def process_data(global_positions, phase, gait, type='flat', scale=False):
    """

    Arguments:
        global_positions {numpy.array3d} -- n_frames * n_joints * 3
        phase {[type]} -- [description]
        gait {[type]} -- [description]

    Keyword Arguments:
        type {str} -- [description] (default: {'flat'})
        scale {bool} -- [description] (default: {False})

    Returns:
        [type] -- [description]
    """
    """ Do FK """

    # print('global position shape: ', global_positions.shape)
    """ Extract Forward Direction """
    ref_dir = np.array([0, 0, 1])
    n_frames, n_joints, _ = global_positions.shape
    # sdr_l, sdr_r, hip_l, hip_r = 18, 25, 2, 7   ## todo: figure out these magic number (LeftArm, RightArm, LeftUpLeg, RightUpLeg)
    sdr_l, sdr_r, hip_l, hip_r = 10, 20, 2, 27
    across = (
        (global_positions[:,sdr_l] - global_positions[:,sdr_r]) + 
        (global_positions[:,hip_l] - global_positions[:,hip_r]))
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
    
    """ Smooth Forward Direction """
    
    direction_filterwidth = 20
    forward = filters.gaussian_filter1d(
        np.cross(across, np.array([[0,1,0]])), direction_filterwidth, axis=0, mode='nearest')    
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

    # print('forward direction shape: ', forward.shape)
    root_rotations = get_rotation_to_ref_direction(forward, ref_dir=ref_dir)

    """ Put on Floor """
    fid_l, fid_r = np.array([4, 5]), np.array([29, 30])
    foot_heights = np.minimum(global_positions[:, fid_l, 1], global_positions[:, fid_r, 1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)

    global_positions[:, :, 1] -= floor_height


    """ Local Space """
    
    local_positions = global_positions.copy()
    local_velocities = np.zeros(local_positions.shape)
    local_positions[:,:,0] = local_positions[:,:,0] - local_positions[:,0:1,0]
    local_positions[:,:,2] = local_positions[:,:,2] - local_positions[:,0:1,2]
    for i in range(n_frames - 1):
        for j in range(n_joints):
            local_positions[i, j] = root_rotations[i] * local_positions[i, j]
            local_velocities[i, j] = root_rotations[i] * (global_positions[i+1, j] - global_positions[i, j])
    
    """ Get Root Velocity """
    root_velocity = (global_positions[1:, 0:1] - global_positions[:-1, 0:1]).copy()

    """ Rotate Root Velocity """
    for i in range(n_frames - 1):

        root_velocity[i, 0] = root_rotations[i+1] * root_velocity[i, 0]
    """ Get Rotation Velocity """
    root_rvelocity = np.zeros(n_frames - 1)
    for i in range(n_frames - 1):
        q = root_rotations[i+1] * (-root_rotations[i])
        root_rvelocity[i] = Quaternion.get_angle_from_quaternion(q, ref_dir)
    """ Foot Contacts """
    
    # fid_l, fid_r = np.array([4,5]), np.array([9,10])   ## todo: figure out the corrent foot joints

    if not scale:
        velfactor = np.array([0.02, 0.02])
    else:
        velfactor = np.array([0.05, 0.05]) / to_meters
    
    feet_l_x = (global_positions[1:,fid_l,0] - global_positions[:-1,fid_l,0])**2
    feet_l_y = (global_positions[1:,fid_l,1] - global_positions[:-1,fid_l,1])**2
    feet_l_z = (global_positions[1:,fid_l,2] - global_positions[:-1,fid_l,2])**2
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor)).astype(np.float)
    
    feet_r_x = (global_positions[1:,fid_r,0] - global_positions[:-1,fid_r,0])**2
    feet_r_y = (global_positions[1:,fid_r,1] - global_positions[:-1,fid_r,1])**2
    feet_r_z = (global_positions[1:,fid_r,2] - global_positions[:-1,fid_r,2])**2
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)
    #
    # print('feet_l shape: ', feet_l.shape)
    # print('feet_r shape: ', feet_r.shape)
    """ Phase """
    
    dphase = phase[1:] - phase[:-1]
    dphase[dphase < 0] = (1.0-phase[:-1]+phase[1:])[dphase < 0]
    
    """ Adjust Crouching Gait Value """
    
    if type == 'flat':
        crouch_low, crouch_high = 80, 130
        head = 16
        gait[:-1,3] = 1 - np.clip((global_positions[:-1,head,1] - 80) / (130 - 80), 0, 1)
        gait[-1,3] = gait[-2,3]

    """ Start Windows """
    
    Pc, Xc, Yc = [], [], []
    # print('root rotation shape: ', root_rotation.shape)
    for i in range(window, n_frames-window-1, 1):
        rootposs = (global_positions[i-window:i+window:10,0] - global_positions[i:i+1,0]) ### 12*3
        rootdirs = forward[i-window:i+window:10] 
        for j in range(len(rootposs)):
            rootposs[j] = root_rotations[i] * rootposs[j]
            rootdirs[j] = root_rotations[i] * rootdirs[j]
  
        rootgait = gait[i-window:i+window:10]
        
        Pc.append(phase[i])

        Xc.append(np.hstack([
                rootposs[:,0].ravel(), rootposs[:,2].ravel(), # Trajectory Pos
                rootdirs[:,0].ravel(), rootdirs[:,2].ravel(), # Trajectory Dir
                rootgait[:,0].ravel(), rootgait[:,1].ravel(), # Trajectory Gait
                rootgait[:,2].ravel(), rootgait[:,3].ravel(), 
                rootgait[:,4].ravel(), rootgait[:,5].ravel(), 
                local_positions[i-1].ravel(),  # Joint Pos
                local_velocities[i-1].ravel(), # Joint Vel
                ]))
            
        rootposs_next = global_positions[i+1:i+window+1:10,0] - global_positions[i+1:i+2,0]
        rootdirs_next = forward[i+1:i+window+1:10]  
        for j in range(len(rootposs_next)):
            rootposs_next[j] = root_rotations[i+1] * rootposs_next[j]
            rootdirs_next[j] = root_rotations[i+1] * rootdirs_next[j]

        Yc.append(np.hstack([
                root_velocity[i,0,0].ravel(), # Root Vel X
                root_velocity[i,0,2].ravel(), # Root Vel Y
                root_rvelocity[i].ravel(),    # Root Rot Vel
                dphase[i],                    # Change in Phase
                np.concatenate([feet_l[i], feet_r[i]], axis=-1), # Contacts
                rootposs_next[:,0].ravel(), rootposs_next[:,2].ravel(), # Next Trajectory Pos
                rootdirs_next[:,0].ravel(), rootdirs_next[:,2].ravel(), # Next Trajectory Dir
                local_positions[i].ravel(),  # Joint Pos
                local_velocities[i].ravel(), # Joint Vel
                local_rotations[i].ravel()   # Joint Rot
                ]))
    Xc = np.asarray(Xc)
    Yc = np.asarray(Yc)

    return np.array(Pc), np.array(Xc), np.array(Yc)




if __name__ == "__main__":
    data_folder = r'E:\workspace\mocap_data\mk_cmu_retargeting_default_pose\pfnn_data'
    generate_database(data_folder, scale=False, save_filename='mk_cmu_database.npz')