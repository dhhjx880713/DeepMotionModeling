import numpy as np 


def estimate_floor_height(foot_heights):
    """estimate offset from foot to floor
    
    Arguments:
        foot_heights {numpy.array} -- the heights of contact point in each frame
    """
    median_value = np.median(foot_heights)
    min_value = np.min(foot_heights)
    abs_diff = np.abs(median_value - min_value)
    return (2 * abs_diff) / (np.exp(abs_diff) + np.exp(-abs_diff)) + min_value


def sliding_window(data, window_size):
    """Slide Over Windows
    
    """
    windows = []
    window_step = int(window_size / 2.0)
    if len(data) % window_step == 0:
        n_clips = (len(data) - len(data) % window_step) // window_step
    else:
        n_clips = (len(data) - len(data) % window_step) // window_step + 1

    for j in range(0, n_clips):
        """ If slice too small pad out by repeating start and end poses """
        slice = data[j * window_step: j * window_step + window_size]
        if len(slice) < window_size:
            left = slice[:1].repeat((window_size - len(slice)) // 2 + (window_size - len(slice)) % 2, axis=0)
            right = slice[-1:].repeat((window_size - len(slice)) // 2, axis=0)
            slice = np.concatenate([left, slice, right], axis=0)
        if len(slice) != window_size: raise Exception()

        windows.append(slice)
    return windows


def combine_motion_clips(clips, motion_len, window_step):
    """combine motion clips to reconstruct original motion
    
    Arguments:
        clips {numpy.array} -- n_clips * n_frame * n_dims
        motion_len {int} -- number of original motion frames
        overlapping_len {int}
    
    """

    clips = np.asarray(clips)
    n_clips, window_size, n_dims = clips.shape

    ## case 1: motion length is smaller than window_step
    if motion_len <= window_step:
        assert n_clips == 1
        left_index = (window_size - motion_len) // 2 + (window_size - motion_len) % 2
        right_index = window_size - (window_size - motion_len) // 2
        return clips[0][left_index: right_index]

    ## case 2: motion length is larger than window_step and smaller than window
    if motion_len > window_step and motion_len <= window_size:
        assert n_clips == 2
        left_index = (window_size - motion_len) // 2 + (window_size - motion_len) % 2
        right_index = window_size - (window_size - motion_len) // 2
        return clips[0][left_index: right_index]

    residue_frames = motion_len % window_step
    print('residue_frames: ', residue_frames)
    ## case 3: residue frames is smaller than window step
    if residue_frames <= window_step:
        residue_frames += window_step
        combined_frames = np.concatenate(clips[0:n_clips-2, :window_step], axis=0)
        left_index = (window_size - residue_frames) // 2 + (window_size - residue_frames) % 2
        right_index = window_size - (window_size - residue_frames) // 2
        combined_frames = np.concatenate((combined_frames, clips[-2, left_index:right_index]), axis=0)
        return combined_frames