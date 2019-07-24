import numpy as np
import pickle
import os

with open('bounding_box_train.pkl', 'rb') as f:
    data = pickle.load(f)


# curr_dict.update({"video_frames_dir": curr_video_frames_path})
# curr_dict.update({"range_of_frames": [start, end]})
# curr_dict.update({"label": label})
# curr_dict.update({"bounding_box": bounding_box})

def getFrameCut(frame, bounding_box):
    bottom_left = bounding_box[0]
    top_left = bounding_box[1]
    top_right = bounding_box[2]
    bottom_right = bounding_box[3]

    frame_cut = frame[top_left : bottom_left, bottom_left : bottom_right]

    return frame_cut

def toDoPerEpoch(data, win_size, num_clips):
    '''

    :param data: list of dictionaries
    :param win_size: window size
    :param num_clips: number of clips per video sample
    :return:
    '''

    mini_batch_x = []
    mini_match_y = []
    batch_filled = 0
    batch_size = 64

    for dict in data:
        video_frames_dir = dict['video_frames_dir']
        bounding_box = dict['bounding_box']
        range_of_frames = dict['range_of_frames']
        label = dict['label']

        cut_frames_list = []
        for frame in os.listdir(video_frames_dir)[range_of_frames[0]: range_of_frames[1]]:
            frame_cut = getFrameCut(frame, bounding_box)
            cut_frames_list.append(frame_cut)

        num_frames = len(cut_frames_list)
        num_clips_per_video = num_clips
        start_frame = 0
        window_size = win_size
        end_frame = window_size + start_frame
        num_clip_index = 0
        while num_clip_index < num_clips_per_video:
            start_frame = np.randint(0, num_frames - window_size)
            end_frame = window_size

            if batch_filled == batch_size:
                train_batch(mini_batch_x, mini_match_y)

                mini_batch_x = []
                mini_match_y = []
                batch_filled = 0

            mini_batch_x.append((cut_frames_list[start_frame : end_frame]))
            mini_match_y.append(oneHotEncoded(label))



    mini_batch_x = np.array(mini_batch_x)
    mini_batch_y = np.array(mini_match_y)

    return mini_batch_x, mini_match_y
