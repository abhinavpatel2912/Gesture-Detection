import sys
sys.path.insert(1, "/home/axp798/axp798gallinahome/Gesture-Detection/")
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time

video_store = "/mnt/ConGD_phase_1/valid/"
labels_path = '/mnt/ConGD_labels/valid.txt'
store_frames_path = "/shared/valid/frames/"

with open(labels_path, 'r') as f:
    lines = f.readlines()

num_sample = 0
prev_mean = 0
prev_squared_mean = 0
prev_samples = -1
global_squared_mean = 0
global_mean = 0

for curr_line in lines:

    net_start_time = time.time()

    line_ele = curr_line.split(' ')
    dir_info = line_ele[0]
    par_name, vid_name = dir_info.split('/')[0], dir_info.split('/')[1]
    label_info = line_ele[1:]

    for j, curr_label in enumerate(label_info):

        video_start_time = time.time()
        curr_label_info = curr_label.split(':')

        rang = curr_label_info[0]
        label = curr_label_info[1]

        start_frame = rang.split(',')[0]
        end_frame = rang.split(',')[1]

        start_frame, end_frame = int(start_frame), int(end_frame)

        img_height = 240
        img_width = 320
        num_sample += 1
        print("{} video started.".format(num_sample))

        frame_list = []
        for ind in range(start_frame - 1, end_frame):
            frame_name = "frame{}.jpg".format(ind + 1)
            video_frame_dir = os.path.join(store_frames_path, "{}/{}".format(par_name, vid_name))
            frame_path = os.path.join(video_frame_dir, frame_name)
            frame = plt.imread(frame_path)
            frame_list.append(frame)

        frame_array = np.array(frame_list)
        curr_mean = np.mean(frame_array, axis=(0, 1, 2))
        curr_samples = len(frame_list) * img_height * img_width
        if prev_samples != -1:
            multiply_factor = (curr_samples / (curr_samples + prev_samples), prev_samples / (curr_samples + prev_samples))
            global_mean = curr_mean * multiply_factor[0] + prev_mean * multiply_factor[1]
            prev_samples += curr_samples
        else:
            global_mean = curr_mean
            prev_samples = curr_samples

        prev_mean = global_mean
        print(prev_mean)

global_channel_mean = global_mean

num_sample = 0
for curr_line in lines:

    net_start_time = time.time()

    line_ele = curr_line.split(' ')
    dir_info = line_ele[0]
    par_name, vid_name = dir_info.split('/')[0], dir_info.split('/')[1]
    label_info = line_ele[1:]

    for j, curr_label in enumerate(label_info):
        # TODO:
        # get bounding box
        # convert to frames
        # get cut frames

        video_start_time = time.time()
        curr_label_info = curr_label.split(':')

        rang = curr_label_info[0]
        label = curr_label_info[1]

        start_frame = rang.split(',')[0]
        end_frame = rang.split(',')[1]

        start_frame, end_frame = int(start_frame), int(end_frame)

        img_height = 240
        img_width = 320
        num_sample += 1
        print("{} video started.".format(num_sample))

        frame_list = []
        for ind in range(start_frame - 1, end_frame):
            frame_name = "frame{}.jpg".format(ind + 1)
            video_frame_dir = os.path.join(store_frames_path, "{}/{}".format(par_name, vid_name))
            frame_path = os.path.join(video_frame_dir, frame_name)
            frame = plt.imread(frame_path)
            frame_list.append(frame - global_channel_mean)

        frame_array = np.array(frame_list)
        curr_squared_mean = np.mean(np.square(frame_array), axis=(0, 1, 2))
        curr_samples = len(frame_list) * img_height * img_width
        if prev_samples != -1:
            multiply_factor = (curr_samples / (curr_samples + prev_samples), prev_samples / (curr_samples + prev_samples))
            global_squared_mean = curr_squared_mean * multiply_factor[0] + prev_squared_mean * multiply_factor[1]
            prev_samples += curr_samples
        else:
            global_squared_mean = curr_squared_mean
            prev_samples = curr_samples

        prev_squared_mean = global_squared_mean
        print(prev_squared_mean)

global_channel_std = np.sqrt(prev_squared_mean)
print(global_channel_mean)
print(global_channel_std)

