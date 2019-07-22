import image_acquisition2 as img_ac
import os
import pickle

videos_path = "/home/axp798/axp798gallinahome/ConGD/ConGD_phase_1/train"
labels_path = "/home/axp798/axp798gallinahome/ConGD/ConGD_labels/train.txt"
json_store_path = "/home/axp798/axp798gallinahome/store/train/json"


global_min_x = 100000
global_max_x = 0
global_min_y = 0
global_max_y = 240

global_val = [global_min_x, global_max_x, global_min_y, global_max_y]


data = []

num_videos = 0
lines = []

with open(labels_path, 'r') as f:
    lines = f.readlines()


for curr_line in lines:
    print("curr_line: {}".format(curr_line))
    line_ele = curr_line.split(' ')
    dir_info = line_ele[0]
    print("dir_info: {}".format(dir_info))

    par_name, vid_name = dir_info.split('/')[0], dir_info.split('/')[1]
    label_info = line_ele[1:]
    print("label_info: {}".format(label_info))
    num_videos += len(label_info)
    print("num_videos: {}".format(num_videos))

    print("************** ################# ********************")

    for curr_label in label_info:
        curr_label_info = curr_label.split(':')
        range = curr_label_info[0]
        label = curr_label_info[1]

        start_frame = range.split(',')[0]
        end_frame = range.split(',')[1]

        start, end = int(start_frame), int(end_frame)

        # Get bounding box
        dir_path = os.path.join(json_store_path, par_name)
        vid_path = os.path.join(dir_path, vid_name)
        if not os.path.exists(dir_path) or os.path.exists(vid_path):
            continue

        json_path = vid_path
        bounding_box = img_ac.getBoundingBox(json_path, global_val, start, end)

        curr_dict = {}
        curr_video_path = os.path.join(videos_path, dir_info)

        curr_dict.update({"video_dir": curr_video_path})
        curr_dict.update({"range_of_frames": [start, end]})
        curr_dict.update({"label": label})
        curr_dict.update({"bounding_box": bounding_box})

        data.append(curr_dict)

with open('bounding_box_train.pkl', 'wb') as f:
    pickle.dump(data, f)

#
#
