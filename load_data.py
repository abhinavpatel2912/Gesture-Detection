import os
import pickle
import cv2

# curr_dict.update({"video_frames_dir": curr_video_frames_path})
# curr_dict.update({"range_of_frames": [start, end]})
# curr_dict.update({"label": label})
# curr_dict.update({"bounding_box": bounding_box})

with open('bounding_box_train.pkl', 'rb') as f:
    data = pickle.load(f)

for curr_video_dict in data:

    curr_video_frames_path = curr_video_dict["video_frames_dir"]
    start_and_end = curr_video_dict["range_of_frames"]
    label = curr_video_dict["label"]
    bounding_box = curr_video_dict["bounding_box"]

    

