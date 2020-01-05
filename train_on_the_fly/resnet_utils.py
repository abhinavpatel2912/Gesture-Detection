import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def getFrameCut(frame, bounding_box):
    '''

    :param frame: numpy array of image
    :param bounding_box: list of bounds of the box
    :return: bounding box of the numpy array
    '''

    left_most = int(bounding_box[0])
    right_most = int(bounding_box[1])
    top_most = int(bounding_box[2])
    bottom_most = int(bounding_box[3])

    return frame[top_most: bottom_most, left_most: right_most]


def get_frame_array(lst_item, video_store, image_height, image_width):
    '''

    :param lst_item: [video_name, bounding_box, label]
    :param video_store: path to jester videos
    :param image_height: required height of each frame
    :param image_width: required width of each frame
    :return: frame array with given bounding box and image dimensions
    '''

    mod_video_name = lst_item[0]
    video_name = mod_video_name.lstrip("0")
    bounding_box = lst_item[1]
    frame_list = os.listdir(os.path.join(video_store, video_name))

    cut_frame_list = []
    for ind in range(len(frame_list)):
        frame_name = str(ind + 1).zfill(5) + ".jpg"
        frame_path = os.path.join(video_store + video_name, frame_name)
        frame = plt.imread(frame_path)
        frame_cut = getFrameCut(frame, bounding_box)

        res = cv2.resize(frame_cut, dsize=(image_height, image_width), interpolation=cv2.INTER_CUBIC)
        cut_frame_list.append(res)

    cut_frame_array = np.array(cut_frame_list)

    return cut_frame_array


