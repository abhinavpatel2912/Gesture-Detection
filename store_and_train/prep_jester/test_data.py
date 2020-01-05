import json
import os
import pickle
import sys
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

sys.path.insert(1, "/home/axp798/axp798gallinahome/Gesture-Detection/")
from image_acquisition import getBoundingBox

video_store = "/home/axp798/axp798gallinahome/jester_datase/20bn-jester-v1/"
json_store = "/home/axp798/axp798gallinahome/JD_JSON"
test_lables_path = '/home/axp798/axp798gallinahome/data/jester/jester-v1-test.csv'
valid_lables_path = '/home/axp798/axp798gallinahome/data/jester/jester-v1-validation.csv'
lables_path = '/home/axp798/axp798gallinahome/data/jester/jester-v1-labels.csv'


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


def returnBoundingBox(json_path, img_height, img_width):
    '''

    :param json_path: path of folder containing json files of the current video
    :param img_height: height of image frame
    :param img_width: width of image frame
    :return: left, right top and bottom bounds
    '''

    global_val = [100000, 0, 100000, img_height]
    return getBoundingBox(json_path, global_val, img_height, img_width)


def getVideoDimension(video_path):
    '''

    :param video_name: name of the video for which dimension is asked
    :return: shape of the video frame
    '''

    frame_list = os.listdir(video_path)
    sample_image_path = os.path.join(video_path, frame_list[0])
    curr_image = plt.imread(sample_image_path)
    return curr_image.shape


def video_labels(valid_lables_path, lables_path):
    '''

    :param train_lables_path: path of file containing video names with their labels
    :param labels_path: path of file containing index of each gesture
    :return: sorted list of video names along with their label (from 0 to 26)
    '''

    vpd = pd.read_csv(valid_lables_path, sep=';', header=None)
    lab = pd.read_csv(lables_path, sep=';', header=None)

    arr_val = np.array(vpd)
    lst_val = list(arr_val)
    lst_val = [list(arr) for arr in lst_val]

    lst_val = sorted(lst_val, key=lambda x: x[0])

    lst_labels = np.array(lab)
    lst_labels = list(lst_labels)
    lst_labels = [list(arr)[0] for arr in lst_labels]

    ref_dict = {label: ind for ind, label in enumerate(lst_labels)}

    for ind, row in enumerate(lst_val):
        lst_val[ind][1] = ref_dict[row[1]]
        ind += 1

    return lst_val


def save_frames_and_bounding_boxes(lst_test):
    num_videos = 0
    #     lst_val = video_labels(valid_lables_path, lables_path)

    bounding_box_list = []
    incorrect_box = []

    log_file_path = "/home/axp798/axp798gallinahome/Gesture-Detection/assets/jes_dat/log_test.txt"
    save_video_path = '/home/axp798/axp798gallinahome/data/jester/test_64/'
    save_bounding_box_path = "/home/axp798/axp798gallinahome/data/jester/boxes/"

    file1 = open(log_file_path, "w")
    file1.write("Logs: \n")
    file1.close()

    for videoName in lst_test[num_videos:]:
        num_videos += 1
        video_name = videoName[0]

        video_name = str(video_name)

        modified_video_name = video_name.zfill(6)  # to get json files
        json_path = os.path.join(json_store, modified_video_name)
        video_path = os.path.join(video_store, video_name)

        print("{}-{}-{} started!".format(num_videos, video_name, modified_video_name))

        img_shape = getVideoDimension(video_path)
        img_height = img_shape[0]
        img_width = img_shape[1]

        bounding_box = returnBoundingBox(json_path, img_height, img_width)

        frame_list = os.listdir(os.path.join(video_store, video_name))

        cut_frame_list = []
        check = 0
        for ind in range(len(frame_list)):
            frame_name = str(ind + 1).zfill(5) + ".jpg"
            frame_path = os.path.join(video_store + video_name, frame_name)
            frame = plt.imread(frame_path)
            frame_cut = getFrameCut(frame, bounding_box)

            if frame_cut.size == 0:
                incorrect_box.append(modified_video_name)
                check = 1
                break

            res = cv2.resize(frame_cut, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
            cut_frame_list.append(res)

        if check:
            log1 = "{}-{}-{} ; incorrect boxes: {}-{}! \n".format(num_videos, video_name, modified_video_name,
                                                                  len(incorrect_box), incorrect_box)
            file1 = open(log_file_path, "a")
            file1.write(log1)
            file1.close()
            print(log1)
            continue

        cut_frame_array = np.array(cut_frame_list)
        video_shape = cut_frame_array.shape

        bounding_box_list.append(["{}".format(modified_video_name), bounding_box])

        pkl_vid_name = "{}_{}.pkl".format(num_videos, modified_video_name)

        with open(save_video_path + pkl_vid_name, 'wb') as f:
            pickle.dump(cut_frame_array, f)

        with open(save_bounding_box_path + "test_boxes.pkl", 'wb') as f:
            pickle.dump(bounding_box_list, f)

        log = "{}-{}-{} done! ----- bounding box: {}, video shape: {} \n".format(num_videos, video_name,
                                                                                 modified_video_name,
                                                                                 bounding_box, video_shape)
        file1 = open(log_file_path, "a")
        file1.write(log)
        file1.close()
        print(log)


if __name__ == '__main__':
    lst_val = video_labels(valid_lables_path, lables_path)
    test_pd = pd.read_csv(test_lables_path, header=None)
    lst_test = list(np.array(test_pd))
    save_frames_and_bounding_boxes(lst_test)
