import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, "/home/axp798/axp798gallinahome/Gesture-Detection/")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import pickle
from image_acquisition import getBoundingBox
import time

video_store = "/home/axp798/axp798gallinahome/jester_datase/20bn-jester-v1/"
json_store = "/home/axp798/axp798gallinahome/JD_JSON"
train_lables_path = '/home/axp798/axp798gallinahome/Gesture-Detection/jester-v1-train.csv'
labels_path = '/home/axp798/axp798gallinahome/Gesture-Detection/jester-v1-labels.csv'



def video_labels(train_lables_path, labels_path):
    '''

    :param train_lables_path: path of file containing video names with their labels
    :param labels_path: path of file containing index of each gesture
    :return: sorted list of video names along with their label (from 0 to 26)
    '''

    tpd = pd.read_csv(train_lables_path, sep=';', header=None)
    lab = pd.read_csv(labels_path, sep=';', header=None)

    arr_train = np.array(tpd)
    lst_train = list(arr_train)
    lst_train = [list(arr) for arr in lst_train]

    lst_train = sorted(lst_train, key=lambda x: x[0])

    lst_labels = np.array(lab)
    lst_labels = list(lst_labels)
    lst_labels = [list(arr)[0] for arr in lst_labels]

    ref_dict = {label: ind for ind, label in enumerate(lst_labels)}

    for ind, row in enumerate(lst_train):
        lst_train[ind][1] = ref_dict[row[1]]
        ind += 1

    return lst_train


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


def saveFramesWithLabels(video_store, json_store, train_lables_path, labels_path):
    '''

    :param video_store: path of folder containing all the videos
    :param json_store: path of the folder containing json files of all the videos
    :param train_lables_path: path of file containing video names with their labels
    :param labels_path: path of file containing index of each gesture
    :return: stores a dictionary with key as a list of numpy array of video frame and value as video label
    '''

    list_train_videos_name = video_labels(train_lables_path, labels_path)
    
    start = 90000
    end = 100000
    num_videos = start

    net_start_time = time.time()
    log_file_path = "/home/axp798/axp798gallinahome/Gesture-Detection/assets/jes_dat/log10.txt"

    file1 = open(log_file_path, "w")
    file1.write("Logs: \n")
    file1.close()

    bounding_box_list = []
    incorrect_box = []

    save_video_path = '/home/axp798/axp798gallinahome/data/jester/train_64/'
    save_bounding_box_path = "/home/axp798/axp798gallinahome/data/jester/boxes/"

    for videoName_and_label in list_train_videos_name[start : end]:
        num_videos += 1
        video_start_time = time.time()
        video_name = videoName_and_label[0]
        label = videoName_and_label[1]

        video_name = str(video_name)

        modified_video_name = video_name.zfill(6)  # to get json files
        json_path = os.path.join(json_store, modified_video_name)
        video_path = os.path.join(video_store, video_name)

        print ("{}-{}-{} started!".format(num_videos, video_name, modified_video_name))

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
            log1 = "{}-{}-{} ; incorrect boxes: {}-{}!".format(num_videos, video_name, modified_video_name, len(incorrect_box), incorrect_box)
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
            pickle.dump([cut_frame_array, label], f)

        with open(save_bounding_box_path + "{}_{}.pkl".format(start + 1, end), 'wb') as f:
            pickle.dump(bounding_box_list, f)

        video_end_time = time.time()

        log = "{}-{}-{} done in {}! ----- bounding box: {}, video shape: {} \n".format(num_videos, video_name,
                                                                                     modified_video_name,
                                                                                     video_end_time - video_start_time,
                                                                                     bounding_box, video_shape)
        file1 = open(log_file_path, "a")
        file1.write(log)
        file1.close()
        print (log)
        
    net_end_time = time.time()

    print ("Done in {}s".format(net_end_time - net_start_time))


if __name__ == '__main__':

    saveFramesWithLabels(video_store, json_store, train_lables_path, labels_path)
