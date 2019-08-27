import sys
sys.path.insert(1, "/home/axp798/axp798gallinahome/Gesture-Detection/")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import pickle
import image_acquisition2 as img_ac
import time

video_store = "/home/axp798/axp798gallinahome/ConGD/ConGD_phase_1/train/"
json_store = "/home/axp798/axp798gallinahome/store/train/json/"
labels_path = '/home/axp798/axp798gallinahome/ConGD/ConGD_labels/train.txt'
store_frames_path = "/home/axp798/axp798gallinahome/store/train/frames/"

def store_frames(dir_name, vid_name):
    vid_path = os.path.join(video_store, "{}/{}.M.avi".format(dir_name, vid_name))

    video_par_dir = os.path.join(store_frames_path, dir_name)
    video_frame_dir = os.path.join(store_frames_path, "{}/{}".format(dir_name, vid_name))

    if not os.path.exists(video_par_dir):
        os.mkdir(video_par_dir)

    if not os.path.exists(video_frame_dir):
        os.mkdir(video_frame_dir)

    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    count = 1
    while success:
        cv2.imwrite(os.path.join(video_frame_dir, "frame{}.jpg".format(count)), image)     # save frame as JPEG file
        success, image = vidcap.read()
        count += 1


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


def saveFramesWithLabels(video_store, json_store, labels_path):
    '''

    :param video_store: path of folder containing all the videos
    :param json_store: path of the folder containing json files of all the videos
    :param train_lables_path: path of file containing video names with their labels
    :param labels_path: path of file containing index of each gesture
    :return: stores a dictionary with key as a list of numpy array of video frame and value as video label
    '''

    start = 3000
    end = 4500
    num_videos = start

    start_time = time.time()

    save_video_path = '/home/axp798/axp798gallinahome/data/chalearn/train_64/'
    save_bounding_box_path = "/home/axp798/axp798gallinahome/data/chalearn/train_boxes/"
    log_file_path = "/home/axp798/axp798gallinahome/Gesture-Detection/assets/chalearn_dataset/log3.txt"

    file1 = open(log_file_path, "w")
    file1.write("Logs: \n")
    file1.close()

    bounding_box_list = []
    incorrect_box = []

    with open(labels_path, 'r') as f:
        lines = f.readlines()

    for curr_line in lines[start : end]:
        
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

            num_videos += 1
            video_start_time = time.time()
            curr_label_info = curr_label.split(':')

            rang = curr_label_info[0]
            label = curr_label_info[1]

            start_frame = rang.split(',')[0]
            end_frame = rang.split(',')[1]

            start_frame, end_frame = int(start_frame), int(end_frame)

            # Get bounding box
            dir_path = os.path.join(json_store, par_name)
            vid_path = os.path.join(dir_path, vid_name)

            img_height = 240
            img_width = 320

            json_path = vid_path
            global_val = [100000, 0, 100000, img_height] 
            bounding_box = img_ac.getBoundingBox(json_path, vid_name, global_val, img_height, img_width, start_frame, end_frame)

            # store frames
            if j == 0:
                store_frames(par_name, vid_name)

            # get cut frames
            cut_frame_list = []
            check = 0
            for ind in range(start_frame - 1, end_frame):
                frame_name = "frame{}.jpg".format(ind + 1)
                video_frame_dir = os.path.join(store_frames_path, "{}/{}".format(par_name, vid_name))
                frame_path = os.path.join(video_frame_dir, frame_name)
                frame = plt.imread(frame_path)
                frame_cut = getFrameCut(frame, bounding_box)

                if frame_cut.size == 0:
                    incorrect_box.append([dir_info, start_frame, end_frame])
                    check = 1
                    break

                res = cv2.resize(frame_cut, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
                cut_frame_list.append(res)

            if check:
                log1 = "{}-{} ; incorrect boxes: {}-{}! \n".format(num_videos, dir_info,
                                                                len(incorrect_box), incorrect_box)
                file1 = open(log_file_path, "a")
                file1.write(log1)
                file1.close()
                print(log1)
                continue

            cut_frame_array = np.array(cut_frame_list)
            video_shape = cut_frame_array.shape

            bounding_box_list.append(["{}_{}-{}".format(dir_info, start_frame, end_frame), bounding_box, label])

            pkl_vid_name = "{}-{}-{}_{}-{}.pkl".format(num_videos, par_name, vid_name, start_frame, end_frame)

            with open(save_video_path + pkl_vid_name, 'wb') as f:
                pickle.dump([cut_frame_array, label], f)

            with open(save_bounding_box_path + "{}_{}.pkl".format(start + 1, end), 'wb') as f:
                pickle.dump(bounding_box_list, f)

            video_end_time = time.time()

            log = "{}-{}-{} done in {}s ----- bounding box: {}, video shape: {} \n".format(num_videos, dir_info,
                                                                                        "{}_{}".format(start_frame, end_frame),
                                                                                        video_end_time - video_start_time,
                                                                                        bounding_box, video_shape)
            file1 = open(log_file_path, "a")
            file1.write(log)
            file1.close()
            print (log)

            video_end_time = time.time()

        net_end_time = time.time()

        print ("########### Line {} done in {}s ###############".format(dir_info, net_end_time - net_start_time))
    
    end_time = time.time()
    print ("Done in {}s".format(end_time - start_time))


if __name__ == '__main__':
    saveFramesWithLabels(video_store, json_store, labels_path)
