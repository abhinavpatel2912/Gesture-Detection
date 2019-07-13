# Importing Libraries

import sys
import json
import os
import numpy as np

# Default video frame dimensions
<<<<<<< HEAD

img_height = 240
img_width = 320

offset = 25  # amount of shift in extreme points (in pixels)

def getExtrmCoords(points_list):

    '''

    :param points_list: list of x and y coordinates of joints along with their confidence scores
    :return: extreme coordinates in x and y directions

=======

img_height = 240
img_width = 320

offset = 25  # amount of shift in extreme points (in pixels)

def getExtrmCoords(points_list):

    '''

    :param points_list: list of x and y coordinates of joints along with their confidence scores
    :return: extreme coordinates in x and y directions
    
>>>>>>> 87803fcc53a05e4692baba2fd44429cc36b7b2a2
    '''

    x_coord = []
    y_coord = []
    print(points_list)
    for ptr in range(0, len(points_list), 3):

        # print("ptr is {} and len(points_list) is {}".format(ptr, len(points_list)))
        curr_x = points_list[ptr]
        curr_y = points_list[ptr + 1]
        
        if curr_x >= 0 and curr_x <= img_width:
            x_coord.append(curr_x)
        if curr_y >= 0 and curr_y <= img_height:
            y_coord.append(curr_y)

    print(x_coord)
    print(y_coord)

    if len(x_coord) > 0 and len(y_coord) > 0:
        points_min_x = min(i for i in x_coord if i > 0)
        points_max_x = max(i for i in x_coord if i > 0)

        points_min_y = min(i for i in y_coord if i > 0)

        return points_min_x, points_max_x, points_min_y



def getBoundingBox(json_path, global_val):
<<<<<<< HEAD

    '''

=======
    
    '''
    
>>>>>>> 87803fcc53a05e4692baba2fd44429cc36b7b2a2
    :param json_path: path of directory containing all the json files of the respective video
    :param global_val: list of variables storing extreme coordinates
    :return: bounding box around the person in the given video
    '''

    global_min_x, global_max_x, global_min_y, global_max_y = global_val

    file_list = os.listdir(json_path)

    for json_file in file_list:

        arr = json_file.split('.')
        if arr[-1] != "json":
            continue

        print("{} json started!".format(json_file))

        json_file_path = os.path.join(json_path, json_file)

        with open(json_file_path, 'r') as myfile:
            data = myfile.read()

        obj = json.loads(data)

        main_dict = obj['people'][0]


        print("main_dict.keys() is {}".format(main_dict.keys()))
        pose = main_dict['pose_keypoints']
        hand_left = main_dict['hand_left_keypoints']
        hand_right = main_dict['hand_right_keypoints']

        # face = main_dict['face_keypoints']

        pose_min_x, pose_max_x, pose_min_y = getExtrmCoords(pose)
        hand_left_min_x, hand_left_max_x, hand_left_min_y = getExtrmCoords(hand_left)
        hand_right_min_x, hand_right_max_x, hand_right_min_y = getExtrmCoords(hand_right)
        # face_min_x, face_max_x, face_min_y = getExtrmCoords(face)


        global_min_x = min(global_min_x, pose_min_x, hand_left_min_x, hand_right_min_x)
        global_max_x = max(global_max_x, pose_max_x, hand_left_max_x, hand_right_max_x)

        # global_min_y = min(global_min_y, pose_min_y, hand_left_min_y, hand_right_min_y, face_min_y)

    print("global_min_x is {} and global_max_x is {}".format(global_min_x, global_max_x))

    if global_min_x - offset <= 0:
        global_min_x = 0
    else:
        global_min_x -= offset

    if global_max_x + offset > img_width:
        global_max_x = img_width
    else:
        global_max_x += offset

    # if global_min_y - offset <= 0:
    #     global_min_y = 0
    # else:
    #     global_min_y -= offset


    bounding_box = [
        (global_min_x, global_min_y),
        (global_max_x, global_min_y),
        (global_min_x, global_max_y),
        (global_max_x, global_max_y)
    ]

    return bounding_box


def main(global_val):
    '''
<<<<<<< HEAD

    :param global_val: list of variables storing extreme coordinates
    :return: bounding box for each of the video of ChaLearn Dataset
    '''

    store_json = "/home/axp798/axp798gallinahome/store/json/"

    dir_list = os.listdir(store_json)
    for dir_name in dir_list:
        video_list = os.listdir(os.path.join(store_json, dir_name))

        for video_name in video_list:
            json_path = os.path.join(store_json, "{}/{}/".format(dir_name, video_name))

            bounding_box = getBoundingBox(json_path, global_val)

            return bounding_box

    # print("started!")
    # bounding_box = getBoundingBox(store_json, global_val)
    #
=======
    
    :param global_val: list of variables storing extreme coordinates
    :return: bounding box for each of the video of ChaLearn Dataset
    '''

    store_json = "/home/axp798/axp798gallinahome/store/json/"

    dir_list = os.listdir(store_json)
    for dir_name in dir_list:
        video_list = os.listdir(os.path.join(store_json, dir_name))

        for video_name in video_list:
            json_path = os.path.join(store_json, "{}/{}/".format(dir_name, video_name))

            bounding_box = getBoundingBox(json_path)

            return bounding_box

    # print("started!")
    # bounding_box = getBoundingBox(store_json, global_val)
    # 
>>>>>>> 87803fcc53a05e4692baba2fd44429cc36b7b2a2
    # print(bounding_box)

if __name__ == '__main__':
    
    global_min_x = 100000
    global_max_x = 0

    global_min_y = 0
    global_max_y = img_height

<<<<<<< HEAD
    global_min_x = 100000
    global_max_x = 0

    global_min_y = 0
    global_max_y = img_height

=======
>>>>>>> 87803fcc53a05e4692baba2fd44429cc36b7b2a2
    global_val = [global_min_x, global_max_x, global_min_y, global_max_y]

    main(global_val)


