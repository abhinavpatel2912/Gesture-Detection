import os
import cv2

videos_path = "/home/axp798/axp798gallinahome/ConGD/ConGD_phase_1/train"
store_frames = "/home/axp798/axp798gallinahome/store/train/frames"

for dir in os.listdir(videos_path):
    curr_dir_path = os.path.join(videos_path, dir)

    for video in os.listdir(curr_dir_path):

        vidcap = cv2.VideoCapture(os.path.join(curr_dir_path, video))
        success, image = vidcap.read()
        count = 1
        # make directory // video = 00001.K.avi therefore video[:-6]
        video_frame_dir = os.path.join(store_frames, "{}/{}".format(dir, video[ : -6]))
        os.mkdir(video_frame_dir)
        while success:
          cv2.imwrite(os.path.join(video_frame_dir, "frame{}".format(count)), image)     # save frame as JPEG file
          success, image = vidcap.read()
          print('Read a new frame: ', success)
          count += 1