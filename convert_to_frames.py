import os
import cv2
import time

videos_path = "/home/axp798/axp798gallinahome/ConGD/ConGD_phase_1/train"
store_frames = "/home/axp798/axp798gallinahome/store/train/frames"

script_start_time = time.time()

# print(os.listdir(videos_path)[270])

# dir = "246"
# curr_dir_path = os.path.join(videos_path, dir)
# print(curr_dir_path)
# print(os.listdir(curr_dir_path))

dir_count = 271
for dir in os.listdir(videos_path)[270:]:
    print("{}) {} started!".format(dir_count, dir))
    curr_dir_path = os.path.join(videos_path, dir)

    par_dir = os.path.join(store_frames, dir)
    if not os.path.exists(par_dir):
        os.mkdir(par_dir)

    dir_start_time = time.time()

    video_count = 1
    for video in os.listdir(curr_dir_path):
        if video[-5] == 'K':
            continue
        video_start_time = time.time()
        print("{}) {}-{} started!".format(video_count, dir, video))
        vidcap = cv2.VideoCapture(os.path.join(curr_dir_path, video))
        success, image = vidcap.read()
        count = 1
        # make directory // video = 00001.K.avi therefore video[:-6]
        video_frame_dir = os.path.join(store_frames, "{}/{}".format(dir, video[ : -6]))

        if not os.path.exists(video_frame_dir):
            os.mkdir(video_frame_dir)

        while success:
            # print("frames started!")
            cv2.imwrite(os.path.join(video_frame_dir, "frame{}.jpg".format(count)), image)     # save frame as JPEG file
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            count += 1

        video_end_time = time.time()
        print("{}) {}-{} is done in {} sec".format(video_count, dir, video, video_end_time - video_start_time))
        video_count += 1

    dir_end_time = time.time()
    print("{}) {} is done in {} sec".format(dir_count, dir, dir_end_time - dir_start_time))
    dir_count += 1

script_end_time = time.time()
print("Done in {} sec".format(script_end_time - script_start_time))
