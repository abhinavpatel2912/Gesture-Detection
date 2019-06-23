# Importing Libraries

import os
import cv2

openpose_simg = "/home/axp798/axp798gallinahome/openpose_1_0_5.simg"  # Path to openpose singularity image being used
store_path = "/home/axp798/axp798gallinahome/store/valid/"  # Path to directory for storing rendered videos and json files from valid videos

videos_store = os.path.join(store_path, "videos/")  # directory to store rendered videos in the same format as valid videos are stored
frames_store = os.path.join(store_path, "frames/")  # directory to store video frames obtained from rendered videos stored in videos_store
json_store = os.path.join(store_path, "json/")  # directory to store json files containing hand, face and body coordinates from the videos

valid_dir = "/home/axp798/axp798gallinahome/ConGD/ConGD_phase_1/valid/"  # Path to list of directories containing validing videos
lib_dir = "/home/axp798/axp798gallinahome/"  # Path to directory to be accessible from within the singularity container


def create_frames(video_path, dir_name, video_name, parts):
    # Creates and stores video frames from the given video

    vidcap = cv2.VideoCapture(video_path)
    done, frame = vidcap.read()
    num = 0
    if not os.path.exists(os.path.join(frames_store, "{}/{}/".format(dir_name, parts[0]))):
        # create directory to store frames
        os.mkdir(os.path.join(frames_store, "{}/{}/".format(dir_name, parts[0])))

    while done:
        frame_store = os.path.join(frames_store, "/{}/{}/frame{}.jpg".format(dir_name, parts[0], num))
        cv2.imwrite(frame_store, frame)  # save frame as JPEG file
        done, frame = vidcap.read()
        print('Read a new frame: ', done)

        num += 1

    return


def delete_video(write_video_path):
    os.system("rm {}".format(write_video_path))  # execute the delete video command
    return


def main():

    # Uses the videos in the valid_dir to extract json files from them using openpose,
    # stores videos having coordinates in them,
    # creates frames from them and
    # deletes the rendered videos finally

    dir_ptr = 46

    for dir_name in os.listdir(valid_dir):

        print("Directory {} started!".format(dir_ptr))
        # dir_name = 001, 002, ..... , 027, 028, ..... , 128, 129, ....
        print("{} directory started ".format(dir_name))

        # video_dir = "/home/axp798/axp798gallinahome/ConGD/ConGD_phase_1/valid/001/"
        video_dir = os.path.join(valid_dir, dir_name)

        if not os.path.exists(os.path.join(frames_store, "{}/".format(dir_name))):
            # creating the directory to store frames from the rendered_videos
            # os.path.join(frames_store, "{}/".format(dir_name) = "/home/axp798/axp798gallinahome/store/valid/frames/001/"

            os.mkdir(os.path.join(frames_store, "{}/".format(dir_name)))

        if not os.path.exists(os.path.join(videos_store, "{}/".format(dir_name))):
            # creating the directory to store videos
            # os.path.join(videos_store, "{}/".format(dir_name)) = "/home/axp798/axp798gallinahome/store/valid/videos/001/"

            os.mkdir(os.path.join(videos_store, "{}/".format(dir_name)))

        if not os.path.exists(os.path.join(json_store, "{}/".format(dir_name))):
            # creating the directory to store json files
            # os.path.join(json_store, "{}/".format(dir_name)) = "/home/axp798/axp798gallinahome/store/valid/videos/001/"

            os.mkdir(os.path.join(json_store, "{}/".format(dir_name)))

        video_ptr = 1

        for video_name in os.listdir(video_dir):
            # video_name = 00001.K.avi, 00001.M.avi etc.
            # .k extension is for depth frames videos
            # .M extension videos is for 3D RGB videos

            parts = video_name.split('.')  # parts = [00001, k, avi] or parts = [00001, M, avi]

            if (parts[1] == 'K'):
                # excluding the depth frames videos
                continue

            else:
                # prcessing the RGB videos
                print("Directory {} and video {} started!".format(dir_ptr, video_ptr))
                print("{} video directory started ".format(video_name))

                add = "{}/{}".format(dir_name, video_name)  # add = "001/00001.M.avi"
                video_path = os.path.join(valid_dir, add)
                # video_path = "/home/axp798/axp798gallinahome/ConGD/ConGD_phase_1/valid/001/00001.M.avi"
                print(video_path)

                # write_video_path = "/home/axp798/axp798gallinahome/store/valid/videos/001/result_00001.avi"
                write_video_path = os.path.join(videos_store, "{}/result_{}.avi".format(dir_name, parts[0]))

                # write_json_path = "/home/axp798/axp798gallinahome/store/valid/json/001/00001"
                write_json_path = os.path.join(json_store, "{}/{}/".format(dir_name, parts[0]))

                print("command started!")

                command = "singularity run --nv --bind {},{} {} --video {} --write_keypoint_json {} --no_display --render_pose 0 --hand".format(
                    video_path,
                    lib_dir,
                    openpose_simg,
                    video_path,
                    write_json_path

                )

                os.system(command)  # run the command on terminal

                create_frames(write_video_path, dir_name, video_name, parts)  # convert the rendered videos with openpose coordinates to frames
                print("Frames creation for {} video in {} directory is done!".format(video_name, dir_name))

                delete_video(write_video_path)  # delete the rendered videos to save disk space
                print("Rendered video result_{}.avi is deleted!".format(parts[0]))

                video_ptr += 1
                print("Directory {} and video {} completed!".format(dir_ptr, video_ptr))

        dir_ptr += 1
        print("Directory {} completed!".format(dir_ptr))


if __name__ == '__main__':
    main()

