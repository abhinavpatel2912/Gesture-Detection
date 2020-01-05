import os

openpose_simg = "/home/axp798/axp798gallinahome/openpose_1_0_5.simg"  # Path to openpose singularity image being used
store_path = "/home/axp798/axp798gallinahome/"  # Path to directory for storing rendered videos and json files from train videos
json_store = os.path.join(store_path, "JD_JSON/")  # directory to store json files containing hand, face and body coordinates from the videos

jd_videos_dir = "/home/axp798/axp798gallinahome/jester_datase/"  # Path to list of directories containing training videos
lib_dir = "/home/axp798/axp798gallinahome/"  # Path to directory to be accessible from within the singularity container


def main():
    video_ptr = 0
    videos_list = os.listdir(jd_videos_dir)

    for video_name in videos_list:

        print("Video {}-{} started!".format(video_ptr + 1, video_name))
        images_dir = os.path.join(jd_videos_dir, video_name)
        modified_video_name = ('0' * (6 - len(video_name))) + video_name  ## 2 --> 000002
        print(modified_video_name)
        if not os.path.exists(os.path.join(json_store, "{}/".format(modified_video_name))):
            os.mkdir(os.path.join(json_store, "{}/".format(modified_video_name)))

        video_path = images_dir
        write_json_path = os.path.join(json_store, "{}/".format(modified_video_name))

        command = "singularity run --nv --bind {} {} --image_dir {} --write_keypoint_json {} --no_display --render_pose 0 --hand".format(
                    lib_dir,
                    openpose_simg,
                    video_path,
                    write_json_path)
        os.system(command)  # run the command on terminal
        video_ptr += 1
        print("Video {} completed!".format(video_ptr))


if __name__ == '__main__':
    main()

