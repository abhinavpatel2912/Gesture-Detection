# Importing Libraries
import os
import cv2

openpose_simg = "/home/axp798/axp798gallinahome/openpose_105.img"  # Path to openpose singularity image being used
store_path = "/home/axp798/axp798gallinahome/store/valid/"         # Path to directory for storing rendered videos and json files from valid videos

videos_store = os.path.join(store_path, "videos/")  # directory to store rendered videos in the same format as valid videos are stored
json_store = os.path.join(store_path, "json/")   # directory to store json files containing hand, face and body coordinates from the videos

valid_dir = "/home/axp798/axp798gallinahome/ConGD/ConGD_phase_1/valid/"  # Path to list of directories containing validation videos
video_bind = ""
path_bind = "/home/axp798/axp798gallinahome/,{}".format(video_bind)      # Path to directory to be accessible from within the singularity container


def create_frames(video_path, dir_name, video_name, parts):

    # Creates and stores video frames from the given video

    vidcap = cv2.VideoCapture(video_path)
    done, frame = vidcap.read()
    num = 0
    if not os.path.exists(os.path.join(store_path, "frames/{}/{}/".format(dir_name, parts[0]))):
        # create directory to store frames
        os.mkdir(os.path.join(store_path, "frames/{}/{}/".format(dir_name, parts[0])))
    while done:
        frame_store = os.path.join(store_path, "frames/{}/{}/frames{}.jpg".format(dir_name, parts[0], num))
        cv2.imwrite(frame_store, frame)     # save frame as JPEG file      
        done, frame = vidcap.read()
        print('Read a new frame: ', done)
        num += 1
    
    return

def delete_video(write_video_path):
    os.system("rm {}".format(write_video_path))  # execute the delete video command
    return

def main():

    # Uses the videos in the valid_dir to extract json files from them using openpose, stores videos having coordinates in them, creates frames from them and deletes the videos finally
    

    for dir_name in os.listdir(valid_dir):

        # dir_name = 001, 002, ..... , 027, 028, ..... , 128, 129, ....
        print("{} directory started ".format(dir_name))

        video_dir = os.path.join(valid_dir, dir_name)
        # video_dir = "/home/axp798/axp798gallinahome/ConGD/ConGD_phase_1/valid/001/"

        if not os.path.exists(os.path.join(store_path, "frames/{}/".format(dir_name))):
            os.mkdir(os.path.join(store_path, "frames/{}/".format(dir_name)))

	    
        if not os.path.exists(os.path.join(videos_store, "{}/".format(dir_name))):
            # creating the directory to store videos
            # os.path.join(videos_store, "{}/".format(dir_name)) = "/home/axp798/axp798gallinahome/store/valid/videos/001/"

        	os.mkdir(os.path.join(videos_store, "{}/".format(dir_name)))

        
        if not os.path.exists(os.path.join(json_store, "{}/".format(dir_name))):
            os.mkdir(os.path.join(json_store, "{}/".format(dir_name)))

        for video_name in os.listdir(video_dir):

            # video_name = 00001.K.avi, 00001.M.avi etc.
            # .k extension is for depth frames videos
            # .M extension videos is for 3D RGB videos

            print("{} video directory started ".format(video_name))
            parts = video_name.split('.')   # parts = [00001, k, avi] or parts = [00001, M, avi]

            if(parts[1] == 'K'):
                # excluding the depth frames videos
                continue

            else:
                # prcessing the RGB videos 
                add = "{}/{}".format(dir_name, video_name)    # add = "001/00001.M.avi"
                video_path = os.path.join(valid_dir, add)     
                # video_path = "/home/axp798/axp798gallinahome/ConGD/ConGD_phase_1/valid/001/00001.M.avi"
                print(video_path)


                # write_video_path = "/home/axp798/axp798gallinahome/store/valid/videos/001/result_00001.avi"
                write_video_path = os.path.join(videos_store, "{}/result_{}.avi".format(dir_name, parts[0]))

                
                # write_json_path = "/home/axp798/axp798gallinahome/store/valid/json/001/00001" 
                write_json_path = os.path.join(json_store, "{}/{}/".format(dir_name, parts[0]))

                video_bind = video_path

                print(path_bind)
                print(openpose_simg)
                print(video_path)
                print(write_video_path)
                print(write_json_path)

                
                command = "singularity run --bind " + path_bind + " " + openpose_simg + " --video " + video_path +  " --write_video " + write_video_path + " --write_json " + write_json_path + " --display 0 --face --hand"   # command to be executed on terminal
                
                os.system(command)  # run the command on terminal

                create_frames(write_video_path, dir_name, video_name, parts)   # convert the rendered videos with openpose coordinates to frames

                delete_video(write_video_path)                                 # delete the rendered videos to save disk space
                


if __name__ == '__main__':
    main()

