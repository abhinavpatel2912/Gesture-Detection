import argparse
import os
import pickle
import random
import sys
import time

from resnext import ResNeXt
import c3D as conv3d4
import c3D2 as conv3d2
import c3D3 as conv3d3
import c3D_main as conv3d1
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from chalearn_utils import data_augmentation
from tensorflow.python.client import device_lib


# from AdamWOptimizer import create_optimizer
# from tensorflow.keras import *
# from tensorflow.keras.layers import *
# from tensorflow.keras.models import *


def append_frames(cut_frame_array, required_num_frames):
    appended_list_of_frames = list(cut_frame_array)

    num_curr_frames = cut_frame_array.shape[0]
    num_more_frames = required_num_frames - num_curr_frames

    for i in range(num_more_frames):
        appended_list_of_frames.append(cut_frame_array[i % num_curr_frames])

    return np.array(appended_list_of_frames)


def get_video_reduced_fps(appended_video, factor):
    num_total_frames = appended_video.shape[0]
    final_video = []
    for index in range(0, num_total_frames, factor):
        final_video.append(appended_video[index])

    return np.array(final_video)


def get_final_video(par_name, vid_name, label_info, num_video_in_curr_line, window_size, set='train'):

    curr_label = label_info[num_video_in_curr_line]
    curr_label_info = curr_label.split(':')

    rang = curr_label_info[0]
    label = curr_label_info[1]

    start_frame = rang.split(',')[0]
    end_frame = rang.split(',')[1]

    start_frame, end_frame = int(start_frame), int(end_frame)

    frame_list = []

    for ind in range(start_frame - 1, end_frame):
        frame_name = "frame{}.jpg".format(ind + 1)
        if set == 'train':
            video_frame_dir = os.path.join(train_frames_path, "{}/{}".format(par_name, vid_name))
        else:
            video_frame_dir = os.path.join(valid_frames_path, "{}/{}".format(par_name, vid_name))
        frame_path = os.path.join(video_frame_dir, frame_name)
        frame = plt.imread(frame_path)
        frame_list.append(frame)

    frame_array = np.array(frame_list)
    num_frames = frame_array.shape[0]

    required_num_frames = (int(num_frames / window_size) + 1) * window_size
    appended_video = append_frames(frame_array, required_num_frames)  # returns numpy array
    factor = int(appended_video.shape[0] / window_size)
    final_video = get_video_reduced_fps(appended_video, factor)  # returns numpy array

    return final_video, label


def extract_line_info(curr_line):
    line_ele = curr_line.split(' ')
    dir_info = line_ele[0]
    par_name, vid_name = dir_info.split('/')[0], dir_info.split('/')[1]
    label_info = line_ele[1:]
    # total_video_in_curr_line = len(label_info)

    return par_name, vid_name, label_info


def train_neural_network(x_inpuT,
                         y_inpuT,
                         labels_path,
                         val_labels_path,
                         save_loss_path,
                         save_model_path,
                         batch_size,
                         val_batch_size,
                         image_height,
                         image_width,
                         learning_rate,
                         weight_decay,
                         num_iter,
                         epochs,
                         which_model,
                         num_train_videos,
                         num_val_videos,
                         win_size):

    with tf.name_scope("cross_entropy"):

        prediction = 0
        shapes_list = None
        if which_model == 1:
            prediction, shapes_list = conv3d1.inference(x_inpuT)

        elif which_model == 2:
            resnext_model = ResNeXt(x_inpuT, tf.constant(True, dtype=tf.bool))
            prediction = resnext_model.Build_ResNext(x_inpuT)
        elif which_model == 3:
            prediction = conv3d2.inference(x_inpuT)
        elif which_model == 4:
            prediction = conv3d3.inference(x_inpuT)
        elif which_model == 5:
            prediction, shapes_list = conv3d4.inference(x_inpuT)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_inpuT))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):

        # optimizer = 0
        if weight_decay is not None:
            print("weight decay applied.")
            optimizer = create_optimizer(cost, learning_rate, weight_decay)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.name_scope("accuracy"):
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_inpuT, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    # saver = tf.train.Saver(save_relative_paths=True)
    print("Calculating total parameters in the model")
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)
    if shapes_list is not None:
        print(shapes_list)

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        print("session starts!")

        sess.run(tf.global_variables_initializer())

        start_time = time.time()
        epoch_loss_list = []
        val_loss_list = []

        ori_height = 240
        ori_width = 320

        with open(labels_path, 'r') as f:
            lines = f.readlines()

        with open(val_labels_path, 'r') as f:
            val_lines = f.readlines()

        for epoch in range(epochs):

            print("Epoch {} started!".format(epoch + 1))
            epoch_start_time = time.time()

            epoch_loss = 0
            train_acc = 0

            window_size = win_size

            random.seed(7)
            print("Random seed fixed for training.")

            num_videos = num_train_videos
            num_batches = int(num_videos / batch_size)
            num_iter_total_data_per_epoch = num_iter

            for iter_index in range(num_iter_total_data_per_epoch):

                prev_line = -1
                num_line = 0
                num_video_in_curr_line = 0

                # for batch_num in range(num_batches):

                batch_x = np.zeros((batch_size, depth, ori_height, ori_width, 3))
                batch_y = np.zeros((batch_size, num_classes))
                batch_index = 0
                num_batch_completed = 0

                while True:
                    if prev_line != num_line:
                        curr_line = lines[num_line]
                        par_name, vid_name, label_info = extract_line_info(curr_line)
                        total_video_in_curr_line = len(label_info)

                    final_video, label = get_final_video(par_name, vid_name, label_info, num_video_in_curr_line, window_size, set='train')
                    # print(final_video.shape)

                    batch_x[batch_index, :, :, :, :] = final_video

                    basic_line = [0] * num_classes
                    basic_line[int(label) - 1] = 1
                    basic_label = basic_line
                    batch_y[batch_index, :] = np.array(basic_label)
                    batch_index += 1

                    if batch_index == batch_size:
                        # train_batch
                        batch_start_time = time.time()

                        mini_batch_x = data_augmentation(batch_x, (image_height, image_width))
                        # mini_batch_x = mini_batch_x / 255.0
                        mini_batch_y = batch_y

                        perm = np.random.permutation(batch_size)
                        mini_batch_x = mini_batch_x[perm]
                        mini_batch_y = mini_batch_y[perm]

                        _optimizer, _cost, _prediction, _accuracy = sess.run([optimizer, cost, prediction, accuracy],
                                                                             feed_dict={x_inpuT: mini_batch_x,
                                                                                        y_inpuT: mini_batch_y})
                        epoch_loss += _cost
                        train_acc += _accuracy

                        num_batch_completed += 1
                        batch_end_time = time.time()

                        total_train_batch_completed = iter_index * num_batches + num_batch_completed

                        log1 = "\rEpoch: {}, " \
                               "iter: {}, " \
                               "batches completed: {}, " \
                               "time taken: {:.5f}, " \
                               "loss: {:.6f}, " \
                               "accuracy: {:.4f} \n". \
                            format(
                            epoch + 1,
                            iter_index + 1,
                            total_train_batch_completed,
                            batch_end_time - batch_start_time,
                            epoch_loss / (batch_size * total_train_batch_completed),
                            _accuracy)

                        print(log1)
                        sys.stdout.flush()

                        if num_batch_completed == num_batches:
                            break

                        batch_index = 0
                        batch_x = np.zeros((batch_size, depth, ori_height, ori_width, 3))
                        batch_y = np.zeros((batch_size, num_classes))

                    prev_line = num_line
                    num_video_in_curr_line += 1
                    if num_video_in_curr_line == total_video_in_curr_line:
                        num_video_in_curr_line = 0
                        num_line += 1

            # validation loss
            print("<---------------- Validation Set started ---------------->")
            val_loss = 0
            val_acc = 0

            val_num_videos = num_val_videos
            val_num_batches = int(val_num_videos / val_batch_size)

            random.seed(23)
            print("Random seed fixed for validation.")

            for __ in range(num_iter_total_data_per_epoch):
                prev_line = -1
                num_line = 0
                num_video_in_curr_line = 0

                # for batch_num in range(val_num_batches):
                #
                val_batch_x = np.zeros((val_batch_size, depth, ori_height, ori_width, 3))
                val_batch_y = np.zeros((val_batch_size, num_classes))
                batch_index = 0
                val_num_batch_completed = 0

                while True:
                    if prev_line != num_line:
                        curr_line = val_lines[num_line]
                        par_name, vid_name, label_info = extract_line_info(curr_line)
                        total_video_in_curr_line = len(label_info)

                    # process video
                    final_video, label = get_final_video(par_name, vid_name, label_info, num_video_in_curr_line, window_size, set='valid')
                    # print(final_video.shape)

                    val_batch_x[batch_index, :, :, :, :] = final_video

                    basic_line = [0] * num_classes
                    basic_line[int(label) - 1] = 1
                    basic_label = basic_line
                    val_batch_y[batch_index, :] = np.array(basic_label)
                    batch_index += 1

                    if batch_index == val_batch_size:
                        val_batch_x = data_augmentation(val_batch_x, (image_height, image_width))
                        # val_batch_x = val_batch_x / 255.0

                        perm = np.random.permutation(batch_size)
                        val_batch_x = val_batch_x[perm]
                        val_batch_y = val_batch_y[perm]

                        val_cost, val_batch_accuracy = sess.run([cost, accuracy],
                                                                feed_dict={x_inpuT: val_batch_x, y_inpuT: val_batch_y})

                        val_acc += val_batch_accuracy
                        val_loss += val_cost

                        val_num_batch_completed += 1

                        if val_num_batch_completed == val_num_batches:
                            break

                        val_batch_x = np.zeros((val_batch_size, depth, ori_height, ori_width, 3))
                        val_batch_y = np.zeros((val_batch_size, num_classes))
                        batch_index = 0

                    prev_line = num_line
                    num_video_in_curr_line += 1
                    if num_video_in_curr_line == total_video_in_curr_line:
                        num_video_in_curr_line = 0
                        num_line += 1

            epoch_end_time = time.time()

            total_train_batch_completed = num_iter_total_data_per_epoch * num_batch_completed
            total_val_num_batch_completed = num_iter_total_data_per_epoch * val_num_batch_completed

            epoch_loss = epoch_loss / (batch_size * total_train_batch_completed)
            train_acc = train_acc / total_train_batch_completed

            val_loss /= (val_batch_size * total_val_num_batch_completed)
            val_acc = val_acc / total_val_num_batch_completed

            log3 = "Epoch {} done; " \
                   "Time Taken: {:.4f}s; " \
                   "Train_loss: {:.6f}; " \
                   "Val_loss: {:.6f}; " \
                   "Train_acc: {:.4f}; " \
                   "Val_acc: {:.4f}; " \
                   "Train batches: {}; " \
                   "Val batches: {};\n". \
                format(epoch + 1, epoch_end_time - epoch_start_time, epoch_loss, val_loss, train_acc, val_acc,
                       num_iter_total_data_per_epoch * num_batch_completed,
                       num_iter_total_data_per_epoch * val_num_batch_completed)

            print(log3)

            if save_loss_path is not None:
                file1 = open(save_loss_path, "a")
                file1.write(log3)
                file1.close()

            epoch_loss_list.append(epoch_loss)
            val_loss_list.append(val_loss)

            if save_model_path is not None:
                saver.save(sess, save_model_path)

        end_time = time.time()
        print('Time elapse: ', str(end_time - start_time))
        print(epoch_loss_list)

        if save_loss_path is not None:
            file1 = open(save_loss_path, "a")
            file1.write("Train Loss List: {} \n".format(str(epoch_loss_list)))
            file1.write("Val Loss List: {} \n".format(str(val_loss_list)))
            file1.close()


if __name__ == '__main__':

    np.random.seed(0)
    parser = argparse.ArgumentParser()

    parser.add_argument('-cs', action='store', dest='check_singularity', type=int)
    parser.add_argument('-ih', action='store', dest='height', type=int)
    parser.add_argument('-iw', action='store', dest='width', type=int)
    parser.add_argument('-bs', action='store', dest='batch_size', type=int)
    parser.add_argument('-vbs', action='store', dest='val_batch_size', type=int)
    parser.add_argument('-lr', action='store', dest='learning_rate', type=float)
    parser.add_argument('-wd', action='store', dest='weight_decay', type=float, const=None)
    parser.add_argument('-ni', action='store', dest='num_iter', type=int)
    parser.add_argument('-e', action='store', dest='epochs', type=int)
    parser.add_argument('-ntv', action='store', dest='num_train_videos', type=int)
    parser.add_argument('-nvv', action='store', dest='num_val_videos', type=int)
    parser.add_argument('-ws', action='store', dest='win_size', type=int)
    parser.add_argument('-slp', action='store', dest='save_loss_path', const=None)
    parser.add_argument('-smp', action='store', dest='save_model_path', const=None)
    parser.add_argument('-mn', action='store', dest='model_num', type=int)
    parser.add_argument('-vd', action='store', dest='visible_devices')
    parser.add_argument('-nd', action='store', dest='num_device', type=int)

    results = parser.parse_args()

    arg_check_singularity = results.check_singularity
    arg_height = results.height
    arg_width = results.width
    arg_batch_size = results.batch_size
    arg_val_batch_size = results.val_batch_size
    arg_lr = results.learning_rate
    arg_wd = results.weight_decay
    arg_num_iter = results.num_iter
    arg_epochs = results.epochs
    arg_num_val_videos = results.num_val_videos
    arg_num_train_videos = results.num_train_videos
    arg_win_size = results.win_size
    arg_save_loss_path = results.save_loss_path
    arg_save_model_path = results.save_model_path
    arg_model_num = results.model_num
    arg_visible_devices = results.visible_devices
    arg_num_device = results.num_device

    labels_path = '/home/axp798/axp798gallinahome/ConGD/ConGD_labels/train.txt'
    val_labels_path = '/home/axp798/axp798gallinahome/ConGD/ConGD_labels/valid.txt'
    train_frames_path = "/home/axp798/axp798gallinahome/store/train/frames/"
    valid_frames_path = "/home/axp798/axp798gallinahome/store/valid/frames/"
    save_loss = "/home/axp798/axp798gallinahome/Gesture-Detection/models/loss_chalearn/"
    save_model = "/home/axp798/axp798gallinahome/Gesture-Detection/models/saved_models_chalearn/"

    if arg_check_singularity:
        labels_path = '/mnt/ConGD_labels/train.txt'
        val_labels_path = '/mnt/ConGD_labels/valid.txt'
        train_frames_path = "/shared/train/frames/"
        valid_frames_path = "/shared/valid/frames/"
        save_loss = "/home/models/loss_chalearn/"
        save_model = "/home/models/saved_models_chalearn/"

    ar_save_loss_path = None
    if arg_save_loss_path is not None:
        ar_save_loss_path = save_loss + "{}".format(arg_save_loss_path)

    ar_save_model_path = None
    if arg_save_model_path is not None:
        path = save_model + "{}/".format(arg_save_model_path)
        if not os.path.exists(path):
            os.mkdir(path)

        ar_save_model_path = path + "model"

    if ar_save_loss_path is not None:
        file1 = open(ar_save_loss_path, "w")
        file1.write("Params: {} \n".format(results))
        file1.write("Losses: \n")
        file1.close()

    depth = arg_win_size
    num_classes = 249

    os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(arg_visible_devices)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
    print(device_lib.list_local_devices())

    choose_device = "/device:GPU:{}".format(arg_num_device)
    tf.reset_default_graph()
    with tf.device(choose_device):
        x_inpuT = tf.placeholder(tf.float32, shape=[arg_batch_size, depth, arg_height, arg_width, 3])
        y_inpuT = tf.placeholder(tf.float32, shape=[arg_batch_size, num_classes])

        train_neural_network(x_inpuT, y_inpuT, labels_path, val_labels_path,
                             save_loss_path=ar_save_loss_path,
                             save_model_path=ar_save_model_path,
                             batch_size=arg_batch_size,
                             val_batch_size=arg_val_batch_size,
                             image_height=arg_height,
                             image_width=arg_width,
                             learning_rate=arg_lr,
                             weight_decay=arg_wd,
                             num_iter=arg_num_iter,
                             epochs=arg_epochs,
                             which_model=arg_model_num,
                             num_train_videos=arg_num_train_videos,
                             num_val_videos=arg_num_val_videos,
                             win_size=arg_win_size)

