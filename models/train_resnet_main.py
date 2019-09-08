import tensorflow as tf
import sys
import os
import numpy as np
import resnet_10 as resnet
import time
import pickle
import random
import argparse
from tensorflow.python.client import device_lib
from AdamWOptimizer import create_optimizer


def append_frames(cut_frame_array, required_num_frames):
    appended_list_of_frames = list(cut_frame_array)

    num_curr_frames = cut_frame_array.shape[0]
    num_more_frames = required_num_frames - num_curr_frames

    for i in range(num_more_frames):
        appended_list_of_frames.append(cut_frame_array[i % num_curr_frames])

    return np.array(appended_list_of_frames)


def train_neural_network(x_inpuT,
                         y_inpuT,
                         data_path,
                         val_data_path,
                         save_loss_path,
                         save_model_path,
                         batch_size,
                         val_batch_size,
                         learning_rate,
                         weight_decay,
                         epochs,
                         which_model,
                         num_val_videos,
                         random_clips,
                         win_size,
                         ignore_factor):
    with tf.name_scope("cross_entropy"):

        prediction = 0
        if which_model == 1:
            prediction = resnet.inference(x_inpuT)

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

    saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        print("session starts!")

        sess.run(tf.global_variables_initializer())

        start_time = time.time()
        epoch_loss_list = []
        val_loss_list = []

        pkl_files_list = os.listdir(data_path)
        val_pkl_files_list = os.listdir(val_data_path)

        for epoch in range(epochs):

            print("Epoch {} started!".format(epoch + 1))
            epoch_start_time = time.time()

            epoch_loss = 0
            train_acc = 0
            num_batch_completed = 0

            window_size = win_size
            num_clips_per_video = random_clips

            random.seed(7)
            print("Random seed fixed for training.")

            # pkl_files_list = pkl_files_list[ : 64]
            num_videos = len(pkl_files_list) - (len(pkl_files_list) % batch_size)
            num_blocks = int(num_videos / batch_size)

            block_x = np.zeros((num_clips_per_video, batch_size, depth, height, width, 3))
            block_y = np.zeros((num_clips_per_video, batch_size, num_classes))

            for block_index in range(num_blocks):

                for index_in_batch, pkl_file in enumerate(
                        pkl_files_list[block_index * batch_size: (block_index + 1) * batch_size]):

                    with open(os.path.join(data_path, pkl_file), 'rb') as f:
                        frames_and_label = pickle.load(f)

                    cut_frame_array = frames_and_label[0]
                    label = frames_and_label[1]

                    num_frames = cut_frame_array.shape[0]
                    required_num_frames = int(window_size * ignore_factor)

                    if num_frames <= required_num_frames:
                        # print(num_frames, required_num_frames)
                        cut_frame_array = append_frames(cut_frame_array, required_num_frames)
                        num_frames = cut_frame_array.shape[0]
                        # print(num_frames)

                    for batch_index in range(num_clips_per_video):
                        start_frame = random.randint(0, num_frames - window_size)
                        end_frame = start_frame + window_size

                        block_x[batch_index, index_in_batch, :, :, :, :] = np.array(
                            cut_frame_array[start_frame: end_frame, :, :, :])

                        basic_line = [0] * num_classes
                        basic_line[int(label) - 1] = 1
                        basic_label = basic_line
                        block_y[batch_index, index_in_batch, :] = np.array(basic_label)

                for batch_index in range(num_clips_per_video):
                    batch_start_time = time.time()

                    mini_batch_x = block_x[batch_index, :, :, :, :, :]
                    mini_batch_x = mini_batch_x / 255.0
                    mini_batch_y = block_y[batch_index, :, :]

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

                    log1 = "\rEpoch: {}, " \
                           "batches completed: {}, " \
                           "time taken: {:.5f}, " \
                           "loss: {:.6f}, " \
                           "accuracy: {:.4f} \n". \
                        format(
                        epoch + 1,
                        num_batch_completed,
                        batch_end_time - batch_start_time,
                        epoch_loss / (batch_size * num_batch_completed),
                        _accuracy)

                    print(log1)
                    sys.stdout.flush()

            del block_x, block_y

            # validation loss
            val_loss = 0
            val_acc = 0
            val_num_batch_completed = 0

            num_clips_per_video = 1

            val_num_videos = num_val_videos
            val_num_blocks = int(val_num_videos / val_batch_size)

            val_block_x = np.zeros((num_clips_per_video, val_batch_size, depth, height, width, 3))
            val_block_y = np.zeros((num_clips_per_video, val_batch_size, num_classes))

            random.seed(23)
            print("Random seed fixed for validation.")

            for block_index in range(val_num_blocks):

                for index_in_batch, val_pkl_file in enumerate(
                        val_pkl_files_list[block_index * val_batch_size: (block_index + 1) * val_batch_size]):

                    with open(os.path.join(val_data_path, val_pkl_file), 'rb') as f:
                        frames_and_label = pickle.load(f)

                    cut_frame_array = frames_and_label[0]
                    label = frames_and_label[1]

                    num_frames = cut_frame_array.shape[0]
                    required_num_frames = int(window_size * ignore_factor)

                    if num_frames <= window_size:
                        cut_frame_array = append_frames(cut_frame_array, required_num_frames)
                        num_frames = cut_frame_array.shape[0]

                    for batch_index in range(num_clips_per_video):
                        start_frame = random.randint(0, num_frames - window_size)
                        end_frame = start_frame + window_size

                        val_block_x[batch_index, index_in_batch, :, :, :, :] = np.array(
                            cut_frame_array[start_frame: end_frame, :, :, :])

                        basic_line = [0] * num_classes
                        basic_line[int(label) - 1] = 1
                        basic_label = basic_line
                        val_block_y[batch_index, index_in_batch, :] = np.array(basic_label)

                for batch_index in range(num_clips_per_video):
                    val_batch_x = val_block_x[batch_index, :, :, :, :, :]
                    val_batch_x = val_batch_x / 255.0
                    val_batch_y = val_block_y[batch_index, :, :]

                    val_cost, val_batch_accuracy = sess.run([cost, accuracy],
                                                            feed_dict={x_inpuT: val_batch_x, y_inpuT: val_batch_y})

                    val_acc += val_batch_accuracy
                    val_loss += val_cost

                    val_num_batch_completed += 1

            del val_block_x, val_block_y

            epoch_loss = epoch_loss / (batch_size * num_batch_completed)
            train_acc = train_acc / num_batch_completed

            val_loss /= (val_batch_size * val_num_batch_completed)
            val_acc = val_acc / val_num_batch_completed

            epoch_end_time = time.time()

            log3 = "Epoch {} done; " \
                   "Time Taken: {:.4f}s; " \
                   "Train_loss: {:.6f}; " \
                   "Val_loss: {:.6f}; " \
                   "Train_acc: {:.4f}; " \
                   "Val_acc: {:.4f}; " \
                   "Train batches: {}; " \
                   "Val batches: {};\n". \
                format(epoch + 1, epoch_end_time - epoch_start_time, epoch_loss, val_loss, train_acc, val_acc,
                       num_batch_completed, val_num_batch_completed)

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

    parser.add_argument('-bs', action='store', dest='batch_size', type=int)
    parser.add_argument('-vbs', action='store', dest='val_batch_size', type=int)
    parser.add_argument('-lr', action='store', dest='learning_rate', type=float)
    parser.add_argument('-wd', action='store', dest='weight_decay', type=float, const=None)
    parser.add_argument('-e', action='store', dest='epochs', type=int)
    parser.add_argument('-nvv', action='store', dest='num_val_videos', type=int)
    parser.add_argument('-rc', action='store', dest='random_clips', type=int)
    parser.add_argument('-ws', action='store', dest='win_size', type=int)
    parser.add_argument('-slp', action='store', dest='save_loss_path', const=None)
    parser.add_argument('-smp', action='store', dest='save_model_path', const=None)
    parser.add_argument('-mn', action='store', dest='model_num', type=int)
    parser.add_argument('-vd', action='store', dest='visible_devices')
    parser.add_argument('-nd', action='store', dest='num_device', type=int)
    parser.add_argument('-if', action='store', dest='ign_fact', type=float, const=None)

    results = parser.parse_args()

    arg_batch_size = results.batch_size
    arg_val_batch_size = results.val_batch_size
    arg_lr = results.learning_rate
    arg_wd = results.weight_decay
    arg_epochs = results.epochs
    arg_num_val_videos = results.num_val_videos
    arg_random_clips = results.random_clips
    arg_win_size = results.win_size
    arg_save_loss_path = results.save_loss_path
    arg_save_model_path = results.save_model_path
    arg_model_num = results.model_num
    arg_visible_devices = results.visible_devices
    arg_num_device = results.num_device
    arg_ign_fact = results.ign_fact

    data_path = "/home/axp798/axp798gallinahome/data/jester/train_64/"
    val_data_path = "/home/axp798/axp798gallinahome/data/jester/valid_64/"

    ar_save_loss_path = None
    if arg_save_loss_path is not None:
        ar_save_loss_path = "/home/axp798/axp798gallinahome/Gesture-Detection/models/loss_jester/{}".format(
            arg_save_loss_path)

    ar_save_model_path = None
    if arg_save_model_path is not None:
        path = '/home/axp798/axp798gallinahome/Gesture-Detection/models/saved_models_jester/{}/'.format(
            arg_save_model_path)
        if not os.path.exists(path):
            os.mkdir(path)
        ar_save_model_path = path + "model"

    if ar_save_loss_path is not None:
        file1 = open(ar_save_loss_path, "w")
        file1.write("Params: {} \n".format(results))
        file1.write("Losses: \n")
        file1.close()

    depth = arg_win_size
    height = 64
    width = 64
    num_classes = 27

    os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(arg_visible_devices)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
    print(device_lib.list_local_devices())

    choose_device = "/device:GPU:{}".format(arg_num_device)

    with tf.device(choose_device):
        x_inpuT = tf.placeholder(tf.float32, shape=[arg_batch_size, depth, height, width, 3])
        y_inpuT = tf.placeholder(tf.float32, shape=[arg_batch_size, num_classes])

    train_neural_network(x_inpuT, y_inpuT, data_path, val_data_path,
                         save_loss_path=ar_save_loss_path,
                         save_model_path=ar_save_model_path,
                         batch_size=arg_batch_size,
                         learning_rate=arg_lr,
                         weight_decay=arg_wd,
                         epochs=arg_epochs,
                         val_batch_size=arg_val_batch_size,
                         which_model=arg_model_num,
                         num_val_videos=arg_num_val_videos,
                         random_clips=arg_random_clips,
                         win_size=arg_win_size,
                         ignore_factor=arg_ign_fact)
