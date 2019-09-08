import tensorflow as tf
import os
import numpy as np
import resnet_10 as res10
import time
import pickle
import random


def train_neural_network(x_inpuT,
                         y_inpuT,
                         data_path,
                         val_data_path,
                         save_loss_path,
                         save_model_path,
                         learning_rate=0.001,
                         epochs=2,
                         num_val_videos=2048,
                         val_batch_size=64,
                         random_clips=16,
                         win_size=12):
    with tf.name_scope("cross_entropy"):

        prediction = res10.inference(x_inpuT)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_inpuT))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
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

        file1 = open(save_loss_path, "w")
        file1.write("Losses: \n")
        file1.close()

        for epoch in range(epochs):

            print("Epoch {} started!".format(epoch + 1))
            epoch_start_time = time.time()

            epoch_loss = 0
            train_acc = 0

            num_batch_completed = 0

            # mini batch
            batch_start_time = time.time()

            mini_batch_x = []
            mini_batch_y = []
            batch_filled = 0

            for pkl_file in pkl_files_list:

                with open(os.path.join(data_path, pkl_file), 'rb') as f:
                    frames_and_label = pickle.load(f)

                cut_frame_array = frames_and_label[0]
                label = frames_and_label[1]

                num_frames = cut_frame_array.shape[0]
                num_clips_per_video = random_clips
                window_size = win_size

                num_clip_index = 0

                while num_clip_index < num_clips_per_video:
                    start_frame = random.randint(0, num_frames - window_size)
                    end_frame = start_frame + window_size

                    mini_batch_x.append(cut_frame_array[start_frame: end_frame, :, :, :])
                    basic_line = [0] * num_classes
                    curr_label = 0 if int(label) == 25 or int(label) == 26 else 1
                    basic_line[curr_label] = 1
                    basic_label = basic_line

                    mini_batch_y.append(basic_label)

                    batch_filled += 1
                    num_clip_index += 1

                if batch_filled == batch_size:
                    num_batch_completed += 1

                    mini_batch_x = np.array(mini_batch_x)
                    mini_batch_x = mini_batch_x / 255.0
                    mini_batch_y = np.array(mini_batch_y)

                    perm = np.random.permutation(batch_size)
                    mini_batch_x = mini_batch_x[perm]
                    mini_batch_y = mini_batch_y[perm]

                    _optimizer, _cost, _prediction, _accuracy = sess.run([optimizer, cost, prediction, accuracy],
                                                                         feed_dict={x_inpuT: mini_batch_x,
                                                                                    y_inpuT: mini_batch_y})
                    epoch_loss += _cost
                    train_acc += _accuracy
                    batch_end_time = time.time()

                    log1 = "Epoch: {}, batches completed: {}, time taken: {:.5f}, loss: {:.6f}, accuracy: {:.4f} \n".format(
                        epoch + 1,
                        num_batch_completed,
                        batch_end_time - batch_start_time,
                        epoch_loss / (batch_size * num_batch_completed),
                        _accuracy)

                    print(log1)
                    batch_start_time = time.time()

                    mini_batch_x = []
                    mini_batch_y = []
                    batch_filled = 0

            num_itr = int(num_val_videos / val_batch_size)

            # validation loss
            val_loss = 0
            val_acc = 0
            for i in range(num_itr):
                val_batch_x = []
                val_batch_y = []

                for val_pkl_file in val_pkl_files_list[i * val_batch_size: (i + 1) * val_batch_size]:
                    with open(os.path.join(val_data_path, val_pkl_file), 'rb') as f:
                        frames_and_label = pickle.load(f)

                    cut_frame_array = frames_and_label[0]
                    label = frames_and_label[1]

                    num_frames = cut_frame_array.shape[0]
                    window_size = win_size

                    start_frame = random.randint(0, num_frames - window_size)
                    end_frame = start_frame + window_size

                    val_batch_x.append(cut_frame_array[start_frame: end_frame, :, :, :])

                    basic_line = [0] * num_classes
                    curr_label = 0 if int(label) == 25 or int(label) == 26 else 1
                    basic_line[curr_label] = 1
                    basic_label = basic_line

                    val_batch_y.append(basic_label)

                val_batch_x = np.array(val_batch_x)
                val_batch_x = val_batch_x / 255.0
                val_batch_y = np.array(val_batch_y)

                val_cost, batch_accuracy = sess.run([cost, accuracy],
                                                    feed_dict={x_inpuT: val_batch_x, y_inpuT: val_batch_y})

                val_acc += batch_accuracy
                val_loss += val_cost

            epoch_loss = epoch_loss / (batch_size * num_batch_completed)
            train_acc = train_acc / num_batch_completed

            val_loss /= (num_val_videos)
            val_acc = val_acc * batch_size / num_val_videos

            epoch_end_time = time.time()

            log3 = "Epoch {} done; Time Taken: {:.4f}s; Train_loss: {:.6f}; Val_loss: {:.6f}; Train_acc: {:.4f}; Val_acc: {:.4f} \n".format(
                epoch + 1, epoch_end_time - epoch_start_time, epoch_loss, val_loss, train_acc, val_acc)

            print(log3)
            file1 = open(save_loss_path, "a")
            file1.write(log3)
            file1.close()

            epoch_loss_list.append(epoch_loss)
            val_loss_list.append(val_loss)

            saver.save(sess, save_model_path)

        end_time = time.time()
        print('Time elapse: ', str(end_time - start_time))
        print(epoch_loss_list)

        file1 = open(save_loss_path, "a")
        file1.write("Train Loss List: {} \n".format(str(epoch_loss_list)))
        file1.write("Val Loss List: {} \n".format(str(val_loss_list)))
        file1.close()


if __name__ == '__main__':
    np.random.seed(0)

    data_path = "/home/axp798/axp798gallinahome/data/jester/train_64/"
    val_data_path = "/home/axp798/axp798gallinahome/data/jester/valid_64/"
    save_loss_path = "/home/axp798/axp798gallinahome/Gesture-Detection/models/loss_jester/f6.txt"
    save_model_path = '/home/axp798/axp798gallinahome/Gesture-Detection/models/model_path6/model'

    batch_size = 64
    depth = 12
    height = 64
    width = 64
    num_classes = 2

    with tf.device('/device:GPU:0'):
        x_inpuT = tf.placeholder(tf.float32, shape=[batch_size, depth, height, width, 3])
        y_inpuT = tf.placeholder(tf.float32, shape=[batch_size, num_classes])

    train_neural_network(x_inpuT, y_inpuT, data_path, val_data_path, save_loss_path, save_model_path)
