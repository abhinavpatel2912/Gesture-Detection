import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import time
import scipy.misc as smisc
import pickle
import random

# input_video = [batch_size, depth, height, width, num_channels]

print(device_lib.list_local_devices())

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

height = 96
width = 96
depth = 16

DTYPE = tf.float32
batch_size = 24
kernel_shape = 3

num_classes = 249
num_channels = 3
input_filter_shape = [3, 7, 7, num_channels, num_channels]
input_stride = [1, 1, 2, 2, 1]
conv_stride = [1, 1, 1, 1, 1]
kernel_name = "weights"
num_filters_list = [16, 32, 64, 128]  # as per the paper

with tf.device('/device:GPU:0'):
    x_input = tf.placeholder(tf.float32, shape=[None, depth, height, width, 3])
    y_input = tf.placeholder(tf.float32, shape=[None, num_classes])

def getKernel(name, shape):
    kernel = tf.get_variable(name, shape, DTYPE, tf.contrib.layers.xavier_initializer(uniform=True))
    return kernel

def _bias_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(0.1, dtype=DTYPE))

def conv3DBlock(prev_layer, layer_name, in_filters, out_filters):
    with tf.variable_scope(layer_name):

        kernel = getKernel(kernel_name, [kernel_shape, kernel_shape, kernel_shape, in_filters, out_filters])
        prev_layer = tf.nn.conv3d(prev_layer, kernel, strides=conv_stride, padding="SAME")
        print(layer_name, prev_layer.get_shape())

    return prev_layer

def input_block(input, input_filter_shape, input_stride):

    with tf.variable_scope("inpuT_layer"):
        out_filters = 3
        kernel = getKernel(kernel_name, inpuT_filter_shape)
        curr_layer = tf.nn.conv3d(inpuT, kernel, strides=inpuT_stride, padding="VALID")
        print("inpuT_layer", curr_layer.get_shape())

        curr_layer = tf.layers.batch_normalization(curr_layer, training=True)

        biases = _bias_variable('biases', [out_filters])
        print(biases.get_shape())
        curr_layer = tf.nn.bias_add(curr_layer, biases)
        curr_layer = tf.nn.relu(curr_layer)

    return curr_layer

def basic_block(input, in_filters, out_filters, layer_num, add_kernel_name):

    layer_name = "conv{}a".format(layer_num)
    prev_layer = conv3DBlock(input, layer_name, in_filters, out_filters)
    prev_layer = tf.layers.batch_normalization(prev_layer, training=True)
    prev_layer = tf.nn.relu(prev_layer)
    print(layer_name, prev_layer.get_shape())

    in_filters = out_filters

    layer_name = "conv{}b".format(layer_num)
    prev_layer = conv3DBlock(prev_layer, layer_name, in_filters, out_filters)
    prev_layer = tf.layers.batch_normalization(prev_layer, training=True)

    residual = prev_layer


    prev_layer = _shortcut3d(input, residual, add_kernel_name)
    prev_layer = tf.nn.relu(prev_layer)

    print(layer_name, prev_layer.get_shape())

    return prev_layer

def _shortcut3d(input, residual, add_kernel_name):
    """3D shortcut to match input and residual and sum them."""

    input_shape = input.get_shape().as_list()
    residual_shape = residual.get_shape().as_list()

    equal_channels = input_shape[-1] == residual_shape[-1]
    shortcut = input

    if not equal_channels:
        kernel = getKernel(add_kernel_name, [1, 1, 1, input_shape[-1], residual_shape[-1]])
        shortcut = tf.nn.conv3d(shortcut, kernel, strides=conv_stride, padding="SAME")

    return (shortcut + residual)

def inference(input):

    prev_layer = input_block(input, input_filter_shape, input_stride)
    in_filters = 3

    for index, out_filters in enumerate(num_filters_list):
        add_kernel_name = "add{}".format(index + 1)
        prev_layer = basic_block(prev_layer, in_filters, out_filters, index + 2, add_kernel_name)
        in_filters = out_filters

    shape_list = prev_layer.get_shape().as_list()

    num_frames = shape_list[1]
    height = shape_list[2]
    width = shape_list[3]

    prev_layer = tf.nn.avg_pool3d(prev_layer, [1, num_frames, height, width, 1], [1, 1, 1, 1, 1], padding="VALID")
    print(prev_layer.get_shape())

    with tf.variable_scope('fc1') as scope:

        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = getKernel('weights', [dim, FC_SIZE])
        biases = _bias_variable('biases', [FC_SIZE])

        fc1 = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)


    prev_layer = fc1

    with tf.variable_scope('softmax_linear') as scope:

        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])

        weights = getKernel('weights', [dim, num_classes])
        biases = _bias_variable('biases', [num_classes])

        softmax_linear = tf.add(tf.matmul(prev_layer_flat, weights), biases, name=scope.name)
        print(softmax_linear.get_shape())

    return softmax_linear


random_clips = 12
win_size = 16

with open('./../bounding_box_train.pkl', 'rb') as f:
    data = pickle.load(f)


def getFrameCut(frame, bounding_box):
    ext_left = np.ceil(bounding_box[0])
    ext_right = np.ceil(bounding_box[1])
    top = np.ceil(bounding_box[2])
    bottom = np.ceil(bounding_box[3])

    frame_cut = frame[top: bottom, ext_left: ext_right]

    return frame_cut

import scipy

with open('/home/axp798/axp798gallinahome/store/train/pickled_files_list.pkl', 'rb') as f:
    pickled_files_list = pickle.load(f)

file1 = open("epoch_loss.txt", "w")
file1.write("Losses: \n")
file1.close()

def train_neural_network(x_input, y_input, learning_rate=0.001, keep_rate=0.7, epochs=10):

    with tf.name_scope("cross_entropy"):

        prediction = inference(x_input)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_input))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))


    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        sess.run(tf.global_variables_initializer())
        import datetime

        start_time = time.time()

        epoch_loss = 0
        print("session starts!")

        num_batch_completed = 0
        for epoch in range(epochs):

            start_time_epoch = time.time()
            batch_start_time = time.time()
            print("Epoch {} started!".format(epoch + 1))
            epoch_loss = 0
            # mini batch

            mini_batch_x = []
            mini_batch_y = []
            batch_filled = 0

            pickle_files_index = 0
            for dict in data:

                video_frames_dir = dict['video_dir']
                bounding_box = dict['bounding_box']
                print(bounding_box)
                range_of_frames = dict['range_of_frames']
                label = dict['label']

                with open(pickled_files_list[pickle_files_index], 'rb') as f:
                    curr_video = pickle.load(f)

                cut_frames_list = curr_video

                num_frames = len(cut_frames_list)
                num_clips_per_video = random_clips
                window_size = win_size

                num_clip_index = 0

                while num_clip_index < num_clips_per_video:
                    start_frame = random.randint(0, num_frames - window_size)
                    end_frame = start_frame + window_size

                    if batch_filled == batch_size:
                        num_batch_completed += 1
                        mini_batch_x = np.array(mini_batch_x)
                        mini_batch_x = mini_batch_x / 255.0
                        mini_batch_y = np.array(mini_batch_y)

                        perm = np.random.permutation(batch_size)

                        mini_batch_x = mini_batch_x[perm]
                        mini_batch_y = mini_batch_y[perm]

                        _optimizer, _cost = sess.run([optimizer, cost], feed_dict={x_input: mini_batch_x, y_input: mini_batch_y})
                        epoch_loss += _cost
                        batch_end_time = time.time()

                        file1 = open("epoch_loss.txt", "a")
                        file1.write("batches completed: {}, time taken: {}, loss: {} \n".format(num_batch_completed, batch_end_time - batch_start_time, epoch_loss))
                        file1.close()
                        print("time taken: {}, epoch loss: {}".format(batch_end_time - batch_start_time, epoch_loss))
                        batch_start_time = time.time()

                        mini_batch_x = []
                        mini_batch_y = []
                        batch_filled = 0

                    mini_batch_x.append(cut_frames_list[start_frame: end_frame])
                    basic_line = [0] * num_classes
                    basic_line[int(label) - 1] = 1
                    basic_label = basic_line
                    # print("basic_label: {}".format(basic_label))

                    mini_batch_y.append(basic_label)
                    batch_filled += 1

                    num_clip_index += 1

        end_time = time.time()
        print('Time elapse: ', str(end_time - start_time))



train_neural_network(x_input, y_input)




