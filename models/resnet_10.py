import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import time
import scipy.misc as smisc
import pickle
import random

def getKernel(name, shape):
    kernel = tf.get_variable(name, shape, tf.float32, tf.contrib.layers.xavier_initializer(uniform=True))
    return kernel


def _bias_variable(name, shape):
    return tf.get_variable(name, shape, tf.float32, tf.constant_initializer(0.1, dtype=tf.float32))


def conv3DBlock(prev_layer, layer_name, in_filters, out_filters):
    kernel_shape = 3
    kernel_name = "weights"
    conv_stride = [1, 1, 1, 1, 1]
    with tf.variable_scope(layer_name):
        kernel = getKernel(kernel_name, [kernel_shape, kernel_shape, kernel_shape, in_filters, out_filters])
        prev_layer = tf.nn.conv3d(prev_layer, kernel, strides=conv_stride, padding="SAME")
        print(layer_name, prev_layer.get_shape())
        # biases = _bias_variable('biases', [out_filters])
        # prev_layer = tf.nn.bias_add(prev_layer, biases)
        # prev_layer = tf.nn.relu(prev_layer)

    return prev_layer


def inpuT_block(inpuT, inpuT_filter_shape, inpuT_stride):
    with tf.variable_scope("inpuT_layer"):
        out_filters = 3
        kernel_name = "weights"
        kernel = getKernel(kernel_name, inpuT_filter_shape)
        curr_layer = tf.nn.conv3d(inpuT, kernel, strides=inpuT_stride, padding="VALID")
        print("inpuT_layer", curr_layer.get_shape())

        curr_layer = tf.layers.batch_normalization(curr_layer, training=True)

        biases = _bias_variable('biases', [out_filters])
        print(biases.get_shape())
        curr_layer = tf.nn.bias_add(curr_layer, biases)
        curr_layer = tf.nn.relu(curr_layer)

    return curr_layer


def basic_block(inpuT, in_filters, out_filters, layer_num, add_kernel_name):
    layer_name = "conv{}a".format(layer_num)
    prev_layer = conv3DBlock(inpuT, layer_name, in_filters, out_filters)
    prev_layer = tf.layers.batch_normalization(prev_layer, training=True)
    prev_layer = tf.nn.relu(prev_layer)
    print(layer_name, prev_layer.get_shape())

    in_filters = out_filters

    layer_name = "conv{}b".format(layer_num)
    prev_layer = conv3DBlock(prev_layer, layer_name, in_filters, out_filters)
    prev_layer = tf.layers.batch_normalization(prev_layer, training=True)

    residual = prev_layer

    prev_layer = _shortcut3d(inpuT, residual, add_kernel_name)
    prev_layer = tf.nn.relu(prev_layer)

    print(layer_name, prev_layer.get_shape())

    return prev_layer


def _shortcut3d(inpuT, residual, add_kernel_name):
    """3D shortcut to match inpuT and residual and sum them."""
    conv_stride = [1, 1, 1, 1, 1]
    inpuT_shape = inpuT.get_shape().as_list()
    residual_shape = residual.get_shape().as_list()

    equal_channels = inpuT_shape[-1] == residual_shape[-1]
    shortcut = inpuT

    if not equal_channels:
        kernel = getKernel(add_kernel_name, [1, 1, 1, inpuT_shape[-1], residual_shape[-1]])
        shortcut = tf.nn.conv3d(shortcut, kernel, strides=conv_stride, padding="SAME")

    return (shortcut + residual)


def inference(inpuT):
    
    print(device_lib.list_local_devices())

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
    num_filters_list = [16, 32, 64, 128]  # as per the paper

    num_classes = 2
    num_channels = 3
    inpuT_filter_shape = [3, 7, 7, num_channels, num_channels]
    inpuT_stride = [1, 1, 2, 2, 1]

    prev_layer = inpuT_block(inpuT, inpuT_filter_shape, inpuT_stride)
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
        weights = getKernel('weights', [dim, 1024])
        biases = _bias_variable('biases', [1024])

        prev_layer = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)

    with tf.variable_scope('dropout1') as scope:
        prev_layer = tf.nn.dropout(prev_layer, 0.4)

    with tf.variable_scope('fc2') as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = getKernel('weights', [dim, 512])
        biases = _bias_variable('biases', [512])

        prev_layer = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)

    with tf.variable_scope('dropout2') as scope:
        prev_layer = tf.nn.dropout(prev_layer, 0.4)

    with tf.variable_scope('softmax_linear') as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])

        weights = getKernel('weights', [dim, num_classes])
        biases = _bias_variable('biases', [num_classes])

        softmax_linear = tf.add(tf.matmul(prev_layer_flat, weights), biases, name=scope.name)

        print(softmax_linear.get_shape())

    return softmax_linear
