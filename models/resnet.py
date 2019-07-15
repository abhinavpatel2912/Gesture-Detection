import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

# input_video = [batch_size, depth, height, width, num_channels]

print(device_lib.list_local_devices())

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

mean = 0.0
variance = 0.0
offset = 0.0
scale = None
variance_epsilon = 0.0

height = 112
width = 112
depth = 24


DTYPE = tf.float32
batch_size = 32
kernel_shape = 3

num_classes = 249
num_channels = 3
input_filter_shape = [3, 7, 7, num_channels, num_channels]
input_stride = [1, 1, 2, 2, 1]
conv_stride = [1, 1, 1, 1, 1]
kernel_name = "weights"
num_filters_list = [16, 32, 64, 128] # as per the paper

with tf.device('/device:GPU:0'):
    x_input = tf.placeholder(tf.float32, shape=[None, depth, height, width, 3])
    y_input = tf.placeholder(tf.float32, shape=[None, num_classes])

def getKernel(name, shape):
    kernel = tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=0.1))
    return kernel

def _bias_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(0.1, dtype=DTYPE))

def conv3DBlock(prev_layer, layer_name, in_filters, out_filters):
    with tf.variable_scope(layer_name):

        kernel = getKernel(kernel_name, [kernel_shape, kernel_shape, kernel_shape, in_filters, out_filters])
        prev_layer = tf.nn.conv3d(prev_layer, kernel, strides=conv_stride, padding="SAME")
        print(layer_name, prev_layer.get_shape())
        # biases = _bias_variable('biases', [out_filters])
        # prev_layer = tf.nn.bias_add(prev_layer, biases)
        # prev_layer = tf.nn.relu(prev_layer)

    return prev_layer

def input_block(input, input_filter_shape, input_stride):

    with tf.variable_scope("input_layer"):
        out_filters = 3
        kernel = getKernel(kernel_name, input_filter_shape)
        curr_layer = tf.nn.conv3d(input, kernel, strides=input_stride, padding="VALID")
        print("input_layer", curr_layer.get_shape())
        curr_layer = tf.nn.batch_normalization(curr_layer, mean,
                                                variance,
                                                offset,
                                                scale,
                                                variance_epsilon,
                                                name=None)

        biases = _bias_variable('biases', [out_filters])
        print(biases.get_shape())
        curr_layer = tf.nn.bias_add(curr_layer, biases)
        curr_layer = tf.nn.relu(curr_layer)


    return curr_layer

def basic_block(input, in_filters, out_filters, layer_num, add_kernel_name):

    layer_name = "conv{}a".format(layer_num)
    prev_layer = conv3DBlock(input, layer_name, in_filters, out_filters)
    prev_layer = tf.nn.batch_normalization(prev_layer, mean, variance, offset, scale, variance_epsilon, name=None)
    prev_layer = tf.nn.relu(prev_layer)
    print(layer_name, prev_layer.get_shape())

    in_filters = out_filters

    layer_name = "conv{}b".format(layer_num)
    prev_layer = conv3DBlock(prev_layer, layer_name, in_filters, out_filters)
    prev_layer = tf.nn.batch_normalization(prev_layer, mean, variance, offset, scale, variance_epsilon, name=None)

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


    #
    # stride_dim1 = input._keras_shape[DIM1_AXIS] // residual._keras_shape[DIM1_AXIS]
    #
    # stride_dim2 = input._keras_shape[DIM2_AXIS] \
    #     // residual._keras_shape[DIM2_AXIS]
    # stride_dim3 = input._keras_shape[DIM3_AXIS] \
    #     // residual._keras_shape[DIM3_AXIS]
    # equal_channels = residual._keras_shape[CHANNEL_AXIS] \
    #     == input._keras_shape[CHANNEL_AXIS]
    #
    # shortcut = input
    # if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 \
    #         or not equal_channels:
    #     shortcut = Conv3D(
    #         filters=residual._keras_shape[CHANNEL_AXIS],
    #         kernel_size=(1, 1, 1),
    #         strides=(stride_dim1, stride_dim2, stride_dim3),
    #         kernel_initializer="he_normal", padding="valid",
    #         kernel_regularizer=l2(1e-4)
    #         )(input)

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

    with tf.variable_scope('softmax_linear') as scope:

        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])

        weights = getKernel('weights', [dim, num_classes])
        biases = _bias_variable('biases', [num_classes])

        softmax_linear = tf.add(tf.matmul(prev_layer_flat, weights), biases, name=scope.name)
        print(prev_layer.get_shape())

    return softmax_linear


def train_neural_network(x_input, y_input, learning_rate=0.05, keep_rate=0.7, epochs=10):

    with tf.name_scope("cross_entropy"):

        prediction = inference(x_input)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_input))

    with tf.name_scope("training"):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    # iterations = int(len(x_train_data) / batch_size) + 1

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        sess.run(tf.global_variables_initializer())
        import datetime

        start_time = datetime.datetime.now()

        # iterations = int(len(x_train_data) / batch_size) + 1
        # run epochs

        epoch_loss = 0
        print("session starts!")
        mini_batch_x = np.ones((batch_size, depth, height, width, 3))  # testing
        mini_batch_y = np.ones((batch_size, num_classes))  # testing

        _optimizer, _cost = sess.run([optimizer, cost],
                                     feed_dict={x_input: mini_batch_x, y_input: mini_batch_y})
        epoch_loss += _cost

        print("Done!")

        # for epoch in range(epochs):
        #
        #     start_time_epoch = datetime.datetime.now()
        #     print("Epoch {} started!".format(epoch))
        #     epoch_loss = 0
        #     # mini batch
        #     for itr in range(iterations):
        #
        #         mini_batch_x = x_train_data[itr * batch_size: (itr + 1) * batch_size]
        #         mini_batch_y = y_train_data[itr * batch_size: (itr + 1) * batch_size]
        #         _optimizer, _cost = sess.run([optimizer, cost],
        #                                      feed_dict={x_input: mini_batch_x, y_input: mini_batch_y})
        #         epoch_loss += _cost
        #
        #     #  using mini batch in case not enough memory
        #     acc = 0
        #     itrs = int(len(x_test_data) / batch_size) + 1
        #     for itr in range(itrs):
        #         mini_batch_x_test = x_test_data[itr * batch_size: (itr + 1) * batch_size]
        #         mini_batch_y_test = y_test_data[itr * batch_size: (itr + 1) * batch_size]
        #         acc += sess.run(accuracy, feed_dict={x_input: mini_batch_x_test, y_input: mini_batch_y_test})
        #
        #     end_time_epoch = datetime.datetime.now()
        #     print(' Testing Set Accuracy:', acc / itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))
        #
        # end_time = datetime.datetime.now()
        # print('Time elapse: ', str(end_time - start_time))



train_neural_network(x_input, y_input)




