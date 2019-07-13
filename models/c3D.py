import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib
import os

print(device_lib.list_local_devices())

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

FC_SIZE = 2048
DTYPE = tf.float32
num_classes = 249
batch_size = 32

stride = 1
pool_stride = [1, 1, 1, 1, 1]
conv_stride = [1, 1, 1, 1, 1]
kernel_shape = 3
kernel_name = "weights"


def getKernel(name, shape):
    kernel = tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=0.1))
    return kernel

def _bias_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(0.1, dtype=DTYPE))


# block 1

def conv3DBlock(prev_layer, layer_name, in_filters, out_filters):
    with tf.variable_scope(layer_name):

        kernel = getKernel(kernel_name, [kernel_shape, kernel_shape, kernel_shape, in_filters, out_filters])
        prev_layer = tf.nn.conv3d(prev_layer, kernel, strides=conv_stride, padding="SAME")
        biases = _bias_variable('biases', [out_filters])
        prev_layer = tf.nn.bias_add(prev_layer, biases)
        prev_layer = tf.nn.relu(prev_layer)

    return prev_layer

def maxPool3DBlock(prev_layer, ksize):
    prev_layer = tf.nn.max_pool3d(prev_layer, ksize, pool_stride, padding="SAME")

    return prev_layer


with tf.name_scope('inputs'):
    x_input = tf.placeholder(tf.float32, shape=[None, 16, 16, 16, 3])
    y_input = tf.placeholder(tf.float32, shape=[None, num_classes])

def inference(video):
    prev_layer = video
    in_filters = 3
    print(prev_layer.get_shape())


    # block 1
    out_filters = 64
    layer_name = "conv1a"
    prev_layer = conv3DBlock(prev_layer, layer_name, in_filters, out_filters)
    prev_layer = maxPool3DBlock(prev_layer, [1, 1, 2, 2, 1])
    in_filters = out_filters
    print(prev_layer.get_shape())

    # block 2
    out_filters = 128
    layer_name = "conv2a"
    prev_layer = conv3DBlock(prev_layer, layer_name, in_filters, out_filters)
    prev_layer = maxPool3DBlock(prev_layer, [1, 2, 2, 2, 1])
    in_filters = out_filters
    print(prev_layer.get_shape())

    # block 3
    out_filters = 256
    layer_name = "conv3a"
    prev_layer = conv3DBlock(prev_layer, layer_name, in_filters, out_filters)
    in_filters = out_filters
    print(prev_layer.get_shape())

    out_filters = 256
    layer_name = "conv3b"
    prev_layer = conv3DBlock(prev_layer, layer_name, in_filters, out_filters)
    print(prev_layer.get_shape())

    prev_layer = maxPool3DBlock(prev_layer, [1, 2, 2, 2, 1])
    in_filters = out_filters
    print(prev_layer.get_shape())

    # block 4
    out_filters = 512
    layer_name = "conv4a"
    prev_layer = conv3DBlock(prev_layer, layer_name, in_filters, out_filters)
    in_filters = out_filters
    print(prev_layer.get_shape())

    out_filters = 512
    layer_name = "conv4b"
    prev_layer = conv3DBlock(prev_layer, layer_name, in_filters, out_filters)
    print(prev_layer.get_shape())

    prev_layer = maxPool3DBlock(prev_layer, [1, 2, 2, 2, 1])
    in_filters = out_filters
    print(prev_layer.get_shape())

    # block 5
    out_filters = 512
    layer_name = "conv5a"
    prev_layer = conv3DBlock(prev_layer, layer_name, in_filters, out_filters)
    in_filters = out_filters
    print(prev_layer.get_shape())

    out_filters = 512
    layer_name = "conv5b"
    prev_layer = conv3DBlock(prev_layer, layer_name, in_filters, out_filters)
    print(prev_layer.get_shape())

    prev_layer = maxPool3DBlock(prev_layer, [1, 2, 2, 2, 1])
    in_filters = out_filters
    print(prev_layer.get_shape())

    # with tf.variable_scope('fc1') as scope:
    #
    #     dim = np.prod(prev_layer.get_shape().as_list()[1:])
    #     prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
    #     weights = getKernel('weights', [dim, FC_SIZE])
    #     biases = _bias_variable('biases', [FC_SIZE])
    #
    #     fc1 = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)
    #
    #
    # prev_layer = fc1
    #
    # with tf.variable_scope('fc2') as scope:
    #
    #     dim = np.prod(prev_layer.get_shape().as_list()[1:])
    #     prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
    #     weights = getKernel('weights', [dim, FC_SIZE])
    #     biases = _bias_variable('biases', [FC_SIZE])
    #     fc2 = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)
    #
    # prev_layer = fc2

    with tf.variable_scope('softmax_linear') as scope:

        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = getKernel('weights', [dim, num_classes])
        biases = _bias_variable('biases', [num_classes])
        softmax_linear = tf.add(tf.matmul(prev_layer_flat, weights), biases, name=scope.name)
        print(softmax_linear.get_shape())



    return softmax_linear


def train_neural_network(x_input, y_input, learning_rate=0.05, keep_rate=0.7,
                         epochs=10):

    with tf.name_scope("cross_entropy"):
        prediction = inference(x_input)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_input))

    with tf.name_scope("training"):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

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
        mini_batch_x = np.ones((batch_size, 16, 16, 16, 3))  # testing
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





