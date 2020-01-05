import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.python.client import device_lib

weight_decay = 0.0005
momentum = 0.9

init_learning_rate = 0.001
cardinality = 8  # how many split ?
blocks = 3  # res_block ! (split + transition)

"""
So, the total number of layers is (3*blokcs)*residual_layer_num + 2
because, blocks = split(conv 2) + transition(conv 1) = 3 layer
and, first conv layer 1, last dense layer 1
thus, total number of layers = (3*blocks)*residual_layer_num + 2
"""

depth = 64  # out channel

image_size = 112
image_depth = 16
num_channels = 3
num_classes = 249


def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv3d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,
                                   padding=padding)
        return network


def Global_Average_Pooling(x):
    x_shape = x.get_shape().as_list()
    depth = x_shape[1]
    height = x_shape[2]
    width = x_shape[3]
    return tf.layers.average_pooling3d(inputs=x, pool_size=[depth, height, width], strides=[1, 1, 1], padding='valid')


def Max_pooling(x, pool_size=[2, 2, 2], stride=2, padding='SAME'):
    return tf.layers.max_pooling3d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def Relu(x):
    return tf.nn.relu(x)


def Concatenation(layers):
    return tf.concat(layers, axis=4)


def Linear(x):
    return tf.layers.dense(inputs=x, use_bias=False, units=num_classes, name='linear' + str(random.random()))


def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    add = 1000

    for it in range(test_iteration):
        test_batch_x = test_x[test_pre_index: test_pre_index + add]
        test_batch_y = test_y[test_pre_index: test_pre_index + add]
        test_pre_index = test_pre_index + add

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_
        test_acc += acc_

    test_loss /= test_iteration  # average loss
    test_acc /= test_iteration  # average accuracy

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary


class ResNeXt(object):
    def __init__(self, x, training):
        self.training = training
        self.model = self.Build_ResNext(x)

    def first_layer(self, x, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=64, kernel=[7, 7, 7], stride=[1, 2, 2], padding='SAME',
                           layer_name=scope + '_conv1' + str(random.random()))
            x = Batch_Normalization(x, training=self.training,
                                    scope=scope + '_batch1{}'.format(random.randint(10, 100)))
            x = Max_pooling(x, pool_size=[3, 3, 3], stride=2, padding='SAME')
            x = Relu(x)

            return x

    def transform_layer(self, x, stride, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=depth, kernel=[1, 1, 1], stride=stride,
                           layer_name=scope + '_conv1' + str(random.random()))
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1'.format(random.randint(20, 200)))
            x = Relu(x)

            x = conv_layer(x, filter=depth, kernel=[3, 3, 3], stride=1,
                           layer_name=scope + '_conv2' + str(random.random()))
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch2'.format(random.randint(30, 300)))
            x = Relu(x)
            return x

    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=out_dim, kernel=[1, 1, 1], stride=1,
                           layer_name=scope + '_conv1' + str(random.random()))
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1'.format(random.randint(40, 400)))
            # x = Relu(x)

            return x

    def split_layer(self, input_x, stride, layer_name):
        with tf.name_scope(layer_name):
            layers_split = list()
            for i in range(cardinality):
                splits = self.transform_layer(input_x, stride=stride,
                                              scope=layer_name + '_splitN_' + str(i) + str(random.random()))
                layers_split.append(splits)

            return Concatenation(layers_split)

    def residual_layer(self, input_x, out_dim, layer_num, res_block=blocks):
        # split + transform(bottleneck) + transition + merge
        for i in range(res_block):
            # input_dim = input_x.get_shape().as_list()[-1]
            input_dim = int(np.shape(input_x)[-1])

            channel = None
            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = int(input_dim / 2)

            else:
                flag = False
                stride = 1

            x = self.split_layer(input_x, stride=stride,
                                 layer_name='split_layer_' + layer_num + '_' + str(i) + str(random.random()))
            x = self.transition_layer(x, out_dim=out_dim,
                                      scope='trans_layer_' + layer_num + '_' + str(i) + str(random.random()))

            if flag is True:
                pad_input_x = Max_pooling(input_x)
                pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [0, 0], [channel, channel]])
            else:

                x = conv_layer(input_x, filter=out_dim, kernel=[1, 1, 1], stride=[2, 2, 2],
                               layer_name='conv_residual' + str(random.random()))
                x = Batch_Normalization(x, training=self.training, scope='batch_residual{}'.format(random.random()))
                pad_input_x = x

            input_x = Relu(x + pad_input_x)
            print(input_x.get_shape())

        return input_x

    def Build_ResNext(self, input_x):
        # only cifar10 architecture
        input_x = self.first_layer(input_x, scope='first_layer')
        print(input_x.get_shape())
        x = self.residual_layer(input_x, out_dim=256, layer_num='1', res_block=3)
        x = self.residual_layer(x, out_dim=512, layer_num='2', res_block=4)
        x = self.residual_layer(x, out_dim=1024, layer_num='3', res_block=6)
        x = self.residual_layer(x, out_dim=2048, layer_num='4', res_block=3)

        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)

        # x = tf.reshape(x, [-1,10])
        return x

# train_x, train_y, test_x, test_y = prepare_data()
# train_x, test_x = color_preprocessing(train_x, test_x)


# image_size = 32, img_channels = 3, class_num = 10 in cifar10
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
# print(device_lib.list_local_devices())
#
# choose_device = "/device:GPU:0"
# with tf.device(choose_device):
#
#     x = tf.placeholder(tf.float32, shape=[None, image_depth, image_size, image_size, num_channels])
#     label = tf.placeholder(tf.float32, shape=[None, num_classes])
#
#     training_flag = tf.placeholder(tf.bool)
#
#     learning_rate = tf.placeholder(tf.float32, name='learning_rate')
#
#     with tf.name_scope("cross_entropy"):
#         logits = ResNeXt(x, training=training_flag).model
#         cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
#         l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
#
#     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#     with tf.control_dependencies(update_ops):
#         optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
#         train = optimizer.minimize(cost + l2_loss * weight_decay)
#
#     with tf.name_scope("accuracy"):
#         correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#     # saver = tf.train.Saver(tf.global_variables())
#
#     gpu_options = tf.GPUOptions(allow_growth=True)
#     with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#         # ckpt = tf.train.get_checkpoint_state('./model')
#         ckpt = 0
#         if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
#             saver.restore(sess, ckpt.model_checkpoint_path)
#         else:
#             sess.run(tf.global_variables_initializer())
#
#         summary_writer = tf.summary.FileWriter('./../graphs', sess.graph)
#
#         epoch_learning_rate = init_learning_rate
#         print("Session started.")
#         for epoch in range(1, total_epochs + 1):
#             if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
#                 epoch_learning_rate = epoch_learning_rate / 10
#
#             pre_index = 0
#             train_acc = 0.0
#             train_loss = 0.0
#             iteration = 10
#             for step in range(1, iteration + 1):
#                 # if pre_index + batch_size < 50000:
#                 #     batch_x = train_x[pre_index: pre_index + batch_size]
#                 #     batch_y = train_y[pre_index: pre_index + batch_size]
#                 # else:
#                 #     batch_x = train_x[pre_index:]
#                 #     batch_y = train_y[pre_index:]
#                 print("{} iter started.".format(step))
#                 batch_x = np.ones((batch_size, image_depth, image_size, image_size, num_channels))
#                 batch_y = np.zeros((batch_size, num_classes))
#                 # batch_x = data_augmentation(batch_x)
#
#                 train_feed_dict = {
#                     x: batch_x,
#                     label: batch_y,
#                     learning_rate: epoch_learning_rate,
#                     training_flag: True
#                 }
#
#                 _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
#                 batch_acc = accuracy.eval(feed_dict=train_feed_dict)
#
#                 train_loss += batch_loss
#                 train_acc += batch_acc
#                 pre_index += batch_size
#
#                 print("{}-{}, epoch: {}, iter: {} done.".format(batch_loss, batch_acc, epoch, iteration))
#
#
#             train_loss /= iteration # average loss
#             train_acc /= iteration # average accuracy
#
#             train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
#                                               tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])
#
#             # test_acc, test_loss, test_summary = Evaluate(sess)
#
#             summary_writer.add_summary(summary=train_summary, global_step=epoch)
#             # summary_writer.add_summary(summary=test_summary, global_step=epoch)
#             summary_writer.flush()
#
#             line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f \n" % (epoch, total_epochs, train_loss, train_acc)
#             print(line)
#
#             # with open('logs.txt', 'a') as f:
#             #     f.write(line)
#
#             # saver.save(sess=sess, save_path='./model/ResNeXt.ckpt')
