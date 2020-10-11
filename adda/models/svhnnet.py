from contextlib import ExitStack
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from adda.models import register_model_fn


@register_model_fn('svhnnet')
def svhnnet(inputs, scope='svhnnet', is_training=True, reuse=False):
    layers = OrderedDict()
    net = inputs
    with tf.variable_scope(scope, reuse=reuse):
        with ExitStack() as stack:
            stack.enter_context(
                slim.arg_scope(
                    [slim.fully_connected, slim.conv2d],
                    activation_fn=tf.nn.relu,
                    weights_regularizer=slim.l2_regularizer(2.5e-5)))
            stack.enter_context(
                slim.arg_scope([slim.max_pool2d, slim.conv2d],
                               padding='SAME'))
            net = slim.conv2d(net, 64, 5, scope='conv1',activation_fn=None)
            #net = tf.layers.batch_normalization(net, training=is_training, name='bn1',momentum=0.98)
            net=tf.nn.relu(net)
            #net = tf.nn.dropout(net, keep_prob=0.5)
            net = slim.max_pool2d(net, 3, stride=2, scope='pool1')
            layers['pool1'] = net
            net = slim.conv2d(net, 64, 5, scope='conv2',activation_fn=None)
            #net = tf.layers.batch_normalization(net, training=is_training, name='bn2',momentum=0.98)
            net = tf.nn.relu(net)
            #net = tf.nn.dropout(net, keep_prob=0.5)
            net = slim.max_pool2d(net, 3, stride=2, scope='pool2')
            layers['pool2'] = net
            net = slim.conv2d(net, 128, 5, scope='conv3',activation_fn=None)
            #net = tf.layers.batch_normalization(net, training=is_training, name='bn3',momentum=0.98)
            net = tf.nn.relu(net)
            #net = tf.nn.dropout(net, keep_prob=0.5)
            net = tf.contrib.layers.flatten(net)
            layers['pooling_last'] = net

            # net = slim.fully_connected(net, 3072, scope='fc4',activation_fn=None)
            # net = tf.layers.batch_normalization(net, training=is_training, name='bn4',momentum=0.95)
            # net = tf.nn.relu(net)
            # layers['fc_1'] = net
            # net = tf.nn.dropout(net, keep_prob=0.5)
            # net = slim.fully_connected(net, 2048, scope='fc5',activation_fn=None)
            # net = tf.layers.batch_normalization(net, training=is_training, name='bn5',momentum=0.95)
            # net = tf.nn.relu(net)
            # layers['fc_2'] = net

            #net = tf.nn.dropout(net, keep_prob=0.5)
            #layers['fc5'] = net
            #net = slim.fully_connected(net, 10, activation_fn=None, scope='fc6')
            #layers['fc6'] = net
    return net,layers
svhnnet.default_image_size = 32
svhnnet.num_channels = 1
svhnnet.range = 255
svhnnet.mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
svhnnet.bgr = False
