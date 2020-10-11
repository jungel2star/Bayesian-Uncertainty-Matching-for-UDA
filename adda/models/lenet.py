from collections import OrderedDict
from contextlib import ExitStack

import tensorflow as tf
from tensorflow.contrib import slim

from adda.models import register_model_fn
#from utilize import netvlad
import numpy as np


@register_model_fn('lenet')
def lenet(inputs, scope='lenet', is_training=True, reuse=False,batch_size=128):
    layers = OrderedDict()
    net = inputs
    with tf.variable_scope(scope, reuse=reuse):
        with ExitStack() as stack:
            stack.enter_context(
                slim.arg_scope(
                    [slim.fully_connected, slim.conv2d],
                    activation_fn=tf.nn.relu,
                    weights_regularizer=slim.l2_regularizer(2.5e-5)))
            stack.enter_context(slim.arg_scope([slim.conv2d,slim.max_pool2d]))
            net = slim.conv2d(net, 32, 5, scope='conv1', activation_fn=tf.nn.relu, padding='VALID')
            layers['conv1'] = net
            net = slim.max_pool2d(net, 2, stride=2, scope='pool1', padding='VALID')
            layers['pool1'] = net
            net = slim.conv2d(net, 48, 5, scope='conv2', activation_fn=tf.nn.relu, padding='VALID')
            layers['conv2'] = net
            net = slim.max_pool2d(net, 2, stride=2, scope='pool2', padding='VALID')

            net = tf.contrib.layers.flatten(net)
            layers['pooling_last'] = net
            #net = tf.nn.dropout(net, keep_prob=0.5)
            # net = slim.fully_connected(net, 100, scope='fc3')
            # layers['fc_1'] = net
            # net = slim.fully_connected(net, 100, scope='fc4')
            # layers['fc_2'] = net


            #net = tf.nn.dropout(net, keep_prob=0.5)
            #layers['fc3'] = net
            #net = slim.fully_connected(net, 10, activation_fn=None, scope='fc4')
            #layers['fc4'] = net

    return net, layers
lenet.default_image_size = 28
lenet.num_channels = 1
lenet.mean = None
lenet.bgr = False




# net = slim.conv2d(net, 64, 5, scope='conv1', padding="SAME")
#             net = slim.max_pool2d(net, 3, stride=2, scope='pool1', padding='VALID')
#             net = slim.conv2d(net, 64, 5, scope='conv2', padding="SAME")
#             net = slim.max_pool2d(net, 3, stride=2, scope='pool2', padding='VALID')
#             net = slim.conv2d(net, 128, 5, scope='conv3', padding='VALID')
#             layers['conv3'] = net
#             net = tf.contrib.layers.flatten(net)
#             net = slim.fully_connected(net, 256, scope='fc3')
#             layers['fc3'] = net
#             net = slim.fully_connected(net, 10, activation_fn=None, scope='fc4')
#             layers['fc4'] = net