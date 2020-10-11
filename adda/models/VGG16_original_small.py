from collections import OrderedDict
from contextlib import ExitStack

import tensorflow as tf
from tensorflow.contrib import slim

from adda.models import register_model_fn
#from utilize import netvlad
import numpy as np


@register_model_fn('VGG16_original_small')
def VGG16_original_small(inputs, scope='VGG16_original_small',class_number=65, is_training=True, reuse=False,batch_size=128,dropout_ratio=1.0):
    layers = OrderedDict()
    net = inputs
    with tf.variable_scope(scope, reuse=reuse):
        net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.conv2d(net, 512, [3, 3], scope='conv5/conv5_1')
        net = slim.conv2d(net, 512, [3, 3], scope='conv5/conv5_2')
        net = slim.conv2d(net, 512, [3, 3], scope='conv5/conv5_3')
        layers['conv5'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        layers['pool5']=net
        net = slim.conv2d(net, 1024, [7, 7], padding='VALID', scope='fc6')
        layers['fc6'] = tf.squeeze(net, [1, 2])
        net=tf.nn.dropout(net,keep_prob=dropout_ratio)
        net = slim.conv2d(net, 128, [1, 1], scope='fc7')
        layers['fc7'] = tf.squeeze(net, [1, 2])
        net = slim.conv2d(
            net,
            class_number, [1, 1],
            activation_fn=None,
            normalizer_fn=None,
            scope='fc8')
        net = tf.squeeze(net, [1, 2])
        layers['fc8'] = net
    return net, layers

VGG16_original_small.default_image_size = 224  # fully convolutional
VGG16_original_small.num_channels = 3
VGG16_original_small.mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
VGG16_original_small.bgr = False
