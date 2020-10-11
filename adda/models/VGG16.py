from collections import OrderedDict
from contextlib import ExitStack

import tensorflow as tf
from tensorflow.contrib import slim

from adda.models import register_model_fn
#from utilize import netvlad
import numpy as np


@register_model_fn('VGG16')
def VGG16(inputs, scope='VGG16',class_number=65, is_training=True, reuse=False,batch_size=128,dropout_ratio=1.0):
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
        layers['conv5_1'] = net
        net = slim.conv2d(net, 512, [3, 3], scope='conv5/conv5_2')
        layers['conv5_2'] = net
        net = slim.conv2d(net, 512, [3, 3], scope='conv5/conv5_3')
        layers['conv5_3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        layers['pool5']=net
        net = slim.conv2d(net, 256, [7, 7], padding='VALID', scope='fc6')
        layers['fc6'] = net
        net = slim.conv2d(net, class_number, [1, 1], scope='fc7', activation_fn=None)
        net = tf.squeeze(net, [1, 2])
        # net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        # net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        # net = slim.conv2d(
        #     net,
        #     65, [1, 1],
        #     activation_fn=None,
        #     normalizer_fn=None,
        #     scope='fc8')
        # net = tf.squeeze(net, [1, 2])
    return net, layers

VGG16.default_image_size = 224  # fully convolutional
VGG16.num_channels = 3
VGG16.mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
VGG16.bgr = False
