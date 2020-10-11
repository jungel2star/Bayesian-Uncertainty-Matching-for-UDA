from collections import OrderedDict
from contextlib import ExitStack

import tensorflow as tf
from tensorflow.contrib import slim

from adda.models import register_model_fn
#from utilize import netvlad
import numpy as np


@register_model_fn('lenet_twostream')
def lenet_twostream(inputs_s,inputs_t, scope='lenet_twostream', is_training=True, reuse=False,batch_size=128,source_only=True):
    layers = OrderedDict()
    layers_t = OrderedDict()
    net_s = inputs_s
    net_t=inputs_t
    if source_only==True:
        t2s_weight=0
    else:
        t2s_weight=1

    with tf.variable_scope(scope, reuse=reuse):
        with ExitStack() as stack:
            stack.enter_context(
                slim.arg_scope(
                    [slim.fully_connected, slim.conv2d],
                    activation_fn=tf.nn.relu,
                    weights_regularizer=slim.l2_regularizer(2.5e-5)))

            s2t_weight_conv1 = np.random.normal(size=(32))
            s2t_weight_conv12 = np.random.normal(size=(1))
            s2t_weight_conv2 = np.random.normal(size=(64))
            s2t_weight_conv22 = np.random.normal(size=(1))

            s2t_weight_conv1_variable = slim.model_variable(
                's2t_weight_conv1_target',
                shape=s2t_weight_conv1.shape,
                initializer=tf.constant_initializer(s2t_weight_conv1))
            s2t_weight_conv12_variable = slim.model_variable(
                's2t_weight_conv12_target',
                shape=s2t_weight_conv12.shape,
                initializer=tf.constant_initializer(s2t_weight_conv12))
            s2t_weight_conv2_variable = slim.model_variable(
                's2t_weight_conv2_target',
                shape=s2t_weight_conv2.shape,
                initializer=tf.constant_initializer(s2t_weight_conv1))
            s2t_weight_conv22_variable = slim.model_variable(
                's2t_weight_conv22_target',
                shape=s2t_weight_conv22.shape,
                initializer=tf.constant_initializer(s2t_weight_conv22))


            #s2t_weight_conv1=(tf.nn.tanh(s2t_weight_conv1_variable+1.0))/2.0
            #s2t_weight_conv2=(tf.nn.tanh(s2t_weight_conv2_variable)+1.0)/2.0
            #s2t_weight_fcn=(tf.nn.tanh(s2t_weight_fcn_variable+1.0))/2.0

            #s2t_weight_conv1=tf.minimum(tf.maximum(s2t_weight_conv1_variable+0.0,1.0),1.0)/1.0
            #s2t_weight_conv2=tf.minimum(tf.maximum(s2t_weight_conv2_variable+0.0,1.0),1.0)/1.0
            #s2t_weight_fcn=tf.minimum(tf.maximum(s2t_weight_fcn_variable+0.0,1.0),1.0)/1.0

            s2t_weight_conv1=t2s_weight*s2t_weight_conv1_variable*0
            s2t_weight_conv12=t2s_weight*s2t_weight_conv12_variable*0
            s2t_weight_conv2=t2s_weight*s2t_weight_conv2_variable*0
            s2t_weight_conv22=t2s_weight*s2t_weight_conv22_variable*0


            stack.enter_context(slim.arg_scope([slim.conv2d], padding='VALID'))

            net_s_target=net_t
            net_s = slim.conv2d(net_s, 32, 5, scope='conv1_source')
            net_s_target = slim.conv2d(net_s_target, 32, 5, scope='conv1_source',reuse=True)
            net_t = slim.conv2d(net_t, 32, 5, scope='conv1_target')
            net_t=net_t*(1-s2t_weight_conv1)+net_s_target*s2t_weight_conv1


            net_s = slim.max_pool2d(net_s, 2, stride=2, scope='pool1_source')
            layers['pool1_s'] = net_s
            net_t = slim.max_pool2d(net_t, 2, stride=2, scope='pool1_target')
            layers_t['pool1_t'] = net_t

            net_s_target=net_t
            net_s = slim.conv2d(net_s, 64, 5, scope='conv2_source')
            net_s_target = slim.conv2d(net_s_target, 64, 5, scope='conv2_source',reuse=True)
            net_t = slim.conv2d(net_t, 64, 5, scope='conv2_target')
            net_t=net_t*(1-s2t_weight_conv2)+net_s_target*s2t_weight_conv2

            layers['conv2_s'] = net_s
            layers_t['conv2_t'] = net_t

            net_s = slim.max_pool2d(net_s, 2, stride=2, scope='pool2_source')
            layers['pool2_s'] = net_s
            net_t = slim.max_pool2d(net_t, 2, stride=2, scope='pool2_target')
            layers_t['pool2_t'] = net_t
            net_s = tf.contrib.layers.flatten(net_s)
            net_t = tf.contrib.layers.flatten(net_t)

            net_s = slim.fully_connected(net_s, 500, scope='fcnect_3_source')
            layers['fc3_s'] = net_s
            net_s_target=net_t
            net_s_target = slim.fully_connected(net_s_target, 500, scope='fcnect_3_source',reuse=True)
            net_t = slim.fully_connected(net_t, 500, scope='fcnect_3_target')
            net_t=net_t*(1-s2t_weight_conv22)+net_s_target*s2t_weight_conv22
            layers_t['fc3_t'] = net_t



            net_s = slim.fully_connected(net_s, 10,activation_fn=None, scope='fcnect_4_source')
            layers['fc4_s'] = net_s
            net_t = slim.fully_connected(net_t, 10,activation_fn=None, scope='fcnect_4_target')
            layers_t['fc4_t'] = net_t

    return net_s,net_t, layers,layers_t
lenet_twostream.default_image_size = 28
lenet_twostream.num_channels = 1
lenet_twostream.mean = None
lenet_twostream.bgr = False
