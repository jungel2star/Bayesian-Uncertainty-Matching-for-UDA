from collections import OrderedDict
from contextlib import ExitStack

import tensorflow as tf
from tensorflow.contrib import slim

from adda.models import register_model_fn
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_v2
#from utilize import netvlad
import numpy as np
#from resnetv1 import resnet50v1


@register_model_fn('Resnet')
def Resnet(inputs, scope='Resnet',class_number=65, is_training=True, reuse=False,batch_size=128,dropout_ratio=1.0):
    #with tf.variable_scope(scope,reuse=reuse):
    #net,end_points=resnet50v1.resnet_v1_50(inputs,is_training=is_training,scope=scope,reuse=reuse)
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, end_points = resnet_v2.resnet_v2_50(inputs, is_training=is_training,reuse=reuse)
    net = tf.squeeze(net, [1, 2])
    return net

Resnet.default_image_size = 224  # fully convolutional
Resnet.num_channels = 3
Resnet.mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
Resnet.bgr = False
