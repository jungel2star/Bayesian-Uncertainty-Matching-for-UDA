from contextlib import ExitStack

import tensorflow as tf
#import tflearn
from tensorflow.contrib import slim
from tensorflow.python.framework import ops

global_step_grl = 0
class FlipGradientBuilder(object):
    def __init__(self):
        pass

    def __call__(self, x, l):
        global global_step_grl
        grad_name = "FlipGradient{:d}".format(global_step_grl)
        global_step_grl += 1

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
        return y

def adversarial_discriminator(net, layers, lamda_w,scope='adversary', leaky=False,reuse=False,dropout_keep=1.0):
    flip_gradient = FlipGradientBuilder()
    net = flip_gradient(net, lamda_w)

    def LeakyReLU(x, alpha=0.2):
        return tf.maximum(alpha*x, x)
    if leaky:
        activation_fn = LeakyReLU
    else:
        activation_fn = tf.nn.relu

    with tf.variable_scope(scope) as scope:
        if reuse:
                scope.reuse_variables()
        with ExitStack() as stack:
            stack.enter_context(tf.variable_scope(scope))
            stack.enter_context(
                slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=activation_fn,
                    weights_regularizer=slim.l2_regularizer(2.5e-5)))
            net = slim.fully_connected(net,num_outputs=layers[0], activation_fn=activation_fn)
            net = tf.nn.dropout(net, keep_prob=dropout_keep)
            layer2 = slim.fully_connected(net, num_outputs=layers[1], activation_fn=activation_fn,scope="adver_layer2")
            net = tf.nn.dropout(layer2, keep_prob=dropout_keep)

            net_1 = slim.fully_connected(net, 1, activation_fn=None, scope="adver_layer3")
            net_2 = slim.fully_connected(net, 2, activation_fn=None, scope="adver_layer4")
    return net_1,net_2


def adversarial_discriminator_compare_save(net, layers, lamda_w,scope='adversary', leaky=False,reuse=False,dropout_keep=1.0):
    def LeakyReLU(x, alpha=0.2):
        return tf.maximum(alpha*x, x)
    if leaky:
        activation_fn = LeakyReLU
    else:
        activation_fn = tf.nn.relu

    with tf.variable_scope(scope) as scope:
        if reuse:
                scope.reuse_variables()
        with ExitStack() as stack:
            stack.enter_context(tf.variable_scope(scope))
            stack.enter_context(
                slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=activation_fn,
                    weights_regularizer=slim.l2_regularizer(2.5e-5)))

            #net = tf.nn.dropout(net, keep_prob=0.5)

            layer1_compare = slim.fully_connected(net, num_outputs=200, activation_fn=activation_fn,
                                                  scope="compare_layer1")
            #layer1_compare = tf.nn.dropout(layer1_compare, keep_prob=dropout_keep)
            layer2_compare = slim.fully_connected(layer1_compare, num_outputs=layers[1], activation_fn=activation_fn,
                                          scope="compare_layer2")
            layer2_compare = tf.nn.dropout(layer2_compare, keep_prob=dropout_keep)
            net_c = slim.fully_connected(net, 1, activation_fn=None, scope="compare_layer3")


            layer1 = slim.fully_connected(net,num_outputs=layers[0], activation_fn=activation_fn, scope="adver_layer1")
            layer1 = tf.nn.dropout(layer1, keep_prob=dropout_keep)
            layer2 = slim.fully_connected(layer1, num_outputs=layers[1], activation_fn=activation_fn,scope="adver_layer2")
            layer2 = tf.nn.dropout(layer2, keep_prob=dropout_keep)
            net_1 = slim.fully_connected(layer2, 1, activation_fn=None, scope="adver_layer3")

    return net_1,net_c


def adversarial_discriminator_compare_save(net, layers, lamda_w,scope='adversary', leaky=False,reuse=False,dropout_keep=1.0):
    def LeakyReLU(x, alpha=0.2):
        return tf.maximum(alpha*x, x)
    if leaky:
        activation_fn = LeakyReLU
    else:
        activation_fn = tf.nn.relu

    with tf.variable_scope(scope) as scope:
        if reuse:
                scope.reuse_variables()
        with ExitStack() as stack:
            stack.enter_context(tf.variable_scope(scope))
            stack.enter_context(
                slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=activation_fn,
                    weights_regularizer=slim.l2_regularizer(2.5e-5)))
            net = slim.fully_connected(net,num_outputs=layers[0], activation_fn=activation_fn)
            net = tf.nn.dropout(net, keep_prob=dropout_keep)
            layer2 = slim.fully_connected(net, num_outputs=layers[1], activation_fn=activation_fn,scope="adver_layer2")
            net = tf.nn.dropout(layer2, keep_prob=dropout_keep)
            net_1 = slim.fully_connected(net, 1, activation_fn=None, scope="adver_layer3")
            net_c = slim.fully_connected(net, 1, activation_fn=None, scope="compare_layer3")
    return net_1,net_c


def adversarial_discriminator_ralation(net, layers,lamda_w,scope='adversary', neighbor_num=1, leaky=False,reuse=False,dropout_keep=1.0):
    flip_gradient = FlipGradientBuilder()
    net = flip_gradient(net, lamda_w)

    def LeakyReLU(x, alpha=0.2):
        return tf.maximum(alpha*x, x)
    if leaky:
        activation_fn = LeakyReLU
    else:
        activation_fn = tf.nn.relu

    with tf.variable_scope(scope) as scope:
        if reuse:
                scope.reuse_variables()
        with ExitStack() as stack:
            stack.enter_context(tf.variable_scope(scope))
            stack.enter_context(
                slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=activation_fn,
                    weights_regularizer=slim.l2_regularizer(2.5e-5)))

            layer1=[]
            for neighbori in range(neighbor_num-1):
                if neighbori>0:
                    reuse_temp=True
                else:
                    reuse_temp=False
                layer_temp=slim.fully_connected(net[:,neighbori],num_outputs=layers[0], activation_fn=activation_fn, scope="adver_layer1",reuse=reuse_temp)
                layer_temp = tf.nn.dropout(layer_temp, keep_prob=dropout_keep)
                layer_temp = slim.fully_connected(layer_temp, num_outputs=layers[0], activation_fn=activation_fn,
                                                  scope="adver_layer1_2", reuse=reuse_temp)
                layer1.append(layer_temp)
            net=tf.reduce_sum(layer1,axis=0)
            print ("gan_layer1:",net)
            net = tf.nn.dropout(net, keep_prob=dropout_keep)
            net = slim.fully_connected(net,num_outputs=layers[1], activation_fn=activation_fn,scope="adver_layer2")
            #net = tf.nn.dropout(net, keep_prob=dropout_keep)
            net_1 = slim.fully_connected(net, 1, activation_fn=None, scope="adver_layer3")
            net_2 = slim.fully_connected(net, 2, activation_fn=None, scope="adver_layer4")
    return net_1,net_2






def adversarial_discriminator_relation_triple(net, layers,lamda_w,scope='adversary', neighbor_num=1, leaky=False,reuse=False,dropout_keep=1.0):
    flip_gradient = FlipGradientBuilder()
    source_ft_gan, source_ft_gan_differ1, source_ft_gan_differ2 = net

    source_ft_gan = flip_gradient(source_ft_gan, lamda_w)
    source_ft_gan_differ1 = flip_gradient(source_ft_gan_differ1, lamda_w)
    source_ft_gan_differ2 = flip_gradient(source_ft_gan_differ2, lamda_w)

    def LeakyReLU(x, alpha=0.2):
        return tf.maximum(alpha*x, x)
    if leaky:
        activation_fn = LeakyReLU
    else:
        activation_fn = tf.nn.relu


    with tf.variable_scope(scope) as scope:
        if reuse:
                scope.reuse_variables()
        with ExitStack() as stack:
            stack.enter_context(tf.variable_scope(scope))
            stack.enter_context(
                slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=activation_fn,
                    weights_regularizer=slim.l2_regularizer(2.5e-5)))

            layer_s_s_differ1 = slim.fully_connected(tf.concat([source_ft_gan,source_ft_gan_differ1],axis=1),
                                                     num_outputs=int(layers[0]/3),
                                              activation_fn=activation_fn,
                                              scope="adver_layer1_1")
            layer_s_s_differ1 = tf.nn.dropout(layer_s_s_differ1, keep_prob=0.9)
            layer_s_s_differ1 = slim.fully_connected(layer_s_s_differ1,
                                                     num_outputs=int(layers[0] / 3),
                                                     activation_fn=activation_fn,
                                                     scope="adver_layer1_2")
            layer_s_s_differ1 = tf.nn.dropout(layer_s_s_differ1, keep_prob=0.9)

            layer_s_s_differ2 = slim.fully_connected(tf.concat([source_ft_gan, source_ft_gan_differ2], axis=1),
                                                     num_outputs=int(layers[0]/3 ),
                                                     activation_fn=activation_fn,
                                                     scope="adver_layer1_12")
            layer_s_s_differ2 = tf.nn.dropout(layer_s_s_differ2, keep_prob=0.9)
            layer_s_s_differ2 = slim.fully_connected(layer_s_s_differ2,
                                                     num_outputs=int(layers[0] / 3),
                                                     activation_fn=activation_fn,
                                                     scope="adver_layer1_22")
            layer_s_s_differ2 = tf.nn.dropout(layer_s_s_differ2, keep_prob=0.9)

            layer_s_s_differ1_differ2 = slim.fully_connected(tf.concat([source_ft_gan_differ2, source_ft_gan_differ1], axis=1),
                                                     num_outputs=int(layers[0]/3 ),
                                                     activation_fn=activation_fn,
                                                     scope="adver_layer1_13")
            layer_s_s_differ1_differ2 = tf.nn.dropout(layer_s_s_differ1_differ2, keep_prob=0.9)
            layer_s_s_differ1_differ2 = slim.fully_connected(layer_s_s_differ1_differ2,
                num_outputs=int(layers[0] / 3),
                activation_fn=activation_fn,
                scope="adver_layer1_23")
            layer_s_s_differ1_differ2 = tf.nn.dropout(layer_s_s_differ1_differ2, keep_prob=0.9)

            layer_s_s_final = slim.fully_connected(tf.concat([layer_s_s_differ1, layer_s_s_differ2,layer_s_s_differ1_differ2],1),
                                                     num_outputs=int(layers[1]),
                                                     activation_fn=activation_fn,
                                                     scope="adver_layer2")
            net = tf.nn.dropout(layer_s_s_final, keep_prob=0.8)
            net_1 = slim.fully_connected(net, 1, activation_fn=None, scope="adver_layer3")
            net_2 = slim.fully_connected(net, 2, activation_fn=None, scope="adver_layer4")
    return net_1,net_2



def adversarial_discriminator_local(net,conditional_net, layers, scope='adversary', leaky=False,reuse=False):
    def LeakyReLU(x, alpha=0.2):
        return tf.maximum(alpha*x, x)

    if leaky:
        activation_fn = LeakyReLU
    else:
        activation_fn = tf.nn.relu

    with tf.variable_scope(scope) as scope:
        if reuse:
                scope.reuse_variables()
        with ExitStack() as stack:
            stack.enter_context(tf.variable_scope(scope))
            stack.enter_context(
                slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=activation_fn,
                    weights_regularizer=slim.l2_regularizer(2.5e-5)))

            layer1 = slim.fully_connected(net,num_outputs=layers[0], activation_fn=activation_fn)
            layer2 = slim.fully_connected(layer1, num_outputs=layers[1], activation_fn=activation_fn)
            net = slim.fully_connected(layer2, 2, activation_fn=None)
    return net,layer2
