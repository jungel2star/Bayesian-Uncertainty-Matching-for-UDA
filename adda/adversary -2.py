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

