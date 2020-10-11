import logging
import logging.config
import os.path
from collections import OrderedDict
from tensorflow.contrib import slim
from functools import partial

import tensorflow as tf
import yaml
import numpy as np
from tqdm import tqdm


class TqdmHandler(logging.StreamHandler):

    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def config_logging(logfile=None):
    path = os.path.join(os.path.dirname(__file__), 'logging.yml')
    with open(path, 'r') as f:
        config = yaml.load(f.read())
    if logfile is None:
        del config['handlers']['file_handler']
        del config['root']['handlers'][-1]
    else:
        config['handlers']['file_handler']['filename'] = logfile
    logging.config.dictConfig(config)


def remove_first_scope(name):
    return '/'.join(name.split('/')[1:])

def collect_vars(scope, start=None, end=None, prepend_scope=None):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    var_dict = OrderedDict()
    if isinstance(start, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(start):
                start = i
                break
    if isinstance(end, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(end):
                end = i
                break
    for var in vars[start:end]:
        var_name = remove_first_scope(var.op.name)
        if prepend_scope is not None:
            var_name = os.path.join(prepend_scope, var_name)
        var_dict[var_name] = var
    return var_dict

def netvlad(net,vlad_centers=None,vlad_W=None,vlad_B=None,l2_norm_flag=True,netvlad_alpha = 5000.0,
        scope='source/NetVLAD',netvlad_initCenters=16, weight_decay=0.00000,reuse=False):
    #net=tf.expand_dims(net,0)
    videos_per_batch=net.get_shape().as_list()[0]
    netvlad_initCenters = int(netvlad_initCenters)
    cluster_centers = np.random.normal(size=(
        netvlad_initCenters, net.get_shape().as_list()[-1]),loc=30.0, scale=10.0,)
    with tf.variable_scope(scope) as scope:
        if reuse:
                scope.reuse_variables()
        #net_normed = tf.nn.l2_normalize(net, 3, name='FeatureNorm')

        net_normed=net
        if vlad_centers==None:
            vlad_centers = slim.model_variable(
                    'centers_vlad',
                    shape=cluster_centers.shape,
                    initializer=tf.constant_initializer(cluster_centers),
                    regularizer=slim.l2_regularizer(weight_decay))

        netvlad_alpha_loss=0.9
        vlad_W_loss=tf.expand_dims(tf.expand_dims(tf.transpose(vlad_centers)*2 * netvlad_alpha_loss,axis=0),axis=0)
        vlad_B_loss=tf.reduce_sum(tf.square(vlad_centers),axis=1)*(netvlad_alpha_loss)*(-1)
        conv_output_loss = tf.nn.conv2d(net_normed, vlad_W_loss, [1, 1, 1, 1], 'VALID')
        dists_loss = tf.nn.bias_add(conv_output_loss, vlad_B_loss)
        assgnment_loss = dists_loss  #tf.nn.softmax(dists_loss, dim=3)

        conv_output = tf.nn.conv2d(net_normed, vlad_W, [1, 1, 1, 1], 'VALID')
        dists = tf.nn.bias_add(conv_output, vlad_B)
        assgnment = dists #tf.nn.softmax(dists, dim=3)
        print ("net_normed", net_normed)
        print ("----assgnment:", assgnment)
        vid_splits = tf.split(net_normed, videos_per_batch, 0)
        assgn_splits = tf.split(assgnment, videos_per_batch, 0)

        num_vlad_centers = vlad_centers.get_shape()[0]
        vlad_centers_split = tf.split(vlad_centers, netvlad_initCenters, 0)

        final_vlad = []
            #self.loss_smooth=tf.reduce_sum(tf.square(tf.subtract(assgn[0,:,:,1:],assgn[0,:,:,:-1])))
        for feats, assgn in zip(vid_splits, assgn_splits):
            vlad_vectors = []
            assgn_split_byCluster = tf.split(assgn, netvlad_initCenters, 3)
            for k in range(num_vlad_centers):
                    res = tf.reduce_sum(
                        tf.multiply(tf.subtract(
                            feats,
                            vlad_centers_split[k]), assgn_split_byCluster[k]),
                        [0, 1, 2])
                    vlad_vectors.append(res)
            vlad_vectors_frame = tf.stack(vlad_vectors, axis=0)
            final_vlad.append(vlad_vectors_frame)
        vlad_rep = tf.stack(final_vlad, axis=0, name='unnormed-vlad')

        with tf.name_scope('intranorm'):
            if l2_norm_flag==True:
                intranormed = tf.nn.l2_normalize(vlad_rep, dim=2)
            else:
                intranormed=vlad_rep

        with tf.name_scope('finalnorm'):
            if l2_norm_flag==True:
                vlad_rep_output = tf.nn.l2_normalize(tf.reshape(
                        intranormed,
                        [intranormed.get_shape().as_list()[0], -1]),
                        dim=1)
            else:
                vlad_rep_output=tf.reshape(
                        intranormed,
                        [intranormed.get_shape().as_list()[0], -1])

    # print("__1111_assgn_splits_loss:", assgnment_loss)
    #
    # assgnment_loss = -tf.reduce_sum(assgnment_loss * tf.log(assgnment_loss + 1e-10), -1)
    # assgnment_loss = tf.reduce_mean(assgnment_loss)
    # print("__1111_assgn_splits_loss:", assgnment_loss)
    # loss_vlad_sparse = assgnment_loss
    #assgn_splits_sum=tf.reshape(assgn_splits,[videos_per_batch,-1, netvlad_initCenters])
    #loss_vlad_sum =  - (tf.reduce_sum(assgn_splits_sum*tf.log(assgn_splits_sum+1e-10)))*1.0



    print ("vlad_rep_output:", vlad_rep_output)

    #return vlad_rep_output,assgnment,loss_vlad,vlad_centers
    return vlad_rep_output,vlad_rep,assgnment,assgnment_loss,vlad_centers


def Lenet_decoder(codes,scope="decoder",reuse=False):
    with tf.variable_scope(scope) as scope:
        if reuse:
            scope.reuse_variables()
        #net = tf.image.resize_nearest_neighbor(codes, (10, 10))
        net = slim.conv2d(codes, 64, 5, scope='decoder_conv11')
        net = tf.image.resize_nearest_neighbor(net, (12, 12))
        net = slim.conv2d(net, 32, 5, scope='decoder_conv21')
        net = tf.image.resize_nearest_neighbor(net, (16, 16))
        #net = slim.conv2d(net, 32, 3, scope='decoder_conv3')
        net = slim.conv2d(net, 1, 5, scope='decoder_conv3_1',activation_fn=None)
        #net=net-tf.reduce_min(net, reduction_indices=None, keep_dims=False, name=None)
        #net=net/tf.reduce_max(net, reduction_indices=None, keep_dims=False, name=None)
        net = tf.image.resize_nearest_neighbor(net, (28, 28))
    return net

def compute_pairwise_distances(x, y):
  """Computes the squared pairwise Euclidean distances between x and y.
  Args:
    x: a tensor of shape [num_x_samples, num_features]
    y: a tensor of shape [num_y_samples, num_features]
  Returns:
    a distance matrix of dimensions [num_x_samples, num_y_samples].
  Raises:
    ValueError: if the inputs do no matched the specified dimensions.
  """

  if not len(x.get_shape()) == len(y.get_shape()) == 2:
    raise ValueError('Both inputs should be matrices.')

  if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
    raise ValueError('The number of features should be the same.')

  norm = lambda x: tf.reduce_sum(tf.square(x), 1)

  # By making the `inner' dimensions of the two matrices equal to 1 using
  # broadcasting then we are essentially substracting every pair of rows
  # of x and y.
  # x will be num_samples x num_features x 1,
  # and y will be 1 x num_features x num_samples (after broadcasting).
  # After the substraction we will get a
  # num_x_samples x num_features x num_y_samples matrix.
  # The resulting dist will be of shape num_y_samples x num_x_samples.
  # and thus we need to transpose it again.
  return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

def gaussian_kernel_matrix(x, y, sigmas):
  r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
  We create a sum of multiple gaussian kernels each having a width sigma_i.
  Args:
    x: a tensor of shape [num_samples, num_features]
    y: a tensor of shape [num_samples, num_features]
    sigmas: a tensor of floats which denote the widths of each of the
      gaussians in the kernel.
  Returns:
    A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
  """
  beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

  dist = compute_pairwise_distances(x, y)

  s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

  return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
  r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
  Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
  the distributions of x and y. Here we use the kernel two sample estimate
  using the empirical mean of the two distributions.
  MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
  where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.
  Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.
  Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
  """
  with tf.name_scope('MaximumMeanDiscrepancy'):
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))

    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
  return cost

def mmd_loss(source_samples, target_samples, weight, scope=None):
  """Adds a similarity loss term, the MMD between two representations.
  This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
  different Gaussian kernels.
  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the MMD loss.
    scope: optional name scope for summary tags.
  Returns:
    a scalar tensor representing the MMD loss value.
  """
  sigmas = [
      1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
      1e3, 1e4, 1e5, 1e6
  ]
  gaussian_kernel = partial(
      gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

  loss_value = maximum_mean_discrepancy(
      source_samples, target_samples, kernel=gaussian_kernel)
  loss_value = tf.maximum(1e-4, loss_value) * weight

  # assert_op = tf.Assert(tf.is_finite(loss_value), [loss_value])
  # with tf.control_dependencies([assert_op]):
  #   tag = 'MMD Loss'
  #   if scope:
  #     tag = scope + tag
  #   tf.summary.scalar(tag, loss_value)
  #   tf.losses.add_loss(loss_value)

  return loss_value
