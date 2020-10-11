import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

def netvlad(net, netvlad_initCenters=32, weight_decay=0.00001):
    l2_norm_flag=1
    netvlad_alpha = 1.0

    #net=tf.expand_dims(net,0)
    videos_per_batch=net.get_shape().as_list()[0]

    netvlad_initCenters = int(netvlad_initCenters)
    cluster_centers = np.random.normal(size=(
        netvlad_initCenters, net.get_shape().as_list()[-1]))
    with tf.variable_scope('NetVLAD'):
        if l2_norm_flag==1:
                net_normed = tf.nn.l2_normalize(net, 3, name='FeatureNorm')
        else:
            net_normed=net
        vlad_centers = slim.model_variable(
                'centers',
                shape=cluster_centers.shape,
                initializer=tf.constant_initializer(cluster_centers),
                regularizer=slim.l2_regularizer(weight_decay))


        vlad_W=tf.expand_dims(tf.expand_dims(tf.transpose(vlad_centers)*2 * netvlad_alpha,axis=0),axis=0)
        vlad_B=tf.reduce_sum(tf.square(vlad_centers),axis=1)*(-netvlad_alpha)

        print ("vlad_w:",vlad_W)
        print ("vlad_B:",vlad_B)

        conv_output = tf.nn.conv2d(net_normed, vlad_W, [1, 1, 1, 1], 'VALID')
        dists = tf.nn.bias_add(conv_output, vlad_B)
        assgn = tf.nn.softmax(dists, dim=3)

        print ("net_normed", net_normed)
        print ("assgn:", assgn)




        vid_splits = tf.split(net_normed, videos_per_batch, 0)
        assgn_splits = tf.split(assgn, videos_per_batch, 0)


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
            if l2_norm_flag==1:
                intranormed = tf.nn.l2_normalize(vlad_rep, dim=2)
            else:
                intranormed=vlad_rep

        with tf.name_scope('finalnorm'):
            if l2_norm_flag==1:
                vlad_rep_output = tf.nn.l2_normalize(tf.reshape(
                        intranormed,
                        [intranormed.get_shape().as_list()[0], -1]),
                        dim=1)
            else:
                vlad_rep_output=tf.reshape(
                        intranormed,
                        [intranormed.get_shape().as_list()[0], -1])

    assgn_splits=tf.reshape(assgn_splits,[-1, netvlad_initCenters])

    loss_vlad =  - (tf.reduce_sum(assgn_splits*tf.log(assgn_splits+1e-10)))*1.0/videos_per_batch

    print ("vlad_rep_output:", vlad_rep_output)
    print ("assgn_splits:", assgn_splits)

    return vlad_rep_output,assgn_splits,loss_vlad