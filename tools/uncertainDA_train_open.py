import logging
import os
import random
from collections import deque
from collections import OrderedDict

import click
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tqdm import tqdm
import math

import adda
import pickle


@click.command()
@click.argument('source')
@click.argument('target')
@click.argument('model')
@click.argument('output')
@click.option('--gpu', default='0')
@click.option('--iterations', default=20000)
@click.option('--batch_size', default=50)
@click.option('--display', default=10)
@click.option('--lr', default=1e-4)
@click.option('--stepsize', type=int)
@click.option('--snapshot', default=5000)
@click.option('--netvladflag', type=int)
@click.option('--adverflag', type=int, default=1)
@click.option('--reweightflag', type=int, default=0)  # reweightflag
@click.option('--tempe_uncertain_value', default=1.0)
@click.option('--poolinglayer_mode', type=int, default=2)
@click.option('--cluster_number', type=int, default=32)
@click.option('--weights', required=True)
@click.option('--solver', default='Adam')
@click.option('--adapt_layer', default='fc_adapt')  # pooling2  fc_adapt  fc4
@click.option('--uncertain_metric', default='var')  # var     entropy  var_oneshot
@click.option('--adversary', 'adversary_layers', default=[500, 500],
              multiple=True)
@click.option('--adversary_leaky/--adversary_relu', default=True)
@click.option('--seed', type=int)
def main(source, target, model, output,
         gpu, iterations, batch_size, display, lr, stepsize, snapshot, weights, tempe_uncertain_value,
         solver, adversary_layers, adversary_leaky, seed, netvladflag, adapt_layer,
         cluster_number, uncertain_metric, poolinglayer_mode, adverflag, reweightflag):
    # miscellaneous setup
    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    if seed is None:
        seed = random.randrange(2 ** 32 - 2)
    logging.info('Using random seed {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed + 1)
    tf.set_random_seed(seed + 2)
    error = False
    print("----adapt_layer:", adapt_layer)
    if netvladflag == 0:
        adapt_layer = "pooling2"
        model = "lenet"

    optimizer_name = solver
    if netvladflag == 1:
        uncertainty_flag = 1.0
        print("---------------using uncertainty------------")
    else:
        uncertainty_flag = 0
        print("----------------no uncertainty-------------")

    if solver == "sgd":
        lr = 0.002
    else:
        lr = 0.001
        print("____________adam____lr-0.001------------")

    datasetname = source.split(":")

    target_dataset_name, target_split_name = target.split(':')
    print("____using MNIST dataset source:", datasetname[0])
    print("____using MNIST dataset target:", target_dataset_name)
    class_number = 10

    def classifier_fcn(net, category=class_number, dropout_keepratio=0.5, scope="source", reuse=False,
                       is_training=True):
        layers = OrderedDict()
        with tf.variable_scope(scope, reuse=reuse):
            layers['pooling2'] = net
            net = tf.nn.dropout(net, keep_prob=0.5)
            net = slim.fully_connected(net, 500, scope='fc_adapt')
            layers['fc_adapt'] = net
            # layers['fc_adapt'] = net
            net = tf.nn.dropout(net, keep_prob=dropout_keepratio)
            net = slim.fully_connected(net, category, activation_fn=None, scope='softmax')
            # layers['logits'] = net
            layers['fc4'] = net
        return net, layers

    def inference(source_im_batch, source_label_batch, target_im_batch, target_label_batch, target_im_batch_temp_test,
                  target_label_batch_test, model_fn, layer_source, layer_target):

        source_ft, layers_s_dp = classifier_fcn(source_im_batch, scope='source')
        target_ft, layers_t_dp = classifier_fcn(target_im_batch, scope='source', reuse=True)
        target_ft_test, layers_t_dp_test = classifier_fcn(target_im_batch_temp_test, scope='source', reuse=True)

        super_loss = tf.losses.sparse_softmax_cross_entropy(source_label_batch, source_ft / tempe_uncertain)
        correct_prediction = tf.equal(tf.argmax(source_ft, -1), tf.cast(source_label_batch, dtype="int64"))
        accuracy_source = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        correct_prediction_target = tf.equal(tf.argmax(target_ft, -1), tf.cast(target_label_batch, dtype="int64"))
        accuracy_target = tf.reduce_mean(tf.cast(correct_prediction_target, 'float'))

        correct_prediction_target_test = tf.equal(tf.argmax(target_ft_test, -1),
                                                  tf.cast(target_label_batch_test, dtype="int64"))
        accuracy_target_test = tf.reduce_mean(tf.cast(correct_prediction_target_test, 'float'))


        source_ft_gan = layers_s_dp[adapt_layer]  # layer_source['fc_1']    pooling_last
        target_ft_gan = layers_t_dp[adapt_layer]  # layer_target['fc_1']    pooling_last


        adversary_ft = tf.concat([source_ft_gan, target_ft_gan], 0)
        source_adversary_label = tf.ones([batch_size, 1], tf.float32)
        target_adversary_label = tf.zeros([batch_size, 1], tf.float32)
        adversary_label = tf.concat(
            [source_adversary_label, target_adversary_label], 0)

        if adapt_layer == "fc4":
            adver_layer = [512, 512]
        else:
            adver_layer = [2048, 2048]
        if adapt_layer == "pooling2":
            adver_layer = [512, 512]

        adversary_logits, adversary_logits_two = adda.adversary.adversarial_discriminator(
            net=adversary_ft, lamda_w=weights_mapping, layers=adver_layer, leaky=False,
            scope="local_adversary", dropout_keep=0.5)
        adversary_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=adversary_logits, labels=adversary_label)
        mapping_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=adversary_logits, labels=1 - adversary_label)
        mapping_loss = -adversary_loss


        mapping_loss_mmd = adda.util.mmd_loss(source_ft_gan, target_ft_gan, 1.0)
        adversary_logits_softmax = tf.nn.softmax(adversary_logits_two)
        source_sparseloss = -tf.reduce_sum(
            adversary_logits_softmax * tf.log(adversary_logits_softmax + 1e-10) / tf.log(2.0)) * 1.0 / batch_size

        return source_ft, target_ft, super_loss, accuracy_source, accuracy_target_test, mapping_loss, adversary_loss, mapping_loss_mmd, source_sparseloss, adversary_logits

    with tf.Graph().as_default(), tf.device('cpu:0'):
        try:
            source_dataset_name, source_split_name = source.split(':')
        except ValueError:
            error = True
            logging.error(
                'Unexpected source dataset {} (should be in format dataset:split)'
                    .format(source))
        try:
            target_dataset_name, target_split_name = target.split(':')
        except ValueError:
            error = True
            logging.error(
                'Unexpected target dataset {} (should be in format dataset:split)'
                    .format(target))
        if error:
            raise click.Abort

        # setup data
        logging.info('Adapting {} -> {}'.format(source, target))
        source_dataset = getattr(adda.data.get_dataset(source_dataset_name),
                                 source_split_name)
        target_dataset = getattr(adda.data.get_dataset(target_dataset_name),
                                 target_split_name)
        target_dataset_test = getattr(adda.data.get_dataset(target_dataset_name),
                                      'test')

        source_im, source_label = source_dataset.tf_ops()
        target_im, target_label = target_dataset.tf_ops()
        target_im_test, target_label_test = target_dataset_test.tf_ops()

        model_fn = adda.models.get_model_fn(model)
        source_im = adda.models.preprocessing(source_im, model_fn)
        target_im = adda.models.preprocessing(target_im, model_fn)
        target_im_test = adda.models.preprocessing(target_im_test, model_fn)
        lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
        weights_uncertain = tf.Variable(0.0, name='weights_uncertain', trainable=False)
        weights_uncertain_before = tf.Variable(0.0, name='weights_uncertain', trainable=False)
        tempe_uncertain = tf.Variable(tempe_uncertain_value, name='tempe_uncertain', trainable=False)
        weights_mapping = tf.Variable(0.0, name='weights_mapping', trainable=False)
        weight_classifier = tf.Variable(1.0, name='weight_classifier', trainable=False)

        source_super_loss = []
        source_acc = []
        target_acc_total = []
        target_ft_total_ori = []
        source_ft_total_ori = []
        if uncertainty_flag == True:
            gpu_visible = [0, 1, 2]
        else:
            gpu_visible = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15,16,17,18,19,20]

        if optimizer_name == "sgd":
            optimizer = tf.train.MomentumOptimizer(lr_var, 0.9)
            optimizer_fcn = tf.train.MomentumOptimizer(lr_var * 5.0, 0.9)
            print("using sgd......")
        else:
            optimizer = tf.train.AdamOptimizer(lr_var, 0.5)
            optimizer_fcn = tf.train.AdamOptimizer(lr_var * 5.0, 0.5)
            print("using Adam......")


        mapping_loss = []
        adversary_loss = []
        mapping_loss_mmd = []
        source_sparseloss = []
        source_ft_total_logits = []
        target_ft_total_logits = []
        adversary_logits_total = []

        adverbegin_flag = 0

        source_im_batch, source_label_batch = tf.train.batch(
            [source_im, source_label], batch_size=batch_size)
        target_im_batch, target_label_batch = tf.train.batch(
            [target_im, target_label], batch_size=batch_size)
        target_im_batch_test, target_label_batch_test = tf.train.batch(
            [target_im_test, target_label_test], batch_size=batch_size)

        with tf.device('/gpu:%d' % gpu_visible[0]):
            source_im_batch_temp, layer_source = model_fn(source_im_batch, scope='source')
            target_im_batch_temp, layer_target = model_fn(target_im_batch, scope='source', reuse=True)
            target_im_batch_temp_test, layer_target_test = model_fn(target_im_batch_test, scope='source', reuse=True)

        with tf.variable_scope(tf.get_variable_scope()):
            for gpui in range(1, len(gpu_visible)):
                with tf.device('/gpu:%d' % gpu_visible[gpui]):
                    source_ft, target_ft, super_loss, accuracy_source, accuracy_target, mapping_loss_temp, \
                    adversary_loss_temp, mapping_loss_mmd_temp, source_sparseloss_temp, adversary_logits_temp \
                        = inference(source_im_batch_temp, source_label_batch, target_im_batch_temp, target_label_batch,
                                    target_im_batch_temp_test, target_label_batch_test, model_fn, layer_source,
                                    layer_target)
                    tf.get_variable_scope().reuse_variables()
                    target_ft_total_ori.append(target_ft)
                    source_ft_total_ori.append(source_ft)
                    source_ft_total_logits.append(tf.nn.softmax(source_ft))
                    target_ft_total_logits.append(tf.nn.softmax(target_ft))
                    source_sparseloss.append(source_sparseloss_temp)
                    source_super_loss.append(super_loss)
                    source_acc.append(accuracy_source)
                    target_acc_total.append(accuracy_target)
                    mapping_loss.append(mapping_loss_temp)
                    mapping_loss_mmd.append(mapping_loss_mmd_temp)
                    adversary_loss.append(adversary_loss_temp)
                    adversary_logits_total.append(adversary_logits_temp)

        source_super_loss = tf.reduce_mean(source_super_loss)
        mapping_loss = tf.reduce_mean(mapping_loss)
        mapping_loss_mmd = tf.reduce_mean(mapping_loss_mmd)
        # adversary_loss = tf.reduce_mean(adversary_loss)
        source_acc = tf.reduce_mean(source_acc)
        target_acc = tf.reduce_mean(target_acc_total)
        tf.summary.scalar('Accuracy_target', target_acc)
        source_sparseloss = tf.reduce_mean(source_sparseloss)
        print("_____len(target_ft_total):", len(target_ft_total_ori))
        print("_____target_ft_total[0]:", target_ft_total_ori[0])
        print("_____target_ft_total:", target_ft_total_ori)

        print("source_ft_total_logits[0]:", source_ft_total_logits[0])
        # source_ft_total_logits=tf.convert_to_tensor(source_ft_total_logits)

        target_ft_total = tf.convert_to_tensor(tf.nn.softmax(target_ft_total_ori / tempe_uncertain))
        source_ft_total = tf.convert_to_tensor(tf.nn.softmax(source_ft_total_ori / tempe_uncertain))
        adversary_logits_total = tf.convert_to_tensor(tf.nn.softmax(adversary_logits_total))
        print("target_ft_total:", target_ft_total)

        source_ft_total_logits = tf.convert_to_tensor(source_ft_total_logits)
        target_ft_total_logits = tf.convert_to_tensor(target_ft_total_logits)
        source_ft_total_logits = tf.argmax(source_ft_total_logits, axis=-1)
        target_ft_total_logits = tf.argmax(target_ft_total_logits, axis=-1)

        if uncertain_metric == "var_oneshot":
            assgn_s_arg = tf.argmax(source_ft_total, axis=-1)
            assgn_t_arg = tf.argmax(target_ft_total, axis=-1)
            print("-----assgn_target_ft_total:", assgn_t_arg)

            source_ft_total = tf.one_hot(assgn_s_arg, depth=10)
            target_ft_total = tf.one_hot(assgn_t_arg, depth=10)
            print("target_ft_total_oneshot:", target_ft_total)


        source_ft_total_mean, source_ft_total_var = tf.nn.moments(source_ft_total, axes=0)
        target_ft_total_mean, target_ft_total_var = tf.nn.moments(target_ft_total, axes=0)

        if netvladflag == 1 and reweightflag == 1:

            target_ft_total_weight = tf.convert_to_tensor(tf.nn.softmax(target_ft_total_ori / (tempe_uncertain * 1.2)))
            source_ft_total_weight = tf.convert_to_tensor(tf.nn.softmax(source_ft_total_ori / (tempe_uncertain * 1.2)))

            source_ft_total_mean_weight, source_ft_total_var_weight = tf.nn.moments(source_ft_total_weight, axes=0)
            target_ft_total_mean_weight, target_ft_total_var_weight = tf.nn.moments(target_ft_total_weight, axes=0)

            weight_uncer_source = tf.exp(
                -tf.reduce_sum(-source_ft_total_mean_weight * tf.log(source_ft_total_mean_weight + 1e-10), -1))

            weight_uncer_source_flag = tf.cast(tf.greater(weight_uncer_source, 0.9), dtype=tf.float32)  # e**(-0.1)=0.9
            weight_uncer_source = weight_uncer_source * weight_uncer_source_flag
            weight_uncer_source = (weight_uncer_source * batch_size) / (tf.reduce_sum(weight_uncer_source) + 1e-10)

            weight_uncer_target = tf.exp(
                -tf.reduce_sum(-target_ft_total_mean_weight * tf.log(target_ft_total_mean_weight + 1e-10), -1))

            weight_uncer_target_flag = tf.cast(tf.greater(weight_uncer_target, 0.812),
                                               dtype=tf.float32)  # e**(-0.3)=0.74   e**(-0.2)=0.812
            weight_uncer_target = weight_uncer_target * weight_uncer_target_flag

            weight_uncer_target = (weight_uncer_target * batch_size) / (tf.reduce_sum(weight_uncer_target) + 1e-10)

            print("weight_uncer_target-1:", weight_uncer_target)
            weight_uncer_all = tf.concat([weight_uncer_source, weight_uncer_target], axis=0)
            adversary_loss = tf.convert_to_tensor(adversary_loss)
            print("adversary_loss-1:", adversary_loss)
            adversary_loss = tf.multiply(adversary_loss, weight_uncer_all)
            print("adversary_loss-2:", adversary_loss)

            adversary_loss = tf.reduce_mean(adversary_loss)
        else:
            adversary_loss = tf.reduce_mean(adversary_loss)

        adversary_logits_total_mean, adversary_logits_total_var = tf.nn.moments(adversary_logits_total, axes=0)
        print("source_ft_total_mean:", source_ft_total_mean)
        print("source_ft_total_var:", source_ft_total_var)

        adversary_ft = tf.concat([source_ft_total_mean, target_ft_total_mean], 0)
        source_adversary_label = tf.zeros([tf.shape(source_ft_total_mean)[0]], tf.int32)
        target_adversary_label = tf.ones([tf.shape(target_ft_total_mean)[0]], tf.int32)
        adversary_label = tf.concat(
            [source_adversary_label, target_adversary_label], 0)

        weights_mapping_global = 0.0

        adversary_logits_1, adversary_logits_two = adda.adversary.adversarial_discriminator(
            net=adversary_ft, lamda_w=weights_mapping_global, layers=[8, 8], leaky=False,
            scope="global_adversary", dropout_keep=0.5)

        mapping_loss_global = tf.losses.sparse_softmax_cross_entropy(
            1 - adversary_label, adversary_logits_two)
        adversary_loss_global = tf.losses.sparse_softmax_cross_entropy(
            adversary_label, adversary_logits_two)

        '''
        loss_uncertainty_temp=tf.reduce_sum(tf.nn.softmax(adversary_logits_total)*tf.log(tf.nn.softmax(adversary_logits_total)+1e-10))/ \
                              tf.log(2.0) * 1.0/(batch_size*len(adversary_logits_total))
         '''

        adversary_logits_total = tf.reduce_mean(adversary_logits_total, axis=0)
        # print("_____len(target_ft_total)_after:", len(target_ft_total))
        print("_____target_ft_total[0]_after:", target_ft_total[0])
        # print("_____target_ft_total[1]_after:", target_ft_total[1])
        print("_____ltarget_ft_total_after:", target_ft_total)

        print("_____target_ft_total:", target_ft_total)

        source_ft_gan = tf.reshape(target_ft_total, [-1, int(target_ft_total.get_shape()[-1])])
        target_ft_gan = tf.reshape(source_ft_total, [-1, int(source_ft_total.get_shape()[-1])])
        adversary_ft = tf.concat([source_ft_gan, target_ft_gan], 0)
        source_adversary_label = tf.zeros([tf.shape(source_ft_gan)[0]], tf.int32)
        target_adversary_label = tf.ones([tf.shape(target_ft_gan)[0]], tf.int32)
        adversary_label = tf.concat(
            [source_adversary_label, target_adversary_label], 0)

        mapping_loss_mmd_global = adda.util.mmd_loss(source_ft_gan, target_ft_gan, 1.0)

        source_ft_total_var = tf.reduce_sum(source_ft_total_var)
        target_ft_total_var = tf.reduce_sum(target_ft_total_var)

        #####

        #######
        loss_uncertainty = -tf.reduce_sum(tf.nn.softmax(adversary_logits_total) *
                                          tf.log(tf.nn.softmax(adversary_logits_total) + 1e-10)) / tf.log(
            2.0) * 1.0 / batch_size \
            # +loss_uncertainty_temp
        loss_uncertainty_source = -tf.reduce_sum(
            tf.nn.softmax(target_ft_total) * tf.log(tf.nn.softmax(target_ft_total) + 1e-10)) / tf.log(
            2.0) * 1.0 / batch_size

        #####
        if uncertain_metric == "var" or uncertain_metric == "var_oneshot":
            print("using metric----------------", uncertain_metric)
            loss_uncertainty_source = source_ft_total_var / batch_size
            loss_uncertainty = target_ft_total_var / batch_size
        else:
            print("using  metric----------------", uncertain_metric)

            loss_uncertainty = -tf.reduce_sum(target_ft_total_mean *
                                              tf.log(target_ft_total_mean + 1e-10)) / batch_size
            # +loss_uncertainty_temp
            loss_uncertainty_source = -tf.reduce_sum(source_ft_total_mean
                                                     * tf.log(source_ft_total_mean + 1e-10)) / batch_size

        tf.summary.scalar('Uncertainty_target', loss_uncertainty)
        #
        # loss_uncertainty_adver = -tf.reduce_sum(adversary_logits_total_mean * tf.log(adversary_logits_total_mean + 1e-10)) / tf.log(
        #     2.0) * 1.0 / batch_size

        train_variables = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        train_variables += bn_moving_vars

        dropout_ratio = 0.5
        tau = 1.0
        lengthscale = 2.0
        N = len(source_dataset) + len(target_dataset)
        # N=2500
        weights_uncertain_value = 0.1
        weights_uncertain_value_before = 0.1
        print("____N:", N)
        reg = (1 - dropout_ratio) / (2. * N * tau)
        loss_L2 = reg * tf.add_n([tf.nn.l2_loss(v) for v in train_variables
                                  if ('weights' in v.name) and (
                                              ('softmax' in v.name) or ('fc_adapt' in v.name) or ('fc4' in v.name))])
        l2_loss_total = 0.00001 * tf.add_n(
            [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])

        total_loss = source_super_loss + l2_loss_total + adversary_loss * adverflag \
                     + (weight_classifier * abs(
            loss_uncertainty - loss_uncertainty_source) ** 2) * uncertainty_flag * 0.25 * weights_mapping
        # - 0.1*loss_uncertainty_source
        classifer_loss = source_super_loss + l2_loss_total \
                         + adversary_loss * adverflag + loss_L2 \
                         + (weight_classifier * abs(
            loss_uncertainty - loss_uncertainty_source) ** 2) * uncertainty_flag * 0.25 * weights_mapping


        adversary_vars = [v for v in train_variables if "local_adversary" in v.name]
        adversary_vars_global = [v for v in train_variables if "global_adversary" in v.name]
        encoder_vars = [v for v in train_variables if
                        ("adversaryfdsafd" not in v.name) and ("softmax" not in v.name) and ("fc3fdafd" not in v.name)]
        classifier_vars = [v for v in train_variables if
                           ("adversary" not in v.name) and (("softmax" in v.name) or ("fc3fdsafd" in v.name))]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            step = optimizer.minimize(total_loss, var_list=encoder_vars)
            step_classifier = optimizer.minimize(classifer_loss, var_list=classifier_vars)

        adversary_step = optimizer.minimize(adversary_loss, var_list=adversary_vars)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("tensorboard_results" + '/mnist')

        # set up session and initialize
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(init)

        # restore weights
        if os.path.isdir(weights):
            weights = tf.train.latest_checkpoint(weights)

        var_dict_save = adda.util.collect_vars("source")

        target_saver = tf.train.Saver(var_list=var_dict_save)
        # optimization loop (finally)
        output_dir = os.path.join('snapshot', output)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        source_acc_val_total = deque(maxlen=10)
        target_acc_val_total = deque(maxlen=80)
        uncertainty_val_total = deque(maxlen=80)
        source_uncertainty_val_total = deque(maxlen=80)

        bar = tqdm(range(iterations))
        bar.set_description('{} (lr: {:.0e})'.format(output, lr))
        bar.refresh()
        adverbegin_flag = 0
        interration = 0
        acc_maximum = 0
        begin_adv_i = 0

        uncertainty_save = []
        source_uncertainty_save = []
        test_acc_save = []
        interration_total = []

        for i in bar:
            if i % 3 == 0:

                # weights_mapping_value=(i*1.0/iterations)*0.9+0.1
                if optimizer_name == "sgd":
                    if i > begin_adv_i:
                        p_value = 1.0 * (i - begin_adv_i) / (iterations - begin_adv_i) * 1.0
                        weights_mapping_value = 2.0 / (1.0 + math.exp(-10.0 * p_value)) - 1.0
                        learning_rate_value = lr / math.pow(1 + 10.0 * p_value, 0.75)
                        # learning_rate_value = lr * (1 + 0.001 * i) ** (-0.75)
                        sess.run(tf.assign(lr_var, learning_rate_value))
                        # sess.run(tf.assign(weights_mapping, weights_mapping_value))
                        sess.run(tf.assign(weights_mapping, 0.8 * weights_mapping_value))
                        sess.run(tf.assign(weights_uncertain, weights_mapping_value * 0.25))
                else:
                    if i > begin_adv_i:
                        p_value = 1.0 * (i - begin_adv_i) / (iterations - begin_adv_i) * 1.0
                        weights_mapping_value = 2.0 / (1.0 + math.exp(-10.0 * p_value)) - 1.0

                        sess.run(tf.assign(weights_mapping, 0.8 * weights_mapping_value))


                        if i < int(begin_adv_i / 3):
                            sess.run(tf.assign(weights_uncertain, weights_mapping_value * 0.05))

                        elif i > int(begin_adv_i / 3) and i < int(begin_adv_i / 2):
                            sess.run(tf.assign(weights_uncertain, weights_mapping_value * 0.1))

                        elif i > int(begin_adv_i / 2) and i < int(begin_adv_i * 2 / 3):
                            sess.run(tf.assign(weights_uncertain, weights_mapping_value * 0.15))
                        else:
                            sess.run(tf.assign(weights_uncertain, weights_mapping_value * 0.25))


            sess.run([step, step_classifier])


            interration = interration + 1

            if i % 10 == 0:
                source_super_loss_val, source_acc_val, target_acc_val, loss_uncertainty_val, loss_uncertainty_source_val, \
                mapping_loss_val, adversary_loss_val, mapping_loss_mmd_val, adversary_loss_global_val \
                    = sess.run(
                    [source_super_loss, source_acc, target_acc, loss_uncertainty, loss_uncertainty_source,
                     mapping_loss, adversary_loss, mapping_loss_mmd, adversary_loss_global])

                # train_writer.add_summary(summary, i)
                source_acc_val_total.append(source_acc_val)
                target_acc_val_total.append(target_acc_val)
                uncertainty_val_total.append(loss_uncertainty_val)
                source_uncertainty_val_total.append(loss_uncertainty_source_val)

                test_acc_save.append(np.mean(target_acc_val_total))
                uncertainty_save.append(np.mean(uncertainty_val_total))
                interration_total.append(int(i))
                source_uncertainty_save.append(np.mean(source_uncertainty_val_total))

            if i == iterations - 1:
                uncertainty_save_t = np.array(uncertainty_save)
                uncertainty_save_s = np.array(source_uncertainty_save)
                interration_total_s = np.array(interration_total)
                test_acc_save_s = np.array(test_acc_save)
                if netvladflag == 1:
                    savename_resutls = str(output) + "uncer" + str(uncertain_metric) + "_" + str(
                        int(i)) + "_adv_" + str(adverflag)
                else:
                    savename_resutls = str(output) + "_no_uncer" + str(uncertain_metric) + "_" + str(
                        int(i)) + "_adv_" + str(adverflag)

                pickle.dump((test_acc_save_s, uncertainty_save_t, uncertainty_save_s, interration_total_s),
                            open("acc_results/" + savename_resutls + "_.pkl", "wb"))

            if np.mean(target_acc_val_total) > acc_maximum:
                acc_maximum = np.mean(target_acc_val_total)

            if i % 1000 == 0:

                f = open("results_reubttal/" + source + "2" + target + ".txt", 'a')
                if i == 0:
                    f.write("-net-" + str(model) + "-adapt_layer-" + str(adapt_layer) + "-Uncertain_metric-"
                            + str(uncertain_metric) + "-tempe_uncertain-" + str(tempe_uncertain_value) + "-" + str(
                        solver))
                    f.write("\r")
                f.write(str(model) + "-" + str(uncertain_metric) + "-" + "-training_num:%d, accuracy:%4f" % (
                i, acc_maximum))
                f.write("\r")
                f.close()


            if np.mean(source_acc_val_total) > 99999 and adverbegin_flag == 0 and i >= 1000 and i % 500 == 0:
                # adverbegin_flag=1
                if (i / 500) % 2 == 1:
                    print("weights_uncertain_value assigned....")
                    sess.run(tf.assign(weights_uncertain, weights_uncertain_value))
                    sess.run(tf.assign(weights_uncertain_before, weights_uncertain_value_before))

                else:
                    print("weights_uncertain_value assigned 0 0 0 0 0....")
                    sess.run(tf.assign(weights_uncertain, 0.0))
                    sess.run(tf.assign(weights_uncertain_before, 0.0))


            if i % 200 == 0:
                print("source_ft_total_logits:", source_ft_total_logits)
                source_ft_total_logits_val = sess.run(source_ft_total_logits)

                print("---source_ft_total_logits[0]:", source_ft_total_logits_val[:, 1])

                target_total_logits_val = sess.run(target_ft_total_logits)
                # print ("source_ft_total_logits_val:",source_ft_total_logits_val)
                print("----------target_total_logits_val[0]:", target_total_logits_val[:, 1])

            if i % display == 0:
                logging.info('{:10} T_acc:{:5.4f}  S_acc:{:5.4f}  '
                             ' T_uncer:{:5.4f} S_uncer:{:5.4f} '
                             'Mapping:{:5.4f} Adv_glo:{:5.4f}  acc_max:{:5.4f} '
                             .format('Iteration {}:'.format(i),
                                     np.mean(target_acc_val_total),
                                     np.mean(source_acc_val_total),
                                     loss_uncertainty_val, loss_uncertainty_source_val,
                                     mapping_loss_val, adversary_loss_global_val, acc_maximum))


            if stepsize is not None and (i + 1) % stepsize == 0:
                lr = sess.run(lr_var.assign(lr * 0.1))
                logging.info('Changed learning rate to {:.0e}'.format(lr))
                bar.set_description('{} (lr: {:.0e})'.format(output, lr))
            if (i + 1) % snapshot == 0:
                snapshot_path = target_saver.save(
                    sess, os.path.join(output_dir, output), global_step=i + 1)
                logging.info('Saved snapshot to {}'.format(snapshot_path))

        coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    main()
