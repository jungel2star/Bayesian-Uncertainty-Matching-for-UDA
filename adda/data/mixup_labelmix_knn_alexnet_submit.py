import logging
import os
import random
from collections import deque
from collections import OrderedDict
import math
import click
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tqdm import tqdm
import pickle
import adda
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw

from tensorflow.contrib.slim.nets import resnet_v1


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
@click.option('--tempe_uncertain_value', default=1.0)
@click.option('--stepsize', type=int)
@click.option('--snapshot', default=5000)
@click.option('--netvladflag', type=int)
@click.option('--adverflag', type=int, default=1)
@click.option('--semantic_mix_flag', type=int, default=1)
@click.option('--reweightflag', type=int, default=0)  # reweightflag
@click.option('--weights', required=True)
@click.option('--solver', default='Adam')
@click.option('--adapt_layer', default='adapt_256')  # adapt_256  pooling5  fc8  fc6  fc7
@click.option('--training_mode', default="multi_target")  # adapt_256  pooling5  fc8  fc6  fc7
@click.option('--uncertain_metric', default='entropy')  # entropy  var
@click.option('--adversary', 'adversary_layers', default=[500, 500],
              multiple=True)
@click.option('--adversary_leaky/--adversary_relu', default=True)
@click.option('--seed', type=int)
def main(source, target, model, output,
         gpu, iterations, batch_size, display, lr, stepsize, snapshot, weights,
         solver, adversary_layers, adversary_leaky, seed,
         netvladflag, uncertain_metric, adapt_layer, tempe_uncertain_value, adverflag,
         reweightflag,semantic_mix_flag,training_mode):
    # miscellaneous setup
    adda.util.config_logging()
    semantic_mix_flag = semantic_mix_flag
    training_mode =training_mode    #  one_target    multi_target
    print("-------training mode: ", training_mode)

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
    optimizer_name = solver
    if netvladflag == 1:
        uncertainty_flag = 1.0
        print("----------------using uncertainty-------------")
    else:
        uncertainty_flag = 0
        print("----------------no uncertainty-------------")
    print("----adapt_layer:", adapt_layer)

    if solver == "sgd":
        lr = 0.002
    else:
        lr = 0.0001
        print("____________adam____lr-0.0001------------")


    target_dataset_name, target_split_name = target.split(':')

    datasetname = source.split(":")
    if datasetname[0] == "art" or datasetname[0] == "realworld" or datasetname[0] == "clipart" or datasetname[
        0] == "product":
        print("____using officehome dataset:", datasetname[0])

        class_number = 65
    else:
        print("____using office31 dataset:", datasetname[0])
        if training_mode == "one_target":
            class_number = 31
        else:
            class_number = 10


    def load_datasets(dataname,training_mode="train"):
        if True:
            filename = "./data/office31/" + dataname + "_256.pkl"
            category_num = 31
        dataset, label = pickle.load(open(filename, "rb"))
        train_images = dataset
        train_labels = label
        totalfile = dataset.shape[0]
        category_total=[[] for i in range(category_num)]
        for samplei in range(totalfile):
            category_total[label[samplei]].append(dataset[samplei])
        return (train_images, train_labels,category_total)


    def load_datasets_one_multi(dataname,training_mode="train"):
        if True:
            category_num = 10
            dataname_total=["amazon","dslr","webcam","caltech"]
            dataset, label = pickle.load(open("./data/office31/"+dataname+ "_10_10label.pkl", "rb"))
            dataname_total.remove(dataname)
            dataset_target, label_target = pickle.load(open("./data/office31/"+dataname_total[0] +  "_10_10label.pkl", "rb"))
            dataset_target1, label_target1 = pickle.load(open("./data/office31/"+dataname_total[1] +  "_10_10label.pkl", "rb"))
            dataset_target2, label_target2 = pickle.load(open("./data/office31/"+dataname_total[2] +  "_10_10label.pkl", "rb"))

            dataset_target=np.concatenate((dataset_target,dataset_target1),axis=0)
            label_target = np.concatenate((label_target, label_target1), axis=0)
            dataset_target = np.concatenate((dataset_target, dataset_target2), axis=0)
            label_target = np.concatenate((label_target, label_target2), axis=0)

        train_images = dataset
        train_labels = label
        totalfile = dataset.shape[0]
        category_total=[[] for i in range(category_num)]
        for samplei in range(totalfile):
            category_total[label[samplei]].append(dataset[samplei])
        return (train_images, train_labels,category_total,dataset_target,label_target,category_total)

    def getdata_batch(data, label,batchsize, category_total,begini=-1,differ_mode=True):
        data_batch = []
        data_batch_samelabel=[]
        data_batch_differ1=[]
        data_batch_differ2=[]
        label_batch_differ1 = []
        label_batch_differ2 = []

        label_batch = []
        if begini==-1:
            indexlist= [samplei for samplei in range(len(data))]
            for i in range(batchsize):
                initial_flag = random.choice(indexlist)
                indexlist.remove(initial_flag)
                data_batch.append(data[initial_flag])
                label_batch.append(label[initial_flag])
                samelabel_index = random.randint(0, len(category_total[label[initial_flag]]) - 1)
                data_batch_samelabel.append(category_total[label[initial_flag]][samelabel_index])
                if differ_mode==True:
                    #print ("---differ model: train....")
                    labellist = [labeli for labeli in range(class_number)]
                    del (labellist[label[initial_flag]])
                    differ_index1 = random.choice(labellist)
                    labellist.remove(differ_index1)
                    differ_index2 = random.choice(labellist)
                    label_batch_differ1.append(differ_index1)
                    random_index = random.randint(0, len(category_total[differ_index1]) - 1)
                    data_batch_differ1.append(category_total[differ_index1][random_index])
                    label_batch_differ2.append(differ_index2)
                    random_index2 = random.randint(0, len(category_total[differ_index2]) - 1)
                    data_batch_differ2.append(category_total[differ_index2][random_index2])
                else:
                    #print ("----differ mode: validation 10")
                    data_batch_differ1.append(data[initial_flag])
                    data_batch_differ2.append(data[initial_flag])
                    label_batch_differ1.append(label[initial_flag])
                    label_batch_differ2.append(label[initial_flag])


            return np.array(data_batch), np.array(label_batch),np.array(data_batch_samelabel), \
                   np.array(data_batch_differ1), np.array(label_batch_differ1), \
                   np.array(data_batch_differ2), np.array(label_batch_differ2),
        else:
            indexlist = [samplei for samplei in range(len(data))]
            for i in range(batchsize):
                reali=i+begini
                if reali>=len(data)-1:
                    reali = random.choice(indexlist)
                    indexlist.remove(reali)
                    #reali=reali %len(data)
                data_batch.append(data[reali])
                label_batch.append(label[reali])
                samelabel_index = random.randint(0, len(category_total[label[reali]]) - 1)
                data_batch_samelabel.append(category_total[label[reali]][samelabel_index])

                if differ_mode==True:
                    #print ("---differ model: train....")
                    labellist = [labeli for labeli in range(class_number)]
                    del (labellist[label[reali]])
                    differ_index1 = random.choice(labellist)
                    labellist.remove(differ_index1)
                    differ_index2 = random.choice(labellist)
                    label_batch_differ1.append(differ_index1)
                    random_index = random.randint(0, len(category_total[differ_index1]) - 1)
                    data_batch_differ1.append(category_total[differ_index1][random_index])
                    label_batch_differ2.append(differ_index2)
                    random_index2 = random.randint(0, len(category_total[differ_index2]) - 1)
                    data_batch_differ2.append(category_total[differ_index2][random_index2])
                else:
                    #print ("----differ mode: validation 10")
                    data_batch_differ1.append(data[reali])
                    data_batch_differ2.append(data[reali])
                    label_batch_differ1.append(label[reali])
                    label_batch_differ2.append(label[reali])

            return  np.array(data_batch), np.array(label_batch),np.array(data_batch_samelabel), \
                    np.array(data_batch_differ1), np.array(label_batch_differ1),  \
                    np.array(data_batch_differ2), np.array(label_batch_differ2)


    def load_initial_weights(session, path_file, scope, skip_layer="fcn"):
        if skip_layer == "fcn":
            skip_layer = ["fc6", "fc7", "fc8"]
        else:
            skip_layer = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc8"]
        skip_layer = ["fc8", 'fc_adapt']
        # Load the weights into memory
        weights_dict = np.load(path_file, encoding='bytes',allow_pickle=True).item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            # Check if the layer is one of the layers that should be reinitialized
            if op_name not in skip_layer:
                with tf.variable_scope(scope + "/" + op_name, reuse=True):
                    # Loop over list of weights/biases and assign them to their corresponding tf variable
                    for data in weights_dict[op_name]:
                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))

    def train_image_process(image):
        mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
        image = tf.image.resize_images(image, [256, 256], method=2)
        image_ori = tf.image.resize_images(image, [227, 227], method=2) - tf.constant(mean)
        image_total = tf.random_crop(image[0, :, :, :], [227, 227, 3])
        image_total = tf.image.random_flip_left_right(image_total)
        image_total = image_total - tf.constant(mean)
        image_total = tf.expand_dims(image_total, axis=0)
        image_total = tf.concat([image_total, image_ori[1:15]], axis=0)
        for i in range(15, batch_size):
            image_temp = tf.random_crop(image[i, :, :, :], [227, 227, 3])
            image_temp = tf.image.random_flip_left_right(image_temp)
            image_temp = image_temp - tf.constant(mean)
            image_temp = tf.expand_dims(image_temp, axis=0)
            image_total = tf.concat([image_total, image_temp], axis=0)
        # print ("image_total:",image_total)
        return image_total

    def Alex_fcn(net,dropout_keepratio, category=class_number,  scope="source", reuse=False):
        layers = OrderedDict()
        with tf.variable_scope(scope, reuse=reuse):
            layers['pooling5'] = net
            print ("--pooling5: ",net)
            net = tf.nn.dropout(net, keep_prob=dropout_keepratio)
            net = slim.fully_connected(net, 4096, scope='fc6' )
            layers['fc6'] = net

            net = tf.nn.dropout(net, keep_prob=dropout_keepratio)
            # 7th Layer: FC (w ReLu) -> Dropout
            net = slim.fully_connected(net, 4096, scope='fc7')
            layers['fc7'] = net



        return net, layers

    def Alex_fcn2(net, dropout_keepratio,category=class_number, scope="source", reuse=False):
        layers = OrderedDict()
        with tf.variable_scope(scope, reuse=reuse):

            net = tf.nn.dropout(net, keep_prob=dropout_keepratio)
            net = slim.fully_connected(net, 256, scope='fc_adapt')
            layers['adapt_256'] = net
            net = tf.nn.dropout(net, keep_prob=dropout_keepratio)
            net = slim.fully_connected(net, category, activation_fn=None, scope='fc8')
            layers['fc8'] = net
        return net, layers

    def inference(source_im_batch, source_label_batch, target_im_batch, target_label_batch,
                  source_im_batch_samelabel,source_im_batch_differ1, source_label_differ1,
                                        source_im_batch_differ2, source_label_differ2,dropoupratio):
        source_im_batch, layers_s_dp = Alex_fcn(source_im_batch, scope='source',dropout_keepratio=dropoupratio)
        source_im_batch_samelabel, layers_s_dp_samelabel = \
            Alex_fcn(source_im_batch_samelabel, scope='source', reuse=True,dropout_keepratio=dropoupratio)
        target_im_batch, layers_t_dp = Alex_fcn(target_im_batch, scope='source', reuse=True,dropout_keepratio=dropoupratio)

        source_im_batch_differ1, layers_s_differ1_dp1 = Alex_fcn(source_im_batch_differ1,
                                                    scope='source', reuse=True,dropout_keepratio=dropoupratio)
        source_im_batch_differ2, layers_s_differ2_dp1 = \
            Alex_fcn(source_im_batch_differ2, scope='source', reuse=True,dropout_keepratio=dropoupratio)

        source_ft, layers_s_dp2 = Alex_fcn2(source_im_batch, scope='source',dropout_keepratio=dropoupratio)
        target_ft, layers_t_dp2 = Alex_fcn2(target_im_batch, scope='source', reuse=True,dropout_keepratio=dropoupratio)
        source_ft_samelabel, layers_s_dp_samelabel = \
            Alex_fcn2(source_im_batch_samelabel, scope='source', reuse=True,dropout_keepratio=dropoupratio)

        pred_index_t_t_total=[]
        pred_index_t_s_total=[]
        neighbor_num = 1
        target_im_batch_mix=[]
        target_im_batch_mix2 = []
        source_im_batch_mix=[]
        label_batch_source_mix=[]
        source_im_batch_target=[]
        source_im_batch_samelabel_target=[]
        target_im_batch_target=[]
        label_batch_target_target=[]
        source_differ_im_batch_target=[]


        target_im_batch_256=layers_t_dp2[adapt_layer]   #v  'fc8'    adapt_layer
        source_im_batch_256 = layers_s_dp2[adapt_layer]

        # target_im_batch_256 = tf.cast(tf.nn.softmax(layers_t_dp2['fc8']/0.1),dtype=tf.float32)  # v  'fc8'    adapt_layer
        # source_im_batch_256 = tf.cast(tf.nn.softmax(layers_s_dp2['fc8']/0.1),dtype=tf.float32)   #tf.argmax(target_ft, -1)

        target_im_batch_256 = tf.one_hot(tf.argmax(layers_t_dp2['fc8'], axis=-1), dtype=tf.float32, depth=class_number)
        source_im_batch_256 = tf.one_hot(tf.argmax(layers_s_dp2['fc8'], axis=-1), dtype=tf.float32, depth=class_number)

        distance_t_t_total=[]
        distance_t_s_total=[]

        #gama1 = tf.random_uniform([1], minval=0.0, maxval=1.0)
        #gama2 = tf.random_uniform([1], minval=0.0, maxval=1.0)



        # gama01 = tf.cast(tf.random_uniform([4096], minval=0, maxval=1, dtype=tf.float32), dtype=tf.float32)
        # gama02 = tf.cast(tf.random_uniform([4096], minval=0, maxval=1, dtype=tf.float32), dtype=tf.float32)
        # gama03 = tf.cast(tf.random_uniform([4096], minval=0, maxval=1, dtype=tf.float32), dtype=tf.float32)
        # gama1 = tf.cast(tf.random_uniform([4096], minval=0, maxval=2, dtype=tf.int32), dtype=tf.float32)
        # gama2 = tf.cast(tf.random_uniform([4096], minval=0, maxval=2, dtype=tf.int32), dtype=tf.float32)
        # gama3 = tf.cast(tf.random_uniform([4096], minval=0, maxval=1, dtype=tf.float32), dtype=tf.float32)
        # gama4 = tf.cast(tf.random_uniform([4096], minval=0, maxval=1, dtype=tf.float32), dtype=tf.float32)
        # gama5 = tf.cast(tf.random_uniform([4096], minval=0, maxval=1, dtype=tf.float32), dtype=tf.float32)
        # gama6 = tf.cast(tf.random_uniform([4096], minval=0, maxval=1, dtype=tf.float32), dtype=tf.float32)
        # gama7 = tf.cast(tf.random_uniform([4096], minval=0, maxval=1, dtype=tf.float32), dtype=tf.float32)
   #       9216 9216 9216  9216    4096
        gama01 = tf.cast(tf.random_uniform([4096], minval=-10, maxval=10, dtype=tf.float32), dtype=tf.float32)
        gama02 = tf.cast(tf.random_uniform([4096], minval=-10, maxval=10, dtype=tf.float32), dtype=tf.float32)
        gama03 = tf.cast(tf.random_uniform([4096], minval=-10, maxval=10, dtype=tf.float32), dtype=tf.float32)
        gama1 = tf.cast(tf.random_uniform([4096], minval=-10, maxval=10, dtype=tf.float32), dtype=tf.float32)
        gama2 = tf.cast(tf.random_uniform([4096], minval=-10, maxval=10, dtype=tf.float32), dtype=tf.float32)
        gama3 = tf.cast(tf.random_uniform([4096], minval=-10, maxval=10, dtype=tf.float32), dtype=tf.float32)
        gama4 = tf.cast(tf.random_uniform([4096], minval=-10, maxval=10, dtype=tf.float32), dtype=tf.float32)
        gama5 = tf.cast(tf.random_uniform([4096], minval=-10, maxval=10, dtype=tf.float32), dtype=tf.float32)
        gama6 = tf.cast(tf.random_uniform([4096], minval=-10, maxval=10, dtype=tf.float32), dtype=tf.float32)
        gama7 = tf.cast(tf.random_uniform([4096], minval=-10, maxval=10, dtype=tf.float32), dtype=tf.float32)
        with tf.variable_scope("gama"):
            gama1 = tf.nn.sigmoid(tf.Variable(gama1, name="gama1"))
            gama2 = tf.nn.sigmoid(tf.Variable(gama2, name="gama2"))
            gama01 = tf.nn.sigmoid(tf.Variable(gama01, name="gama01"))
            gama02 = tf.nn.sigmoid(tf.Variable(gama02, name="gama02"))
            gama03 = tf.nn.sigmoid(tf.Variable(gama03, name="gama03"))
            gama3 = tf.nn.sigmoid(tf.Variable(gama3, name="gama3"))
            gama4 = tf.nn.sigmoid(tf.Variable(gama4, name="gama4"))
            gama5 = tf.nn.sigmoid(tf.Variable(gama5, name="gama5"))
            gama6 = tf.nn.sigmoid(tf.Variable(gama6, name="gama6"))
            gama7 = tf.nn.sigmoid(tf.Variable(gama7, name="gama7"))

        for samplei_real in range(batch_size):
            # if samplei_real%2==0:
            #     gama1 = tf.random_uniform([1],minval=0.9,maxval=1.0)
            # else:
            distance_t_t = tf.reduce_sum(tf.multiply(target_im_batch_256, target_im_batch_256[samplei_real]), axis=-1)
            distance_t_t = distance_t_t / tf.sqrt(tf.reduce_sum(tf.square(target_im_batch_256[samplei_real]), axis=-1))
            distance_t_t = distance_t_t / tf.sqrt(tf.reduce_sum(tf.square(target_im_batch_256), axis=-1))

            #distance_t_t = tf.reduce_sum(tf.square(target_im_batch - target_im_batch[samplei_real]), axis=-1)


            distance_t_s = tf.reduce_sum(tf.multiply(source_im_batch_256, target_im_batch_256[samplei_real]), axis=-1)
            distance_t_s = distance_t_s / tf.sqrt(tf.reduce_sum(tf.square(target_im_batch_256[samplei_real]), axis=-1))
            distance_t_s = distance_t_s / tf.sqrt(tf.reduce_sum(tf.square(source_im_batch_256), axis=-1))
            #distance_t_s = tf.reduce_sum(tf.square(source_im_batch - target_im_batch[samplei_real]), axis=-1)


            values_t_t, pred_index_t_t = tf.nn.top_k(distance_t_t, k=neighbor_num + 1, sorted=True)
            values_t_s, pred_index_t_s = tf.nn.top_k(distance_t_s, k=neighbor_num, sorted=True)
            distance_t_t_total.append(values_t_t[1])
            distance_t_s_total.append(values_t_s[0])

            pred_index_t_t_total.append(pred_index_t_t[1])
            pred_index_t_s_total.append(pred_index_t_s[0])
            #### find the target neighbour from target
            target_im_batch_mix.append( (1-gama01)*target_im_batch[samplei_real] \
                                            + gama01*target_im_batch[pred_index_t_t[1]])
            target_im_batch_mix2.append((1 - gama02) * target_im_batch[samplei_real] \
                                       + gama02 * target_im_batch[pred_index_t_t[1]])
            target_im_batch_target.append(target_im_batch[pred_index_t_t[1]])

            source_im_batch_target.append(source_im_batch[pred_index_t_s[0]])
            source_im_batch_samelabel_target.append(source_im_batch_samelabel[pred_index_t_s[0]])
            source_im_batch_mix.append((1 - gama03) * source_im_batch[pred_index_t_s[0]] \
                                            + gama03 * source_im_batch_samelabel[pred_index_t_s[0]])

            source_differ_im_batch_target.append(source_im_batch_differ1[pred_index_t_s[0]])

            label_batch_target_target.append(target_label_batch[pred_index_t_t[1]])
            label_batch_source_mix.append(source_label_batch[pred_index_t_s[0]])


        source_im_batch_target = tf.convert_to_tensor(source_im_batch_target)
        source_im_batch_samelabel_target = tf.convert_to_tensor(source_im_batch_samelabel_target)
        target_im_batch_mix = tf.convert_to_tensor(target_im_batch_mix)
        target_im_batch_mix2 = tf.convert_to_tensor(target_im_batch_mix2)
        target_im_batch_target = tf.convert_to_tensor(target_im_batch_target)
        source_im_batch_mix = tf.convert_to_tensor(source_im_batch_mix)
        label_batch_source_mix= tf.convert_to_tensor(label_batch_source_mix)
        label_batch_target_target = tf.convert_to_tensor(label_batch_target_target)
        distance_t_t_total = tf.convert_to_tensor(distance_t_t_total)
        distance_t_s_total=tf.convert_to_tensor(distance_t_s_total)
        print ("-----distance_t_t_total:",distance_t_t_total)

        distance_t_t_total=tf.expand_dims(distance_t_t_total,axis=-1)
        distance_t_s_total = tf.expand_dims(distance_t_s_total, axis=-1)



        source_differ_im_batch_target = tf.convert_to_tensor(source_differ_im_batch_target)

        #gama1 = tf.cast(tf.random_uniform([4096], minval=0, maxval=1, dtype=tf.float32), dtype=tf.float32)
        #gama1 = tf.random_uniform([1], minval=0.0, maxval=1.0)

        source_im_batch_mix_samelabel = (1 - gama1) * source_im_batch_samelabel + gama1 * source_im_batch
        #gama3 = tf.random_uniform([1], minval=0.0, maxval=1.0)
        source_im_batch_mix_samelabel2 = (1 - gama2) * source_im_batch_samelabel + gama2 * source_im_batch

        #gama2 = tf.cast(tf.random_uniform([4096], minval=0, maxval=1, dtype=tf.float32), dtype=tf.float32)
        #gama2 = tf.random_uniform([1], minval=0.0, maxval=1.0)

        source_target_mix = (1 - gama3) * target_im_batch + gama3 * source_im_batch_target
        #gama4 = tf.cast(tf.random_uniform([4096], minval=0, maxval=1, dtype=tf.float32), dtype=tf.float32)
        #gama4 = tf.random_uniform([1], minval=0.0, maxval=1.0)
        source_target_mix2 = (1 - gama4) * target_im_batch + gama4 * source_im_batch_samelabel_target
        source_im_batch_samelabel_mix_target = (1 - gama7) * source_im_batch_target + gama7 * source_im_batch_samelabel_target

        #gama5 = tf.cast(tf.random_uniform([4096], minval=0, maxval=1, dtype=tf.float32), dtype=tf.float32)
        #gama5 = tf.random_uniform([1], minval=0.0, maxval=1.0)

        source_differ_im_batch_mix = (1 - gama5) * source_im_batch_differ1 + gama5 * source_im_batch_differ2
        #gama6 = tf.cast(tf.random_uniform([4096], minval=0, maxval=1, dtype=tf.float32), dtype=tf.float32)
        #gama6 = tf.random_uniform([1], minval=0.0, maxval=1.0)
        source_differ_target_mix = (1 - gama6) * target_im_batch + gama6 * source_differ_im_batch_target




        source_ft_mix_samelabel, layers_s_dp_mix_samelabel = \
            Alex_fcn2(source_im_batch_mix_samelabel, scope='source', reuse=True,dropout_keepratio=dropoupratio)
        source_ft_mix_samelabel2, layers_s_dp_mix_samelabel2 =\
            Alex_fcn2(source_im_batch_mix_samelabel2, scope='source',reuse=True,dropout_keepratio=dropoupratio)
        source_ft_mix, layers_s_dp_mix = Alex_fcn2(source_im_batch_mix, scope='source', reuse=True,dropout_keepratio=dropoupratio)
        source_ft_target, layers_s_dp_target = Alex_fcn2(source_im_batch_target, scope='source', reuse=True,dropout_keepratio=dropoupratio)
        source_ft_samelabel_target, layers_s_samelabel_dp_target = \
            Alex_fcn2(source_im_batch_samelabel_target, scope='source', reuse=True,dropout_keepratio=dropoupratio)
        source_ft_samelabel_mix_target, layers_s_samelabel_mix_dp_target = \
            Alex_fcn2(source_im_batch_samelabel_mix_target,scope='source', reuse=True,dropout_keepratio=dropoupratio)

        target_ft_target, layers_t_t_dp = Alex_fcn2(target_im_batch_target, scope='source', reuse=True,dropout_keepratio=dropoupratio)
        target_ft_mix, layers_t_dp_mix = Alex_fcn2(target_im_batch_mix, scope='source', reuse=True,dropout_keepratio=dropoupratio)
        target_ft_mix2, layers_t_dp_mix2 = Alex_fcn2(target_im_batch_mix2, scope='source', reuse=True,dropout_keepratio=dropoupratio)
        source_differ1_ft, layers_s_differ1_dp2 = Alex_fcn2(source_im_batch_differ1, scope='source', reuse=True,dropout_keepratio=dropoupratio)
        source_differ2_ft, layers_s_differ2_dp2 = Alex_fcn2(source_im_batch_differ2, scope='source', reuse=True,dropout_keepratio=dropoupratio)

        source_target_mix_ft, layers_s_t_mix_dp = Alex_fcn2(source_target_mix, scope='source', reuse=True,dropout_keepratio=dropoupratio)
        source_target_mix2_ft, layers_s_t_mix2_dp = Alex_fcn2(source_target_mix2, scope='source', reuse=True,dropout_keepratio=dropoupratio)

        source_differ_mix_ft, layers_s_differ_mix_dp = Alex_fcn2(source_differ_im_batch_mix, scope='source', reuse=True,dropout_keepratio=dropoupratio)
        source_differ_target_mix_ft, layers_s_differ_t_mix_dp = Alex_fcn2(source_differ_target_mix, scope='source', reuse=True,dropout_keepratio=dropoupratio)

        # correct_prediction_target_target = tf.equal(tf.cast(target_label_batch, dtype="int64"),
        #                                             tf.cast(label_batch_target_target, dtype="int64"))
        # correct_prediction_target_target1 = tf.equal(tf.argmax(target_ft, -1),
        #                                             tf.argmax(target_ft_mix, -1))
        correct_prediction_target_target1 = tf.equal(tf.cast(target_label_batch, dtype="int64"),
                                                     tf.cast(label_batch_target_target, dtype="int64"))
        accuracy_t_t1 = tf.reduce_mean(tf.cast(correct_prediction_target_target1, 'float'))
        correct_prediction_target_target2 = tf.equal(tf.argmax(target_ft_mix, -1),
                                                    tf.argmax(target_ft_mix2, -1))
        accuracy_t_t2 = tf.reduce_mean(tf.cast(correct_prediction_target_target2, 'float'))
        accuracy_t_t=accuracy_t_t1

        # correct_prediction_target_source = tf.equal(tf.cast(target_label_batch, dtype="int64"),
        #                                             tf.cast(label_batch_source_mix, dtype="int64"))
        correct_prediction_target_source1 = tf.equal(tf.cast(target_label_batch, dtype="int64"),
                                                    tf.cast(label_batch_source_mix, dtype="int64"))

        accuracy_t_s1 = tf.reduce_mean(tf.cast(correct_prediction_target_source1, 'float'))
        correct_prediction_target_source2 = tf.equal(tf.argmax(source_target_mix_ft, -1),
                                                    tf.argmax(source_target_mix2_ft, -1))
        accuracy_t_s2= tf.reduce_mean(tf.cast(correct_prediction_target_source2, 'float'))
        accuracy_t_s=accuracy_t_s1

        distance_threshlod=1.0
        distance_threshlod_min=0.0

        source_ft_gan = layers_s_dp2[adapt_layer]
        source_ft_samelabel_gan = layers_s_dp_samelabel[adapt_layer]
        source_ft_mix_gan_samelabel = layers_s_dp_mix_samelabel[adapt_layer]
        source_ft_mix_gan_samelabel2 = layers_s_dp_mix_samelabel2[adapt_layer]

        source_differ1_ft_gan = layers_s_differ1_dp2[adapt_layer]
        #source_differ2_ft_gan = layers_s_differ2_dp2[adapt_layer]
        source_differ_mix_ft_gan = layers_s_differ_mix_dp[adapt_layer]

        source_differ_target_mix_ft_gan=layers_s_differ_t_mix_dp[adapt_layer]

        target_ft_gan = layers_t_dp2[adapt_layer]
        target_ft_mix_gan = layers_t_dp_mix[adapt_layer]
        target_ft_mix2_gan = layers_t_dp_mix2[adapt_layer]
        source_ft_gan_samelabel_target = layers_s_samelabel_dp_target[adapt_layer]
        source_ft_gan_samelabel_mix_target=layers_s_samelabel_mix_dp_target[adapt_layer]
        source_ft_gan_target = layers_s_dp_target[adapt_layer]
        target_ft_gan_target = layers_t_t_dp[adapt_layer]
        source_ft_mix_gan = layers_s_dp_mix[adapt_layer]
        source_target_mix_ft_gan = layers_s_t_mix_dp[adapt_layer]
        source_target_mix2_ft_gan = layers_s_t_mix2_dp[adapt_layer]

        source_ft_softmax = tf.nn.softmax(source_ft)
        source_uncertaity = -tf.reduce_sum(source_ft_softmax * tf.log(source_ft_softmax + 1e-10), axis=-1)
        source_certainty = tf.exp(-source_uncertaity)
        source_certainty = tf.nn.softmax(source_certainty) * batch_size
        source_certainty=tf.expand_dims(source_certainty,axis=-1)

        print ("-----source_certainty:",source_certainty)

        target_ft_softmax = tf.nn.softmax(target_ft)
        target_uncertaity = -tf.reduce_sum(target_ft_softmax * tf.log(target_ft_softmax + 1e-10), axis=-1)
        target_certainty = tf.exp(-target_uncertaity)
        target_certainty=tf.nn.softmax(target_certainty)*batch_size
        target_certainty = tf.expand_dims(target_certainty, axis=-1)

        print("-----target_certainty:", target_certainty)


        distance_semantic_differ_mix = tf.reduce_sum(source_certainty*tf.maximum(0.0,distance_threshlod
                - tf.reduce_sum(tf.square(source_differ_mix_ft_gan - source_ft_mix_gan_samelabel), -1))) / batch_size
        distance_semantic_differ = tf.reduce_sum(source_certainty*tf.maximum(0.0, distance_threshlod -
                tf.reduce_sum(tf.square(source_differ1_ft_gan - source_ft_gan), -1))) / batch_size
        distance_semantic_source_differ_target_mix = tf.reduce_sum(tf.maximum(0.0, distance_threshlod -
                    tf.reduce_sum(tf.square(target_ft_gan - source_differ_target_mix_ft_gan),
                                                                -1))) / batch_size

        distance_semantic_differ_show = tf.reduce_sum(tf.square(source_differ1_ft_gan - source_ft_gan)) / batch_size

        distance_semantic_s_s_samelabel = tf.reduce_sum\
                            (source_certainty*tf.square(source_ft_gan - source_ft_samelabel_gan)) / batch_size
        distance_semantic_s_s_mix_samelabel = tf.reduce_sum\
                    (source_certainty*tf.square(source_ft_mix_gan_samelabel - source_ft_mix_gan_samelabel2)) / batch_size

        distance_semantic_s_t1 = tf.reduce_sum(target_certainty*distance_t_s_total*tf.square
                        (target_ft_gan - source_ft_gan_target)) / batch_size
        distance_semantic_s_t_samelabel = tf.reduce_sum(target_certainty * distance_t_s_total * tf.square
        (target_ft_gan - source_ft_gan_samelabel_target)) / batch_size

        distance_semantic_s_t=distance_semantic_s_t1*0.8+distance_semantic_s_t_samelabel*0.2


        distance_semantic_st_mix1 = tf.reduce_sum(
            target_certainty*distance_t_s_total * tf.square(source_target_mix_ft_gan - source_target_mix2_ft_gan)) / batch_size
        distance_semantic_st_mix_samelabel = tf.reduce_sum(
            target_certainty * distance_t_s_total * tf.square(
                source_target_mix_ft_gan - source_ft_gan_samelabel_mix_target)) / batch_size

        distance_semantic_st_mix=distance_semantic_st_mix1*0.8+distance_semantic_st_mix_samelabel*0.2

        distance_semantic_t_t = tf.reduce_sum(target_certainty*distance_t_t_total*tf.square
                        (target_ft_gan - target_ft_gan_target)) / batch_size
        distance_semantic_t_t_mix = tf.reduce_sum(
            target_certainty*distance_t_t_total * tf.square(target_ft_mix2_gan - target_ft_mix_gan)) / batch_size

        distance_semantic = (distance_semantic_s_s_samelabel + distance_semantic_s_s_mix_samelabel+
                             distance_semantic_differ_mix+distance_semantic_differ) /4.0
        distance_semantic_st = (distance_semantic_s_t*1.6+distance_semantic_t_t*0.4)/2.0
        distance_semantic_st_mix = (distance_semantic_st_mix*1.6+distance_semantic_t_t_mix*0.4)/2.0



        super_loss = tf.losses.sparse_softmax_cross_entropy(source_label_batch, source_ft )
        super_loss_samelabel = tf.losses.sparse_softmax_cross_entropy(source_label_batch, source_ft_samelabel)
        super_loss_mix_samelabel = tf.losses.sparse_softmax_cross_entropy(source_label_batch, source_ft_mix_samelabel)
        super_loss_mix_samelabel2 = tf.losses.sparse_softmax_cross_entropy(source_label_batch, source_ft_mix_samelabel2)
        super_loss_mix_target = tf.losses.sparse_softmax_cross_entropy(label_batch_source_mix, source_ft_mix)
        #super_loss=(super_loss+super_loss_mix+super_loss_samelabel)/3.0
        super_loss_differ1 = tf.losses.sparse_softmax_cross_entropy(source_label_differ1, source_differ1_ft)
        super_loss_differ2 = tf.losses.sparse_softmax_cross_entropy(source_label_differ2, source_differ2_ft)
        if semantic_mix_flag==1:
            super_loss = super_loss * 0.15  + super_loss_samelabel * 0.15\
                         +super_loss_differ1*0.15+super_loss_differ2*0.15\
                         +super_loss_mix_samelabel*0.2  +super_loss_mix_samelabel2*0.2
        else:
            super_loss = (super_loss +super_loss_differ1+super_loss_differ2)/3.0




        correct_prediction = tf.equal(tf.argmax(source_ft, -1), tf.cast(source_label_batch, dtype="int64"))
        correct_prediction_target = tf.equal(tf.argmax(target_ft, -1), tf.cast(target_label_batch, dtype="int64"))
        accuracy_target = tf.reduce_mean(tf.cast(correct_prediction_target, 'float'))
        accuracy_source = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        source_adversary_label = tf.ones([batch_size, 1], tf.float32)
        target_adversary_label = tf.zeros([batch_size, 1], tf.float32)
        if semantic_mix_flag == 1:

            adversary_ft = tf.concat([source_ft_gan,source_ft_mix_gan_samelabel,source_ft_mix_gan_samelabel2,
                                      target_ft_gan, target_ft_gan,source_ft_mix_gan_samelabel2], 0)   #  target_ft_mix_gan  source_ft_mix_gan_samelabel2
            adversary_label = tf.concat(
                [source_adversary_label, source_adversary_label,source_adversary_label,
                 target_adversary_label,target_adversary_label,target_adversary_label], 0)
        else:
            adversary_ft = tf.concat([source_ft_gan,target_ft_gan], 0)
            adversary_label = tf.concat([source_adversary_label, target_adversary_label], 0)


        print("tf.shape(source_ft_gan)[0]:", tf.shape(source_ft_gan)[0])
        print("adversary_label:", adversary_label)
        print("tf.shape(adversary_label):", tf.shape(adversary_label))
        # source_adversary_label = tf.zeros([tf.shape(source_ft_gan)[0]]*2, tf.int32)
        if adapt_layer == "fc8" or netvladflag == 0:
            adver_layer = [1024, 1024]
        else:
            adver_layer = [1024, 1024]

        adversary_logits, adversary_layerfeature = adda.adversary.adversarial_discriminator(
            net=adversary_ft, layers=adver_layer, lamda_w=weights_mapping, leaky=False,
            scope="local_adversary", dropout_keep=0.5)
        adversary_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=adversary_logits, labels=adversary_label)
        mapping_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=adversary_logits, labels=1 - adversary_label)
        mapping_loss = -adversary_loss

        mapping_loss_mmd = adda.util.mmd_loss(source_ft_gan, target_ft_gan, 1.0)

        adversary_logits_softmax = adversary_logits

        return source_ft, target_ft, super_loss, accuracy_source, \
               accuracy_target, mapping_loss, adversary_loss, mapping_loss_mmd, \
               distance_semantic, adversary_logits, \
                accuracy_t_t,accuracy_t_s,distance_semantic_st,distance_semantic_st_mix,distance_semantic_t_t_mix,distance_semantic_differ_show

    with tf.Graph().as_default(), tf.device('/gpu:0'):
        source_im=tf.placeholder(dtype=tf.float32, shape=(batch_size,256,256,3), name='source_im')

        source_im_samelabel = tf.placeholder(dtype=tf.float32, shape=(batch_size, 256, 256, 3), name='source_im_samelabel')
        source_label = tf.placeholder(dtype=tf.int32, shape=(batch_size), name='source_label')
        target_im = tf.placeholder(dtype=tf.float32, shape=(batch_size, 256, 256, 3), name='target_im')
        dropoupratio = tf.placeholder(dtype=tf.float32, name='dropout_ratio')

        target_label = tf.placeholder(dtype=tf.int32, shape=(batch_size), name='target_label')
        if training_mode=="one_target":
            (source_images, source_labels,source_category_total)= load_datasets(datasetname[0])
            (target_images, target_labels,target_category_total)= load_datasets(target_dataset_name)
        else:
            (source_images, source_labels, source_category_total,
             target_images, target_labels,target_category_total) = load_datasets_one_multi(datasetname[0],target_dataset_name)

        print ("----len(source_images): ",len(source_images))
        print("----len(target_images): ", len(target_images))

        model_fn = adda.models.get_model_fn(model)



        source_im_differ1 = tf.placeholder(dtype=tf.float32, shape=(batch_size, 256, 256, 3), name='source_im_differ1')
        source_label_differ1 = tf.placeholder(dtype=tf.int32, shape=(batch_size), name='source_label_differ1')
        source_im_differ2 = tf.placeholder(dtype=tf.float32, shape=(batch_size, 256, 256, 3), name='source_im_differ2')
        source_label_differ2 = tf.placeholder(dtype=tf.int32, shape=(batch_size), name='source_label_differ2')

        weights_uncertain = tf.Variable(0.0, name='weights_uncertain', trainable=False)
        weights_uncertain_before = tf.Variable(0.0, name='weights_uncertain', trainable=False)
        weights_mapping = tf.Variable(0.0, name='weights_mapping', trainable=False)
        tempe_uncertain = tf.Variable(tempe_uncertain_value, name='tempe_uncertain', trainable=False)
        weight_classifier = tf.Variable(1.0, name='weight_classifier', trainable=False)

        weights_semantic_st = tf.Variable(0.0, name='weights_semantic_st', trainable=False)
        weights_semantic_st_mix = tf.Variable(0.0, name='weights_semantic_st_mix', trainable=False)
        source_super_loss = []
        source_acc = []
        target_acc = []
        target_ft_total_ori = []
        source_ft_total_ori = []
        if uncertainty_flag == True:
            gpu_visible = [0,1, 2]
        else:
            gpu_visible = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        lr_var = tf.Variable(lr, name='learning_rate', trainable=False)



        if optimizer_name == "sgd":
            optimizer = tf.train.MomentumOptimizer(lr_var, 0.9)
            optimizer_fcn = tf.train.MomentumOptimizer(lr_var * 10.0, 0.9)
            print("using sgd......")
        else:
            optimizer = tf.train.AdamOptimizer(lr_var, 0.5)
            optimizer_fcn = tf.train.AdamOptimizer(lr_var * 10.0, 0.5)
            print("using Adam......")

        mapping_loss = []
        adversary_loss = []
        mapping_loss_mmd = []
        distance_semantic = []
        source_ft_total_logits = []
        target_ft_total_logits = []
        adversary_logits_total = []
        accuracy_t_t=[]
        accuracy_t_s=[]
        distance_semantic_st=[]
        distance_semantic_st_mix=[]
        distance_semantic_tt_mix = []
        distance_semantic_differ_show=[]

        source_im_batch = train_image_process(source_im)
        source_im_batch_samelabel = train_image_process(source_im_samelabel)
        source_label_batch = source_label
        target_im_batch = train_image_process(target_im)

        source_im_batch_differ1 = train_image_process(source_im_differ1)
        source_im_batch_differ2 = train_image_process(source_im_differ2)
        target_label_batch = target_label
        with tf.device('/gpu:%d' % gpu_visible[1]):
            source_im_batch, layer_source = model_fn(source_im_batch, scope='source')
            source_im_batch_samelabel, layer_target_samelabel = model_fn(source_im_batch_samelabel, scope='source', reuse=True)
            target_im_batch, layer_target = model_fn(target_im_batch, scope='source', reuse=True)

            source_im_batch_differ1, layer_source_differ1 = model_fn(source_im_batch_differ1, scope='source',
                                                                     reuse=True)
            source_im_batch_differ2, layer_source_differ2 = model_fn(source_im_batch_differ2, scope='source',
                                                                     reuse=True)

        with tf.variable_scope(tf.get_variable_scope()):
            for gpui in range(2, len(gpu_visible)):
                with tf.device('/gpu:%d' % gpu_visible[gpui]):
                    with tf.name_scope('Tower_%d' % (gpu_visible[gpui])) as scope:
                        source_ft, target_ft, super_loss, accuracy_source, accuracy_target_temp, mapping_loss_temp, \
                        adversary_loss_temp, mapping_loss_mmd_temp, distance_semantic_temp, adversary_logits_temp , \
                        accuracy_t_t_temp,accuracy_t_s_temp,distance_semantic_st_temp,\
                        distance_semantic_st_mix_temp ,distance_semantic_tt_mix_temp, \
                        distance_semantic_differ_show_temp                            \
                            = inference(source_im_batch, source_label_batch, target_im_batch, target_label_batch,
                                        source_im_batch_samelabel,
                                        source_im_batch_differ1, source_label_differ1,
                                        source_im_batch_differ2, source_label_differ2,dropoupratio=dropoupratio)
                        tf.get_variable_scope().reuse_variables()
                        target_ft_total_ori.append(target_ft)
                        source_ft_total_ori.append(source_ft)
                        accuracy_t_t.append(accuracy_t_t_temp)
                        accuracy_t_s.append(accuracy_t_s_temp)
                        source_ft_total_logits.append(tf.nn.softmax(source_ft))
                        target_ft_total_logits.append(tf.nn.softmax(target_ft))
                        distance_semantic.append(distance_semantic_temp)
                        source_super_loss.append(super_loss)
                        source_acc.append(accuracy_source)
                        target_acc.append(accuracy_target_temp)
                        mapping_loss.append(mapping_loss_temp)
                        mapping_loss_mmd.append(mapping_loss_mmd_temp)
                        adversary_loss.append(adversary_loss_temp)
                        adversary_logits_total.append(adversary_logits_temp)
                        distance_semantic_st.append(distance_semantic_st_temp)
                        distance_semantic_st_mix.append(distance_semantic_st_mix_temp)
                        distance_semantic_differ_show.append(distance_semantic_differ_show_temp)
                        distance_semantic_tt_mix.append(distance_semantic_tt_mix_temp)

        source_super_loss = tf.reduce_mean(source_super_loss)
        mapping_loss = tf.reduce_mean(mapping_loss)
        mapping_loss_mmd = tf.reduce_mean(mapping_loss_mmd)

        source_acc = tf.reduce_mean(source_acc)
        target_acc = tf.reduce_mean(target_acc)
        accuracy_t_t = tf.reduce_mean(accuracy_t_t)
        accuracy_t_s = tf.reduce_mean(accuracy_t_s)
        distance_semantic = tf.reduce_mean(distance_semantic)
        distance_semantic_st = tf.reduce_mean(distance_semantic_st)
        distance_semantic_st_mix = tf.reduce_mean(distance_semantic_st_mix)
        distance_semantic_tt_mix=tf.reduce_mean(distance_semantic_tt_mix)
        distance_semantic_differ_show = tf.reduce_mean(distance_semantic_differ_show)
        print("_____len(target_ft_total):", len(target_ft_total_ori))
        print("_____target_ft_total[0]:", target_ft_total_ori[0])
        print("_____target_ft_total:", target_ft_total_ori)

        # source_ft_total_logits=tf.convert_to_tensor(source_ft_total_logits)

        target_ft_total = tf.convert_to_tensor(tf.nn.softmax(target_ft_total_ori / tempe_uncertain))
        source_ft_total = tf.convert_to_tensor(tf.nn.softmax(source_ft_total_ori / tempe_uncertain))
        # adversary_logits_total=tf.convert_to_tensor(tf.nn.softmax(adversary_logits_total))

        print("target_ft_total:", target_ft_total)

        source_ft_total_logits = tf.convert_to_tensor(source_ft_total_logits)
        target_ft_total_logits = tf.convert_to_tensor(target_ft_total_logits)
        source_ft_total_logits = tf.argmax(source_ft_total_logits, axis=-1)
        target_ft_total_logits = tf.argmax(target_ft_total_logits, axis=-1)



        source_ft_total_mean, source_ft_total_var = tf.nn.moments(source_ft_total, axes=0)
        target_ft_total_mean, target_ft_total_var = tf.nn.moments(target_ft_total, axes=0)



        adversary_loss = tf.reduce_mean(adversary_loss)

        # adversary_logits_total_mean, adversary_logits_total_var = tf.nn.moments(adversary_logits_total, axes=0)
        print("source_ft_total_mean:", source_ft_total_mean)
        print("source_ft_total_var:", source_ft_total_var)

        target_ft_mean_batch = tf.reduce_mean(target_ft_total_mean, axis=0)
        print("target_ft_mean_batch:", target_ft_mean_batch)

        adversary_logits_total = tf.reduce_mean(adversary_logits_total, axis=0)
        # print("_____len(target_ft_total)_after:", len(target_ft_total))
        print("_____target_ft_total[0]_after:", target_ft_total[0])
        # print("_____target_ft_total[1]_after:", target_ft_total[1])
        print("_____ltarget_ft_total_after:", target_ft_total)

        print("_____target_ft_total:", target_ft_total)

        source_ft_total_var = tf.reduce_sum(source_ft_total_var)
        target_ft_total_var = tf.reduce_sum(target_ft_total_var)

        loss_balacelabel = -tf.reduce_sum(target_ft_mean_batch * tf.log(target_ft_mean_batch + 1e-10)) / tf.log(
            2.0) * 1.0

        #
        # loss_uncertainty_adver = -tf.reduce_sum(adversary_logits_total_mean * tf.log(adversary_logits_total_mean + 1e-10)) / tf.log(
        #     2.0) * 1.0 / batch_size

        if uncertain_metric == "var":
            print("using var metric----------------")
            loss_uncertainty_source = source_ft_total_var / batch_size
            loss_uncertainty = target_ft_total_var / batch_size
        else:
            print("using  entropy metric----------------")

            loss_uncertainty = -tf.reduce_sum(target_ft_total_mean *
                                              tf.log(target_ft_total_mean + 1e-10)) / batch_size
            # +loss_uncertainty_temp  tf.log(2.0) *
            loss_uncertainty_source = -tf.reduce_sum(
                source_ft_total_mean * tf.log(source_ft_total_mean + 1e-10)) / batch_size  # tf.log(2.0)

        vars = tf.trainable_variables()

        dropout_ratio = 0.5
        tau = 1.0
        lengthscale = 2.0
        # N = len(source_dataset)
        # N=2000
        N = max(2000, len(source_images)+len(target_images))
        # N = 2000
        print("------------N:", N)
        weights_uncertain_value = 0.1
        weights_uncertain_value_before = 0.1
        print("____N:", N)
        reg = (1 - dropout_ratio) / (2. * N * tau)

        clip_max_value = 7.0  ##############gradient clip

        loss_L2 = reg * tf.add_n([tf.nn.l2_loss(v) for v in vars
                                  if ('weights' in v.name) and (
                                          ('fc8' in v.name) or ('fc7' in v.name) or ('fc6' in v.name) or (
                                              'fc_adapt' in v.name))])

        l2_loss_total = 0.0001 * tf.add_n(
            [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])

        total_loss = source_super_loss + 0.0 * mapping_loss \
                     + loss_L2 - loss_uncertainty * weights_uncertain_value

        total_loss = source_super_loss +adversary_loss*adverflag  \
                    # + semantic_mix_flag*weights_mapping * 0.1 * distance_semantic

        classifer_loss = source_super_loss   \
                         + semantic_mix_flag*weights_mapping * 0.1 * \
                         (distance_semantic * (1.0 - weights_semantic_st - weights_semantic_st_mix)
                          + distance_semantic_st * weights_semantic_st
                          + distance_semantic_st_mix * weights_semantic_st_mix)
        softmax_loss = source_super_loss
        gama_loss=-distance_semantic * (1.0 - weights_semantic_st - weights_semantic_st_mix) \
                          + distance_semantic_st * weights_semantic_st  \
                          + distance_semantic_st_mix * weights_semantic_st_mix

        # - 0.1 * loss_uncertainty_source

        # total_loss = source_super_loss+weights_mapping*(mapping_loss)
        # classifer_loss=source_super_loss+weights_mapping*(mapping_loss)

        train_variables = tf.trainable_variables()
        adversary_vars = [v for v in train_variables if "local_adversary" in v.name]

        encoder_vars = [v for v in train_variables if
                        ("conv1" not in v.name) and ("conv2" not in v.name) and ("conv3" not in v.name) and
                        ("fc8" not in v.name) and ("fc_adapt" not in v.name) and ("conv4fdfsdf" not in v.name)
                        and ("conv53" not in v.name) and ("conv53343" not in v.name) and ("gama" not in v.name)]
        classifier_vars = [v for v in train_variables if "fc_adapt" in v.name ]
        softmax_vars = [v for v in train_variables if "fc8" in v.name]
        gama_vars = [v for v in train_variables if "gama" in v.name]

        # step = optimizer.minimize(total_loss, var_list=encoder_vars)
        # step_classifier = optimizer_fcn.minimize(classifer_loss, var_list=classifier_vars)

        grads_encoder = optimizer.compute_gradients(total_loss, var_list=encoder_vars)
        for i, (g, v) in enumerate(grads_encoder):
            if g is not None:
                grads_encoder[i] = (tf.clip_by_norm(g, clip_max_value),
                                    v)  # (tf.clip_by_norm(g, 10), v)  tf.clip_by_value(g, -clip_max_value, clip_max_value)
        step = optimizer.apply_gradients(grads_encoder)

        grads_c = optimizer_fcn.compute_gradients(classifer_loss, var_list=classifier_vars)
        for i, (g, v) in enumerate(grads_c):
            if g is not None:
                grads_c[i] = (tf.clip_by_norm(g, clip_max_value), v)
        step_classifier = optimizer_fcn.apply_gradients(grads_c)

        grads_softmax = optimizer_fcn.compute_gradients(softmax_loss, var_list=softmax_vars)
        for i, (g, v) in enumerate(grads_softmax):
            if g is not None:
                grads_softmax[i] = (tf.clip_by_norm(g, clip_max_value), v)
        step_softmax = optimizer_fcn.apply_gradients(grads_softmax)

        step_gama=optimizer.minimize(gama_loss,var_list=gama_vars)


        # set up session and initialize
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(init)

        train_variables = tf.trainable_variables()
        logging.info('    Restoring model from Alexnet:')

        if weights:
            print("____________pretraining, retore from bvlc_alexnet.npy")
            load_initial_weights(session=sess, path_file=weights + "/bvlc_alexnet.npy", scope="source")

        var_dict_save = adda.util.collect_vars("source")
        target_saver = tf.train.Saver(var_list=var_dict_save)

        output_dir = os.path.join('snapshot', output)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if training_mode=="one_target" and datasetname[0]=="amazon":
            keep_length=9
        else:
            keep_length=27
        source_acc_val_total = deque(maxlen=10)
        target_acc_val_total = deque(maxlen=keep_length)
        target_acc_val_total_test = deque(maxlen=keep_length)
        uncertainty_val_total = deque(maxlen=10)
        source_uncertainty_val_total = deque(maxlen=10)
        acc_t_t_total = deque(maxlen=10)
        acc_t_s_total = deque(maxlen=10)

        distance_t_t_val_total = deque(maxlen=9)
        distance_s_t_val_total = deque(maxlen=9)
        distance_s_s_val_total = deque(maxlen=9)
        distance_t_t_val_total_save=[]
        distance_s_t_val_total_save=[]
        distance_s_s_val_total_save=[]


        bar = tqdm(range(iterations))
        bar.set_description('{} (lr: {:.0e})'.format(output, lr))
        bar.refresh()
        adverbegin_flag = 0

        acc_maximum = 0
        begin_adv_i = 10
        tempe_uncertain_max = 1.0
        test_acc_save = []

        uncertainty_save = []
        source_uncertainty_save = []
        test_acc_save = []
        interration_total = []

        test_test__acc_save=[]

        acc_target_test_val=0
        for i in bar:
            if i % 2 == 0:
                # weights_mapping_value=(i*1.0/iterations)*0.9+0.1
                if optimizer_name == "sgd":

                    # if i == int(begin_adv_i*2):
                    # print ("begin weight uncertain")

                    if i > begin_adv_i:
                        p_value = 1.0 * i / iterations * 1.0
                        weights_mapping_value = 2.0 / (1.0 + math.exp(-10.0 * p_value)) - 1.0
                        learning_rate_value = lr / math.pow(1 + 10.0 * p_value, 0.75)
                        # learning_rate_value = lr * (1 + 0.001 * i) ** (-0.75)
                        sess.run(tf.assign(lr_var, learning_rate_value))
                        # sess.run(tf.assign(weights_mapping, weights_mapping_value))
                        sess.run(tf.assign(weights_mapping, 0.99 * weights_mapping_value))
                        # sess.run(tf.assign(weights_uncertain, 0.25*weights_mapping_value))


                else:
                    if i > begin_adv_i:
                        p_value = 1.0 * (i - begin_adv_i) / (iterations - begin_adv_i) * 1.0
                        weights_mapping_value = 2.0 / (1.0 + math.exp(-10.0 * p_value)) - 1.0
                        sess.run(tf.assign(weights_mapping, 0.99 * weights_mapping_value))

                        learning_rate_value = lr / math.pow(1 + 10.0 * p_value, 0.75)
                        # learning_rate_value = lr * (1 + 0.001 * i) ** (-0.75)
                        sess.run(tf.assign(lr_var, learning_rate_value))





            source_batch, source_batch_label, source_batch_samelabel, \
            source_batch_differ1, source_batch_label_differ1, \
            source_batch_differ2, source_batch_label_differ2 = getdata_batch(source_images, source_labels,
                                                                             batch_size, source_category_total,
                                                                             begini=-1)
            target_batch, target_batch_label, target_batch_samelabel, \
            target_batch_differ1, target_batch_label_differ1, \
            target_batch_differ2, target_batch_label_differ2 = getdata_batch(target_images, target_labels,
                                                                             batch_size, target_category_total,
                                                                             begini=-1,differ_mode=False)
            inter_number=20
            if i >0:
                if i <1500:

                    sess.run(tf.assign(weights_semantic_st, 0.05))
                    sess.run(tf.assign(weights_semantic_st_mix, 0.2))
                else:
                    sess.run(tf.assign(weights_semantic_st, 0.05))
                    sess.run(tf.assign(weights_semantic_st_mix, 0.2))
            #  step, step_classifier

            if i % 30 == 0:
                _,_,_,\
                source_super_loss_val, source_acc_val, target_acc_val, loss_uncertainty_val, loss_uncertainty_source_val, \
                mapping_loss_val, adversary_loss_val, mapping_loss_mmd_val,distance_semantic_val, \
                accuracy_t_t_val,accuracy_t_s_val, \
                distance_semantic_st_val,distance_semantic_st_mix_val,distance_semantic_tt_mix_val,\
                distance_semantic_differ_show_val               \
                    = sess.run(
                    [step,step_classifier,step_softmax,
                     source_super_loss, source_acc, target_acc, loss_uncertainty, loss_uncertainty_source,
                     mapping_loss, adversary_loss, mapping_loss_mmd,distance_semantic,
                     accuracy_t_t,accuracy_t_s,
                     distance_semantic_st,distance_semantic_st_mix,distance_semantic_tt_mix,
                     distance_semantic_differ_show],feed_dict={source_im:source_batch,source_label:source_batch_label,
                                            target_im:target_batch,target_label:target_batch_label,
                                                        source_im_samelabel:source_batch_samelabel,
                                        source_im_differ1: source_batch_differ1, source_label_differ1: source_batch_label_differ1,
                                        source_im_differ2: source_batch_differ2,
                                    source_label_differ2: source_batch_label_differ2,dropoupratio:0.5})

                # train_writer.add_summary(summary, i)
                source_acc_val_total.append(source_acc_val)
                target_acc_val_total.append(target_acc_val)
                uncertainty_val_total.append(loss_uncertainty_val)
                source_uncertainty_val_total.append(loss_uncertainty_source_val)
                acc_t_t_total.append(accuracy_t_t_val)
                acc_t_s_total.append(accuracy_t_s_val)
                test_acc_save.append(np.mean(target_acc_val_total))
                uncertainty_save.append(np.mean(uncertainty_val_total))
                interration_total.append(int(i))
                source_uncertainty_save.append(np.mean(source_uncertainty_val_total))

                distance_t_t_val_total.append(distance_semantic_tt_mix_val)
                distance_s_t_val_total.append(distance_semantic_st_val)
                distance_s_s_val_total.append(distance_semantic_st_mix_val)

                distance_t_t_val_total_save.append(np.mean(distance_t_t_val_total))
                distance_s_t_val_total_save.append(np.mean(distance_s_t_val_total))
                distance_s_s_val_total_save.append(np.mean(distance_s_s_val_total))



                if np.mean(target_acc_val_total) > acc_maximum and i > 99:
                    acc_maximum = np.mean(target_acc_val_total)

            else:
                if i>1500:   #### update gama
                    _,_,_,_,target_acc_val=sess.run([step, step_classifier, step_softmax,step_gama,target_acc],
                             feed_dict={source_im: source_batch, source_label: source_batch_label,
                                        target_im: target_batch, target_label: target_batch_label,
                                        source_im_samelabel: source_batch_samelabel,
                                        source_im_differ1: source_batch_differ1,
                                        source_label_differ1: source_batch_label_differ1,
                                        source_im_differ2: source_batch_differ2,
                                        source_label_differ2: source_batch_label_differ2,
                                        dropoupratio: 0.5})
                    target_acc_val_total.append(target_acc_val)
                    if np.mean(target_acc_val_total) > acc_maximum and i > 99:
                        acc_maximum = np.mean(target_acc_val_total)

                else:
                    _, _, _, target_acc_val = sess.run([step, step_classifier, step_softmax, target_acc],
                                                       feed_dict={source_im: source_batch,
                                                                  source_label: source_batch_label,
                                                                  target_im: target_batch,
                                                                  target_label: target_batch_label,
                                                                  source_im_samelabel: source_batch_samelabel,
                                                                  source_im_differ1: source_batch_differ1,
                                                                  source_label_differ1: source_batch_label_differ1,
                                                                  source_im_differ2: source_batch_differ2,
                                                                  source_label_differ2: source_batch_label_differ2,
                                                                  dropoupratio: 0.5})
                    target_acc_val_total.append(target_acc_val)
                    if np.mean(target_acc_val_total) > acc_maximum and i > 99:
                        acc_maximum = np.mean(target_acc_val_total)
            if i%30==0:
                target_acc_val_test= sess.run([target_acc],
                                         feed_dict={target_im: target_batch,
                                                    target_label: target_batch_label,dropoupratio:1.0})
                # train_writer.add_summary(summary, i)
                target_acc_val_total_test.append(target_acc_val_test)

                if np.mean(target_acc_val_total_test) > acc_maximum and i > 90:
                    acc_maximum = np.mean(target_acc_val_total_test)

            if i%50==0:
                #sess.run(tf.assign(dropoupratio, 1.0))
                acc_target_test_total=[]
                for testbatchi in range(int(len(target_images)/batch_size+1)):
                    target_batch, target_batch_label, target_batch_samelabel, \
                    target_batch_differ1, target_batch_label_differ1, \
                    target_batch_differ2, target_batch_label_differ2 = getdata_batch(target_images, target_labels,
                                                                                     batch_size, target_category_total,
                                                                                     begini=testbatchi*batch_size, differ_mode=False)
                    target_acc_val = sess.run([target_acc],
                        feed_dict={target_im: target_batch, target_label: target_batch_label,dropoupratio:1.0})
                    acc_target_test_total.append(target_acc_val)
                    acc_target_test_val=np.mean(np.array(acc_target_test_total[0:len(target_labels)]))
                test_test__acc_save.append(acc_target_test_val)
                if acc_target_test_val > acc_maximum  and i> 590:
                    acc_maximum = acc_target_test_val



            if i%1000==0 or i==iterations-2:
                distance_t_t_val_total_save_s = np.array(distance_t_t_val_total_save)
                distance_s_t_val_total_save_s = np.array(distance_s_t_val_total_save)
                distance_s_s_val_total_save_s = np.array(distance_s_s_val_total_save)
                test_test__acc_save_s=np.array(test_test__acc_save)

                interration_total_s = np.array(interration_total)
                test_acc_save_s = np.array(test_acc_save)

                savename_resutls = str(output) + "_semantic_mix_flag_" + str(semantic_mix_flag)  + "_adv_" + str(adverflag)






            if i % 1000 == 0:

                f = open("results_mixup2/" + source + "2" + target + ".txt", 'a')
                if i == 0:
                    f.write("-net-" + str(model) + "-training_mode-" + str(training_mode) + "-semantic_mix_flag-" + str(semantic_mix_flag) + "-" + str(
                        solver))
                    f.write("\r")
                f.write("training_num:%d, accuracy:%4f" % (i, acc_maximum))
                f.write("\r")
                f.close()
            if i == iterations - 1:
                f = open("results_mixup2/" + source + "2" + target + ".txt", 'a')
                f.write("training_num:%d, current:%4f" % (i, np.mean(target_acc_val_total_test)))
                f.write("\r")
                f.close()
            if np.mean(source_acc_val_total) > 100 and adverbegin_flag == 0 and i >= 500 and i % 500 == 0:
                # adverbegin_flag=1

                # sess.run(tf.assign(weights_uncertain, weights_uncertain_value))
                # sess.run(tf.assign(weights_uncertain_before, weights_uncertain_value_before))

                if int(int(i / 500) % 2) == 1:
                    # print("weights_uncertain_value assigned....")
                    sess.run(tf.assign(weights_uncertain, weights_uncertain_value))
                    sess.run(tf.assign(weights_uncertain_before, weights_uncertain_value_before))
                else:
                    # print("weights_uncertain_value assigned 0 0 0 0 0....")
                    sess.run(tf.assign(weights_uncertain, 0.0))
                    sess.run(tf.assign(weights_uncertain_before, 0.0))

            if i % (display) == 0:
                logging.info('{:10} Target_acc_test: {:5.4f}  Source_ac: {:5.4f}  '
                             'Mpig: {:5.4f} '
                             .format('Iteration {}:'.format(i),
                                     acc_target_test_val,
                                     np.mean(source_acc_val_total),
                                      mapping_loss_val))
            if stepsize is not None and (i + 1) % stepsize == 0:
                # lr = sess.run(lr_var.assign(lr * 0.1))
                logging.info('Changed learning rate to {:.0e}'.format(lr))
                bar.set_description('{} (lr: {:.0e})'.format(output, lr))
            if (i ) % 1500 == 0 and i>2000:
                snapshot_path = target_saver.save(
                    sess, os.path.join(output_dir, output), global_step=i + 1)
                logging.info('Saved snapshot to {}'.format(snapshot_path))

        coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    main()
