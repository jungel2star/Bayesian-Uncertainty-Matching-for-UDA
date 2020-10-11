import os
from urllib.parse import urljoin

import numpy as np
from scipy.io import loadmat

from adda.data import DatasetGroup
from adda.data import ImageDataset
from adda.data import util
from adda.data.dataset import register_dataset
import pickle
import random
from PIL import Image
@register_dataset('amazon')
class Amazon(DatasetGroup):
    """The Street View House Numbers Dataset.

    This DatasetGroup corresponds to format 2, which consists of center-cropped
    digits.

    Homepage: http://ufldl.stanford.edu/housenumbers/

    Images are 32x32 RGB images in the range [0, 1].
    """

    num_classes = 31

    def __init__(self, path=None, shuffle=True):
        DatasetGroup.__init__(self, 'amazon', path=path)
        self.image_shape = (256, 256, 3)
        self.label_shape = ()
        self.shuffle = shuffle
        self._load_datasets()
    def _load_datasets(self):
        dataset,label=pickle.load(open("./data/office31/amazon_256.pkl","rb"))
        train_images =dataset
        train_labels =label

        test_images = dataset[0:10]
        test_labels = label[0:10]
        self.train = ImageDataset(train_images, train_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)

        self.test = ImageDataset(test_images, test_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)

        dataset_validation, label_validation = dataset,label

        self.validation = ImageDataset(dataset_validation, label_validation,
                                       image_shape=self.image_shape,
                                       label_shape=self.label_shape,
                                       shuffle=self.shuffle)
       ##########################
        colors = np.array([[0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]])
        train_total_color = []
        label_total_color = []
        noise_p = 0.0 * 100
        color_p = 0.1 * 100
        for i in range(len(train_labels)):
            if int(train_labels[i]) < 16:
                if random.randint(low=0, high=100) < noise_p:
                    label_total_color.append(1)
                    Y = 1
                else:
                    label_total_color.append(0)
                    Y = 0
                img = np.array(train_images[i])
                # print("np.array(train_images[i]).shape: ", img.shape)

                if random.randint(low=0, high=100) < color_p:
                    colori = abs(1 - Y)
                else:
                    colori = abs(Y)
                colori = random.randint(a=0, b=len(colors) - 1)
                img = (img + colors[colori]) / 2
                train_total_color.append(img)
                img = np.uint8(img)
                savedirr = "amazon_colored/"
                savename = savedirr + ("%d.jpg" % i)
                Image.fromarray(img).save(savename)
            else:
                if random.randint(low=0, high=100) < noise_p:
                    label_total_color.append(0)
                    Y = 0
                else:
                    label_total_color.append(1)
                    Y = 1
                img = np.array(train_images[i])
                # print ("np.array(train_images[i]).shape: ",img.shape)
                train_total_color.append(img)
                if random.randint(low=0, high=100) < color_p:
                    colori = abs(1 - Y)
                else:
                    colori = abs(Y)

                colori = random.randint(a=0, b=len(colors) - 1)
                img = (img + colors[colori]) / 2
                img = np.uint8(img)
                savedirr = "amazon_colored/"
                savename = savedirr + ("%d.jpg" % i)
                Image.fromarray(img).save(savename)
        train_total_color = np.array(train_total_color)
        label_total_color = np.array(label_total_color)
        print ("---colored amazon")
        #dataset_validation_10, label_validation_10 = pickle.load(open("./data/office31/amazon_10.pkl", "rb"))
        self.validation_10 = ImageDataset(train_total_color, label_total_color,
                                          image_shape=self.image_shape,
                                          label_shape=self.label_shape,
                                          shuffle=self.shuffle)