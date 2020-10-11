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

@register_dataset('art')
class Art(DatasetGroup):
    """The Street View House Numbers Dataset.

    This DatasetGroup corresponds to format 2, which consists of center-cropped
    digits.

    Homepage: http://ufldl.stanford.edu/housenumbers/

    Images are 32x32 RGB images in the range [0, 1].
    """

    num_classes = 65

    def __init__(self, path=None, shuffle=True):
        DatasetGroup.__init__(self, 'art', path=path)
        self.image_shape = (256, 256, 3)
        self.label_shape = ()
        self.shuffle = shuffle
        self._load_datasets()
    def _load_datasets(self):
        dataset,label=pickle.load(open("./data/officehome/art_256.pkl","rb"))
        train_images =dataset
        train_labels =label

        validation_images, validation_labels = pickle.load(open("./data/officehome/art_256.pkl", "rb"))
        test_images = []
        test_labels = []
        totalfile = dataset.shape[0]
        for i in range(int(totalfile / 2)):
            imgi = random.randint(0, totalfile - 1)
            test_images.append(dataset[imgi])
            test_labels.append(label[imgi])
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)

        self.train = ImageDataset(train_images, train_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)

        self.testtest = ImageDataset(test_images, test_labels,
                                 image_shape=self.image_shape,
                                 label_shape=self.label_shape,
                                 shuffle=self.shuffle)

        self.validation = ImageDataset(validation_images, validation_labels,
                                       image_shape=self.image_shape,
                                       label_shape=self.label_shape,
                                       shuffle=self.shuffle)

