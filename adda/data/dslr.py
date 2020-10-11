import os
from urllib.parse import urljoin

import numpy as np
from scipy.io import loadmat

from adda.data import DatasetGroup
from adda.data import ImageDataset
from adda.data import util
from adda.data.dataset import register_dataset
import pickle


@register_dataset('dslr')
class Dslr(DatasetGroup):
    """The Street View House Numbers Dataset.

    This DatasetGroup corresponds to format 2, which consists of center-cropped
    digits.

    Homepage: http://ufldl.stanford.edu/housenumbers/

    Images are 32x32 RGB images in the range [0, 1].
    """

    num_classes = 31

    def __init__(self, path=None, shuffle=True):
        DatasetGroup.__init__(self, 'dslr', path=path)
        self.image_shape = (256, 256, 3)
        self.label_shape = ()
        self.shuffle = shuffle
        self._load_datasets()
    def _load_datasets(self):
        dataset,label=pickle.load(open("./data/office31/dslr_256.pkl","rb"))
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

        dataset_validation, label_validation = dataset,label  #pickle.load(open("./data/office31/dslr25.pkl", "rb"))

        self.validation = ImageDataset(dataset_validation, label_validation,
                                 image_shape=self.image_shape,
                                 label_shape=self.label_shape,
                                 shuffle=self.shuffle)

        dataset_validation_10, label_validation_10 = pickle.load(open("./data/office31/dslr_10.pkl", "rb"))

        self.validation_10 = ImageDataset(dataset_validation_10, label_validation_10,
                                       image_shape=self.image_shape,
                                       label_shape=self.label_shape,
                                       shuffle=self.shuffle)