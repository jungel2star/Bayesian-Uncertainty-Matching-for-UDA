import gzip
import os
from urllib.parse import urljoin
from PIL import Image
import numpy as np
import random
from adda.data import DatasetGroup
from adda.data import ImageDataset
from adda.data import util
from adda.data.dataset import register_dataset
import pickle

@register_dataset('usps')
class USPS(DatasetGroup):
    """USPS handwritten digits.

    Homepage: http://statweb.stanford.edu/~hastie/ElemStatLearn/data.html

    Images are 16x16 grayscale images in the range [0, 1].
    """

    base_url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'

    data_files = {
        'train': 'zip.train.gz',
        'test': 'zip.test.gz'
        }

    num_classes = 10

    def __init__(self, path=None, shuffle=True, download=True):
        DatasetGroup.__init__(self, 'usps', path=path, download=download)
        self.image_shape = (16, 16, 1)
        self.label_shape = ()
        self.shuffle = shuffle
        self._load_datasets()

    def download(self):
        data_dir = self.get_path()
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for filename in self.data_files.values():
            path = self.get_path(filename)
            if not os.path.exists(path):
                url = urljoin(self.base_url, filename)
                util.maybe_download(url, path)

    def _load_datasets(self):
        abspaths = {name: self.get_path(path)
                    for name, path in self.data_files.items()}
        train_images, train_labels = self._read_datafile(abspaths['train'])
        test_images, test_labels = self._read_datafile(abspaths['test'])
        self.train = ImageDataset(train_images, train_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)
        self.test = ImageDataset(test_images, test_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)

    def _read_datafile(self, path):
        """Read the proprietary USPS digits data file."""
        labels, images = [], []
        with gzip.GzipFile(path) as f:
            for line in f:
                vals = line.strip().split()
                labels.append(float(vals[0]))
                images.append([float(val) for val in vals[1:]])
        labels = np.array(labels, dtype=np.int32)
        labels[labels == 10] = 0  # fix weird 0 labels
        images = np.array(images, dtype=np.float32).reshape(-1, 16, 16, 1)
        images = (images + 1) / 2
        return images, labels


@register_dataset('usps_colored')
class USPS_COLORED(USPS):

    name = 'usps_colored'

    def __init__(self, seed=None, path=None, shuffle=True):
        if seed is None:
            self.seed = hash(self.name) & 0xffffffff
        else:
            self.seed = seed
        USPS.__init__(self, path=path, shuffle=shuffle)

    def _load_datasets(self):
        abspaths = {name: self.get_path(path)
                    for name, path in self.data_files.items()}
        rand = np.random.RandomState(self.seed)
        train_images, train_labels = self._read_datafile(abspaths['train'])
        inds = rand.permutation(len(train_images))[:1800]
        inds.sort()
        train_images = train_images[inds]
        train_labels = train_labels[inds]
        test_images, test_labels = self._read_datafile(abspaths['test'])

        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]]
        print("-------------------------------len(train_images: ", np.array(train_images).shape)
        print("--------------------------------len(train_labels: ", np.array(test_labels).shape)
        train_total = []
        label_total = []
        train_1_total = []
        train_0_total = []
        label_1_total = []
        label_0_total = []

        for i in range(len(train_labels)):
            if int(train_labels[i]) < 4:
                label_total.append(0)
                img = np.reshape(np.array(train_images[i]), [16, 16])
                #print("np.array(train_images[i]).shape: ", img.shape)
                img = np.uint8(img)
                img = Image.fromarray(img)
                img = np.array(img.convert("RGB").resize((28,28),Image.BICUBIC))
                if random.random() > 0.1:
                    img = img + colors[random.randint(0, 5)] #random.randint(0, 5)
                train_total.append(img)
                train_0_total.append(img)
                label_0_total.append(0)
            else:
                label_total.append(1)
                img = np.reshape(np.array(train_images[i]), [16, 16])
                #print ("np.array(train_images[i]).shape: ",img.shape)
                img=np.uint8(img)
                img=Image.fromarray(img)
                img = np.array(img.convert("RGB").resize((28,28),Image.BICUBIC))
                if random.random() > 0.1:
                    img = img + colors[random.randint(0, 5)]
                train_total.append(img)
                train_1_total.append(img)
                label_1_total.append(1)

        train_total = np.array(train_total)
        label_total=np.array(label_total)
        print ("-----------train_total.shape:",train_total.shape)
        train_1_total = np.array(train_1_total)
        train_0_total = np.array(train_0_total)
        label_1_total = np.array(label_1_total)
        label_0_total = np.array(label_0_total)

        print("dataset.shape:", train_total.shape, "dataset.shape:", label_total.shape)
        pickle.dump((train_total, label_total,train_1_total,label_1_total,train_0_total,label_0_total),
                    open( "usps_colored.pkl", "wb"))
        print("end..")

        self.train = ImageDataset(train_images, train_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)
        self.test = ImageDataset(test_images, test_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)
