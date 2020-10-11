import gzip
import os
from urllib.parse import urljoin

import numpy as np

from adda.data import DatasetGroup
from adda.data import ImageDataset
from adda.data import util
from adda.data.dataset import register_dataset

import pickle
from PIL import Image
import random

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

        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]]

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
                # print("np.array(train_images[i]).shape: ", img.shape)
                img = np.uint8(img)
                img = Image.fromarray(img)
                img = np.array(img.convert("RGB").resize((28, 28), Image.BICUBIC))
                if random.random() > 0.1:
                    img = (img + colors[random.randint(0, 5)] )/2 # random.randint(0, 5)
                train_total.append(img)
                train_0_total.append(img)
                label_0_total.append(0)

                img = np.uint8(img)
                savedirr = "usps_colored/"
                savename = savedirr + ("%d.jpg" % i)
                Image.fromarray(img).save(savename)
            else:
                label_total.append(1)
                img = np.reshape(np.array(train_images[i]), [16, 16])
                # print ("np.array(train_images[i]).shape: ",img.shape)
                img = np.uint8(img)
                img = Image.fromarray(img)
                img = np.array(img.convert("RGB").resize((28, 28), Image.BICUBIC))
                if random.random() > 0.1:
                    img = (img +colors[random.randint(0, 5)])/2
                train_total.append(img)
                train_1_total.append(img)
                label_1_total.append(1)

                img = np.uint8(img)
                savedirr="usps_colored/"
                savename=savedirr+("%d.jpg"%i)
                Image.fromarray(img).save(savename)

        train_total = np.array(train_total)
        label_total = np.array(label_total)
        print("-----------train_total.shape:", train_total.shape)
        train_1_total = np.array(train_1_total)
        train_0_total = np.array(train_0_total)
        label_1_total = np.array(label_1_total)
        label_0_total = np.array(label_0_total)

        print("dataset.shape:", train_total.shape, "dataset.shape:", label_total.shape)
        pickle.dump((train_total, label_total, train_1_total, label_1_total, train_0_total, label_0_total),
                    open("usps_colored.pkl", "wb"))
        print("end..")

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


@register_dataset('usps1800')
class USPS1800(USPS):

    name = 'usps1800'

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
        inds = rand.permutation(len(train_images))[:7500]
        inds.sort()
        train_images = train_images[inds]
        train_labels = train_labels[inds]
        test_images, test_labels = self._read_datafile(abspaths['test'])
        self.train = ImageDataset(train_images, train_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)
        self.test = ImageDataset(test_images, test_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)

        colors = np.array([ [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]])

        train_total = []
        label_total = []
        train_1_total = []
        train_0_total = []
        label_1_total = []
        label_0_total = []
        label_total_true = []
        noise_p = 0.0 * 100
        color_p = 0.9 * 100



        for i in range(len(train_labels)):
            if int(train_labels[i]) < 5:
                label_total_true.append(0)
                if random.randint(a=0,b=100) < noise_p:
                    label_total.append([1, 0])
                    Y = 1
                else:
                    label_total.append([0, 0])
                    Y = 0
                img = np.reshape(np.array(train_images[i]), [16, 16])
                # print("np.array(train_images[i]).shape: ", img.shape)
                img = np.uint8(img*255)
                img = Image.fromarray(img)
                img = np.array(img.convert("RGB").resize((16, 16), Image.BICUBIC))
                if random.randint(a=0,b=100) < color_p:
                    colori=abs(1-Y)
                else:
                    colori = abs(Y)

                colori = random.randint(a=0, b=len(colors)-1)
                img = (img + colors[colori])/2
                train_total.append(img)
                train_0_total.append(img)
                if random.randint(a=0,b=100) < noise_p:
                    label_0_total.append([1, 0])
                else:
                    label_0_total.append([0, 0])

                img = np.uint8(img)
                savedirr = "usps_colored/"
                savename = savedirr + ("%d.jpg" % i)
                Image.fromarray(img).save(savename)
            else:
                label_total_true.append(1)
                if random.randint(a=0,b=100) < noise_p:
                    label_total.append([0, 1])
                    Y = 0
                else:
                    label_total.append([1, 1])
                    Y = 1
                img = np.reshape(np.array(train_images[i]), [16, 16])
                # print ("np.array(train_images[i]).shape: ",img.shape)
                img = np.uint8(img*255)
                img = Image.fromarray(img)
                img = np.array(img.convert("RGB").resize((16, 16), Image.BICUBIC))
                train_total.append(img)
                if random.randint(a=0,b=100) < color_p:
                    colori = abs(1 - Y)
                else:
                    colori = abs(Y)

                colori=random.randint(a=0,b=len(colors)-1)
                img = (img + colors[colori]) / 2

                train_1_total.append(img)
                if random.randint(a=0,b=100) < noise_p:
                    label_1_total.append([0,1])
                else:
                    label_1_total.append([1,1])

                img = np.uint8(img)
                savedirr = "usps_colored/"
                savename = savedirr + ("%d.jpg" % i)
                Image.fromarray(img).save(savename)

        train_total = np.array(train_total)
        label_total = np.array(label_total)
        label_total_true = np.array(label_total_true)
        print("-----------train_total.shape:", train_total.shape)
        train_1_total = np.array(train_1_total)
        train_0_total = np.array(train_0_total)
        label_1_total = np.array(label_1_total)
        label_0_total = np.array(label_0_total)


        print("dataset.shape:", train_total.shape, "dataset.shape:", label_total.shape)
        pickle.dump((train_total, label_total,label_total_true, train_1_total, label_1_total, train_0_total, label_0_total),
                    open("usps_colored.pkl", "wb"))
        print("end..")
