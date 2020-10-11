import gzip
import operator
import os
import struct
from functools import reduce
from urllib.parse import urljoin

import numpy as np

from adda.data import DatasetGroup
from adda.data import ImageDataset
from adda.data import util
from adda.data.dataset import register_dataset


import pickle
from PIL import Image
import random
@register_dataset('mnist')
class MNIST(DatasetGroup):
    """The MNIST database of handwritten digits.

    Homepage: http://yann.lecun.com/exdb/mnist/

    Images are 28x28 grayscale images in the range [0, 1].
    """

    base_url = 'http://yann.lecun.com/exdb/mnist/'

    data_files = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images': 't10k-images-idx3-ubyte.gz',
            'test_labels': 't10k-labels-idx1-ubyte.gz',
            }

    num_classes = 10

    def __init__(self, path=None, shuffle=True):
        DatasetGroup.__init__(self, 'mnist', path)
        self.image_shape = (28, 28, 1)
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
        train_images = self._read_images(abspaths['train_images'])
        train_labels = self._read_labels(abspaths['train_labels'])
        test_images = self._read_images(abspaths['test_images'])
        test_labels = self._read_labels(abspaths['test_labels'])
        print("--train_images", train_images.shape)
        print("--train_labels" ,train_labels)

        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]]
        print ("--------------------------mnist----------------")
        print("-------------------------------len(train_images: ", np.array(train_images).shape)
        print("--------------------------------len(train_labels: ", np.array(test_labels).shape)
        train_total = []
        label_total = []
        train_1_total = []
        train_0_total = []
        label_1_total = []
        label_0_total = []

        for i in range(len(train_labels)):
            if int(train_labels[i]) <= 4:
                label_total.append(0)
                img = np.reshape(np.array(train_images[i]), [28, 28])
                # print("np.array(train_images[i]).shape: ", img.shape)
                img = np.uint8(img)
                img = Image.fromarray(img)
                img = np.array(img.convert("RGB"))
                if random.random() > 0.1:
                    img = img + colors[0]
                train_total.append(img)
                train_0_total.append(img)
                label_0_total.append(0)
            else:
                label_total.append(1)
                img = np.reshape(np.array(train_images[i]), [28, 28])
                # print ("np.array(train_images[i]).shape: ",img.shape)
                img = np.uint8(img)
                img = Image.fromarray(img)
                img = np.array(img.convert("RGB"))
                if random.random() > 0.1:
                    img = img + colors[1]
                train_total.append(img)
                train_1_total.append(img)
                label_1_total.append(1)

        train_total = np.array(train_total)
        label_total = np.array(label_total)
        print("-----------train_total.shape:", train_total.shape)
        train_1_total = np.array(train_1_total)
        train_0_total = np.array(train_0_total)
        label_1_total = np.array(label_1_total)
        label_0_total = np.array(label_0_total)

        print("dataset.shape:", train_total.shape, "dataset.shape:", label_total.shape)
        pickle.dump((train_total, label_total, train_1_total, label_1_total, train_0_total, label_0_total),
                    open("mnist_colored.pkl", "wb"))
        print("end..")

        self.train = ImageDataset(np.array(train_images), np.array(train_labels),
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)
        self.test = ImageDataset(test_images, test_labels,
                                 image_shape=self.image_shape,
                                 label_shape=self.label_shape,
                                 shuffle=self.shuffle)

    def _read_datafile(self, path, expected_dims):
        """Helper function to read a file in IDX format."""
        base_magic_num = 2048
        with gzip.GzipFile(path) as f:
            magic_num = struct.unpack('>I', f.read(4))[0]
            expected_magic_num = base_magic_num + expected_dims
            if magic_num != expected_magic_num:
                raise ValueError('Incorrect MNIST magic number (expected '
                                 '{}, got {})'
                                 .format(expected_magic_num, magic_num))
            dims = struct.unpack('>' + 'I' * expected_dims,
                                 f.read(4 * expected_dims))
            buf = f.read(reduce(operator.mul, dims))
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(*dims)
            return data

    def _read_images(self, path):
        """Read an MNIST image file."""
        return (self._read_datafile(path, 3)
                .astype(np.float32)
                .reshape(-1, 28, 28, 1)
                / 255)

    def _read_labels(self, path):
        """Read an MNIST label file."""
        return self._read_datafile(path, 1)


@register_dataset('mnist2000')
class MNIST2000(MNIST):

    name = 'mnist2000'

    def __init__(self, seed=None, path=None, shuffle=True):
        if seed is None:
            self.seed = hash(self.name) & 0xffffffff
        else:
            self.seed = seed
        MNIST.__init__(self, path=path, shuffle=shuffle)

    def _load_datasets(self):
        abspaths = {name: self.get_path(path)
                    for name, path in self.data_files.items()}
        rand = np.random.RandomState(self.seed)
        train_images = self._read_images(abspaths['train_images'])
        train_labels = self._read_labels(abspaths['train_labels'])
        inds = rand.permutation(len(train_images))[:50000] #45000
        inds.sort()
        train_images = train_images[inds]
        train_labels = train_labels[inds]
        test_images = self._read_images(abspaths['test_images'])
        test_labels = self._read_labels(abspaths['test_labels'])
        self.train = ImageDataset(train_images, train_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)
        self.test = ImageDataset(test_images, test_labels,
                                 image_shape=self.image_shape,
                                 label_shape=self.label_shape,
                                 shuffle=self.shuffle)
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]])
        print("--------------------------mnist----------------")
        print("-------------------------------np.array(train_images).shape: ", np.array(train_images).shape)
        print("--------------------------------np.array(test_labels).shape: ", np.array(test_labels).shape)
        train_total = []
        label_total = []
        train_1_total = []
        train_0_total = []
        label_1_total = []
        label_0_total = []
        savedirr="./data/mnist/"
        label_total_true = []
        noise_p=0.0*100
        color_p=0.1*100

        for i in range(len(train_labels)):
            if int(train_labels[i]) < 5:
                label_total_true.append(0)
                if random.randint(a=0,b=100)<noise_p:
                    label_total.append([1,0])
                    Y=1
                else:
                    label_total.append([0,0])
                    Y=0
                img_ori = np.reshape(np.array(train_images[i]), [28, 28])
                # print("np.array(train_images[i]).shape: ", img.shape)
                img = np.uint8(img_ori*255)
                img = Image.fromarray(img)
                img = np.array(img.convert("RGB").resize((16, 16), Image.BICUBIC))
                img = np.array(img)
                if random.randint(a=0,b=100) < color_p:
                    colori=abs(1-Y)
                else:
                    colori = abs(Y)
                img = (img + colors[colori])/2
                train_total.append(img)
                train_0_total.append(img)

                if random.randint(a=0,b=100) < noise_p:
                    label_0_total.append([1,0])
                else:
                    label_0_total.append([0,0])


                #img = np.reshape(img, [28, 28, 3])
                img = np.uint8(img)
                savedirr_img = "mnist_colored/"
                savename = savedirr_img + ("%d.jpg" % i)
                #Image.fromarray(img).save(savename)
            else:
                label_total_true.append(1)
                if random.randint(a=0,b=100) < noise_p:
                    label_total.append([0,1])
                    Y=0
                else:
                    label_total.append([1,1])
                    Y=1

                img_ori = np.reshape(np.array(train_images[i]), [28, 28])
                # print ("np.array(train_images[i]).shape: ",img.shape)


                img = np.uint8(img_ori*255)
                img = Image.fromarray(img)
                img = np.array(img.convert("RGB").resize((16, 16), Image.BICUBIC))
                img = np.array(img)

                if random.randint(a=0,b=100) < color_p:
                    colori = abs(1 - Y)
                else:
                    colori = abs(Y)
                img = (img + colors[colori]) / 2

                train_total.append(img)
                train_1_total.append(img)

                if rand.randint(low=0, high=100) < noise_p:
                    label_1_total.append([0,1])
                else:
                    label_1_total.append([1,1])

                #img = np.reshape(img, [28, 28,3])
                img = np.uint8(img)
                savedirr_img =  "mnist_colored/"
                savename = savedirr_img + ("%d.jpg" % i)
                #Image.fromarray(img).save(savename)

        train_total = np.array(train_total)
        label_total = np.array(label_total)
        label_total_true=np.array(label_total_true)
        print("-----------train_total.shape:", train_total.shape)
        train_1_total = np.array(train_1_total)
        train_0_total = np.array(train_0_total)
        label_1_total = np.array(label_1_total)
        label_0_total = np.array(label_0_total)

        print("dataset.shape:", train_total.shape, "dataset.shape:", label_total.shape)
        pickle.dump((train_total, label_total, label_total_true,train_1_total, label_1_total, train_0_total, label_0_total),
                    open("mnist_colored.pkl", "wb"))
        print(" ----------mnist end..")
