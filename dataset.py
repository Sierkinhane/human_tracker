import torch
from torch.utils import data
import os
from math import ceil, floor
import cv2
import numpy as np

class Dataset(data.Dataset):

    def __init__(self, datasets, to_shape=(96, 128)):
        self.datasets = datasets
        self.transform = True
        self.num_classes = len(datasets)
        self.height = to_shape[1]
        self.width = to_shape[0]

    def normalize(self, img):

        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
        #resize
        if np.random.rand() < 0.2:
            r_h = np.random.randint(80, 128, 1)
            r_w = np.random.randint(0, 3, 1)
            x = np.random.randint(5, 10, 1)
            if r_w == 0:
                img = img[0:int(r_h), 0:54+int(x), :]
            elif r_w == 1:
                img = img[0:int(r_h), 10-int(x):64, :]
            else:
                img = img[0:int(r_h), :, :]

        height, width = img.shape[:2]
        img = cv2.resize(img, (0,0), fx=self.width/width, fy=self.height/height, interpolation=cv2.INTER_CUBIC)

        return img.astype(np.float32) / 255.

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        image = cv2.imread(self.datasets[index][0])
        if self.transform:
            # image = self.transform(image)
            image = self.normalize(image)
        image = torch.from_numpy(image.transpose(2, 0, 1))
        return (image, self.datasets[index][1], self.datasets[index][2])


class Data(object):

    def __init__(self, data_dir=None, infer_shape=(96, 128)):
        super(Data, self).__init__()
        self.data_dir = data_dir
        self.infer_shape = infer_shape

    def getDataloader(self, batch_size=128, num_workers=3, shuffle=True):
        
        t, v, n = self.createDataset()
        train_set = Dataset(t, to_shape=self.infer_shape)
        valid_set = Dataset(v, to_shape=self.infer_shape)

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle)

        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle)

        return (train_loader, valid_loader), n


    def createDataset(self, train_val_split=0.9):
        """
        thanks to
        """

        images_root = os.path.join(self.data_dir, '')
        names = os.listdir(images_root)

        if len(names) == 0:
            raise RuntimeError('Empty dataset')

        training_set = []
        validation_set = []
        for klass, name in enumerate(names):
            def add_class(image):
                image_path = os.path.join(images_root, name, image)
                return (image_path, klass, name)

            images_of_person = os.listdir(os.path.join(images_root, name))

            total = len(images_of_person)

            training_set += map(
                    add_class,
                    images_of_person[:ceil(total * train_val_split)])
            validation_set += map(
                    add_class,
                    images_of_person[floor(total * train_val_split):])

        return training_set, validation_set, len(names)
