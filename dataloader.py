import os
from os.path import isdir, exists, abspath, join

import random
from random import randint
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageEnhance
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import torchvision

class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=.1):
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')

        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]

    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            endId = len(self.data_files)

        while current < endId:


            # todo: load images and labels
            # hint: scale images between 0 and 1
            # hint: if training takes too long or memory overflow, reduce image size!

            #GET IMAGE
            single_image_name = self.data_files[current]
            img_as_img = Image.open(single_image_name)

            # Crop the image
            img_as_np = img_as_img.resize((572, 572))

            #Data Augumentation
            data_image=self.applyDataAugmentation(img_as_np)
            data_image=np.divide(np.asarray(data_image,dtype=np.float32),255.)
            #img_as_np = np.asarray(data_image)
            #data_image = (img_as_np - np.min(img_as_np)) / (np.max(img_as_np) - np.min(img_as_np)) * 255.


            #Mirroring the image
            img_crop = (3 * 572 - 572) // 2
            img_flip=np.flipud(data_image)
            img_rot=np.rot90(data_image,k=1,axes=(0,1))
            concat1=np.concatenate((np.flipud(img_rot),img_flip,np.flipud(img_rot)),axis=1)
            concat2 = np.concatenate((img_rot, data_image, img_rot), axis=1)
            concat3 = np.concatenate((np.flipud(img_rot), img_flip, np.flipud(img_rot)), axis=1)
            concat_image=np.concatenate((concat1,concat2,concat3),axis=0)
            dim1,dim2=concat_image.shape
            data_image=concat_image[img_crop:dim1-572, 572:dim2-572]
            #data_image=data_image.resize((572,572))

            #Normalization of the image
            data_image = (data_image - np.min(data_image)) / (np.max(data_image) - np.min(data_image))*255

            #GET MASK-LABELS
            single_mask_name = self.label_files[current]
            current += 1
            msk_as_img = Image.open(single_mask_name)


            #Cropping the mask
            label_image=msk_as_img.resize((388,388))
            label_image = self.applyDataAugmentation(label_image)
            label_image = np.asarray(label_image,dtype=np.float32)


            #Mask should be in 0 and 1 Hence no normalization for Mask
            yield (data_image, label_image)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))

    def applyDataAugmentation(self, img):

        # Add the Brightness for the image, add the Horizontal flip, Vertical Flip
        self.img_as_np = torchvision.transforms.functional.adjust_hue(img, 0.5)
        self.img_as_np = torchvision.transforms.functional.hflip(self.img_as_np)
        self.img_as_np = torchvision.transforms.functional.vflip(self.img_as_np)
        return self.img_as_np




