import os
from os.path import isdir, exists, abspath, join

import random
from random import randint

import numpy as np
from PIL import Image

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
            current += 1

            # todo: load images and labels
            # hint: scale images between 0 and 1
            # hint: if training takes too long or memory overflow, reduce image size!

            #GET IMAGE
            single_image_name = self.data_files[current]
            img_as_img = Image.open(single_image_name)
            img_as_np = np.asarray(img_as_img)
            input_size=572
            output_size=388

            # Augmentation
            # flip {0: vertical, 1: horizontal, 2: both, 3: none}
            flip_num = randint(0, 3)
            #img_as_np = dataAugumentation(img_as_np, flip_num)

            if flip_num == 0:
                # vertical
                img_as_np = np.flip(img_as_np, flip_num)
            elif flip_num == 1:
                # horizontal
                imaimg_as_npge = np.flip(img_as_np, flip_num)
            elif flip_num == 2:
                # horizontally and vertically flip
                img_as_np = np.flip(img_as_np, 0)
                img_as_np = np.flip(img_as_np, 1)
            else:
                img_as_np = img_as_np
                # no effect
            #return img_as_np

            # Crop the image
            img_height, img_width = img_as_np.shape[0], img_as_np.shape[1]
            pad_size = int((input_size-output_size) / 2)
            img_as_np = np.pad(img_as_np, pad_size, mode="symmetric")
            y_loc, x_loc = randint(0, img_height - output_size), randint(0, img_width - output_size)

            #img_as_np = cropping(img_as_np, crop_size=input_size, dim1=y_loc, dim2=x_loc)
            img_as_np = img_as_np[y_loc:y_loc + input_size, x_loc:x_loc + input_size]

            #Normalization of the image
            data_image = (img_as_np - np.min(img_as_np)) * (1 - 0) / (np.max(img_as_np) - np.min(img_as_np)) + 0
            #data_image = normalization2(img_as_np, max=1, min=0)

            #GET MASK-LABELS
            single_mask_name = self.label_files[current]
            msk_as_img = Image.open(single_mask_name)
            msk_as_np = np.asarray(msk_as_img)

            #Cropping the mask
            #label_image = cropping(msk_as_np, crop_size=self.out_size, dim1=y_loc, dim2=x_loc)
            label_image = msk_as_np[y_loc:y_loc + output_size, x_loc:x_loc + output_size]

            #Mask should be in 0 and 1 Hence no normalization for Mask
            yield (data_image, label_image)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))

    def cropping(image, crop_size, dim1, dim2):
        """crop the image and pad it to in_size
        Args :
            images : numpy array of images
            crop_size(int) : size of cropped image
            dim1(int) : vertical location of crop
            dim2(int) : horizontal location of crop
        Return :
            cropped_img: numpy array of cropped image
        """
        cropped_img = image[dim1:dim1 + crop_size, dim2:dim2 + crop_size]
        return cropped_img


