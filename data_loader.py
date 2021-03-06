# Matplotlib
import matplotlib.pyplot as plt
# Numpy
import numpy as np
# Pillow
from PIL import Image
# Torch
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from datetime import datetime

class Lung_Dataset(Dataset):

    def __init__(self, types, data_args,classification):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        Parameters:
        - type : string that is either val, train, test
        - data_args: int that is either 0: no argumentation, 1: flipping
        - classifiction : string that is either binary or trinary
        """
        assert (data_args==0 or data_args==1)
        self.data_args=data_args
        
        # All images are of size 150 x 150
        self.img_size = (150, 150)

        # Only two classes will be considered here (normal and infected)
        assert (classification=="binary" or classification=="trinary" or classification=="infected_only")
        self.classification =classification
        if classification=="binary":
            self.classes = {0: 'normal', 1: 'infected', 2:"infected_covid",3: "infected_non_covid"}
        elif classification=="trinary":
            self.classes = {0: 'normal', 1: 'infected_covid', 2: "infected_non_covid"}
        elif classification=="infected_only":
            self.classes = { 0: 'infected_covid', 1: "infected_non_covid"}
       
        # The dataset consists only of training images
        assert (types=="train" or types=="val" or types=="test")
        self.groups = types
        
        # Number of images in each part of the dataset
        if types=="train":
            if classification=="infected_only":
                self.dataset_numbers = {'train_infected_covid': 1334, \
                                        'train_infected_non_covid': 2529}
            else:
                self.dataset_numbers = {'train_normal': 1341, \
                                        'train_infected_covid': 1334, \
                                        'train_infected_non_covid': 2529}
        elif types=="test":
            if classification == "infected_only":
                self.dataset_numbers = {'test_infected_covid': 139, \
                                        'test_infected_non_covid': 242}
            else:
                self.dataset_numbers = {'test_normal': 234, \
                                        'test_infected_covid': 139, \
                                        'test_infected_non_covid': 242}
        elif types=="val":
            if classification== "infected_only":
                self.dataset_numbers = {'val_infected_covid': 8, \
                                        'val_infected_non_covid': 7}
            else:
                self.dataset_numbers = {'val_normal': 7, \
                                        'val_infected_covid': 8, \
                                        'val_infected_non_covid': 7}


        # Path to images for different parts of the dataset
        if classification== "infected_only":
            self.dataset_paths = {
                              '{}_infected_covid'.format(types): './dataset/{}/infected/covid'.format(types), \
                              '{}_infected_non_covid'.format(types): './dataset/{}/infected/non-covid'.format(types)}
        else :
            self.dataset_paths = {'{}_normal'.format(types): './dataset/{}/normal/'.format(types), \
                              '{}_infected_covid'.format(types): './dataset/{}/infected/covid'.format(types), \
                              '{}_infected_non_covid'.format(types): './dataset/{}/infected/non-covid'.format(types)}

    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """

        # Generate description
        msg = "This is the {} dataset of the Lung Dataset".format(self.groups)
        msg += " used for the Small Project in the 50.039 Deep Learning class"
        msg += " in 21 March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)

    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.

        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal', 'infected covid' or 'infected non-covid.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.

        Returns loaded image as a normalized Numpy array.
        """
        
        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train', 'test' or 'val'."
        assert group_val in self.groups, err_msg

        err_msg = "Error - class_val variable should be set to 'normal', 'infected' or 'infected non-covid."
        assert class_val in self.classes.values(), err_msg

        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        err_msg += "\n Your index value is {}".format(index_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg

        # Open file as before
        path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        # with open(path_to_file, 'rb') as f:
            # im = np.asarray(Image.open(f)) / 255
        # f.close()
        im = Image.open(path_to_file)
        return im
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """

        # Length function
        add=0
        if self.classification=="infected_only":
            for key in self.dataset_numbers:
                if "infected_covid"in key:
                    add=self.dataset_numbers[key]
        else:
            for key in self.dataset_numbers:
                if "infected_non_covid" in key:
                    add+=self.dataset_numbers[key]
        length =sum(self.dataset_numbers.values())+(self.data_args*(add))
        return int(length)

    def __getitem__(self, index):
        """
        Getitem special method.

        Expects an integer value index, between 0 and len(self) - 1.

        Returns the image and its label as a one hot vector, both
        in torch tensor format in dataset.
        """

        # Get item special method
        first_val = int(list(self.dataset_numbers.values())[0])
        second_val = int(list(self.dataset_numbers.values())[1])
        if self.classification=="infected_only":
            if index < first_val:
                class_val = 'infected_covid'
                label = 0
            elif index < (second_val+first_val):
                class_val = 'infected_non_covid'
                index = index - first_val
                label = 1
            else:
                return self.get_flip(index -sum(self.dataset_numbers.values()))
        else:
            if index < first_val:
                class_val = 'normal'
                label = 0
            elif index < (second_val+first_val):
                class_val = 'infected_covid'
                index = index - first_val
                label = 1
            elif index < sum(self.dataset_numbers.values()):
                class_val = 'infected_non_covid'
                index = index - first_val - second_val
                label = 2
            else:
                return self.get_flip(index -sum(self.dataset_numbers.values()))

        im = self.open_img(self.groups, class_val, index)



        train_transforms = transforms.Compose([

                                               transforms.ToTensor(),
                                               transforms.Normalize([0.5],
                                                                   [0.250])])
        im = train_transforms(im)
        # im = transforms.functional.to_tensor(np.array(im)).float()
        if self.classification=="binary":
            if label==1 or label ==2:
                return im, 1, label
            else:
                return im ,0,0
        else:
            return im, label
    def get_flip(self,index):
        first_val = int(list(self.dataset_numbers.values())[0])
        second_val = int(list(self.dataset_numbers.values())[1])
        if self.classification=="infected_only":
            if index < first_val:
                class_val = 'infected_covid'
                label = 0
        else:
            if index < first_val:
                class_val = 'normal'
                label = 0
            elif index < (second_val + first_val):
                class_val = 'infected_covid'
                index = index - first_val
                label = 1

        im = self.open_img(self.groups, class_val, index)
        transforms_image = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5],
                                                           [0.250])])
        im =transforms_image(im)
        im= torch.flip(im,[0, 1])
        if self.classification == "binary":
            if label == 1 or label == 2:
                return im, 1, label
            else:
                return im, 0, 0
        else:
            return im, label


# ld_test= Lung_Dataset("val",0,"infected_only")
# print(ld_test[10])
# test_loader=DataLoader(ld_test, batch_size = 10, shuffle = True)
# for i,label1 in test_loader:
#     print(label1)
