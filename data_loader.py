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


class Lung_Train_Dataset(Dataset):

    def __init__(self):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        """

        # All images are of size 150 x 150
        self.img_size = (150, 150)

        # Only two classes will be considered here (normal and infected)
        self.classes = {0: 'normal', 1: 'infected_covid', 2: "infected_non_covid"}

        # The dataset consists only of training images
        self.groups = 'train'

        # Number of images in each part of the dataset
        self.dataset_numbers = {'train_normal': 1341, \
                                'train_infected_covid': 1334, \
                                'train_infected_non_covid': 2529}

        # Path to images for different parts of the dataset
        self.dataset_paths = {'train_normal': './dataset/train/normal/', \
                              'train_infected_covid': './dataset/train/infected/covid', \
                              'train_infected_non_covid': './dataset/train/infected/non-covid'}

    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """

        # Generate description
        msg = "This is the training dataset of the Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
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
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.

        Returns loaded image as a normalized Numpy array.
        """

        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train', 'test' or 'val'."
        assert group_val in self.groups, err_msg

        err_msg = "Error - class_val variable should be set to 'normal' or 'infected'."
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

    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.

        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """

        # Open image
        im = self.open_img(group_val, class_val, index_val)

        # Display
        plt.imshow(im)

    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """

        # Length function
        return sum(self.dataset_numbers.values())

    def test_funct(self):
        print(list(self.dataset_numbers.values()))

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
        if index < first_val:
            class_val = 'normal'
            label =0
        elif index < (second_val+first_val):
            class_val = 'infected_covid'
            index = index - first_val
            label = 1
        else:
            class_val = 'infected_non_covid'
            index = index - first_val - second_val
            label = 2

        im = self.open_img(self.groups, class_val, index)
        train_transforms = transforms.Compose([
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.5],
                                                                    [0.250])])
        im = train_transforms(im)
        return im, label


class Lung_Val_Dataset(Dataset):

    def __init__(self):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        """

        # All images are of size 150 x 150
        self.img_size = (150, 150)

        # Only two classes will be considered here (normal and infected)
        self.classes = {0: 'normal', 1: 'infected_covid', 2: "infected_non_covid"}

        # The dataset consists only of validation images
        self.groups = 'val'

        # Number of images in each part of the dataset
        self.dataset_numbers = {'val_normal': 7, \
                                'val_infected_covid': 8, \
                                'val_infected_non_covid': 7}

        # Path to images for different parts of the dataset
        self.dataset_paths = {'val_normal': './dataset/val/normal/', \
                              'val_infected_covid': './dataset/val/infected/covid', \
                              'val_infected_non_covid': './dataset/val/infected/non-covid', }

    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """

        # Generate description
        msg = "This is the validation dataset of the Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
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
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.

        Returns loaded image as a normalized Numpy array.
        """

        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train', 'test' or 'val'."
        assert group_val in self.groups, err_msg

        err_msg = "Error - class_val variable should be set to 'normal' or 'infected'."
        assert class_val in self.classes.values(), err_msg

        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg

        # Open file as before
        path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        # with open(path_to_file, 'rb') as f:
            # im = np.asarray(Image.open(f)) / 255
        # f.close()
        im = Image.open(path_to_file)
        
        return im

    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.

        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """

        # Open image
        im = self.open_img(group_val, class_val, index_val)

        # Display
        plt.imshow(im)

    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """

        # Length function
        return sum(self.dataset_numbers.values())

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
        if index < first_val:
            class_val = 'normal'
            label = 0
        elif index < (second_val+first_val):
            class_val = 'infected_covid'
            index = index - first_val
            label = 1
        else:
            class_val = 'infected_non_covid'
            index = index - first_val - second_val
            label = 2
        im = self.open_img(self.groups, class_val, index)
        test_valid_transforms = transforms.Compose([ 
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5],
                                                           [0.250])])
        im = test_valid_transforms(im)
        return im, label


class Lung_Test_Dataset(Dataset):

    def __init__(self):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        """

        # All images are of size 150 x 150
        self.img_size = (150, 150)

        # Only two classes will be considered here (normal and infected)
        self.classes = {0: 'normal', 1: 'infected_covid', 2: "infected_non_covid"}

        # The dataset consists only of test images
        self.groups = 'test'

        # Number of images in each part of the dataset
        self.dataset_numbers = {'test_normal': 234, \
                                'test_infected_covid': 139, \
                                'test_infected_non_covid': 242}

        # Path to images for different parts of the dataset
        self.dataset_paths = {'test_normal': './dataset/test/normal/', \
                              'test_infected_covid': './dataset/test/infected/covid', \
                              'test_infected_non_covid': './dataset/test/infected/non-covid'}

    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """

        # Generate description
        msg = "This is the test dataset of the Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
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
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.

        Returns loaded image as a normalized Numpy array.
        """

        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train', 'test' or 'val'."
        assert group_val in self.groups, err_msg

        err_msg = "Error - class_val variable should be set to 'normal' or 'infected'."
        assert class_val in self.classes.values(), err_msg

        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg

        # Open file as before
        path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        with open(path_to_file, 'rb') as f:
            im = np.asarray(Image.open(f)) / 255
        f.close()
        return im

    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.

        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """

        # Open image
        im = self.open_img(group_val, class_val, index_val)

        # Display
        plt.imshow(im)

    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """

        # Length function
        return sum(self.dataset_numbers.values())

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
        if index < first_val:
            class_val = 'normal'
            label = 0
        elif index < (second_val+first_val):
            class_val = 'infected_covid'
            index = index - first_val
            label = 1
        else:
            class_val = 'infected_non_covid'
            index = index - first_val - second_val
            label = 2
        im = self.open_img(self.groups, class_val, index)
        im = transforms.functional.to_tensor(np.array(im)).float()
        return im, label

# ld_test= Lung_Test_Dataset()
# print(ld_test[10])
# test_loader=DataLoader(ld_test, batch_size = 10, shuffle = True)
# for i,j in test_loader:
#     print(j)


class Lung_Dataset(Dataset):

    def __init__(self, types, data_args,classification):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        Parameters:
        - type : string that is either val, train, test
        - data_args: int that is either 0: no argumentation, 1: flipping 2: contrast and flipping
        -classifiction : string that is either binary or trinary
        """
        assert (data_args==0 or data_args==1 or data_args ==2)
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
                self.dataset_numbers = {
                                        'val_infected_covid': 8, \
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
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
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
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.

        Returns loaded image as a normalized Numpy array.
        """

        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train', 'test' or 'val'."
        assert group_val in self.groups, err_msg

        err_msg = "Error - class_val variable should be set to 'normal' or 'infected'."
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

        length =sum(self.dataset_numbers.values())*(self.data_args+1)
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
            if index < first_val:
                class_val = 'normal'
                label = 0
            elif index < (second_val+first_val):
                class_val = 'infected_covid'
                index = index - first_val
                label = 1
            else:
                class_val = 'infected_non_covid'
                index = index - first_val - second_val
                label = 2
        im = self.open_img(self.groups, class_val, index)


        train_transforms = transforms.Compose([
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.5],
                                                                    [0.250])])
        im= train_transforms(im)
        # im = transforms.functional.to_tensor(np.array(im)).float()
        if self.classification=="binary":
            if label==1 or label ==2:
                return im, 1, label
            else:
                return im ,0,0
        else:
            return im, label
    def get_normal(self,index):
        first_val = int(list(self.dataset_numbers.values())[0])
        second_val = int(list(self.dataset_numbers.values())[1])
        if index < first_val:
            class_val = 'normal'
            label = 0
        elif index < (second_val + first_val):
            class_val = 'infected_covid'
            index = index - first_val
            label = 1
        else:
            class_val = 'infected_non_covid'
            index = index - first_val - second_val
            label = 2
        im = self.open_img(self.groups, class_val, index)
        test_valid_transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5],
                                                           [0.250])])
        im = transforms.functional.to_tensor(np.array(im)).float()
        return im, label


ld_test= Lung_Dataset("val",0,"infected_only")
print(ld_test[10])
test_loader=DataLoader(ld_test, batch_size = 10, shuffle = True)
for i,label1 in test_loader:
    print(label1)