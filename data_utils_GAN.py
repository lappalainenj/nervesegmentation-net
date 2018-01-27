"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

import _pickle as pickle

SEG_LABELS_LIST = [
            {"id": 0,  "name": "unspecific", "grey_value": 0},
            {"id": 1,  "name": "brachial plexus", "grey_value": 255}]

class SegmentationData(data.Dataset):

    def __init__(self, image_paths_file, transform = None, mode = 'train'):
        """
        Returns the data object that can be loaded with pytorch
        (i.e. torch.utils.data.DataLoader( object ) ).
        Methods of this object can be used to return certain images
        as torchtensors.

        Parameters
        ----------
        image_paths_file = '../data/image_files.txt'
        mode = either 'train', 'val' or 'test'

        Notes
        -----

        """
        self.root_dir_name = os.path.dirname(image_paths_file)

        self.num_train = 758   #60%clean_no_zero_full_dataset_1300
        self.num_val = 252     #20%
        self.num_test = 252    #20%

        #self.num_train = 8   #60%clean_no_zero_full_dataset_50
        #self.num_val = 2     #20%
        #self.num_test = 2    #20%

        self.transform = transform

        self.image_names = self.get_image_names(mode, image_paths_file)
        #with open(image_paths_file) as f:
        #    self.image_names = f.read().splitlines()


    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return len(self.image_names)

    def get_item_from_index(self, index):
        to_tensor = transforms.ToTensor()
        img_id = self.image_names[index].replace('.tif', '')

        img = Image.open(os.path.join(self.root_dir_name,
                                      'images',
                                      img_id + '.tif'))

        img = Image.open(os.path.join(self.root_dir_name,
                                         'targets',
                                         img_id + '_mask.tif'))

        if self.transform:
            img = self.transform(img)
        #center_crop = transforms.CenterCrop(240)
        #img = center_crop(img)
        img = to_tensor(img)

        target = Image.open(os.path.join(self.root_dir_name,
                                         'targets',
                                         img_id + '_mask.tif'))
        # target = 1.0
        # target = np.array(target, dtype=np.int64)

        if self.transform:
            target = self.transform(target)
        #target = center_crop(target)
        target = np.array(target, dtype=np.int64)

        target_labels = np.zeros_like(target, dtype = np.int64)
        for label in SEG_LABELS_LIST:
            mask = (target == label['grey_value'])
            target_labels[mask] = label['id']

        target_labels = torch.from_numpy(target_labels.copy())
        # target_labels = torch.from_numpy(target)



        return img, target_labels

    def get_image_names(self, mode, image_paths_file):
        '''Evaluates the mode and returns the related image file names.
        The image_paths_file is shuffled (the same way for each initialized
        object of this class), to distribute images in train-, val- and testsets.

        Parameters:
        -----------
        mode = train, val or test
        image_paths_file = file that contains all filenames of the whole dataset

        Returns:
        ---------
        Image file names that are related to the mode.
        '''
        randomstate = np.random.RandomState(33)
        with open(image_paths_file) as f:
            all_image_names = np.array(f.read().splitlines())
            randomstate.shuffle(all_image_names)

        if mode == 'train':
            return all_image_names[0:self.num_train]
        elif mode == 'val':
            return all_image_names[self.num_train : self.num_train + self.num_val]
        elif mode == 'test':
            return all_image_names[self.num_train + self.num_val::]
        elif mode == 'all':
            return all_image_names

    def get_image(self, index, return_image_name = False):
        '''Returns certain image and mask as numpy array.

        Parameters:
        -----------
        index = arbitrary index that will be used to return an image
        return_image_name = if True, the name of the related image is returned

        Returns:
        --------
        X = ultrasound nerve image
        y = ultrasound nerve image segmentation
        optional: image_name

        '''

        X, y = self.get_item_from_index(index)
        if return_image_name:
            return np.squeeze(X.numpy()), y.numpy(), self.image_names[index]
        else:
            return np.squeeze(X.numpy()), y.numpy()

    def show_image(self, index, thickness = 1, cmap = 'gray'):
        '''Plots a certain image.

        Parameters:
        -----------
        index = arbitrary index that will be used to return an image
        thickness = thickness of the mask
        cmap = colormap for the greyscale image


        Returns:
        --------
        fig = matplotlib figure object
        ax =  matplotlib ax object
        '''
        import matplotlib.pylab as plt
        X, y, image_name = self.get_image(index, return_image_name = True)
        mask = y
        temp1 = mask - np.hstack([np.zeros([mask.shape[0], thickness]), mask[:, 0:-thickness]])
        cropped_mask = np.abs(temp1)
        print(image_name)
        plt.imshow(X + cropped_mask, cmap = cmap)
        plt.show()
        return plt.gcf(), plt.gca()




def get_nerve_seg_data(num_training = 3381, num_validation = 1127, num_test = 1127):
    """
    Load the dataset from disk and perform preprocessing to prepare
    it for classifiers.
    """
    # Load the raw CIFAR-10 data
    data_dir = '../data/'

    X = []
    y = []

    for file in os.listdir(data_dir):
        imarray = np.array(Image.open(data_dir + file))
        if not 'mask' in file:
            X.append(imarray)
        else:
            y.append(imarray)

    X = np.array(X)
    y = np.array(y)

    # Subsample the data
    # Our training set will be the first num_train points from the original
    # training set.
    mask = list(range(num_training))
    X_train = X[mask]
    y_train = y[mask]

    # Our validation set will be num_validation points from the original
    # training set.
    mask = list(range(num_training, num_training + num_validation))
    X_val = X[mask]
    y_val = y[mask]

    # We use a small subset of the training set as our test set.
    mask = list(range(num_training + num_validation,
                      num_training + num_validation + num_test))
    X_test = X[mask]
    y_test = y[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Package data into a dictionary
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }
