"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


SEG_LABELS_LIST = [  
            {"id": 0,  "name": "unspecific", "grey_value": 0},
            {"id": 1,  "name": "brachial plexus", "grey_value": 255}]

class SegmentationData(data.Dataset):

    def __init__(self, image_paths_file, transform = None, mode = 'train', 
                 num_train = 3381, num_val = 1127, num_test = 1127):
        """
        Returns the data object that can be loaded with pytorch
        (i.e. torch.utils.data.DataLoader( object ) ).
        Methods of this object can be used to return certain images 
        as torchtensors.
        
        Parameters
        ----------
        image_paths_file = '../data/image_files.txt'
        mode = either 'train', 'val', 'test', 'all' or 'debug'

        Notes
        -----
        
        """
        self.root_dir_name = os.path.dirname(image_paths_file)
        
        self.num_train = num_train
        self.num_val = num_val  
        self.num_test = num_test
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
        
        img_id = self.image_names[index].replace('.tif', '')

        img = Image.open(os.path.join(self.root_dir_name,
                                      'images',
                                      img_id + '.tif'))
        target = Image.open(os.path.join(self.root_dir_name,
                                 'targets',
                                 img_id + '_mask.tif'))
        
        targets_binary = int(np.sum(np.array(target)) > 0)
        
        if self.transform:
            inputs, targets = self._transform(img, target)
        
        targets = self._labelize(targets)
        targets['down5'] = targets_binary
            
        return inputs, targets
    
    def _transform(self, img, target):
        
        targets = {}
        keys = ['in', 'up4', 'up5']
        
        resize_transforms = self.transform[0:3]
        random_transforms = self.transform[3]
        
        inputs = resize_transforms[0](img)
        
        for i, transformation in enumerate(resize_transforms):
            
            targets[keys[i]] = transformation(target)
        
        rt = random_transforms([inputs] + list(targets.values()))
        
        inputs = rt[0]
        
        for i, key in enumerate(keys, 1):
            
            targets[key] = rt[i]
             
        return inputs, targets
    
    def _labelize(self, sample):
        
        targets = list(sample.values())
        #to_tensor = transforms.ToTensor()
        
        for i, key in enumerate(sample.keys()):
            
            target = targets[i].numpy()
            target_labels = np.where(target != 0, 1, 0)
            
            sample[key] = torch.FloatTensor(target_labels)
           # print(sample[key].size())
        
        return sample
            
    
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
        elif mode == 'debug':
            return all_image_names[0:50]
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



