"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import types



SEG_LABELS_LIST = [  
            {"id": 0,  "name": "unspecific", "grey_value": 0},
            {"id": 1,  "name": "brachial plexus", "grey_value": 255}]

class SegmentationData(data.Dataset):

    def __init__(self, image_paths_file, transform = None, mode = 'train', 
                 num_train = 3381, num_val = 1127, num_test = 1127,
                 binary_out = True, mask_only = False, shuffle = True):
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
        self.root_dir_name = '../data/'
        
        self.num_train = num_train
        self.num_val = num_val  
        self.num_test = num_test
        self.transform = transform
        self.binary_out = binary_out
        self.mask_only = mask_only
        self.shuffle = shuffle
        
        self.image_names = self.get_image_names(mode, image_paths_file)
           

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
        
        targets = {}
        img_id = self.image_names[index].replace('.tif', '')
        
        target = Image.open(os.path.join(self.root_dir_name,
                                 'targets',
                                 img_id + '_mask.tif'))
        
        if self.mask_only and int(np.sum(np.array(target)) == 0):
            try:
                return self.__getitem__(index + 1)
            except IndexError:
                rnd = int(np.random.randint(0, len(self.image_names), 1)[0])
                return self.__getitem__(rnd)

        img = Image.open(os.path.join(self.root_dir_name,
                                      'images',
                                      img_id + '.tif'))
        
        
        if self.transform:
            inputs, target = self.transform([img, target])
        
        inputs = torch.FloatTensor(inputs)
        targets['main'] = self._labelize(target)
        
        if self.binary_out:
            binary_targets = int(np.sum(np.array(target)) > 0)
            #binary_targets = np.zeros([2])
            #binary_targets[int(np.sum(np.array(target)) > 0)] = 1
            targets['binary'] = binary_targets
            
        return inputs, targets
    
    def _labelize(self, target):
                
        target = target.numpy()
        target_labels = np.where(target != 0, 1, 0)
            
        target_labels = torch.FloatTensor(target_labels)
           
        return target_labels
    
    def eval_mode(self):               
                
        self.get_item_from_index = types.MethodType(get_item_from_index_eval, self)
    
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
            if self.shuffle:
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
        y = y['main']
        if return_image_name:
            return X.squeeze().numpy(), y.squeeze().numpy(), self.image_names[index]
        else:
            return X.squeeze().numpy(), y.squeeze().numpy()
        
    def show_image(self, index, thickness = 3, cmap = 'gray'):
        '''Plots a certain image.
        
        Parameters:
        -----------
        index = arbitrary index that will be used to return an image
        thickness = thickness of the mask
        cmap = colormap for the greyscale image
        
        
        Returns:
        --------
      
        '''
        import matplotlib.pylab as plt
        X, y, image_name = self.get_image(index, return_image_name = True)
        mask = y
        temp1 = mask - np.hstack([np.zeros([mask.shape[0], thickness]), mask[:, 0:-thickness]])
        cropped_mask = np.abs(temp1)      
        print(image_name)
        
        ax1 = plt.subplot(131)
        ax1.imshow(X, cmap = cmap, vmin = 0, vmax = 1)
        ax2 = plt.subplot(132)
        ax2.imshow(y)
        ax3 = plt.subplot(133)
        ax3.imshow(X + cropped_mask, cmap = cmap, vmin = 0, vmax = 1)        
        plt.show()
        
def get_item_from_index_eval(self, index):

    img_id = self.image_names[index].replace('.tif', '')
    
    img = Image.open(os.path.join(self.root_dir_name,
                                  'test',
                                  img_id + '.tif'))
    
    
    if self.transform:
        inputs = self.transform(img)
    
    inputs = torch.FloatTensor(inputs)
        
    return inputs        
        

 



