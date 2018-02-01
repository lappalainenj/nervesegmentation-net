#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 20:15:31 2018

@author: yy
"""

import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torchvision import transforms, datasets

from classifiers.NerveNet import NerveNET, BinaryOut
from data_utils_NerveNet import SegmentationData
from solver_NerveNet import Solver
import transform_utils_NerveNet as tu
from dice_loss import DiceLoss

def dice_coefficient(ground_truth, predicted):
    gt = ground_truth
    p = predicted
    if np.sum(p) + np.sum(gt) == 0:
        return 1
    else:
        dice = np.sum(p[gt==1])*2.0 / (np.sum(p) + np.sum(gt))
        return dice
    
def plot_gt_pred(num_example_imgs):
    plt.figure(figsize=(15, 5 * num_example_imgs))
    for i, (img, targets) in enumerate(test_data[20:40]):
        
        target = targets['main']
        inputs = img.unsqueeze(0)
        inputs = Variable(inputs)
    
        model.cpu()
        outputs = model.forward(inputs)
        pred = outputs['main']
        _, pred = torch.max(pred, 1)
        if model.binary_out == True:
            binary_probs, binary = torch.max(outputs['binary'], 1)
            #If binary predicts no nerve we just multiply the predicted image with zeros.
            if binary.data.numpy() == 0:
                pred = pred * 0
       
            
        pred = pred.squeeze().data.cpu().numpy()
    
        img=np.squeeze(img)
        target = target.squeeze().numpy()
            
        # img
        plt.subplot(num_example_imgs, 3, i * 3 + 1)
        plt.axis('off')
        plt.imshow(img, cmap='gray')
        if i == 0:
            plt.title("Input image")
        
        # target
        plt.subplot(num_example_imgs, 3, i * 3 + 2)
        plt.axis('off')
        plt.imshow(target, cmap='gray')
        if i == 0:
            plt.title("Target image")
    
        # pred
        plt.subplot(num_example_imgs, 3, i * 3 + 3)
        plt.axis('off')
        plt.imshow(pred, cmap='gray')
        if i == 0:
            plt.title("Prediction image")    
    plt.show()
    
def test_acc():
    test_scores = []
    model.eval()
    for inputs, targets in test_loader:
        inputs, targets = Variable(inputs), Variable(targets['main'])
        if model.is_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        outputs = model.forward(inputs)
        pred = outputs['main']
        _, pred = torch.max(pred, 1)
        if model.binary_out == True:
            binary_probs, binary = torch.max(outputs['binary'], 1)
            if binary.data.numpy() == 0:
                pred = pred * 0
    
        pred = pred.squeeze().data.cpu().numpy()
        test_scores.append(dice_coefficient(np.squeeze(targets.data.numpy()), pred))
    print(np.mean(test_scores))
    return np.mean(test_scores)
    
def load_ttv():
    #not doing validation but needed as an input
    val_transforms = tu.Compose([tu.Resize(input_dim[1::]),
                                   tu.ToTensor()])
    val_data = SegmentationData(img_files,  transform = val_transforms, mode = 'val', **nums)
    val_loader = torch.utils.data.DataLoader(val_data,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=4)
    
    train_transforms = tu.Compose([tu.Resize(input_dim[1::]),
                                   tu.RandomHorizontalFlip(),
                                   tu.RandomVerticalFlip(),                               
                                   tu.ToTensor()])
    train_data = SegmentationData(img_files, transform = train_transforms, mode = 'train', **nums, 
                                  binary_out = binary_out, mask_only = mask_only)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size = batch_size,
                                               shuffle=True,
                                               num_workers=4)
    
    test_data = SegmentationData(img_files,  transform = val_transforms, mode = 'test', **nums,
                                 mask_only = mask_only)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=4)
    return test_data, train_data, val_loader, train_loader, test_loader
    


img_files = '../data/clean_38_percent.txt' 
num_lines = sum(1 for line in open(img_files, 'r'))
num_lines = sum(1 for line in open(img_files, 'r'))
nums = {'num_train' : int(0.8*num_lines)+1,
        'num_val'   : int(0.0*num_lines),
        'num_test' : int(0.2*num_lines)}

print(nums, num_lines)
print(np.sum(list(nums.values())) == num_lines)

#-----------------------------------------------------------------------------------

input_dim = (1, 128, 128)
binary_out =  True
mask_only = False 
batch_size = 4
num_epochs = 5
learning_rate = 0.00005
class_weights = 0.25
num_classes = 2

#LOAD THE DATA
test_data, train_data, val_loader, train_loader, test_loader = load_ttv()

#CREATE OR LOAD THE MODEL 

#model = torch.load("../models/NerveNet_binary_input128_clear38_10_10_epochs_lr"
#                   +str(learning_rate)+"_cw"+str(class_weights)+ ".model")

model = torch.load("../goodmodels/NerveNet_upsamp_binary_input128_clear38_10_10_epochs_acc0.871782739093.model")
model.binary_out = binary_out

#TRAINING
    
#transpose convolution training
#model = NerveNET(input_dim, num_classes = num_classes, weight_scale = 0.01, dropout = 0.05, 
#                binary_out = binary_out, upsample_unit='ConvTranspose2d')


#freeze the binary output learning
for param in list(model.parameters()): #'''True/False for all layers'''
    param.requires_grad = True

for param in list(model.binary.parameters()): #'''True/False for binary output layers'''
    param.requires_grad = True

solver = Solver(optim_args={"lr": learning_rate, #0.0025, #1.e-3, #1.e-2
                            "betas": (0.9, 0.999),
                            "eps": 1e-8,
                            "weight_decay": 0.001},
                loss_func = DiceLoss(num_classes = num_classes, classweights = [0.25, 0.75]), 
                binary_out = 0.5,
                num_classes = num_classes)


model.train()
outputs = solver.train(model, train_loader, val_loader, log_nth=100, num_epochs=num_epochs)

model.cpu()
test = test_acc()
model.save("../goodmodels/NerveNet_upsamp_nobinary_input128_clear38_10_10_5_epochs_acc"+str(test)+".model")
































