from random import shuffle
import numpy as np
#import cv2
import torch
from torch.autograd import Variable
#from skimage import transform
from torchvision import transforms
from PIL import Image


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.BCELoss(),
                 loss_weights = [1.0, 0.1, 0.05, 0.01]):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.loss_weights = loss_weights

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []
        
    def dice_coefficient(self, gt, p):
        """gt = ground_truth
            p = predicted
        """
        if np.sum(p) + np.sum(gt) == 0:
            return 1
        else:
            dice = np.sum(p[gt==1])*2.0 / (np.sum(p) + np.sum(gt))
            return dice

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """

        optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()), **self.optim_args)
        
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')

        for epoch in range(num_epochs):
            # TRAINING

            for i, (inputs, targets) in enumerate(train_loader, 1):
                #inputs contains batchsize input images
                #inputs.size() = [batchsize, num_channels, height, width]
                #targets contains batchsize target images
                #targets.size() = [batchsize, height, width]
                
                inputs = Variable(inputs)
                keys = ['in', 'up4', 'up5', 'down5']
                
                for key in keys:
                    targets[key] = Variable(targets[key])

                
                targets1 = targets['in']
                targets2 = targets['down5']
                targets3 = targets['up5']
                targets4 = targets['up4']
                
                #print('Shape: Inputs %s // Targets %s'%(inputs.size(), targets.size()))
                if model.is_cuda:
                    inputs, targets1 = inputs.cuda(), targets1.cuda()
                    targets2, targets3 = targets2.cuda(), targets3.cuda()
                    targets4 = targets4.cuda()
                    
                optim.zero_grad()
                outputs = model(inputs)
                #outputs contains num_outputs*batchsize output images
                #outputs.size() = [num_outputs, batchsize, channels, height, width]
                #print('Shape: Outputs ', outputs.size())

                outputs = list(map(lambda x: x.squeeze(), outputs))

                loss1 = self.loss_func(outputs[0].float(), targets1.float())
                loss2 = self.loss_func(outputs[1].float(), targets2.float()) 
                loss3 = self.loss_func(outputs[2].float(), targets3.float())              
                loss4 = self.loss_func(outputs[3].float(), targets4.float())
                
                total_loss = sum([w*l for w,l in 
                          zip(self.loss_weights,[loss1, loss2, loss3, loss4])])
                
                total_loss.backward()
                optim.step()

                self.train_loss_history.append(total_loss.data.cpu().numpy())
                
                if log_nth and i % log_nth == 0:
                    
                    last_log_nth_losses = self.train_loss_history[-log_nth:]
                    train_loss = np.mean(last_log_nth_losses)
                    print('[Iteration %d/%d] TRAIN loss: %.3f' % \
                        (i + epoch * iter_per_epoch,
                         iter_per_epoch * num_epochs,
                         train_loss))

            #_, preds = torch.max(outputs, 1) 
            #torch.max(outputs, 1) takes max over channels, we have just one

            gt = np.squeeze(targets1.data.cpu().numpy()) 
            p  = outputs[0].data.cpu().numpy()

            train_acc = self.dice_coefficient(gt, p)
            self.train_acc_history.append(train_acc)
            if log_nth:
                print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   train_acc,
                                                                   train_loss))
            # VALIDATION
            val_losses = []
            val_scores = []
            model.eval()
            for inputs, targets in val_loader:
                
                inputs = Variable(inputs)
                keys = ['in', 'up4', 'up5', 'down5']
                
                for key in keys:
                    targets[key] = Variable(targets[key])

                
                targets1 = targets['in']
                targets2 = targets['down5']
                targets3 = targets['up5']
                targets4 = targets['up4']

                if model.is_cuda:
                    inputs, targets1 = inputs.cuda(), targets1.cuda()
                    targets2, targets3 = targets2.cuda(), targets3.cuda()
                    targets4 = targets4.cuda()
                    
                optim.zero_grad()
                outputs = model(inputs)


                outputs = list(map(lambda x: x.squeeze(), outputs))

                loss1 = self.loss_func(outputs[0].float(), targets1.float())
                loss2 = self.loss_func(outputs[1].float(), targets2.float()) 
                loss3 = self.loss_func(outputs[2].float(), targets3.float())              
                loss4 = self.loss_func(outputs[3].float(), targets4.float())
                
                total_loss = sum([w*l for w,l in 
                          zip(self.loss_weights,[loss1, loss2, loss3, loss4])])

                val_losses.append(total_loss.data.cpu().numpy())

                
                gt = np.squeeze(targets1.data.cpu().numpy()) 
                p  = outputs[0].data.cpu().numpy()
                scores = self.dice_coefficient(gt, p)
                val_scores.append(scores)

            model.train()
            val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            if log_nth:
                print('[Epoch %d/%d] VAL   acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   val_acc,
                                                                   val_loss))
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
        
def _resize(Y, outshape):
    transform = transforms.ToPILImage()
    img = transform(Y.data.cpu().numpy())
    resized = Image.resize(img, outshape)
    transform = transforms.ToTensor()
    return transform(resized)
