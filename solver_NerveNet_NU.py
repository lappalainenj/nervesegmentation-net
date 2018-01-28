import numpy as np
import torch
from torch.autograd import Variable



class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.BCELoss(),
                 binary_out = 0.1):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.binary_out = binary_out

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
        self.iter_per_epoch = len(train_loader)
        self.log_nth = log_nth

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')

        for epoch in range(num_epochs):

            for i, (inputs, targets) in enumerate(train_loader, 1):
                #inputs contains #batchsize input images
                #inputs.size() = [batchsize, num_channels, height, width]
                #targets contains batchsize target images
                #targets.size() = [batchsize, height, width]
                
                inputs = Variable(inputs)
                
                target_main = Variable(targets['main'])
                
                if self.binary_out:
                    target_binary = Variable(targets['binary'])

                if model.is_cuda:
                    inputs = inputs.cuda()
                    target_main = target_main.cuda()
                    if self.binary_out:
                        target_binary = target_binary.cuda()
                    
                optim.zero_grad()
                outputs = model(inputs)
                
                loss = self.loss_func(outputs['main'].float(), target_main.float())
                
                if self.binary_out:
                    binary_loss = self.loss_func(outputs['binary'].float(),
                                                 target_binary)
                    loss = loss + binary_loss
                
                loss.backward()
                optim.step()

                self.train_loss_history.append(loss.data.cpu().numpy())
                
                if log_nth and i % log_nth == 0:
                    
                    self.print_iteration_results(i, epoch, num_epochs, 'TRAIN')
            
            _, pred_label = torch.max(outputs['main'], 1)
            pred_label = pred_label.unsqueeze(1)
            train_acc = 1 - self.loss_func(pred_label.float(), target_main.float())
            
            self.train_acc_history.append(train_acc)
            if log_nth:
                
                self.print_epoch_results(train_acc, epoch, num_epochs, 'TRAIN')
                
            val_acc = self.validation(model, val_loader, optim)
            self.print_epoch_results(val_acc, epoch, num_epochs, 'VAL')

        print('FINISH.')

    def validation(self, model, val_loader, optim):
                
        val_losses = []
        val_scores = []
        model.eval()
        for inputs, targets in val_loader:
            
            inputs = Variable(inputs)
        
            target_main = Variable(targets['main'])
            
            if self.binary_out:
                target_binary = Variable(targets['binary'])

            if model.is_cuda:
                inputs = inputs.cuda()
                target_main = target_main.cuda()
                if self.binary_out:
                    target_binary = target_binary.cuda()
                
            optim.zero_grad()
            outputs = model(inputs)

            loss = self.loss_func(outputs['main'].float(), target_main.float())
        
            if self.binary_out:
                binary_loss = self.loss_func(outputs['binary'].float(),
                                             target_binary)
                loss = loss + binary_loss

            val_losses.append(loss.data.cpu().numpy())
            
            _, pred_label = torch.max(outputs['main'], 1)
            pred_label = pred_label.unsqueeze(1)
            scores = 1 - self.loss_func(pred_label.float(), target_main.float())
            val_scores.append(scores)

        model.train()
        val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
        self.val_acc_history.append(val_acc)
        self.val_loss_history.append(val_loss)
        return val_acc
      
    def print_iteration_results(self, i, epoch, num_epochs, mode):
        '''Print iteration results'''
        
        iter_per_epoch = self.iter_per_epoch
        last_log_nth_losses = self.train_loss_history[-self.log_nth:]
        train_loss = np.mean(last_log_nth_losses)
        phrase = '|Iteration %d/%d| %s loss: %.3f'
        display = (i + epoch * iter_per_epoch, iter_per_epoch * num_epochs,
                   mode, train_loss)
        print(phrase%display)
        
    def print_epoch_results(self, train_acc, epoch, num_epochs, mode):
        '''Print epoch results'''

        phrase = '|Epoch %d/%d| %s acc: %.3f'
        display = (epoch + 1, num_epochs, mode, train_acc)
        print(phrase%display)
        
                
    