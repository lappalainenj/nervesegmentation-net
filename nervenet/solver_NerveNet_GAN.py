import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pylab as plt


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={}, discr_optim_args={},
                 loss_func=torch.nn.MSELoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        optim_args_merged.update(discr_optim_args)
        self.discr_optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []
        self.discr_train_loss_history = []
        self.discr_train_acc_history = []
        self.discr_val_acc_history = []
        self.discr_val_loss_history = []


    def train(self, model, discr_model, train_loader, val_loader, 
                  active_layers, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        
        for param in list(model.parameters()): 
            param.requires_grad = False

        for module in active_layers:
            for param in list(module.parameters()): 
                param.requires_grad = True
                
        optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()), **self.optim_args)
        discr_optim = self.optim(filter(lambda p: p.requires_grad, discr_model.parameters()), **self.discr_optim_args)
        
        self._reset_histories()
        self.iter_per_epoch = len(train_loader)
        self.log_nth = log_nth

        if torch.cuda.is_available():
            model.cuda()
            discr_model.cuda()

        print('START TRAIN.')

        for epoch in range(num_epochs):

            for i, (inputs, targets) in enumerate(train_loader, 1):
                #inputs contains #batchsize input images
                #inputs.size() = [batchsize, num_channels, height, width]
                #targets contains batchsize target images
                #targets.size() = [batchsize, height, width]
                
                inputs = Variable(inputs)
                
                #Batchsize ones
                batchsize = inputs.size(0)
                real = torch.FloatTensor(batchsize)
                real.fill_(1)
                real = Variable(real)
                
                #Batchsize zeros
                fake = real - 1
                
                target_main = torch.cat((1-targets['main'], targets['main']), 1)
                target_main = Variable(target_main)
                #target_binary = Variable(targets['binary'])

                if model.is_cuda:
                    inputs = inputs.cuda()
                    target_main = target_main.cuda()
                    
                #Discr_Model
                discr_optim.zero_grad()
                
                #on real mask
                discr_outputs = discr_model(target_main)
                discr_loss_real = self.loss_func(discr_outputs, real)
                discr_loss_real.backward()
                self.d_on_real = discr_outputs.mean()
                
                #on predicted mask
                outputs = model(inputs)
                pred = outputs['main']
                
#                _, binary = torch.max(outputs['binary'], 1)
#                for i, bc in enumerate(binary.data.cpu().numpy()):
#                    if bc == 0:
#                        eps = 0.0000001
#                        pred[i, 0] = pred[i, 0]/(pred[i, 0]+eps) #0 class probability to one
#                        pred[i, 1] = 0 * pred[i, 1]               #1 class probability to one
                
                discr_outputs = discr_model(pred)
                discr_loss_fake = self.loss_func(discr_outputs, fake)
                discr_loss_fake.backward()
                discr_loss = discr_loss_real + discr_loss_fake
                discr_optim.step()
                self.d_on_fake = discr_outputs.mean()
                
                self.discr_train_loss_history.append(discr_loss.data.cpu().numpy())
                
                #NerveNET G Model
                optim.zero_grad()
                nng_output = model(inputs)              
                discr_on_nng_out = discr_model(nng_output['main'])
                
                nng_loss = self.loss_func(discr_on_nng_out, real)
                nng_loss.backward()
                optim.step()
                
                self.d_on_nng = discr_on_nng_out.mean()

                self.train_loss_history.append(nng_loss.data.cpu().numpy())
                
                if log_nth and i % log_nth == 0:
                    
                    self.print_iteration_results(i, epoch, num_epochs, 'TRAIN')
                    
                    ax0 = plt.subplot(131)
                    plt.axis('off')
                    ax0.imshow(inputs[0].squeeze().data.cpu().numpy(), cmap='gray')
                    
                    ax = plt.subplot(132)
                    plt.axis('off')
                    ax.imshow(target_main[0, 1].squeeze().data.cpu().numpy(), cmap='gray')

                    ax1 = plt.subplot(133)
                    _, pred = torch.max(nng_output['main'], 1)
                    plt.axis('off')
                    ax1.imshow(pred[0].squeeze().data.cpu().numpy(), cmap='gray')
                    
                    plt.show()
                    
        print('FINISH.')
        
        

 
        
      
    def print_iteration_results(self, i, epoch, num_epochs, mode):
        '''Print iteration results'''
        
        iter_per_epoch = self.iter_per_epoch
        last_log_nth_losses = self.train_loss_history[-self.log_nth:]
        discr_last_log_nth_losses = self.discr_train_loss_history[-self.log_nth:]
        train_loss = np.mean(last_log_nth_losses)
        discr_loss = np.mean(discr_last_log_nth_losses)
        phrase = '|Iteration %d/%d| %s loss NNG/DISCR: %.3f / %.3f'
        add    = ' || DISCRout REAL / FAKE / NNG: %.3f / %.3f / %.3f'
        phrase = phrase + add
        display = (i + epoch * iter_per_epoch, iter_per_epoch * num_epochs,
                   mode, train_loss, discr_loss, self.d_on_real, self.d_on_fake,
                   self.d_on_nng)
        print(phrase%display)
        
    def print_epoch_results(self, train_acc, epoch, num_epochs, mode):
        '''Print epoch results'''

        phrase = '|Epoch %d/%d| %s acc: %.3f'
        display = (epoch + 1, num_epochs, mode, train_acc)
        print(phrase%display)
        
                
    