import matplotlib
matplotlib.use('Agg')
from random import shuffle
import numpy as np
import scipy.misc
import uuid

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Solver(object):
    default_adam_args = {"lr": 0.002,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.001}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim_args_generator = {"lr": 0.002,#generator
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.001}
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_loss_history_GNet = []
        self.train_loss_history_DNet = []
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

    def train(self, model_DNet, model_GNet, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """

        optim = self.optim(filter(lambda p: p.requires_grad, model_DNet.parameters()), **self.optim_args)
        g_optim = self.optim(filter(lambda p: p.requires_grad, model_GNet.parameters()), **self.optim_args_generator)

        self._reset_histories()
        iter_per_epoch = len(train_loader)
        
        model_NerveNet = torch.load('models/NerveNet_binary_input128_clear38_10_10_epochs_lr0.00025_cw0.25.model')
        
        for param in model_NerveNet.parameters():
            param.requires_grad = False

        cuda_param = False

        if torch.cuda.is_available():
            cuda_param = True
            model_DNet = model_DNet.cuda()
            model_GNet = model_GNet.cuda()
            model_NerveNet = model_NerveNet.cuda()
            print('Using Cuda')

        print('START TRAIN.')

        for epoch in range(num_epochs):
            # TRAINING

            for i, (inputs, targets) in enumerate(train_loader, 1):

                ########################Train D-Net on real#####################
                #print('Running the loop')
                a = np.array([1.0])
                targets_DNet = Variable(torch.from_numpy(a))
                inputs_DNet = Variable(targets.float(), requires_grad = True).unsqueeze(0)
                if cuda_param == True:
                    targets_DNet = targets_DNet.cuda()
                    inputs_DNet = inputs_DNet.cuda()
                    #print('setting inputs to cuda')
                
                outputs_DNet = model_DNet(inputs_DNet)
                optim.zero_grad()
                loss_DNet_real = torch.mean((outputs_DNet - 1) ** 2)
                ########################Train D-Net on real#####################

                ########################Train D-Net on fake#####################
                if epoch < 1000:
                    
                    a = np.array([0.0])#fake
                    target_DNet2 = torch.from_numpy(a)
                    targets_DNet2 = Variable(target_DNet2)
                    
                    #img = Variable(img)
                    #inputs = img.unqueeze(0)
                    #inputs = Variable(inputs)

                    inputs_NerveNet = Variable(inputs.float(), requires_grad = False)#.unsqueeze(0)
                    
                    if cuda_param == True:
                        targets_DNet2 = targets_DNet2.cuda()
                        #noise = noise.cuda()
                        inputs_NerveNet = inputs_NerveNet.cuda()
                        
                    check_NerveNet_input_picture = inputs_NerveNet.cpu().data.numpy()#how does yagmur process the targets?
                    check_NerveNet_input_picture = check_NerveNet_input_picture.squeeze()
                    noise = model_NerveNet(inputs_NerveNet)
                    
                    
                        
                    predicted_mask_NerveNet = noise['main']
                    #print(noise)
                    _,predicted_mask_NerveNet = torch.max(predicted_mask_NerveNet,1)
                        
                    _, binary = torch.max(noise['binary'],1)
                    if (binary == 0).cpu().data.numpy():
                        predicted_mask_NerveNet = predicted_mask_NerveNet * 0
                        
                    check_NerveNet_output_picture = predicted_mask_NerveNet.cpu().data.numpy()#how does yy process the targets?
                    check_NerveNet_output_picture = check_NerveNet_output_picture.squeeze()
                    
                    check_NerveNet_output_picture_ud = np.flipud(check_NerveNet_output_picture)
                    check_NerveNet_output_picture_ud_lr = np.fliplr(check_NerveNet_output_picture_ud)
                    
                    #ax = plt.subplot(141)
                    #ax.imshow(check_NerveNet_input_picture)

                    #ax2 = plt.subplot(142)
                    #ax2.imshow(check_NerveNet_output_picture)    
                    
                    #ax3 = plt.subplot(143)
                    #ax3.imshow(check_NerveNet_output_picture_ud)   
                    
                    #ax4 = plt.subplot(144)
                    #ax4.imshow(check_NerveNet_output_picture_ud_lr)
                    
                    #plt.show()
                    
                    flipped_predicted_mask_NerveNet = torch.from_numpy(check_NerveNet_output_picture_ud_lr.copy())
                    flipped_predicted_mask_NerveNet_var = Variable(flipped_predicted_mask_NerveNet)
                    if cuda_param == True:
                        flipped_predicted_mask_NerveNet_var = flipped_predicted_mask_NerveNet_var.cuda()
                    
                    #print(flipped_predicted_mask_NerveNet_var)
                    flipped_predicted_mask_NerveNet_var = flipped_predicted_mask_NerveNet_var.float()
                            
                    bare_output = flipped_predicted_mask_NerveNet_var
                    predicted_mask_NerveNet = predicted_mask_NerveNet.float()
                    
                    #predicted_mask_NerveNet = predicted_mask_NerveNet.view(-1)
                  
                    outputs_GNet = model_GNet(flipped_predicted_mask_NerveNet_var.view(-1))
                    #print(list(outputs_GNet.size()))
                    outputs_GNet = outputs_GNet#.view(1, 1, 128, 128)###################################maybe transpose

                    outputs_DNet = model_DNet(outputs_GNet)
                    optim.zero_grad()
                    loss_DNet_fake = torch.mean(outputs_DNet ** 2)
                    ###################Code for new implementation##################
                    
                    loss_DNet = loss_DNet_real + loss_DNet_fake
                    if (binary == 1).cpu().data.numpy():
                        loss_DNet.backward()
                        optim.step()
                        loss_DNet_CPU = loss_DNet.data.cpu().numpy()
                    else:
                        loss_DNet_CPU = 0
                else:
                    loss_DNet = loss_DNet_real
                    loss_DNet.backward()
                    optim.step()
                    loss_DNet_CPU = loss_DNet.cpu()

              
                self.train_loss_history_DNet.append(loss_DNet_CPU)
                if log_nth and i % log_nth == 0:
                    last_log_nth_losses_DNet = self.train_loss_history_DNet[-log_nth:]
                    train_loss_DNet = np.mean(last_log_nth_losses_DNet)
                    #print('[Iteration %d/%d] TRAIN loss_DNet: %.3f' % \
                    #      (i + epoch * iter_per_epoch,
                    #       iter_per_epoch * num_epochs,
                    #       train_loss_DNet))
                ########################Train D-Net on fake#####################

                ############################Train G-Net#########################
                if epoch > -1:
                    # Train G so that D recognizes G(z) as real.
                    #fake_images = model_GNet(noise_G)
                    
                    fake_images = model_GNet(predicted_mask_NerveNet.view(-1))
                    fake_images = fake_images.unsqueeze(0)#.view(1, 1, 128, 128)

                    outputs_G = model_DNet(fake_images)
                    g_optim.zero_grad()
                    loss_GNet = torch.mean((outputs_G - 1) ** 2)

                    # Backprop + optimize
                    if (binary == 1).cpu().data.numpy():
                        loss_GNet.backward()
                        g_optim.step()
                        loss_GNet_CPU = loss_GNet.cpu().data.numpy()
                    else:
                        loss_GNet_CPU = 0
                        fake_images = bare_output
                        

                    self.train_loss_history_GNet.append(loss_GNet_CPU)
                    if log_nth and i % log_nth == 0:
                        last_log_nth_losses_GNet = self.train_loss_history_GNet[-log_nth:]
                        train_loss_GNet = np.mean(last_log_nth_losses_GNet)
                        #print('[Iteration %d/%d] TRAIN loss_GNet: %.3f' % \
                        #      (i + epoch * iter_per_epoch,
                        #       iter_per_epoch * num_epochs,
                        #       train_loss_GNet))
                    ############################Train G-Net#########################

                    
                    #######################Plot current G-Net#######################
                    #if (i % 100 == 0):
                        ax = plt.subplot(131)
                        inputs_DNet = inputs_DNet.cpu()
                        picture_target = inputs_DNet.squeeze()
                        picture_target_np = picture_target.data.numpy()
                        plt.axis('off')
                        ax.imshow(picture_target_np, cmap='gray')

                        ax1 = plt.subplot(132)
                        bare_output = bare_output.cpu()
                        picture_NerveNet_CPU = bare_output.squeeze()
                        picture_NerveNet_np = picture_NerveNet_CPU.data.numpy()
                        pred = bare_output.squeeze().data.cpu().numpy()
                        plt.axis('off')
                        ax1.imshow(pred, cmap='gray')

                        ax2 = plt.subplot(133)
                        picture = fake_images.cpu()
                        picture_CPU = picture.squeeze()
                        picture_np = picture_CPU.data.numpy()
                        plt.axis('off')
                        ax2.imshow(picture_np, cmap='gray')
                        plt.show()

                        #outfile = 'image%s%s.jpg' % (str(epoch), str(i))
                        #scipy.misc.imsave(outfile, picture_np)
                    #######################Plot current G-Net#######################

        print('FINISH.')