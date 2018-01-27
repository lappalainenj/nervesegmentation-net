from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_loss_history_GNet = []
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
        g_optim = self.optim(filter(lambda p: p.requires_grad, model_GNet.parameters()), **self.optim_args)

        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model_DNet.cuda()
            model_GNet.cuda()

        print('START TRAIN.')

        for epoch in range(num_epochs):
            # TRAINING

            for i, (inputs, targets) in enumerate(train_loader, 1):
                #inputs contains batchsize input images
                #inputs.size() = [batchsize, num_channels, height, width]
                #targets contains batchsize target images
                #targets.size() = [batchsize, height, width]

                ########################Train D-Net on real#####################
                #print(inputs.type(), targets.type())
                #if (i%2 == 1):
                print('Running the loop')

                a = np.array([1.0])
                targets_DNet = Variable(torch.from_numpy(a))

                inputs_DNet = Variable(targets.float(), requires_grad = True)
                inputs_DNet = inputs_DNet.view(-1)


                # x = -torch.ones(128, 128)
                # inputs_DNet = Variable(x, requires_grad = True)
                # inputs_DNet = Variable(noise_image)

                outputs_DNet = model_DNet(inputs_DNet)
                # outp_DNet = outputs_DNet.squeeze()
                #print(outputs_DNet.size())

                optim.zero_grad()

                #loss_DNet_real = self.loss_func(outputs_DNet.float(), targets_DNet.float())
                loss_DNet_real = torch.mean((outputs_DNet - 1) ** 2)
                #print('real', loss_DNet_real)

                loss_DNet = loss_DNet_real
                loss_DNet.backward()
                optim.step()

                if (i%100 == 0):
                    print('loss after real image:  ', loss_DNet.data.numpy())

                # d_optim.zero_grad()
                # loss_DNet_real.backward()
                # d_optim.step()
                #loss_DNet_real = torch.mean((outp_DNet - 1) ** 2)

                #print('loss_real:  ', loss_DNet_real)

                ########################Train D-Net on real#####################

                # ########################Train D-Net on fake#####################
                # #if (i%2 == 0):
                # a = np.array([0.0])
                # target_DNet2 = torch.from_numpy(a)
                # targets_DNet2 = Variable(target_DNet2)
                #
                # noise = Variable(torch.randn(128*128, 1))
                # noise = noise.view(-1)
                #
                # # noise = Variable(targets.float() / torch.randn(128*128, 1))
                # #noise = Variable(targets.float())
                # #noise = noise.view(-1)
                #
                # outputs_GNet = model_GNet(noise)
                # outp_DNet2 = model_DNet(outputs_GNet)
                #
                # # noise_image = Variable(torch.randn(128, 128), requires_grad = True)
                # # inputs_DNet2 = noise_image.view(-1)
                # # outp_DNet2 = model_DNet(inputs_DNet2)
                #
                # #print(outp_DNet)
                #
                # #loss_DNet_fake = self.loss_func(outp_DNet.float(), targets_DNet.float())
                # #loss_DNet_fake = self.loss_func(outp_DNet2.float(), targets_DNet2.float())
                # loss_DNet_fake = torch.mean(outp_DNet2 ** 2)
                # #print('fake', loss_DNet_fake)
                # #loss_DNet_fake = torch.mean(outp_DNet ** 2)
                #
                # loss_DNet = loss_DNet_real + loss_DNet_fake
                #
                #
                # loss_DNet.backward()
                # optim.step()
                #
                # if (i%100 == 0):
                #     print('loss after fake image:  ', loss_DNet.data.numpy())
                #
                #
                # # self.train_loss_history.append(loss_DNet.data.cpu().numpy())
                # # if log_nth and i % log_nth == 0:
                # #     last_log_nth_losses_DNet = self.train_loss_history[-log_nth:]
                # #     train_loss_DNet = np.mean(last_log_nth_losses_DNet)
                # #     print('[Iteration %d/%d] TRAIN loss_DNet: %.3f' % \
                # #         (i + epoch * iter_per_epoch,
                # #          iter_per_epoch * num_epochs,
                # #          train_loss_DNet))
                #
                # ########################Train D-Net on fake#####################

                # if (i > 100):
                #
                #     ############################Train G-Net#########################
                #     noise_G = Variable(torch.randn(128*128, 1))
                #     noise_G = noise_G.view(-1)
                #
                #     # Train G so that D recognizes G(z) as real.
                #     fake_images = model_GNet(noise_G)
                #
                #     outputs_G = model_DNet(fake_images)
                #     g_optim.zero_grad()
                #     loss_GNet = torch.mean((outputs_G - 1) ** 2)
                #
                #     # Backprop + optimize
                #     loss_GNet.backward()
                #     g_optim.step()
                #
                #     if (i%100 == 0):
                #         print('GNET loss:   ', loss_GNet)
                #
                #     self.train_loss_history_GNet.append(loss_GNet.data.cpu().numpy())
                #     if log_nth and i % log_nth == 0:
                #         last_log_nth_losses_GNet = self.train_loss_history_GNet[-log_nth:]
                #         train_loss_GNet = np.mean(last_log_nth_losses_GNet)
                #         print('[Iteration %d/%d] TRAIN loss_GNet: %.3f' % \
                #             (i + epoch * iter_per_epoch,
                #              iter_per_epoch * num_epochs,
                #              train_loss_GNet))
                #         ############################Train G-Net#########################
                #
                #     #######################Plot current G-Net#######################
                #     if (i % (log_nth*1) == 0):
                #         ax = plt.subplot(121)
                #         picture_tar = inputs_DNet.view(128, 128)
                #         picture_target = picture_tar.data.numpy()
                #         ax.imshow(picture_target)
                #
                #         ax1 = plt.subplot(122)
                #         picture = fake_images.view(128, 128)
                #         picture_np = picture.data.numpy()
                #         ax1.imshow(picture_np)
                #         plt.show()
                #     #######################Plot current G-Net#######################


            # gt = np.squeeze(targets.data.cpu().numpy())
            # p  = outp.data.cpu().numpy()
            #
            # train_acc = self.dice_coefficient(gt, p)
            # self.train_acc_history.append(train_acc)
            # if log_nth:
            #     print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (epoch + 1,
            #                                                        num_epochs,
            #                                                        train_acc,
            #                                                        train_loss))
            # # VALIDATION
            # val_losses = []
            # val_scores = []
            # model_DNet.eval()
            # for inputs, targets in val_loader:
            #     inputs, targets = Variable(inputs), Variable(targets)
            #     if model_DNet.is_cuda:
            #         inputs, targets = inputs.cuda(), targets.cuda()
            #
            #     outputs = model_DNet.forward(inputs)
            #     outp = outputs.squeeze()
            #     #loss = self.loss_func(outp.float(), targets.float())
            #     loss = self.loss_func(outp.float(), 1)
            #     val_losses.append(loss.data.cpu().numpy())
            #
            #     #_, preds = torch.max(outputs, 1)
            #
            #     gt = np.squeeze(targets.data.cpu().numpy())
            #     p  = outp.data.cpu().numpy()
            #     scores = self.dice_coefficient(gt, p)
            #     val_scores.append(scores)
            #
            # model_DNet.train()
            # val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
            # self.val_acc_history.append(val_acc)
            # self.val_loss_history.append(val_loss)
            # if log_nth:
            #     print('[Epoch %d/%d] VAL   acc/loss: %.3f/%.3f' % (epoch + 1,
            #                                                        num_epochs,
            #                                                        val_acc,
            #                                                        val_loss))
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
