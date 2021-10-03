from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


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
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=1):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        # initialize variables
        n_mini_batch = 0
        num_iter = train_loader.__len__()*num_epochs
        val_size = val_loader.batch_size*val_loader.__len__()

        # start training
        for epoch in range(num_epochs):
            # initialize loss to sum over all mini-bachtes and whole epoch
            current_loss = 0.0
            epoch_loss = 0.0
            # count mini_batches in epoch
            n_epoch = 0
            for curr_n, data in enumerate(train_loader):
                n_mini_batch += 1
                n_epoch = +1
                # zero gradients
                optim.zero_grad()

                x, y = data
                # run forward path
                yHat = model(x)
                # calculate loss
                loss =  self.loss_func(yHat, y)
                current_loss += loss.item()
                epoch_loss += loss.item()
                # run backward path
                loss.backward()
                # optimize
                optim.step()

                # logging
                if n_mini_batch%log_nth == 0:
                    train_loss = current_loss/log_nth
                    print('[Iteration %i/%i] TRAIN loss: %.4f'%(n_mini_batch, num_iter,
                                                                train_loss))
                    current_loss = 0.0

                if curr_n == train_loader.__len__()-1:
                    # using detach() as var requiring grad can not be converted to numpy
                    yHat = np.argmax(yHat.detach().numpy(), axis = 1)
                    # calculate measures
                    train_acc = float(sum(yHat == y))/y.size(0)
                    train_loss = epoch_loss/curr_n
                    print('[EPOCH %i/%i] TRAIN acc/loss: %.3f/%.3f'%(epoch+1, num_epochs,
                                                                     train_acc, train_loss))
                    self.train_acc_history.append(train_acc)
                    self.train_loss_history.append(train_loss)

            # validation set
            val_loss = 0.0
            val_acc = 0.0
            with (torch.no_grad()):
                for data in val_loader:
                    x, y = data
                    # run forward path
                    yHat = model(x)
                    # calculate loss
                    loss = self.loss_func(yHat, y)
                    val_loss += loss.item()

                    yHat = np.argmax(yHat, axis = 1)
                    val_acc += float(sum(yHat == y))

            print('[EPOCH %d/%d] VAL acc/loss: %.3f/%.3f'%(epoch+1, num_epochs,
                                                           val_acc/val_size,
                                                           val_loss/val_size))
            self.val_acc_history.append(val_acc/val_size)
            self.val_loss_history.append(val_loss/val_size)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')

    def train_segmentation(self, model, train_loader, val_loader, num_epochs=10, log_nth=1):
            """
            Train a given model with the provided data.

            Inputs:
            - model: model object initialized from a torch.nn.Module
            - train_loader: train data in torch.utils.data.DataLoader
            - val_loader: val data in torch.utils.data.DataLoader
            - num_epochs: total number of training epochs
            - log_nth: log training accuracy and loss every nth iteration
            """
            optim = self.optim(model.parameters(), **self.optim_args)
            self._reset_histories()
            iter_per_epoch = len(train_loader)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)

            print('START TRAIN.')
            ########################################################################
            # TODO:                                                                #
            # Write your own personal training method for our solver. In each      #
            # epoch iter_per_epoch shuffled training batches are processed. The    #
            # loss for each batch is stored in self.train_loss_history. Every      #
            # log_nth iteration the loss is logged. After one epoch the training   #
            # accuracy of the last mini batch is logged and stored in              #
            # self.train_acc_history. We validate at the end of each epoch, log    #
            # the result and store the accuracy of the entire validation set in    #
            # self.val_acc_history.                                                #
            #                                                                      #
            # Your logging could like something like:                              #
            #   ...                                                                #
            #   [Iteration 700/4800] TRAIN loss: 1.452                             #
            #   [Iteration 800/4800] TRAIN loss: 1.409                             #
            #   [Iteration 900/4800] TRAIN loss: 1.374                             #
            #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
            #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
            #   ...                                                                #
            ########################################################################
            # initialize variables
            n_mini_batch = 0
            num_iter = train_loader.__len__()*num_epochs
            val_size = val_loader.batch_size*val_loader.__len__()

            # start training
            for epoch in range(num_epochs):
                # initialize loss to sum over all mini-bachtes and whole epoch
                current_loss = 0.0
                epoch_loss = 0.0
                # count mini_batches in epoch
                n_epoch = 0
                for curr_n, data in enumerate(train_loader):
                    n_mini_batch += 1
                    n_epoch = +1
                    # zero gradients
                    optim.zero_grad()

                    x, y = data
                    # run forward path
                    yHat = model(x)
                    # calculate loss
                    loss =  self.loss_func(yHat, y)
                    current_loss += loss.item()
                    epoch_loss += loss.item()
                    # run backward path
                    loss.backward()
                    # optimize
                    optim.step()

                    yHat = yHat.argmax(dim = 1)
                    # logging
                    if n_mini_batch%log_nth == 0:
                        train_loss = current_loss/log_nth
                        acc, iu = accuracy_measures(y, yHat, model.num_classes)
                        print('[Iteration %i/%i] TRAIN acc/meanIU/loss: %.4f/%.4f/%.4f'%(n_mini_batch,
                                                                               num_iter,
                                                                    acc, iu, train_loss))
                        current_loss = 0.0

                    if curr_n == train_loader.__len__()-1:
                        # calculate measures
                        # NOTE: training error in variables still refered to as accuracy
                        #       in order to match trainers for image classification
                        acc, iu = accuracy_measures(y, yHat, model.num_classes)
                        train_loss = epoch_loss/curr_n

                        print('[EPOCH %i/%i] TRAIN label acc/meanIU/loss: %.3f/%.3f/%.3f'%(epoch+1, num_epochs,
                                                                         acc, iu, train_loss))
                        self.train_acc_history.append(acc)
                        self.train_loss_history.append(train_loss)

                # validation set
                val_loss = 0.0
                val_acc = 0.0
                with (torch.no_grad()):
                    for data in val_loader:
                        x, y = data
                        # run forward path
                        yHat = model(x)
                        # calculate loss
                        loss = self.loss_func(yHat, y)
                        val_loss += loss.item()

                        # calculate accuracy
                        yHat = yHat.argmax(dim = 1)
                        acc, iu = accuracy_measures(y, yHat, model.num_classes)

                print('[EPOCH %d/%d] VAL label acc/meanIU/loss: %.3f/%.3f/%.3f'%(epoch+1, num_epochs,
                                                               acc, iu, val_loss/val_size))
                self.val_acc_history.append(acc)
                self.val_loss_history.append(val_loss/val_size)

            ########################################################################
            #                             END OF YOUR CODE                         #
            ########################################################################
            print('FINISH.')

def accuracy_measures(true_labels, pred_labels, num_classes = 23):
    """
    """
    # convert tensors to numpy
    true_labels = true_labels.numpy()
    pred_labels = pred_labels.numpy()
    # calculate label accuracy
    mask = (true_labels > 0)
    label_accuracy = np.mean((true_labels == pred_labels)[mask])
    ## calculate confusion matrix
    #confusion_matrix = np.zeros((num_classes, num_classes))
    #for tl, pl, in zip(true_labels, pred_labels):
    #    confusion_matrix += calc_cm(tl.flatten(), pl.flatten(), num_classes)

    ## calculate accuracy
    #acc = np.diag(confusion_matrix).sum()/confusion_matrix.sum()

    ## calculate IU and class accuracy
    ## use np.errstate to cope with not all classes present in all images
    #with np.errstate(divide = 'ignore', invalid = 'ignore'):
    #    acc_cls = np.diag(confusion_matrix)/confusion_matrix.sum(axis = 1)
    #    iu = np.diag(confusion_matrix)/(confusion_matrix.sum(axis = 0) + \
    #                                    confusion_matrix.sum(axis = 1) - np.diag(confusion_matrix))

    ## use np.nanmean instead of np.mean because likely NaNs create in before due to zero divison
    #acc_cls = np.nanmean(acc_cls)
    #mean_iu = np.nanmean(iu)

    #return acc, acc_cls, mean_iu
    return label_accuracy, 0

def calc_cm(tl, pl, nc):
    # mask to ignore unlabeled pixels
    mask = (tl >= 0)&(tl<nc)
    cm = np.bincount(nc*tl[mask]+pl[mask], minlength = nc).reshape(nc,nc)
    return cm
