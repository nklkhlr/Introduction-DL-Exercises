import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        #################################################################################
        # TODO: Define all the layers of this CNN, the only requirements are:           #
        # 1. This network takes in a square (same width and height), grayscale image as #
        #   input                                                                       #
        # 2. It ends with a linear layer that represents the keypoints                  #
        #   it's suggested that you make this last layer output 30 values, 2 for each of#
        #   the 15 keypoint (x, y) pairs                                                #
        #                                                                               #
        # Note that among the layers to add, consider including:                        #
        #   maxpooling layers, multiple conv layers, fully-connected layers, and other  #
        #   layers (such as dropout or batch normalization) to avoid overfitting.       #
        #################################################################################
        self.init_layer = nn.Sequential(
                                        nn.Conv2d(1, 32, kernel_size = 4),
                                        nn.ReLU(True),
                                        nn.MaxPool2d(2)
                                        )
        self.conv_layer1 = nn.Sequential(
                                        nn.Conv2d(32, 64, kernel_size = 3),
                                        nn.ReLU(True),
                                        nn.MaxPool2d(2),
                                        nn.Dropout(0.1)
                                        )
        self.conv_layer2 = nn.Sequential(
                                        nn.Conv2d(64, 128, kernel_size = 2),
                                        nn.ReLU(True),
                                        nn.MaxPool2d(2),
                                        nn.Dropout(0.2)
                                        )
        self.conv_layer3 = nn.Sequential(
                                        nn.Conv2d(128, 256, kernel_size = 1),
                                        nn.ReLU(True),
                                        nn.MaxPool2d(2),
                                        nn.Dropout(0.3)
                                        )


        # initialize conv weights from uniform distribution
        for layer in [self.init_layer, self.conv_layer1, self.conv_layer2, self.conv_layer3]:
            layer[0].weight.data.normal_(std = 0.001)

        # calculate input dimension for fully connected layer assuming squared size
        dim = 5

        self.fc1 = nn.Sequential(
                                nn.Linear(dim**2*256, 1000),
                                nn.ReLU(True),
                                nn.Dropout(0.5)
                                )
        self.fc2 = nn.Sequential(
                                nn.Linear(1000, 1000),
                                nn.ReLU(True),
                                nn.Dropout(0.5))
        self.final = nn.Linear(1000, 30)

        # initialize linear weights via xavier
        #for layer in [self.fc1, self.fc2]:
        #    nn.init.xavier_uniform_(layer[0].weight.data, gain = 1)
        #nn.init.xavier_uniform_(self.final.weight.data, gain = 1)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        #################################################################################
        # TODO: Define the feedforward behavior of this model                           #
        #       x is the input image and, as an example, here you may choose to include #
        #       a pool/conv step:                                                       #
        #                           x = self.pool(F.relu(self.conv1(x)))                #
        #        a modified x, having gone through all the layers of your model, should #
        #        be returned                                                            #
        #################################################################################
        x = self.init_layer(x)
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = x.view(-1, self.fc1[0].in_features)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.final(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
