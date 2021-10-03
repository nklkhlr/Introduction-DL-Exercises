"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, pretrained_model = 'resnet50'):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        # define inpute dimensions
        channels, height, width = 3, 240, 240
        self.num_classes = num_classes

        # initialized pretrained model
        pretrained_model = getattr(models, pretrained_model)
        self.resnet = pretrained_model(pretrained = True)

        #### Resnet architecture ####
        expansion_rate = self.resnet.layer1[0].expansion

        self.score_8 = nn.Conv2d(128*expansion_rate, num_classes, kernel_size = 1)
        self.score_16 = nn.Conv2d(256*expansion_rate, num_classes, kernel_size = 1)
        self.score_32 = nn.Conv2d(512*expansion_rate, num_classes, kernel_size = 1)

        ##### VGG archtitecture ####
        ## get last channel size of last convolution
        ## ADJUST IF USING OTHER MODEL!
        #last_conv = pretrained_layers[-3].out_channels
        ## leave out last layer as this will result in invalid dimensions
        #self.pretrained_features = nn.Sequential(*pretrained_layers[:-1])

        # note: might have to be adapted to fit to all available pytorch models
        #assert channels == self.pretrained_features[0].in_channels,\
        #        'Size innput channels: %i does not match expected channel size: %i'%(channels,\
        #                                                self.pretrained_features[0].in_channels)

        ## calculate output dimensions of pretrained model to fit model
        ## might have to be adjusted for models other than vgg (and non squared pictures)
        #sizes = [height]
        #for i, layer in enumerate(list(self.pretrained_features)):
        #    if isinstance(layer, nn.ReLU):
        #        sizes.append(sizes[i])
        #    elif isinstance(layer, nn.Conv2d):
        #        pad = layer.padding[0]
        #        kernel_size = layer.kernel_size[0]
        #        stride = layer.stride[0]
        #        size = ((sizes[i]+2*pad-(kernel_size-1)-1)//stride+1)
        #        sizes.append(size)
        #    else:
        #        pad = layer.padding
        #        kernel_size = layer.kernel_size
        #        stride = layer.stride
        #        size = ((sizes[i]+2*pad-(kernel_size-1)-1)//stride+1)
        #        sizes.append(size)

        ## calculate kernel sizes for last conv layers to ensure correct output dimensions
        #dim_conv_m = sizes[-1]-4
        #kernel_conv_l = -(height-1)+dim_conv_m
        # initialize segmentation model
        #self.model = nn.Sequential( nn.Conv2d(last_conv.out_channels, 256, 1),
        #                            nn.ReLU(True),
        #                            nn.Dropout2d(),
        #                            nn.Conv2d(256, 128, 1),
        #                            nn.ReLU(True),
        #                            nn.Conv2d(128, 128, 1),
        #                            nn.ReLU(True),
        #                            nn.ConvTranspose2d(128, num_classes, 4,
        #                                               stride = 4),
        #                            nn.ConvTranspose2d(num_classes, num_classes, 4, stride = 4)
        #                           )
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        input_dim = x.size()[2:]

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        l2  = self.score_8(x)
        x = self.resnet.layer3(x)
        l3  = self.score_16(x)
        x = self.resnet.layer4(x)
        l4 = self.score_32(x)

        l3 += nn.functional.upsample_bilinear(l4, size = l3.size()[2:])
        l2 += nn.functional.upsample_bilinear(l3, size = l2.size()[2:])
        x = nn.functional.upsample_bilinear(l2, size = input_dim)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
