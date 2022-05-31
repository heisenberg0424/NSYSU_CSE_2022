"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from torchvision import transforms
class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=101, hparams=None):
        super().__init__()
        #self.hparams = hparams
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        self.num_classes = num_classes
        self.alexnet = models.alexnet(pretrained=True).features
        for param in self.alexnet.parameters():
            param.requires_grad = False
        self.pre = nn.Sequential(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.trans = nn.Sequential(
            nn.Conv2d(256,num_classes,1,1),
            nn.Upsample((341,256),mode = 'bilinear'),

        )

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        x = self.pre(x)
        x = self.alexnet(x)
        x = self.trans(x)
        # 256,6,6


        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

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

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
