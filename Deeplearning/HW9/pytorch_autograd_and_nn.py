"""
Implements pytorch autograd and nn in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
import torch.nn as nn
from a4_helper import *
import torch.nn.functional as F
import torch.optim as optim

def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from pytorch_autograd_and_nn.py!')



################################################################################
# Part II. Barebones PyTorch                         
################################################################################
# Before we start, we define the flatten function for your convenience.
def flatten(x, start_dim=1, end_dim=-1):
  return x.flatten(start_dim=start_dim, end_dim=end_dim)


def three_layer_convnet(x, params):
  """
  Performs the forward pass of a three-layer convolutional network with the
  architecture defined above.

  Inputs:
  - x: A PyTorch Tensor of shape (N, C, H, W) giving a minibatch of images
  - params: A list of PyTorch Tensors giving the weights and biases for the
    network; should contain the following:
    - conv_w1: PyTorch Tensor of shape (channel_1, C, KH1, KW1) giving weights
      for the first convolutional layer
    - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
      convolutional layer
    - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
      weights for the second convolutional layer
    - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
      convolutional layer
    - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
      figure out what the shape should be?
    - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
      figure out what the shape should be?
  
  Returns:
  - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
  """
  conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
  scores = None
  ##############################################################################
  # TODO: Implement the forward pass for the three-layer ConvNet.              
  # The network have the following architecture:                               
  # 1. Conv layer (with bias) with 32 5x5 filters, with zero-padding of 2     
  #   2. ReLU                                                                  
  # 3. Conv layer (with bias) with 16 3x3 filters, with zero-padding of 1     
  # 4. ReLU                                                                   
  # 5. Fully-connected layer (with bias) to compute scores for 10 classes    
  # Hint: F.linear, F.conv2d, F.relu, flatten (implemented above)                                   
  ##############################################################################
  # Replace "pass" statement with your code
  x = F.relu( F.conv2d(x,conv_w1,conv_b1,padding=2))
  x = F.relu( F.conv2d(x,conv_w2,conv_b2,padding=1))
  x = flatten(x)
  scores = F.linear(x,fc_w,fc_b)

  ##############################################################################
  #                                 END OF YOUR CODE                             
  ##############################################################################
  return scores


def initialize_three_layer_conv_part2(dtype=torch.float, device='cpu'):
  '''
  Initializes weights for the three_layer_convnet for part II
  Inputs:
    - dtype: A torch data type object; all computations will be performed using
        this datatype. float is faster but less accurate, so you should use
        double for numeric gradient checking.
      - device: device to use for computation. 'cpu' or 'cuda'
  '''
  # Input/Output dimenssions
  C, H, W = 3, 32, 32
  num_classes = 10

  # Hidden layer channel and kernel sizes
  channel_1 = 32
  channel_2 = 16
  kernel_size_1 = 5
  kernel_size_2 = 3

  # Initialize the weights
  conv_w1 = None
  conv_b1 = None
  conv_w2 = None
  conv_b2 = None
  fc_w = None
  fc_b = None

  ##############################################################################
  # TODO: Define and initialize the parameters of a three-layer ConvNet           
  # using nn.init.kaiming_normal_. You should initialize your bias vectors    
  # using the zero_weight function.                         
  # You are given all the necessary variables above for initializing weights. 
  ##############################################################################
  # Replace "pass" statement with your code
  conv_w1 = nn.init.kaiming_normal_(torch.empty(channel_1, C, kernel_size_1, kernel_size_1, dtype=dtype, device='cuda'))
  conv_w1.requires_grad = True
  conv_b1 = nn.init.zeros_(torch.empty(channel_1, dtype=dtype, device='cuda'))
  conv_b1.requires_grad = True

  conv_w2 = nn.init.kaiming_normal_(torch.empty(channel_2, channel_1, kernel_size_2, kernel_size_2, dtype=dtype, device='cuda'))
  conv_w2.requires_grad = True
  conv_b2 = nn.init.zeros_(torch.empty(channel_2, dtype=dtype, device='cuda'))
  conv_b2.requires_grad = True

  fc_w = nn.init.kaiming_normal_(torch.empty(num_classes, channel_2*H*H, dtype=dtype, device='cuda'))
  fc_w.requires_grad = True
  fc_b = nn.init.zeros_(torch.empty(num_classes, dtype=dtype, device='cuda'))
  fc_b.requires_grad = True
  ##############################################################################
  #                                 END OF YOUR CODE                            
  ##############################################################################
  return [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]




################################################################################
# Part III. PyTorch Module API                         
################################################################################

class ThreeLayerConvNet(nn.Module):
  def __init__(self, in_channel, channel_1, channel_2, num_classes):
    super().__init__()
    ############################################################################
    # TODO: Set up the layers you need for a three-layer ConvNet with the       
    # architecture defined below. You should initialize the weight  of the
    # model using Kaiming normal initialization, and zero out the bias vectors.     
    #                                       
    # The network architecture should be the same as in Part II:          
  #   1. Convolutional layer with channel_1 5x5 filters with zero-padding of 2  
    #   2. ReLU                                   
    #   3. Convolutional layer with channel_2 3x3 filters with zero-padding of 1
    #   4. ReLU                                   
    #   5. Fully-connected layer to num_classes classes               
    #                                       
    # We assume that the size of the input of this network is `H = W = 32`, and   
    # there is no pooing; this information is required when computing the number  
    # of input channels in the last fully-connected layer.              
    #                                         
    # HINT: nn.Conv2d, nn.init.kaiming_normal_, nn.init.zeros_            
    ###########################################################C#################
    # Replace "pass" statement with your code
    self.conv1 = nn.Conv2d(in_channel,channel_1,5,padding=2)
    self.conv2 = nn.Conv2d(channel_1,channel_2,3,padding=1)
    self.fc = nn.Linear(32*32*channel_2,num_classes)

    nn.init.kaiming_normal_(self.conv1.weight)
    nn.init.kaiming_normal_(self.conv2.weight)
    nn.init.zeros_(self.conv1.bias)
    nn.init.zeros_(self.conv2.bias)
    nn.init.kaiming_normal_(self.fc.weight)
    nn.init.zeros_(self.fc.bias)
    ############################################################################
    #                           END OF YOUR CODE                            
    ############################################################################

  def forward(self, x):
    scores = None
    ############################################################################
    # TODO: Implement the forward function for a 3-layer ConvNet. you      
    # should use the layers you defined in __init__ and specify the       
    # connectivity of those layers in forward()   
    # Hint: flatten (implemented at the start of part II)                          
    ############################################################################
    # Replace "pass" statement with your code
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = flatten(x)
    scores = self.fc(x)
    ############################################################################
    #                            END OF YOUR CODE                          
    ############################################################################
    return scores


def initialize_three_layer_conv_part3():
  '''
  Instantiates a ThreeLayerConvNet model and a corresponding optimizer for part III
  '''

  # Parameters for ThreeLayerConvNet
  C = 3
  num_classes = 10

  channel_1 = 32
  channel_2 = 16

  # Parameters for optimizer
  learning_rate = 3e-3
  weight_decay = 1e-4

  model = None
  optimizer = None
  ##############################################################################
  # TODO: Instantiate ThreeLayerConvNet model and a corresponding optimizer.     
  # Use the above mentioned variables for setting the parameters.                
  # You should train the model using stochastic gradient descent without       
  # momentum, with L2 weight decay of 1e-4.                    
  ##############################################################################
  # Replace "pass" statement with your code
  model = ThreeLayerConvNet(C,channel_1,channel_2,num_classes)
  optimizer = optim.SGD(model.parameters(),lr = learning_rate, weight_decay=weight_decay)
  ##############################################################################
  #                                 END OF YOUR CODE                            
  ##############################################################################
  return model, optimizer


################################################################################
# Part IV. PyTorch Sequential API                        
################################################################################

# Before we start, We need to wrap `flatten` function in a module in order to stack it in `nn.Sequential`.
# As of 1.3.0, PyTorch supports `nn.Flatten`, so this is not required in the latest version.
# However, let's use the following `Flatten` class for backward compatibility for now.
class Flatten(nn.Module):
  def forward(self, x):
    return flatten(x)


def initialize_three_layer_conv_part4():
  '''
  Instantiates a ThreeLayerConvNet model and a corresponding optimizer for part IV
  '''
  # Input/Output dimenssions
  C, H, W = 3, 32, 32
  num_classes = 10

  # Hidden layer channel and kernel sizes
  channel_1 = 32
  channel_2 = 16
  kernel_size_1 = 5
  pad_size_1 = 2
  kernel_size_2 = 3
  pad_size_2 = 1

  # Parameters for optimizer
  learning_rate = 1e-2
  weight_decay = 1e-4
  momentum = 0.5

  model = None
  optimizer = None
  ##################################################################################
  # TODO: Rewrite the 3-layer ConvNet with bias from Part III with Sequential API and 
  # a corresponding optimizer.
  # You don't have to re-initialize your weight matrices and bias vectors.  
  # Here you should use `nn.Sequential` to define a three-layer ConvNet with:
  #   1. Convolutional layer (with bias) with 32 5x5 filters, with zero-padding of 2 
  #   2. ReLU                                      
  #   3. Convolutional layer (with bias) with 16 3x3 filters, with zero-padding of 1 
  #   4. ReLU                                      
  #   5. Fully-connected layer (with bias) to compute scores for 10 classes        
  #                                            
  # You should optimize your model using stochastic gradient descent with Nesterov   
  # momentum 0.5, with L2 weight decay of 1e-4 as given in the variables above.   
  # Hint: nn.Sequential, Flatten (implemented at the start of Part IV)   
  ####################################################################################
  # Replace "pass" statement with your code
  model = nn.Sequential(OrderedDict([
    ('conv1',nn.Conv2d(C,channel_1,5,padding=2)),
    ('relu1',nn.ReLU()),
    ('conv2',nn.Conv2d(channel_1,channel_2,3,padding=1)),
    ('relu2',nn.ReLU()),
    ('flat',Flatten()),
    ('fc',nn.Linear(32*32*channel_2,num_classes))
  ]))

  optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                      weight_decay=weight_decay,
                      momentum=momentum, nesterov=True)
  ################################################################################
  #                                 END OF YOUR CODE                             
  ################################################################################
  return model, optimizer


################################################################################
# Part V. ResNet for CIFAR-10                        
################################################################################

class PlainBlock(nn.Module):
  def __init__(self, Cin, Cout, downsample=False):
    super().__init__()

    self.net = None
    ############################################################################
    # TODO: Implement PlainBlock.                                             
    # Hint: Wrap your layers by nn.Sequential() to output a single module.     
    #       You don't have use OrderedDict.                                    
    # Inputs:                                                                  
    # - Cin: number of input channels                                          
    # - Cout: number of output channels                                        
    # - downsample: add downsampling (a conv with stride=2) if True            
    # Store the result in self.net.                                            
    ############################################################################
    # Replace "pass" statement with your code
    stride=2 if downsample else 1

    self.net = nn.Sequential(
      nn.BatchNorm2d(Cin),
      nn.ReLU(),
      nn.Conv2d(Cin,Cout,3,stride,1),
      nn.BatchNorm2d(Cout),
      nn.ReLU(),
      nn.Conv2d(Cout,Cout,3,1,1)
    )
    ############################################################################
    #                                 END OF YOUR CODE                         #
    ############################################################################

  def forward(self, x):
    return self.net(x)


class ResidualBlock(nn.Module):
  def __init__(self, Cin, Cout, downsample=False):
    super().__init__()

    self.block = None # F
    self.shortcut = None # G
    ############################################################################
    # TODO: Implement residual block using plain block. Hint: nn.Identity()    #
    # Inputs:                                                                  #
    # - Cin: number of input channels                                          #
    # - Cout: number of output channels                                        #
    # - downsample: add downsampling (a conv with stride=2) if True            #
    # Store the main block in self.block and the shortcut in self.shortcut.    #
    ############################################################################
    # Replace "pass" statement with your code
    self.block = PlainBlock(Cin, Cout, downsample)
    if downsample :
      self.shortcut = nn.Conv2d(Cin, Cout, 1, 2, 0)
    else :
      if Cin == Cout:
        self.shortcut = nn.Identity()
      else:
        self.shortcut = nn.Conv2d(Cin, Cout, 1, 1, 0)
    ############################################################################
    #                                 END OF YOUR CODE                         #
    ############################################################################
  
  def forward(self, x):
    return self.block(x) + self.shortcut(x)


class ResNet(nn.Module):
  def __init__(self, stage_args, Cin=3, block=ResidualBlock, num_classes=100):
    super().__init__()

    self.cnn = None
    ############################################################################
    # TODO: Implement the convolutional part of ResNet using ResNetStem,       #
    #       ResNetStage, and wrap the modules by nn.Sequential.                #
    # Store the model in self.cnn.                                             #
    ############################################################################
    # Replace "pass" statement with your code
    self.stage_args = stage_args

    self.cnn = nn.Sequential(
      ResNetStem(Cin, stage_args[0][0]), #cin, cout? why 32
      *[ResNetStage(*stage, block) for stage in stage_args]
    )


    ############################################################################
    #                                 END OF YOUR CODE                         #
    ############################################################################
    self.fc = nn.Linear(stage_args[-1][1], num_classes)
  
  def forward(self, x):
    scores = None
    ############################################################################
    # TODO: Implement the forward function of ResNet.                          #
    # Store the output in `scores`.                                            #
    ############################################################################
    # Replace "pass" statement with your code
    pooling = nn.AdaptiveAvgPool2d(1)

    x = self.cnn(x)
    x = pooling(x)
    x = flatten(x)
    scores = self.fc(x)

    ############################################################################
    #                                 END OF YOUR CODE                         #
    ############################################################################
    return scores


class ResidualBottleneckBlock(nn.Module):
  def __init__(self, Cin, Cout, downsample=False):
    super().__init__()

    self.block = None
    self.shortcut = None
    ############################################################################
    # TODO: Implement residual bottleneck block.                               #
    # Inputs:                                                                  #
    # - Cin: number of input channels                                          #
    # - Cout: number of output channels                                        #
    # - downsample: add downsampling (a conv with stride=2) if True            #
    # Store the main block in self.block and the shortcut in self.shortcut.    #
    ############################################################################
    # Replace "pass" statement with your code
    stride=2 if downsample else 1

    self.block = nn.Sequential(
      nn.BatchNorm2d(Cin),
      nn.ReLU(),
      nn.Conv2d(Cin,Cout//4,1),
      nn.BatchNorm2d(Cout//4),
      nn.ReLU(),
      nn.Conv2d(Cout//4,Cout//4,3,stride,1),
      nn.BatchNorm2d(Cout//4),
      nn.ReLU(),
      nn.Conv2d(Cout//4,Cout,1)
    )

    if downsample :
      self.shortcut = nn.Conv2d(Cin, Cout, 1, 2, 0)
    else :
      if Cin == Cout:
        self.shortcut = nn.Identity()
      else:
        self.shortcut = nn.Conv2d(Cin, Cout, 1, 1, 0)
    ############################################################################
    #                                 END OF YOUR CODE                         #
    ############################################################################

  def forward(self, x):
    return self.block(x) + self.shortcut(x)

##############################################################################
# No need to implement anything here                     
##############################################################################
class ResNetStem(nn.Module):
  def __init__(self, Cin=3, Cout=8):
    super().__init__()
    layers = [
        nn.Conv2d(Cin, Cout, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
    ]
    self.net = nn.Sequential(*layers)
    
  def forward(self, x):
    return self.net(x)

class ResNetStage(nn.Module):
  def __init__(self, Cin, Cout, num_blocks, downsample=True,
               block=ResidualBlock):
    super().__init__()
    blocks = [block(Cin, Cout, downsample)]
    for _ in range(num_blocks - 1):
      blocks.append(block(Cout, Cout))
    self.net = nn.Sequential(*blocks)
  
  def forward(self, x):
    return self.net(x)


"""
Creates a EfficientNetV2 Model as defined in:
Mingxing Tan, Quoc V. Le. (2021). 
EfficientNetV2: Smaller Models and Faster Training
arXiv preprint arXiv:2104.00298.
import from https://github.com/d-li14/mobilenetv2.pytorch
"""

import torch
import torch.nn as nn
import math

#__all__ = ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

 
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, cfgs, num_classes=100, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_m(**kwargs):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  3, 1, 0],
        [4,  48,  5, 2, 0],
        [4,  80,  5, 2, 0],
        [4, 160,  7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512,  5, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_l(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  7, 2, 0],
        [4,  96,  7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640,  7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_xl(**kwargs):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  8, 2, 0],
        [4,  96,  8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640,  8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)