import torch
import torch.nn as nn

class VGG11(nn.Module):
  def __init__(self, input_channels, out_channels):
    super(VGG11, self).__init__()
    inplace = True
    num_classes = 101
    pool_kernel_size = (2, 2)
    conv_kernel_size = (3, 3)

    self.conv_layer = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(input_channels, out_channels, kernel_size=conv_kernel_size, padding=1),
        nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=inplace)),

      nn.Sequential(    
        nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
        nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=inplace)),

      nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=inplace),
        nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=inplace),
        nn.MaxPool2d(kernel_size=(2, 2), stride=2)),

      nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=inplace),
        nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
        nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=inplace)),

      nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=inplace),
        nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
        nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=inplace))
      ])

    self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
     
    self.linear = nn.Sequential(
      nn.Linear(512*7*7, 4096),
      nn.ReLU(inplace=inplace),
      nn.Dropout(0.5),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=inplace),
      nn.Dropout(0.5),
      nn.Linear(4096, num_classes),
      nn.ReLU(inplace=inplace),
      nn.Softmax(dim=1)
    )
    
    self._init_layer()

    
  def _init_layer(self):
    for layer in self.linear:
      if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, 0, 0.01)
        nn.init.constant_(layer.bias, 0)
    
  def forward(self, x):
    for conv_layer in self.conv_layer:
      x = conv_layer(x)
    x = self.avg_pool(x)
    x = x.view(x.size(0), -1)
    x = self.linear(x)
    return x
  