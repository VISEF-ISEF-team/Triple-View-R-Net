import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.original import UNet, NestedUNet
from networks.new import UNetBlock 
from networks.Unet_arch import UnetArch

def train():
    input = torch.randn(1, 3, 512, 512)
    
    # print("UNET 1")
    # model = UnetArch(in_channels=3, num_classes=12)
    # output = model(input)
    
    # print("UNET 2")
    # model2 = UNet(num_classes=12)
    # output2 = model2(input)
    
    model = NestedUNet(num_classes=12, deep_supervision=True)
    output = model(input)
    print(len(output))

train()