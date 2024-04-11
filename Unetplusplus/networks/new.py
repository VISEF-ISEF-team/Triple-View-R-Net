import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        # print('VGGBLOCK')
        x = self.conv1(x)
        # print(f'x after conv1: {x.size()}')
        x = self.bn1(x)
        x = self.relu(x)
        # print(f'x after bn1 + relu: {x.size()}')
        
        x = self.conv2(x)
        # print(f'x after conv2: {x.size()}')
        x = self.bn2(x)
        x = self.relu(x)
        # print(f'x after bn2 + relu: {x.size()}')
        
        return x
    
class ConVMaxPool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv = VGGBlock(in_channels, in_channels, out_channels)
        
    def forward(self, x):
        x = self.max_pool(x)
        x = self.conv(x)
        return x
    
class UNetBlock(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super().__init__()
        
        self.filters = [input_channels, 32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2) 
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.downsample = nn.ModuleList([
            ConVMaxPool(self.filters[index], self.filters[index+1]) 
            for index in range(len(self.filters)-1)
        ])
        
        self.upsample = nn.ModuleList([
            ConVMaxPool(self.filters[index] + self.filters[index+1], self.filters[index]) 
            for index in range(len(self.filters)-2, 0, -1)
        ])
           
        self.final = nn.Conv2d(self.filters[0], num_classes, kernel_size=1) 
            
    def forward(self, x):
        print('UNETBLOCK')
        print(f'Input X: {x.size()}')
        
        down_outputs = []
        
        for downsample_layer in self.downsample:
            x = downsample_layer(x)
            down_outputs.append(x)
        
        x = down_outputs[-1]
        
        for upsample_layer, down_output in zip(self.upsample, reversed(down_outputs[:-1])):
            x = torch.cat([x, self.up(down_output)], dim=1)
            x = upsample_layer(x)
            
        output = self.final(x)
        print(f'Output Y: {output.size()}')
        
        return output

