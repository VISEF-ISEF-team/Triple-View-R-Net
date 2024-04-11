import torch
from torch import nn

__all__ = ['UNet', 'NestedUNet']


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


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[4] + nb_filter[3], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[3] + nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[2] + nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[1] + nb_filter[0], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        print('UNET')
        print(f'Input X: {input.size()}')
        
        # Downsampling
        print("DOWNSAMPLING")
        x0_0 = self.conv0_0(input)
        print(f'x0_0: {x0_0.size()}')

        x1_0 = self.conv1_0(self.pool(x0_0))
        print(f'x1_0: {x1_0.size()}')
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        print(f'x2_0: {x2_0.size()}')
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        print(f'x3_0: {x3_0.size()}')
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        print(f'x4_0: {x4_0.size()}')

        # Upsampling
        print("UPSAMPLING")
        print(f'Convtranspose x4_0: {self.up(x4_0).size()}')
        print(f'Concat: {torch.cat([x3_0, self.up(x4_0)], 1).size()}')
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))        
        print(f'Conv one more time after concat - x3_1: {x3_1.size()}')
        
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        print(f'x2_2: {x2_2.size()}')
        
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        print(f'x1_3: {x1_3.size()}')
        
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        print(f'x0_4: {x0_4.size()}')

        output = self.final(x0_4)
        print(f'Output Y: {output.size()}')
        
        return output
    
'''    
class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        print('NESTED UNET')
        
        
        x0_0 = self.conv0_0(input)
        print(f'x0_0: {x0_0.size()}')
        x1_0 = self.conv1_0(self.pool(x0_0))
        print(f'x1_0: {x1_0.size()}')
        up_x1_0 = self.up(x1_0)
        print(f'up_x1_0: {up_x1_0.size()}')
        concat_x0_0_x1_0 = torch.cat([x0_0, up_x1_0], 1)
        print(f'concat_x0_0__x1_0: {concat_x0_0_x1_0.size()}')
        x0_1 = self.conv0_1(concat_x0_0_x1_0)
        print(f'x0_1 after conv: {x0_1.size()}')
        

        x2_0 = self.conv2_0(self.pool(x1_0))
        print(f'\nx2_0: {x2_0.size()}')
        
        up_x2_0 = self.up(x2_0)
        print(f'up_x2_0: {up_x2_0.size()}')
        concat_x1_0_x2_0 = torch.cat([x1_0, up_x2_0], 1)
        print(f'concat_x1_0__x2_0: {concat_x1_0_x2_0.size()}')
        x1_1 = self.conv1_1(concat_x1_0_x2_0)
        print(f'x1_1 after conv: {x1_1.size()}')
        
        up_x1_1 = self.up(x1_1)
        print(f'up_x1_1: {up_x1_1.size()}')
        concat_x0_0__x0_1__x1_1 = torch.cat([x0_0, x0_1, up_x1_1], 1)
        print(f'concat_x0_0__x1_1: {concat_x0_0__x0_1__x1_1.size()}')
        x0_2 = self.conv0_2(concat_x0_0__x0_1__x1_1)
        print(f'x0_2 after conv: {x0_2.size()}')


        x3_0 = self.conv3_0(self.pool(x2_0))
        print(f'x3_0: {x3_0.size()}')
        up_x3_0 = self.up(x3_0)
        print(f'up_x3_0: {up_x3_0.size()}')
        concat_x2_0__x3_0 = torch.cat([x2_0, up_x3_0], 1)
        print(f'concat_x2_0__x3_0: {concat_x2_0__x3_0.size()}')
        x2_1 = self.conv2_1(concat_x2_0__x3_0)
        print(f'x2_1 after conv: {x2_1.size()}')
        
        up_x2_1 = self.up(x2_1)
        print(f'up_x2_1: {up_x2_1.size()}')
        concat_x1_0__x1_1__up_x2_1 = torch.cat([x1_0, x1_1, up_x2_1], 1)
        print(f'concat_x1_0__x1_1__up_x2_1: {concat_x1_0__x1_1__up_x2_1.size()}')
        x1_2 = self.conv1_2(concat_x1_0__x1_1__up_x2_1)
        print(f'x1_2 after conv: {x1_2.size()}')
        
        up_x1_2 = self.up(x1_2)
        print(f'up_x1_2: {up_x1_2.size()}')
        concat_x0_0__x0_1__x0_2__up_x1_2 = torch.cat([x0_0, x0_1, x0_2, up_x1_2], 1)
        print(f'concat_x0_0__x0_1__x0_2__up_x1_2: {concat_x0_0__x0_1__x0_2__up_x1_2.size()}')
        x0_3 = self.conv0_3(concat_x0_0__x0_1__x0_2__up_x1_2)
        print(f'x0_3 after conv: {x0_3.size()}')





        x4_0 = self.conv4_0(self.pool(x3_0))
        print(f'x4_0: {x4_0.size()}')        
                
        up_x4_0 = self.up(x4_0)
        print(f'up_x4_0: {up_x4_0.size()}')
        concat_x3_0__x4_0 = torch.cat([x3_0, up_x4_0], 1)
        print(f'concat_x3_0__x4_0: {concat_x3_0__x4_0.size()}')
        x3_1 = self.conv3_1(concat_x3_0__x4_0)
        print(f'x3_1 after conv: {x3_1.size()}')
        
        up_x3_1 = self.up(x3_1)
        print(f'up_x3_1: {up_x3_1.size()}')
        concat_x2_0__x2_1__x3_1 = torch.cat([x2_0, x2_1, up_x3_1], 1)
        print(f'concat_x2_0__x2_1__x3_1: {concat_x2_0__x2_1__x3_1.size()}') 
        x2_2 = self.conv2_2(concat_x2_0__x2_1__x3_1)
        print(f'x2_2 after conv: {x2_2.size()}')
        
        up_x2_2 = self.up(x2_2)
        print(f'up_x2_2: {up_x2_2.size()}')
        concat_x1_0__x1_1__x1_2 = torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1)
        print(f'concat_x1_0__x1_1__x1_2: {concat_x1_0__x1_1__x1_2.size()}')
        x1_3 = self.conv1_3(concat_x1_0__x1_1__x1_2)
        print(f'x1_3 after conv: {x1_3.size()}')
        
        up_x1_3 = self.up(x1_3)
        print(f'up_x1_3: {up_x1_3.size()}')
        concat_x0_0__x0_1__x0_2__x0_3 = torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1)
        print(f'concat_x0_0__x0_1__x0_2__x0_3: {concat_x0_0__x0_1__x0_2__x0_3.size()}')
        x0_4 = self.conv0_4(concat_x0_0__x0_1__x0_2__x0_3)
        print(f'x0_4 after conv: {x0_4.size()}')
        

        print('DEEP SUPERVISION')
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            print(f'output1 after final1: {output1.size()}')
            output2 = self.final2(x0_2)
            print(f'output2 after final2: {output2.size()}')
            output3 = self.final3(x0_3)
            print(f'output3 after final3: {output3.size()}')
            output4 = self.final4(x0_4)
            print(f'output4 after final4: {output4.size()}')
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output
'''        
        
class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        
        # conv bình thường cho downsample
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        # 2 lần là block nào cũng có
        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0]) # 2 lần
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1]) # 2 lần
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2]) # 2 lần
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3]) # 2 lần

        # 3 lần là block cuối (block 3) không có
        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0]) # 3 lần
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1]) # 3 lần
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2]) # 3 lần

        # 4 lần là block gần cuối (block 2) không có
        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0]) # 4 lần
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1]) # 4 lần

        # 5 lần là chỉ có block đầu có
        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0]) # 5 lần

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        
        x1_0 = self.conv1_0(self.pool(x0_0)) # downsample bình thường
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))  # liên kết của block 1 với block 0 -> layer 1 của block 0

        x2_0 = self.conv2_0(self.pool(x1_0)) # downsample bình thường
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))  # liên kết của block 2 với block 1 -> layer 1 của block 1
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1)) # liên kết của block 2 với block 0 -> layer 2 của block 0

        x3_0 = self.conv3_0(self.pool(x2_0)) # downsample bình thường
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1)) # liên kết của block 3 với block 2 -> layer 1 của block 2
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1)) # liên kết của block 3 với block 1 -> layer 2 của block 1
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1)) # liên kết của block 3 với block 0 -> layer 3 của block 0

        x4_0 = self.conv4_0(self.pool(x3_0)) # downsample bình thường
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))  # liên kết của block 4 với block 3 -> layer 1 của block 3
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1)) # liên kết của block 4 với block 2 -> layer 2 của block 2
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1)) # liên kết của block 4 với block 1 -> layer 3 của block 1
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))  # liên kết của block 4 với block 0 -> layer 4 của block 1
        
        print(f'x0_4 after conv: {x0_4.size()}')
        

        print('DEEP SUPERVISION')
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            print(f'output block 0 layer 1 after final1: {output1.size()}')
            output2 = self.final2(x0_2)
            print(f'output block 0 layer 2 after final2: {output2.size()}')
            output3 = self.final3(x0_3)
            print(f'output block 0 layer 3 after final3: {output3.size()}')
            output4 = self.final4(x0_4)
            print(f'output block 0 layer 4 after final4: {output4.size()}')
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output