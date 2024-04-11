import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_extraction import ConvBatchNorm


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x)
        )
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d(
            x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)

        avg_pool_g = F.avg_pool2d(
            g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)

        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(
            2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale

        out = self.relu(x_after_channel)
        return out


class UpBlockAtt(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)

        layers = []
        layers.append(ConvBatchNorm(in_channels, out_channels))
        for _ in range(nb_Conv - 1):
            layers.append(ConvBatchNorm(out_channels, out_channels))
        self.nConvs = nn.Sequential(*layers)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)
        return self.nConvs(x)


class UpSampling(nn.Module):
    def __init__(self, config):
        super().__init__()
        df = config.df
        self.up4 = UpBlockAtt(df[3]*2, df[2], nb_Conv=2)
        self.up3 = UpBlockAtt(df[3], df[1], nb_Conv=2)
        self.up2 = UpBlockAtt(df[2], df[0], nb_Conv=2)
        self.up1 = UpBlockAtt(df[1], df[0], nb_Conv=2)

    def forward(self, enc1, enc2, enc3, enc4, x5):
        x = self.up4(x5, enc4)
        x = self.up3(x, enc3)
        x = self.up2(x, enc2)
        x = self.up1(x, enc1)
        return x
