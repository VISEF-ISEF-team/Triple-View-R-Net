import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class DoubleConvolution(nn.Module):
    def __init__(self, inc, outc):
        super(DoubleConvolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(inc, outc, 3, 1, padding=1, bias=False),
            nn.BatchNorm3d(outc),
            nn.ReLU(inplace=True),

            nn.Conv3d(outc, outc, 3, 1, padding=1, bias=False),
            nn.BatchNorm3d(outc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Unet3D(nn.Module):
    def __init__(self, inc=1, outc=1, features=[64, 128, 256, 512]):
        super(Unet3D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # down part
        for feature in features:
            self.downs.append(DoubleConvolution(inc, feature))
            inc = feature

        # up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConvolution(feature*2, feature))

        self.bottle_neck = DoubleConvolution(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv3d(features[0], outc, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottle_neck(x)

        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            x = torch.concat((x, skip_connections[i // 2]), dim=1)
            x = self.ups[i + 1](x)

        return self.final_conv(x)


if __name__ == "__main__":
    x = torch.rand((1, 1, 128, 128, 128))
    model = Unet3D(inc=1, outc=8)
    preds = model(x)

    print(preds.shape)
    print(x.shape)

    print(preds)
