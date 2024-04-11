import torch
import torch.nn as nn
from rotatory_attention_model import RotatoryAttentionModule


class conv_block(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class encoder_block(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = conv_block(inc, outc)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)

        return s, p


class decoder_block(nn.Module):
    def __init__(self, inc, outc, flattened_dim):
        super().__init__()

        self.inc = inc
        self.outc = outc
        self.flattend_dim = flattened_dim

        self.up = nn.ConvTranspose2d(
            in_channels=inc, out_channels=outc, kernel_size=2, stride=2)

        self.rag = RotatoryAttentionModule(
            inc, flattened_dim, inc, flattened_dim, inc, flattened_dim, flattened_dim, inc // 4, flattened_dim, inc // 4, flattened_dim, inc // 4)

        self.c1 = conv_block(outc + outc, outc)

    def forward(self, x, s):
        # x and s must have same height and width
        # print(f"S: {s.shape}")
        # print(f"X: {x.shape}")

        # print(f'Inc: {self.inc} || Outc: {self.outc}')
        """get left right vector from output"""

        left = x[0, :, :, :].view(self.inc, -1)
        current = x[1, :, :, :].view(self.inc, -1)
        right = x[2, :, :, :].view(self.inc, -1)

        # print(
        #     f"Left: {left.shape} || Current: {current.shape} || Right: {right.shape}")

        output = self.rag(left, current, right)

        left = torch.unsqueeze(
            left.view(self.inc, int(self.flattend_dim ** 0.5), int(self.flattend_dim ** 0.5)), dim=0)

        right = torch.unsqueeze(right.view(
            self.inc, int(self.flattend_dim ** 0.5), int(self.flattend_dim ** 0.5)), dim=0)

        output = torch.unsqueeze(output.view(
            self.inc, int(self.flattend_dim ** 0.5), int(self.flattend_dim ** 0.5)), dim=0)

        new_output = torch.cat((left, output, right), dim=0)

        new_output = self.up(new_output)

        x = torch.concat([new_output, s], dim=1)
        x = self.c1(x)
        return x


class Rotatory_Attention_Unet(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)

        self.b1 = conv_block(256, 512)

        self.d1 = decoder_block(512, 256, int(image_size // (2 ** 3)) ** 2)
        self.d2 = decoder_block(256, 128, int(image_size // (2 ** 2)) ** 2)
        self.d3 = decoder_block(128, 64, int(image_size // (2 ** 1)) ** 2)

        self.output = nn.Conv2d(64, 8, kernel_size=1, padding=0)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        b1 = self.b1(p3)

        d1 = self.d1(b1, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        output = self.output(d3)

        return output


if __name__ == "__main__":
    model = Rotatory_Attention_Unet(image_size=256)
    x = torch.rand(3, 1, 256, 256)
    output = model(x)
    output = torch.unsqueeze(output[1, :, :, :], dim=0)
    print(output.shape)
