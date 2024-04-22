import torch
import torch.nn as nn
import numpy as np

torch.set_printoptions(profile="full")


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


class attention_gate(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(inc[0], outc, kernel_size=1, padding=0),
            nn.BatchNorm2d(outc),
        )

        self.Ws = nn.Sequential(
            nn.Conv2d(inc[1], outc, kernel_size=1, padding=0),
            nn.BatchNorm2d(outc),
        )

        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(outc, outc, kernel_size=1, padding=0),
            # nn.Softmax(dim=1),
            nn.Sigmoid(),
        )

    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)

        # with open(f"{g.shape[1]}_before_sigmoid_tensor.txt", "w") as f:
        #     tmp = out.detach().cpu()
        #     x_tmp = str(tmp)
        #     f.write(x_tmp)
        #     print(out.max(), out.min())

        out = self.output(out)

        # with open(f"{g.shape[1]}_tensor.txt", "w") as f:
        #     tmp = out.detach().cpu()
        #     x_tmp = str(tmp)
        #     f.write(x_tmp)
        #     print(out.max(), out.min())

        return out * s


class EncoderPath(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.e1 = encoder_block(inc, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        return p3, s1, s2, s3


class decoder_block(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True)
        self.ag = attention_gate(inc, outc)
        self.c1 = conv_block(inc[0] + outc, outc)

    def forward(self, x, s):
        x = self.up(x)
        s = self.ag(x, s)
        x = torch.concat([x, s], dim=1)
        x = self.c1(x)
        return x


class DecoderPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = decoder_block([512, 256], 256)
        self.d2 = decoder_block([256, 128], 128)
        self.d3 = decoder_block([128, 64], 64)

    def forward(self, b1, s1, s2, s3):
        d1 = self.d1(b1, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        return d3


class Attention_Unet(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)

        self.b1 = conv_block(256, 512)

        self.d1 = decoder_block([512, 256], 256)
        self.d2 = decoder_block([256, 128], 128)
        self.d3 = decoder_block([128, 64], 64)

        self.output = nn.Conv2d(64, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        b1 = self.b1(p3)

        d1 = self.d1(b1, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        # print(f"b1: {b1.shape}")
        # print(f"s1: {s1.shape} || s2: {s2.shape} || s3: {s3.shape}")

        output = self.output(d3)

        return output


class Attention_Unet_Refracted(nn.Module):
    def __init__(self, num_class=8):
        super().__init__()

        self.features = EncoderPath(inc=1)
        self.bottle_neck = conv_block(256, 512)
        self.reconstructor = DecoderPath()

        self.output = nn.Conv2d(64, num_class, kernel_size=1, padding=0)

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        p3, s1, s2, s3 = self.features(x)

        b1 = self.bottle_neck(p3)

        d3 = self.reconstructor(b1, s1, s2, s3)

        output = self.output(d3)

        return output


class SegGradModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features_conv = model.features
        self.bottle_neck = model.bottle_neck
        self.reconstructor = model.reconstructor
        self.output = model.output

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        p3, s1, s2, s3 = self.features_conv(x)

        return self.bottle_neck(p3)

    def forward(self, x):
        p3, s1, s2, s3 = self.features_conv(x)
        b1 = self.bottle_neck(p3)

        # register hook on bottle neck layer
        h = b1.register_hook(self.activations_hook)

        d3 = self.reconstructor(b1, s1, s2, s3)

        output = self.output(d3)

        return output


class Attention_Unet_Seg_Grad_Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.e1 = model.e1
        self.e2 = model.e2
        self.e3 = model.e3
        self.b1 = model.b1
        self.d1 = model.d1
        self.d2 = model.d2
        self.d3 = model.d3
        self.output = model.output

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        _, p1 = self.e1(x)
        _, p2 = self.e2(p1)
        _, p3 = self.e3(p2)

        b1 = self.b1(p3)

        return b1

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        b1 = self.b1(p3)

        h = b1.register_hook(self.activations_hook)

        d1 = self.d1(b1, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        output = self.output(d3)

        return output


if __name__ == "__main__":
    model = Attention_Unet()
    x = torch.rand(3, 1, 128, 128)
    output = model(x)
    print(output.shape)
    print(model.bottle_neck)
