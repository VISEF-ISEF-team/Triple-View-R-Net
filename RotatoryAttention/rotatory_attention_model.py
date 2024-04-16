import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from config import get_config


class RotatoryAttentionModule(nn.Module):
    def __init__(self, nft=4, dft=3, nfl=6, dfl=5, nfr=8, dfr=7, dkl=8, dvl=8, dkr=9, dvr=9, dkt=10, dvt=10):
        super().__init__()
        """Dimension of input feature matrices"""
        self.dft = dft
        self.nft = nft
        self.dfl = dfl
        self.nfl = nfl
        self.dfr = dfr
        self.nfr = nfr

        """Dimension of keys and values"""
        self.dkl = dkl
        self.dvl = dvl
        self.dkr = dkr
        self.dvr = dvr
        self.dkt = dkt
        self.dvt = dvt

        """Key and Value matrices from Feature matrix"""
        self.lkeyweight = nn.Linear(dfl, self.dkl, bias=False)
        self.lvalueweight = nn.Linear(dfl, self.dvl, bias=False)

        self.rkeyweight = nn.Linear(dfr, self.dkr, bias=False)
        self.rvalueweight = nn.Linear(dfr, self.dvr, bias=False)

        self.tkeyweight = nn.Linear(dft, self.dkt, bias=False)
        self.tvalueweight = nn.Linear(dft, self.dvt, bias=False)

        """Weights for scoring function"""
        self.left_attention_score_bias = nn.Parameter(torch.zeros(1))
        self.left_attention_score_weight = nn.Parameter(
            torch.rand(self.dkl, self.dft))

        self.right_attention_score_bias = nn.Parameter(torch.zeros(1))
        self.right_attention_score_weight = nn.Parameter(
            torch.rand(self.dkr, dft))

        self.left_t_attention_score_bias = nn.Parameter(torch.zeros(1))
        self.left_t_attention_score_weight = nn.Parameter(
            torch.rand(self.dkt, self.dvl))

        self.right_t_attention_score_bias = nn.Parameter(torch.zeros(1))
        self.right_t_attention_score_weight = nn.Parameter(
            torch.rand(self.dkt, self.dvr))

        self.final_t_weights = nn.Linear(
            1, self.dft, bias=False)

    def calculate_attention_score(self, query, key, weight, bias):
        E = torch.matmul(key, torch.unsqueeze(torch.matmul(
            weight, query), 1)) + bias

        E = nn.functional.tanh(E)
        return E

    def calculate_attention_score_t(self, query, key, weight, bias):
        E = torch.matmul(key, torch.matmul(
            weight, query)) + bias

        E = nn.functional.tanh(E)
        return E

    def attention_align(self, E, V):
        # E has shape (N x 1)
        # V has shape (N, dv)
        A = nn.functional.softmax(E, dim=0)
        A = torch.transpose(A, 0, 1)

        C = torch.matmul(A, V) / torch.sum(A)
        C = torch.transpose(C, 0, 1)
        return C

    def final_context_transformation(self, c, weights):
        return torch.matmul(c, weights)

    def forward(self, Ft, Fl, Fr):
        rt = torch.mean(Ft, dim=0)  # (dft, 1)

        Kl = self.lkeyweight(Fl)  # nfl x dkl
        Vl = self.lvalueweight(Fl)  # nfl x dvl
        Kr = self.rkeyweight(Fr)  # nfr, dkr
        Vr = self.rvalueweight(Fr)  # nfr, dvr
        Kt = self.tkeyweight(Ft)  # nft, dkt
        Vt = self.tvalueweight(Ft)  # nft, dvt

        El = self.calculate_attention_score(
            rt, Kl, self.left_attention_score_weight, self.left_attention_score_bias)
        Er = self.calculate_attention_score(
            rt, Kr, self.right_attention_score_weight, self.right_attention_score_bias)

        Rl = self.attention_align(El, Vl)
        Rr = self.attention_align(Er, Vr)

        Elt = self.calculate_attention_score_t(
            Rl, Kt, self.left_t_attention_score_weight, self.left_t_attention_score_bias)
        Ert = self.calculate_attention_score_t(
            Rr, Kt, self.right_t_attention_score_weight, self.right_t_attention_score_bias)

        Rlt = self.attention_align(Elt, Vt)
        Rrt = self.attention_align(Ert, Vt)

        R = torch.concat((Rl, Rr, Rlt, Rrt), dim=0)
        R = self.final_t_weights(R)
        return R


class ActivatedGeneral(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W = nn.Linear(config.d_k, config.d_f)
        self.bias = nn.Parameter(torch.zeros(1))

        self.act = nn.Tanh()
        self.align = nn.Softmax(dim=0)

    def forward(self, r, K):
        e = self.act(self.W(K) @ r + self.bias)
        a = self.align(e)
        return a


class SingleConvolutionalRotatoryAttentionModule(nn.Module):
    def __init__(self, config, channels=768, kernel_size=3, stride_q=1, stride_kv=1, padding="same"):
        super().__init__()

        self.conv_k = nn.Conv2d(channels, channels, kernel_size,
                                stride_kv, padding, bias=True, groups=channels)
        self.layernorm_k = nn.LayerNorm(channels, eps=1e-5)

        self.conv_v = nn.Conv2d(channels, channels, kernel_size,
                                stride_kv, padding, bias=True, groups=channels)
        self.layernorm_v = nn.LayerNorm(channels, eps=1e-5)

        self.W_K = nn.Linear(config.d_f, config.d_k)
        self.W_V = nn.Linear(config.d_f, config.d_v)

        self.score = ActivatedGeneral(config)

    def _build_projection(self, x, qkv):
        if qkv == "k":
            x1 = F.relu(self.conv_k(x))  # (768, 1, 256)
            x1 = x1.permute(1, 2, 0)  # (1, 256, 768)
            x1 = self.layernorm_k(x1)
            proj = torch.squeeze(x1, 0)
            proj = self.W_K(proj)

        elif qkv == "v":
            x1 = F.relu(self.conv_v(x))
            x1 = x1.permute(1, 2, 0)  # (1, 256, 768)
            x1 = self.layernorm_v(x1)
            proj = torch.squeeze(x1, 0)
            proj = self.W_V(proj)

        return proj

    def forward_conv(self, x):
        # left (256, 768)
        x = torch.unsqueeze(x, 0)  # (1, 256, 768)
        x = x.permute(2, 0, 1)  # (768, 1, 256)
        k = self._build_projection(x, "k")
        v = self._build_projection(x, "v")

        return k, v

    def forward(self, r, x):

        K, V = self.forward_conv(x)  # (1, 256, dk) / (1, 256, dv)
        a = self.score(r, K)
        r_x = torch.matmul(V.transpose(-1, -2), a)
        return r_x


class ConvolutionalRotatoryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.single_attention_left = SingleConvolutionalRotatoryAttentionModule(
            config, channels=config.d_v, kernel_size=3, stride_q=1, stride_kv=1, padding="same")

        self.single_attention_right = SingleConvolutionalRotatoryAttentionModule(
            config, channels=config.d_v, kernel_size=3, stride_q=1, stride_kv=1, padding="same")

        self.single_attention_left_t = SingleConvolutionalRotatoryAttentionModule(
            config, channels=config.d_v, kernel_size=3, stride_q=1, stride_kv=1, padding="same")

        self.single_attention_right_t = SingleConvolutionalRotatoryAttentionModule(
            config, channels=config.d_v, kernel_size=3, stride_q=1, stride_kv=1, padding="same")

    def forward(self, x):
        left = x[0]
        target = x[1]
        right = x[2]

        r_t = torch.mean(target, dim=0)
        r_l = self.single_attention_left(r_t, left)
        r_r = self.single_attention_right(r_t, right)

        r_l_t = self.single_attention_left_t(r_l, target)
        r_r_t = self.single_attention_right_t(r_r, target)

        r = torch.concat([r_l, r_r, r_l_t, r_r_t])
        return r


class RotatoryAttentionOnModel(nn.Module):
    def __init__(self, image_width=256, image_height=256, outc=8):
        super().__init__()
        dft = image_height
        nft = image_width
        dfl = image_height
        nfl = image_width
        dfr = image_height
        nfr = image_width
        dkl = dft
        dvl = nft // 4
        dkr = dft
        dvr = nft // 4
        dkt = dft
        dvt = nft // 4

        self.attention_module = RotatoryAttentionModule(
            nft=nft, dft=dft, nfl=nfl, dfl=dfl, nfr=nfr, dfr=dfr, dkl=dkl, dvl=dvl, dkr=dkr, dvr=dvr, dkt=dkt, dvt=dvt)
        self.output_conv = nn.Conv2d(
            in_channels=1, out_channels=outc, kernel_size=3, padding=1, stride=1)

    def forward(self, left, current, right):
        attention_context = [self.attention_module(left[i], current[i], right[i])
                             for i in range(current.shape[0])]
        attention_context = torch.stack(attention_context)

        output = current * attention_context
        output = torch.unsqueeze(output, dim=1)
        output = self.output_conv(output)

        return output


def standalone_module_testing():
    dft = 256
    nft = 256
    dfl = 256
    nfl = 256
    dfr = 256
    nfr = 256
    dkl = 256
    dvl = 64
    dkr = 256
    dvr = 64
    dkt = 256
    dvt = 64

    # dvr, dvt, dvl = nft / 4
    attention_module = RotatoryAttentionModule(
        nft=nft, dft=dft, nfl=nfl, dfl=dfl, nfr=nfr, dfr=dfr, dkl=dkl, dvl=dvl, dkr=dkr, dvr=dvr, dkt=dkt, dvt=dvt)

    Ft = torch.randn(nft, dft)
    Fl = torch.randn(nfl, dfl)
    Fr = torch.randn(nfr, dfr)

    output = attention_module(Ft, Fl, Fr)

    print(f"Model output after forward: {output.shape}")


if __name__ == "__main__":
    standalone_module_testing()
    # model = RotatoryAttentionOnModel(image_width=256, image_height=256, outc=8)

    # input to attention will be a 3D tensor: (batch_size, sequence length, features)
    # current = torch.randn(3, 256, 256)
    # left = torch.randn(3, 256, 256)
    # right = torch.randn(3, 256, 256)

    # output = model(left, current, right)
    # print(output.shape)

    # i have picture of features, H, W => transform into features, H x W (flatten image)
    # let's say at last layer, it would be 1024, 16x16 = 1024, 256
    # if image of shape (256, 256) is paseed through model, and at some point it is (256, 64, 64)

    # image = torch.randn(256, 64, 64)
    # # multiply image by attention score
    # image_attended = output * image

    # print(image_attended.shape)

    # model = RotatoryAttentionModule(
    #     256, 64, 256, 64, 256, 64, 64, 256 // 4, 64, 256 // 4, 64, 256 // 4)

    # x = torch.rand(3, 256, 8, 8)

    # left = x[0, :, :, :].view(256, -1)
    # current = x[1, :, :, :].view(256, -1)
    # right = x[2, :, :, :].view(256, -1)

    # attention_context = model(left, current, right)
    # output = current * attention_context

    # left = torch.unsqueeze(left.view(256, 8, 8), dim=0)
    # right = torch.unsqueeze(right.view(256, 8, 8), dim=0)
    # output = torch.unsqueeze(output.view(256, 8, 8), dim=0)

    # print(left.shape, right.shape, current.shape)

    # new_output = torch.cat((left, output, right), dim=0)

    # print(new_output.shape)

    config = get_config()

    model = ConvolutionalRotatoryAttention(config)

    a = torch.rand(3, 256, 768)

    output = model(a)

    print(output.shape)
