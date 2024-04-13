from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import get_config
from feature_extraction import *
from encoder_transformer import *
from reconstruction import *
from decoder_attention import *
from rotatory_attention import *


class RotCAttTransUNetDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        df = config.df
        num_classes = config.num_classes

        self.down_sampling = DownSampling(config)
        self.channel_transformer = ChannelTransformer(config)
        self.rotatory_attention = RotatoryAttention(config)
        self.reconstruction = Reconstruction(config)
        self.up_sampling = UpSampling(config)
        self.out = nn.Conv2d(df[0], num_classes,
                             kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.down_sampling(x)
        emb1, emb2, emb3, emb4, enc1, enc2, enc3, enc4, a_weights = self.channel_transformer(
            x1, x2, x3, x4)
        r1, r2, r3, r4 = self.rotatory_attention(emb1, emb2, emb3, emb4)

        # Combine intra-slice information and interslice information
        er1 = enc1 + r1
        er2 = enc2 + r2
        er3 = enc3 + r3
        er4 = enc4 + r4

        d1, d2, d3, d4 = self.reconstruction(er1, er2, er3, er4)
        y = self.up_sampling(d1, d2, d3, d4, x5)
        y = self.out(y)
        return y, a_weights


if __name__ == "__main__":
    config = get_config()
    model = RotCAttTransUNetDense(config=config).cuda()
    input = torch.rand(3, 1, 256, 256).cuda()
    logits, _ = model(input)
    print(logits.size())
