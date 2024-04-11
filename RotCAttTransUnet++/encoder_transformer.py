import math
import copy
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair


class ChannelEmbeddings(nn.Module):
    def __init__(self, config, img_size, patch_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        n_patches = (img_size[0] // patch_size[0]) * \
            (img_size[1] // patch_size[1])

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=in_channels,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, in_channels))
        self.dropout = nn.Dropout(config.transformer.dropout_rate)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiheadChannelAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vis = config.vis
        self.KV_size = config.KV_size
        df = config.df
        self.n_heads = config.transformer.num_heads

        self.W_Q_1 = nn.ModuleList()
        self.W_Q_2 = nn.ModuleList()
        self.W_Q_3 = nn.ModuleList()
        self.W_Q_4 = nn.ModuleList()

        self.W_K = nn.ModuleList()
        self.W_V = nn.ModuleList()

        for _ in range(self.n_heads):
            W_Q_1_H = nn.Linear(df[0], df[0], bias=False)
            W_Q_2_H = nn.Linear(df[1], df[1], bias=False)
            W_Q_3_H = nn.Linear(df[2], df[2], bias=False)
            W_Q_4_H = nn.Linear(df[3], df[3], bias=False)

            W_K_H = nn.Linear(self.KV_size,  self.KV_size, bias=False)
            W_V_H = nn.Linear(self.KV_size,  self.KV_size, bias=False)

            self.W_Q_1.append(copy.deepcopy(W_Q_1_H))
            self.W_Q_2.append(copy.deepcopy(W_Q_2_H))
            self.W_Q_3.append(copy.deepcopy(W_Q_3_H))
            self.W_Q_4.append(copy.deepcopy(W_Q_4_H))

            self.W_K.append(copy.deepcopy(W_K_H))
            self.W_V.append(copy.deepcopy(W_V_H))

        self.psi = nn.InstanceNorm2d(self.n_heads)
        self.softmax = nn.Softmax(dim=3)

        self.W_O_1 = nn.Linear(df[0], df[0], bias=False)
        self.W_O_2 = nn.Linear(df[1], df[1], bias=False)
        self.W_O_3 = nn.Linear(df[2], df[2], bias=False)
        self.W_O_4 = nn.Linear(df[3], df[3], bias=False)

        self.attn_dropout = nn.Dropout(config.transformer.att_dropout_rate)
        self.proj_dropout = nn.Dropout(config.transformer.att_dropout_rate)

    def forward(self, emb1, emb2, emb3, emb4, emb_all):

        Q_1 = []
        Q_2 = []
        Q_3 = []
        Q_4 = []
        K = []
        V = []

        for W_Q_1_H in self.W_Q_1:
            Q_1_H = W_Q_1_H(emb1)
            Q_1.append(Q_1_H)

        for W_Q_2_H in self.W_Q_2:
            Q_2_H = W_Q_2_H(emb2)
            Q_2.append(Q_2_H)

        for W_Q_3_H in self.W_Q_3:
            Q_3_H = W_Q_3_H(emb3)
            Q_3.append(Q_3_H)

        for W_Q_4_H in self.W_Q_4:
            Q_4_H = W_Q_4_H(emb4)
            Q_4.append(Q_4_H)

        for W_K_H in self.W_K:
            K_H = W_K_H(emb_all)
            K.append(K_H)

        for W_V_H in self.W_V:
            V_H = W_V_H(emb_all)
            V.append(V_H)

        Q_1 = torch.stack(Q_1, dim=1)
        Q_2 = torch.stack(Q_2, dim=1)
        Q_3 = torch.stack(Q_3, dim=1)
        Q_4 = torch.stack(Q_4, dim=1)
        K = torch.stack(K, dim=1)
        V = torch.stack(V, dim=1)

        Q_1 = Q_1.transpose(-1, -2)
        Q_2 = Q_2.transpose(-1, -2)
        Q_3 = Q_3.transpose(-1, -2)
        Q_4 = Q_4.transpose(-1, -2)

        S1 = torch.matmul(Q_1, K) / math.sqrt(self.KV_size)
        S2 = torch.matmul(Q_2, K) / math.sqrt(self.KV_size)
        S3 = torch.matmul(Q_3, K) / math.sqrt(self.KV_size)
        S4 = torch.matmul(Q_4, K) / math.sqrt(self.KV_size)

        A1 = self.softmax(self.psi(S1))
        A2 = self.softmax(self.psi(S2))
        A3 = self.softmax(self.psi(S3))
        A4 = self.softmax(self.psi(S4))

        if self.vis:
            weights = []
            weights.append(A1.mean(1))
            weights.append(A2.mean(1))
            weights.append(A3.mean(1))
            weights.append(A4.mean(1))

        else:
            weights = None

        A1 = self.attn_dropout(A1)
        A2 = self.attn_dropout(A2)
        A3 = self.attn_dropout(A3)
        A4 = self.attn_dropout(A4)

        V = V.transpose(-1, -2)
        C1 = torch.matmul(A1, V)
        C2 = torch.matmul(A2, V)
        C3 = torch.matmul(A3, V)
        C4 = torch.matmul(A4, V)

        C1 = C1.permute(0, 3, 2, 1).contiguous()
        C2 = C2.permute(0, 3, 2, 1).contiguous()
        C3 = C3.permute(0, 3, 2, 1).contiguous()
        C4 = C4.permute(0, 3, 2, 1).contiguous()
        C1 = C1.mean(dim=3)
        C2 = C2.mean(dim=3)
        C3 = C3.mean(dim=3)
        C4 = C4.mean(dim=3)

        O1 = self.W_O_1(C1)
        O2 = self.W_O_2(C2)
        O3 = self.W_O_3(C3)
        O4 = self.W_O_4(C4)

        O1 = self.proj_dropout(O1)
        O2 = self.proj_dropout(O2)
        O3 = self.proj_dropout(O3)
        O4 = self.proj_dropout(O4)

        return O1, O2, O3, O4, weights


class MLP(nn.Module):
    def __init__(self, config, in_channel, mlp_channel):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()  # F.gelu
        self.dropout = nn.Dropout(config.transformer.dropout_rate)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        mlp_ratio = config.mlp_ratio
        df = config.df
        self.attn_norm1 = nn.LayerNorm(df[0], eps=1e-6)
        self.attn_norm2 = nn.LayerNorm(df[1], eps=1e-6)
        self.attn_norm3 = nn.LayerNorm(df[2], eps=1e-6)
        self.attn_norm4 = nn.LayerNorm(df[3], eps=1e-6)
        self.attn_norm = nn.LayerNorm(config.KV_size, eps=1e-6)
        self.multi_channel_attn = MultiheadChannelAttention(config)

        self.ffn_norm1 = nn.LayerNorm(df[0], eps=1e-6)
        self.ffn_norm2 = nn.LayerNorm(df[1], eps=1e-6)
        self.ffn_norm3 = nn.LayerNorm(df[2], eps=1e-6)
        self.ffn_norm4 = nn.LayerNorm(df[3], eps=1e-6)

        self.ffn1 = MLP(config, df[0], df[0]*mlp_ratio)
        self.ffn2 = MLP(config, df[1], df[1]*mlp_ratio)
        self.ffn3 = MLP(config, df[2], df[2]*mlp_ratio)
        self.ffn4 = MLP(config, df[3], df[3]*mlp_ratio)

    def forward(self, emb1, emb2, emb3, emb4):
        '''  Block 1 '''
        h1, h2, h3, h4 = emb1, emb2, emb3, emb4
        embcat = []
        for i in range(4):
            var_name = "emb" + str(i+1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)
        h_all = torch.cat(embcat, dim=2)

        # Layer norm
        emb1 = self.attn_norm1(emb1)
        emb2 = self.attn_norm2(emb2)
        emb3 = self.attn_norm3(emb3)
        emb4 = self.attn_norm4(emb4)
        h_all = self.attn_norm(h_all)

        # Multi head channel attention + Residual
        emb1, emb2, emb3, emb4, weights = self.multi_channel_attn(
            emb1, emb2, emb3, emb4, h_all)
        emb1 = emb1 + h1
        emb2 = emb2 + h2
        emb3 = emb3 + h3
        emb4 = emb4 + h4

        '''  Block 2 '''
        h1, h2, h3, h4 = emb1, emb2, emb3, emb4

        # Layer norm + MLP
        emb1 = self.ffn1(self.ffn_norm1(emb1))
        emb2 = self.ffn2(self.ffn_norm2(emb2))
        emb3 = self.ffn3(self.ffn_norm3(emb3))
        emb4 = self.ffn4(self.ffn_norm4(emb4))

        # Residual
        emb1 = emb1 + h1
        emb2 = emb2 + h2
        emb3 = emb3 + h3
        emb4 = emb4 + h4

        return emb1, emb2, emb3, emb4, weights


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.vis = config.vis
        df = config.df
        self.layers = nn.ModuleList()
        self.encoder_norm1 = nn.LayerNorm(df[0], eps=1e-6)
        self.encoder_norm2 = nn.LayerNorm(df[1], eps=1e-6)
        self.encoder_norm3 = nn.LayerNorm(df[2], eps=1e-6)
        self.encoder_norm4 = nn.LayerNorm(df[3], eps=1e-6)

        for _ in range(config.transformer.num_layers):
            layer = EncoderBlock(config)
            self.layers.append(copy.deepcopy(layer))

    def forward(self, emb1, emb2, emb3, emb4):
        a_weight_layers = []
        for layer_block in self.layers:
            emb1, emb2, emb3, emb4, weights = layer_block(
                emb1, emb2, emb3, emb4)
            if self.vis:
                a_weight_layers.append(weights)

        emb1 = self.encoder_norm1(emb1)
        emb2 = self.encoder_norm2(emb2)
        emb3 = self.encoder_norm3(emb3)
        emb4 = self.encoder_norm4(emb4)

        return emb1, emb2, emb3, emb4, a_weight_layers


class ChannelTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Extract config
        self.config = config
        img_size = config.img_size
        df = config.df
        p = config.p

        # Embedding layers
        self.embeddings_1 = ChannelEmbeddings(
            config, img_size=img_size,      patch_size=p[0], in_channels=df[0])
        self.embeddings_2 = ChannelEmbeddings(
            config, img_size=img_size // 2, patch_size=p[1], in_channels=df[1])
        self.embeddings_3 = ChannelEmbeddings(
            config, img_size=img_size // 4, patch_size=p[2], in_channels=df[2])
        self.embeddings_4 = ChannelEmbeddings(
            config, img_size=img_size // 8, patch_size=p[3], in_channels=df[3])

        # Encode embedding layers
        self.encoder = Encoder(config)

    def forward(self, x1, x2, x3, x4):
        emb1 = self.embeddings_1(x1)
        emb2 = self.embeddings_2(x2)
        emb3 = self.embeddings_3(x3)
        emb4 = self.embeddings_4(x4)

        enc1, enc2, enc3, enc4, a_weights = self.encoder(
            emb1, emb2, emb3, emb4)
        return emb1, emb2, emb3, emb4, enc1, enc2, enc3, enc4, a_weights
