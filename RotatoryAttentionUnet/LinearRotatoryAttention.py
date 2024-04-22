import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class LinearRotatoryAttentionModule(nn.Module):
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
        self.left_attention_score_bias = nn.Parameter(torch.randn(1))
        self.left_attention_score_weight = nn.Parameter(
            torch.randn(self.dkl, self.dft), requires_grad=True)

        self.right_attention_score_bias = nn.Parameter(torch.randn(1))
        self.right_attention_score_weight = nn.Parameter(
            torch.randn(self.dkr, dft), requires_grad=True)

        self.left_t_attention_score_bias = nn.Parameter(torch.randn(1))
        self.left_t_attention_score_weight = nn.Parameter(
            torch.randn(self.dkt, self.dvl), requires_grad=True)

        self.right_t_attention_score_bias = nn.Parameter(torch.randn(1))
        self.right_t_attention_score_weight = nn.Parameter(
            torch.randn(self.dkt, self.dvr), requires_grad=True)

        self.final_t_weights = nn.Linear(
            1, self.dft, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.left_attention_score_weight)

        nn.init.xavier_uniform_(self.right_attention_score_weight)

        nn.init.xavier_uniform_(self.left_t_attention_score_weight)

        nn.init.xavier_uniform_(self.right_t_attention_score_weight)

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


def standalone_module_testing():
    # 32 x 512
    dft = 512
    nft = 32
    dfl = 512
    nfl = 32
    dfr = 512
    nfr = 32
    dkl = 48
    dvl = 8
    dkr = 48
    dvr = 8
    dkt = 48
    dvt = 8

    # dvr, dvt, dvl = nft / 4
    attention_module = LinearRotatoryAttentionModule(
        nft=nft, dft=dft, nfl=nfl, dfl=dfl, nfr=nfr, dfr=dfr, dkl=dkl, dvl=dvl, dkr=dkr, dvr=dvr, dkt=dkt, dvt=dvt)

    Ft = torch.randn(nft, dft)
    Fl = torch.randn(nfl, dfl)
    Fr = torch.randn(nfr, dfr)

    output = attention_module(Ft, Fl, Fr)

    print(f"Model output after forward: {output.shape}")


if __name__ == "__main__":
    standalone_module_testing()
