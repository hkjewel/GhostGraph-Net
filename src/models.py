"""Models for the GhostGraph seizure-detection study.

Proposed: GhostGraph-Net (GG-Net) -- a lightweight Ghost-convolution temporal
encoder, a GGN-style probabilistic connectivity-graph generator (mixture-Gaussian
edge logits + relaxed-Bernoulli/Gumbel sampling), a multi-head attentive graph
decoder, a parallel Ghost-CNN spatial branch, and interpretable spectral node
descriptors.  Baselines: EEGNet, ShallowConvNet, CNN-LSTM, static-graph GCN,
EEG Transformer, and a GGN re-implementation (standard-conv encoder).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

C_DEFAULT = 18


# ------------------------------------------------------------------ building blocks
class Ghost1d(nn.Module):
    """Ghost module (Han et al., CVPR 2020) for 1-D signals: a primary conv makes
    out/ratio intrinsic maps, a cheap depthwise conv makes the remaining 'ghost'
    maps."""

    def __init__(self, cin, cout, k, stride=1, ratio=2, dw_k=3):
        super().__init__()
        init = math.ceil(cout / ratio)
        self.cout = cout
        self.primary = nn.Sequential(nn.Conv1d(cin, init, k, stride, k // 2, bias=False),
                                     nn.BatchNorm1d(init), nn.ELU())
        self.cheap = nn.Sequential(nn.Conv1d(init, init * (ratio - 1), dw_k, 1, dw_k // 2,
                                             groups=init, bias=False),
                                   nn.BatchNorm1d(init * (ratio - 1)), nn.ELU())

    def forward(self, x):
        a = self.primary(x)
        return torch.cat([a, self.cheap(a)], 1)[:, : self.cout]


class Std1d(nn.Module):
    """Standard conv drop-in for Ghost1d (ablation)."""

    def __init__(self, cin, cout, k, stride=1, **_):
        super().__init__()
        self.f = nn.Sequential(nn.Conv1d(cin, cout, k, stride, k // 2, bias=False),
                               nn.BatchNorm1d(cout), nn.ELU())

    def forward(self, x):
        return self.f(x)


class TemporalEncoder(nn.Module):
    """Channel-shared temporal encoder: (B,C,T) -> node features (B,C,d)."""

    def __init__(self, d=48, ghost=True):
        super().__init__()
        B = Ghost1d if ghost else Std1d
        self.net = nn.Sequential(B(1, 16, 15, 4), B(16, 24, 9, 2), nn.MaxPool1d(4),
                                 B(24, 32, 7, 1), nn.MaxPool1d(2), B(32, d // 2, 5, 1))
        self.d = d

    def forward(self, x):
        b, c, t = x.shape
        h = self.net(x.reshape(b * c, 1, t))  # (BC, d/2, t')
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        return h.reshape(b, c, self.d)


def self_loops(A):
    """Set the diagonal to 1 so that a flat (zero-variance) channel, whose
    correlation row is all zero, still has a non-zero degree."""
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return A * (1 - eye) + eye


def pearson_adj(x):
    """|Pearson correlation| between channels of the raw window: (B,C,C)."""
    x = x - x.mean(-1, keepdim=True)
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
    return (x @ x.transpose(1, 2)).abs()


class GraphGenerator(nn.Module):
    """GGN-style connectivity-graph generator.

    For every electrode pair a Para-Learner MLP outputs the parameters of a
    K-component Gaussian mixture over the edge logit; a component is chosen with
    Gumbel-softmax, the logit is drawn by re-parameterisation, and the edge is
    sampled with a relaxed Bernoulli (binary-concrete) gate.  H heads give H
    graphs with different learned neighbourhood ranges.  At inference the
    expected edge probability is used (deterministic)."""

    def __init__(self, d, heads=3, K=3, tau=0.5, use_prior=True):
        super().__init__()
        self.H, self.K, self.tau, self.use_prior = heads, K, tau, use_prior
        ein = 2 * d + (1 if use_prior else 0)
        self.para = nn.Sequential(nn.Linear(ein, 32), nn.ELU(), nn.Linear(32, heads * K * 3))

    def forward(self, h, prior=None):
        b, c, d = h.shape
        hi = h.unsqueeze(2).expand(b, c, c, d)
        hj = h.unsqueeze(1).expand(b, c, c, d)
        e = [hi * hj, (hi - hj).abs()]
        if self.use_prior:
            e.append(prior.unsqueeze(-1))
        p = self.para(torch.cat(e, -1)).view(b, c, c, self.H, self.K, 3)
        pi_logit, mu, logsig = p[..., 0], p[..., 1], p[..., 2].clamp(-4, 2)
        if self.training:
            w = F.gumbel_softmax(pi_logit, tau=1.0, hard=False, dim=-1)
            z = (w * (mu + torch.randn_like(mu) * logsig.exp())).sum(-1)
            u = torch.rand_like(z).clamp(1e-6, 1 - 1e-6)
            A = torch.sigmoid((z + torch.log(u) - torch.log1p(-u)) / self.tau)
        else:
            w = pi_logit.softmax(-1)
            A = (w * torch.sigmoid(mu / torch.sqrt(1 + math.pi * (2 * logsig).exp() / 8))).sum(-1)
        A = A.permute(0, 3, 1, 2)  # (B,H,C,C)
        A = 0.5 * (A + A.transpose(-1, -2))
        eye = torch.eye(c, device=h.device)
        A = A * (1 - eye) + eye
        return A


class AttentiveGraphDecoder(nn.Module):
    """Graph conv over each generated graph + attention across graph heads."""

    def __init__(self, d, layers=2):
        super().__init__()
        self.W = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.att = nn.ModuleList([nn.Sequential(nn.Linear(d, 16), nn.Tanh(), nn.Linear(16, 1))
                                  for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.last_alpha = None

    def forward(self, h, A):  # h (B,C,d); A (B,H,C,C)
        deg = A.sum(-1, keepdim=True)
        An = A / deg
        for W, att, ln in zip(self.W, self.att, self.norm):
            m = torch.einsum("bhij,bjd->bhid", An, W(h))  # (B,H,C,d)
            s = att(m.mean(2))  # (B,H,1)
            alpha = s.softmax(1).unsqueeze(-1)  # (B,H,1,1)
            self.last_alpha = alpha.detach()
            h = ln(h + F.elu((alpha * m).sum(1)))
        return h


class GhostSpatialBranch(nn.Module):
    """Parallel CNN branch: depthwise-separable spatial filter across electrodes
    then a Ghost temporal stack (global, non-graph view of the window)."""

    def __init__(self, C=C_DEFAULT, out=32, ghost=True):
        super().__init__()
        B = Ghost1d if ghost else Std1d
        self.spatial = nn.Sequential(nn.Conv1d(C, 16, 1, bias=False), nn.BatchNorm1d(16), nn.ELU())
        self.temporal = nn.Sequential(B(16, 24, 15, 2), nn.MaxPool1d(4), B(24, out, 7, 1),
                                      nn.AdaptiveAvgPool1d(1))

    def forward(self, x):
        return self.temporal(self.spatial(x)).squeeze(-1)


class GhostGraphNet(nn.Module):
    """Proposed GhostGraph-Net (GG-Net)."""

    def __init__(self, C=C_DEFAULT, nfeat=8, d=48, heads=3, K=3, ghost=True,
                 dynamic_graph=True, handcrafted=True, cnn_branch=True):
        super().__init__()
        self.handcrafted, self.cnn_branch, self.dynamic = handcrafted, cnn_branch, dynamic_graph
        self.enc = TemporalEncoder(d, ghost)
        self.feat_proj = nn.Linear(nfeat, d) if handcrafted else None
        self.gen = GraphGenerator(d, heads if dynamic_graph else 1, K) if dynamic_graph else None
        self.dec = AttentiveGraphDecoder(d)
        self.cnn = GhostSpatialBranch(C, 32, ghost) if cnn_branch else None
        fin = 2 * d + (32 if cnn_branch else 0)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(fin, 32), nn.ELU(), nn.Linear(32, 2))
        self.last_A = None

    def forward(self, x, f=None):
        h = self.enc(x)
        if self.handcrafted:
            h = h + self.feat_proj(f)
        prior = pearson_adj(x)
        A = self.gen(h, prior) if self.dynamic else self_loops(prior).unsqueeze(1)
        self.last_A = A.detach()
        h = self.dec(h, A)
        z = [h.mean(1), h.amax(1)]
        if self.cnn_branch:
            z.append(self.cnn(x))
        return self.head(torch.cat(z, 1))


class GGNReimpl(nn.Module):
    """Re-implementation of the GGN of Li et al. (Sci. Rep. 2022) adapted to binary
    detection: standard-conv temporal encoder, graph generator, attentive graph
    decoder, shallow CNN branch on temporal features; no Ghost modules and no
    handcrafted features."""

    def __init__(self, C=C_DEFAULT, d=64, heads=3, K=3):
        super().__init__()
        self.enc = TemporalEncoder(d, ghost=False)
        self.gen = GraphGenerator(d, heads, K, use_prior=True)
        self.dec = AttentiveGraphDecoder(d)
        self.cnn = nn.Sequential(nn.Conv1d(C, 32, 3, padding=1), nn.ELU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * d + 32, 32), nn.ELU(), nn.Linear(32, 2))

    def forward(self, x, f=None):
        h0 = self.enc(x)
        A = self.gen(h0, pearson_adj(x))
        h = self.dec(h0, A)
        c = self.cnn(h0).squeeze(-1)
        return self.head(torch.cat([h.mean(1), h.amax(1), c], 1))


# ------------------------------------------------------------------------ baselines
class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al., J. Neural Eng. 2018), kernel scaled to 128 Hz."""

    def __init__(self, C=C_DEFAULT, T=512, F1=8, D=2, F2=16, k=64, p=0.25):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, F1, (1, k), padding=(0, k // 2), bias=False),
                                nn.BatchNorm2d(F1),
                                nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False),
                                nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(p))
        self.b2 = nn.Sequential(nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                                nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
                                nn.AvgPool2d((1, 8)), nn.Dropout(p))
        with torch.no_grad():
            n = self.b2(self.b1(torch.zeros(1, 1, C, T))).numel()
        self.fc = nn.Linear(n, 2)

    def forward(self, x, f=None):
        return self.fc(self.b2(self.b1(x.unsqueeze(1))).flatten(1))


class ShallowConvNet(nn.Module):
    """ShallowFBCSPNet (Schirrmeister et al., HBM 2017) at 128 Hz."""

    def __init__(self, C=C_DEFAULT, T=512, nf=40):
        super().__init__()
        self.c1 = nn.Conv2d(1, nf, (1, 13))
        self.c2 = nn.Conv2d(nf, nf, (C, 1), bias=False)
        self.bn = nn.BatchNorm2d(nf)
        self.pool = nn.AvgPool2d((1, 37), (1, 7))
        self.drop = nn.Dropout(0.5)
        with torch.no_grad():
            n = self._feat(torch.zeros(1, C, T)).numel()
        self.fc = nn.Linear(n, 2)

    def _feat(self, x):
        z = self.bn(self.c2(self.c1(x.unsqueeze(1))))
        return torch.log(torch.clamp(self.pool(z * z), 1e-6)).flatten(1)

    def forward(self, x, f=None):
        return self.fc(self.drop(self._feat(x)))


class CNNLSTM(nn.Module):
    """1-D CNN front-end + bidirectional LSTM over time."""

    def __init__(self, C=C_DEFAULT):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(C, 32, 7, 2, 3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                                 nn.Conv1d(32, 64, 5, 1, 2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4))
        self.lstm = nn.LSTM(64, 48, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(96, 2))

    def forward(self, x, f=None):
        z, _ = self.lstm(self.cnn(x).transpose(1, 2))
        return self.fc(z.mean(1))


class StaticGCN(nn.Module):
    """GCN on a fixed correlation graph (static-topology GNN baseline)."""

    def __init__(self, C=C_DEFAULT, nfeat=8, d=48):
        super().__init__()
        self.enc = TemporalEncoder(d, ghost=False)
        self.W1, self.W2 = nn.Linear(d, d), nn.Linear(d, d)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * d, 2))

    def forward(self, x, f=None):
        A = self_loops(pearson_adj(x))
        An = A / A.sum(-1, keepdim=True)
        h = self.enc(x)
        h = F.elu(An @ self.W1(h))
        h = F.elu(An @ self.W2(h))
        return self.head(torch.cat([h.mean(1), h.amax(1)], 1))


class EEGTransformer(nn.Module):
    """Compact EEG Transformer: conv patch embedding (0.25 s patches) + 2 encoder
    layers + CLS token (in the spirit of EEG Conformer, Song et al. 2023)."""

    def __init__(self, C=C_DEFAULT, T=512, d=64, patch=32):
        super().__init__()
        self.embed = nn.Sequential(nn.Conv1d(C, d, patch, patch), nn.BatchNorm1d(d))
        n = T // patch
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, n + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 128, 0.2, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 2)
        self.fc = nn.Linear(d, 2)

    def forward(self, x, f=None):
        z = self.embed(x).transpose(1, 2)
        z = torch.cat([self.cls.expand(len(z), -1, -1), z], 1) + self.pos
        return self.fc(self.tf(z)[:, 0])


def build(name, C=C_DEFAULT):
    return {
        "GG-Net": lambda: GhostGraphNet(C),
        "GG-Net w/o graph generator": lambda: GhostGraphNet(C, dynamic_graph=False),
        "GG-Net w/o Ghost (std conv)": lambda: GhostGraphNet(C, ghost=False),
        "GG-Net w/o handcrafted": lambda: GhostGraphNet(C, handcrafted=False),
        "GG-Net w/o CNN branch": lambda: GhostGraphNet(C, cnn_branch=False),
        "GGN (re-impl.)": lambda: GGNReimpl(C),
        "EEGNet": lambda: EEGNet(C),
        "ShallowConvNet": lambda: ShallowConvNet(C),
        "CNN-LSTM": lambda: CNNLSTM(C),
        "Static GCN": lambda: StaticGCN(C),
        "EEG Transformer": lambda: EEGTransformer(C),
    }[name]()


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
