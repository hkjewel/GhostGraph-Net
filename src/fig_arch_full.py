"""Full pictorial architecture of GhostGraph-Net (Fig. 1). Schematic: signals and
graphs are illustrative, not experimental results."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Ellipse
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.linewidth": 0.5})
out = sys.argv[1]
rng = np.random.default_rng(7)

INK = "#1F2328"; MUTED = "#5B6470"
COL = dict(inp="#4C566A", ghost="#2E7D6B", hc="#7A4E9C", graph="#B05A2A", dec="#9A3E1F",
           cnn="#3B5BA5", out="#A33B3B")
LOBE = dict(F="#C0392B", T="#2E5FA8", C="#B07D1A", P="#7A4E9C", O="#2E7D6B")

W, H = 7.2, 9.33
fig = plt.figure(figsize=(W, H), dpi=300)
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")


def panel(x, y, w, h, c, label, title):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=c + "0F", ec=c, lw=0.9, zorder=0))
    bg.text(x + 0.08, y + h - 0.07, label, fontsize=8, weight="bold", color=c, va="top")
    bg.text(x + 0.30, y + h - 0.075, title, fontsize=7.2, weight="bold", color=c, va="top")


def arrow(x0, y0, x1, y1, c=INK, lw=0.9, rad=0.0, ls="-"):
    bg.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=lw,
                                 color=c, connectionstyle=f"arc3,rad={rad}", linestyle=ls, zorder=5))


def ax_at(x, y, w, h, **kw):
    a = fig.add_axes([x / W, y / H, w / W, h / H], **kw)
    return a


def txt(x, y, s, **kw):
    kw.setdefault("fontsize", 6.2); kw.setdefault("color", INK); kw.setdefault("ha", "center"); kw.setdefault("va", "center")
    bg.text(x, y, s, **kw)


# ----------------------------------------------------------------- montage
POS = dict(FP1=(-.30, .92), FP2=(.30, .92), F7=(-.72, .55), F3=(-.36, .48), FZ=(0, .45), F4=(.36, .48), F8=(.72, .55),
           T7=(-.92, 0), C3=(-.46, 0), CZ=(0, 0), C4=(.46, 0), T8=(.92, 0), P7=(-.72, -.55), P3=(-.36, -.48),
           PZ=(0, -.45), P4=(.36, -.48), P8=(.72, -.55), O1=(-.30, -.92), O2=(.30, -.92))
CH = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4",
      "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ"]
LOBE_OF = ["F", "T", "T", "O", "F", "C", "P", "O", "F", "C", "P", "O", "F", "T", "T", "O", "C", "P"]
NODE = np.array([np.mean([POS[a], POS[b]], 0) for a, b in (c.split("-") for c in CH)])


def scalp(a, A=None, thr=0.0, ncol=None, lw_scale=2.2, show_elec=False, node_size=11, edge_col="#B05A2A", topk=None):
    a.set_xlim(-1.18, 1.18); a.set_ylim(-1.18, 1.22); a.set_aspect("equal"); a.axis("off")
    a.add_patch(Circle((0, 0), 1.08, fc="white", ec=MUTED, lw=0.6))
    a.add_patch(FancyArrowPatch((-0.12, 1.06), (0, 1.2), arrowstyle="-", lw=0.6, color=MUTED))
    a.add_patch(FancyArrowPatch((0.12, 1.06), (0, 1.2), arrowstyle="-", lw=0.6, color=MUTED))
    if show_elec:
        for k, (x, y) in POS.items():
            a.plot(x, y, "o", ms=2.2, mfc="white", mec=MUTED, mew=0.5)
        for c in CH:
            p, q = c.split("-"); a.plot(*zip(POS[p], POS[q]), color="#C9CED6", lw=0.5, zorder=0)
    if A is not None:
        iu = np.triu_indices(18, 1)
        if topk is not None:
            thr = np.sort(A[iu])[-topk - 1]
        order = np.argsort(A[iu])
        for k in order:
            i, j = iu[0][k], iu[1][k]
            if A[i, j] > thr:
                a.plot(*zip(NODE[i], NODE[j]), color=edge_col, lw=lw_scale * (A[i, j] - thr) / (1 - thr) + 0.1,
                       alpha=0.25 + 0.75 * (A[i, j] - thr) / (1 - thr), solid_capstyle="round", zorder=1)
    for i, (x, y) in enumerate(NODE):
        a.scatter(x, y, s=node_size, c=(ncol[i] if ncol is not None else LOBE[LOBE_OF[i]]), edgecolors="white",
                  linewidths=0.4, zorder=3)


def make_adj(seed, focus, strength=0.9, spread=0.35):
    r = np.random.default_rng(seed)
    d = np.linalg.norm(NODE[:, None] - NODE[None], axis=-1)
    base = np.exp(-d ** 2 / (2 * spread ** 2)) * 0.55
    f = np.zeros(18); f[focus] = 1
    A = base + strength * np.outer(f, f) * 0.55 + r.uniform(0, 0.12, (18, 18))
    A = np.clip((A + A.T) / 2, 0, 1); np.fill_diagonal(A, 1)
    return A


# ===================================================================== BAND A
yA, hA = 6.55, 2.72
panel(0.08, yA, 2.30, hA, COL["inp"], "(a)", "Scalp EEG input window")
panel(2.48, yA, 2.62, hA, COL["ghost"], "(b)", "Ghost temporal encoder (shared)")
panel(5.20, yA, 1.92, hA, COL["hc"], "(c)", "Spectral node descriptors")

# (a) EEG traces
a = ax_at(0.42, yA + 0.95, 1.00, 1.50)
t = np.linspace(0, 12, 1536)
on = 6.0
for k in range(18):
    s = 0.22 * np.cumsum(rng.normal(0, 1, t.size)) / np.sqrt(t.size) * 6 + 0.25 * rng.normal(0, 1, t.size) * 0.4
    s = s - s.mean()
    if k in (0, 1, 4, 5, 12, 13):
        env = np.clip((t - on) / 0.8, 0, 1)
        s = s + env * 0.75 * np.sin(2 * np.pi * 3.2 * t + k) * (1 + 0.3 * np.sin(2 * np.pi * 0.3 * t))
    a.plot(t, s * 0.35 - k, lw=0.35, color=LOBE[LOBE_OF[k]])
a.axvspan(5.0, 9.0, color="#F2C14E", alpha=0.25, lw=0)
a.axvline(on, color=COL["out"], lw=0.6, ls="--")
a.text(on + 0.1, 0.9, "onset", fontsize=5, color=COL["out"], va="bottom")
a.text(7.0, -18.2, "4-s window", fontsize=5.2, ha="center", va="top", color="#8A6A00")
a.set_ylim(-18.6, 1.5); a.set_xlim(0, 12); a.axis("off")
for k in range(0, 18, 1):
    a.text(-0.2, -k, CH[k], fontsize=3.3, ha="right", va="center", color=MUTED)
# montage
a2 = ax_at(1.45, yA + 1.42, 0.88, 0.88)
scalp(a2, show_elec=True, node_size=7)
txt(1.89, yA + 1.30, "18 bipolar derivations", fontsize=5.0, color=MUTED)
# legend lobes
for i, (k, nm) in enumerate([("F", "frontal"), ("T", "temporal"), ("C", "central"), ("P", "parietal"), ("O", "occipital")]):
    bg.scatter(1.70, yA + 1.14 - i * 0.10, s=7, c=LOBE[k])
    txt(1.75, yA + 1.14 - i * 0.10, nm, fontsize=4.6, ha="left", color=MUTED)
txt(1.23, yA + 0.32, "X ∈ ℝ$^{18×512}$  (4 s × 128 Hz)\nfold-wise median/IQR scaling", fontsize=5.6)

# (b) Ghost module internals
gx, gy = 2.62, yA + 1.15
a = ax_at(gx, gy + 0.55, 0.42, 0.62)
for k in range(3):
    a.plot(t[:512], np.sin(2 * np.pi * 0.7 * t[:512] * (k + 1)) * 0.3 + 0.2 * rng.normal(0, 1, 512) - k, lw=0.35, color=COL["inp"])
a.axis("off")
txt(gx + 0.21, gy + 0.45, "x$_i$ ∈ ℝ$^{512}$\n(each channel)", fontsize=5)
arrow(gx + 0.46, gy + 0.86, gx + 0.62, gy + 0.86, c=COL["ghost"])


def fmap(x, y, n, c, alpha=1.0, hatch=None, w=0.28, h=0.075):
    for k in range(n):
        bg.add_patch(Rectangle((x + k * 0.025, y - k * 0.06), w, h, fc=c, ec="white", lw=0.4, alpha=alpha, hatch=hatch, zorder=3))


txt(gx + 0.95, gy + 1.28, "Ghost module", fontsize=6, weight="bold", color=COL["ghost"])
bg.add_patch(FancyBboxPatch((gx + 0.62, gy + 0.40), 1.10, 0.80, boxstyle="round,pad=0.01,rounding_size=0.05", fc="white", ec=COL["ghost"], lw=0.6, ls="--"))
txt(gx + 0.86, gy + 1.08, "primary conv\nk×1, n/2 maps", fontsize=4.2, color=COL["ghost"])
fmap(gx + 0.70, gy + 0.90, 4, COL["ghost"])
txt(gx + 1.46, gy + 1.08, "cheap depthwise\nΦ, 3×1", fontsize=4.2, color="#4E937A")
fmap(gx + 1.32, gy + 0.90, 4, "#8CC7AE")
arrow(gx + 1.02, gy + 0.78, gx + 1.30, gy + 0.78, c="#6AA88F", lw=0.6)
txt(gx + 1.17, gy + 0.53, "Y = [ Y′ ‖ Φ(Y′) ]\nn maps, ≈ ½ the weights", fontsize=4.4)
arrow(gx + 1.74, gy + 0.86, gx + 1.92, gy + 0.86, c=COL["ghost"])
# stack of 4 ghost blocks
for k, (nm, sz) in enumerate([("G16 k15 s4", 0.62), ("G24 k9 s2", 0.52), ("G32 k7", 0.42), ("G24 k5", 0.34)]):
    xx = gx + 1.95 + k * 0.105
    bg.add_patch(Rectangle((xx, gy + 0.86 - sz / 2), 0.08, sz, fc=COL["ghost"], alpha=0.35 + 0.15 * k, ec=COL["ghost"], lw=0.4))
txt(gx + 2.15, gy + 0.42, "×4 Ghost blocks", fontsize=4.6, color=COL["ghost"])
# node feature matrix
a = ax_at(gx + 0.55, yA + 0.12, 0.95, 0.80)
Hm = rng.normal(0, 1, (18, 48)); Hm[[0, 1, 4, 5, 12, 13]] += 1.2
a.imshow(Hm, cmap="BuGn", aspect="auto", interpolation="nearest"); a.set_xticks([]); a.set_yticks([])
for s in a.spines.values(): s.set_color(COL["ghost"]); s.set_linewidth(0.6)
txt(gx + 2.10, yA + 0.66, "mean ‖ max pool\nover time", fontsize=4.6, color=MUTED)
txt(gx + 2.10, yA + 0.34, "E ∈ ℝ$^{18×48}$\none row e$_i$\nper electrode", fontsize=4.8)
arrow(gx + 2.30, gy + 0.28, gx + 1.52, yA + 0.85, c=COL["ghost"], rad=-0.25)

# (c) spectral descriptors
a = ax_at(5.38, yA + 1.35, 1.58, 0.92)
bands = ["δ", "θ", "α", "β", "γ", "LL", "Mo", "Co"]
v1 = np.array([1.0, 0.7, 0.45, 0.35, 0.2, 0.55, 0.5, 0.6]); v2 = np.array([1.35, 1.2, 0.5, 0.4, 0.28, 0.95, 0.35, 0.45])
xb = np.arange(8)
a.bar(xb - 0.2, v1, 0.38, color="#C9B6DB", label="interictal")
a.bar(xb + 0.2, v2, 0.38, color=COL["hc"], label="ictal")
a.set_xticks(xb); a.set_xticklabels(bands, fontsize=4.8); a.set_yticks([])
for s in ["top", "right", "left"]: a.spines[s].set_visible(False)
a.legend(fontsize=4.4, frameon=False, loc="upper right", handlelength=0.8)
txt(6.16, yA + 1.08, "log band power (δ θ α β γ), line length,\nHjorth mobility & complexity", fontsize=4.9, color=MUTED)
txt(6.16, yA + 0.80, "φ$_i$ ∈ ℝ$^8$  (z-scored on training fold)", fontsize=5.4)
txt(6.16, yA + 0.50, "h$_i$ = e$_i$ + W$_φ$ φ$_i$", fontsize=7.2, weight="bold", color=COL["hc"])
txt(6.16, yA + 0.22, "node features H ∈ ℝ$^{18×48}$", fontsize=5.4)
arrow(5.08, yA + 0.50, 5.62, yA + 0.50, c=COL["hc"])
txt(5.35, yA + 0.60, "+", fontsize=10, color=COL["hc"], weight="bold")

# flow from (a) to (b) and (c)
arrow(2.36, yA + 1.85, 2.62, gy + 0.86, c=INK)
bg.plot([2.30, 2.30, 5.30], [yA + 2.62, yA + 2.62 + 0.0, yA + 2.62], color=MUTED, lw=0)

# ===================================================================== BAND B
yB, hB = 3.42, 3.03
panel(0.08, yB, 7.04, hB, COL["graph"], "(d)", "Connectivity graph generator (GGN-style, per window)")

# pair features
px = 0.25
a = ax_at(px, yB + 1.30, 0.95, 1.05)
scalp(a, node_size=8)
i0, j0 = 1, 13
a.plot(*zip(NODE[i0], NODE[j0]), color=COL["graph"], lw=1.0, ls="--")
a.scatter(*NODE[i0], s=22, c=LOBE["T"], edgecolors=INK, linewidths=0.6, zorder=4)
a.scatter(*NODE[j0], s=22, c=LOBE["T"], edgecolors=INK, linewidths=0.6, zorder=4)
a.text(NODE[i0][0] - 0.05, NODE[i0][1] + 0.16, "i", fontsize=6, weight="bold")
a.text(NODE[j0][0] + 0.02, NODE[j0][1] + 0.16, "j", fontsize=6, weight="bold")
txt(0.72, yB + 1.12, "for every pair (i, j):", fontsize=5.4)
txt(0.72, yB + 0.88, "[ h$_i$⊙h$_j$ , |h$_i$ − h$_j$| , |ρ$_{ij}$| ]", fontsize=5.8, weight="bold", color=COL["graph"])
txt(0.72, yB + 0.62, "ρ$_{ij}$: Pearson correlation\nof the raw window (prior)", fontsize=4.8, color=MUTED)
arrow(1.20, yB + 1.80, 1.42, yB + 1.80, c=COL["graph"])

# Para-learner MLP
mx = 1.45
bg.add_patch(FancyBboxPatch((mx, yB + 1.05), 1.05, 1.45, boxstyle="round,pad=0.01,rounding_size=0.05", fc="white", ec=COL["graph"], lw=0.6))
txt(mx + 0.52, yB + 2.36, "Para-Learner (MLP)", fontsize=5.6, weight="bold", color=COL["graph"])
layers = [5, 4, 6]
for li, n in enumerate(layers):
    for k in range(n):
        yy = yB + 1.75 + (k - (n - 1) / 2) * 0.13
        xx = mx + 0.18 + li * 0.35
        if li < len(layers) - 1:
            for k2 in range(layers[li + 1]):
                y2 = yB + 1.75 + (k2 - (layers[li + 1] - 1) / 2) * 0.13
                bg.plot([xx, xx + 0.35], [yy, y2], color="#E3C3AE", lw=0.25, zorder=2)
        bg.add_patch(Circle((xx, yy), 0.035, fc=COL["graph"] if li == 2 else "white", ec=COL["graph"], lw=0.5, zorder=3))
txt(mx + 0.52, yB + 1.20, "→ (π$_k$, μ$_k$, σ$_k$)$_{k=1..3}$\nfor each head h = 1..3", fontsize=4.9)
arrow(mx + 1.07, yB + 1.80, mx + 1.27, yB + 1.80, c=COL["graph"])

# mixture gaussian
gx2 = 2.75
a = ax_at(gx2, yB + 1.32, 1.20, 0.98)
z = np.linspace(-5, 5, 400)
comps = [(0.25, -2.4, 0.7, "#E8B795"), (0.45, 0.6, 0.9, "#D17C45"), (0.30, 2.8, 0.6, "#8E3E12")]
tot = 0
for w_, m_, s_, c_ in comps:
    pdf = w_ * np.exp(-(z - m_) ** 2 / (2 * s_ ** 2)) / (s_ * np.sqrt(2 * np.pi))
    a.fill_between(z, pdf, color=c_, alpha=0.45, lw=0); tot = tot + pdf
a.plot(z, tot, color=INK, lw=0.7)
a.set_yticks([]); a.set_xticks([-4, 0, 4]); a.tick_params(labelsize=4.5, length=1.5)
for s in ["top", "right", "left"]: a.spines[s].set_visible(False)
a.set_xlabel("edge logit z$_{ij}$", fontsize=4.8, labelpad=1)
txt(gx2 + 0.60, yB + 2.43, "Mix-Gaussian (K = 3)", fontsize=5.6, weight="bold", color=COL["graph"])
txt(gx2 + 0.60, yB + 1.02, "p(z$_{ij}$) = Σ$_k$ π$_k$ N(μ$_k$, σ$_k^2$)", fontsize=5.4)
arrow(gx2 + 1.25, yB + 1.80, gx2 + 1.42, yB + 1.80, c=COL["graph"])

# sampler
sx = 4.20
bg.add_patch(FancyBboxPatch((sx, yB + 1.05), 1.02, 1.45, boxstyle="round,pad=0.01,rounding_size=0.05", fc="white", ec=COL["graph"], lw=0.6))
txt(sx + 0.51, yB + 2.36, "Gumbel sampler", fontsize=5.6, weight="bold", color=COL["graph"])
txt(sx + 0.51, yB + 2.08, "Training:", fontsize=5, weight="bold", ha="center")
txt(sx + 0.51, yB + 1.80, "c ~ Gumbel-softmax(π)\nz = Σc$_k$(μ$_k$+σ$_k$ε)\nA$_{ij}$ = σ((z + log u/(1−u))/τ)", fontsize=4.6)
txt(sx + 0.51, yB + 1.43, "Inference:", fontsize=5, weight="bold")
txt(sx + 0.51, yB + 1.22, "A$_{ij}$ = E[A$_{ij}$] (closed form)\n→ deterministic, no resampling", fontsize=4.6)
arrow(sx + 1.04, yB + 1.80, sx + 1.22, yB + 1.80, c=COL["graph"])

# three adjacency heads
hx = 5.45
heads = [make_adj(1, [1, 2, 13, 14], 0.9, 0.30), make_adj(2, [0, 1, 4, 5, 12], 1.0, 0.55), make_adj(3, [5, 6, 9, 10, 16, 17], 0.8, 0.9)]
for k, A in enumerate(heads):
    a = ax_at(hx + k * 0.55, yB + 1.83, 0.50, 0.50)
    a.imshow(A, cmap="Oranges", vmin=0, vmax=1, interpolation="nearest"); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_linewidth(0.4)
    a.set_title(f"A$^{{({k+1})}}$", fontsize=5.4, pad=1.5, color=COL["graph"])
    a2 = ax_at(hx + k * 0.55 - 0.02, yB + 1.20, 0.54, 0.58)
    scalp(a2, A, thr=0.55, node_size=4, lw_scale=1.4)
txt(hx + 0.80, yB + 1.07, "H = 3 graphs per window, A ∈ [0,1]$^{18×18}$\n(symmetrised, self-loops, row-normalised Â)", fontsize=4.8)

# dynamic connectivity strip
txt(3.60, yB + 0.80, "Generated connectivity changes with the brain state (illustrative)", fontsize=5.6, weight="bold", color=COL["graph"])
states = [("interictal", make_adj(11, [], 0.0, 0.25), 0.62), ("pre-onset", make_adj(12, [1, 13], 0.7, 0.32), 0.60),
          ("onset", make_adj(13, [0, 1, 4, 5, 12, 13], 1.2, 0.45), 0.58), ("ictal", make_adj(14, list(range(0, 18, 1)), 0.6, 0.65), 0.58),
          ("post-ictal", make_adj(15, [6, 7, 10, 11], 0.5, 0.3), 0.62)]
for k, (nm, A, th) in enumerate(states):
    a = ax_at(1.05 + k * 1.10, yB + 0.10, 0.55, 0.58)
    scalp(a, A, node_size=4, lw_scale=1.6, topk=[6, 8, 16, 30, 7][k])
    txt(1.33 + k * 1.10, yB + 0.06, nm, fontsize=4.8, color=MUTED)
    if k < 4:
        arrow(1.62 + k * 1.10, yB + 0.40, 2.12 + k * 1.10, yB + 0.40, c=MUTED, lw=0.5)

# ===================================================================== BAND C
yC, hC = 0.08, 3.22
panel(0.08, yC, 2.62, hC, COL["dec"], "(e)", "Attentive graph decoder (×2 layers)")
panel(2.80, yC, 1.72, hC, COL["cnn"], "(f)", "Ghost-CNN branch")
panel(4.62, yC, 2.50, hC, COL["out"], "(g)", "Fusion, classifier & post-processing")

# (e) message passing on each head + attention
for k, A in enumerate(heads):
    a = ax_at(0.18 + k * 0.80, yC + 1.95, 0.78, 0.82)
    focus = 13
    w_ = A[focus] / A[focus].sum()
    scalp(a, None, node_size=6)
    for j in np.argsort(w_)[-6:]:
        if j == focus: continue
        a.add_patch(FancyArrowPatch(NODE[j], NODE[focus], arrowstyle="-|>", mutation_scale=4, lw=0.4 + 6 * w_[j],
                                    color=COL["dec"], alpha=0.85, zorder=2))
    a.scatter(*NODE[focus], s=20, c="#F2C14E", edgecolors=INK, linewidths=0.5, zorder=5)
    txt(0.57 + k * 0.80, yC + 1.90, f"M$^{{({k+1})}}$ = Â$^{{({k+1})}}$ H W", fontsize=4.8)
# attention bars
a = ax_at(0.40, yC + 1.02, 0.85, 0.55)
al = np.array([0.22, 0.53, 0.25])
a.bar([1, 2, 3], al, color=[COL["dec"] + "88", COL["dec"], COL["dec"] + "88"], width=0.6)
a.set_xticks([1, 2, 3]); a.set_xticklabels(["α$_1$", "α$_2$", "α$_3$"], fontsize=5)
a.set_yticks([]); a.set_ylim(0, 0.65)
for s in ["top", "right", "left"]: a.spines[s].set_visible(False)
txt(0.83, yC + 1.65, "head attention α$_h$", fontsize=4.8, color=MUTED)
txt(1.95, yC + 1.42, "α$_h$ = softmax$_h$(v$^⊤$tanh(U·mean$_i$M$^{(h)}$))\nH ← LN(H + ELU(Σ$_h$ α$_h$ M$^{(h)}$))", fontsize=4.7)
txt(1.40, yC + 0.52, "graph read-out  r = [ mean$_i$ H$_i$ ‖ max$_i$ H$_i$ ] ∈ ℝ$^{96}$", fontsize=5.3, weight="bold", color=COL["dec"])
for k in range(12):
    bg.add_patch(Rectangle((0.70 + k * 0.105, yC + 0.20), 0.095, 0.16, fc=plt.cm.Oranges(0.25 + 0.06 * k), ec="white", lw=0.3))

# (f) Ghost-CNN branch
cx = 2.95
a = ax_at(cx, yC + 2.02, 0.55, 0.75)
for k in range(18):
    a.plot(np.sin(np.linspace(0, 8 + k, 200)) * 0.3 + 0.1 * rng.normal(0, 1, 200) - k, lw=0.3, color=LOBE[LOBE_OF[k]])
a.axis("off")
txt(cx + 0.28, yC + 1.93, "X (from a)", fontsize=4.8)
arrow(cx + 0.58, yC + 2.40, cx + 0.74, yC + 2.40, c=COL["cnn"])
bg.add_patch(Rectangle((cx + 0.76, yC + 2.05), 0.16, 0.70, fc=COL["cnn"], alpha=0.5, ec=COL["cnn"], lw=0.4))
txt(cx + 0.84, yC + 1.93, "1×1 conv\n18→16", fontsize=4.5)
arrow(cx + 0.94, yC + 2.40, cx + 1.06, yC + 2.40, c=COL["cnn"])
for k, sz in enumerate([0.66, 0.50]):
    bg.add_patch(Rectangle((cx + 1.08 + k * 0.17, yC + 2.40 - sz / 2), 0.13, sz, fc=COL["cnn"], alpha=0.35 + 0.25 * k, ec=COL["cnn"], lw=0.4))
txt(cx + 1.25, yC + 1.93, "Ghost\nk15, k7", fontsize=4.5)
txt(cx + 0.72, yC + 1.40, "global average pool", fontsize=4.8, color=MUTED)
arrow(cx + 1.25, yC + 1.80, cx + 0.80, yC + 1.10, c=COL["cnn"], rad=-0.2)
for k in range(6):
    bg.add_patch(Rectangle((cx + 0.30 + k * 0.105, yC + 0.82), 0.095, 0.16, fc=plt.cm.Blues(0.3 + 0.1 * k), ec="white", lw=0.3))
txt(cx + 0.72, yC + 0.60, "g ∈ ℝ$^{32}$  (global, graph-free view)", fontsize=5.0, weight="bold", color=COL["cnn"])

# (g) fusion + classifier + output
fx = 4.78
for k in range(10):
    bg.add_patch(Rectangle((fx + 0.02, yC + 2.70 - k * 0.075), 0.16, 0.07, fc=plt.cm.Oranges(0.3 + 0.05 * k) if k < 7 else plt.cm.Blues(0.4 + 0.1 * (k - 7)), ec="white", lw=0.3))
txt(fx + 0.10, yC + 1.82, "[ r ‖ g ]\nℝ$^{128}$", fontsize=5)
layers = [6, 4, 2]
for li, n in enumerate(layers):
    for k in range(n):
        yy = yC + 2.35 + (k - (n - 1) / 2) * 0.14
        xx = fx + 0.45 + li * 0.38
        if li < 2:
            for k2 in range(layers[li + 1]):
                y2 = yC + 2.35 + (k2 - (layers[li + 1] - 1) / 2) * 0.14
                bg.plot([xx, xx + 0.38], [yy, y2], color="#E2B9B9", lw=0.25, zorder=2)
        bg.add_patch(Circle((xx, yy), 0.04, fc=COL["out"] if li == 2 else "white", ec=COL["out"], lw=0.5, zorder=3))
txt(fx + 0.83, yC + 1.82, "Dropout → FC 32 (ELU)\n→ FC 2 → softmax", fontsize=4.8)
txt(fx + 1.50, yC + 2.52, "P(seizure)", fontsize=5, ha="left")
txt(fx + 1.50, yC + 2.18, "P(non-seizure)", fontsize=5, ha="left")
txt(fx + 1.22, yC + 2.98, "weighted cross-entropy (w$_1$ = N$_0$/N$_1$)", fontsize=4.8, color=MUTED)
# output trace
a = ax_at(5.10, yC + 0.52, 1.90, 1.00)
tt = np.arange(0, 600, 2.0)
pr = 0.08 + 0.05 * rng.random(tt.size)
pr += 0.85 * np.exp(-((tt - 260) / 40) ** 2 / 2) * (tt > 222)
pr[(tt > 121) & (tt < 124)] += 0.45
sm = np.convolve(pr, np.ones(3) / 3, mode="same")
a.plot(tt, pr, color="#D9A3A3", lw=0.4, label="window prob.")
a.plot(tt, sm, color=COL["out"], lw=0.8, label="6-s smoothed")
a.axhline(0.5, color=INK, lw=0.5, ls="--")
a.axvspan(230, 292, color="#F2C14E", alpha=0.35, lw=0, label="annotated seizure")
det = sm >= 0.5
idx = np.where(det)[0]
if idx.size:
    a.axvspan(tt[idx[0]], tt[idx[-1]] + 4, ymin=0.92, ymax=1.0, color=COL["out"], lw=0, label="detected event")
a.set_ylim(0, 1.08); a.set_xlim(0, 600)
a.set_xlabel("time (s)", fontsize=4.8, labelpad=1); a.set_ylabel("P(seizure)", fontsize=4.8, labelpad=1)
a.tick_params(labelsize=4.3, length=1.5)
for s in ["top", "right"]: a.spines[s].set_visible(False)
a.text(8, 0.53, "θ* (validation patient)", fontsize=4.2, color=INK)
a.legend(fontsize=3.9, frameon=False, loc="center right", ncol=1, bbox_to_anchor=(1.02, 0.78))
txt(5.95, yC + 0.15, "events: merge < 90 s, SzCORE 30 s / 60 s tolerance → Sen, FP/h, latency", fontsize=4.5, color=MUTED)
arrow(fx + 1.22, yC + 1.70, fx + 1.22, yC + 1.55, c=COL["out"])

# ------------------------------------------------------------- inter-panel flows
bg.plot([6.16, 6.16], [yA + 0.12, yA - 0.05], color=COL["hc"], lw=0.9)
bg.plot([6.16, 0.12], [yA - 0.05, yA - 0.05], color=COL["hc"], lw=0.9)
bg.plot([0.12, 0.12], [yA - 0.05, yB + 1.80], color=COL["hc"], lw=0.9)
arrow(0.12, yB + 1.80, 0.27, yB + 1.80, c=COL["hc"], lw=0.9)
txt(3.4, yA - 0.05, " node features H ", fontsize=5.2, weight="bold", color=COL["hc"], bbox=dict(fc="white", ec="none", pad=0.6))
# (d) -> (e): graphs + H
bg.plot([6.30, 6.30], [yB + 1.00, yB - 0.06], color=COL["graph"], lw=1.0)
bg.plot([6.30, 2.55], [yB - 0.06, yB - 0.06], color=COL["graph"], lw=1.0)
arrow(2.55, yB - 0.06, 2.55, yC + hC - 0.36, c=COL["graph"], lw=1.0)
txt(3.9, yB - 0.06, " graphs Â$^{(1..3)}$ + node features H ", fontsize=5.2, weight="bold", color=COL["graph"], bbox=dict(fc="white", ec="none", pad=0.6))
# (a) -> (f): raw window

# (e) r and (f) g -> (g)
bg.plot([2.25, 4.70], [yC + 0.28, yC + 0.28], color=COL["dec"], lw=1.0)
arrow(4.70, yC + 0.28, 4.70, yC + 2.40, c=COL["dec"], lw=1.0)
bg.plot([3.72, 4.76], [yC + 0.90, yC + 0.90], color=COL["cnn"], lw=1.0)
arrow(4.76, yC + 0.90, 4.76, yC + 2.10, c=COL["cnn"], lw=1.0)
txt(4.40, yC + 0.36, "r", fontsize=6, weight="bold", color=COL["dec"])
txt(4.40, yC + 0.98, "g", fontsize=6, weight="bold", color=COL["cnn"])

plt.savefig(out, facecolor="white")
print("saved", out)
