"""Fig. 4: per-patient heatmaps for the eight models (two-seed means)."""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
J = json.load(open(sys.argv[1])); out = sys.argv[2]
order = ["GG-Net", "CNN-LSTM", "ShallowConvNet", "EEGNet", "EEG Transformer", "Static GCN", "GGN (re-impl.)", "RF (handcrafted)"]
lab = {"GG-Net": "GG-Net (proposed)", "GGN (re-impl.)": "GGN (re-impl.)", "RF (handcrafted)": "RF (spectral)", "Static GCN": "Static-graph GCN", "EEGNet": "EEGNet-8,2"}
PP = J["per_patient"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), dpi=300, gridspec_kw=dict(wspace=0.08))
specs = [("auroc", "(a) window AUROC", "YlOrBr_r", 0.45, 1.0, "%.2f"), ("ev_sen", "(b) event sensitivity (%)", "YlOrBr_r", 0, 100, "%.0f"),
         ("fp_h", "(c) false positives per hour", "YlOrBr", 0, 10, "%.1f")]
for i, (k, t, cm, lo, hi, fmt) in enumerate(specs):
    M = np.array([PP[m][k] for m in order]); ax = axs[i]
    ax.imshow(M, cmap=cm, vmin=lo, vmax=hi, aspect="auto")
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            v = M[r, c]; dark = (v - lo) / (hi - lo) < 0.35 if cm.endswith("_r") else (v - lo) / (hi - lo) > 0.65
            ax.text(c, r, fmt % v, ha="center", va="center", fontsize=4.6, color="white" if dark else "#222")
    ax.set_xticks(range(10)); ax.set_xticklabels([f"{j:02d}" for j in range(1, 11)], fontsize=6)
    ax.set_xlabel("held-out patient (chb)", fontsize=6.5); ax.set_title(t, fontsize=7.5, loc="left")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([lab.get(m, m) for m in order] if i == 0 else [], fontsize=6.5)
    ax.axhline(0.5, color="black", lw=0.8); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
fig.savefig(out, bbox_inches="tight", facecolor="white"); print("saved", out)
