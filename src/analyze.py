"""Aggregate LOSO runs into manuscript tables, statistics and figures.

usage: python analyze.py --runs RESULTS/runs --out RESULTS/analysis
Writes: summary_tables.md, per_patient.csv, stats.json, fig2_roc.png,
fig3_graphs.png (if GG-Net graphs exist), fig4_per_patient.png
"""
import os, glob, json, argparse
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAIN = ["RF (handcrafted)", "EEGNet", "ShallowConvNet", "CNN-LSTM", "Static GCN", "EEG Transformer", "GGN (re-impl.)", "GG-Net"]
ABL = ["GG-Net", "GG-Net w/o graph generator", "GG-Net w/o Ghost (std conv)", "GG-Net w/o handcrafted", "GG-Net w/o CNN branch"]
LABEL = {"RF (handcrafted)": "RF (spectral descriptors)", "EEGNet": "EEGNet-8,2", "Static GCN": "Static-graph GCN",
         "GGN (re-impl.)": "GGN (re-implemented)", "GG-Net": "GG-Net (proposed)"}
COLORS = {"GG-Net": "#B05A2A", "GGN (re-impl.)": "#3B5BA5", "EEGNet": "#2E7D6B", "EEG Transformer": "#7A4E9C",
          "Static GCN": "#B07D1A", "CNN-LSTM": "#5B6470", "ShallowConvNet": "#9AA3AD", "RF (handcrafted)": "#C9CED6"}


def load(runs):
    R = {}
    for f in glob.glob(os.path.join(runs, "*.json")):
        d = json.load(open(f))
        R.setdefault(d["model"], {}).setdefault(d["test"], []).append(d)
    return R


def per_patient(R, m, key):
    """mean over seeds for each test patient."""
    out = {}
    for p, lst in R.get(m, {}).items():
        vals = [key(d) for d in lst]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if vals:
            out[p] = float(np.mean(vals))
    return out


METRICS = {
    "auroc": lambda d: d["seg"]["auroc"], "auprc": lambda d: d["seg"]["auprc"],
    "sen": lambda d: 100 * d["seg"]["sen"], "spe": lambda d: 100 * d["seg"]["spe"], "f1": lambda d: 100 * d["seg"]["f1"],
    "ev_sen": lambda d: 100 * d["event"]["ev_sen"], "ev_prec": lambda d: 100 * d["event"]["ev_prec"],
    "ev_f1": lambda d: 100 * d["event"]["ev_f1"], "fp_h": lambda d: d["event"]["fp_h"],
    "lat_med": lambda d: d["event"]["latency_median"],
}


def ms(v, fmt):
    v = np.array(list(v.values()))
    return (fmt % v.mean()) + " ± " + (fmt % v.std(ddof=1) if len(v) > 1 else "0")


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * pvals[i]); adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    R = load(a.runs)
    models = [m for m in MAIN if m in R]; abl = [m for m in ABL if m in R]
    patients = sorted({p for m in R for p in R[m]})
    PP = {m: {k: per_patient(R, m, f) for k, f in METRICS.items()} for m in R}
    seeds = {m: sorted({d["seed"] for lst in R[m].values() for d in lst}) for m in R}
    L = ["# GhostGraph LOSO results", "", f"Patients: {', '.join(patients)}", ""]
    L += ["Seeds per model: " + "; ".join(f"{m}: {seeds[m]}" for m in R), ""]

    # ---- window level
    L += ["## Table 3 - window level (mean ± SD over patients)", "", "| Model | AUROC | AUPRC | Sen (%) | Spe (%) | F1 (%) |", "|---|---|---|---|---|---|"]
    for m in models:
        P = PP[m]
        L.append(f"| {LABEL.get(m, m)} | {ms(P['auroc'], '%.3f')} | {ms(P['auprc'], '%.3f')} | {ms(P['sen'], '%.1f')} | {ms(P['spe'], '%.1f')} | {ms(P['f1'], '%.1f')} |")
    # ---- event level
    L += ["", "## Table 4 - event level (SzCORE)", "", "| Model | Detected/total | Sen (%) | Prec (%) | F1 (%) | FP/h | Latency median [IQR] (s) |", "|---|---|---|---|---|---|---|"]
    for m in models:
        P = PP[m]
        s0 = [d for lst in R[m].values() for d in lst if d["seed"] == seeds[m][0]]
        det = sum(d["event"]["tp"] for d in s0); tot = sum(d["event"]["n_sz"] for d in s0)
        lats = [l for d in s0 for l in d["event"]["latencies"]]
        lq = f"{np.median(lats):.0f} [{np.percentile(lats, 25):.0f}–{np.percentile(lats, 75):.0f}]" if lats else "—"
        L.append(f"| {LABEL.get(m, m)} | {det}/{tot} | {ms(P['ev_sen'], '%.1f')} | {ms(P['ev_prec'], '%.1f')} | {ms(P['ev_f1'], '%.1f')} | {ms(P['fp_h'], '%.2f')} | {lq} |")

    # ---- statistics
    st = {}
    for metric in ["auroc", "ev_f1", "fp_h"]:
        common = [p for p in patients if all(p in PP[m][metric] for m in models)]
        if len(models) >= 3 and len(common) >= 5:
            M_ = np.array([[PP[m][metric][p] for m in models] for p in common])
            chi, pf = stats.friedmanchisquare(*M_.T)
            raw, names = [], []
            for j, m in enumerate(models):
                if m == "GG-Net": continue
                try:
                    raw.append(stats.wilcoxon(M_[:, models.index("GG-Net")], M_[:, j], zero_method="zsplit").pvalue)
                except ValueError:
                    raw.append(1.0)
                names.append(m)
            st[metric] = dict(n=len(common), friedman_chi2=float(chi), friedman_p=float(pf),
                              wilcoxon_raw=dict(zip(names, map(float, raw))), wilcoxon_holm=dict(zip(names, map(float, holm(np.array(raw))))))
    json.dump(st, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    L += ["", "## Statistics (Friedman; Wilcoxon GG-Net vs each, Holm)", "", "```", json.dumps(st, indent=1), "```"]

    # ---- ablation
    if abl:
        L += ["", "## Table 5 - ablation", "", "| Variant | Params | AUROC | Event F1 (%) | Δ F1 | p (Wilcoxon vs full) |", "|---|---|---|---|---|---|"]
        full = PP.get("GG-Net", {}).get("ev_f1", {})
        for m in abl:
            prm = next((d["params"] for lst in R[m].values() for d in lst), None)
            f1 = PP[m]["ev_f1"]; common = [p for p in f1 if p in full]
            d = np.mean([f1[p] - full[p] for p in common]) if common else float("nan")
            try:
                pv = stats.wilcoxon([f1[p] for p in common], [full[p] for p in common], zero_method="zsplit").pvalue if m != "GG-Net" else None
            except ValueError:
                pv = 1.0
            L.append(f"| {m} | {prm} | {np.mean(list(PP[m]['auroc'].values())):.3f} | {np.mean(list(f1.values())):.1f} | "
                     f"{'—' if m == 'GG-Net' else f'{d:+.1f}'} | {'—' if pv is None else f'{pv:.3f}'} |")

    # ---- per patient (GG-Net)
    if "GG-Net" in R:
        L += ["", "## Table 6 - GG-Net per patient (seed 0)", "", "| Test | Val | AUROC | Detected/total | Event F1 (%) | FP/h | Median latency (s) |", "|---|---|---|---|---|---|---|"]
        rows = ["test,val,auroc,tp,n_sz,ev_f1,fp_h,lat_med,thr"]
        for p in patients:
            d = next((x for x in R["GG-Net"].get(p, []) if x["seed"] == 0), None)
            if not d: continue
            e = d["event"]
            lm = "—" if e["latency_median"] is None else "%.0f" % e["latency_median"]
            L.append(f"| {p} | {d['val']} | {d['seg']['auroc']:.3f} | {e['tp']}/{e['n_sz']} | {100*e['ev_f1']:.1f} | {e['fp_h']:.2f} | {lm} |")
            rows.append(f"{p},{d['val']},{d['seg']['auroc']:.4f},{e['tp']},{e['n_sz']},{e['ev_f1']:.4f},{e['fp_h']:.4f},{e['latency_median']},{d['thr']}")
        open(os.path.join(a.out, "per_patient.csv"), "w").write("\n".join(rows))
    open(os.path.join(a.out, "summary_tables.md"), "w").write("\n".join(L))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    # ---- Fig 2 ROC pooled
    fig, ax = plt.subplots(figsize=(3.6, 3.4), dpi=300)
    for m in models:
        ys, ps = [], []
        for f in glob.glob(os.path.join(a.runs, m.replace(" ", "_").replace("/", "-").replace(".", "") + "__chb*__s0_pred.npz")):
            d = np.load(f); ys.append(d["y"]); ps.append(d["p"].astype(np.float32))
        if not ys: continue
        y, p = np.concatenate(ys), np.concatenate(ps)
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=1.6 if m == "GG-Net" else 0.9, color=COLORS.get(m), label=f"{LABEL.get(m, m)} ({roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=0.6, color="#999")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(fontsize=5.5, frameon=False, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig2_roc.png")); plt.close(fig)

    # ---- Fig 4 per-patient event F1 and FP/h
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), dpi=300)
    for ax, key, lab in [(axs[0], "ev_f1", "Event F1 (%)"), (axs[1], "fp_h", "False positives per hour")]:
        data = [list(PP[m][key].values()) for m in models]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops=dict(color="black", lw=0.8))
        for b, m in zip(bp["boxes"], models):
            b.set_facecolor(COLORS.get(m, "#ccc")); b.set_alpha(0.75); b.set_linewidth(0.5)
        for i, d in enumerate(data):
            ax.scatter(np.full(len(d), i + 1) + np.random.default_rng(i).uniform(-0.15, 0.15, len(d)), d, s=5, color="black", zorder=3)
        ax.set_xticks(range(1, len(models) + 1)); ax.set_xticklabels([LABEL.get(m, m).replace(" (", "\n(") for m in models], rotation=45, ha="right", fontsize=6)
        ax.set_ylabel(lab); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig4_per_patient.png")); plt.close(fig)

    # ---- Fig 3 learned graphs (GG-Net, best-detected seizure)
    gfiles = sorted(glob.glob(os.path.join(a.runs, "GG-Net__chb*__s0_graphs.npz")))
    if gfiles:
        best = None
        for f in gfiles:
            g = np.load(f); keys = sorted({k.split("__")[0] for k in g.keys()})
            for k in keys:
                A, t, on, off = g[k + "__A"].astype(np.float32), g[k + "__t"], float(g[k + "__onset"]), float(g[k + "__offset"])
                if len(t) < 10: continue
                score = off - on
                if best is None or score > best[0]:
                    best = (score, os.path.basename(f).split("__")[1], k, A, t, on, off)
        if best:
            _, pat, k, A, t, on, off = best
            phases = [("interictal", (t + 4 <= on - 30)), ("onset", (t >= on - 2) & (t <= on + 10)), ("ictal", (t > on + 10) & (t + 4 <= off))]
            fig, axs = plt.subplots(3, 3, figsize=(6.0, 6.0), dpi=300)
            for c, (nm, msk) in enumerate(phases):
                if msk.sum() == 0: continue
                Am = A[msk].mean(0)
                for h in range(3):
                    im = axs[h, c].imshow(Am[h], vmin=0, vmax=1, cmap="Oranges")
                    axs[h, c].set_xticks([]); axs[h, c].set_yticks([])
                    if h == 0: axs[h, c].set_title(f"{nm} (n={int(msk.sum())})", fontsize=8)
                    if c == 0: axs[h, c].set_ylabel(f"head {h+1}", fontsize=8)
            fig.suptitle(f"GG-Net expected connectivity, {pat} {k} (onset {on:.0f} s)", fontsize=8)
            fig.colorbar(im, ax=axs, shrink=0.6, label="edge probability")
            fig.savefig(os.path.join(a.out, "fig3_graphs.png")); plt.close(fig)
    print("\n".join(L))


if __name__ == "__main__":
    main()
