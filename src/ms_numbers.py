"""Compute every manuscript number/figure from the Colab results package.
usage: python ms_numbers.py RESULTS_DIR OUT_DIR"""
import os, sys, glob, json, re
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import parse_summary

P = [f"chb{i:02d}" for i in range(1, 11)]
runs = [json.load(open(f)) for f in glob.glob(f"{RES}/runs/*.json")]
by = {}
for d in runs:
    by.setdefault(d["model"], {}).setdefault(d["test"], {})[d["seed"]] = d
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
complete = [m for m in by if all(p in by[m] and all(sd in by[m][p] for sd in SEEDS) for p in P)]
ALL = ["RF (handcrafted)", "EEGNet", "ShallowConvNet", "CNN-LSTM", "Static GCN", "EEG Transformer", "GGN (re-impl.)", "GG-Net"]
order = [m for m in ALL if m in complete]
two_seed = [m for m in ALL if m in by and all(p in by[m] and 0 in by[m][p] and 1 in by[m][p] for p in P)]
print("complete models:", order)
fn = lambda m: m.replace(" ", "_").replace("/", "-").replace(".", "")
OUTJ = {"complete_models": order}

# ---------------------------------------------------------------- Table 1
t1 = []
d0 = by["GG-Net"]
for p in P:
    summ = parse_summary(f"{RES}/{p}-summary.txt")
    pr = np.load(f"{RES}/runs/GG-Net__{p}__s0_pred.npz")
    stems = [str(s) for s in pr["stems"]]
    sz = [(s, e) for st in stems for (s, e) in summ.get(st, [])]
    t1.append(dict(p=p, rec=len(stems), hours=d0[p][0]["event"]["hours"], nsz=len(sz), ictal=sum(e - s for s, e in sz),
                   n_seiz_rec=sum(1 for st in stems if summ.get(st))))
OUTJ["table1"] = t1
OUTJ["table1_total"] = dict(rec=sum(r["rec"] for r in t1), hours=sum(r["hours"] for r in t1), nsz=sum(r["nsz"] for r in t1),
                            ictal=sum(r["ictal"] for r in t1))
info = open(f"{RES}/SUBJECT-INFO").read()
OUTJ["subject_info"] = info

# ---------------------------------------------------------------- per-patient metrics (seed mean)
MET = {"auroc": lambda d: d["seg"]["auroc"], "auprc": lambda d: d["seg"]["auprc"], "sen": lambda d: 100 * d["seg"]["sen"],
       "spe": lambda d: 100 * d["seg"]["spe"], "f1": lambda d: 100 * d["seg"]["f1"], "ev_sen": lambda d: 100 * d["event"]["ev_sen"],
       "ev_prec": lambda d: 100 * d["event"]["ev_prec"], "ev_f1": lambda d: 100 * d["event"]["ev_f1"], "fp_h": lambda d: d["event"]["fp_h"]}
seeds = SEEDS
PP = {m: {k: np.array([np.mean([f(by[m][p][s]) for s in seeds]) for p in P]) for k, f in MET.items()} for m in order}
rng = np.random.default_rng(0)


def boot(v, n=10000):
    idx = rng.integers(0, len(v), (n, len(v)))
    return np.percentile(v[idx].mean(1), [2.5, 97.5])


tab = {}
for m in order:
    r = {k: dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), ci=list(map(float, boot(v)))) for k, v in PP[m].items()}
    det = {s: (sum(by[m][p][s]["event"]["tp"] for p in P), sum(by[m][p][s]["event"]["n_sz"] for p in P)) for s in seeds}
    fps = {s: sum(by[m][p][s]["event"]["fp"] for p in P) for s in seeds}
    hrs = sum(by[m][p][0]["event"]["hours"] for p in P)
    lats = [l for p in P for s in seeds for l in by[m][p][s]["event"]["latencies"]]
    r["detected"] = det; r["fp_total"] = fps; r["hours"] = hrs
    r["fp_h_pooled"] = {s: fps[s] / hrs for s in seeds}
    r["lat"] = dict(median=float(np.median(lats)), q1=float(np.percentile(lats, 25)), q3=float(np.percentile(lats, 75)), n=len(lats))
    r["thr"] = [by[m][p][s]["thr"] for p in P for s in seeds]
    r["params"] = by[m][P[0]][0]["params"]
    tab[m] = r
OUTJ["tab"] = tab
stab = {}
for m in two_seed:
    a = np.array([[by[m][p][sd]["seg"]["auroc"] for sd in (0, 1)] for p in P]); es = np.array([[by[m][p][sd]["event"]["ev_sen"] for sd in (0, 1)] for p in P])
    stab[m] = dict(dauroc=float(np.abs(a[:, 0] - a[:, 1]).mean()), devsen=float(100 * np.abs(es[:, 0] - es[:, 1]).mean()),
                   det={sd: sum(by[m][p][sd]["event"]["tp"] for p in P) for sd in (0, 1)})
OUTJ["stability"] = stab
OUTJ["per_patient"] = {m: {k: list(map(float, v)) for k, v in PP[m].items()} for m in order}


# ---------------------------------------------------------------- statistics
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj


st = {}
for k in ["auroc", "auprc", "ev_sen", "ev_prec", "ev_f1", "fp_h"]:
    M = np.array([PP[m][k] for m in order]).T
    chi, pf = stats.friedmanchisquare(*M.T)
    others = [m for m in order if m != "GG-Net"]
    raw = [stats.wilcoxon(PP["GG-Net"][k], PP[m][k], zero_method="zsplit").pvalue for m in others]
    st[k] = dict(friedman_chi2=float(chi), friedman_p=float(pf), wilcoxon={m: dict(p=float(r), p_holm=float(h),
                 wins=int((PP["GG-Net"][k] > PP[m][k]).sum()), diff=float(np.mean(PP["GG-Net"][k] - PP[m][k])))
                 for m, r, h in zip(others, raw, holm(raw))})
OUTJ["stats"] = st

# ---------------------------------------------------------------- pooled ROC (seeds 0 and 1)
COL = {"GG-Net": "#B05A2A", "GGN (re-impl.)": "#3B5BA5", "EEGNet": "#2E7D6B", "EEG Transformer": "#7A4E9C", "CNN-LSTM": "#B07D1A", "Static GCN": "#5B6470", "ShallowConvNet": "#9AA3AD", "RF (handcrafted)": "#C9CED6"}
LAB = {"GG-Net": "GG-Net (proposed)", "GGN (re-impl.)": "GGN (re-implemented)", "EEGNet": "EEGNet-8,2", "Static GCN": "Static-graph GCN", "RF (handcrafted)": "RF (spectral)"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6})
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1), dpi=300)
pooled = {}
for m in order:
    ys, ps = [], []
    for p in P:
        d = np.load(f"{RES}/runs/{fn(m)}__{p}__s0_pred.npz"); ys.append(d["y"]); ps.append(d["p"].astype(np.float32))
    y, pr = np.concatenate(ys), np.concatenate(ps)
    fpr, tpr, _ = roc_curve(y, pr); auc = roc_auc_score(y, pr); pooled[m] = float(auc)
    for ax in axs:
        ax.plot(fpr, tpr, lw=1.5 if m == "GG-Net" else 1.0, color=COL[m], label=f"{LAB.get(m, m)} (AUC {auc:.3f})")
for ax, lim, t in [(axs[0], 1.0, "(a) pooled ROC, all held-out windows"), (axs[1], 0.1, "(b) low false-positive region")]:
    ax.plot([0, 1], [0, 1], ls="--", lw=0.5, color="#999"); ax.set_xlim(0, lim); ax.set_ylim(0, 1.0)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title(t, fontsize=8, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
axs[0].legend(fontsize=6.5, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_roc.png"); plt.close(fig)
OUTJ["pooled_auroc_s0"] = pooled
OUTJ["n_windows"] = int(len(y)); OUTJ["n_ictal_windows"] = int(y.sum())

# ---------------------------------------------------------------- Fig 4 per patient
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), dpi=300)
x = np.arange(len(P)); w = 0.8 / len(order)
for ax, k, lab in [(axs[0], "auroc", "Window AUROC"), (axs[1], "ev_sen", "Event sensitivity (%)"), (axs[2], "fp_h", "False positives per hour")]:
    for i, m in enumerate(order):
        ax.bar(x + (i - (len(order) - 1) / 2) * w, PP[m][k], w, color=COL[m], label=LAB.get(m, m))
    ax.set_xticks(x); ax.set_xticklabels([p[3:] for p in P], fontsize=6.5); ax.set_xlabel("held-out patient (chb)")
    ax.set_title(lab, fontsize=8, loc="left"); ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylim(0.4, 1.0)
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, fontsize=7, frameon=False, ncol=len(order), loc="upper center")
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(f"{OUT}/fig4_per_patient.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3 connectivity analysis (GG-Net seed 0)
iu = np.triu_indices(18, 1)
phase_vals = {"interictal": [], "onset": [], "ictal": [], "post-ictal": []}
curves = []
per_head = {h: {"interictal": [], "ictal": [], "post-ictal": []} for h in range(3)}
example = None
for p in P:
    f = f"{RES}/runs/GG-Net__{p}__s0_graphs.npz"
    if not os.path.exists(f): continue
    g = np.load(f)
    for k in sorted({kk.split("__")[0] for kk in g.keys()}):
        A = g[k + "__A"].astype(np.float32); t = g[k + "__t"]; on = float(g[k + "__onset"]); off = float(g[k + "__offset"])
        dens = A[:, :, iu[0], iu[1]].mean(-1)  # (n, heads)
        m_int = (t + 4 <= on - 30); m_on = (t >= on - 2) & (t <= on + 10); m_ict = (t > on + 10) & (t + 4 <= off)
        m_post = (t >= off) & (t + 4 <= off + 60)
        if m_int.sum() >= 3 and m_ict.sum() >= 3 and m_on.sum() >= 1 and m_post.sum() >= 3:
            phase_vals["post-ictal"].append(dens[m_post].mean())
            phase_vals["interictal"].append(dens[m_int].mean()); phase_vals["onset"].append(dens[m_on].mean())
            phase_vals["ictal"].append(dens[m_ict].mean())
            for h in range(3):
                per_head[h]["interictal"].append(dens[m_int, h].mean()); per_head[h]["ictal"].append(dens[m_ict, h].mean())
                per_head[h]["post-ictal"].append(dens[m_post, h].mean())
            rel = t + 2 - on
            bins = np.arange(-120, 121, 4)
            c = np.full(len(bins) - 1, np.nan)
            for b in range(len(bins) - 1):
                mm = (rel >= bins[b]) & (rel < bins[b + 1]) & (t + 4 <= off + 60)
                if mm.any(): c[b] = dens[mm].mean()
            curves.append(c)
            if example is None or (off - on) > example[0]:
                example = (off - on, p, k, A, t, on, off)
pv = {k: np.array(v) for k, v in phase_vals.items()}
w_ii = stats.wilcoxon(pv["ictal"], pv["interictal"])
w_oi = stats.wilcoxon(pv["onset"], pv["interictal"])
w_pi = stats.wilcoxon(pv["post-ictal"], pv["interictal"]); w_pc = stats.wilcoxon(pv["post-ictal"], pv["ictal"])
OUTJ["graphs"] = dict(n_seizures=len(pv["ictal"]), mean={k: float(v.mean()) for k, v in pv.items()}, sd={k: float(v.std(ddof=1)) for k, v in pv.items()},
                      ictal_gt_inter=int((pv["ictal"] > pv["interictal"]).sum()), p_ictal_vs_inter=float(w_ii.pvalue), p_onset_vs_inter=float(w_oi.pvalue), p_post_vs_inter=float(w_pi.pvalue), p_post_vs_ictal=float(w_pc.pvalue),
                      post_lt_inter=int((pv["post-ictal"] < pv["interictal"]).sum()), post_lt_ictal=int((pv["post-ictal"] < pv["ictal"]).sum()),
                      head_tests={h: {ph: dict(mean=float(np.mean(v[ph])), p_vs_inter=(float(stats.wilcoxon(v[ph], v["interictal"]).pvalue) if ph != "interictal" else None),
                                  n_lower=int((np.array(v[ph]) < np.array(v["interictal"])).sum())) for ph in v} for h, v in per_head.items()},
                      per_head={h: dict(inter=float(np.mean(v["interictal"])), ictal=float(np.mean(v["ictal"]))) for h, v in per_head.items()})
# figure: (a) example adjacency per phase (head-averaged), (b) density time course, (c) paired phases
_, ep, ek, A, t, on, off = example
fig = plt.figure(figsize=(7.2, 4.6), dpi=300)
phases = [("interictal", (t + 4 <= on - 30)), ("onset", (t >= on - 2) & (t <= on + 10)), ("ictal", (t > on + 10) & (t + 4 <= off))]
for c, (nm, msk) in enumerate(phases):
    ax = fig.add_axes([0.05 + c * 0.155, 0.55, 0.14, 0.36])
    im = ax.imshow(A[msk].mean(0).mean(0), vmin=0.2, vmax=0.7, cmap="Oranges")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{nm}", fontsize=7.5)
cax = fig.add_axes([0.515, 0.58, 0.008, 0.3]); fig.colorbar(im, cax=cax); cax.tick_params(labelsize=6)
fig.text(0.05, 0.95, f"(a) mean generated adjacency, {ep} seizure {ek.split('_sz')[0]} (heads averaged)", fontsize=7.5)
ax = fig.add_axes([0.64, 0.55, 0.33, 0.36])
C = np.array(curves); centers = np.arange(-120, 120, 4) + 2
mu = np.nanmean(C, 0); se = np.nanstd(C, 0) / np.sqrt(np.sum(~np.isnan(C), 0))
ax.plot(centers, mu, color="#B05A2A", lw=1.2); ax.fill_between(centers, mu - se, mu + se, color="#B05A2A", alpha=0.25, lw=0)
ax.axvline(0, color="black", lw=0.6, ls="--"); ax.set_xlabel("time from annotated onset (s)", fontsize=7)
ax.set_ylabel("mean edge probability", fontsize=7); ax.tick_params(labelsize=6.5); ax.spines[["top", "right"]].set_visible(False)
ax.set_title(f"(b) aligned to onset (mean ± s.e., n = {len(curves)})", fontsize=7.5, loc="left")
ax = fig.add_axes([0.08, 0.08, 0.38, 0.36])
PH = ["interictal", "onset", "ictal", "post-ictal"]
for i in range(len(pv["ictal"])):
    ax.plot(range(4), [pv[k][i] for k in PH], color="#999", lw=0.4, alpha=0.6)
ax.plot(range(4), [pv[k].mean() for k in PH], color="#B05A2A", lw=1.6, marker="o", ms=3)
ax.set_xticks(range(4)); ax.set_xticklabels(PH, fontsize=7); ax.tick_params(labelsize=6.5)
ax.set_ylabel("mean edge probability", fontsize=7); ax.spines[["top", "right"]].set_visible(False)
ax.set_title(f"(c) per seizure (n = {len(pv['ictal'])}); post-ictal < interictal: p = {w_pi.pvalue:.1e}", fontsize=7.5, loc="left")
ax = fig.add_axes([0.58, 0.08, 0.39, 0.36])
cols = {"interictal": "#E8C3A8", "ictal": "#D17C45", "post-ictal": "#7A3412"}
for j, ph in enumerate(["interictal", "ictal", "post-ictal"]):
    ax.bar(np.arange(3) + (j - 1) * 0.26, [np.mean(per_head[h][ph]) for h in range(3)], 0.26, color=cols[ph], label=ph)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["head 1", "head 2", "head 3"], fontsize=7); ax.tick_params(labelsize=6.5)
ax.set_ylabel("mean edge probability", fontsize=7); ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper left"); ax.set_ylim(0, 0.38)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("(d) by graph head", fontsize=7.5, loc="left")
fig.savefig(f"{OUT}/fig3_graphs.png"); plt.close(fig)

json.dump(OUTJ, open(f"{OUT}/ms_numbers.json", "w"), indent=1)
print(json.dumps({k: v for k, v in OUTJ.items() if k not in ("subject_info", "per_patient")}, indent=1)[:6000])
