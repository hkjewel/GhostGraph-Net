"""CHB-MIT data handling for the GhostGraph study.

Reads the compact .npz files produced by prepare_chbmit.py (18 bipolar channels,
128 Hz, float16) and the PhysioNet summary files; builds 4-s windows, labels,
and leakage-free (training-fold-only) normalisation statistics.
"""
import os, re, glob
import numpy as np

FS = 128
WIN_S = 4
WIN = FS * WIN_S


def parse_summary(path):
    """Return {recording_stem: [(start_s, end_s), ...]} from a chbXX-summary.txt."""
    out, cur = {}, None
    starts = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"File Name:\s*(\S+)\.edf", line)
            if m:
                cur = m.group(1); out[cur] = []; starts = []
                continue
            if cur is None:
                continue
            m = re.match(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)", line)
            if m:
                starts.append(int(m.group(1))); continue
            m = re.match(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)", line)
            if m and starts:
                out[cur].append((starts.pop(0), int(m.group(1))))
    return out


def load_patient(prep_dir, patient):
    """List of dicts: stem, x (C,N float32), seizures [(s,e)] for one patient."""
    summ = parse_summary(os.path.join(prep_dir, f"{patient}-summary.txt"))
    recs = []
    for f in sorted(glob.glob(os.path.join(prep_dir, f"{patient}_*.npz"))):
        stem = os.path.basename(f)[:-4]
        if stem not in summ:
            continue
        d = np.load(f)
        recs.append(dict(stem=stem, x=d["x"].astype(np.float32), seizures=summ[stem]))
    return recs


def window_starts(n_samples, stride_s):
    stride = int(stride_s * FS)
    return np.arange(0, n_samples - WIN + 1, stride)


def label_windows(starts, seizures, min_frac=0.5):
    """Window is ictal if >= min_frac of it lies inside an annotated seizure."""
    y = np.zeros(len(starts), np.int64)
    t0 = starts / FS; t1 = t0 + WIN_S
    for s, e in seizures:
        ov = np.clip(np.minimum(t1, e) - np.maximum(t0, s), 0, None)
        y[ov >= min_frac * WIN_S] = 1
    return y


def build_train_windows(recs, rng, ictal_stride=2.0, inter_stride=4.0, ratio=5,
                        exclude_margin_s=60):
    """Return X (n,C,WIN) float16, y. Ictal windows densely sampled; interictal
    windows drawn ratio:1 at random, excluding +/- margin around seizures."""
    Xi, Xn = [], []
    for r in recs:
        x, N = r["x"], r["x"].shape[1]
        st = window_starts(N, ictal_stride)
        y = label_windows(st, r["seizures"])
        for s in st[y == 1]:
            Xi.append(x[:, s:s + WIN])
        st2 = window_starts(N, inter_stride)
        keep = np.ones(len(st2), bool)
        t0 = st2 / FS
        for s, e in r["seizures"]:
            keep &= ~((t0 + WIN_S > s - exclude_margin_s) & (t0 < e + exclude_margin_s))
        Xn.extend([(r["stem"], s) for s in st2[keep]])
    n_int = min(len(Xn), max(ratio * len(Xi), 2000))
    pick = rng.choice(len(Xn), n_int, replace=False)
    lookup = {r["stem"]: r["x"] for r in recs}
    Xn = [lookup[Xn[k][0]][:, Xn[k][1]:Xn[k][1] + WIN] for k in pick]
    X = np.stack(Xi + Xn).astype(np.float16)
    y = np.r_[np.ones(len(Xi)), np.zeros(len(Xn))].astype(np.int64)
    return X, y


def build_test_windows(recs, stride_s=2.0):
    """Continuous windows over every recording (for segment + event scoring)."""
    out = []
    for r in recs:
        st = window_starts(r["x"].shape[1], stride_s)
        X = np.stack([r["x"][:, s:s + WIN] for s in st]).astype(np.float16)
        y = label_windows(st, r["seizures"])
        out.append(dict(stem=r["stem"], X=X, y=y, t=st / FS,
                        seizures=r["seizures"], dur=r["x"].shape[1] / FS))
    return out


def channel_scale(X):
    """Per-channel robust scale from TRAINING windows only."""
    Xf = X[:: max(1, len(X) // 4000)].astype(np.float32)
    med = np.median(Xf, axis=(0, 2))
    iqr = np.subtract(*np.percentile(Xf, [75, 25], axis=(0, 2)))
    return med.astype(np.float32), (iqr / 1.349 + 1e-3).astype(np.float32)


# ---------------------------------------------------------------- handcrafted
BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
FEAT_NAMES = ["logP_delta", "logP_theta", "logP_alpha", "logP_beta", "logP_gamma",
              "log_line_length", "hjorth_mobility", "hjorth_complexity"]


def handcrafted(X):
    """X (n,C,T) -> (n,C,8): band log-power, log line length, Hjorth mob/comp."""
    X = X.astype(np.float32)
    n, C, T = X.shape
    F = np.fft.rfft(X * np.hanning(T)[None, None], axis=-1)
    P = (np.abs(F) ** 2) / T
    f = np.fft.rfftfreq(T, 1 / FS)
    feats = [np.log(P[..., (f >= lo) & (f < hi)].mean(-1) + 1e-6) for lo, hi in BANDS]
    d1 = np.diff(X, axis=-1); d2 = np.diff(d1, axis=-1)
    feats.append(np.log(np.abs(d1).sum(-1) + 1e-6))
    v0, v1, v2 = X.var(-1) + 1e-8, d1.var(-1) + 1e-8, d2.var(-1) + 1e-8
    mob = np.sqrt(v1 / v0); comp = np.sqrt(v2 / v1) / mob
    feats += [mob, comp]
    return np.stack(feats, -1).astype(np.float32)
