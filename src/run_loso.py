"""Leave-one-subject-out experiments on CHB-MIT for the GhostGraph study.

usage: python run_loso.py --prep DIR --out DIR --models "GG-Net,EEGNet" [--epochs 15] [--seeds 0,1,2]
Every (model, test patient, seed) result is saved as JSON + NPZ, so runs resume
after an interruption. Uses the GPU when one is available.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import models as M
from events import event_scores, smooth

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cpu":
    torch.set_num_threads(max(1, os.cpu_count()))
else:
    torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------- caching
def patient_cache(prep, cache, p, seed=0):
    f = os.path.join(cache, f"{p}_train.npz")
    if os.path.exists(f):
        d = np.load(f)
        return d["X"], d["y"], d["F"]
    recs = D.load_patient(prep, p)
    X, y = D.build_train_windows(recs, np.random.default_rng(seed))
    Fh = D.handcrafted(X)
    np.savez(f, X=X, y=y, F=Fh)
    return X, y, Fh


def patient_test(prep, cache, p):
    f = os.path.join(cache, f"{p}_test.npz")
    if os.path.exists(f):
        d = np.load(f, allow_pickle=True)
        return list(d["T"])
    recs = D.load_patient(prep, p)
    T = D.build_test_windows(recs)
    for r in T:
        r["F"] = D.handcrafted(r["X"])
    np.savez(f, T=np.array(T, dtype=object))
    return T


# --------------------------------------------------------------------- training
def norm(X, med, sc):
    return np.clip((X.astype(np.float32) - med[None, :, None]) / sc[None, :, None], -20, 20)


def predict(model, X, Fh, med, sc, fmu, fsd, bs=1024, graphs=False):
    model.eval(); out, As = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = torch.from_numpy(norm(X[i:i + bs], med, sc)).to(DEV)
            f = torch.from_numpy(((Fh[i:i + bs] - fmu) / fsd).astype(np.float32)).to(DEV)
            out.append(model(x, f).softmax(-1)[:, 1].float().cpu().numpy())
            if graphs:
                As.append(model.last_A.float().cpu().numpy().astype(np.float16))
    p = np.concatenate(out) if out else np.zeros(0)
    return (p, np.concatenate(As)) if graphs else p


def train_nn(name, Xtr, ytr, Ftr, Xva, yva, Fva, med, sc, fmu, fsd, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    model = M.build(name, Xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    w = torch.tensor([1.0, (ytr == 0).sum() / max(1, (ytr == 1).sum())], dtype=torch.float32, device=DEV)
    best, best_state, bs = -1, None, 128
    hist = []
    for ep in range(epochs):
        model.train(); idx = rng.permutation(len(Xtr)); tl = 0
        for i in range(0, len(idx), bs):
            j = np.sort(idx[i:i + bs])
            x = norm(Xtr[j], med, sc)
            x *= rng.uniform(0.8, 1.2, (len(j), 1, 1)).astype(np.float32)          # amplitude scaling
            x += rng.normal(0, 0.05, x.shape).astype(np.float32)                  # additive noise
            sh = rng.integers(-13, 14); x = np.roll(x, sh, -1)                     # +-100 ms shift
            f = ((Ftr[j] - fmu) / fsd).astype(np.float32)
            logit = model(torch.from_numpy(x).to(DEV), torch.from_numpy(f).to(DEV))
            loss = F.cross_entropy(logit, torch.from_numpy(ytr[j]).to(DEV), weight=w)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tl += loss.item() * len(j)
        sched.step()
        pv = predict(model, Xva, Fva, med, sc, fmu, fsd)
        auc = roc_auc_score(yva, pv) if len(set(yva)) > 1 else 0.5
        hist.append(dict(epoch=ep, loss=tl / len(Xtr), val_auc=float(auc)))
        if auc > best:
            best = auc; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, hist


# --------------------------------------------------------------------- metrics
def choose_threshold(val_tests, probs):
    """Threshold maximising event-level F1 on the VALIDATION patient (never the
    test patient); ties broken toward the higher threshold (fewer false alarms)."""
    best, thr = -1, 0.5
    for t in np.arange(0.30, 0.96, 0.05):
        res = [event_scores(r, smooth(p) >= t) for r, p in zip(val_tests, probs)]
        tp = sum(x["tp"] for x in res); n = sum(x["n_sz"] for x in res); fp = sum(x["fp"] for x in res)
        sen = tp / max(1, n); prec = tp / max(1, tp + fp)
        f1 = 2 * sen * prec / max(1e-9, sen + prec)
        if f1 >= best:
            best, thr = f1, t
    return float(thr)


def segment_metrics(y, p, thr):
    yh = (smooth(p) >= thr).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    sen = tp / max(1, tp + fn); spe = tn / max(1, tn + fp); prec = tp / max(1, tp + fp)
    return dict(acc=(tp + tn) / len(y), sen=sen, spe=spe, prec=prec,
                f1=2 * sen * prec / max(1e-9, sen + prec), bacc=(sen + spe) / 2,
                auroc=float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
                auprc=float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan"),
                tp=tp, tn=tn, fp=fp, fn=fn)


def latency_ms(model, X, Fh, med, sc, fmu, fsd):
    xb = torch.from_numpy(norm(X[:1], med, sc)).to(DEV)
    fb = torch.from_numpy(((Fh[:1] - fmu) / fsd).astype(np.float32)).to(DEV)
    model.eval()
    with torch.no_grad():
        for _ in range(10): model(xb, fb)
        if DEV.type == "cuda": torch.cuda.synchronize()
        ts = time.perf_counter()
        for _ in range(100): model(xb, fb)
        if DEV.type == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - ts) / 100 * 1000


# --------------------------------------------------------------------- main
def run_one(a, name, k, P, seed, cache):
    test_p = P[k]
    tag = f"{name}__{test_p}__s{seed}".replace(" ", "_").replace("/", "-").replace(".", "")
    fout = os.path.join(a.out, "runs", tag + ".json")
    if os.path.exists(fout):
        return
    val_p = P[(k + 1) % len(P)]
    tr_p = [q for q in P if q not in (test_p, val_p)]
    t0 = time.time()
    parts = [patient_cache(a.prep, cache, q) for q in tr_p]
    Xtr = np.concatenate([x[0] for x in parts]); ytr = np.concatenate([x[1] for x in parts])
    Ftr = np.concatenate([x[2] for x in parts]); del parts
    Xva, yva, Fva = patient_cache(a.prep, cache, val_p)
    med, sc = D.channel_scale(Xtr)
    fmu = Ftr.mean((0, 1), keepdims=True)[0]; fsd = Ftr.std((0, 1), keepdims=True)[0] + 1e-6
    val_T = patient_test(a.prep, cache, val_p); test_T = patient_test(a.prep, cache, test_p)
    hist, params, lat, graphs = [], None, None, None
    if name == "RF (handcrafted)":
        rf = RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=seed)
        rf.fit(Ftr.reshape(len(Ftr), -1), ytr)
        pr = lambda T: [rf.predict_proba(r["F"].reshape(len(r["F"]), -1))[:, 1] for r in T]
        pt = pr(test_T)
    else:
        model, hist = train_nn(name, Xtr, ytr, Ftr, Xva, yva, Fva, med, sc, fmu, fsd, a.epochs, seed)
        pr = lambda T: [predict(model, r["X"], r["F"], med, sc, fmu, fsd) for r in T]
        params = M.n_params(model)
        lat = latency_ms(model, test_T[0]["X"], test_T[0]["F"], med, sc, fmu, fsd)
        if name == "GG-Net" and seed == 0:
            pt, graphs = [], {}
            for r in test_T:
                p_, A_ = predict(model, r["X"], r["F"], med, sc, fmu, fsd, graphs=True)
                pt.append(p_)
                for s_i, (s, e) in enumerate(r["seizures"]):
                    keep = (r["t"] >= s - 120) & (r["t"] <= e + 60)
                    graphs[f"{r['stem']}_sz{s_i}"] = dict(A=A_[keep], t=r["t"][keep], onset=s, offset=e)
            alpha = getattr(model.dec, "last_alpha", None)
        else:
            pt = pr(test_T)
    thr = choose_threshold(val_T, pr(val_T))
    y = np.concatenate([r["y"] for r in test_T]); p = np.concatenate(pt)
    seg = segment_metrics(y, p, thr)
    ev = [event_scores(r, smooth(q) >= thr) for r, q in zip(test_T, pt)]
    hours = sum(r["dur"] for r in test_T) / 3600
    tp = sum(e["tp"] for e in ev); nsz = sum(e["n_sz"] for e in ev); fp = sum(e["fp"] for e in ev)
    lats = [float(l) for e in ev for l in e["latency"]]
    evm = dict(n_sz=nsz, tp=tp, fp=fp, hours=hours, ev_sen=tp / max(1, nsz), ev_prec=tp / max(1, tp + fp),
               fp_h=fp / hours, latency_mean=float(np.mean(lats)) if lats else None,
               latency_median=float(np.median(lats)) if lats else None, latencies=lats)
    evm["ev_f1"] = 2 * evm["ev_sen"] * evm["ev_prec"] / max(1e-9, evm["ev_sen"] + evm["ev_prec"])
    res = dict(model=name, test=test_p, val=val_p, seed=seed, thr=thr, params=params, latency_ms=lat,
               device=str(DEV), epochs=a.epochs, train_n=int(len(ytr)), train_pos=int(ytr.sum()),
               seg=seg, event=evm, hist=hist, minutes=(time.time() - t0) / 60)
    np.savez_compressed(os.path.join(a.out, "runs", tag + "_pred.npz"), p=p.astype(np.float16), y=y.astype(np.int8),
                        stems=np.array([r["stem"] for r in test_T]), lens=np.array([len(r["y"]) for r in test_T]))
    if graphs:
        np.savez_compressed(os.path.join(a.out, "runs", tag + "_graphs.npz"),
                            **{f"{k}__{f}": v[f] for k, v in graphs.items() for f in ("A", "t", "onset", "offset")})
    json.dump(res, open(fout, "w"), indent=1)
    print(f"{name:28s} {test_p} s{seed} AUROC={seg['auroc']:.3f} sen={seg['sen']:.3f} spe={seg['spe']:.3f} "
          f"evSen={evm['ev_sen']:.2f} FP/h={evm['fp_h']:.2f} ({res['minutes']:.1f} min)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None, help="fast local dir for window caches (default OUT/cache)")
    ap.add_argument("--models", required=True); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patients", default=",".join(f"chb{i:02d}" for i in range(1, 11)))
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    P = a.patients.split(","); cache = a.cache or os.path.join(a.out, "cache")
    os.makedirs(cache, exist_ok=True); os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)
    print("device:", DEV, flush=True)
    for seed in [int(s) for s in a.seeds.split(",")]:
        for name in a.models.split(","):
            for k in range(len(P)):
                run_one(a, name, k, P, seed, cache)


if __name__ == "__main__":
    main()
