"""Event-level scoring following the SzCORE conventions (Dan et al., Epilepsia 2024):
any-overlap matching with 30 s pre-onset and 60 s post-offset tolerance,
hypothesis events closer than 90 s merged, events longer than 5 min split."""
import numpy as np

STRIDE = 2.0      # s between window starts
WIN = 4.0         # s window length
PRE, POST, MERGE, MAXLEN = 30.0, 60.0, 90.0, 300.0


def smooth(p, k=3):
    """Causal moving average over k consecutive windows (k*2 s)."""
    p = np.asarray(p, np.float64)
    c = np.cumsum(np.r_[0.0, p])
    i = np.arange(len(p))
    lo = np.maximum(0, i - k + 1)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)


def to_events(binary, t):
    ev, on = [], None
    for b, s in zip(binary, t):
        if b and on is None:
            on = s
        if not b and on is not None:
            ev.append([on, prev + WIN]); on = None
        prev = s
    if on is not None:
        ev.append([on, t[-1] + WIN])
    merged = []
    for e in ev:
        if merged and e[0] - merged[-1][1] < MERGE:
            merged[-1][1] = e[1]
        else:
            merged.append(e)
    out = []
    for s, e in merged:
        while e - s > MAXLEN:
            out.append([s, s + MAXLEN]); s += MAXLEN
        out.append([s, e])
    return out


def event_scores(rec, binary):
    hyp = to_events(np.asarray(binary, bool), rec["t"])
    ref = rec["seizures"]
    tp, lat, used = 0, [], np.zeros(len(hyp), bool)
    for s, e in ref:
        hit = [k for k, (hs, he) in enumerate(hyp) if he >= s - PRE and hs <= e + POST]
        if hit:
            tp += 1; used[hit] = True
            lat.append(max(0.0, hyp[hit[0]][0] + WIN - s))   # decision available at window end
    fp = int((~used).sum())
    return dict(tp=tp, n_sz=len(ref), fp=fp, latency=lat)
