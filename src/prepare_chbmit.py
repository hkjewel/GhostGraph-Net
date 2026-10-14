"""Download and compact the CHB-MIT subset (chb01-chb10) for the GhostGraph study.

usage: python prepare_chbmit.py --out /content/drive/MyDrive/GhostGraph/chbmit_prep [--workers 6]
Keeps every recording with a seizure plus the first 3 seizure-free recordings
per patient, the 18 common bipolar channels, 256 -> 128 Hz (windowed-sinc
low-pass at 50 Hz), float16 .npz.  Safe to re-run: finished files are skipped.
"""
import os, time, argparse, urllib.request, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

BASE = "https://physionet.org/files/chbmit/1.0.0/"
CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FZ-CZ", "CZ-PZ"]


def fetch(url, dst, tries=6):
    for k in range(tries):
        try:
            tmp = dst + ".part"
            with urllib.request.urlopen(url, timeout=180) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            os.replace(tmp, dst)
            return True
        except Exception as e:
            print(f"   retry {k+1}/{tries} {url}: {e}", flush=True)
            time.sleep(5 * (k + 1))
    return False


def read_edf(path):
    with open(path, "rb") as f:
        h = f.read(256)
        nrec = int(h[236:244]); dur = float(h[244:252]); ns = int(h[252:256])
        sh = f.read(256 * ns)
        nf = lambda start, width: [sh[start + i * width: start + (i + 1) * width].decode("latin-1").strip() for i in range(ns)]
        labels = nf(0, 16)
        p = 16 * ns + 80 * ns + 8 * ns
        pmin = np.array(nf(p, 8), float); p += 8 * ns
        pmax = np.array(nf(p, 8), float); p += 8 * ns
        dmin = np.array(nf(p, 8), float); p += 8 * ns
        dmax = np.array(nf(p, 8), float); p += 8 * ns
        p += 80 * ns
        nsamp = np.array(nf(p, 8), int)
        raw = np.frombuffer(f.read(), dtype="<i2")
    rec = int(nsamp.sum())
    nrec = min(nrec, raw.size // rec) if nrec > 0 else raw.size // rec
    raw = raw[: nrec * rec].reshape(nrec, rec)
    gain = (pmax - pmin) / (dmax - dmin)
    offs = np.concatenate([[0], np.cumsum(nsamp)])
    sigs = [((raw[:, offs[i]:offs[i + 1]].reshape(-1).astype(np.float32) - dmin[i]) * gain[i] + pmin[i]) for i in range(ns)]
    return labels, sigs, nsamp[0] / dur


def lowpass_decimate(x, fs):
    n = 129; t = np.arange(n) - (n - 1) / 2; fc = 50.0 / fs
    h = 2 * fc * np.sinc(2 * fc * t) * np.hamming(n); h /= h.sum()
    return np.convolve(x, h, mode="same")[::2]


def convert(edf, dst):
    labels, sigs, fs = read_edf(edf)
    up = [l.upper() for l in labels]
    out = []
    for ch in CHANNELS:
        idx = [i for i, l in enumerate(up) if l == ch or l.startswith(ch + "-")]
        if not idx:
            return f"missing channel {ch}"
        out.append(sigs[idx[0]])
    if abs(fs - 256) > 1:
        return f"unexpected fs {fs}"
    L = min(len(s) for s in out)
    X = np.stack([lowpass_decimate(s[:L], fs) for s in out]).astype(np.float16)
    np.savez_compressed(dst, x=X, fs=np.float32(fs / 2), ch=np.array(CHANNELS))
    return "ok"


def job(r, out, tmpdir):
    stem = os.path.basename(r)[:-4]
    dst = os.path.join(out, stem + ".npz")
    if os.path.exists(dst):
        return r, "exists"
    edf = os.path.join(tmpdir, os.path.basename(r))
    if not fetch(BASE + r, edf):
        return r, "download failed"
    try:
        st = convert(edf, dst)
    except Exception as e:
        st = f"error {e}"
    if os.path.exists(edf):
        os.remove(edf)
    return r, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--patients", default=",".join(f"chb{i:02d}" for i in range(1, 11)))
    ap.add_argument("--n_free", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="chbmit_edf_")
    for name in ["RECORDS", "RECORDS-WITH-SEIZURES", "SUBJECT-INFO"]:
        if not os.path.exists(os.path.join(a.out, name)):
            fetch(BASE + name, os.path.join(a.out, name))
    records = open(os.path.join(a.out, "RECORDS")).read().split()
    seiz = set(open(os.path.join(a.out, "RECORDS-WITH-SEIZURES")).read().split())
    todo = []
    for p in a.patients.split(","):
        s = os.path.join(a.out, f"{p}-summary.txt")
        if not os.path.exists(s):
            fetch(BASE + f"{p}/{p}-summary.txt", s)
        mine = [r for r in records if r.startswith(p + "/")]
        chosen = [r for r in mine if r in seiz] + [r for r in mine if r not in seiz][: a.n_free]
        print(f"{p}: {len(chosen)} recordings", flush=True)
        todo += chosen
    log = open(os.path.join(a.out, "log.txt"), "a")
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(job, r, a.out, tmpdir) for r in todo]
        for k, fu in enumerate(as_completed(futs), 1):
            r, st = fu.result()
            log.write(f"{r}\t{st}\n"); log.flush()
            print(f"[{k}/{len(todo)}] {r}: {st}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    print("Done. Folder:", a.out)


if __name__ == "__main__":
    main()
