"""NumPy side of the numj benchmarks. Same protocol and same input bits as java/bench/numj/bench/Bench.java.

Implementations:
  numpy        ordinary expressions (allocate temporaries and results)       -> mode "e2e"
  numpy-reuse  ufuncs with preallocated out= buffers (+ np.dot for sums)     -> mode "compute"

Usage: python -I bench/numpy_bench.py --out results/numpy.csv [--quick]
"""
import os

# Single-threaded backend, set before NumPy loads its BLAS. (Elementwise ufuncs are single-threaded anyway.)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import io
import json
import math
import platform
import statistics
import sys
import time
from contextlib import redirect_stdout

import numpy as np

WARMUP_S, SAMPLE_S, SAMPLES = 0.5, 0.020, 21


def splitmix(seed: int, n: int) -> np.ndarray:
    """Bit-identical to numj.bench.Data.splitmix (uint64 arithmetic wraps mod 2**64)."""
    i = np.arange(1, n + 1, dtype=np.uint64)
    z = np.uint64(seed) + i * np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return ((z >> np.uint64(11)).astype(np.float64) * 2.0**-53) * 2.0 - 1.0


def fingerprint(x: np.ndarray) -> int:
    """Same FNV-style fold as Data.fingerprint (Java long arithmetic, as unsigned)."""
    h = 0xCBF29CE484222325
    for bits in x.view(np.uint64).tolist():
        h = ((h ^ bits) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


rows_out = []


def run(kernel, shape, n, impl, mode, op):
    end = time.perf_counter() + WARMUP_S
    r = op()
    while time.perf_counter() < end:
        r = op()
    reps = 1
    while True:
        t0 = time.perf_counter_ns()
        for _ in range(reps):
            op()
        dt = time.perf_counter_ns() - t0
        if dt >= SAMPLE_S * 1e9:
            break
        reps = max(reps * 2, math.ceil(reps * (SAMPLE_S * 1e9 / max(dt, 1)) * 1.1))
    per = []
    for _ in range(SAMPLES):
        t0 = time.perf_counter_ns()
        for _ in range(reps):
            op()
        per.append((time.perf_counter_ns() - t0) / reps)
    result = float(np.asarray(op()).ravel()[-1])
    rows_out.append((kernel, shape, n, impl, mode, per, reps, result))
    s = sorted(per)
    print(f"{kernel:15s} {shape:16s} {impl:22s} {mode:8s} median {statistics.median(s):14.1f} ns  "
          f"cv {100 * statistics.stdev(s) / statistics.fmean(s):5.1f}%", flush=True)


def pct(s, p):
    idx = p * (len(s) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def main():
    global WARMUP_S, SAMPLE_S, SAMPLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        WARMUP_S, SAMPLE_S, SAMPLES = 0.05, 0.005, 5

    for n in (16, 1_000, 100_000, 10_000_000):
        a, b, c = splitmix(1, n), splitmix(2, n), splitmix(3, n)
        t = np.empty(n)
        shape = f"[{n}]"
        run("sqdist", shape, n, "numpy", "e2e", lambda: np.sum((a - b) ** 2))

        def sq_reuse():
            np.subtract(a, b, out=t)
            return np.dot(t, t)
        run("sqdist", shape, n, "numpy-reuse", "compute", sq_reuse)

        run("sumsq_muladd", shape, n, "numpy", "e2e", lambda: np.sum((a * b + c) ** 2))

        def ma_reuse():
            np.multiply(a, b, out=t)
            np.add(t, c, out=t)
            return np.dot(t, t)
        run("sumsq_muladd", shape, n, "numpy-reuse", "compute", ma_reuse)
        del a, b, c, t

    for r, k in ((4, 4), (100, 10), (1_000, 100), (100_000, 100)):
        n = r * k
        x = splitmix(4, n).reshape(r, k)
        y, tmp, nrm = np.empty_like(x), np.empty_like(x), np.empty(r)
        shape = f"[{r},{k}]"

        def norm_plain():
            nn = np.linalg.norm(x, axis=1, keepdims=True)
            return x / np.where(nn == 0, 1.0, nn)          # zero rows copied, like numj
        run("normalize_rows", shape, n, "numpy", "e2e", norm_plain)

        def norm_reuse():
            np.multiply(x, x, out=tmp)
            np.sum(tmp, axis=1, out=nrm)
            np.sqrt(nrm, out=nrm)
            nrm[nrm == 0] = 1.0
            np.divide(x, nrm[:, None], out=y)
            return y
        run("normalize_rows", shape, n, "numpy-reuse", "compute", norm_reuse)

    for r, k in ((1_000, 16), (10_000, 100), (100_000, 100)):
        n = r * k
        x = splitmix(4, n).reshape(r, k)
        q = splitmix(5, k)
        tmp, o = np.empty_like(x), np.empty(r)
        shape = f"[{r},{k}]"
        run("sqdist_rows", shape, n, "numpy", "e2e", lambda: ((x - q) ** 2).sum(axis=1))

        def rows_reuse():
            np.subtract(x, q, out=tmp)
            np.multiply(tmp, tmp, out=tmp)
            np.sum(tmp, axis=1, out=o)
            return o
        run("sqdist_rows", shape, n, "numpy-reuse", "compute", rows_reuse)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("suite,kernel,shape,n,impl,mode,threads,profile,median_ns,min_ns,p10_ns,p90_ns,mean_ns,stdev_ns,samples,reps,result".split(","))
        for kernel, shape, n, impl, mode, per, reps, result in rows_out:
            s = sorted(per)
            w.writerow(["main", kernel, shape, n, impl, mode, 1, "numpy",
                        f"{statistics.median(s):.3f}", f"{s[0]:.3f}", f"{pct(s, 0.1):.3f}", f"{pct(s, 0.9):.3f}",
                        f"{statistics.fmean(s):.3f}", f"{statistics.stdev(s):.3f}", len(s), reps, repr(result)])

    # Environment record: versions, backend, SIMD dispatch, and input fingerprints.
    buf = io.StringIO()
    with redirect_stdout(buf):
        np.show_runtime()
    env = {
        "python": sys.version,
        "numpy": np.__version__,
        "numpy_config": np.show_config(mode="dicts"),
        "numpy_runtime": buf.getvalue(),
        "platform": platform.platform(),
        "thread_env": {v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "fingerprints": {f"splitmix({s},{n})": f"{fingerprint(splitmix(s, n)):016x}"
                         for s, n in ((1, 1000), (2, 1000), (3, 1000), (4, 10_000), (5, 100))},
    }
    with open(os.path.splitext(args.out)[0] + "_env.json", "w") as f:
        json.dump(env, f, indent=2, default=str)
    print("# wrote", args.out)


if __name__ == "__main__":
    main()
