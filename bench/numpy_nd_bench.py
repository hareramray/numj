"""Python side of the numj n-d benchmarks (milestone 1): same workloads, shapes and input bits as the JMH
benchmarks in java/jmh/numj (ElementwiseBench, ReduceBench, FusedBench).

Implementations (all single-threaded):
  numpy          ordinary expressions; results and temporaries allocated per call
  numpy-out      ufuncs with reusable out= buffers
  numexpr        numexpr.evaluate(..., out=...) with numexpr.set_num_threads(1)
  numba          @njit loops (compilation excluded from timings; first-call time recorded separately)

Protocol (same as bench/numpy_bench.py): warm up 0.5 s, calibrate repetitions so one sample lasts >= 20 ms, take
21 samples; report median, p10, p90 per call. Every result is checked against NumPy's (elementwise: bitwise;
sums: relative 1e-12) before timing.

Usage: python -I bench/numpy_nd_bench.py --out results/nd/python.csv [--quick]
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "NUMEXPR_MAX_THREADS", "NUMBA_NUM_THREADS"):
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

try:
    import numexpr as ne
    ne.set_num_threads(1)
except ImportError:  # optional
    ne = None
try:
    import numba
    from numba import njit
except ImportError:  # optional
    numba = None

WARMUP_S, SAMPLE_S, SAMPLES = 0.5, 0.020, 21
DIMS = {16: (4, 4), 1000: (25, 40), 100_000: (250, 400), 10_000_000: (2500, 4000)}


def splitmix(seed: int, n: int) -> np.ndarray:
    """Bit-identical to numj.bench.Data.splitmix."""
    i = np.arange(1, n + 1, dtype=np.uint64)
    z = np.uint64(seed) + i * np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return ((z >> np.uint64(11)).astype(np.float64) * 2.0**-53) * 2.0 - 1.0


rows_out = []
compile_times = {}


def run(suite, kernel, layout, n, impl, op):
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
    rows_out.append((suite, kernel, layout, n, impl, per, reps, result))
    s = sorted(per)
    print(f"{suite:8s} {kernel:12s} {layout:10s} {n:>9d} {impl:12s} median {statistics.median(s):14.1f} ns  "
          f"cv {100 * statistics.stdev(s) / statistics.fmean(s):5.1f}%", flush=True)


def pct(s, p):
    idx = p * (len(s) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def same_bits(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return x.shape == y.shape and np.array_equal(x.view(np.uint64), y.view(np.uint64))


# ------------------------------------------------------------------------------------------------ numba kernels

if numba is not None:
    @njit(cache=False)
    def nb_add1(a, b, o):
        for i in range(o.shape[0]):
            o[i] = a[i] + b[i]

    @njit(cache=False)
    def nb_add2(a, b, o):
        for i in range(o.shape[0]):
            for j in range(o.shape[1]):
                o[i, j] = a[i, j] + b[i, j]

    @njit(cache=False)
    def nb_addrow(a, row, o):
        for i in range(o.shape[0]):
            for j in range(o.shape[1]):
                o[i, j] = a[i, j] + row[j]

    @njit(cache=False)
    def nb_sum(a):
        s = 0.0
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                s += a[i, j]
        return s

    @njit(cache=False)
    def nb_sum_axis0(a, o):
        o[:] = 0.0
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                o[j] += a[i, j]

    @njit(cache=False)
    def nb_sum_axis1(a, o):
        for i in range(a.shape[0]):
            s = 0.0
            for j in range(a.shape[1]):
                s += a[i, j]
            o[i] = s

    @njit(cache=False)
    def nb_muladd(a, b, c):
        s = 0.0
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                t = a[i, j] * b[i, j] + c[i, j]
                s += t * t
        return s


def timed_first_call(name, f, *args):
    t0 = time.perf_counter()
    f(*args)
    compile_times.setdefault(name, []).append(time.perf_counter() - t0)


# ------------------------------------------------------------------------------------------------ suites

def elementwise():
    for n in (16, 1000, 100_000, 10_000_000):
        r, c = DIMS[n]
        for layout in ("contig", "transposed", "stepped", "bcast"):
            if layout == "contig":
                a, b = splitmix(1, n), splitmix(2, n)
                o = np.empty(n)
                nb_args = (nb_add1, a, b, o) if numba else None
            elif layout == "transposed":
                a, b = splitmix(1, n).reshape(c, r).T, splitmix(2, n).reshape(c, r).T
                o = np.empty((r, c))
                nb_args = (nb_add2, a, b, o) if numba else None
            elif layout == "stepped":
                a, b = splitmix(1, 2 * n)[::2], splitmix(2, 2 * n)[::2]
                o = np.empty(n)
                nb_args = (nb_add1, a, b, o) if numba else None
            else:
                a, b = splitmix(1, n).reshape(r, c), splitmix(2, c)
                o = np.empty((r, c))
                nb_args = (nb_addrow, a, b, o) if numba else None
            ref = a + b
            run("ew", "add", layout, n, "numpy", lambda: a + b)
            run("ew", "add", layout, n, "numpy-out", lambda: np.add(a, b, out=o))
            if ne is not None:
                ne.evaluate("a + b", local_dict={"a": a, "b": b}, out=o)
                assert same_bits(o, ref), "numexpr add differs"
                run("ew", "add", layout, n, "numexpr", lambda: ne.evaluate("a + b", local_dict={"a": a, "b": b}, out=o))
            if numba is not None:
                f, x, y, z = nb_args
                timed_first_call(f.__name__ + ":" + layout, f, x, y, z)
                assert same_bits(z, ref), "numba add differs"
                run("ew", "add", layout, n, "numba", lambda: (f(x, y, z), z)[1])
            del a, b, o, ref


def reductions():
    for n in (1000, 100_000, 10_000_000):
        r, c = DIMS[n]
        A = splitmix(1, n).reshape(r, c)
        o0, o1 = np.empty(c), np.empty(r)
        for kind in ("all", "axis0", "axis1", "allT"):
            if kind == "all":
                run("reduce", "sum", kind, n, "numpy", lambda: np.sum(A))
                if numba:
                    timed_first_call("nb_sum", nb_sum, A)
                    run("reduce", "sum", kind, n, "numba", lambda: nb_sum(A))
            elif kind == "axis0":
                run("reduce", "sum", kind, n, "numpy", lambda: np.sum(A, axis=0))
                run("reduce", "sum", kind, n, "numpy-out", lambda: np.sum(A, axis=0, out=o0))
                if numba:
                    timed_first_call("nb_sum_axis0", nb_sum_axis0, A, o0)
                    run("reduce", "sum", kind, n, "numba", lambda: (nb_sum_axis0(A, o0), o0)[1])
            elif kind == "axis1":
                run("reduce", "sum", kind, n, "numpy", lambda: np.sum(A, axis=1))
                run("reduce", "sum", kind, n, "numpy-out", lambda: np.sum(A, axis=1, out=o1))
                if numba:
                    timed_first_call("nb_sum_axis1", nb_sum_axis1, A, o1)
                    run("reduce", "sum", kind, n, "numba", lambda: (nb_sum_axis1(A, o1), o1)[1])
            else:
                AT = A.T
                run("reduce", "sum", kind, n, "numpy", lambda: np.sum(AT))
                if numba:
                    timed_first_call("nb_sum:T", nb_sum, AT)
                    run("reduce", "sum", kind, n, "numba", lambda: nb_sum(AT))


def fused():
    for n in (1000, 100_000, 10_000_000):
        r, c = DIMS[n]
        for layout in ("contig", "transposed"):
            A, B, C = (splitmix(s, n).reshape(r, c) for s in (1, 2, 3))
            if layout == "transposed":
                a, b, cc = A.T, B.T, C.T
            else:
                a, b, cc = A, B, C
            t = np.empty(a.shape)
            ref = np.sum((a * b + cc) ** 2)
            run("fused", "muladd", layout, n, "numpy", lambda: np.sum((a * b + cc) ** 2))

            def reuse():
                np.multiply(a, b, out=t)
                np.add(t, cc, out=t)
                np.multiply(t, t, out=t)
                return np.sum(t)
            run("fused", "muladd", layout, n, "numpy-out", reuse)
            if ne is not None:
                v = ne.evaluate("sum((a * b + c) ** 2)", local_dict={"a": a, "b": b, "c": cc})
                assert abs(v - ref) <= 1e-12 * abs(ref), "numexpr fused differs"
                run("fused", "muladd", layout, n, "numexpr",
                    lambda: ne.evaluate("sum((a * b + c) ** 2)", local_dict={"a": a, "b": b, "c": cc}))
            if numba is not None:
                timed_first_call("nb_muladd:" + layout, nb_muladd, a, b, cc)
                assert abs(nb_muladd(a, b, cc) - ref) <= 1e-12 * abs(ref), "numba fused differs"
                run("fused", "muladd", layout, n, "numba", lambda: nb_muladd(a, b, cc))
            del A, B, C, a, b, cc, t


def short_rows():
    """Same as java/jmh/numj/ShortRowBench: 10^6 elements in rows of k."""
    N = 1_000_000
    for k in (2, 4, 8, 16, 64):
        rows = N // k
        M = splitmix(1, N).reshape(rows, k)
        o = np.empty(rows)
        X = splitmix(2, rows * (k + 1)).reshape(rows, k + 1)
        Y = splitmix(3, rows * (k + 1)).reshape(rows, k + 1)
        xv, yv = X[:, :k], Y[:, :k]
        out = np.empty((rows, k))
        run("short", "sumRows", f"k={k}", N, "numpy-out", lambda: np.sum(M, axis=1, out=o))
        run("short", "addRows", f"k={k}", N, "numpy-out", lambda: np.add(xv, yv, out=out))
        if ne is not None:
            run("short", "addRows", f"k={k}", N, "numexpr",
                lambda: ne.evaluate("x + y", local_dict={"x": xv, "y": yv}, out=out))
        if numba is not None:
            timed_first_call("nb_sum_axis1:short", nb_sum_axis1, M, o)
            run("short", "sumRows", f"k={k}", N, "numba", lambda: (nb_sum_axis1(M, o), o)[1])
            timed_first_call("nb_add2:short", nb_add2, xv, yv, out)
            run("short", "addRows", f"k={k}", N, "numba", lambda: (nb_add2(xv, yv, out), out)[1])


def main():
    global WARMUP_S, SAMPLE_S, SAMPLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--suites", default="ew,reduce,fused,short")
    args = ap.parse_args()
    if args.quick:
        WARMUP_S, SAMPLE_S, SAMPLES = 0.05, 0.005, 5
    suites = args.suites.split(",")
    if "ew" in suites:
        elementwise()
    if "reduce" in suites:
        reductions()
    if "fused" in suites:
        fused()
    if "short" in suites:
        short_rows()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("suite,kernel,layout,n,impl,median_ns,min_ns,p10_ns,p90_ns,mean_ns,stdev_ns,samples,reps,result".split(","))
        for suite, kernel, layout, n, impl, per, reps, result in rows_out:
            s = sorted(per)
            w.writerow([suite, kernel, layout, n, impl, f"{statistics.median(s):.3f}", f"{s[0]:.3f}",
                        f"{pct(s, 0.1):.3f}", f"{pct(s, 0.9):.3f}", f"{statistics.fmean(s):.3f}",
                        f"{statistics.stdev(s):.3f}", len(s), reps, repr(result)])

    buf = io.StringIO()
    with redirect_stdout(buf):
        np.show_runtime()
    env = {
        "python": sys.version,
        "numpy": np.__version__,
        "numexpr": getattr(ne, "__version__", None),
        "numexpr_threads": ne.get_num_threads() if ne else None,
        "numexpr_vml": getattr(ne, "use_vml", None),
        "numba": getattr(numba, "__version__", None),
        "llvmlite": __import__("llvmlite").__version__ if numba else None,
        "numba_compile_seconds": compile_times,
        "numpy_config": np.show_config(mode="dicts"),
        "numpy_runtime": buf.getvalue(),
        "platform": platform.platform(),
        "thread_env": {v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                                                      "NUMBA_NUM_THREADS")},
    }
    with open(os.path.splitext(args.out)[0] + "_env.json", "w") as f:
        json.dump(env, f, indent=2, default=str)
    print("# wrote", args.out)


if __name__ == "__main__":
    main()
