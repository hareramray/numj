"""Generates differential test cases and NumPy reference results for numj.DiffTests.

Usage:  python -I difftest/gen_numpy_cases.py build/difftest      (scripts/difftest.ps1 does this)

Writes <out>/cases.json and one .npy file per expected array. Each case describes inputs (deterministic splitmix64
data, optionally with special values, shaped and then transformed by a pipeline of view operations), one operation,
and NumPy's outcome: result values, shape, strides and flags for views, whether the result shares memory with its
input, or the exception category NumPy raised. The Java side replays the same description against numj.

Comparison rules (implemented in DiffTests.java):
  * views, elementwise results and copies: bitwise (NaN == NaN), same shape;
  * reductions: |numj - numpy| <= (gamma(k_numj) + gamma(n - 1)) * sum|x| per output, where n is the number of
    summed terms. Both libraries satisfy |s - exact| <= gamma(k) * sum|x| for their own summation trees (numj's k is
    NumJ.summationDepth(n); n - 1 bounds any tree NumPy may use), so the difference is bounded by the sum of the two.
    Means divide both sides by n, which adds at most one rounding each (2u|mean|);
  * views: same strides (ignoring axes of length 1, whose stride is irrelevant), same C/F flags, and numj's
    view/copy behaviour matches np.shares_memory;
  * errors: NumPy ValueError/AxisError/IndexError map to Java IllegalArgumentException or IndexOutOfBoundsException
    as documented in docs/COMPATIBILITY.md; the case records the expected Java exception.
"""
import json
import os
import sys

import numpy as np

PINNED_NUMPY = "2.5.2"


def splitmix(seed: int, n: int) -> np.ndarray:
    """Bit-identical to numj.bench.Data.splitmix."""
    i = np.arange(1, n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed) + i * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return ((z >> np.uint64(11)).astype(np.float64) * 2.0**-53) * 2.0 - 1.0


SPECIALS = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 5e-324, -5e-324, 2.2250738585072014e-308,
                     1.7976931348623157e308, -1.7976931348623157e308, 1.0, -1.0, 3.0, 0.1])


def make_base(spec):
    n = int(np.prod(spec["shape"], dtype=np.int64)) if spec["shape"] else 1
    data = splitmix(spec["seed"], n)
    if spec.get("special"):
        # deterministic sprinkle of special values at every 3rd position
        for k, pos in enumerate(range(0, n, 3)):
            data[pos] = SPECIALS[(k + spec["seed"]) % len(SPECIALS)]
    return data.reshape(spec["shape"])


def parse_index(expr):
    """Same grammar as numj.Ix.parse."""
    expr = expr.strip()
    if not expr:
        return ()
    out = []
    for raw in expr.split(","):
        t = raw.strip()
        if t == "...":
            out.append(Ellipsis)
        elif t in ("None", "newaxis"):
            out.append(None)
        elif ":" not in t:
            out.append(int(t))
        else:
            p = t.split(":")
            out.append(slice(*[int(x) if x.strip() else None for x in p]))
    return tuple(out)


def apply_view(x, op):
    kind = op[0]
    if kind == "slice":
        return x[parse_index(op[1])]
    if kind == "T":
        return x.T
    if kind == "permute":
        return np.transpose(x, op[1])
    if kind == "swap":
        return np.swapaxes(x, op[1], op[2])
    if kind == "reshape":                          # numj reshape = view only = NumPy copy=False
        return np.reshape(x, op[1], copy=False)
    if kind == "bcast":
        return np.broadcast_to(x, op[1])
    raise ValueError(kind)


def build_inputs(case):
    bases, arrays = {}, {}
    for name, spec in case["bases"].items():
        bases[name] = make_base(spec)
    for name, spec in case["inputs"].items():
        x = bases[spec["base"]]
        for op in spec.get("view", []):
            x = apply_view(x, op)
        arrays[name] = x
    return bases, arrays


def operand(arrays, ref):
    return ref["scalar"] if isinstance(ref, dict) else arrays[ref]


def run_op(case, bases, arrays):
    op = case["op"]
    kind = op["kind"]
    if kind == "view":
        return arrays[op["of"]]
    if kind in ("add", "subtract", "multiply", "divide"):
        f = getattr(np, kind)
        a, b = operand(arrays, op["a"]), operand(arrays, op["b"])
        out = op.get("out")
        with np.errstate(all="ignore"):
            if out is None:
                return np.asarray(f(a, b))
            return f(a, b, out=arrays[out])
    if kind in ("sum", "mean"):
        f = getattr(np, kind)
        axes = op.get("axes")
        axis = None if axes is None else tuple(axes)
        out = op.get("out")
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if out is None:
                    return np.asarray(f(arrays[op["of"]], axis=axis, keepdims=op.get("keepdims", False)))
                return f(arrays[op["of"]], axis=axis, keepdims=op.get("keepdims", False), out=arrays[out])
    if kind == "copy":
        return arrays[op["of"]].copy(order="C")
    if kind == "flattenCopy":
        return arrays[op["of"]].flatten()
    if kind == "reshapeCopy":
        return np.array(np.reshape(arrays[op["of"]], op["shape"]), order="C", copy=True)
    raise ValueError(kind)


def java_error(exc):
    # AxisError derives from both ValueError and IndexError: numj reports axis problems as IllegalArgumentException
    if isinstance(exc, np.exceptions.AxisError):
        return "IllegalArgumentException"
    if isinstance(exc, IndexError):
        msg = str(exc)
        if "too many indices" in msg or "single ellipsis" in msg:
            return "IllegalArgumentException"
        return "IndexOutOfBoundsException"
    if isinstance(exc, ValueError):
        if "read-only" in str(exc):
            return "ReadOnlyArrayException"
        return "IllegalArgumentException"
    raise exc


# ------------------------------------------------------------------------------------------------ case generation

cases = []


def case(cid, bases, inputs, op, tags=()):
    cases.append({"id": cid, "bases": bases, "inputs": inputs, "op": op, "tags": list(tags)})


def base(seed, shape, special=False):
    return {"seed": seed, "shape": list(shape), "special": special}


def gen():
    seed = 1
    # --- views: slicing / transpose / permute / reshape / broadcast
    view_pipelines = [
        ((3, 4, 5), [["slice", "1:, ::-1, 0"]]),
        ((3, 4, 5), [["slice", "::2, None, ..., -1"]]),
        ((3, 4, 5), [["slice", "100:"]]),
        ((3, 4, 5), [["slice", "-100:, 1:3:-1"]]),
        ((3, 4, 5), [["slice", "..., 4:0:-2"]]),
        ((3, 4, 5), [["slice", "::-1, ::-1, ::-1"]]),
        ((3, 4, 5), [["slice", "2, 3, 4"]]),
        ((3, 4, 5), [["slice", "2, 3, 4, ..."]]),
        ((3, 4, 5), [["slice", "-1, :, -5"]]),
        ((3, 4, 5), [["slice", "1::-1, -2::-2"]]),
        ((3, 4, 5), [["slice", "None, None, ..., None"]]),
        ((7,), [["slice", "-3:-8:-1"]]),
        ((7,), [["slice", "::3"]]),
        ((7,), [["slice", "5:2"]]),
        ((), [["slice", "None"]]),
        ((), [["slice", "..."]]),
        ((3, 4, 5), [["T"]]),
        ((3, 4, 5), [["permute", [1, -1, 0]]]),
        ((3, 4, 5), [["swap", 0, -1], ["slice", "::-1"]]),
        ((4, 6), [["reshape", [2, -1, 3]]]),
        ((4, 6), [["slice", ":, ::2"], ["reshape", [12]]]),
        ((4, 6), [["slice", ":, ::2"], ["reshape", [2, 2, 3]]]),
        ((4, 6), [["slice", "::2"], ["reshape", [2, 2, 3]]]),
        ((4, 6), [["slice", ":, :3"], ["reshape", [12]]]),
        ((4, 6), [["T"], ["reshape", [24]]]),
        ((4, 6), [["T"], ["reshape", [6, 2, 2]]]),
        ((4, 6), [["slice", "None, :, None, 1:2"], ["reshape", [4]]]),
        ((2, 3, 4), [["permute", [2, 0, 1]], ["reshape", [4, 6]]]),
        ((2, 3, 4), [["slice", ":, ::-1"], ["reshape", [2, 12]]]),
        ((0, 3), [["reshape", [3, 0]]]),
        ((0, 3), [["reshape", [0, -1]]]),
        ((4, 6), [["reshape", [-1, -1]]]),
        ((4, 6), [["reshape", [5, 5]]]),
        ((3,), [["bcast", [4, 3]]]),
        ((3, 1), [["bcast", [2, 3, 5]]]),
        ((1,), [["bcast", [0]]]),
        ((3,), [["bcast", [4, 2]]]),
        ((3, 4), [["slice", "3"]]),
        ((3, 4), [["slice", ":, -5"]]),
        ((3, 4), [["slice", "0, 0, 0"]]),
        ((3, 4), [["slice", "..., ..."]]),
        ((3, 4), [["slice", "::0"]]),
        ((3, 4, 5), [["permute", [0, 0, 1]]]),
        ((3, 4, 5), [["permute", [0, 1, 3]]]),
    ]
    for shape, pipe in view_pipelines:
        case(f"view_{seed}", {"B": base(seed, shape)}, {"x": {"base": "B", "view": pipe}},
             {"kind": "view", "of": "x"}, ["view"])
        seed += 1

    # --- copies of strided views
    for shape, pipe in [((4, 5, 6), [["T"]]), ((4, 5, 6), [["slice", "::-1, 1::2"]]), ((3, 4), [["bcast", [2, 3, 4]]])]:
        for kind in ("copy", "flattenCopy"):
            case(f"{kind}_{seed}", {"B": base(seed, shape)}, {"x": {"base": "B", "view": pipe}},
                 {"kind": kind, "of": "x"}, ["copy"])
        seed += 1
    case(f"reshapeCopy_{seed}", {"B": base(seed, (4, 6))}, {"x": {"base": "B", "view": [["T"]]}},
         {"kind": "reshapeCopy", "of": "x", "shape": [3, -1]}, ["copy"])
    seed += 1

    # --- elementwise: layouts x broadcasting x specials
    layouts = {
        "C": lambda nd: [],
        "T": lambda nd: [["T"]],
        "rev": lambda nd: [["slice", ",".join(["::-1"] * nd)]] if nd else [],
        "step": lambda nd: [["slice", ",".join(["1::2"] * nd)]] if nd else [],
    }

    def base_shape(layout, shape):
        if layout == "T":
            return tuple(reversed(shape))
        if layout == "step":
            return tuple(2 * d + 1 for d in shape)
        return tuple(shape)

    pairs = [((), ()), ((5,), ()), ((3, 4), (4,)), ((3, 1), (1, 4)), ((2, 3, 4), (3, 1)), ((0, 3), (3,)),
             ((1,), (0,)), ((4, 1, 6), (5, 1)), ((40, 70), (70,)), ((33, 40), (33, 1)), ((1000,), (1000,)),
             ((6, 7, 8), (6, 1, 8))]
    lay_pairs = [("C", "C"), ("T", "rev"), ("step", "T"), ("rev", "step")]
    for op in ("add", "subtract", "multiply", "divide"):
        for k, (sa, sb) in enumerate(pairs):
            la, lb = lay_pairs[k % len(lay_pairs)]
            special = k % 3 == 0
            case(f"{op}_{seed}", {"A": base(seed, base_shape(la, sa), special), "B": base(seed + 1, base_shape(lb, sb), special)},
                 {"a": {"base": "A", "view": layouts[la](len(sa))}, "b": {"base": "B", "view": layouts[lb](len(sb))}},
                 {"kind": op, "a": "a", "b": "b"}, ["elementwise"])
            seed += 2
        for s in (2.5, -0.0, float("inf"), float("nan")):
            case(f"{op}_s_{seed}", {"A": base(seed, (4, 5), True)}, {"a": {"base": "A", "view": [["T"]]}},
                 {"kind": op, "a": "a", "b": {"scalar": s}}, ["elementwise", "scalar"])
            case(f"{op}_sl_{seed}", {"A": base(seed, (4, 5), True)}, {"a": {"base": "A", "view": [["slice", "::-1"]]}},
                 {"kind": op, "a": {"scalar": s}, "b": "a"}, ["elementwise", "scalar"])
            seed += 1
        # incompatible shapes
        case(f"{op}_bad_{seed}", {"A": base(seed, (2, 3)), "B": base(seed + 1, (4,))},
             {"a": {"base": "A"}, "b": {"base": "B"}}, {"kind": op, "a": "a", "b": "b"}, ["elementwise", "error"])
        seed += 2
        # overlapping out: in place, reversed, shifted, broadcast row of out
        case(f"{op}_inplace_{seed}", {"X": base(seed, (10, 20)), "Y": base(seed + 1, (10, 20))},
             {"xs": {"base": "X", "view": [["slice", "::-1, ::2"]]}, "ys": {"base": "Y", "view": [["slice", "::-1, ::2"]]}},
             {"kind": op, "a": "xs", "b": "ys", "out": "xs"}, ["overlap"])
        case(f"{op}_inplace_b_{seed}", {"X": base(seed, (10, 20)), "Y": base(seed + 1, (10, 20))},
             {"xs": {"base": "X", "view": [["T"]]}, "ys": {"base": "Y", "view": [["T"]]}},
             {"kind": op, "a": "ys", "b": "xs", "out": "xs"}, ["overlap"])
        case(f"{op}_self_{seed}", {"X": base(seed, (10, 20))}, {"x": {"base": "X", "view": [["slice", ":, 1::3"]]}},
             {"kind": op, "a": "x", "b": "x", "out": "x"}, ["overlap"])
        case(f"{op}_rev_{seed}", {"X": base(seed, (200,))},
             {"x": {"base": "X"}, "r": {"base": "X", "view": [["slice", "::-1"]]}},
             {"kind": op, "a": "r", "b": "x", "out": "x"}, ["overlap"])
        case(f"{op}_shift_{seed}", {"X": base(seed, (200,))},
             {"lo": {"base": "X", "view": [["slice", ":-1"]]}, "hi": {"base": "X", "view": [["slice", "1:"]]}},
             {"kind": op, "a": "lo", "b": "hi", "out": "hi"}, ["overlap"])
        case(f"{op}_bcastrow_{seed}", {"M": base(seed, (10, 20))},
             {"m": {"base": "M"}, "r": {"base": "M", "view": [["slice", "0"]]}},
             {"kind": op, "a": "m", "b": "r", "out": "m"}, ["overlap"])
        case(f"{op}_ro_{seed}", {"M": base(seed, (3,)), "N": base(seed + 1, (2, 3))},
             {"m": {"base": "M"}, "o": {"base": "M", "view": [["bcast", [2, 3]]]}, "n": {"base": "N"}},
             {"kind": op, "a": "n", "b": "m", "out": "o"}, ["error"])
        seed += 2

    # --- reductions
    red_cases = [
        ((), None), ((7,), [0]), ((3, 4), [0]), ((3, 4), [1]), ((3, 4), [-1]), ((3, 4), None), ((3, 4), [1, 0]),
        ((2, 3, 4), [0, 2]), ((2, 3, 4), [1]), ((2, 3, 4), [-1, 0]), ((2, 3, 4), [2, 1, 0]), ((2, 1, 3), [1]),
        ((2, 3, 4), []), ((4, 0, 3), [1]), ((4, 0, 3), [0]), ((0,), None), ((5000, 7), [0]), ((6, 9000), [1]),
        ((3000, 400), None), ((300, 600), [0]), ((20, 30, 40), [0, 2]), ((3, 4), [2]), ((3, 4), [0, -2]),
        ((3, 4), [-3]),
    ]
    for kind in ("sum", "mean"):
        for k, (shape, axes) in enumerate(red_cases):
            lay = ["C", "T", "rev", "step"][k % 4]
            big = int(np.prod(shape)) > 100_000 if shape else False
            for keep in (False, True):
                if big and keep:
                    continue
                case(f"{kind}_{seed}", {"A": base(seed, base_shape(lay, shape), k % 5 == 0 and not big)},
                     {"x": {"base": "A", "view": layouts[lay](len(shape))}},
                     {"kind": kind, "of": "x", "axes": axes, "keepdims": keep}, ["reduction"])
                seed += 1
        case(f"{kind}_negzero_{seed}", {"A": base(seed, (3, 5))}, {"x": {"base": "A"}},
             {"kind": kind, "of": "x", "axes": [0], "keepdims": False, "negzero": True}, ["reduction"])
        seed += 1
        # out overlapping the input
        case(f"{kind}_outov_{seed}", {"A": base(seed, (3, 4, 5))},
             {"x": {"base": "A"}, "o": {"base": "A", "view": [["slice", ":, 0, :"]]}},
             {"kind": kind, "of": "x", "axes": [1], "keepdims": False, "out": "o"}, ["reduction", "overlap"])
        seed += 1


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("build", "difftest")
    if np.__version__ != PINNED_NUMPY:
        print(f"WARNING: NumPy {np.__version__} != pinned {PINNED_NUMPY}; results may differ", file=sys.stderr)
    os.makedirs(out_dir, exist_ok=True)
    gen()
    for c in cases:
        bases, arrays = build_inputs_safe(c)
        if bases is None:
            continue
        if c["op"].get("negzero"):
            bases["A"][...] = -0.0
        try:
            r = run_op(c, bases, arrays)
            fname = c["id"] + ".npy"
            res = np.asarray(r)
            # NumPy returns a scalar (a copy) when every axis gets an integer index; numj returns a 0-d view.
            numpy_scalar = isinstance(r, np.generic)
            np.save(os.path.join(out_dir, fname), np.array(res, order="C"), allow_pickle=False)
            exp = {"status": "ok", "file": fname, "shape": list(res.shape)}
            if c["op"]["kind"] == "view":
                exp["strides"] = list(res.strides)
                exp["c"] = bool(res.flags.c_contiguous)
                exp["f"] = bool(res.flags.f_contiguous)
                exp["writeable"] = bool(res.flags.writeable)
            src = c["op"].get("of")
            if src is not None:
                ref = bases[c["inputs"][src]["base"]] if c["op"]["kind"] == "view" else arrays[src]
                exp["shares"] = bool(np.shares_memory(res, ref))
                exp["numpy_scalar"] = numpy_scalar
            if c["op"]["kind"] in ("sum", "mean"):
                axes = c["op"].get("axes")
                x = arrays[c["op"]["of"]]
                if "out" in c["op"]:
                    x = make_input_snapshot(c)
                ax = None if axes is None else tuple(axes)
                with np.errstate(all="ignore"):
                    absum = np.asarray(np.sum(np.abs(x), axis=ax, keepdims=c["op"].get("keepdims", False)))
                np.save(os.path.join(out_dir, c["id"] + ".abs.npy"), np.array(absum, order="C"), allow_pickle=False)
                exp["absfile"] = c["id"] + ".abs.npy"
                exp["count"] = int(np.prod([x.shape[a] for a in range(x.ndim)] if ax is None else
                                           [x.shape[a] for a in ax], dtype=np.int64)) if x.ndim else 1
            # memory written through "out": record the whole base afterwards
            if c["op"].get("out") is not None:
                bname = c["inputs"][c["op"]["out"]]["base"]
                np.save(os.path.join(out_dir, c["id"] + ".base.npy"), bases[bname], allow_pickle=False)
                exp["basefile"] = c["id"] + ".base.npy"
                exp["outbase"] = bname
        except Exception as e:
            exp = {"status": "error", "java": java_error(e), "numpy": type(e).__name__, "message": str(e)}
        c["expect"] = exp
    meta = {"numpy": np.__version__, "python": sys.version.split()[0], "cases": [c for c in cases if "expect" in c]}
    with open(os.path.join(out_dir, "cases.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    n_err = sum(1 for c in meta["cases"] if c["expect"]["status"] == "error")
    print(f"wrote {len(meta['cases'])} cases ({n_err} expected errors) to {out_dir} with NumPy {np.__version__}")


def make_input_snapshot(c):
    bases, arrays = build_inputs(c)
    return arrays[c["op"]["of"]]


def build_inputs_safe(c):
    try:
        return build_inputs(c)
    except Exception as e:
        # the view pipeline itself fails in NumPy: that is the expected outcome
        c["expect"] = {"status": "error", "java": java_error(e), "numpy": type(e).__name__, "message": str(e)}
        return None, None


if __name__ == "__main__":
    main()
