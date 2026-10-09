"""Compares numj's verdicts across independent benchmark runs.

Usage: python -I bench/compare_runs.py results results/run2 [...]  -> writes <first dir>/RUNS_COMPARED.md

A verdict (win/tie/lose, non-overlapping p10-p90 rule from report.py) is called ROBUST only if it is the same
in every run; otherwise it is reported as 'mixed' and should be read as a tie.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report  # noqa: E402  (reuses the CSV loader)

OTHERS = [("numpy-reuse", "compute"), ("numpy", "e2e"), ("java-naive", "compute"), ("java-blocked", "compute")]


def verdict(n, o):
    if n["p90_ns"] < o["p10_ns"]:
        return "win"
    if n["p10_ns"] > o["p90_ns"]:
        return "lose"
    return "tie"


def main(dirs):
    runs = []
    for d in dirs:
        rows = report.load(os.path.join(d, "java_main.csv")) + report.load(os.path.join(d, "numpy_main.csv"))
        runs.append({(r["kernel"], r["shape"], r["impl"], r["mode"]): r for r in rows})
    keys = []
    for (k, s, impl, mode) in runs[0]:
        if impl == "numj" and mode == "compute" and (k, s) not in keys:
            keys.append((k, s))
    out = ["# numj verdicts across independent runs\n",
           f"Runs: {', '.join(dirs)}. Cell = `X / numj` median ratio per run, then the verdict. "
           "**robust** = same verdict in every run.\n",
           "| kernel | shape | " + " | ".join(f"vs {o}" for o, _ in OTHERS) + " |",
           "|---|---|" + "---|" * len(OTHERS)]
    summary = {}
    for k, s in keys:
        cells = []
        for o, mode in OTHERS:
            ratios, verdicts = [], []
            for run in runs:
                n, x = run.get((k, s, "numj", "compute")), run.get((k, s, o, mode))
                if n and x:
                    ratios.append(x["median_ns"] / n["median_ns"])
                    verdicts.append(verdict(n, x))
            if not ratios:
                cells.append("—")
                continue
            v = verdicts[0] if len(set(verdicts)) == 1 else "mixed"
            label = f"**{v}** (robust)" if v != "mixed" and len(verdicts) == len(runs) else v
            summary.setdefault(o, {}).setdefault(v, 0)
            summary[o][v] += 1
            cells.append(" / ".join(f"{r:.2f}×" for r in ratios) + f" {label}")
        out.append(f"| {k} | {s.replace(',', '×')} | " + " | ".join(cells) + " |")
    out.append("\n## Tally\n")
    for o, counts in summary.items():
        out.append(f"- vs **{o}**: " + ", ".join(f"{v}: {c}" for v, c in sorted(counts.items())))
    path = os.path.join(dirs[0], "RUNS_COMPARED.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 2 else ["results", "results/run2"])
