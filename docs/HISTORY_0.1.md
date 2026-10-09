# numj 0.1 — benchmark narrative (historical)

> **Historical record.** This is the benchmark section of the numj 0.1 README (2026-10-09), moved here unchanged when
> milestone 1 was added. It describes the 0.1 kernels and the custom harness of `bench/`, measured before the
> n-d work. These numbers were **not re-measured** for milestone 1; treat them as evidence recorded at the time.
> Two statements are superseded: roadmap items 2 (Java path for tiny inputs) and 6 (JMH) are implemented in 0.2, and
> critical downcalls are no longer the default (see [MIGRATION.md](MIGRATION.md)). Raw data: [`results/RESULTS.md`](../results/RESULTS.md),
> [`results/run2/RESULTS.md`](../results/run2/RESULTS.md), [`results/RUNS_COMPARED.md`](../results/RUNS_COMPARED.md).

## Results in one paragraph (0.1)

Single-threaded, with the same input bits, on an i5-13420H pinned to its P-cores: **numj beats NumPy in 14 of 15
kernel/size cases in both runs.** It is about 2–6.5× faster than NumPy written with reused output buffers for 10⁵–10⁷
elements, about 13–16× faster at 10³, and ~100× faster at 16 elements, where NumPy's per-call overhead dominates. The 15th case was disturbed by
system noise in run 2 and counts as inconclusive. **Compared with plain Java loops, the result depends on size.** numj
is 4–6× faster when the data fits in L1 (10³ elements), 1.6–3× faster at 10⁵, and **tied at 10⁷**,
where every implementation is limited by DRAM bandwidth (~17 GB/s on one core). numj is **slower** for very small
inputs (16 elements: 0.63–0.70×, where the ~8 ns Java→Fortran call costs about as much as Java's whole loop) and for **short-row
normalisation** (4- or 10-element rows: 0.34–0.74×). End-to-end calls that copy Java `double[]` data into fresh native memory
every time lose to plain Java at every size. The library only pays off when data stays in `F64Array`.

## Benchmarks (0.1 kernels)

### Methodology

* **Same inputs.** Java and Python both generate data with splitmix64 → `[-1, 1)`, and the fingerprints are checked to be
  bit-identical in `RESULTS.md`. All arithmetic is float64, and all implementations run single-threaded (NumPy's BLAS is forced to 1 thread
  through environment variables; NumPy's elementwise ufuncs are single-threaded anyway).
* **Pinning.** Both processes are pinned to logical CPUs 0–7, which are the four P-cores. Unpinned runs landed on E-cores
  and were up to 6× slower and much noisier for division-heavy code, so unpinned numbers are not comparable.
* **Protocol (both harnesses).** Warm-up (Java 1.5 s per case, so C2 has finished; Python 0.5 s), then calibrate the number of
  repetitions so each sample lasts ≥ 20 ms, then take 21 samples. Reported per call: median, p10–p90, mean, stdev, CV.
  Every Java result goes into a sink that is printed at the end, so dead-code elimination cannot remove work. The JVM runs with
  `-Xms4g -Xmx4g -XX:+AlwaysPreTouch`.
* **Implementations.** `numj` (compute: data already in `F64Array`), `numj e2e` (from Java `double[]`: allocate,
  copy in, compute, copy out, free), `java-naive` (straightforward loops), `java-blocked` (same summation order as
  numj, hand-unrolled 16 lanes, bit-identical results), `numpy` (ordinary expressions such as `np.sum((a-b)**2)`),
  `numpy-reuse` (`out=` buffers, plus `np.dot` for the final sum), and for the batched kernel `numj per-row calls` (one
  native call per row).
* **Verdicts.** numj **wins** only if its p90 is below the other implementation's p10. Overlapping ranges count as a tie. A verdict is
  **robust** only if it is the same in both independent runs (`bench/compare_runs.py`).
* **Correctness of the benchmarked values.** Every case's final result is compared across all implementations, and all agree
  within their rounding-error bounds in both runs.
* **Environment record.** CPU, cache sizes, RAM, OS build, power plan, AC power, affinity, JDK, gfortran version, the flags compiled into the DLL,
  Python/NumPy versions, NumPy's BLAS (scipy-openblas 0.3.34) and SIMD dispatch (baseline X86_V2, dispatched
  X86_V3): `results/environment.json`, `results/numpy_main_env.json`, and the header of `RESULTS.md`.

### Selected results (run 1; run 2 in brackets where informative)

Median per call. `X/numj` > 1 means numj is faster.

| kernel | size | numj | java-naive | numpy-reuse | numpy | java-naive / numj | numpy-reuse / numj |
|---|---|---|---|---|---|---|---|
| sqdist | 16 | 9.6 ns | 6.7 ns | 930 ns | 2.23 µs | 0.70 [0.65] | 97 [87] |
| sqdist | 10³ | 73 ns | 438 ns | 1.10 µs | 2.62 µs | 6.0 [5.5] | 15 [13] |
| sqdist | 10⁵ | 16.4 µs | 47.7 µs | 47.3 µs | 522 µs | 2.9 [3.0] | 2.9 [3.1] |
| sqdist | 10⁷ | 9.32 ms | 10.5 ms | 23.8 ms | 53.7 ms | 1.13 tie [1.47 tie] | 2.6 [2.2] |
| sum((a·b+c)²) | 10³ | 101 ns | 438 ns | 1.63 µs | 3.32 µs | 4.3 [4.2] | 16 [15] |
| sum((a·b+c)²) | 10⁷ | 14.3 ms | 14.0 ms | 43.2 ms | 94.0 ms | 0.98 tie [disturbed] | 3.0 [disturbed] |
| normalize rows | 100×10 | 2.97 µs | 1.13 µs | 8.56 µs | 9.82 µs | **0.38** [0.34] | 2.9 [2.0] |
| normalize rows | 1000×100 | 52 µs | 71 µs | 120 µs | 256 µs | 1.36 [1.12] | 2.3 [2.3] |
| normalize rows | 100000×100 | 14.3 ms | 14.6 ms | 37.6 ms | 65.2 ms | 1.02 tie [0.95 tie] | 2.6 [2.3] |
| sqdist to rows | 1000×16 | 5.3 µs | 5.9 µs | 25.9 µs | 21.9 µs | 1.11 tie [1.15 tie] | 4.9 [3.8] |
| sqdist to rows | 10000×100 | 290 µs | 630 µs | 1.70 ms | 5.18 ms | 2.2 [1.6] | 5.9 [6.5] |

Further measurements (all in `results/RESULTS.md`):

* **Call overhead.** Calling `NumJ.sqdist` with n = 0, validation included, takes 8.2 ns with a critical downcall and 11.9 ns with a regular one.
* **Batching.** At 1000 rows × 16, one batched call takes 5.3 µs, while 1000 separate calls take 9.6 µs.
* **End-to-end from `double[]`.** For sqdist at 10⁵, the e2e path takes 116 µs, compute alone takes 16 µs, and plain Java takes 48 µs. Each fresh
  native allocation is new OS pages that fault on first touch, then get zeroed and filled. NumPy's ordinary expressions pay
  the same Windows cost for their temporaries: 522 µs vs 47 µs with reused buffers.
* **Vectorization profiles.** These use the same source and strict semantics.
  * SSE2 (128-bit) is about 2× slower than AVX2 at 10³ elements and equal from 10⁵ up, because those sizes are memory-bound.
  * Turning auto-vectorization off is 2.7–11× slower at 10³–10⁵ elements.
  * The opt-in `-ffast-math` build is **not consistently faster**. It is 1.5× *slower* at 10³ and 9× slower for batched short rows,
    and faster in one cell (normalize 1000×100, 0.63×, which is within the run-to-run drift). It also breaks the contract.
    We did not investigate why it is slower. There is no measured reason to enable it.
* **Threads (OpenMP inside the DLL).**
  * Streaming kernels gain little. At 10⁷ elements, `sqdist` reaches 1.39× with 8 threads; at 2.6·10⁵, 2 threads are *slower* than 1, which shows the 2¹⁸-element threshold is too low.
  * Compute-heavier, cache-resident work scales. `normalize_rows` on 10⁴×100 reaches 5.5× with 8 threads.
  * The 1-thread baseline of this suite was noisier and slower than the main run, so read these as indicative only.
* **Run-to-run drift.** For reductions, the same build differs by ≤ 1.17× between runs. For `normalize_rows` and `sqdist_rows` at medium and large sizes it differs by up to 1.9×.
  Run 2's 10⁷-element section was disturbed: CV was 40–55% for *every* implementation, including plain Java, and numj's minimum
  sample was 5× below its median. Those cells are reported as inconclusive.

## Where numj wins, ties, or loses, and why

| situation | verdict | reason (supported by the measurements above) |
|---|---|---|
| vs NumPy, any size | **wins** (14/15 robust) | Three effects. (1) Fixed cost per call: NumPy's per-call cost, made of interpreter, ufunc dispatch and temporaries, is ~1–2 µs, while a numj call is ~10 ns, which dominates up to ~10³ elements. (2) Temporaries: ordinary expressions allocate n-sized arrays, which on Windows means fresh pages and page faults (up to 11× slower than reusing buffers at 10⁵). (3) Memory traffic: even with `out=` buffers, NumPy writes a temporary and reads it back. That is 32 B/element for sqdist versus numj's single fused pass at 16 B/element, which matches the ~2.2–2.6× gap at 10⁷. NumPy's arithmetic itself is SIMD (AVX2-dispatched), so this is **not** "Fortran computes faster than NumPy". |
| vs plain Java, data in L1/L2 (10³–10⁵) | **wins** 1.6–6× | HotSpot C2 does not vectorize floating-point reductions, because it must keep the source's left-to-right order. The naive loop is therefore limited by add latency, and hand-unrolled scalar Java (`java-blocked`, the same order as numj) is limited by scalar throughput. numj's fixed 16-lane order lets GCC use four 256-bit accumulators. The gain comes from **SIMD plus a defined summation order, not from Fortran as such**: the SSE2 build halves it and the no-vectorization build loses it. |
| vs plain Java, 10⁷ elements | **ties** | All implementations stream at ~15–17 GB/s from DRAM on one core, so arithmetic speed does not matter. |
| vs plain Java, ≤ 16 elements | **loses** (0.63–0.70×) | The ~8 ns downcall costs about as much as Java's whole 16-element loop. |
| short rows (4–10 columns) | **loses** (0.34–0.74×) | numj has a fixed cost of ~20–25 ns per row: tail handling, which costs ~12 ns at n=17 vs n=16 in the overhead table, plus the norm and division setup. Java's simple loop has almost none. |
| e2e from `double[]` | **loses** to Java at every size | Copying into fresh native memory costs more than the computation. Keep data resident in `F64Array`. |
| batching | helps | 1.8× fewer ns per row than separate calls at 1000×16. |

## Roadmap (ordered by measured impact)

1. **Short rows and tails.** This is the largest robust loss. Specialise the row kernels for small `ncols` (masked
   loads in a separate, never-inlined routine, or fixed-width paths for common widths) to remove the ~12 ns
   tail stall per row. Target: match Java on 100×10. Notes from failed attempts are in *Engineering notes* below.
2. **Pure-Java path for tiny inputs.** For n below ~32, run the Java kernel in-process instead of calling native code.
   `JavaKernels.*Blocked` already produces **bit-identical** results, so this keeps the determinism guarantee while removing
   the 0.63–0.70× loss.
3. **Allocation cost.** Pool arenas or reuse buffers, and offer a non-zeroing allocation path. Measure `copyFrom` with
   pre-touched memory. End-to-end numbers are currently dominated by page faults and zeroing, not by the kernels.
4. **Thread policy from measurements.** Use per-kernel parallel thresholds instead of one global 2¹⁸. Streaming kernels need
   ≳10⁷ elements to gain even 1.4×, while `normalize_rows` scales from ~10⁶. Add a P-core-aware schedule on hybrid CPUs, and
   document `OMP_WAIT_POLICY=PASSIVE`. Keep the default at 1 thread.
5. **A stronger Java baseline.** Compare with the JDK Vector API (`jdk.incubator.vector`). It can express the same
   16-lane order in pure Java and may close the in-cache gap, which would answer whether native code is needed at all
   for these kernels.
6. **Benchmark rigour.** Add a JMH cross-check of the custom harness, run on the *High performance* power plan or a
   fixed-frequency Linux machine, use ≥ 3 independent runs, and report confidence intervals.
7. **Scope.** Add a float32 variant, more fused reductions, an optional compensated (Kahan/Neumaier) mode, and
   Linux/macOS builds (the code is portable, but only Windows was built and tested). For linear algebra, bind a tuned BLAS/LAPACK
   (for example the scipy-openblas library NumPy already ships) through the FFM API instead of writing kernels by hand.

## Engineering notes (what was tried, with measurements)

* **Unpinned runs** were 1.5–6× slower on division-heavy code, with CV up to 37%. Windows scheduled the thread on E-cores,
  so all runs are now pinned.
* **Per-row overhead.** Removing the generic block machinery from the single-block path, and replacing internal procedures
  with plain module procedures, made batched 1000×16 2× faster (26 → 8 µs in quick mode).
* **Tail handling.** Accumulators held in registers but then spilled for a variable-index tail: 8 µs (1000×16, quick mode). A tail padded with
  `+0` (the current version) gives 4.1 µs. A predicated masked-load tail cut the tail cost to ~2 ns, but made GCC
  vectorize the *main* loop across iterations with shuffles, which was 6× slower at n = 64. `!GCC$ novector` on the main loop
  then made it fully scalar. Both were reverted. The current code was checked in the assembly: no shuffles, `ymm`
  arithmetic, accumulators in registers.
* **A quick-mode anomaly.** In quick mode, `sumsq_muladd [16]` took 33 ns. That was JIT state: the case runs second, right after the benchmark's call site
  becomes polymorphic, with only 200 ms of warm-up. A direct probe measured 10.7 ns, and the full run measured 12.1 ns.

