# numj — a small Java float64 library backed by Fortran

Java 25 (Foreign Function & Memory API) → Fortran 2018 (`ISO_C_BINDING`, `bind(C)`) kernels on contiguous
`float64` arrays in reusable native memory. The scope is deliberately small: three kernels plus one batched
variant, a memory model, a numerical contract, tests, and reproducible benchmarks against plain Java and NumPy.

**Status (verified on this machine, 2026-10-09):** builds from a clean checkout with the pinned toolchain; 20/20
correctness tests pass for the default, SSE2 and no-vectorization builds; benchmarks ran twice, independently.
Measured results are in [`results/RESULTS.md`](results/RESULTS.md) (run 1, all suites),
[`results/run2/RESULTS.md`](results/run2/RESULTS.md) (repeat of the main and NumPy suites), and
[`results/RUNS_COMPARED.md`](results/RUNS_COMPARED.md) (verdicts across both runs).

### Results in one paragraph

Single-threaded, with the same input bits, on an i5-13420H pinned to its P-cores: **numj beats NumPy in 14 of 15
kernel/size cases in both runs.** It is about 2–6.5× faster than NumPy written with reused output buffers for 10⁵–10⁷
elements, about 13–16× faster at 10³, and ~100× faster at 16 elements, where NumPy's per-call overhead dominates. The 15th case was disturbed by
system noise in run 2 and counts as inconclusive. **Compared with plain Java loops, the result depends on size.** numj
is 4–6× faster when the data fits in L1 (10³ elements), 1.6–3× faster at 10⁵, and **tied at 10⁷**,
where every implementation is limited by DRAM bandwidth (~17 GB/s on one core). numj is **slower** for very small
inputs (16 elements: 0.63–0.70×, where the ~8 ns Java→Fortran call costs about as much as Java's whole loop) and for **short-row
normalisation** (4- or 10-element rows: 0.34–0.74×). End-to-end calls that copy Java `double[]` data into fresh native memory
every time lose to plain Java at every size. The library only pays off when data stays in `F64Array`.

## Quick start (Windows 11, PowerShell, no admin rights)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1   # pinned JDK 25 + gfortran 16.2 + NumPy venv into .tools/ and .venv/ (SHA-256 verified)
powershell -ExecutionPolicy Bypass -File scripts\build.ps1       # 4 native profiles + Java classes
powershell -ExecutionPolicy Bypass -File scripts\test.ps1        # 20 correctness tests (exit code != 0 on failure)
powershell -ExecutionPolicy Bypass -File scripts\bench.ps1       # ~10 min; writes results\*.csv, *.log, environment.json, RESULTS.md
```

Run the example:

```powershell
. .\scripts\env.ps1
java --enable-native-access=ALL-UNNAMED -cp "build\classes;build\example-classes" Example
```

Other options: `scripts\test.ps1 -Library build\native\numj-sse2.dll` tests another build;
`scripts\bench.ps1 -Suites main,numpy -Quick` is a smoke run (numbers not meaningful).

### Toolchain (pinned in `scripts/bootstrap.ps1`)

| component | version | why |
|---|---|---|
| JDK | Eclipse Temurin 25.0.4.1+1 (LTS) | FFM API is final since JDK 22; 25 is the current LTS |
| Fortran | GCC/gfortran 16.2.0, WinLibs MinGW-w64 UCRT r2 | portable zip, no installer; nothing was installed on the machine |
| Python / NumPy | CPython 3.13.15, NumPy 2.5.2 (in `.venv`) | the NumPy version already on this machine, pinned in `bench/requirements.txt` |
| build | PowerShell scripts, plain `javac` | no Maven/Gradle/JUnit/JMH to download; trade-offs listed under Limitations |

The DLLs are linked with `-static`, so libgfortran, libgomp and libgcc are inside them and they depend only on
Windows system/UCRT DLLs (`objdump -p build\native\numj.dll`).

## Layout

```
native/numj_kernels.f90            Fortran kernels (bind(C) entry points)
java/src/numj/F64Array.java        native float64 array: shape, ownership, lifetime, bounds
java/src/numj/NumJ.java            public kernels + validation + numerical contract (javadoc)
java/src/numj/Native.java          FFM downcall handles (regular + critical)
java/test/numj/NumJTests.java      correctness tests (zero-dependency harness)
java/bench/numj/bench/*.java       Java benchmark harness, plain-Java baselines, shared data generator
java/example/Example.java          minimal usage
bench/numpy_bench.py               NumPy benchmarks (same protocol, same input bits)
bench/report.py                    builds results/RESULTS.md from the CSVs
scripts/*.ps1                      bootstrap, env, build, test, bench
results/                           measured results of the run described below
```

## API

```java
try (F64Array a = F64Array.of(1, 2, 3, 4);
     F64Array b = F64Array.of(0, 2, 0, 4);
     F64Array x = F64Array.allocate(1000, 64);          // [rows, cols], zero-filled, row-major
     F64Array query = F64Array.allocate(64);
     F64Array out = F64Array.allocate(1000);
     F64Array norms = F64Array.allocate(1000)) {
    double d = NumJ.sqdist(a, b);                      // sum((a-b)^2)
    double s = NumJ.sumSqMulAdd(a, b, a);              // sum((a*b + c)^2), one pass, no temporaries
    NumJ.normalizeRows(x, x, norms);                   // in place (same memory) or into another array
    NumJ.sqdistRows(query, x, out);                    // batched: one native call for all rows
}
```

### Memory model

* **Layout.** Rank 1 `[n]` or rank 2 `[rows, cols]`, dense, row-major (C order, NumPy's default). No strides, so
  every kernel receives a plain pointer and never copies. Fortran sees `[rows, cols]` as `x(cols, rows)`,
  which makes each row a contiguous column.
* **Ownership.** `allocate`, `allocateShared`, `of` and `copyOf` create an **owner** with its own `Arena`, and
  `close()` frees the memory at once (idempotent). `row(i)`, `reshape`, `flatten` and `wrap(segment, …)` create **views**
  that share memory and never free it; `close()` on a view does nothing.
* **Lifetime.** Any access after the memory is freed throws `IllegalStateException`. This includes views whose
  parent was closed, and kernels given a closed array. The FFM runtime keeps an arena alive for the whole native call.
  The Fortran code never keeps a pointer after it returns.
* **Threads.** `allocate` is confined to the creating thread; other threads get `WrongThreadException`.
  `allocateShared` can be used from any thread.
* **Bounds.** Every accessor checks its indices and throws `IndexOutOfBoundsException`. Shapes are validated in Java
  before any native call. Calling the Fortran symbols directly bypasses these checks.
* **Copies.** These happen only when you ask for them (`of`, `copyOf`, `copyFrom`, `copyTo`, `toArray`), each as one bulk copy. Arrays are
  allocated 64-byte aligned and zero-initialised.

### Numerical contract

| topic | behaviour |
|---|---|
| FP semantics | Strict IEEE-754 by default: built with `-ffp-contract=off` and no fast-math, so every `*`, `+`, `-`, `/` and `sqrt` in the formulas is rounded separately. Per-element values match NumPy and plain Java exactly. |
| summation order | Fixed and documented: blocks of 4096 elements; within a block, element *j* is added to lane *j* mod 16; lanes and blocks are combined by fixed pairwise trees. **Results are bit-identical for every thread count and every strict build profile** (AVX2, SSE2, scalar), and bit-identical to `JavaKernels.*Blocked` (tested). |
| accuracy bound | For the reductions (all terms ≥ 0): `|computed − exact| ≤ γ(k)·exact`, where `γ(k) = k·u/(1−k·u)`, `u = 2⁻⁵³` and `k = NumJ.summationDepth(n) ≤ 256 + 4 + ⌈log₂(n/4096)⌉`, measured against the exact sum of the rounded per-element terms. For comparison, a naive loop has `k = n`. |
| normalisation accuracy | Each output element is within `(ncols + 8)·u` relative error of the exact `x/‖x‖` (tested against a 50-digit BigDecimal reference). |
| empty input | Reductions return `+0.0`. Zero rows or zero columns: nothing is written, and zero-length rows get norm `0.0`. |
| shape mismatch | `IllegalArgumentException` before any native call. Rank and extents must match exactly. |
| NaN | A NaN input makes the affected result NaN: the scalar for reductions, the whole row for normalisation. |
| ±Inf | IEEE propagation, e.g. `Inf − Inf = NaN`. In normalisation, a row containing ±Inf (and no NaN) gets norm `+Inf` and outputs `x/Inf`: finite → ±0, infinite → NaN. |
| overflow | Reductions are unscaled, like NumPy: if the true value or an intermediate exceeds `DBL_MAX`, the result is `+Inf`. **Normalisation is overflow/underflow-safe**: if the row's sum of squares overflows or drops below 2⁻⁹⁶⁸, the row is recomputed with scaling by its largest absolute value. Rows of 1e300 or 1e-300 values therefore normalise correctly, where `x / sqrt(sum(x*x))` gives zeros or NaN. The scaled path runs only for such rows. |
| zero norm | The row is copied unchanged (all ±0) and its norm is `0.0`; it does not become NaN. |
| overlapping buffers | Read-only inputs may alias (`sqdist(a, a) == 0`). An output must be disjoint from every other argument or, for `normalizeRows`, be *exactly* the input's memory (in place, handled by a separate single-argument Fortran entry so Fortran's no-aliasing rule holds). Any partial overlap throws `IllegalArgumentException`. |
| relaxed FP | Opt-in only, through the `numj-fast` build (`-ffast-math`). It **fails** 5 of the 20 tests: a 1-element row normalises to `-0.9999999999999999` because of reciprocal multiplication, `3/5` becomes `0.6000000000000001`, and the overflow-safe path is silently removed (`-ffinite-math-only`), so huge rows become zeros. |

## Design choices (brief)

* **FFM, not JNI.** There is no C glue: `Linker.downcallHandle` binds the `bind(C)` symbols directly. Each kernel has
  a regular handle and a `critical` handle. Critical calls skip the thread-state transition, which is cheaper, but they
  hold off safepoints and therefore GC, so `NumJ` uses them only for single-threaded calls of up to 65 536 elements
  (`-Dnumj.criticalMaxElements` overrides this).
* **Fusion.** `sqdist` and `sumSqMulAdd` read each input once and write nothing; NumPy's ordinary expressions
  create 2–3 temporaries. `normalizeRows` computes a row's norm while the row is still in L1, then scales it.
* **Batching.** `sqdistRows` and `normalizeRows` handle all rows in one native call, so the about 6–11 ns call cost is
  paid once rather than once per row.
* **Vectorization.** Inside a block, 16 independent partial sums are written as fixed-length array statements, and GCC maps
  them to four 256-bit accumulators held in registers (checked in the generated assembly; see
  `build/vec_report.txt`). This order is also what makes the result independent of SIMD width and thread
  count. Compiler flags reorder nothing.
* **Threads.** Single-threaded by default. `NumJ.setThreads(n)` turns on OpenMP inside the DLL, only for inputs of
  at least 2¹⁸ elements. The library never creates threads otherwise. If your application already calls numj from several Java
  threads, leave it at 1 to avoid two thread pools competing for cores. If you do mix the two, set `OMP_WAIT_POLICY=PASSIVE` so idle
  OpenMP workers do not spin.
* **No linear algebra yet.** Nothing here is a matrix product. If `gemm`, solvers and similar operations are added, they should
  call a tuned BLAS/LAPACK (OpenBLAS, MKL), not hand-written Fortran.

## Benchmarks

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

## Limitations

* Only Windows 11 x64 with MinGW-w64 gfortran was built and tested. Linux and macOS library names are handled in `Native.java`
  but have **not** been tested.
* The DLLs target x86-64-v3 (AVX2/FMA, Haswell or newer). `numj-sse2.dll` runs on any x86-64 machine.
* The benchmark harness is custom (no JMH). It follows JMH's main rules (warm-up, sinks, repeated samples) but not
  forking or a blackhole intrinsic. Its megamorphic `DoubleSupplier` call costs every Java case ~1–2 ns, which matters only
  for the 16-element rows.
* The test suite has no JUnit. It is a 20-test harness with derived tolerances that exits non-zero on failure.
* Results come from one laptop on the *Balanced* power plan. The drift between runs is documented above, and ratios inside it should be read as ties.

## License

Copyright 2026 Hareram

Licensed under the [Apache License, Version 2.0](LICENSE). Unless required by applicable law or agreed to in
writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
and limitations under the License.
