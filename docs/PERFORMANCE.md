# Performance design (milestone 1)

This document describes how numj picks an execution path and why. Every threshold is backed by a measurement in
[`results/nd/RESULTS.md`](../results/nd/RESULTS.md); the section [Measured decisions](#measured-decisions) lists them.
Nothing here relaxes the numerical contract. Strict IEEE evaluation is the default and the only mode of the shipped
builds (`-ffp-contract=off`, no fast-math).

## Execution paths

Every operation first validates arguments in Java (shapes, writability, lifetime, bounds, overlap), then builds an
iteration plan (`Plan`):

1. **Drop axes of extent 1** (their strides are irrelevant).
2. **Elementwise only:** reverse every axis on which the output's stride is negative, and order the axes by decreasing
   output stride. The order of independent elementwise operations cannot change any result, so the output is then
   written in memory order.
3. **Merge adjacent axes** that are contiguous with respect to each other in *every* operand (`stride[i] ==
   stride[i+1] * shape[i+1]`). A C-contiguous, an F-contiguous (after step 2), or a uniformly stepped operand set
   collapses to one run. Reductions merge only adjacent reduced axes and keep their logical order, because that
   order defines the summation order.

Then:

| plan | path | why |
|---|---|---|
| elementwise, ≤ 4 elements (contiguous) or ≤ 64 (strided) | Java loop over `MemorySegment`s | measured crossover: below this, a downcall (plus, for strided operands, planning) costs more than the loop. Same bits as the native kernels. Reductions never take this path: native measured faster at every size |
| all operands C-contiguous with one shape (checked before any planning) | **fast path**: contiguous Fortran kernel with cached segments | no plan, no descriptor, no allocation: ~16 ns per call at small sizes instead of ~200 ns |
| one unit-stride run per operand after planning | contiguous Fortran kernel (`numj_ew_contig*`, `numj_sum`) | vectorised (AVX2: 4 doubles per instruction), no per-row dispatch |
| anything else | n-d Fortran kernel (`numj_ew_nd*`, `numj_sum_nd`, `numj_reduce_seq`) | one downcall for the whole operation; the Fortran code walks the outer axes and dispatches **each row** to a unit-stride kernel, a broadcast-scalar kernel, or a general strided loop |

No input is copied to make it contiguous. The only implicit copy happens when an output overlaps an input in a
way that would otherwise change results (NumPy's "as if copied" rule). Inputs with exactly the output's layout
are updated in place instead.

### Inner loops

* The operation is a compile-time constant inside every loop. The generator (`native/gen_elementwise.py`) emits one
  loop per (operation × layout) combination, with dispatch only per row. No callbacks, boxing, interpretation or
  native calls happen inside a loop.
* Row kernels receive unit-stride rows through Fortran sequence association (`a(ia)` passed to an explicit-shape
  dummy), so GCC vectorises them like contiguous arrays. Broadcast rows (stride 0) use a scalar-operand kernel.
* Fortran's no-aliasing rule holds by construction. Out-of-place kernels only ever see an output disjoint from its
  inputs. In-place updates use dedicated entry points where the modified array is the single `intent(inout)`
  argument, and `x op x` has its own kernel. The Java layer routes every call accordingly.

### Reductions

`sum` reproduces the blocked summation order of numj 0.1 for every output: 16 lanes, blocks of 4096, then pairwise
trees, applied to the output's logical sequence. Three Fortran paths produce identical bits:

* **contiguous sequences** (reduce the innermost, unit-stride axis): the 0.1 block kernels;
* **column tiles** (reduce an outer axis while the kept axis is unit-stride, e.g. `sum(axis=0)` of a C array):
  16 lane accumulators per column for 256-column tiles (32 KiB), streaming each row once;
* **general strided walk**: an odometer over the reduced axes feeding the same lane and block logic.

Results are therefore independent of layout, path and thread count. This guarantee costs speed when the logical
order walks memory with a large stride (e.g. the full sum of a transposed array); `RESULTS.md` (`allT`) measures it.

### Threads

Single-threaded by default. With `NumJ.setThreads(n)`, OpenMP is used only above per-operation thresholds:
`EW_PARALLEL_MIN_ELEMENTS` for elementwise operations, `PARALLEL_MIN_ELEMENTS` for reductions. Work is split by
rows or by 32,768-element chunks. Reductions split whole blocks or whole outputs, so the summation order and the
results do not change with the thread count (tested for 1–8 threads).

### Downcalls

All downcalls are regular. Critical downcalls are documented for functions whose running time is "similar to calling
an empty function" in all cases. No size threshold makes an array kernel satisfy that, and the tiny inputs where a
critical call helped most now take the Java path. `-Dnumj.criticalMaxElements=<n>` re-enables critical downcalls for
the four 0.1 kernels as an explicit opt-in (single-threaded calls only: no OpenMP region, no heap allocation, no
upcalls, no blocking).

### Allocation

Results allocated by numj take their memory from the C runtime's `malloc` without zero-filling (every element is
overwritten by the operation); `-Dnumj.resultAlloc=arena` restores zero-filled arena memory. Factories
(`allocate`, `zeros`, …) always zero-fill. Operations with `out` allocate nothing,
except a temporary copy of an input that overlaps the output. Benchmarks report computation with reused outputs
separately from allocating calls.

## Measured decisions

All numbers come from `results/nd/RESULTS.md` (i5-13420H, 4 P-cores pinned, Balanced power plan, JMH 1.37 with
3 forks or the 21-sample Python harness, medians). The JMH files for each step are kept, so "before" and "after"
can be compared on the same benchmark.

| decision | measurement | outcome |
|---|---|---|
| **contiguous fast path** (retained) | `add` into a reused buffer at n = 16: 201 ns → 17 ns; `sum` 92 → 12 ns; `sqdist` 44 → 14 ns. The ~150–180 ns before was Java-side planning, descriptors and slice allocation | same-shape C-contiguous operands skip planning entirely |
| **Java loop for tiny inputs** | contiguous `add`: Java 10.6 ns vs native 16 ns at n = 1, tie at 4, native ahead from 16 (17 vs 22 ns). Strided `add`: Java ahead up to ~64 elements (planning still costs ~130 ns). `sum`/`sqdist`: native ahead at every n ≥ 4 (12 vs 29 ns), and the Java loop never wins clearly | Java loop for ≤ 4 contiguous / ≤ 64 strided elementwise elements; never for reductions. This **reverses** the 0.1 roadmap's expectation for reductions: a generic `MemorySegment` lane loop is slower than the downcall |
| **critical downcalls** (kept opt-in) | regular − critical = 4.4 ns (n = 0), 7.5 ns (16), 9.1 ns (1000); at 65,536 elements critical was *slower* in this run | the gain is real but small, and the JDK's "running time similar to an empty function" constraint is not met by array kernels. Off by default |
| **non-zeroing result allocation** (retained) | `NumJ.add(a, b)` with a fresh result: 1000: 241 → 201 ns; 10⁵: 49.4 → 38.7 µs; 10⁷: 39.0 → 32.1 ms; n = 16 tie. Two separate runs agree | results come from `malloc` (always fully overwritten); `-Dnumj.resultAlloc=arena` restores zero-fill |
| **short-row sums** (retained) | row sums of 10⁶ elements in rows of k: k=2 7.9 → 4.7 ms, k=4 1.9 → 0.84 ms, k=8 1.06 → 0.51 ms, k=16 0.55 → 0.39 ms (Java: 0.95, 0.67, 0.53, 0.42 ms) | incremental offsets for one kept axis; rows ≤ 16 fill the lanes directly. Rows of 2 still lost 4.9× to Java after this step (per-row lane setup and an out-of-line `lane_tree` call, ~9 ns/row); fixed in round 2 below |
| **round 2: column sums** (retained) | groups of up to 4096 columns (512 KiB of lane accumulators, L2-resident) instead of 256, so each row is streamed once: 10⁷ elements 7.66 → 6.12 / 6.44 ms (final / independent repeat); NumPy 5.20 ms. 10³–10⁵ unchanged within noise | still a narrow loss (0.85×) at 10⁷; within the observed run-to-run spread, so it needs repeated runs to settle |
| **round 2: strided sums** (retained) | each 4096-element block is gathered into a buffer and reduced by the vectorised block kernel; for ≥ 2²⁰ elements with a unit-stride second axis, 8 neighbouring logical rows are read per memory row (tiled). Sum of a transposed view: 10³ 1.62 µs → 648 ns, 10⁵ 89 → 48 µs, 10⁷ 40.4 → 25.3 / 23.9 ms | 2–3× faster; still 0.25× NumPy at 10⁷ (the layout-independent order) |
| **round 2: fused kernels on strided inputs** | the untiled gather *regressed* them (10⁵: 150 → 315 µs; 10⁷: 206 → 519 ms); tiling fixed 10⁷ (78 ms) but not ≤ 10⁵, where the original direct lane loop is fastest. An attempt to inline that loop into the gather function was still slow (~215 µs); restoring it verbatim as a separate function recovered 159 µs | dispatch: direct loop below 2²⁰ elements, tiled gather above. 10⁷ transposed: 206 → 76 / 84 ms, 2.5× faster than NumPy-out |
| **round 2: rows of ≤ 8 elements** (retained) | closed form of the lane tree (`((a1+a5)+(a3+a7)) + ((a2+a6)+(a4+a8))`, exact because lanes are never −0.0): k = 2: 4.65 → 0.62 ms, k = 4: 0.84 → 0.42 ms | short-row sums now beat NumPy-out for k ≤ 8 (2.2–5.4×) and tie or beat plain Java |
| **round 3: `SumOrder.MEMORY`** (added, opt-in) | memory-order sums of a transposed view vs NumPy measured in the same session: 10³ 176 ns vs 1.36 µs (7.7×), 10⁵ 5.1 µs vs 20.6 µs (4.0×), 10⁷ 8.1–9.3 ms vs 6.5 ms (0.8×, both memory-bound; numj's own contiguous sum is ~7 ms) | opt-in only: bits depend on the layout. Default stays `LOGICAL` |
| **round 3: wider tiles for `LOGICAL`** (kept, gain unproven) | tiles up to 64 logical rows (512 KiB budget) and blocks reduced straight from the tile. Transposed sum at 10⁷ in three runs: 16.7, 14.8, 24.1 ms (round 2: 25.3 / 23.9 ms) | bit-identical, never slower in these runs, but the gain is within this laptop's noise; 0.26–0.44× NumPy remains |
| **elementwise threads** | 8 threads: 1.10× at 10⁶, 1.10× at 4·10⁶, 1.31× at 1.6·10⁷ (memory-bound) | `EW_PARALLEL_MIN_ELEMENTS` = 2²³ (was 2¹⁸ for everything) |
| **reduction threads** | 10⁵: 4–7× *slower* with 2–8 threads; 10⁶: 2.4× faster with 4 threads | `PARALLEL_MIN_ELEMENTS` = 2¹⁹ (nothing measured between 10⁵ and 10⁶, so the choice is conservative) |
| **tail processing** | elementwise kernels: GCC emits a 32-byte vector body plus 16-byte and scalar epilogues (vectorisation report); `add` at n = 16, 32, 64 costs 16.8, 18.8, 21.4 ns, so tails show no measurable cliff. Reductions keep 0.1's +0.0-padded tail | no change |
| **fast-math** | the opt-in `numj-fast` build fails 5/20, 5/29 and 2/259 test groups (reciprocal division, reassociation, removed overflow guards) | not used; strict IEEE stays the only default |

### Known losses (measured, not yet addressed)

* **Tiny contiguous inputs vs plain Java** (`double[]` loops that the JIT inlines): 6.5 ns vs 17.6 ns at n = 16.
  The downcall is a fixed cost; Java wins whenever data could stay on the Java heap.
* **Strided small inputs**: ~140–200 ns at n = 16 (planning and descriptor), versus 8–20 ns for Java loops.
* **Transposed elementwise at 10⁷**: numj 124 ms, Java 95 ms. NumPy's *allocating* `a.T + b.T` takes 39 ms because it
  allocates the result in the inputs' (F) order and streams, while NumPy with a C-ordered `out` takes 200 ms.
  A cache-blocked strided path, plus an option to allocate results in the inputs' order, are candidates.
* **Broadcast `[r, c] + [c]` at 10⁷**: 22.7 ms versus 14.1 ms for Java and 16.4 ms for NumPy-out. The row kernel
  is the same vectorised `ew_vv` used by the contiguous path, so the cause is not understood yet. Needs profiling.
* **Column sums at 10⁷**: 6.1–6.4 ms versus NumPy's 5.2 ms after round 2 (was 7.7 ms). The remaining gap is the
  lane bookkeeping (16 accumulators per column) on top of a pure stream; within run-to-run noise of a tie.
* **Sums of transposed views** with the default `SumOrder.LOGICAL` at 10⁷: 15–24 ms versus NumPy's 6.5 ms (was
  40 ms before round 2). This is the price of results that do not depend on layout. `SumOrder.MEMORY` (opt-in)
  removes most of it (8–9 ms) and wins clearly up to 10⁵ elements.
* **Run-to-run variability at 10⁷ elements** is large on this laptop: unchanged code measured 24.8 vs 15.1 ms
  (contiguous fused) and 6.3 vs 11.9 ms (row sums) in two runs. Verdicts at 10⁷ are therefore provisional.
