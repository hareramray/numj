# Changelog

## 0.2.0 (unreleased) — milestone 1: general float64 array foundations

### Added
* N-dimensional strided `F64Array` (0 to 64 dimensions, byte strides that may be negative or zero, byte offset) on a
  sealed `NDArray` base with a `DType` enum, prepared for further element types.
* Views: `slice(Ix...)` / `slice("numpy index")` (integers, stepped/reversed slices, `None`, `...`), `transpose`,
  `permute`, `swapAxes`, `reshape` (view-only, NumPy's no-copy rule), `canReshapeView`, `flattenView`,
  `broadcastTo` (read-only), `asReadOnly`. Copies: `copy`, `reshapeCopy`, `flattenCopy`, `copyFrom(F64Array)`.
* Creation: `allocate(long...)`, `allocate(Arena, long...)` (arena-managed), `zeros`, `ones`, `full`, `scalar`,
  `arange`, `copyOf(double[], long...)`, strided `wrap(segment, offset, shape, strides)`.
* `NumJ.add/subtract/multiply/divide` with broadcasting, `double` operands on either side, and reusable `out`;
  overlapping outputs behave as if inputs were copied.
* `NumJ.sum/mean` over all elements or selected axes, with `keepdims` and `out`. The summation order is independent of
  layout, code path and thread count.
* `Expr`: explicit pattern-based fusion (`sum(a)`, `sum((a-b)**2)`, `sum((a*b+c)**2)`), with an explicit unfused fallback.
* `SumOrder` (`LOGICAL` default, opt-in `MEMORY`) and `sum`/`mean` overloads taking it: memory-order sums of
  transposed or reversed views run at contiguous speed (deterministic per layout, same error bound).
* `ReadOnlyArrayException`; `NumJ.broadcastShapes`; `numj.NativeInfo` diagnostics.
* Fortran: generated elementwise kernels (`native/gen_elementwise.py` → `numj_elementwise.f90`) and a reduction
  module (`numj_reduce.f90`) with contiguous, column-tile and general strided paths.
* Tests: `NDArrayTests` (29 groups) and `DiffTests` (259 cases against NumPy 2.5.2, each replayed on three code paths).
* JMH benchmarks (`java/jmh`), `bench/numpy_nd_bench.py` (NumPy, NumExpr, Numba), `bench/report_nd.py`,
  `scripts/bench_nd.ps1`, `scripts/difftest.ps1`, and the packaging prototype `scripts/package.ps1`.
* Native library loading from a natives jar on the classpath, with a CPU-level fallback (AVX2 build or SSE2 baseline).

### Changed
* Critical downcalls are opt-in (`-Dnumj.criticalMaxElements`, default `-1`; previously 65,536). See docs/MIGRATION.md.
* Elementwise operations on ≤ 4 contiguous (≤ 64 strided) elements are computed in Java (bitwise identical); thresholds come from measurements.
* Operation results are allocated with non-zeroing `malloc` (they are always fully overwritten); `-Dnumj.resultAlloc=arena` restores zero-filled arenas.
* Parallel thresholds: elementwise 2²³ elements, reductions 2¹⁹ (was 2¹⁸ for everything), from the thread-scaling measurements.
* Reductions, round 2 (all bit-identical; see results/nd/RESULTS.md): column sums use L2-sized column groups (up to
  4096 columns) so rows stream once; strided sums gather blocks into buffers for the vectorised block kernels, tiled
  over 8 neighbouring rows for large transposed inputs; rows of ≤ 8 elements use the lane tree's closed form; smaller
  fused kernels on strided inputs keep the direct lane loop, which measured fastest there. Round 3 widened the
  tiles (up to 64 logical rows) and reduces blocks straight from the tile (bit-identical).
* `sqdist` and `sumSqMulAdd` accept any layout. `sqdistRows` and `normalizeRows` accept row-strided views.
* Native ABI version 2: the row kernels take leading dimensions; new entry points.

### Deprecated
* `F64Array.flatten()`: still a view in 0.2. It will return a copy (NumPy semantics) in 0.3. Use `flattenView()` or
  `flattenCopy()`.

## 0.1.0 — 2026-10-09
* Contiguous rank-1/rank-2 `F64Array`, fused Fortran kernels `sqdist`, `sumSqMulAdd`, `sqdistRows`,
  `normalizeRows`, numerical contract, 20 tests, benchmarks against plain Java and NumPy.
