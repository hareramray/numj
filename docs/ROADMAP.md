# Roadmap

Milestone 1 (general float64 array foundations) is complete; see [CHANGELOG.md](../CHANGELOG.md). The stages below
are ordered by priority. Within each stage, the first items unblock the later ones. "Measured" items cite
[`results/nd/RESULTS.md`](../results/nd/RESULTS.md).

## Priority 0 — measured gaps in milestone 1 (do these first)

Ordered by impact in [`results/nd/RESULTS.md`](../results/nd/RESULTS.md) and [PERFORMANCE.md](PERFORMANCE.md):

1. **Strided planning overhead.** Small strided or broadcast calls cost 140–200 ns versus 8–20 ns for Java loops.
   Cache plans per (shape, strides) signature, avoid per-call allocations, and pass small descriptors by value.
2. **Profile the broadcast `[r, c] + [c]` loss at 10⁷** (22.7 vs 14.1 ms for Java), whose cause is unknown.
3. **Cache-blocked strided elementwise kernels** (transposed operands), plus an optional result layout that
   follows the inputs (NumPy's `order='K'`). That choice alone makes NumPy's allocating transposed add 3× faster
   than numj's.
4. **Column sums:** wider or L2-sized tiles, measured against NumPy's row-wise accumulation (currently 0.68×).
5. **Rows of ≤ 4 elements:** inline the lane tree for tiny K (rows of 2 cost ~9 ns per row versus ~2 ns in Java).
6. **An opt-in memory-order reduction** for transposed and other strided inputs: deterministic for a given layout,
   but explicitly not layout-independent. It would remove the 0.16–0.36× loss without weakening the default contract.
7. **Repeat the benchmarks** on a fixed-frequency machine with ≥ 3 independent runs (only one run per configuration
   exists for milestone 1), and add the JDK Vector API as a pure-Java baseline. The tiny-input results suggest that a
   Java path that is fast enough for small arrays needs the Vector API or heap arrays, not `MemorySegment` loops.


## Stage 1 — more operations on float64 (next milestone)

1. **Bool dtype first** (needed by comparisons, masks, `any`/`all`, `where`), then comparisons (`equal`, `less`, … with
   IEEE NaN semantics) and `where(cond, x, y)`.
2. **Reductions:** `min`/`max` (NumPy propagates NaN), `argmin`/`argmax` (first occurrence, NaN wins), `prod`
   (same blocked order as `sum`), `var`/`std` with a documented two-pass algorithm and `ddof`, `cumsum`/`cumprod`
   (sequential order, documented), `any`/`all`. Each reduction reuses `numj_reduce`'s three paths
   (contiguous, column tile, general walk) so results stay layout-independent.
3. **Unary math:** `negative`, `abs`, `sqrt` (correctly rounded, so bitwise comparable), `square`, then
   `exp/log/sin/cos/tanh/…`. Fortran intrinsics call the C runtime's libm, while NumPy uses its own SIMD
   implementations, so results can differ by an ulp or more. The contract must state an ulp bound per function and the
   differential tests must use it. Never claim bitwise equality for these.
4. **Binary math:** `power` (exact for the special cases NumPy treats specially), `maximum`/`minimum`/`fmax`/`fmin`,
   `floor_divide`, `remainder`, `copysign`, `hypot`, `arctan2`.
5. Shape utilities: `squeeze`, `expand_dims`, `moveaxis`, `concatenate`, `stack`, `split`, `tile`, `repeat`.

Every new kernel goes through `gen_elementwise.py` (one loop per operation and layout, no per-element dispatch), uses
the same overlap and aliasing rules, and gets differential cases.

## Stage 2 — data types, casting and promotion

* float32, int8–int64, uint8–uint64, bool, complex64/complex128 as `DType` constants, with one typed `NDArray`
  subclass each and generator templates per type.
* Promotion follows the Array API standard's table, which agrees with NumPy 2's NEP 50 rules: no value-based
  casting, and Java `double`/`long` scalars are weakly typed. Cross-kind promotions that the Array API leaves undefined
  (e.g. int64 + float32) follow NumPy and are documented.
* `casting=` levels (`no`, `equiv`, `safe`, `same_kind`, `unsafe`) for `out` and `astype`. The default for `out` is
  `same_kind`, as in NumPy.
* Integer overflow wraps, as in NumPy; integer division by zero needs a documented result (NumPy returns 0 and warns).
* Mixed-type kernels: either cast in the inner loop (generated) or cast once into a buffer. The choice is measured,
  not assumed.

## Stage 3 — linear algebra through tuned BLAS/LAPACK

* `matmul`/`dot`/`vdot`/`tensordot`/`einsum` subset, `solve`, `inv`, `lstsq`, `cholesky`, `qr`, `eigh`, `eig`, `svd`,
  `det`, `norm`.
* Bind an existing tuned library through FFM, for example the scipy-openblas build NumPy ships, MKL, or a
  platform BLAS. Do not hand-write GEMM in Fortran. Batched (stacked) operation through n-d views; non-contiguous
  inputs use BLAS leading dimensions where possible and documented explicit copies otherwise.
* A thread policy that keeps BLAS threads, numj's OpenMP and application threads from oversubscribing.
* Differential tests with condition-number-aware tolerances.

## Stage 4 — sorting, searching, advanced indexing, random

* `sort`/`argsort` (stable; NaN sorts last, as in NumPy), `partition`, `searchsorted`, `unique`, `nonzero`, `where`.
* Advanced indexing: integer-array and boolean-mask indexing (results are copies), `take`/`put`, assignment through
  masks.
* `random.Generator` with PCG64 and Philox. Bit-exact streams compatible with NumPy's are feasible for the bit generators
  and the simplest distributions; each distribution documents whether it is stream-compatible.

## Stage 5 — FFT and file interoperability

* `.npy`/`.npz` read and write (`.npy` memory-mapped through `FileChannel.map` → `MemorySegment`, so arrays can be views of
  files). A reader already exists in the test sources (`DiffTests.Npy`).
* FFT (`fft`, `ifft`, `rfft`, n-d variants) through a permissively licensed implementation, such as pocketfft, which
  NumPy uses (BSD). FFTW is GPL and therefore unsuitable as a default dependency.

## Stage 6 — specialised functionality

datetime64/timedelta64, structured (record) dtypes, masked arrays, string arrays. These come last because they need
the full dtype machinery of stage 2, and their value for numerical Java users is lower.

## Cross-cutting work (scheduled alongside the stages)

* **Fusion:** grow the explicit `Expr` pattern list from measured demand (dot products `sum(a*b)`, norms, `mean` of
  expressions, axis reductions of expressions). Then, as a separate project, consider a template-based kernel
  generator: expression templates instantiated at build time, still no runtime interpretation. A general runtime
  expression compiler, or automatic fusion of ordinary method chains, is explicitly out of scope until a design shows
  how to keep strict IEEE semantics and the overlap rules.
* **Packaging:** a Maven/Gradle build, CI on Windows/Linux/macOS, signed artifacts, a JPMS module (see PACKAGING.md).
* **Testing:** move the harness to JUnit 5 once a build tool exists. Extend the differential suite with property-based
  generation of shapes and index expressions.
* **Benchmarks:** a fixed-frequency Linux machine, a *High performance* power plan, ≥ 3 independent runs, and the JDK
  Vector API as a stronger pure-Java baseline.
