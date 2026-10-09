# Migrating from numj 0.1 to 0.2 (milestone 1: n-d arrays)

Source compatibility: **every public method of numj 0.1 still exists with the same signature.** Code that only
used rank-1/rank-2 contiguous arrays compiles and behaves as before, with the exceptions listed under
"Behaviour changes". The native library's C ABI changed (version 1 → 2). That matters only if you call the DLL's
symbols directly.

## Behaviour changes

| what | 0.1 | 0.2 | action |
|---|---|---|---|
| `F64Array.flatten()` | rank-1 **view** | still a view, now `@Deprecated`; throws `IllegalArgumentException` for layouts that cannot be flattened without a copy (such layouts did not exist in 0.1) | replace with `flattenView()` (same behaviour) or `flattenCopy()` (NumPy behaviour). See the plan below |
| critical downcalls | used for single-threaded calls up to 65,536 elements | **off by default** (`NumJ.CRITICAL_MAX_ELEMENTS = -1`) | none. To restore 0.1 behaviour, pass `-Dnumj.criticalMaxElements=65536`, but read the next section first |
| tiny elementwise inputs | always a native call | contiguous operations on ≤ 4 elements (strided on ≤ 64) run as an equivalent Java loop (bitwise identical); reductions and fused kernels always call native code, which measured faster | none |
| `rows()` / `cols()` | rank 1 or 2 only | same; rank 0 or ≥ 3 throws `IllegalStateException` | none |
| `segment()` | the array's bytes | the bytes **spanned** by the elements (identical for contiguous arrays) | for strided arrays use `baseSegment()`, `byteOffset()`, `strides()` |
| `reshape(rows, cols)` | always a view | always a view; throws `IllegalArgumentException` if the (new) strided layout cannot be viewed | `reshapeCopy(...)` or `canReshapeView(...)` |
| fused kernels on non-contiguous arrays | impossible (no strides) | `sqdist`/`sumSqMulAdd` accept any layout (same bits); `sqdistRows`/`normalizeRows` accept row-strided views with unit-stride rows and reject other layouts with `IllegalArgumentException` | none |
| writes to read-only arrays | no read-only arrays | `ReadOnlyArrayException` | none |
| native library resolution | `numj.library` / `NUMJ_LIBRARY` / `build/native/numj.dll` | same, plus a classpath resource from a natives jar, and on CPUs without AVX2 the baseline build is chosen automatically (`-Dnumj.cpu=` overrides) | none |

### Why critical downcalls are now opt-in

`Linker.Option.critical` is specified for functions with "an extremely short running time in all cases (similar
to calling an empty function)" (JDK 25 javadoc). During such a call the JVM cannot reach a safepoint, so GC and
other threads that need a safepoint wait. numj 0.1 chose critical calls by element count alone, up to 65,536
elements, which is tens to hundreds of microseconds of work. That size threshold cannot meet the "all cases" condition.
In 0.2 the calls where a critical downcall saved the most (tiny inputs) are served by a Java loop that needs no
downcall at all. `results/nd/RESULTS.md` has the measurements (regular vs critical vs Java path).

## `flatten()` plan

1. **0.2 (this release):** `flatten()` keeps returning a view and is deprecated. `flattenView()` and `flattenCopy()`
   are available, and the compiler flags every remaining use of `flatten()`.
2. **0.3:** `flatten()` returns a C-ordered copy, which is a new owner that must be closed (NumPy semantics). This is
   announced in the changelog as a breaking change.
3. Code that has moved to `flattenView()`/`flattenCopy()` is unaffected by step 2.

A copy is a new native allocation that must be closed, so silently switching `flatten()` to a copy would leak
memory in code that does not close its result. The deprecation cycle gives that code a compiler warning first.

## New concepts in 0.2

* **N-d strided arrays.** Any rank from 0 to 64, byte strides (negative for reversed views, 0 for broadcast views), byte
  offset. `slice(Ix...)` / `slice("numpy index")`, `transpose`, `permute`, `swapAxes`, `broadcastTo`.
* **Ownership.** Operations without an `out` argument return a new **owner**: close it, for example with
  try-with-resources. `F64Array.allocate(arena, shape...)` allocates into your own `Arena`, and the memory is freed when
  that arena closes. Views never free memory.
* **Outputs.** Every operation has an `out` overload for buffer reuse. If `out` overlaps an input, the result is as
  if the input had been copied first. An input with exactly `out`'s layout is updated in place.
* **Explicit fusion.** `Expr` builds an expression and `Expr.Reduction.evaluate()` runs it with a compiled kernel if
  the expression matches a supported pattern. Ordinary method calls are never fused.

## C ABI changes (version 2)

Only relevant if you call the native library directly instead of through Java:

* `numj_sqdist_rows(q, x, nrows, ncols, ldx, out, nthreads)`: new `ldx` (elements between row starts, ≥ ncols).
* `numj_normalize_rows(x, ldx, y, ldy, nrows, ncols, norms, nthreads)` and
  `numj_normalize_rows_inplace(x, ldx, nrows, ncols, norms, nthreads)`: new leading dimensions.
* New: `numj_sum`, `numj_reduce_seq`, `numj_sum_nd`, `numj_ew_contig*`, `numj_ew_nd*`, `numj_copy_nd`
  (see `native/*.f90`).
* `numj_abi_version()` returns 2. The Java layer refuses to load a library with another ABI version.
