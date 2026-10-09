# Compatibility with NumPy and the Array API standard

References: **NumPy 2.5.2** (pinned in `bench/requirements.txt`; every behaviour marked *verified* is checked by
`difftest/gen_numpy_cases.py` + `numj.DiffTests`) and the **Python Array API standard 2024.12** (for naming and
semantics where it is stricter than NumPy). numj does **not** claim NumPy compatibility. It follows NumPy's
numerical semantics where it implements a feature. Each difference is listed below with the reason.

Status key: **implemented** · **partial** (subset, see notes) · **different** (intentional, documented) ·
**unsupported** (not yet; see [ROADMAP.md](ROADMAP.md)).

## 1. Numerical API coverage

### Arrays, dtypes, memory

| feature | NumPy / Array API | numj 0.2 | status | notes |
|---|---|---|---|---|
| float64 arrays, any rank incl. 0-d | `ndarray` | `F64Array` | implemented | up to 64 dimensions (NumPy 2.x limit) |
| other dtypes (float32, ints, bool, complex) | yes | — | unsupported | `DType` enum, `NDArray` base and the kernel generator are prepared for them |
| shape, strides (bytes), offset | `shape`, `strides` | `shape()`, `strides()`, `byteOffset()` | implemented | strides must be multiples of 8 (NumPy allows unaligned strides) |
| C/F contiguity flags | `flags.c_contiguous` … | `isCContiguous()`, `isFContiguous()` | implemented, verified | NumPy's relaxed-strides rules (extent-1 axes ignored; empty arrays are both) |
| writeable flag | `flags.writeable` | `isWritable()`, `asReadOnly()` | implemented, verified | numj never lets a writable array address one element twice |
| memory ownership | GC / `base` | owners, arena-managed arrays, views | **different** | native memory is freed explicitly (`close()` / arena), not by GC |
| `np.zeros/ones/full/empty/arange` | yes | `allocate/zeros`, `ones`, `full`, `arange(n)` | partial | no `empty` (allocation is zero-filled), `arange` only `0..n-1` |
| from/to Java arrays | `np.array`, `tolist` | `of`, `copyOf(data, shape...)`, `toArray`, `copyTo` | implemented | row-major order for every layout |
| wrap foreign memory | `np.frombuffer`, `as_strided` | `wrap(segment, shape)`, `wrap(segment, offset, shape, strides)` | implemented | checked: bounds, alignment, self-overlap ⇒ read-only |
| `.npy` / `.npz` files | yes | — | unsupported | roadmap stage 5 (a reader exists in the test sources) |

### Indexing and views

| feature | NumPy | numj | status | notes |
|---|---|---|---|---|
| basic indexing: ints, slices, `None`, `...` | `a[...]` | `slice(Ix...)`, `slice("::-1, 2, None")` | implemented, verified | Python slice clamping rules; negative steps; same strides as NumPy for axes of extent > 1 |
| all-integer index | returns a NumPy **scalar** (copy) | returns a **0-d view** | **different** | a Java method returning `F64Array` cannot return a primitive; use `get(i, j, k)` for the value. `a[i, j, k, ...]` is a 0-d view in both |
| element read/write | `a[i, j]`, `a.flat[i]` | `get(long...)`, `set(long[], v)`, `get(flat)`, `item()` | implemented | multi-indices are bounds-checked and **not** wrapped (use `slice` for negative indices); `get(long)` is a flat index for every rank |
| advanced (integer-array / boolean) indexing | yes | — | unsupported | roadmap stage 4 |
| transpose, axis permutation | `.T`, `transpose(axes)`, `swapaxes` | `transpose()`, `permute(int...)`, `swapAxes` | implemented, verified | |
| reshape | `reshape` copies when it must | `reshape(...)` is **view-only** (throws if a copy is needed); `reshapeCopy(...)`; `canReshapeView(...)` | **different** | NumPy's default `copy=None` silently allocates; numj makes the copy explicit because a copy is a new owner that must be closed. `reshape` = NumPy `reshape(copy=False)`, `reshapeCopy` = `copy=True` (verified, including which strided layouts allow a view) |
| `flatten` | always a copy | deprecated `flatten()` is a view; `flattenView()`, `flattenCopy()` | **different (transitional)** | see [MIGRATION.md](MIGRATION.md) |
| `ravel` | view if possible | `flattenView()` (throws instead of copying) | partial | |
| `broadcast_to`, `broadcast_shapes` | yes | `broadcastTo`, `NumJ.broadcastShapes` | implemented, verified | broadcast views are read-only, as in NumPy |
| `squeeze`, `expand_dims`, `moveaxis`, `concatenate`, `stack`, `split` | yes | — | unsupported | `slice(Ix.newAxis())` covers `expand_dims` |
| `copy` | `a.copy()` (C order) | `copy()` | implemented, verified | always C order (NumPy's `np.copy` defaults to `order='K'`) |
| `np.copyto` | yes | `dst.copyFrom(src)` | implemented | broadcasting, overlap-safe |

### Elementwise arithmetic (`add`, `subtract`, `multiply`, `divide`)

| aspect | NumPy | numj | status |
|---|---|---|---|
| values | IEEE-754, one rounding per op | same; bitwise identical (verified, including NaN/±Inf/±0/subnormal inputs) | implemented |
| broadcasting, scalar operands, 0-d arrays, empty dimensions | yes | yes (`double` overloads on either side) | implemented, verified |
| `out=` | output may also broadcast the inputs | output must have exactly the result shape | **different** (stricter) |
| `out=` overlapping an input | result as if inputs were copied | same rule (identical layout ⇒ in place; any other overlap ⇒ the input is copied first) | implemented, verified |
| read-only `out` | `ValueError` | `ReadOnlyArrayException` | implemented, verified |
| result memory layout | `order='K'` (may follow the inputs, e.g. F order) | always C order | **different** |
| dtype promotion, `casting=`, `where=`, `dtype=` | yes | float64 only | unsupported |
| floating-point error reporting (`np.errstate`, warnings) | configurable warnings | never reports; IEEE results only | **different** |
| other ufuncs (`power`, `sqrt`, `exp`, comparisons, `maximum` …) | yes | — | unsupported (roadmap stage 1) |

### Reductions (`sum`, `mean`)

| aspect | NumPy | numj | status |
|---|---|---|---|
| `axis=None / int / tuple`, negative axes, `keepdims`, `out=` | yes | `axes` (`null` = all), `keepdims`, `out` | implemented, verified |
| `axis=()` | each element plus the `+0.0` identity | same (`-0.0` becomes `+0.0`) | implemented, verified |
| duplicate / out-of-range axes | `ValueError` / `AxisError` | `IllegalArgumentException` | implemented, verified |
| empty reductions | sum `0.0`; mean `nan` **with RuntimeWarning** | sum `+0.0`; mean NaN, no warning | implemented (warning: different) |
| sum of `-0.0` values | `+0.0` | `+0.0` | implemented, verified |
| summation order | pairwise for contiguous inner loops, sequential otherwise; depends on layout | default `SumOrder.LOGICAL`: fixed blocked order on the logical sequence, **independent of layout, path and thread count**; opt-in `SumOrder.MEMORY`: same algorithm over memory order (layout-dependent, like NumPy, faster for transposed views) | **different** (results differ in the last bits; bounded, verified: max observed difference 0.2× the derived tolerance) |
| `dtype=`, `initial=`, `where=` | yes | — | unsupported |
| other reductions (`prod`, `min`, `max`, `var`, `std`, `argmax`, `any`, `all`, `cumsum`) | yes | — | unsupported (roadmap stage 1) |

### Linear algebra, FFT, random, sorting, I/O, specialised arrays

Not implemented: `linalg`, `matmul`/`dot` (stage 3, through a tuned BLAS/LAPACK), sorting and searching, `random` (stage 4),
`fft`, `.npy` files (stage 5), datetime, structured dtypes, masked arrays (stage 6). numj's fused kernels
(`sqdist`, `sumSqMulAdd`, `sqdistRows`, `normalizeRows`) have no single NumPy equivalent. Their values match the
NumPy expressions documented in their javadoc.

## 2. Python-specific interfaces (not applicable)

These are part of NumPy's Python surface and have no numj counterpart by design: operator overloading
(`a + b`, `a[i]`, `a += b`), the buffer protocol / `__array__` / `__array_interface__` / DLPack (numj's interop
surface is the FFM `MemorySegment`), pickling, `np.errstate` and warnings, object arrays, Python scalars and the
NumPy scalar types, implicit lifetime by garbage collection, and the Array API namespace object
(`__array_namespace__`). numj's Java naming follows NumPy where possible (`add`, `subtract`, `broadcastTo`,
`keepdims`) and uses Java conventions otherwise (`reshapeCopy`, `canReshapeView`, overloads instead of keyword
arguments).

## 3. Exception mapping

| situation | NumPy | numj |
|---|---|---|
| negative extent, more than 64 dimensions, byte size above 2⁶³−1 | `ValueError` | `IllegalArgumentException` |
| shapes that do not broadcast; `out` of the wrong shape | `ValueError` | `IllegalArgumentException` |
| reshape: wrong size, two `-1`, `-1` with size 0, view impossible | `ValueError` | `IllegalArgumentException` |
| axis out of range, duplicate axis, bad permutation | `AxisError` / `ValueError` | `IllegalArgumentException` |
| integer index out of range | `IndexError` | `IndexOutOfBoundsException` |
| too many indices, two ellipses, slice step 0 | `IndexError` / `ValueError` | `IllegalArgumentException` |
| write to a read-only array, read-only `out` | `ValueError` | `ReadOnlyArrayException` (an `UnsupportedOperationException`) |
| write through `segment()` of a read-only array | — | `IllegalArgumentException` (thrown by the JDK's `MemorySegment`) |
| use after `close()` (including views of a closed owner) | — | `IllegalStateException` |
| confined array used from another thread | — | `WrongThreadException` |
