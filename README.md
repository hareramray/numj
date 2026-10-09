# numj — float64 arrays for Java, backed by Fortran

Java 25 (Foreign Function & Memory API) → Fortran 2018 (`ISO_C_BINDING`, `bind(C)`) kernels on `float64` arrays in
native memory. The long-term goal is broad NumPy-like numerical functionality for Java, with performance
advantages **on workloads where they can be demonstrated**. numj does not claim universal speed advantages over NumPy,
and using Fortran guarantees no speed by itself. The measured results below include wins, ties and losses.

**Status (0.2, milestone 1, verified on this machine 2026-10-10):**
- n-dimensional strided arrays, views, broadcasting, elementwise arithmetic, `sum`/`mean` over axes, reusable
  outputs, explicit pattern-based fusion, plus the four fused kernels of 0.1.
- Tests (all passing on the default AVX2 build and on the strict SSE2 and no-vectorization builds):
  - 20 original tests;
  - 29 n-d test groups;
  - 259 differential cases against NumPy 2.5.2, each replayed on the default, forced-native and forced-Java paths.
- Benchmarks: JMH (3 forks) against plain Java, and the same workloads against NumPy, NumPy with `out=`, NumExpr
  and Numba. Results are in [`results/nd/RESULTS.md`](results/nd/RESULTS.md); a summary follows.

### Results in brief (milestone 1)

Single-threaded, same input bits, i5-13420H pinned to its P-cores; medians, with a win or loss only when the p10–p90
ranges do not overlap. Full tables: [`results/nd/RESULTS.md`](results/nd/RESULTS.md).

* **Where numj wins.**
  * Contiguous elementwise `a + b` into a reused buffer: 6.6–19× faster than NumPy with `out=` at 16–10³ elements and
    1.7× at 10⁵.
  * Full and row sums: 4.6–47× faster than NumPy up to 10⁵ elements (the full sum at 10⁵, 2.4×, is a tie by the p10–p90 rule).
  * The explicitly fused `sum((a*b+c)**2)`: 3.3–25× faster than NumPy with reused buffers, 3.1–91× faster than
    NumExpr, and 1.1–6.4× faster than Numba on contiguous data.
  * Across the 16 elementwise cells, numj beats NumPy-out in 13 and ties 3, beats NumExpr in 13 and ties 3, and beats
    Numba in 12 and ties 4. The main reason is per-call overhead: ~17 ns for numj against 0.3–13 µs.
* **Where it ties.** At 10⁷ elements, contiguous and stepped elementwise operations are memory-bound and tie with
  NumPy, NumExpr, Numba and plain Java (≈ 17 GB/s on one core).
* **Where it loses.**
  * Plain Java `double[]` loops win on tiny contiguous inputs (6.5 vs 17.6 ns at 16 elements: the downcall is a fixed
    cost) and on small strided ones (planning overhead). Elementwise vs Java overall: 2 wins, 5 ties, 9 losses.
  * Several 10⁷-element cases lose: transposed and broadcast elementwise, and column sums (0.68× NumPy).
  * Sums over transposed views lose (0.16–0.36× NumPy). This is the deliberate cost of results that do not depend on
    memory layout.
  * Rows of 2 elements still lose to Java and Numba.
* **What changed because of measurements.** Two improvements were measured and retained:
  * a contiguous fast path: small calls went from ~200 ns to ~17 ns;
  * short-row sums: 1.25–2.3× faster.
  Two assumptions were refuted: a Java loop for tiny reductions turned out slower than the downcall, and multi-threaded
  elementwise operations gain at most 1.3×. Critical downcalls save only 4–9 ns and are now opt-in. See
  [docs/PERFORMANCE.md](docs/PERFORMANCE.md).
* The 0.1 kernel benchmarks are kept as historical evidence in [docs/HISTORY_0.1.md](docs/HISTORY_0.1.md). They were
  not re-measured.

## Documents

| document | contents |
|---|---|
| [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) | implemented / partial / intentionally different / unsupported, versus NumPy and the Array API; exception mapping |
| [docs/MIGRATION.md](docs/MIGRATION.md) | 0.1 → 0.2 changes, the `flatten()` plan, critical downcalls, C ABI 2 |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | execution paths, inner loops, reductions, threads, and the measured decisions behind each threshold |
| [docs/ROADMAP.md](docs/ROADMAP.md) | prioritised later milestones |
| [docs/PACKAGING.md](docs/PACKAGING.md) | runtime requirements, CPU fallback, proposed Maven Central layout |
| [results/nd/RESULTS.md](results/nd/RESULTS.md) | milestone-1 measurements (hardware, versions, variability, wins/ties/losses) |
| [docs/HISTORY_0.1.md](docs/HISTORY_0.1.md) | the 0.1 benchmark narrative (historical, not re-measured) |
| [CHANGELOG.md](CHANGELOG.md) | release notes |

## Quick start (Windows 11, PowerShell, no admin rights)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1   # pinned JDK 25, gfortran 16.2, JMH 1.37, NumPy/NumExpr/Numba venv (hash-verified)
powershell -ExecutionPolicy Bypass -File scripts\build.ps1       # 4 native profiles, Java classes, tests, example, JMH benchmarks
powershell -ExecutionPolicy Bypass -File scripts\difftest.ps1    # NumPy reference results + differential tests
powershell -ExecutionPolicy Bypass -File scripts\test.ps1        # all suites (exit code != 0 on failure)
powershell -ExecutionPolicy Bypass -File scripts\bench_nd.ps1    # milestone-1 benchmarks (~1.5 h), then:
.venv\Scripts\python.exe -I bench\report_nd.py results\nd       # -> results\nd\RESULTS.md
powershell -ExecutionPolicy Bypass -File scripts\package.ps1     # packaging prototype + smoke test from the jars (nothing published)
```

Run the example (n-d views, broadcasting, reductions, reusable buffers, explicit fusion, fused kernels):

```powershell
. .\scripts\env.ps1
java --enable-native-access=ALL-UNNAMED -cp "build\classes;build\example-classes" Example
```

Other options:
- `scripts\test.ps1 -Library build\native\numj-sse2.dll` tests another build.
- `scripts\bench_nd.ps1 -Jmh 'numj\.ReduceBench' -NoPython -Tag x` runs one JMH class.
- `scripts\bench.ps1` runs the 0.1 harness.

### Toolchain (pinned in `scripts/bootstrap.ps1`)

| component | version | why |
|---|---|---|
| JDK | Eclipse Temurin 25.0.4.1+1 (LTS) | FFM API is final since JDK 22; 25 is the current LTS |
| Fortran | GCC/gfortran 16.2.0, WinLibs MinGW-w64 UCRT r2 | portable zip, no installer |
| JMH | 1.37 (+ jopt-simple 5.0.4, commons-math3 3.6.1) from Maven Central | Java microbenchmarks with forks |
| Python | CPython 3.13.15; NumPy 2.5.2, NumExpr 2.14.2, Numba 0.68.0 / llvmlite 0.50.0 (`.venv`) | reference results and comparison baselines; pinned in `bench/requirements*.txt` |
| build | PowerShell scripts, plain `javac` | no Maven/Gradle yet (see docs/PACKAGING.md) |

The DLLs are linked with `-static`, so libgfortran, libgomp and libgcc are inside them and they depend only on Windows
system/UCRT DLLs.

## Layout

```
native/numj_kernels.f90            fused kernels + blocked summation (0.1, ABI 2)
native/numj_elementwise.f90        elementwise kernels: contiguous, in-place, n-d strided (GENERATED)
native/gen_elementwise.py          generator for numj_elementwise.f90
native/numj_reduce.f90             axis reductions: contiguous, column-tile and general strided paths
java/src/numj/NDArray.java         layout, flags, ownership, lifetime (dtype-independent)
java/src/numj/F64Array.java        float64 arrays: creation, views, copies, element access
java/src/numj/NumJ.java            public operations + numerical contract (javadoc)
java/src/numj/Expr.java            explicit pattern-based fusion
java/src/numj/{Layout,Plan,Elementwise,Reduce,Native,NativeLoader}.java   internals
java/test/numj/*.java              NumJTests (0.1), NDArrayTests, DiffTests (NumPy)
java/jmh/numj/*.java               JMH benchmarks
java/example/Example.java          usage tour
difftest/gen_numpy_cases.py        NumPy reference generator for DiffTests
bench/numpy_nd_bench.py            NumPy / NumExpr / Numba side of the benchmarks
bench/report_nd.py                 builds results/nd/RESULTS.md
scripts/*.ps1                      bootstrap, build, test, difftest, bench_nd, package
```

## API

```java
try (F64Array x = F64Array.arange(24).reshapeCopy(2, 3, 4);       // owner: close() frees native memory
     F64Array row = F64Array.of(1, 2, 3, 4);
     F64Array y = NumJ.add(x, row);                                // broadcasting; new C-ordered owner
     F64Array s = NumJ.sum(x, new int[] {0, 2}, false);            // axis reduction -> [3]
     F64Array out = F64Array.allocate(2, 3, 4)) {
    F64Array v = x.slice("::-1, 1:, ::2");                         // view: reversed + stepped, no copy
    F64Array t = x.permute(2, 0, 1);                               // view
    NumJ.multiply(v, 2.0, out.slice("::-1, 1:, ::2"));             // reusable, strided output
    NumJ.add(out, out, out);                                       // in place (x = x + x)
    double m = NumJ.mean(t);                                       // same bits for any layout of the same values
    double f = Expr.sum(Expr.of(x).mul(Expr.of(x)).add(Expr.of(x)).square()).evaluate();   // fused, one pass
}
```

### Memory model

* **Layout.** Any rank from 0 to 64. Element `[i0..ik]` is at `byteOffset + Σ i·stride` of the base segment. Strides are
  in bytes and are multiples of 8; negative strides give reversed views, zero strides give broadcast views.
* **Ownership.** Operations and factories without an arena or `out` return an **owner**, and `close()` frees it.
  `allocate(arena, shape…)` places the array in your arena. Views (`slice`, `transpose`, `reshape`, `broadcastTo`,
  `wrap`) never free anything.
* **Lifetime.** Any use after the memory is freed throws `IllegalStateException`, including through views. The FFM
  runtime keeps memory alive during native calls, and Fortran never keeps pointers.
* **Threads.** Confined by default (`WrongThreadException` from other threads, for both Java and native paths).
  `allocateShared` arrays can be used from any thread.
* **Writability.** `broadcastTo`, `asReadOnly` and wrapped read-only segments are read-only, and writes throw
  `ReadOnlyArrayException`. A writable array can never address one element twice, so writes are never ambiguous.
* **Copies.** Nothing is copied implicitly, with one exception: an input that overlaps an operation's output in a way
  that would change the result is copied first, which matches NumPy's "as if copied" rule. `reshape` is view-only;
  `reshapeCopy`, `copy`, `flattenCopy` copy explicitly.
* **Validation.** Shapes, broadcasting, `out` shape and writability, bounds and lifetime are checked in Java before any
  native call. Calling the Fortran symbols directly bypasses every check.

### Numerical contract

| topic | behaviour |
|---|---|
| FP semantics | Strict IEEE-754: `-ffp-contract=off`, no fast-math. Every `+ − × ÷` is rounded separately, so **elementwise results are bitwise identical to NumPy's and to Java arithmetic** (verified, including NaN, ±Inf, ±0 and subnormals) for every layout, path and thread count. |
| summation order | Every reduction sums the output's logical sequence (C order of the reduced axes) in blocks of 4096. Inside a block, element *j* goes to lane *j* mod 16, and lanes and blocks are combined by fixed pairwise trees. **Results depend only on the values and the shape**: not on memory layout, code path, SIMD width or thread count (tested). |
| accuracy | `|computed − exact| ≤ γ(k)·Σ|tᵢ|`, `γ(k) = k·u/(1−k·u)`, `u = 2⁻⁵³`, `k = NumJ.summationDepth(n) ≤ 260 + ⌈log₂(n/4096)⌉`, where a naive loop has `k = n−1`. Compared with NumPy, which uses a different order, sums differ in the last bits. Over 4,212 tested outputs the largest difference was 0.2× the derived bound. |
| signed zero | Sums start from `+0.0`, as in NumPy: a sum of `−0.0` values is `+0.0`, and `axis=()` maps `−0.0` to `+0.0`. |
| empty | `sum` → `+0.0`; `mean` → NaN (NumPy also warns; numj never warns). |
| NaN / Inf / overflow | IEEE propagation; overflow gives ±Inf; `Inf − Inf` = NaN. `normalizeRows` is overflow/underflow-safe (see its javadoc). |
| relaxed FP | Only in the opt-in `numj-fast` build, which is not used by default and is known to break the contract. |

## Limitations

* Only Windows 11 x64 with MinGW-w64 gfortran was built and tested. Linux and macOS names and CPU detection exist but
  are untested.
* float64 only. Many NumPy operations are not implemented yet: see [COMPATIBILITY.md](docs/COMPATIBILITY.md) and
  [ROADMAP.md](docs/ROADMAP.md).
* Tests use a small harness (no JUnit). The NumPy differential tests need the Python venv.
* All measurements come from one laptop (i5-13420H, *Balanced* power plan, P-cores pinned). Ratios close to 1 should be
  read as ties; `RESULTS.md` reports the variability.

## License

Copyright 2026 Hareram

Licensed under the [Apache License, Version 2.0](LICENSE). Unless required by applicable law or agreed to in
writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
and limitations under the License.
