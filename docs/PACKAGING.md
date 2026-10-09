# Packaging and runtime requirements

Status: a **prototype** of the artifact layout exists (`scripts/package.ps1`). It builds the jars and runs the example
from them in an empty directory. Nothing has been published, and no Maven/Gradle build exists yet.

## Runtime requirements

| requirement | value | why |
|---|---|---|
| Java | **25 or newer** (built with `--release 25`) | the FFM API is final since JDK 22; 25 is the current LTS. Lowering the target to 22 is possible (no 23+ API is used) but untested |
| JVM flag | `--enable-native-access=ALL-UNNAMED` (or the module name once numj is a named module) | FFM downcalls are restricted methods; without the flag JDK 25 prints a warning, and later JDKs may refuse |
| OS / CPU | Windows 11 x86-64, **tested**. Linux and macOS: library names and CPU detection are implemented but **untested**; no binaries are built yet | only Windows was available |
| CPU features | x86-64-v3 build (AVX2, FMA, BMI1/2, F16C, MOVBE, LZCNT; Haswell 2013 / Excavator 2015 or newer) **or** the baseline SSE2 build, which runs on every x86-64 CPU | chosen at load time (below) |
| native dependencies | none beyond the OS C runtime (UCRT on Windows): libgfortran, libgomp and libgcc are linked statically | `objdump -p numj.dll` lists only Windows system DLLs |
| threads | single-threaded unless `NumJ.setThreads(n)`. The DLL contains an OpenMP runtime (libgomp) that starts its threads on first parallel use | set `OMP_WAIT_POLICY=PASSIVE` when numj runs alongside other thread pools |

## Library resolution and CPU fallback

`NativeLoader` tries, in order:

1. `-Dnumj.library=<file>` or `NUMJ_LIBRARY`;
2. the classpath resource `numj/native/<os>-<arch>/<level>/<file>` (copied to a temporary directory, then loaded),
   with `<level>` = `x86-64-v3` if the CPU reports AVX2 (Windows `IsProcessorFeaturePresent`, Linux `/proc/cpuinfo`),
   falling back to `baseline`;
3. `build/native/numj.dll` (or `numj-sse2.dll` without AVX2) relative to the working directory, for development builds.

`-Dnumj.cpu=baseline` forces the SSE2 build. Both builds use the same strict floating-point flags, so their results are
**bitwise identical** (the test suites pass on both, and the summation order does not depend on SIMD width).
`java -cp <jars> numj.NativeInfo` prints which library was loaded and why.

Detection is a heuristic. Windows reports AVX2 but not every other x86-64-v3 feature. In practice every CPU with
AVX2 also has the remaining v3 features, but a virtual machine that hides some of them could crash with an
illegal-instruction fault. `-Dnumj.cpu=baseline` is the workaround.

## Artifact layout (proposed for Maven Central)

| artifact | contents | depends on |
|---|---|---|
| `io.github.hareramray:numj` | Java API (`numj.*`), no native code | — |
| `io.github.hareramray:numj-natives-windows-x86_64` | `numj/native/windows-x86_64/{x86-64-v3,baseline}/numj.dll` | — |
| `...:numj-natives-linux-x86_64`, `...-macos-aarch64`, … | per platform (not built yet) | — |
| `io.github.hareramray:numj-platform` (optional) | a POM that depends on the API and all natives jars | all of the above |

Users add the API artifact plus the natives artifact for their platform, or the `-platform` aggregate. The API
jar alone works with `-Dnumj.library`. `scripts/package.ps1` produces `numj-<v>.jar`,
`numj-natives-windows-x86_64-<v>.jar`, `-sources.jar` and `-javadoc.jar` under `build/dist` and smoke-tests them.

## Before a first release (not done)

* A Maven or Gradle build that compiles the Java code, runs the three test suites (the Java-only `NDArrayTests`
  needs the natives jar), attaches sources/javadoc, and signs artifacts (Central requires GPG signatures, POM
  metadata, and a verified `io.github.hareramray` namespace).
* CI that builds the native library on each platform (Windows MinGW-w64, Linux GCC, macOS with gfortran from
  Homebrew or LLVM Flang) and runs the full suite against every CPU level, including the NumPy differential tests.
* A JPMS `module-info.java` (`module numj { exports numj; }`), so users can grant `--enable-native-access=numj`
  instead of `ALL-UNNAMED`.
* Decide whether temporary-file extraction is acceptable for the target environments (read-only `/tmp`, antivirus
  scanning on Windows). The alternative is `-Dnumj.library` with an installed copy.
* Licence notices for the statically linked GCC runtime libraries (GPL with the GCC Runtime Library Exception, which
  permits this use; the notice should still ship in the natives jar).
