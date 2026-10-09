package numj;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Downcall handles for the Fortran kernels (see native/numj_kernels.f90).
 *
 * Every kernel has two handles: a regular one, and a {@link Linker.Option#critical critical} one that skips
 * the Java-to-native thread-state transition. Critical calls are cheaper but delay safepoints (and so GC)
 * for their duration, so {@link NumJ} only uses them for small, single-threaded calls.
 *
 * Library resolution order: system property {@code numj.library}, environment variable
 * {@code NUMJ_LIBRARY}, then {@code build/native/numj.<ext>} relative to the working directory.
 */
@SuppressWarnings("restricted") // native linkage is this class's whole purpose; callers need --enable-native-access
final class Native {
    static final int EXPECTED_ABI = 1;
    static final Path LIBRARY = resolve();

    // double numj_sqdist(const double* a, const double* b, int64 n, int32 nthreads)
    static final MethodHandle SQDIST, SQDIST_C;
    // double numj_sumsq_muladd(const double* a, const double* b, const double* c, int64 n, int32 nthreads)
    static final MethodHandle MULADD, MULADD_C;
    // void numj_sqdist_rows(const double* q, const double* x, int64 nrows, int64 ncols, double* out, int32 nthreads)
    static final MethodHandle SQDIST_ROWS, SQDIST_ROWS_C;
    // void numj_normalize_rows(const double* x, double* y, int64 nrows, int64 ncols, double* norms|NULL, int32 nthreads)
    static final MethodHandle NORMALIZE, NORMALIZE_C;
    // void numj_normalize_rows_inplace(double* x, int64 nrows, int64 ncols, double* norms|NULL, int32 nthreads)
    static final MethodHandle NORMALIZE_IP, NORMALIZE_IP_C;
    // int64 numj_build_info(char* buf, int64 cap)
    private static final MethodHandle BUILD_INFO;

    static {
        Linker linker = Linker.nativeLinker();
        SymbolLookup lib = SymbolLookup.libraryLookup(LIBRARY, Arena.global());
        Linker.Option critical = Linker.Option.critical(false);

        FunctionDescriptor sqdist = FunctionDescriptor.of(JAVA_DOUBLE, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT);
        FunctionDescriptor muladd = FunctionDescriptor.of(JAVA_DOUBLE, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT);
        FunctionDescriptor rows = FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_INT);
        FunctionDescriptor norm = FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_INT);
        FunctionDescriptor normIp = FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_INT);

        MemorySegment abiSym = find(lib, "numj_abi_version");
        int abi;
        try {
            abi = (int) linker.downcallHandle(abiSym, FunctionDescriptor.of(JAVA_INT)).invokeExact();
        } catch (Throwable t) {
            throw new ExceptionInInitializerError(t);
        }
        if (abi != EXPECTED_ABI)
            throw new UnsatisfiedLinkError("numj native ABI " + abi + " != expected " + EXPECTED_ABI + " (" + LIBRARY + ")");

        MemorySegment s;
        s = find(lib, "numj_sqdist");
        SQDIST = linker.downcallHandle(s, sqdist);
        SQDIST_C = linker.downcallHandle(s, sqdist, critical);
        s = find(lib, "numj_sumsq_muladd");
        MULADD = linker.downcallHandle(s, muladd);
        MULADD_C = linker.downcallHandle(s, muladd, critical);
        s = find(lib, "numj_sqdist_rows");
        SQDIST_ROWS = linker.downcallHandle(s, rows);
        SQDIST_ROWS_C = linker.downcallHandle(s, rows, critical);
        s = find(lib, "numj_normalize_rows");
        NORMALIZE = linker.downcallHandle(s, norm);
        NORMALIZE_C = linker.downcallHandle(s, norm, critical);
        s = find(lib, "numj_normalize_rows_inplace");
        NORMALIZE_IP = linker.downcallHandle(s, normIp);
        NORMALIZE_IP_C = linker.downcallHandle(s, normIp, critical);
        BUILD_INFO = linker.downcallHandle(find(lib, "numj_build_info"), FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_LONG));
    }

    private Native() {}

    static String buildInfo() {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment buf = a.allocate(4096);
            long n = (long) BUILD_INFO.invokeExact(buf, 4096L);
            return n <= 0 ? "" : buf.getString(0);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    private static MemorySegment find(SymbolLookup lib, String name) {
        return lib.find(name).orElseThrow(() -> new UnsatisfiedLinkError("symbol " + name + " not found in " + LIBRARY));
    }

    private static Path resolve() {
        String p = System.getProperty("numj.library");
        if (p == null || p.isBlank()) p = System.getenv("NUMJ_LIBRARY");
        if (p != null && !p.isBlank()) return Path.of(p).toAbsolutePath();
        String os = System.getProperty("os.name").toLowerCase();
        String file = os.contains("win") ? "numj.dll" : os.contains("mac") ? "libnumj.dylib" : "libnumj.so";
        Path def = Path.of("build", "native", file).toAbsolutePath();
        if (!Files.exists(def))
            throw new UnsatisfiedLinkError("numj native library not found at " + def + "; build it or set -Dnumj.library=<path>");
        return def;
    }
}
