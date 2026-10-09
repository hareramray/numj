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
 * Downcall handles for the Fortran kernels (native/numj_kernels.f90, numj_elementwise.f90, numj_reduce.f90).
 *
 * <p>All handles are regular downcalls. The four fused kernels of numj 0.1 additionally have a
 * {@link Linker.Option#critical critical} handle, which skips the Java-to-native thread-state transition but
 * holds off safepoints (and therefore GC) for the duration of the call. {@link NumJ} uses it only when the
 * call provably satisfies the documented constraints of critical functions (see {@code NumJ.useCritical}):
 * no upcalls, no blocking, no OpenMP region, no heap allocation, and a measured, bounded run time.
 *
 * <p>Library resolution: see {@link NativeLoader}.
 */
@SuppressWarnings("restricted") // native linkage is this class's whole purpose; callers need --enable-native-access
final class Native {
    static final int EXPECTED_ABI = 2;
    static final Path LIBRARY = NativeLoader.resolve();

    // ---- fused kernels (numj 0.1, row kernels gained leading dimensions in ABI 2)
    // double numj_sqdist(const double* a, const double* b, int64 n, int32 nthreads)
    static final MethodHandle SQDIST, SQDIST_C;
    // double numj_sumsq_muladd(const double* a, const double* b, const double* c, int64 n, int32 nthreads)
    static final MethodHandle MULADD, MULADD_C;
    // void numj_sqdist_rows(const double* q, const double* x, int64 nrows, int64 ncols, int64 ldx, double* out, int32 nthreads)
    static final MethodHandle SQDIST_ROWS, SQDIST_ROWS_C;
    // void numj_normalize_rows(const double* x, int64 ldx, double* y, int64 ldy, int64 nrows, int64 ncols, double* norms|NULL, int32 nthreads)
    static final MethodHandle NORMALIZE, NORMALIZE_C;
    // void numj_normalize_rows_inplace(double* x, int64 ldx, int64 nrows, int64 ncols, double* norms|NULL, int32 nthreads)
    static final MethodHandle NORMALIZE_IP, NORMALIZE_IP_C;

    // ---- reductions
    // double numj_sum(const double* x, int64 n, int32 nthreads)
    static final MethodHandle SUM;
    // double numj_reduce_seq(int32 op, int32 nd, const int64* desc, const double* a, int64 oa, b, ob, c, oc)
    static final MethodHandle REDUCE_SEQ;
    // void numj_sum_nd(int32 nk, int32 nr, const int64* desc, const double* x, int64 ox, double* out, int64 oo, int32 nthreads)
    static final MethodHandle SUM_ND;

    // ---- elementwise
    // void numj_ew_contig(int32 op, int64 n, const double* a, const double* b, double* out, int32 nthreads)
    static final MethodHandle EW_CONTIG;
    // void numj_ew_contig_s(int32 op, int64 n, const double* a, double s, double* out, int32 nthreads)
    static final MethodHandle EW_CONTIG_S;
    // void numj_ew_contig_ip(int32 op, int64 n, double* x, const double* y, int32 nthreads)
    static final MethodHandle EW_CONTIG_IP;
    // void numj_ew_contig_ip_s(int32 op, int64 n, double* x, double s, int32 nthreads)
    static final MethodHandle EW_CONTIG_IP_S;
    // void numj_ew_contig_self(int32 op, int64 n, double* x, int32 nthreads)
    static final MethodHandle EW_CONTIG_SELF;
    // void numj_ew_nd(int32 op, int32 nd, const int64* desc, double* out, int64 oo, const double* a, int64 oa, const double* b, int64 ob, int32 nthreads)
    static final MethodHandle EW_ND;
    // void numj_ew_nd_ip(int32 op, int32 nd, const int64* desc, double* x, int64 ox, const double* y, int64 oy, int32 nthreads)
    static final MethodHandle EW_ND_IP;
    // void numj_ew_nd_self(int32 op, int32 nd, const int64* desc, double* x, int64 ox, int32 nthreads)
    static final MethodHandle EW_ND_SELF;
    // void numj_copy_nd(int32 nd, const int64* desc, double* out, int64 oo, const double* a, int64 oa, int32 nthreads)
    static final MethodHandle COPY_ND;

    // int64 numj_build_info(char* buf, int64 cap)
    private static final MethodHandle BUILD_INFO;

    static {
        Linker linker = Linker.nativeLinker();
        SymbolLookup lib = SymbolLookup.libraryLookup(LIBRARY, Arena.global());
        Linker.Option critical = Linker.Option.critical(false);

        MemorySegment abiSym = find(lib, "numj_abi_version");
        int abi;
        try {
            abi = (int) linker.downcallHandle(abiSym, FunctionDescriptor.of(JAVA_INT)).invokeExact();
        } catch (Throwable t) {
            throw new ExceptionInInitializerError(t);
        }
        if (abi != EXPECTED_ABI)
            throw new UnsatisfiedLinkError("numj native ABI " + abi + " != expected " + EXPECTED_ABI + " (" + LIBRARY + ")");

        FunctionDescriptor sqdist = FunctionDescriptor.of(JAVA_DOUBLE, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT);
        FunctionDescriptor muladd = FunctionDescriptor.of(JAVA_DOUBLE, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT);
        FunctionDescriptor rows = FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_INT);
        FunctionDescriptor norm = FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_INT);
        FunctionDescriptor normIp = FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_INT);

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

        SUM = linker.downcallHandle(find(lib, "numj_sum"), FunctionDescriptor.of(JAVA_DOUBLE, ADDRESS, JAVA_LONG, JAVA_INT));
        REDUCE_SEQ = linker.downcallHandle(find(lib, "numj_reduce_seq"), FunctionDescriptor.of(JAVA_DOUBLE,
                JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG));
        SUM_ND = linker.downcallHandle(find(lib, "numj_sum_nd"), FunctionDescriptor.ofVoid(
                JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT));

        EW_CONTIG = linker.downcallHandle(find(lib, "numj_ew_contig"),
                FunctionDescriptor.ofVoid(JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS, JAVA_INT));
        EW_CONTIG_S = linker.downcallHandle(find(lib, "numj_ew_contig_s"),
                FunctionDescriptor.ofVoid(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_DOUBLE, ADDRESS, JAVA_INT));
        EW_CONTIG_IP = linker.downcallHandle(find(lib, "numj_ew_contig_ip"),
                FunctionDescriptor.ofVoid(JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS, JAVA_INT));
        EW_CONTIG_IP_S = linker.downcallHandle(find(lib, "numj_ew_contig_ip_s"),
                FunctionDescriptor.ofVoid(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_DOUBLE, JAVA_INT));
        EW_CONTIG_SELF = linker.downcallHandle(find(lib, "numj_ew_contig_self"),
                FunctionDescriptor.ofVoid(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_INT));
        EW_ND = linker.downcallHandle(find(lib, "numj_ew_nd"), FunctionDescriptor.ofVoid(
                JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT));
        EW_ND_IP = linker.downcallHandle(find(lib, "numj_ew_nd_ip"), FunctionDescriptor.ofVoid(
                JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT));
        EW_ND_SELF = linker.downcallHandle(find(lib, "numj_ew_nd_self"), FunctionDescriptor.ofVoid(
                JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT));
        COPY_ND = linker.downcallHandle(find(lib, "numj_copy_nd"), FunctionDescriptor.ofVoid(
                JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT));

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

    static RuntimeException rethrow(Throwable e) {
        if (e instanceof RuntimeException r) return r;
        if (e instanceof Error err) throw err;
        return new IllegalStateException(e);
    }
}
