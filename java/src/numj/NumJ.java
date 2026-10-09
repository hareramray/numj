package numj;

import java.lang.foreign.MemorySegment;

/**
 * Fortran-backed float64 kernels operating directly on {@link F64Array} native memory (no copies).
 *
 * <h2>Numerical contract</h2>
 * <ul>
 *   <li><b>Strict IEEE-754 evaluation.</b> Each operation in the formulas below is rounded separately
 *       (no FMA contraction, no reassociation, true division). Only the <i>summation order</i> differs from a
 *       naive left-to-right loop; it is fixed (see below), so results are deterministic and bit-identical
 *       for any thread count.</li>
 *   <li><b>Summation order.</b> Blocks of {@value #BLOCK} elements; inside a block, element {@code j} goes to
 *       partial sum {@code j mod 16}; partials and block results are combined by fixed pairwise trees.
 *       Error bound for a sum of non-negative rounded terms {@code t_i}:
 *       {@code |computed - sum t_i| <= gamma(k) * sum t_i}, {@code gamma(k) = k u / (1 - k u)},
 *       {@code u = 2^-53}, {@code k = }{@link #summationDepth(long)}.</li>
 *   <li><b>Empty input:</b> reductions return {@code 0.0}; row operations with zero rows or zero columns
 *       do nothing (norms of zero-length rows are {@code 0.0}).</li>
 *   <li><b>NaN</b> in any input element makes the affected result NaN. <b>Infinity</b> propagates by IEEE rules
 *       ({@code Inf - Inf = NaN}). <b>Overflow</b> of an intermediate or of the result gives {@code +Inf};
 *       reductions are unscaled (like NumPy), except {@link #normalizeRows} which rescales (see there).</li>
 *   <li><b>Shapes</b> must match exactly; otherwise {@link IllegalArgumentException} before any native call.</li>
 *   <li><b>Aliasing.</b> Read-only inputs may alias each other. An output must either be disjoint from every
 *       other argument or (where documented) be exactly the same memory as the input; partial overlap is
 *       rejected with {@link IllegalArgumentException}.</li>
 * </ul>
 *
 * <h2>Threads</h2>
 * Single-threaded by default. {@link #setThreads(int)} enables the native OpenMP pool for large inputs only
 * (at least {@link #PARALLEL_MIN_ELEMENTS}). The library never starts threads otherwise. If your application
 * already parallelises calls from several Java threads, keep this at 1 to avoid oversubscription.
 */
public final class NumJ {
    /** Elements per summation block (must match the Fortran BLOCK parameter). */
    public static final int BLOCK = 4096;
    /** Partial sums per block (must match the Fortran LANES parameter). */
    public static final int LANES = 16;
    /** Inputs smaller than this always run on the calling thread, whatever {@link #threads()} says. */
    public static final long PARALLEL_MIN_ELEMENTS = 1L << 18;
    /**
     * Single-threaded calls up to this many elements use a critical (lower-overhead) downcall.
     * Override with {@code -Dnumj.criticalMaxElements=<n>} ({@code -1} disables critical calls).
     */
    public static final long CRITICAL_MAX_ELEMENTS = Long.getLong("numj.criticalMaxElements", 1L << 16);

    private static volatile int threads = 1;

    private NumJ() {}

    /** Number of native threads used for inputs of at least {@link #PARALLEL_MIN_ELEMENTS} elements. */
    public static int threads() { return threads; }

    public static void setThreads(int n) {
        if (n < 1) throw new IllegalArgumentException("threads must be >= 1, got " + n);
        threads = n;
    }

    /** Compiler version and options the native library was built with. */
    public static String buildInfo() { return Native.buildInfo(); }

    /** Path of the loaded native library. */
    public static String libraryPath() { return Native.LIBRARY.toString(); }

    /**
     * Worst-case number of rounded additions any term passes through in a numj sum of {@code n} terms:
     * {@code ceil(min(n, BLOCK) / LANES) + log2(LANES) + ceil(log2(ceil(n / BLOCK)))}.
     */
    public static int summationDepth(long n) {
        if (n <= 1) return 0;
        long nb = (n + BLOCK - 1) / BLOCK;
        int inBlock = (int) ((Math.min(n, BLOCK) + LANES - 1) / LANES);
        int blocks = nb <= 1 ? 0 : 64 - Long.numberOfLeadingZeros(nb - 1);
        return inBlock + 4 + blocks;
    }

    // ---------------------------------------------------------------- kernels

    /** Squared Euclidean distance {@code sum_i (a_i - b_i)^2}. {@code a} and {@code b} must have the same shape. */
    public static double sqdist(F64Array a, F64Array b) {
        a.requireAlive();
        b.requireAlive();
        requireSameShape("b", b, a);
        long n = a.size();
        int t = threadsFor(n);
        try {
            return useCritical(n, t)
                    ? (double) Native.SQDIST_C.invokeExact(a.segment(), b.segment(), n, t)
                    : (double) Native.SQDIST.invokeExact(a.segment(), b.segment(), n, t);
        } catch (Throwable e) {
            throw rethrow(e);
        }
    }

    /**
     * {@code sum_i (a_i * b_i + c_i)^2} in one pass, with {@code a_i*b_i} and {@code +c_i} rounded separately
     * (identical per-element values to NumPy's {@code (a*b+c)**2}). All three must have the same shape.
     */
    public static double sumSqMulAdd(F64Array a, F64Array b, F64Array c) {
        a.requireAlive();
        b.requireAlive();
        c.requireAlive();
        requireSameShape("b", b, a);
        requireSameShape("c", c, a);
        long n = a.size();
        int t = threadsFor(n);
        try {
            return useCritical(n, t)
                    ? (double) Native.MULADD_C.invokeExact(a.segment(), b.segment(), c.segment(), n, t)
                    : (double) Native.MULADD.invokeExact(a.segment(), b.segment(), c.segment(), n, t);
        } catch (Throwable e) {
            throw rethrow(e);
        }
    }

    /**
     * Batched squared distances from one query to every row: {@code out[i] = sqdist(x.row(i), q)}.
     * {@code x} is {@code [m, n]}, {@code q} has {@code n} elements, {@code out} has {@code m} elements and must
     * not overlap {@code x} or {@code q}. Each {@code out[i]} is bit-identical to {@code sqdist(x.row(i), q)}.
     */
    public static void sqdistRows(F64Array q, F64Array x, F64Array out) {
        q.requireAlive();
        x.requireAlive();
        out.requireAlive();
        requireRank2("x", x);
        if (q.size() != x.cols())
            throw new IllegalArgumentException("q has " + q.size() + " elements, x has " + x.cols() + " columns");
        if (out.size() != x.rows())
            throw new IllegalArgumentException("out has " + out.size() + " elements, x has " + x.rows() + " rows");
        requireDisjoint("out", out, "x", x);
        requireDisjoint("out", out, "q", q);
        long n = x.size();
        int t = threadsFor(n);
        try {
            if (useCritical(n, t))
                Native.SQDIST_ROWS_C.invokeExact(q.segment(), x.segment(), x.rows(), x.cols(), out.segment(), t);
            else
                Native.SQDIST_ROWS.invokeExact(q.segment(), x.segment(), x.rows(), x.cols(), out.segment(), t);
        } catch (Throwable e) {
            throw rethrow(e);
        }
    }

    /**
     * Normalises every row of the rank-2 array {@code x} to unit Euclidean length, writing to {@code out}
     * (same shape; may be exactly {@code x} for in-place, otherwise must not overlap it). If {@code norms} is
     * non-null it receives each row's norm ({@code x.rows()} elements, disjoint from {@code x} and {@code out}).
     *
     * <p>Per-row rules:
     * <ul>
     *   <li>zero-length row: nothing to write, norm {@code 0.0};</li>
     *   <li>norm zero (all elements are {@code +-0.0}): row copied unchanged, norm {@code 0.0} (no NaN);</li>
     *   <li>any NaN in the row: whole output row NaN, norm NaN;</li>
     *   <li>any {@code +-Inf} (no NaN): norm {@code +Inf}; output {@code x / Inf} (finite &rarr; {@code +-0}, infinite &rarr; NaN);</li>
     *   <li>sum of squares overflows or falls below {@code 2^-968}: recomputed with scaling by the row's max
     *       |element|, so the normalised row is accurate even for huge or tiny values (where a naive
     *       {@code x / sqrt(sum(x*x))} returns zeros or NaN). The reported norm is {@code +Inf} only if the true
     *       norm exceeds {@link Double#MAX_VALUE}.</li>
     * </ul>
     * Output elements are within {@code (ncols + 8) u} relative error of the exact {@code x / ||x||}.
     */
    public static void normalizeRows(F64Array x, F64Array out, F64Array norms) {
        x.requireAlive();
        out.requireAlive();
        requireRank2("x", x);
        requireSameShape("out", out, x);
        boolean inPlace = out.sameMemory(x);
        if (!inPlace) requireDisjoint("out", out, "x", x);
        MemorySegment ns = normsSegment(norms, x, out);
        long n = x.size();
        int t = threadsFor(n);
        try {
            boolean c = useCritical(n, t);
            if (inPlace) {
                if (c) Native.NORMALIZE_IP_C.invokeExact(x.segment(), x.rows(), x.cols(), ns, t);
                else Native.NORMALIZE_IP.invokeExact(x.segment(), x.rows(), x.cols(), ns, t);
            } else {
                if (c) Native.NORMALIZE_C.invokeExact(x.segment(), out.segment(), x.rows(), x.cols(), ns, t);
                else Native.NORMALIZE.invokeExact(x.segment(), out.segment(), x.rows(), x.cols(), ns, t);
            }
        } catch (Throwable e) {
            throw rethrow(e);
        }
    }

    /** In-place {@link #normalizeRows}; {@code norms} may be null. */
    public static void normalizeRowsInPlace(F64Array x, F64Array norms) {
        normalizeRows(x, x, norms);
    }

    // ---------------------------------------------------------------- helpers

    private static MemorySegment normsSegment(F64Array norms, F64Array x, F64Array out) {
        if (norms == null) return MemorySegment.NULL;
        norms.requireAlive();
        if (norms.size() != x.rows())
            throw new IllegalArgumentException("norms has " + norms.size() + " elements, x has " + x.rows() + " rows");
        requireDisjoint("norms", norms, "x", x);
        requireDisjoint("norms", norms, "out", out);
        return norms.segment();
    }

    private static int threadsFor(long n) {
        int t = threads;
        return (t > 1 && n >= PARALLEL_MIN_ELEMENTS) ? t : 1;
    }

    private static boolean useCritical(long n, int t) {
        return t == 1 && n <= CRITICAL_MAX_ELEMENTS;
    }

    private static void requireSameShape(String name, F64Array got, F64Array ref) {
        if (!got.sameShape(ref))
            throw new IllegalArgumentException("shape mismatch: " + name + " is " + got.shapeString() + ", expected " + ref.shapeString());
    }

    private static void requireRank2(String name, F64Array a) {
        if (a.rank() != 2) throw new IllegalArgumentException(name + " must be rank 2, got " + a.shapeString());
    }

    private static void requireDisjoint(String an, F64Array a, String bn, F64Array b) {
        if (a.overlaps(b)) throw new IllegalArgumentException(an + " overlaps " + bn);
    }

    private static RuntimeException rethrow(Throwable e) {
        if (e instanceof RuntimeException r) return r;
        if (e instanceof Error err) throw err;
        return new IllegalStateException(e);
    }
}
