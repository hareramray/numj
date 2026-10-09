package numj;

import java.lang.foreign.MemorySegment;

/**
 * Float64 array operations backed by Fortran kernels through the FFM API.
 *
 * <h2>Operations</h2>
 * <ul>
 *   <li><b>Elementwise</b> {@link #add}, {@link #subtract}, {@link #multiply}, {@link #divide}: NumPy
 *       broadcasting, array or {@code double} operands, optional reusable {@code out}.</li>
 *   <li><b>Reductions</b> {@link #sum}, {@link #mean}: all elements, or selected axes with {@code keepdims} and
 *       optional {@code out}.</li>
 *   <li><b>Fused kernels</b> (one pass, no temporaries): {@link #sqdist}, {@link #sumSqMulAdd},
 *       {@link #sqdistRows}, {@link #normalizeRows}. See also {@link Expr} for explicit pattern-based fusion.</li>
 * </ul>
 *
 * <h2>Results and outputs</h2>
 * Methods without {@code out} return a new C-ordered owner that the caller must close. With {@code out}, the
 * result is written there and {@code out} is returned. {@code out} must be alive, writable
 * ({@link ReadOnlyArrayException} otherwise) and have exactly the result shape ({@link IllegalArgumentException}
 * otherwise; NumPy additionally accepts an {@code out} the inputs broadcast to). If {@code out} shares memory
 * with an input, the result is as if the input had been copied first: an input that addresses exactly the
 * output's elements is updated in place, any other overlap is resolved by a temporary copy of that input.
 * All validation happens before any native call.
 *
 * <h2>Numerical contract</h2>
 * <ul>
 *   <li><b>Strict IEEE-754 evaluation.</b> Each {@code +}, {@code -}, {@code *}, {@code /} is rounded separately
 *       (no FMA contraction, no reassociation, true division, no flush-to-zero). Elementwise results are therefore
 *       bitwise identical to NumPy's and to plain Java arithmetic, for every layout, path and thread count;
 *       NaN, infinities and signed zeros follow IEEE rules ({@code 0*Inf = NaN}, {@code 1/-0.0 = -Inf},
 *       {@code -0.0 - 0.0 = -0.0}). NaN payloads are not specified.</li>
 *   <li><b>Summation order.</b> Every reduction sums a logical sequence (C order of the reduced axes) in blocks of
 *       {@value #BLOCK} elements; inside a block element {@code j} goes to partial sum {@code j mod 16}; partials and
 *       block results are combined by fixed pairwise trees. Results depend only on the values and the shape, never
 *       on the memory layout, the code path or the thread count. Partial sums start at {@code +0.0}, so (as in
 *       NumPy) a sum of {@code -0.0} values is {@code +0.0}.</li>
 *   <li><b>Accuracy.</b> For a sum of {@code n} terms {@code t_i}:
 *       {@code |computed - sum t_i| <= gamma(k) * sum |t_i|}, {@code gamma(k) = k u / (1 - k u)},
 *       {@code u = 2^-53}, {@code k = }{@link #summationDepth(long)} (at most {@code 260 + ceil(log2(n/4096))};
 *       a naive loop has {@code k = n - 1}). For the fused kernels {@code t_i} are the rounded per-element terms.
 *       {@link #mean} divides the sum by the count with one more rounding.</li>
 *   <li><b>Empty reductions:</b> sums are {@code +0.0}; means are NaN (NumPy also warns; numj does not).</li>
 *   <li><b>Overflow</b> gives {@code +-Inf}; {@code Inf - Inf} gives NaN; NaN propagates.</li>
 * </ul>
 *
 * <h2>Threads</h2>
 * Single-threaded by default. {@link #setThreads(int)} lets large calls use the native OpenMP pool (reductions from
 * {@link #PARALLEL_MIN_ELEMENTS} elements, elementwise operations from {@link #EW_PARALLEL_MIN_ELEMENTS}).
 * If your application already parallelises calls from several Java threads, keep this at 1.
 *
 * <h2>Downcalls</h2>
 * All native calls are regular FFM downcalls by default. Elementwise operations on a handful of elements (and
 * strided ones on up to ~64) are computed by an equivalent Java loop instead (same bits), where that measured
 * cheaper than a downcall plus planning; reductions always go native, which measured faster at every size.
 * Critical downcalls
 * ({@link java.lang.foreign.Linker.Option#critical}) are documented for functions with "an extremely short running
 * time in all cases (similar to calling an empty function)", which an array kernel cannot promise in general;
 * they are therefore opt-in only (see {@link #CRITICAL_MAX_ELEMENTS}).
 */
public final class NumJ {
    /** Elements per summation block (must match the Fortran BLOCK parameter). */
    public static final int BLOCK = 4096;
    /** Partial sums per block (must match the Fortran LANES parameter). */
    public static final int LANES = 16;
    /** Reductions and fused kernels smaller than this always run on the calling thread. */
    public static final long PARALLEL_MIN_ELEMENTS = 1L << 19;
    /** Elementwise operations smaller than this always run on the calling thread. */
    public static final long EW_PARALLEL_MIN_ELEMENTS = Elementwise.parallelMinElements;
    /**
     * Opt-in: single-threaded calls of the four fused kernels with at most this many elements use a critical
     * downcall. Default {@code -1} (never). Set with {@code -Dnumj.criticalMaxElements=<n>} only after checking
     * that calls of that size are short enough for your GC pause budget; the JDK documents critical functions
     * as having a running time similar to calling an empty function. (numj 0.1 defaulted to 65536.)
     */
    public static final long CRITICAL_MAX_ELEMENTS = Long.getLong("numj.criticalMaxElements", -1);

    /** Fused kernels with at most this many elements run as a Java loop (bitwise identical). */
    static long fusedJavaMaxElements = Long.getLong("numj.fused.javaMaxElements", 0);   // measured: native wins

    private static volatile int threads = 1;

    private NumJ() {}

    /** Number of native threads used for large inputs. */
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

    /** NumPy {@code broadcast_shapes}: the common shape, or {@link IllegalArgumentException}. */
    public static long[] broadcastShapes(long[]... shapes) {
        return Layout.broadcastShapes(shapes);
    }

    // ================================================================== elementwise

    /** {@code a + b} (broadcast), new array. */
    public static F64Array add(F64Array a, F64Array b) { return Elementwise.binary(Elementwise.ADD, a, b, null); }
    /** {@code a + b} (broadcast) into {@code out}. */
    public static F64Array add(F64Array a, F64Array b, F64Array out) { return Elementwise.binary(Elementwise.ADD, a, b, nn(out)); }
    public static F64Array add(F64Array a, double b) { return Elementwise.scalar(Elementwise.ADD, a, b, null); }
    public static F64Array add(F64Array a, double b, F64Array out) { return Elementwise.scalar(Elementwise.ADD, a, b, nn(out)); }
    public static F64Array add(double a, F64Array b) { return Elementwise.scalar(Elementwise.ADD, b, a, null); }
    public static F64Array add(double a, F64Array b, F64Array out) { return Elementwise.scalar(Elementwise.ADD, b, a, nn(out)); }

    /** {@code a - b} (broadcast), new array. */
    public static F64Array subtract(F64Array a, F64Array b) { return Elementwise.binary(Elementwise.SUB, a, b, null); }
    public static F64Array subtract(F64Array a, F64Array b, F64Array out) { return Elementwise.binary(Elementwise.SUB, a, b, nn(out)); }
    public static F64Array subtract(F64Array a, double b) { return Elementwise.scalar(Elementwise.SUB, a, b, null); }
    public static F64Array subtract(F64Array a, double b, F64Array out) { return Elementwise.scalar(Elementwise.SUB, a, b, nn(out)); }
    public static F64Array subtract(double a, F64Array b) { return Elementwise.scalar(Elementwise.RSUB, b, a, null); }
    public static F64Array subtract(double a, F64Array b, F64Array out) { return Elementwise.scalar(Elementwise.RSUB, b, a, nn(out)); }

    /** {@code a * b} (broadcast), new array. */
    public static F64Array multiply(F64Array a, F64Array b) { return Elementwise.binary(Elementwise.MUL, a, b, null); }
    public static F64Array multiply(F64Array a, F64Array b, F64Array out) { return Elementwise.binary(Elementwise.MUL, a, b, nn(out)); }
    public static F64Array multiply(F64Array a, double b) { return Elementwise.scalar(Elementwise.MUL, a, b, null); }
    public static F64Array multiply(F64Array a, double b, F64Array out) { return Elementwise.scalar(Elementwise.MUL, a, b, nn(out)); }
    public static F64Array multiply(double a, F64Array b) { return Elementwise.scalar(Elementwise.MUL, b, a, null); }
    public static F64Array multiply(double a, F64Array b, F64Array out) { return Elementwise.scalar(Elementwise.MUL, b, a, nn(out)); }

    /** {@code a / b} (IEEE true division, broadcast), new array. */
    public static F64Array divide(F64Array a, F64Array b) { return Elementwise.binary(Elementwise.DIV, a, b, null); }
    public static F64Array divide(F64Array a, F64Array b, F64Array out) { return Elementwise.binary(Elementwise.DIV, a, b, nn(out)); }
    public static F64Array divide(F64Array a, double b) { return Elementwise.scalar(Elementwise.DIV, a, b, null); }
    public static F64Array divide(F64Array a, double b, F64Array out) { return Elementwise.scalar(Elementwise.DIV, a, b, nn(out)); }
    public static F64Array divide(double a, F64Array b) { return Elementwise.scalar(Elementwise.RDIV, b, a, null); }
    public static F64Array divide(double a, F64Array b, F64Array out) { return Elementwise.scalar(Elementwise.RDIV, b, a, nn(out)); }

    // ================================================================== reductions

    /** Sum of all elements ({@code +0.0} if empty). */
    public static double sum(F64Array a) { return Reduce.sumAll(a); }

    /** Sum over one axis (negative counts from the end); new array. */
    public static F64Array sum(F64Array a, int axis, boolean keepdims) {
        return Reduce.sum(a, new int[] {axis}, keepdims, null, false);
    }

    /** Sum over {@code axes} ({@code null} = all; empty = none); new array. Duplicate or out-of-range axes throw. */
    public static F64Array sum(F64Array a, int[] axes, boolean keepdims) {
        return Reduce.sum(a, axes, keepdims, null, false);
    }

    /** Sum over {@code axes} into {@code out}. */
    public static F64Array sum(F64Array a, int[] axes, boolean keepdims, F64Array out) {
        return Reduce.sum(a, axes, keepdims, nn(out), false);
    }

    /** Mean of all elements (NaN if empty). */
    public static double mean(F64Array a) {
        long n = a.size();
        double s = Reduce.sumAll(a);
        return n == 0 ? Double.NaN : s / n;
    }

    /** Mean over one axis; new array. */
    public static F64Array mean(F64Array a, int axis, boolean keepdims) {
        return Reduce.sum(a, new int[] {axis}, keepdims, null, true);
    }

    /** Mean over {@code axes} ({@code null} = all); new array. Means over empty axes are NaN. */
    public static F64Array mean(F64Array a, int[] axes, boolean keepdims) {
        return Reduce.sum(a, axes, keepdims, null, true);
    }

    /** Mean over {@code axes} into {@code out}. */
    public static F64Array mean(F64Array a, int[] axes, boolean keepdims, F64Array out) {
        return Reduce.sum(a, axes, keepdims, nn(out), true);
    }

    // ================================================================== fused kernels

    /**
     * Squared Euclidean distance {@code sum_i (a_i - b_i)^2}. {@code a} and {@code b} must have the same shape
     * (no broadcasting); any layout. Same bits for every layout.
     */
    public static double sqdist(F64Array a, F64Array b) {
        a.requireAlive();
        b.requireAlive();
        requireSameShape("b", b, a);
        long n = a.size();
        if (n <= fusedJavaMaxElements || !a.cContig || !b.cContig) return Reduce.seq(Reduce.K_SQDIST, a, b, a, fusedJavaMaxElements);
        int t = threadsFor(n, Reduce.parallelMinElements);
        try {
            return useCritical(n, t)
                    ? (double) Native.SQDIST_C.invokeExact(a.segment(), b.segment(), n, t)
                    : (double) Native.SQDIST.invokeExact(a.segment(), b.segment(), n, t);
        } catch (Throwable e) {
            throw Native.rethrow(e);
        }
    }

    /**
     * {@code sum_i (a_i * b_i + c_i)^2} in one pass, with {@code a_i*b_i} and {@code +c_i} rounded separately
     * (identical per-element values to NumPy's {@code (a*b+c)**2}). All three must have the same shape; any layout.
     */
    public static double sumSqMulAdd(F64Array a, F64Array b, F64Array c) {
        a.requireAlive();
        b.requireAlive();
        c.requireAlive();
        requireSameShape("b", b, a);
        requireSameShape("c", c, a);
        long n = a.size();
        if (n <= fusedJavaMaxElements || !a.cContig || !b.cContig || !c.cContig) return Reduce.seq(Reduce.K_MULADD, a, b, c, fusedJavaMaxElements);
        int t = threadsFor(n, Reduce.parallelMinElements);
        try {
            return useCritical(n, t)
                    ? (double) Native.MULADD_C.invokeExact(a.segment(), b.segment(), c.segment(), n, t)
                    : (double) Native.MULADD.invokeExact(a.segment(), b.segment(), c.segment(), n, t);
        } catch (Throwable e) {
            throw Native.rethrow(e);
        }
    }

    /**
     * Batched squared distances from one query to every row: {@code out[i] = sqdist(x.row(i), q)}.
     * {@code x} is {@code [m, n]} with unit-stride rows in increasing memory order (any row spacing, e.g. a
     * row-stepped view); {@code q} has {@code n} elements and {@code out} {@code m} elements, both C-contiguous;
     * {@code out} must be writable and must not overlap {@code x} or {@code q}. Layouts outside these rules throw
     * {@link IllegalArgumentException} (copy the input first); nothing is copied implicitly.
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
        out.requireWritable("out");
        requireContiguous("q", q);
        requireContiguous("out", out);
        long ldx = leadingDim("x", x);
        requireDisjoint("out", out, "x", x);
        requireDisjoint("out", out, "q", q);
        long n = x.size();
        int t = threadsFor(n, Reduce.parallelMinElements);
        try {
            if (useCritical(n, t))
                Native.SQDIST_ROWS_C.invokeExact(q.segment(), x.segment(), x.rows(), x.cols(), ldx, out.segment(), t);
            else
                Native.SQDIST_ROWS.invokeExact(q.segment(), x.segment(), x.rows(), x.cols(), ldx, out.segment(), t);
        } catch (Throwable e) {
            throw Native.rethrow(e);
        }
    }

    /**
     * Normalises every row of the rank-2 array {@code x} to unit Euclidean length, writing to {@code out}
     * (same shape; may address exactly the same elements as {@code x} for in-place, otherwise must not overlap it).
     * {@code x} and {@code out} need unit-stride rows in increasing memory order (any row spacing). If
     * {@code norms} is non-null it receives each row's norm ({@code x.rows()} elements, C-contiguous, disjoint from
     * {@code x} and {@code out}).
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
        out.requireWritable("out");
        long ldx = leadingDim("x", x), ldo = leadingDim("out", out);
        boolean inPlace = out.size() > 0 && out.sameLayout(x);
        if (!inPlace) requireDisjoint("out", out, "x", x);
        MemorySegment ns = normsSegment(norms, x, out);
        long n = x.size();
        int t = threadsFor(n, Reduce.parallelMinElements);
        try {
            boolean c = useCritical(n, t);
            if (inPlace) {
                if (c) Native.NORMALIZE_IP_C.invokeExact(x.segment(), ldx, x.rows(), x.cols(), ns, t);
                else Native.NORMALIZE_IP.invokeExact(x.segment(), ldx, x.rows(), x.cols(), ns, t);
            } else {
                if (c) Native.NORMALIZE_C.invokeExact(x.segment(), ldx, out.segment(), ldo, x.rows(), x.cols(), ns, t);
                else Native.NORMALIZE.invokeExact(x.segment(), ldx, out.segment(), ldo, x.rows(), x.cols(), ns, t);
            }
        } catch (Throwable e) {
            throw Native.rethrow(e);
        }
    }

    /** In-place {@link #normalizeRows}; {@code norms} may be null. */
    public static void normalizeRowsInPlace(F64Array x, F64Array norms) {
        normalizeRows(x, x, norms);
    }

    // ================================================================== helpers

    private static F64Array nn(F64Array out) {
        if (out == null) throw new NullPointerException("out");
        return out;
    }

    private static MemorySegment normsSegment(F64Array norms, F64Array x, F64Array out) {
        if (norms == null) return MemorySegment.NULL;
        norms.requireAlive();
        if (norms.size() != x.rows())
            throw new IllegalArgumentException("norms has " + norms.size() + " elements, x has " + x.rows() + " rows");
        norms.requireWritable("norms");
        requireContiguous("norms", norms);
        requireDisjoint("norms", norms, "x", x);
        requireDisjoint("norms", norms, "out", out);
        return norms.segment();
    }

    /** Leading dimension (elements between row starts) of a rank-2 array with unit-stride, ascending rows. */
    private static long leadingDim(String name, F64Array a) {
        long rows = a.layout.shape[0], cols = a.layout.shape[1];
        long rs = a.layout.strides[0], cs = a.layout.strides[1];
        if (cols > 1 && cs != F64Array.ITEM)
            throw new IllegalArgumentException(name + " must have unit-stride rows (column stride " + cs
                    + " bytes); copy it first");
        if (rows <= 1) return Math.max(cols, 1);
        if (rs < cols * F64Array.ITEM || rs <= 0)
            throw new IllegalArgumentException(name + " rows must be in increasing memory order and not overlap "
                    + "(row stride " + rs + " bytes); copy it first");
        return rs / F64Array.ITEM;
    }

    private static void requireContiguous(String name, F64Array a) {
        if (!a.cContig) throw new IllegalArgumentException(name + " must be C-contiguous; copy it first");
    }

    static int threadsFor(long n, long threshold) {
        int t = threads;
        return (t > 1 && n >= threshold) ? t : 1;
    }

    private static boolean useCritical(long n, int t) {
        return t == 1 && n <= CRITICAL_MAX_ELEMENTS;
    }

    private static void requireSameShape(String name, F64Array got, F64Array ref) {
        if (!got.sameShape(ref))
            throw new IllegalArgumentException("shape mismatch: " + name + " is " + got.shapeString() + ", expected " + ref.shapeString());
    }

    private static void requireRank2(String name, F64Array a) {
        if (a.ndim() != 2) throw new IllegalArgumentException(name + " must be rank 2, got " + a.shapeString());
    }

    private static void requireDisjoint(String an, F64Array a, String bn, F64Array b) {
        if (a.overlaps(b)) throw new IllegalArgumentException(an + " overlaps " + bn);
    }
}
