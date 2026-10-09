package numj;

import numj.bench.Data;
import numj.bench.JavaKernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.math.BigDecimal;
import java.math.MathContext;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Correctness tests (no framework; exits non-zero on failure).
 *
 * Tolerances are derived, not guessed:
 *  - reductions: |numj - S| <= gamma(k) * S, where S is the EXACT (BigDecimal) sum of the per-element rounded
 *    terms and k = NumJ.summationDepth(n). All terms are >= 0, so this is a relative bound.
 *  - bitwise equality with JavaKernels.*Blocked (same algorithm, same IEEE operations).
 *  - normalisation: |y - x/||x||| <= (ncols + 8) u |x/||x||| against a 50-digit BigDecimal reference.
 */
public final class NumJTests {
    static final double U = 0x1p-53;
    static int passed, failed;
    static final List<String> failures = new ArrayList<>();

    public static void main(String[] args) {
        System.out.println("numj library: " + NumJ.libraryPath());
        System.out.println("build: " + NumJ.buildInfo());

        test("sqdist: exact-reference bound, many sizes", NumJTests::sqdistAccuracy);
        test("sqdist: bit-identical to Java blocked", NumJTests::sqdistBitwise);
        test("sumSqMulAdd: exact-reference bound + bitwise", NumJTests::mulAddAccuracy);
        test("reductions: identical bits for 1..8 threads", NumJTests::threadInvariance);
        test("reductions: empty input -> 0.0", NumJTests::emptyReductions);
        test("reductions: NaN / Inf / overflow", NumJTests::nonFinite);
        test("reductions: aliased read-only inputs", NumJTests::aliasedInputs);
        test("sqdistRows: equals per-row sqdist and Java", NumJTests::sqdistRows);
        test("normalizeRows: bitwise vs Java + reference bound", NumJTests::normalizeRandom);
        test("normalizeRows: zero, -0, empty rows", NumJTests::normalizeZeros);
        test("normalizeRows: NaN and Inf rows", NumJTests::normalizeNonFinite);
        test("normalizeRows: overflow/underflow-safe rows", NumJTests::normalizeExtremes);
        test("normalizeRows: in-place == out-of-place", NumJTests::normalizeInPlace);
        test("normalizeRows: multi-threaded == single", NumJTests::normalizeThreads);
        test("validation: shape mismatches rejected", NumJTests::shapeMismatch);
        test("validation: partial overlap rejected", NumJTests::overlap);
        test("lifetime: use after close, views, idempotent close", NumJTests::lifetime);
        test("bounds: index checks", NumJTests::bounds);
        test("threads: confined vs shared arrays", NumJTests::confinement);
        test("F64Array: copies, reshape, row views", NumJTests::arrayBasics);

        System.out.printf("%n%d passed, %d failed%n", passed, failed);
        failures.forEach(f -> System.out.println("  FAILED: " + f));
        System.exit(failed == 0 ? 0 : 1);
    }

    // ------------------------------------------------------------------ reductions

    static final int[] SIZES = {0, 1, 2, 15, 16, 17, 31, 4095, 4096, 4097, 8192 + 5, 65_536, 100_003, 300_001};

    static void sqdistAccuracy() {
        for (int n : SIZES) {
            double[] a = Data.splitmix(11, n), b = Data.splitmix(12, n);
            BigDecimal exact = BigDecimal.ZERO;
            for (int i = 0; i < n; i++) {
                double d = a[i] - b[i];
                exact = exact.add(new BigDecimal(d * d));
            }
            try (F64Array x = F64Array.of(a); F64Array y = F64Array.of(b)) {
                checkSumBound("n=" + n, NumJ.sqdist(x, y), exact, n);
            }
        }
    }

    static void sqdistBitwise() {
        for (int n : new int[] {0, 1, 17, 4097, 1_000_003, 3_000_000}) {
            double[] a = Data.splitmix(21, n), b = Data.splitmix(22, n);
            try (F64Array x = F64Array.of(a); F64Array y = F64Array.of(b)) {
                assertBits("n=" + n, JavaKernels.sqdistBlocked(a, b), NumJ.sqdist(x, y));
            }
        }
    }

    static void mulAddAccuracy() {
        for (int n : SIZES) {
            double[] a = Data.splitmix(31, n), b = Data.splitmix(32, n), c = Data.splitmix(33, n);
            BigDecimal exact = BigDecimal.ZERO;
            for (int i = 0; i < n; i++) {
                double t = a[i] * b[i] + c[i];
                exact = exact.add(new BigDecimal(t * t));
            }
            try (F64Array x = F64Array.of(a); F64Array y = F64Array.of(b); F64Array z = F64Array.of(c)) {
                double got = NumJ.sumSqMulAdd(x, y, z);
                checkSumBound("n=" + n, got, exact, n);
                assertBits("bitwise n=" + n, JavaKernels.sumSqMulAddBlocked(a, b, c), got);
            }
        }
    }

    static void threadInvariance() {
        int n = 5_000_011;
        try (F64Array a = F64Array.of(Data.splitmix(41, n)); F64Array b = F64Array.of(Data.splitmix(42, n));
             F64Array c = F64Array.of(Data.splitmix(43, n))) {
            NumJ.setThreads(1);
            double s1 = NumJ.sqdist(a, b), m1 = NumJ.sumSqMulAdd(a, b, c);
            for (int t = 2; t <= 8; t++) {
                NumJ.setThreads(t);
                assertBits("sqdist threads=" + t, s1, NumJ.sqdist(a, b));
                assertBits("muladd threads=" + t, m1, NumJ.sumSqMulAdd(a, b, c));
            }
        } finally {
            NumJ.setThreads(1);
        }
    }

    static void emptyReductions() {
        try (F64Array a = F64Array.allocate(0); F64Array b = F64Array.allocate(0); F64Array c = F64Array.allocate(0);
             F64Array m = F64Array.allocate(0, 7); F64Array m2 = F64Array.allocate(0, 7)) {
            assertBits("sqdist", 0.0, NumJ.sqdist(a, b));
            assertBits("muladd", 0.0, NumJ.sumSqMulAdd(a, b, c));
            assertBits("rank2 empty", 0.0, NumJ.sqdist(m, m2));
        }
    }

    static void nonFinite() {
        double inf = Double.POSITIVE_INFINITY, nan = Double.NaN;
        check("NaN in a", Double.isNaN(sq(new double[] {1, nan, 3}, new double[] {0, 0, 0})));
        check("NaN in b, long", Double.isNaN(sq(withAt(Data.splitmix(1, 50_000), 33_333, nan), Data.splitmix(2, 50_000))));
        assertBits("Inf - finite", inf, sq(new double[] {inf, 1}, new double[] {1, 1}));
        check("Inf - Inf -> NaN", Double.isNaN(sq(new double[] {inf}, new double[] {inf})));
        assertBits("overflow -> +Inf", inf, sq(new double[] {1e200, 0}, new double[] {-1e200, 0}));
        assertBits("square overflow", inf, sq(new double[] {1e155}, new double[] {0}));
        assertBits("subnormal squares underflow to 0", 0.0, sq(new double[] {1e-170}, new double[] {0}));
        try (F64Array a = F64Array.of(1e300, 1); F64Array b = F64Array.of(10, 1); F64Array c = F64Array.of(0, nan)) {
            assertBits("muladd overflow", inf, NumJ.sumSqMulAdd(a, b, F64Array.of(0, 0)));
            check("muladd NaN", Double.isNaN(NumJ.sumSqMulAdd(a, b, c)));
        }
    }

    static void aliasedInputs() {
        try (F64Array a = F64Array.of(Data.splitmix(5, 10_000))) {
            assertBits("sqdist(a, a)", 0.0, NumJ.sqdist(a, a));
            double[] h = a.toArray();
            assertBits("muladd(a, a, a)", JavaKernels.sumSqMulAddBlocked(h, h, h), NumJ.sumSqMulAdd(a, a, a));
        }
    }

    static void sqdistRows() {
        int[][] shapes = {{1, 1}, {7, 3}, {1000, 16}, {300, 4100}, {0, 5}, {5, 0}};
        for (int[] s : shapes) {
            int rows = s[0], cols = s[1];
            double[] xh = Data.splitmix(51, rows * cols), qh = Data.splitmix(52, cols);
            double[] javaOut = new double[rows];
            JavaKernels.sqdistRowsBlocked(qh, xh, rows, cols, javaOut);
            try (F64Array x = F64Array.copyOf(xh, rows, cols); F64Array q = F64Array.of(qh); F64Array out = F64Array.allocate(rows)) {
                NumJ.sqdistRows(q, x, out);
                for (int r = 0; r < rows; r++) {
                    assertBits("row " + r + " vs Java " + rows + "x" + cols, javaOut[r], out.get(r));
                    assertBits("row " + r + " vs sqdist", NumJ.sqdist(x.row(r), q), out.get(r));
                }
            }
        }
    }

    // ------------------------------------------------------------------ normalisation

    static void normalizeRandom() {
        int[][] shapes = {{1, 1}, {3, 2}, {50, 17}, {200, 100}, {4, 9000}, {10_000, 8}};
        for (int[] s : shapes) {
            int rows = s[0], cols = s[1];
            double[] xh = Data.splitmix(61, rows * cols);
            double[] yj = new double[xh.length], nj = new double[rows];
            JavaKernels.normalizeRowsBlocked(xh, yj, rows, cols, nj);
            try (F64Array x = F64Array.copyOf(xh, rows, cols); F64Array y = F64Array.allocate(rows, cols);
                 F64Array norms = F64Array.allocate(rows)) {
                NumJ.normalizeRows(x, y, norms);
                double[] yh = y.toArray();
                for (int i = 0; i < yh.length; i++) assertBits("y[" + i + "] " + rows + "x" + cols, yj[i], yh[i]);
                for (int r = 0; r < rows; r++) assertBits("norm[" + r + "]", nj[r], norms.get(r));
                if (rows * cols <= 40_000) checkNormalizeReference(xh, yh, rows, cols);
            }
        }
    }

    static void normalizeZeros() {
        double[] xh = {0, 0, 0, -0.0, 0, -0.0, 3, 4, 0};
        try (F64Array x = F64Array.copyOf(xh, 3, 3); F64Array y = F64Array.allocate(3, 3); F64Array n = F64Array.allocate(3)) {
            y.fill(99);
            NumJ.normalizeRows(x, y, n);
            assertBits("zero row norm", 0.0, n.get(0));
            assertBits("-0 row norm", 0.0, n.get(1));
            for (int j = 0; j < 3; j++) assertBits("zero row copied " + j, xh[j], y.get(0, j));
            for (int j = 0; j < 3; j++) assertBits("-0 row copied " + j, xh[3 + j], y.get(1, j));
            assertBits("3-4-0 row", 0.6, y.get(2, 0));
            assertBits("3-4-0 row", 0.8, y.get(2, 1));
            assertBits("norm 5", 5.0, n.get(2));
        }
        try (F64Array x = F64Array.allocate(4, 0); F64Array y = F64Array.allocate(4, 0); F64Array n = F64Array.allocate(4)) {
            n.fill(7);
            NumJ.normalizeRows(x, y, n);
            for (int r = 0; r < 4; r++) assertBits("zero-length row norm", 0.0, n.get(r));
        }
        try (F64Array x = F64Array.allocate(0, 5); F64Array y = F64Array.allocate(0, 5); F64Array n = F64Array.allocate(0)) {
            NumJ.normalizeRows(x, y, n);   // no rows: must simply return
        }
    }

    static void normalizeNonFinite() {
        double inf = Double.POSITIVE_INFINITY, nan = Double.NaN;
        double[] xh = {1, nan, 2, inf, 1, -inf, 1, 2, 3};
        try (F64Array x = F64Array.copyOf(xh, 3, 3); F64Array y = F64Array.allocate(3, 3); F64Array n = F64Array.allocate(3)) {
            NumJ.normalizeRows(x, y, n);
            check("NaN row norm", Double.isNaN(n.get(0)));
            for (int j = 0; j < 3; j++) check("NaN row element " + j, Double.isNaN(y.get(0, j)));
            assertBits("Inf row norm", inf, n.get(1));
            check("Inf/Inf -> NaN", Double.isNaN(y.get(1, 0)));
            assertBits("finite/Inf -> 0", 0.0, y.get(1, 1));
            check("-Inf/Inf -> NaN", Double.isNaN(y.get(1, 2)));
            check("normal row unaffected", Math.abs(y.get(2, 2) - 3 / Math.sqrt(14)) <= 2 * U);
        }
    }

    static void normalizeExtremes() {
        double big = 1e300, tiny = 1e-300, max = Double.MAX_VALUE, sub = Double.MIN_VALUE;
        double[][] rows = {
            {big, big, -big},              // sum of squares overflows
            {tiny, -tiny, tiny, tiny},     // squares underflow to 0
            {max, max, max, max},          // true norm 2*MAX overflows; result must still be 0.5 each
            {sub, 0, sub, 0},              // subnormal
            {1e-160, 3e-160, 0, 0},        // squares subnormal
            {3e200, 4e200},                // classic 3-4-5, scaled
        };
        for (double[] row : rows) {
            int cols = row.length;
            try (F64Array x = F64Array.copyOf(row, 1, cols); F64Array y = F64Array.allocate(1, cols); F64Array n = F64Array.allocate(1)) {
                NumJ.normalizeRows(x, y, n);
                double[] yh = y.toArray();
                checkNormalizeReference(row, yh, 1, cols);
                double[] yj = new double[cols], nj = new double[1];
                JavaKernels.normalizeRowsBlocked(row, yj, 1, cols, nj);
                for (int j = 0; j < cols; j++) assertBits("vs Java " + row[0], yj[j], yh[j]);
                assertBits("norm vs Java " + row[0], nj[0], n.get(0));
            }
        }
        try (F64Array x = F64Array.copyOf(new double[] {max, max, max, max}, 1, 4); F64Array y = F64Array.allocate(1, 4);
             F64Array n = F64Array.allocate(1)) {
            NumJ.normalizeRows(x, y, n);
            assertBits("norm > MAX reported +Inf", Double.POSITIVE_INFINITY, n.get(0));
            assertBits("row still normalised", 0.5, y.get(0, 0));
        }
    }

    static void normalizeInPlace() {
        int rows = 300, cols = 37;
        double[] xh = Data.splitmix(71, rows * cols);
        xh[5 * cols + 3] = 1e300;   // one rescaled row
        for (int j = 0; j < cols; j++) xh[9 * cols + j] = 0;  // one zero row
        try (F64Array x = F64Array.copyOf(xh, rows, cols); F64Array y = F64Array.allocate(rows, cols);
             F64Array n1 = F64Array.allocate(rows); F64Array n2 = F64Array.allocate(rows)) {
            NumJ.normalizeRows(x, y, n1);
            NumJ.normalizeRowsInPlace(x, n2);
            for (long i = 0; i < x.size(); i++) assertBits("element " + i, y.get(i), x.get(i));
            for (long r = 0; r < rows; r++) assertBits("norm " + r, n1.get(r), n2.get(r));
            NumJ.normalizeRows(y, y, null);    // same memory passed twice == in-place, norms optional
        }
    }

    static void normalizeThreads() {
        int rows = 20_000, cols = 50;
        try (F64Array x = F64Array.copyOf(Data.splitmix(81, rows * cols), rows, cols); F64Array y1 = F64Array.allocate(rows, cols);
             F64Array y4 = F64Array.allocate(rows, cols); F64Array d1 = F64Array.allocate(rows); F64Array d4 = F64Array.allocate(rows);
             F64Array q = F64Array.of(Data.splitmix(82, cols))) {
            NumJ.normalizeRows(x, y1, null);
            NumJ.sqdistRows(q, x, d1);
            NumJ.setThreads(4);
            NumJ.normalizeRows(x, y4, null);
            NumJ.sqdistRows(q, x, d4);
            for (long i = 0; i < y1.size(); i++) assertBits("normalize " + i, y1.get(i), y4.get(i));
            for (long r = 0; r < rows; r++) assertBits("sqdistRows " + r, d1.get(r), d4.get(r));
        } finally {
            NumJ.setThreads(1);
        }
    }

    // ------------------------------------------------------------------ validation & memory model

    static void shapeMismatch() {
        try (F64Array a10 = F64Array.allocate(10); F64Array a11 = F64Array.allocate(11); F64Array m25 = F64Array.allocate(2, 5);
             F64Array m52 = F64Array.allocate(5, 2); F64Array n3 = F64Array.allocate(3); F64Array q5 = F64Array.allocate(5);
             F64Array o2 = F64Array.allocate(2)) {
            throwsIAE("sqdist [10] vs [11]", () -> NumJ.sqdist(a10, a11));
            throwsIAE("sqdist [10] vs [2,5]", () -> NumJ.sqdist(a10, m25));
            throwsIAE("muladd c mismatch", () -> NumJ.sumSqMulAdd(a10, a10, a11));
            throwsIAE("normalize out shape", () -> NumJ.normalizeRows(m25, m52, null));
            throwsIAE("normalize rank 1", () -> NumJ.normalizeRows(a10, a11, null));
            throwsIAE("normalize norms size", () -> NumJ.normalizeRows(m25, F64Array.allocate(2, 5), n3));
            throwsIAE("sqdistRows q size", () -> NumJ.sqdistRows(n3, m25, o2));
            throwsIAE("sqdistRows out size", () -> NumJ.sqdistRows(q5, m25, n3));
            throwsIAE("reshape size", () -> m25.reshape(3, 3));
            throwsIAE("negative dim", () -> F64Array.allocate(-1));
        }
    }

    static void overlap() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment s = arena.allocate(10 * 4 * 8L, 64);
            F64Array whole = F64Array.wrap(s, 10, 4);
            F64Array top = F64Array.wrap(s.asSlice(0, 8 * 4 * 8L), 8, 4);
            F64Array shifted = F64Array.wrap(s.asSlice(2 * 4 * 8L, 8 * 4 * 8L), 8, 4);
            F64Array row8 = whole.row(8);
            throwsIAE("normalize partial overlap", () -> NumJ.normalizeRows(top, shifted, null));
            throwsIAE("norms overlap input", () -> NumJ.normalizeRows(top, F64Array.allocate(8, 4), F64Array.wrap(s.asSlice(0, 64), 8, 1)));
            throwsIAE("sqdistRows out overlaps x", () -> NumJ.sqdistRows(F64Array.allocate(4), whole, F64Array.wrap(s.asSlice(0, 80), 10, 1)));
            throwsIAE("sqdistRows out overlaps q", () -> NumJ.sqdistRows(row8, top, F64Array.wrap(s.asSlice(8 * 32L, 64), 8, 1)));
            check("whole is a view", !whole.isOwner());
        }
    }

    static void lifetime() {
        F64Array a = F64Array.of(1, 2, 3);
        F64Array b = F64Array.of(1, 2, 3);
        F64Array view = a.reshape(1, 3).row(0);
        view.close();                                   // no-op on a view
        check("view close is a no-op", a.isAlive() && view.isAlive());
        a.close();
        a.close();                                      // idempotent
        check("closed", !a.isAlive() && !view.isAlive());
        throwsType("get after close", IllegalStateException.class, () -> a.get(0));
        throwsType("view after parent close", IllegalStateException.class, () -> view.get(0));
        throwsType("kernel after close", IllegalStateException.class, () -> NumJ.sqdist(a, b));
        b.close();
    }

    @SuppressWarnings("deprecation") // flatten(): kept as a view in 0.2, see docs/MIGRATION.md
    static void bounds() {
        try (F64Array m = F64Array.allocate(3, 4)) {
            throwsType("get(-1)", IndexOutOfBoundsException.class, () -> m.get(-1));
            throwsType("get(size)", IndexOutOfBoundsException.class, () -> m.get(12));
            throwsType("get(3,0)", IndexOutOfBoundsException.class, () -> m.get(3, 0));
            throwsType("get(0,4)", IndexOutOfBoundsException.class, () -> m.get(0, 4));
            throwsType("row(3)", IndexOutOfBoundsException.class, () -> m.row(3));
            throwsType("copyFrom range", IndexOutOfBoundsException.class, () -> m.copyFrom(new double[5], 0, 10, 5));
            throwsIAE("get(r,c) on rank 1", () -> m.flatten().get(0, 0));
            m.set(2, 3, 42);
            assertBits("set/get", 42, m.get(11));
        }
    }

    static void confinement() throws InterruptedException {
        AtomicReference<Throwable> err = new AtomicReference<>();
        try (F64Array conf = F64Array.of(1, 2); F64Array shared = F64Array.allocateShared(2)) {
            shared.set(0, 3);
            Thread t = new Thread(() -> {
                try {
                    conf.get(0);
                    err.set(new AssertionError("confined read from another thread succeeded"));
                } catch (WrongThreadException expected) {
                    // ok
                }
                try {
                    if (NumJ.sqdist(shared, shared) != 0.0) err.set(new AssertionError("shared sqdist wrong"));
                } catch (Throwable e) {
                    err.set(e);
                }
            });
            t.start();
            t.join();
        }
        if (err.get() != null) throw new AssertionError(err.get());
    }

    static void arrayBasics() {
        double[] h = Data.splitmix(91, 12);
        try (F64Array m = F64Array.copyOf(h, 3, 4)) {
            assertBits("row view", h[6], m.row(1).get(2));
            m.row(1).set(2, -5);
            assertBits("row view writes through", -5, m.get(1, 2));
            assertBits("reshape view", -5, m.reshape(4, 3).get(2, 0));
            double[] back = new double[12];
            m.copyTo(back);
            assertBits("copyTo", -5, back[6]);
            check("shape", m.rank() == 2 && m.rows() == 3 && m.cols() == 4 && m.size() == 12);
            check("alignment", m.segment().address() % F64Array.ALIGNMENT == 0);
            check("zero-filled", F64Array.allocate(5).get(4) == 0.0);
        }
    }

    // ------------------------------------------------------------------ helpers

    interface Body { void run() throws Exception; }

    static void test(String name, Body body) {
        try {
            body.run();
            passed++;
            System.out.println("PASS  " + name);
        } catch (Throwable t) {
            failed++;
            failures.add(name + ": " + t);
            System.out.println("FAIL  " + name + "\n      " + t);
        }
    }

    static double sq(double[] a, double[] b) {
        try (F64Array x = F64Array.of(a); F64Array y = F64Array.of(b)) {
            return NumJ.sqdist(x, y);
        }
    }

    static double[] withAt(double[] x, int i, double v) {
        x[i] = v;
        return x;
    }

    static double gamma(int k) {
        return k * U / (1 - k * U);
    }

    static void checkSumBound(String what, double got, BigDecimal exact, int n) {
        BigDecimal err = new BigDecimal(got).subtract(exact).abs();
        BigDecimal bound = exact.multiply(new BigDecimal(gamma(NumJ.summationDepth(n))));
        if (err.compareTo(bound) > 0)
            throw new AssertionError(what + ": |error| " + err.doubleValue() + " > bound " + bound.doubleValue());
    }

    /** |y - x/||x||| <= (cols + 8) u |x/||x||| against a 50-digit reference (NaN-free, nonzero rows only). */
    static void checkNormalizeReference(double[] x, double[] y, int rows, int cols) {
        MathContext mc = new MathContext(50);
        double tol = (cols + 8) * U;
        for (int r = 0; r < rows; r++) {
            BigDecimal ss = BigDecimal.ZERO;
            for (int j = 0; j < cols; j++) {
                BigDecimal v = new BigDecimal(x[r * cols + j]);
                ss = ss.add(v.multiply(v));
            }
            if (ss.signum() == 0) continue;
            BigDecimal nrm = ss.sqrt(mc);
            for (int j = 0; j < cols; j++) {
                double ref = new BigDecimal(x[r * cols + j]).divide(nrm, mc).doubleValue();
                double got = y[r * cols + j];
                if (Math.abs(got - ref) > tol * Math.abs(ref) + Double.MIN_VALUE)
                    throw new AssertionError("normalize row " + r + " col " + j + ": got " + got + " ref " + ref);
            }
        }
    }

    static void assertBits(String what, double expected, double got) {
        if (Double.doubleToRawLongBits(expected) != Double.doubleToRawLongBits(got)
                && !(Double.isNaN(expected) && Double.isNaN(got)))
            throw new AssertionError(what + ": expected " + expected + " got " + got);
    }

    static void check(String what, boolean ok) {
        if (!ok) throw new AssertionError(what);
    }

    static void throwsIAE(String what, Runnable r) {
        throwsType(what, IllegalArgumentException.class, r);
    }

    static void throwsType(String what, Class<? extends Throwable> type, Runnable r) {
        try {
            r.run();
        } catch (Throwable t) {
            if (type.isInstance(t)) return;
            throw new AssertionError(what + ": expected " + type.getSimpleName() + " got " + t, t);
        }
        throw new AssertionError(what + ": expected " + type.getSimpleName() + ", nothing thrown");
    }
}
