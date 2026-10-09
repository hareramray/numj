package numj;

import numj.bench.Data;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static numj.T.*;

/**
 * Tests for the n-dimensional array foundations (milestone 1). References are independent of the engine:
 * per-element Java arithmetic read through the multi-index accessor, an independent from-the-spec implementation
 * of the blocked summation, and exact BigDecimal sums. Every operation is run on several memory layouts and with
 * the Java path forced, the native path forced, and the default selection.
 */
public final class NDArrayTests {

    public static void main(String[] args) {
        System.out.println("numj library: " + NumJ.libraryPath());
        test("creation: shapes, 0-d, empty, factories", NDArrayTests::creation);
        test("creation: invalid shapes and dimension overflow", NDArrayTests::invalidShapes);
        test("creation: arena-managed and shared arrays", NDArrayTests::arenas);
        test("access: multi-index, flat, item, bounds", NDArrayTests::access);
        test("slicing: stepped, reversed, clamped, newaxis, ellipsis", NDArrayTests::slicing);
        test("slicing: errors", NDArrayTests::slicingErrors);
        test("transpose / permute / swapAxes", NDArrayTests::transpose);
        test("reshape: view rules, -1, copy, errors", NDArrayTests::reshape);
        test("flatten: deprecated view, flattenView, flattenCopy", NDArrayTests::flatten);
        test("broadcastTo: stride 0, read-only, errors", NDArrayTests::broadcastTo);
        test("contiguity flags (NumPy rules)", NDArrayTests::contiguity);
        test("wrap: strided views, bounds, self-overlap -> read-only", NDArrayTests::wrapStrided);
        test("read-only views reject every write", NDArrayTests::readOnly);
        test("elementwise: layouts x broadcasting x paths, bitwise", NDArrayTests::elementwiseLayouts);
        test("elementwise: scalar operands and 0-d arrays", NDArrayTests::elementwiseScalars);
        test("elementwise: NaN, Inf, signed zero, subnormals", NDArrayTests::elementwiseSpecials);
        test("elementwise: out reuse and validation", NDArrayTests::outReuse);
        test("elementwise: overlapping out (in place, reversed, shifted)", NDArrayTests::overlap);
        test("copy / copyFrom (broadcast, overlap)", NDArrayTests::copies);
        test("sum/mean: axes, keepdims, negative axes, layouts, paths", NDArrayTests::reductions);
        test("sum/mean: large inputs hit every native path, bitwise", NDArrayTests::reductionsLarge);
        test("sum/mean: empty reductions, -0.0, axis=(), errors", NDArrayTests::reductionEdges);
        test("sum/mean: out reuse, overlap, read-only", NDArrayTests::reductionOut);
        test("threads: results identical for 1..8 native threads", NDArrayTests::threadInvariance);
        test("fused kernels on strided inputs (bitwise)", NDArrayTests::fusedStrided);
        test("Expr: pattern recognition, fused == unfused", NDArrayTests::expr);
        test("lifetime: closed arrays and views, owners", NDArrayTests::lifetime);
        test("threads: confined vs shared arrays (Java and native paths)", NDArrayTests::confinement);
        test("max dimensions (64)", NDArrayTests::maxDims);
        System.exit(finish());
    }

    // ================================================================== layouts

    record V(String name, F64Array a) {}

    /** The same logical values in several memory layouts, all allocated in {@code ar}. */
    static List<V> variants(Arena ar, double[] vals, long[] shape) {
        int nd = shape.length;
        List<V> out = new ArrayList<>();
        out.add(new V("contig", F64Array.allocate(ar, shape)));
        long[] rev = new long[nd], big = new long[nd], rbig = new long[nd];
        StringBuilder step2 = new StringBuilder(), back = new StringBuilder(), back2 = new StringBuilder();
        for (int i = 0; i < nd; i++) {
            rev[i] = shape[nd - 1 - i];
            big[i] = 2 * shape[i] + 1;
            rbig[i] = 2 * shape[nd - 1 - i] + 1;
            if (i > 0) { step2.append(','); back.append(','); back2.append(','); }
            step2.append("1::2");
            back.append("::-1");
            back2.append("-2::-2");
        }
        out.add(new V("transposed", F64Array.allocate(ar, rev).transpose()));
        out.add(new V("stepped", F64Array.allocate(ar, big).slice(step2.toString())));
        out.add(new V("reversed", F64Array.allocate(ar, shape).slice(back.toString())));
        out.add(new V("rev-step-T", F64Array.allocate(ar, rbig).slice(back2.toString()).transpose()));
        for (V v : out) {
            assertShape(v.name, shape, v.a.shape());
            int[] k = {0};
            forEach(shape, idx -> v.a.set(idx.clone(), vals[k[0]++]));
        }
        return out;
    }

    static final long[][] SHAPES = {{}, {1}, {7}, {3, 4}, {2, 3, 4}, {0, 3}, {2, 1, 3}};

    // ================================================================== creation & access

    static void creation() {
        try (F64Array s = F64Array.scalar(2.5); F64Array e = F64Array.allocate(3, 0, 2); F64Array z = F64Array.zeros(2, 3, 4, 5);
             F64Array o = F64Array.ones(2, 2); F64Array f = F64Array.full(-0.0, 3); F64Array r = F64Array.arange(5);
             F64Array c = F64Array.copyOf(new double[] {1, 2, 3, 4, 5, 6}, 2, 1, 3); F64Array zz = F64Array.allocate()) {
            check("scalar ndim", s.ndim() == 0 && s.size() == 1 && s.shape().length == 0);
            assertBits("scalar get()", 2.5, s.get());
            assertBits("scalar item", 2.5, s.item());
            assertBits("scalar flat", 2.5, s.get(0));
            check("scalar shapeString", s.shapeString().equals("[]"));
            check("empty size", e.size() == 0 && e.ndim() == 3);
            check("empty toArray", e.toArray().length == 0);
            check("zeros", z.size() == 120 && z.get(119) == 0.0);
            assertShape("strides", new long[] {480, 160, 40, 8}, z.strides());
            assertBits("ones", 1.0, o.get(1, 1));
            assertBits("full keeps -0.0", -0.0, f.get(2));
            assertBits("arange", 4.0, r.get(4));
            assertBits("copyOf 3-d", 6.0, c.get(new long[] {1, 0, 2}));
            check("allocate() is 0-d", zz.ndim() == 0 && zz.get() == 0.0);
            check("dtype", c.dtype() == DType.FLOAT64 && DType.FLOAT64.itemSize() == 8);
            check("owner", c.isOwner() && c.isWritable() && c.isAlive());
            check("alignment", z.segment().address() % 64 == 0);
        }
        throwsIAE("copyOf length mismatch", () -> F64Array.copyOf(new double[5], 2, 3));
    }

    static void invalidShapes() {
        throwsIAE("negative", () -> F64Array.allocate(2, -1));
        throwsIAE("65 dims", () -> F64Array.allocate(new long[65]));
        throwsIAE("byte size overflow", () -> F64Array.allocate(1L << 31, 1L << 31));
        throwsIAE("element overflow", () -> F64Array.allocate(Long.MAX_VALUE, 2));
        long[] ones = new long[64];
        Arrays.fill(ones, 1);
        try (F64Array a = F64Array.allocate(ones)) {
            check("64 dims ok", a.ndim() == 64 && a.size() == 1);
        }
        // zero extents short-circuit the size, but the other extents must still be representable
        try (F64Array a = F64Array.allocate(0, 1L << 40)) {
            check("0 x 2^40", a.size() == 0);
        }
        try (F64Array a = F64Array.allocate(4)) {
            throwsIAE("broadcast overflow", () -> a.slice("0:1").broadcastTo(1L << 40, 1L << 40));
            throwsIAE("reshape overflow", () -> a.reshape(Long.MAX_VALUE, 2));
            throwsIAE("broadcastShapes mismatch", () -> NumJ.broadcastShapes(new long[] {2}, new long[] {3}));
        }
    }

    static void arenas() throws Exception {
        F64Array kept;
        try (Arena ar = Arena.ofConfined()) {
            F64Array a = F64Array.allocate(ar, 3, 4);
            kept = a;
            check("arena-managed is not an owner", !a.isOwner());
            a.close();                               // no-op
            check("still alive after close()", a.isAlive());
            a.set(1, 2, 7);
            assertBits("value", 7, a.get(6));
        }
        check("freed with the arena", !kept.isAlive());
        throwsType("use after arena close", IllegalStateException.class, () -> kept.get(0));

        try (F64Array sh = F64Array.allocateShared(1000); F64Array sh2 = F64Array.allocateShared(10, 100)) {
            sh.fill(1);
            AtomicReference<Throwable> err = new AtomicReference<>();
            Thread t = new Thread(() -> {
                try {
                    try (F64Array r = NumJ.add(sh, sh2.reshape(1000))) {
                        if (r.get(999) != 1.0) err.set(new AssertionError("shared add"));
                    }
                } catch (Throwable e) {
                    err.set(e);
                }
            });
            t.start();
            t.join();
            if (err.get() != null) throw new AssertionError(err.get());
        }
    }

    static void access() {
        try (F64Array a = F64Array.copyOf(Data.splitmix(1, 24), 2, 3, 4)) {
            double[] v = a.toArray();
            assertBits("get(idx)", v[1 * 12 + 2 * 4 + 3], a.get(new long[] {1, 2, 3}));
            F64Array t = a.transpose();                                   // [4, 3, 2]
            assertBits("flat on transposed", a.get(new long[] {1, 0, 2}), t.get(2 * 6 + 0 * 2 + 1));
            assertBits("multi on transposed", a.get(new long[] {1, 0, 2}), t.get(new long[] {2, 0, 1}));
            t.set(new long[] {3, 2, 1}, 42);
            assertBits("write through transposed view", 42, a.get(new long[] {1, 2, 3}));
            throwsType("multi-index out of range", IndexOutOfBoundsException.class, () -> a.get(new long[] {2, 0, 0}));
            throwsType("negative multi-index", IndexOutOfBoundsException.class, () -> a.get(new long[] {-1, 0, 0}));
            throwsIAE("wrong index count", () -> a.get(new long[] {0, 0}));
            throwsIAE("get(r,c) on rank 3", () -> a.get(0, 0));
            throwsType("flat out of range", IndexOutOfBoundsException.class, () -> t.get(24));
            throwsIAE("item on size 24", a::item);
            throwsType("rows() on rank 3", IllegalStateException.class, a::rows);
            check("shape(axis)", a.shape(-1) == 4 && a.shape(0) == 2);
            double[] tv = new double[24];
            t.copyTo(tv);
            double[] want = values(t);
            for (int i = 0; i < 24; i++) assertBits("copyTo strided " + i, want[i], tv[i]);
            t.copyFrom(new double[] {9, 8}, 0, 22, 2);
            assertBits("copyFrom strided", 9, t.get(22));
            assertBits("copyFrom strided 2", 8, t.get(23));
        }
    }

    static void slicing() {
        try (F64Array a = F64Array.arange(60).reshapeCopy(3, 4, 5)) {
            // value at [i,j,k] is 20i + 5j + k
            F64Array s = a.slice("1:, ::-1, 0");
            assertShape("1:,::-1,0", new long[] {2, 4}, s.shape());
            assertBits("s[0,0]", 20 + 15, s.get(0, 0));
            assertBits("s[1,3]", 40 + 0, s.get(1, 3));
            assertShape("strides", new long[] {160, -40}, s.strides());

            F64Array n = a.slice("::2, None, ..., -1");
            assertShape("newaxis+ellipsis", new long[] {2, 1, 4}, n.shape());
            assertBits("n[1,0,2]", 40 + 10 + 4, n.get(new long[] {1, 0, 2}));
            check("newaxis stride 0", n.strides()[1] == 0);

            assertShape("clamped", new long[] {0, 4, 5}, a.slice("100:").shape());
            assertShape("clamped neg", new long[] {3, 4, 5}, a.slice("-100:").shape());
            assertShape("empty backwards", new long[] {3, 0, 5}, a.slice(":, 1:3:-1").shape());
            F64Array r = a.slice("..., 4:0:-2");
            assertShape("4:0:-2", new long[] {3, 4, 2}, r.shape());
            assertBits("r[0,0,1]", 2, r.get(new long[] {0, 0, 1}));
            F64Array full = a.slice("::-1, ::-1, ::-1");
            assertBits("full reverse first", 59, full.get(0));
            assertBits("full reverse last", 0, full.get(59));
            F64Array sc = a.slice("2, 3, 4");
            check("all integers -> 0-d", sc.ndim() == 0);
            assertBits("0-d value", 59, sc.get());
            F64Array sameIx = a.slice(Ix.range(1, 3), Ix.step(-1), Ix.at(-2));
            assertShape("Ix api", new long[] {2, 4}, sameIx.shape());
            assertBits("Ix api value", 20 + 15 + 3, sameIx.get(0, 0));
            F64Array big = a.slice(Ix.slice(null, null, Long.MIN_VALUE));
            assertShape("min step", new long[] {1, 4, 5}, big.shape());
            assertBits("min step value", 40, big.get(new long[] {0, 0, 0}));

            // views write through, nested views compose
            F64Array v = a.slice("1, ::2").slice("::-1, 1:3");
            v.set(0, 0, -1);
            assertBits("nested write", -1, a.get(new long[] {1, 2, 1}));
            check("view is not owner", !v.isOwner() && v.overlaps(a));
        }
    }

    static void slicingErrors() {
        try (F64Array a = F64Array.allocate(3, 4)) {
            throwsType("index oob", IndexOutOfBoundsException.class, () -> a.slice("3"));
            throwsType("negative index oob", IndexOutOfBoundsException.class, () -> a.slice(":, -5"));
            throwsIAE("too many indices", () -> a.slice("0, 0, 0"));
            throwsIAE("two ellipses", () -> a.slice("..., ..."));
            throwsIAE("step 0", () -> a.slice("::0"));
            throwsIAE("parse error", () -> a.slice("1:2:3:4"));
            throwsIAE("parse error 2", () -> a.slice("x"));
            throwsType("row oob", IndexOutOfBoundsException.class, () -> a.row(3));
        }
    }

    static void transpose() {
        try (F64Array a = F64Array.arange(24).reshapeCopy(2, 3, 4)) {
            F64Array t = a.transpose();
            assertShape("T", new long[] {4, 3, 2}, t.shape());
            F64Array p = a.permute(1, -1, 0);
            assertShape("permute", new long[] {3, 4, 2}, p.shape());
            forEach(a.shape(), i -> {
                assertBits("T value", a.get(i.clone()), t.get(new long[] {i[2], i[1], i[0]}));
                assertBits("permute value", a.get(i.clone()), p.get(new long[] {i[1], i[2], i[0]}));
            });
            F64Array s = a.swapAxes(0, -1);
            assertShape("swapAxes", new long[] {4, 3, 2}, s.shape());
            throwsIAE("duplicate axes", () -> a.permute(0, 0, 1));
            throwsIAE("wrong count", () -> a.permute(0, 1));
            throwsIAE("axis oob", () -> a.permute(0, 1, 3));
            throwsIAE("swap oob", () -> a.swapAxes(0, 3));
            try (F64Array s0 = F64Array.scalar(1)) {
                check("0-d transpose", s0.transpose().ndim() == 0);
            }
        }
    }

    static void reshape() {
        try (F64Array a = F64Array.arange(24).reshapeCopy(4, 6)) {
            F64Array r = a.reshape(2, -1, 3);
            assertShape("-1 inferred", new long[] {2, 4, 3}, r.shape());
            check("contiguous reshape is a view", !r.isOwner() && r.overlaps(a));
            r.set(new long[] {1, 3, 2}, -7);
            assertBits("view writes through", -7, a.get(23));

            F64Array cols = a.slice(":, ::2");                         // [4, 3], strides (48, 16)
            check("split stepped axis: view", cols.canReshapeView(2, 2, 3));
            F64Array split = cols.reshape(2, 2, 3);
            assertShape("split strides", new long[] {96, 48, 16}, split.strides());
            assertBits("split value", cols.get(3, 1), split.get(new long[] {1, 1, 1}));
            assertShape("merge uniformly stepped axes: view", new long[] {16}, cols.reshape(12).strides());
            F64Array left = a.slice(":, :3");                          // [4, 3], strides (48, 8)
            check("merge non-uniform axes: no view", !left.canReshapeView(12));
            throwsIAE("merge needs copy", () -> left.reshape(12));
            try (F64Array c = left.reshapeCopy(12)) {
                check("reshapeCopy owner", c.isOwner() && !c.overlaps(a));
                assertBits("reshapeCopy value", left.get(3, 2), c.get(11));
            }
            F64Array rows = a.slice("::2");                            // [2, 6], strides (96, 8)
            check("merge stepped rows: no view", !rows.canReshapeView(-1));
            check("split contiguous rows: view", rows.canReshapeView(2, 2, 3));
            check("T then reshape: no view", !a.transpose().canReshapeView(24));
            check("size-1 axes are free", a.slice("None, :, None, 1:2").canReshapeView(4));

            throwsIAE("two -1", () -> a.reshape(-1, -1));
            throwsIAE("size mismatch", () -> a.reshape(5, 5));
            throwsIAE("negative", () -> a.reshape(-2, 12));
            try (F64Array e = F64Array.allocate(0, 3)) {
                throwsIAE("-1 with zero size", () -> e.reshape(0, -1));
                assertShape("empty reshape", new long[] {3, 0}, e.reshape(3, 0).shape());
            }
            try (F64Array s = F64Array.scalar(3)) {
                assertShape("0-d to [1,1]", new long[] {1, 1}, s.reshape(1, 1).shape());
                assertShape("[1,1] to 0-d", new long[0], s.reshape(1, 1).reshape().shape());
            }
        }
    }

    @SuppressWarnings("deprecation")
    static void flatten() {
        try (F64Array a = F64Array.arange(6).reshapeCopy(2, 3)) {
            F64Array f = a.flatten();
            check("flatten (deprecated) is still a view", !f.isOwner() && f.overlaps(a));
            F64Array fv = a.flattenView();
            fv.set(5, 99);
            assertBits("flattenView writes through", 99, a.get(1, 2));
            try (F64Array fc = a.flattenCopy()) {
                fc.set(0, -1);
                assertBits("flattenCopy is a copy", 0, a.get(0, 0));
                check("flattenCopy owner", fc.isOwner() && fc.ndim() == 1);
            }
            throwsIAE("flatten of transposed needs a copy", () -> a.transpose().flatten());
            try (F64Array tc = a.transpose().flattenCopy()) {
                assertBits("transposed flattenCopy order", a.get(0, 1), tc.get(2));
            }
        }
    }

    static void broadcastTo() {
        try (F64Array a = F64Array.of(1, 2, 3)) {
            F64Array b = a.broadcastTo(4, 3);
            check("read-only", !b.isWritable());
            assertShape("strides", new long[] {0, 8}, b.strides());
            assertBits("value", 3, b.get(3, 2));
            throwsType("set on broadcast", ReadOnlyArrayException.class, () -> b.set(0, 0, 1));
            throwsType("fill on broadcast", ReadOnlyArrayException.class, () -> b.fill(1));
            throwsType("out=broadcast", ReadOnlyArrayException.class, () -> NumJ.add(b, 1.0, b));
            throwsType("segment of broadcast is read-only", RuntimeException.class,
                    () -> b.segment().set(java.lang.foreign.ValueLayout.JAVA_DOUBLE, 0, 1.0));
            throwsIAE("incompatible", () -> a.broadcastTo(4, 2));
            throwsIAE("fewer dims", () -> a.broadcastTo());
            assertShape("to empty", new long[] {0, 3}, a.broadcastTo(0, 3).shape());
            try (F64Array one = F64Array.of(5)) {
                assertShape("(1) -> (0)", new long[] {0}, one.broadcastTo(0).shape());
            }
            assertShape("broadcastShapes", new long[] {2, 0}, NumJ.broadcastShapes(new long[0], new long[] {2, 0}));
            assertShape("broadcastShapes 1 vs 0", new long[] {0}, NumJ.broadcastShapes(new long[] {1}, new long[] {0}));
            throwsIAE("broadcastShapes 2 vs 0", () -> NumJ.broadcastShapes(new long[] {2}, new long[] {0}));
        }
    }

    static void contiguity() {
        try (F64Array a = F64Array.allocate(3, 4)) {
            check("C", a.isCContiguous() && !a.isFContiguous());
            check("T is F", !a.transpose().isCContiguous() && a.transpose().isFContiguous());
            check("row slice is C", a.slice("1:2").isCContiguous());
            check("single row is C and F", a.slice("1:2").isFContiguous());
            check("column is neither", !a.slice(":, 1").isCContiguous());
            check("size-1 axis with odd stride", a.slice("None, 1:2, :").isCContiguous());
            check("empty is both", a.slice("0:0").isCContiguous() && a.slice("0:0").isFContiguous());
            check("reversed is neither", !a.slice("::-1").isCContiguous() && !a.slice("::-1").isFContiguous());
            try (F64Array s = F64Array.scalar(1)) {
                check("0-d is both", s.isCContiguous() && s.isFContiguous());
            }
        }
    }

    static void wrapStrided() {
        try (Arena ar = Arena.ofConfined()) {
            MemorySegment s = ar.allocate(10 * 8, 64);
            F64Array col = F64Array.wrap(s, 8, new long[] {3}, new long[] {24});
            check("strided wrap writable", col.isWritable());
            col.set(2, 5);
            assertBits("wrap write", 5, s.getAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE, 7));
            F64Array overlapping = F64Array.wrap(s, 0, new long[] {3, 3}, new long[] {8, 8});
            check("self-overlapping layout is read-only", !overlapping.isWritable());
            F64Array neg = F64Array.wrap(s, 72, new long[] {10}, new long[] {-8});
            assertBits("negative stride wrap", 5, neg.get(2));
            throwsIAE("past the end", () -> F64Array.wrap(s, 8, new long[] {10}, new long[] {8}));
            throwsIAE("before the start", () -> F64Array.wrap(s, 0, new long[] {2}, new long[] {-8}));
            throwsIAE("unaligned stride", () -> F64Array.wrap(s, 0, new long[] {2}, new long[] {12}));
            throwsIAE("strides count", () -> F64Array.wrap(s, 0, new long[] {2}, new long[] {8, 8}));
            throwsIAE("heap segment", () -> F64Array.wrap(MemorySegment.ofArray(new double[4]), 4));
            F64Array ro = F64Array.wrap(s.asReadOnly(), 10);
            check("read-only segment -> read-only array", !ro.isWritable());
            throwsType("write to read-only wrap", ReadOnlyArrayException.class, () -> ro.set(0, 1));
        }
    }

    static void readOnly() {
        try (F64Array a = F64Array.arange(6).reshapeCopy(2, 3); F64Array out = F64Array.allocate(2, 3)) {
            F64Array r = a.asReadOnly();
            check("asReadOnly", !r.isWritable() && a.isWritable());
            check("views of read-only stay read-only", !r.slice("::-1").isWritable() && !r.transpose().isWritable());
            throwsType("set", ReadOnlyArrayException.class, () -> r.set(0, 1));
            throwsType("set(r,c)", ReadOnlyArrayException.class, () -> r.set(0, 0, 1));
            throwsType("set(idx)", ReadOnlyArrayException.class, () -> r.set(new long[] {0, 0}, 1));
            throwsType("copyFrom(double[])", ReadOnlyArrayException.class, () -> r.copyFrom(new double[6]));
            throwsType("copyFrom(array)", ReadOnlyArrayException.class, () -> r.copyFrom(a));
            throwsType("out of add", ReadOnlyArrayException.class, () -> NumJ.add(a, a, r));
            throwsType("out of sum", ReadOnlyArrayException.class, () -> NumJ.sum(a, new int[] {0}, true, r.slice("0:1")));
            throwsType("normalizeRows out", ReadOnlyArrayException.class, () -> NumJ.normalizeRows(a, r, null));
            try (F64Array s = NumJ.add(r, r)) {
                assertBits("read-only arrays are fine as inputs", 10, s.get(5));
            }
            NumJ.add(r, 1.0, out);
            assertBits("read-only input, writable out", 6, out.get(5));
        }
    }

    // ================================================================== elementwise

    static final int[] OPS = {Elementwise.ADD, Elementwise.SUB, Elementwise.MUL, Elementwise.DIV};
    static final String[] OP_NAMES = {"", "add", "sub", "mul", "div"};

    static double ref(int op, double x, double y) {
        return switch (op) {
            case Elementwise.ADD -> x + y;
            case Elementwise.SUB -> x - y;
            case Elementwise.MUL -> x * y;
            default -> x / y;
        };
    }

    static F64Array call(int op, F64Array a, F64Array b, F64Array out) {
        return switch (op) {
            case Elementwise.ADD -> out == null ? NumJ.add(a, b) : NumJ.add(a, b, out);
            case Elementwise.SUB -> out == null ? NumJ.subtract(a, b) : NumJ.subtract(a, b, out);
            case Elementwise.MUL -> out == null ? NumJ.multiply(a, b) : NumJ.multiply(a, b, out);
            default -> out == null ? NumJ.divide(a, b) : NumJ.divide(a, b, out);
        };
    }

    /** Runs {@code body} with the Java path forced, the native path forced, and the default threshold. */
    static void allPaths(Runnable body) {
        long saveE = Elementwise.javaMaxElements, saveC = Elementwise.contigJavaMaxElements, saveR = Reduce.javaMaxElements,
                saveF = NumJ.fusedJavaMaxElements;
        try {
            for (long t : new long[] {Long.MAX_VALUE, -1, saveE}) {
                Elementwise.javaMaxElements = t;
                Elementwise.contigJavaMaxElements = t == saveE ? saveC : t;
                Reduce.javaMaxElements = t == saveE ? saveR : t;
                NumJ.fusedJavaMaxElements = t == saveE ? saveF : t;
                body.run();
            }
        } finally {
            Elementwise.javaMaxElements = saveE;
            Elementwise.contigJavaMaxElements = saveC;
            Reduce.javaMaxElements = saveR;
            NumJ.fusedJavaMaxElements = saveF;
        }
    }

    static void checkBinary(String what, int op, F64Array a, F64Array b, F64Array r) {
        long[] shape = NumJ.broadcastShapes(a.shape(), b.shape());
        assertShape(what, shape, r.shape());
        long[] as = a.shape(), bs = b.shape();
        forEach(shape, idx -> {
            double want = ref(op, a.get(bcastIndex(idx, as)), b.get(bcastIndex(idx, bs)));
            assertBits(what + " at " + Arrays.toString(idx), want, r.get(idx.clone()));
        });
    }

    static void elementwiseLayouts() {
        long[][][] pairs = {
            {{}, {}}, {{5}, {}}, {{}, {5}}, {{3, 4}, {4}}, {{3, 1}, {1, 4}}, {{2, 3, 4}, {3, 1}}, {{0, 3}, {3}},
            {{1}, {0}}, {{4, 1, 6}, {5, 1}}, {{2, 1, 3}, {2, 1, 3}}, {{7}, {7}},
            {{1000}, {1000}}, {{40, 70}, {70}}, {{33, 40}, {33, 1}}, {{6, 7, 8}, {6, 1, 8}}, {{3, 50, 4}, {50, 1}},
        };
        try (Arena ar = Arena.ofConfined()) {
            int seed = 0;
            for (long[][] pr : pairs) {
                long[] as = pr[0], bs = pr[1];
                List<V> av = variants(ar, Data.splitmix(++seed, (int) size(as)), as);
                List<V> bv = variants(ar, Data.splitmix(++seed, (int) size(bs)), bs);
                for (int i = 0; i < av.size(); i++) {
                    V va = av.get(i), vb = bv.get((i * 3 + 1) % bv.size());
                    for (int op : OPS) {
                        String w = OP_NAMES[op] + " " + va.name + Arrays.toString(as) + " " + vb.name + Arrays.toString(bs);
                        allPaths(() -> {
                            try (F64Array r = call(op, va.a, vb.a, null)) {
                                check(w + " owner", r.isOwner() && r.isCContiguous());
                                checkBinary(w, op, va.a, vb.a, r);
                            }
                        });
                    }
                }
            }
            // incompatible shapes
            F64Array x = F64Array.allocate(ar, 2, 3), y = F64Array.allocate(ar, 4);
            throwsIAE("broadcast mismatch", () -> NumJ.add(x, y));
        }
    }

    static void elementwiseScalars() {
        double[] scalars = {2.5, -0.0, Double.NaN, Double.POSITIVE_INFINITY, 0x1p-1074};
        try (Arena ar = Arena.ofConfined()) {
            for (long[] shape : new long[][] {{}, {9}, {4, 5}, {300}, {30, 20}, {0, 2}}) {
                for (V v : variants(ar, Data.splitmix(3, (int) size(shape)), shape)) {
                    for (double s : scalars) {
                        allPaths(() -> {
                            String w = v.name + Arrays.toString(shape) + " s=" + s;
                            try (F64Array r1 = NumJ.add(v.a, s); F64Array r2 = NumJ.subtract(s, v.a);
                                 F64Array r3 = NumJ.multiply(s, v.a); F64Array r4 = NumJ.divide(v.a, s);
                                 F64Array r5 = NumJ.divide(s, v.a); F64Array r6 = NumJ.subtract(v.a, s)) {
                                forEach(shape, idx -> {
                                    double x = v.a.get(idx.clone());
                                    assertBits(w + " a+s", x + s, r1.get(idx.clone()));
                                    assertBits(w + " s-a", s - x, r2.get(idx.clone()));
                                    assertBits(w + " s*a", s * x, r3.get(idx.clone()));
                                    assertBits(w + " a/s", x / s, r4.get(idx.clone()));
                                    assertBits(w + " s/a", s / x, r5.get(idx.clone()));
                                    assertBits(w + " a-s", x - s, r6.get(idx.clone()));
                                });
                            }
                        });
                    }
                }
            }
            // 0-d arrays broadcast like scalars
            F64Array s0 = F64Array.allocate(ar);
            s0.set(new long[0], 3.0);
            F64Array m = F64Array.copyOf(Data.splitmix(4, 12), 3, 4);
            try (F64Array r = NumJ.subtract(s0, m); F64Array r0 = NumJ.multiply(s0, s0)) {
                checkBinary("0-d - matrix", Elementwise.SUB, s0, m, r);
                check("0-d op 0-d is 0-d", r0.ndim() == 0);
                assertBits("0-d value", 9.0, r0.get());
            }
            m.close();
        }
    }

    static void elementwiseSpecials() {
        double inf = Double.POSITIVE_INFINITY, nan = Double.NaN;
        double[] sp = {0.0, -0.0, 1.0, -1.0, inf, -inf, nan, Double.MIN_VALUE, -Double.MIN_VALUE, Double.MIN_NORMAL,
            Double.MAX_VALUE, -Double.MAX_VALUE, 1e308, 1e-308, 3.0, 0.1};
        int n = sp.length;
        double[] av = new double[n * n], bv = new double[n * n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) { av[i * n + j] = sp[i]; bv[i * n + j] = sp[j]; }
        try (F64Array a = F64Array.copyOf(av, n, n); F64Array b = F64Array.copyOf(bv, n, n)) {
            for (int op : OPS) {
                allPaths(() -> {
                    try (F64Array r = call(op, a, b, null); F64Array rt = call(op, a.transpose(), b.transpose(), null)) {
                        checkBinary("specials " + OP_NAMES[op], op, a, b, r);
                        checkBinary("specials T " + OP_NAMES[op], op, a.transpose(), b.transpose(), rt);
                    }
                });
            }
            try (F64Array r = NumJ.subtract(a, a)) {
                assertBits("-0 - -0 = +0", 0.0, r.get(1 * n + 1));
            }
            try (F64Array r = NumJ.multiply(a, 0.0)) {
                check("Inf * 0 = NaN", Double.isNaN(r.get(4 * n)));
            }
        }
    }

    static void outReuse() {
        try (Arena ar = Arena.ofConfined()) {
            F64Array a = F64Array.copyOf(Data.splitmix(5, 600), 20, 30);
            F64Array b = F64Array.copyOf(Data.splitmix(6, 30), 30);
            for (V o : variants(ar, new double[600], new long[] {20, 30})) {
                for (int op : OPS) {
                    allPaths(() -> {
                        F64Array r = call(op, a, b, o.a);
                        check("returns out", r == o.a);
                        checkBinary("out " + o.name + " " + OP_NAMES[op], op, a, b, o.a);
                    });
                }
            }
            F64Array wrong = F64Array.allocate(ar, 30, 20);
            throwsIAE("out shape mismatch", () -> NumJ.add(a, b, wrong));
            throwsIAE("out must not be broadcast-expanded", () -> NumJ.add(b, b, F64Array.allocate(ar, 2, 30)));
            throwsType("null out", NullPointerException.class, () -> NumJ.add(a, b, null));
            F64Array closed = F64Array.allocate(20, 30);
            closed.close();
            throwsType("closed out", IllegalStateException.class, () -> NumJ.add(a, b, closed));
            // the same out reused many times: no allocation, results replaced each time
            F64Array out = F64Array.allocate(ar, 20, 30);
            for (int k = 0; k < 3; k++) NumJ.multiply(a, (double) k, out);
            assertBits("reused out holds the last result", a.get(7) * 2, out.get(7));
            a.close();
            b.close();
        }
    }

    static void overlap() {
        double[] v = Data.splitmix(7, 200);
        allPaths(() -> {
            for (int op : OPS) {
                String w = OP_NAMES[op];
                // out == a (in place), out == b, out == a == b, on strided views
                try (F64Array x = F64Array.copyOf(v, 10, 20); F64Array y = F64Array.copyOf(Data.splitmix(8, 200), 10, 20)) {
                    F64Array xs = x.slice("::-1, ::2"), ys = y.slice("::-1, ::2");
                    double[] before = values(xs), yb = values(ys);
                    call(op, xs, ys, xs);
                    double[] after = values(xs);
                    for (int i = 0; i < before.length; i++) assertBits(w + " out==a", ref(op, before[i], yb[i]), after[i]);
                    before = values(xs);
                    call(op, ys, xs, xs);
                    after = values(xs);
                    for (int i = 0; i < before.length; i++) assertBits(w + " out==b", ref(op, yb[i], before[i]), after[i]);
                    before = values(xs);
                    call(op, xs, xs, xs);
                    after = values(xs);
                    for (int i = 0; i < before.length; i++) assertBits(w + " out==a==b", ref(op, before[i], before[i]), after[i]);
                    // untouched elements (odd columns) stay unchanged
                    for (int r = 0; r < 10; r++) assertBits("gap", v[r * 20 + 1], x.get(r, 1));
                }
                // out = x, a = x reversed: NumPy's "as if copied" semantics
                try (F64Array x = F64Array.copyOf(v, 200)) {
                    double[] before = x.toArray();
                    call(op, x.slice("::-1"), x, x);
                    for (int i = 0; i < 200; i++) assertBits(w + " reversed overlap", ref(op, before[199 - i], before[i]), x.get(i));
                }
                // shifted overlap: out = x[1:], a = x[:-1]  (a naive loop would smear x[0] forward)
                try (F64Array x = F64Array.copyOf(v, 200)) {
                    double[] before = x.toArray();
                    call(op, x.slice(":-1"), x.slice("1:"), x.slice("1:"));
                    for (int i = 1; i < 200; i++) assertBits(w + " shifted overlap", ref(op, before[i - 1], before[i]), x.get(i));
                    assertBits("x[0] untouched", before[0], x.get(0));
                }
                // broadcast input that is a view of out: out = m, b = m[0] broadcast over rows
                try (F64Array m = F64Array.copyOf(v, 10, 20)) {
                    double[] before = m.toArray();
                    call(op, m, m.slice("0"), m);
                    for (int r = 0; r < 10; r++)
                        for (int c = 0; c < 20; c++)
                            assertBits(w + " broadcast row of out", ref(op, before[r * 20 + c], before[c]), m.get(r, c));
                }
            }
            // scalar in place on a strided view
            try (F64Array x = F64Array.copyOf(v, 10, 20)) {
                F64Array t = x.transpose();
                double[] before = values(t);
                NumJ.divide(1.0, t, t);
                double[] after = values(t);
                for (int i = 0; i < before.length; i++) assertBits("scalar in place", 1.0 / before[i], after[i]);
            }
        });
    }

    static void copies() {
        allPaths(() -> {
            try (Arena ar = Arena.ofConfined()) {
                double[] v = Data.splitmix(9, 120);
                for (V src : variants(ar, v, new long[] {4, 5, 6})) {
                    try (F64Array c = src.a.copy()) {
                        check("copy is C owner", c.isOwner() && c.isCContiguous() && !c.overlaps(src.a));
                        double[] got = c.toArray();
                        for (int i = 0; i < 120; i++) assertBits("copy " + src.name, v[i], got[i]);
                    }
                    F64Array dst = variants(ar, new double[120], new long[] {4, 5, 6}).get(3).a;
                    dst.copyFrom(src.a);
                    double[] got = values(dst);
                    for (int i = 0; i < 120; i++) assertBits("copyFrom " + src.name, v[i], got[i]);
                }
                F64Array m = F64Array.allocate(ar, 3, 4);
                m.copyFrom(F64Array.copyOf(new double[] {1, 2, 3, 4}, 4).asReadOnly());
                assertBits("copyFrom broadcast", 4, m.get(2, 3));
                try (F64Array x = F64Array.copyOf(v, 120)) {
                    x.slice("1:").copyFrom(x.slice(":-1"));
                    for (int i = 1; i < 120; i++) assertBits("copyFrom overlap", v[i - 1], x.get(i));
                }
                try (F64Array nz = F64Array.of(-0.0, Double.NaN)) {
                    try (F64Array c = nz.copy()) {
                        assertBits("copy keeps -0.0", -0.0, c.get(0));
                    }
                }
            }
        });
    }

    // ================================================================== reductions

    /** Independent implementation of the documented summation order (for comparison with the engine). */
    static double specSum(double[] t) {
        int n = t.length;
        if (n == 0) return 0.0;
        int nb = (n + 4095) / 4096;
        double[] blocks = new double[nb];
        for (int b = 0; b < nb; b++) {
            double[] lane = new double[16];
            for (int j = b * 4096; j < Math.min(n, (b + 1) * 4096); j++) lane[(j - b * 4096) % 16] += t[j];
            double[] q = new double[8];
            for (int i = 0; i < 8; i++) q[i] = lane[i] + lane[i + 8];
            for (int i = 0; i < 4; i++) q[i] = q[i] + q[i + 4];
            for (int i = 0; i < 2; i++) q[i] = q[i] + q[i + 2];
            blocks[b] = q[0] + q[1];
        }
        int len = nb;
        while (len > 1) {
            int half = len / 2;
            for (int i = 0; i < half; i++) blocks[i] = blocks[2 * i] + blocks[2 * i + 1];
            if (len % 2 == 1) { blocks[half] = blocks[len - 1]; len = half + 1; } else len = half;
        }
        return blocks[0];
    }

    /** Expected sum/mean of {@code x} over {@code axes} by the spec, with exact-bound check; returns values in C order. */
    static double[] expectedReduction(F64Array x, int[] axes, boolean keepdims, boolean mean, long[] outShape) {
        int nd = x.ndim();
        boolean[] red = new boolean[nd];
        if (axes == null) Arrays.fill(red, true);
        else for (int a : axes) red[a < 0 ? a + nd : a] = true;
        long[] xs = x.shape();
        long[] kept = new long[nd], rs = new long[nd];
        int nk = 0, nr = 0;
        for (int d = 0; d < nd; d++) if (red[d]) rs[nr++] = xs[d]; else kept[nk++] = xs[d];
        long[] keptShape = Arrays.copyOf(kept, nk), redShape = Arrays.copyOf(rs, nr);
        double[] out = new double[(int) size(keptShape)];
        int[] o = {0};
        forEach(keptShape, kidx -> {
            List<Double> seq = new ArrayList<>();
            forEach(redShape, ridx -> {
                long[] full = new long[nd];
                for (int d = 0, a = 0, b = 0; d < nd; d++) full[d] = red[d] ? ridx[b++] : kidx[a++];
                seq.add(x.get(full));
            });
            double[] t = seq.stream().mapToDouble(Double::doubleValue).toArray();
            double s = specSum(t);
            // exact-sum bound (finite inputs only)
            BigDecimal exact = BigDecimal.ZERO, abs = BigDecimal.ZERO;
            boolean finite = true;
            for (double v : t) {
                if (!Double.isFinite(v)) { finite = false; break; }
                exact = exact.add(new BigDecimal(v));
                abs = abs.add(new BigDecimal(Math.abs(v)));
            }
            if (finite && t.length > 0) {
                BigDecimal err = new BigDecimal(s).subtract(exact).abs();
                BigDecimal bound = abs.multiply(new BigDecimal(gamma(NumJ.summationDepth(t.length))));
                if (err.compareTo(bound) > 0) throw new AssertionError("spec sum exceeds bound: " + err + " > " + bound);
            }
            out[o[0]++] = mean ? s / t.length : s;
        });
        return out;
    }

    static void checkReduction(String what, F64Array x, int[] axes, boolean keepdims, boolean mean) {
        int nd = x.ndim();
        long[] xs = x.shape();
        boolean[] red = new boolean[nd];
        if (axes == null) Arrays.fill(red, true);
        else for (int a : axes) red[a < 0 ? a + nd : a] = true;
        List<Long> os = new ArrayList<>();
        for (int d = 0; d < nd; d++) if (keepdims) os.add(red[d] ? 1L : xs[d]); else if (!red[d]) os.add(xs[d]);
        long[] outShape = os.stream().mapToLong(Long::longValue).toArray();
        double[] want = expectedReduction(x, axes, keepdims, mean, outShape);
        try (F64Array r = mean ? NumJ.mean(x, axes, keepdims) : NumJ.sum(x, axes, keepdims)) {
            assertShape(what, outShape, r.shape());
            double[] got = r.toArray();
            for (int i = 0; i < want.length; i++) assertBits(what + " [" + i + "]", want[i], got[i]);
        }
    }

    static void reductions() {
        Object[][] cases = {
            {new long[] {}, null}, {new long[] {7}, new int[] {0}}, {new long[] {3, 4}, new int[] {0}},
            {new long[] {3, 4}, new int[] {1}}, {new long[] {3, 4}, new int[] {-1}}, {new long[] {3, 4}, null},
            {new long[] {3, 4}, new int[] {1, 0}}, {new long[] {2, 3, 4}, new int[] {0, 2}},
            {new long[] {2, 3, 4}, new int[] {1}}, {new long[] {2, 3, 4}, new int[] {-1, 0}},
            {new long[] {2, 3, 4}, new int[] {2, 1, 0}}, {new long[] {2, 1, 3}, new int[] {1}},
            {new long[] {2, 1, 3}, new int[] {0, 1}}, {new long[] {5, 6, 7, 3}, new int[] {1, 3}},
            {new long[] {40, 50}, new int[] {0}}, {new long[] {40, 50}, new int[] {1}}, {new long[] {9, 10, 11}, new int[] {0, 2}},
        };
        try (Arena ar = Arena.ofConfined()) {
            int seed = 100;
            for (Object[] c : cases) {
                long[] shape = (long[]) c[0];
                int[] axes = (int[]) c[1];
                for (V v : variants(ar, Data.splitmix(++seed, (int) size(shape)), shape)) {
                    for (boolean keep : new boolean[] {false, true}) {
                        for (boolean mean : new boolean[] {false, true}) {
                            String w = (mean ? "mean " : "sum ") + v.name + Arrays.toString(shape) + " axes "
                                    + Arrays.toString(axes) + (keep ? " keepdims" : "");
                            allPaths(() -> checkReduction(w, v.a, axes, keep, mean));
                        }
                    }
                }
            }
            // scalar API
            F64Array m = F64Array.copyOf(Data.splitmix(7, 12), 3, 4);
            double[] mv = m.toArray();
            assertBits("sum(all)", specSum(mv), NumJ.sum(m));
            assertBits("sum(T)", specSum(values(m.transpose())), NumJ.sum(m.transpose()));
            assertBits("mean(all)", specSum(mv) / 12, NumJ.mean(m));
            try (F64Array r = NumJ.sum(m, 1, false)) {
                assertShape("sum(axis) shape", new long[] {3}, r.shape());
            }
            try (F64Array r = NumJ.mean(m, -2, true)) {
                assertShape("mean(axis) keepdims", new long[] {1, 4}, r.shape());
            }
            m.close();
        }
    }

    static void reductionsLarge() {
        long saveR = Reduce.javaMaxElements;
        try {
            Object[][] cases = {
                {new long[] {5000, 7}, new int[] {0}},          // column tiles, 2 blocks per column
                {new long[] {6, 9000}, new int[] {1}},          // contiguous rows, multi-block
                {new long[] {3000, 400}, null},                 // full contiguous sum (293 blocks)
                {new long[] {300, 600}, new int[] {0}},         // tiles: 3 work items
                {new long[] {20, 30, 40}, new int[] {0, 2}},    // general strided walk
            };
            int seed = 200;
            for (Object[] c : cases) {
                long[] shape = (long[]) c[0];
                int[] axes = (int[]) c[1];
                double[] vals = Data.splitmix(++seed, (int) size(shape));
                try (Arena ar = Arena.ofConfined()) {
                    List<V> vs = variants(ar, vals, shape);
                    double[] ref = null;
                    for (V v : vs) {
                        for (long jm : new long[] {-1, Long.MAX_VALUE}) {
                            Reduce.javaMaxElements = jm;
                            try (F64Array r = NumJ.sum(v.a, axes, false)) {
                                double[] got = r.toArray();
                                if (ref == null) {
                                    // validate the first result against the spec and the exact bound
                                    long[] os = r.shape();
                                    double[] want = expectedReduction(v.a, axes, false, false, os);
                                    for (int i = 0; i < want.length; i++) assertBits("spec " + Arrays.toString(shape), want[i], got[i]);
                                    ref = got;
                                } else {
                                    for (int i = 0; i < ref.length; i++)
                                        assertBits(v.name + Arrays.toString(shape) + " path " + jm + " [" + i + "]", ref[i], got[i]);
                                }
                            }
                        }
                    }
                    // full sum over a transposed large array (general path with > 256 block partials)
                    if (axes == null) {
                        Reduce.javaMaxElements = -1;
                        for (V v : vs) assertBits("full sum " + v.name, ref[0], NumJ.sum(v.a));
                    }
                }
            }
        } finally {
            Reduce.javaMaxElements = saveR;
        }
    }

    static void reductionEdges() {
        allPaths(() -> {
            try (F64Array e = F64Array.allocate(4, 0, 3); F64Array z = F64Array.full(-0.0, 3, 5)) {
                try (F64Array r = NumJ.sum(e, new int[] {1}, false); F64Array m = NumJ.mean(e, new int[] {1}, true)) {
                    assertShape("empty-axis sum shape", new long[] {4, 3}, r.shape());
                    for (int i = 0; i < 12; i++) assertBits("empty sum is +0.0", 0.0, r.get(i));
                    assertShape("empty-axis mean keepdims", new long[] {4, 1, 3}, m.shape());
                    for (int i = 0; i < 12; i++) check("empty mean is NaN", Double.isNaN(m.get(i)));
                }
                try (F64Array r = NumJ.sum(e, new int[] {0}, false)) {
                    assertShape("reduce non-empty axis of empty array", new long[] {0, 3}, r.shape());
                }
                assertBits("sum of empty", 0.0, NumJ.sum(e));
                check("mean of empty", Double.isNaN(NumJ.mean(e)));
                assertBits("sum of -0.0s is +0.0", 0.0, NumJ.sum(z));
                try (F64Array r = NumJ.sum(z, new int[] {0}, false); F64Array r2 = NumJ.sum(z, new int[0], false);
                     F64Array r3 = NumJ.mean(z, new int[0], true)) {
                    assertBits("axis sum of -0.0s", 0.0, r.get(0));
                    assertShape("axis=() keeps shape", new long[] {3, 5}, r2.shape());
                    assertBits("axis=() adds the +0.0 identity", 0.0, r2.get(0));
                    assertBits("mean axis=()", 0.0, r3.get(0));
                }
                try (F64Array s = F64Array.of(1, Double.NaN, 2); F64Array i = F64Array.of(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
                     F64Array big = F64Array.of(Double.MAX_VALUE, Double.MAX_VALUE)) {
                    check("NaN propagates", Double.isNaN(NumJ.sum(s)));
                    check("Inf - Inf", Double.isNaN(NumJ.sum(i)));
                    assertBits("overflow", Double.POSITIVE_INFINITY, NumJ.sum(big));
                    assertBits("mean overflow", Double.POSITIVE_INFINITY, NumJ.mean(big));
                }
                try (F64Array s0 = F64Array.scalar(-0.0)) {
                    assertBits("0-d sum", 0.0, NumJ.sum(s0));
                    try (F64Array r = NumJ.sum(s0, (int[]) null, false)) {
                        check("0-d reduction result is 0-d", r.ndim() == 0);
                    }
                }
                throwsIAE("axis oob", () -> NumJ.sum(z, 2, false));
                throwsIAE("negative axis oob", () -> NumJ.sum(z, -3, false));
                throwsIAE("duplicate axes", () -> NumJ.sum(z, new int[] {0, -2}, false));
                throwsIAE("0-d axis", () -> NumJ.sum(F64Array.scalar(1).asReadOnly(), 0, false));
            }
        });
    }

    static void reductionOut() {
        allPaths(() -> {
            try (F64Array x = F64Array.copyOf(Data.splitmix(11, 60), 3, 4, 5); F64Array out = F64Array.allocate(3, 5);
                 F64Array big = F64Array.allocate(4, 10)) {
                F64Array r = NumJ.sum(x, new int[] {1}, false, out);
                check("returns out", r == out);
                try (F64Array want = NumJ.sum(x, new int[] {1}, false)) {
                    for (int i = 0; i < 15; i++) assertBits("out value", want.get(i), out.get(i));
                }
                F64Array strided = big.slice("1:4, ::2");                   // [3, 5] strided out
                NumJ.mean(x, new int[] {1}, false, strided);
                try (F64Array want = NumJ.mean(x, new int[] {1}, false)) {
                    for (int i = 0; i < 15; i++) assertBits("strided out", want.get(i), strided.get(i));
                }
                for (int c = 1; c < 10; c += 2) assertBits("gaps untouched", 0.0, big.get(1, c));
                throwsIAE("wrong out shape", () -> NumJ.sum(x, new int[] {1}, true, out));
                throwsType("read-only out", ReadOnlyArrayException.class, () -> NumJ.sum(x, new int[] {1}, false, out.asReadOnly()));
                // out overlapping the input: as if the input had been copied first
                double[] before = x.toArray();
                F64Array o2 = x.slice(":, 0, :");                          // [3, 5] view inside x
                try (F64Array want = NumJ.sum(x, new int[] {1}, false)) {
                    NumJ.sum(x, new int[] {1}, false, o2);
                    for (int i = 0; i < 15; i++) assertBits("overlapping out", want.get(i), o2.get(i));
                }
                check("input changed only where out is", x.get(new long[] {0, 1, 0}) == before[5]);
            }
        });
    }

    static void threadInvariance() {
        long saveEw = Elementwise.parallelMinElements, saveRed = Reduce.parallelMinElements;
        Elementwise.parallelMinElements = 0;   // exercise OpenMP at this size (defaults are measured, higher)
        Reduce.parallelMinElements = 0;
        try (F64Array a = F64Array.copyOf(Data.splitmix(21, 2_000_000), 1000, 2000);
             F64Array b = F64Array.copyOf(Data.splitmix(22, 2000), 2000)) {
            NumJ.setThreads(1);
            double s1 = NumJ.sum(a), st1 = NumJ.sum(a.transpose());
            try (F64Array c1 = NumJ.sum(a, new int[] {0}, false); F64Array r1 = NumJ.sum(a, new int[] {1}, false);
                 F64Array g1 = NumJ.sum(a.reshape(10, 100, 2000), new int[] {0, 2}, false); F64Array e1 = NumJ.add(a, b);
                 F64Array t1 = NumJ.multiply(a.transpose(), 3.0)) {
                for (int t = 2; t <= 8; t *= 2) {
                    NumJ.setThreads(t);
                    assertBits("sum t=" + t, s1, NumJ.sum(a));
                    assertBits("sum T t=" + t, st1, NumJ.sum(a.transpose()));
                    try (F64Array c = NumJ.sum(a, new int[] {0}, false); F64Array r = NumJ.sum(a, new int[] {1}, false);
                         F64Array g = NumJ.sum(a.reshape(10, 100, 2000), new int[] {0, 2}, false); F64Array e = NumJ.add(a, b);
                         F64Array tt = NumJ.multiply(a.transpose(), 3.0)) {
                        for (int i = 0; i < 2000; i++) assertBits("cols t=" + t, c1.get(i), c.get(i));
                        for (int i = 0; i < 1000; i++) assertBits("rows t=" + t, r1.get(i), r.get(i));
                        for (int i = 0; i < 100; i++) assertBits("general t=" + t, g1.get(i), g.get(i));
                        for (long i = 0; i < e.size(); i += 997) assertBits("add t=" + t, e1.get(i), e.get(i));
                        for (long i = 0; i < tt.size(); i += 991) assertBits("mul T t=" + t, t1.get(i), tt.get(i));
                    }
                }
            }
        } finally {
            NumJ.setThreads(1);
            Elementwise.parallelMinElements = saveEw;
            Reduce.parallelMinElements = saveRed;
        }
    }

    // ================================================================== fused kernels, Expr

    static void fusedStrided() {
        try (Arena ar = Arena.ofConfined()) {
            long[] shape = {30, 40};
            List<V> av = variants(ar, Data.splitmix(31, 1200), shape), bv = variants(ar, Data.splitmix(32, 1200), shape),
                    cv = variants(ar, Data.splitmix(33, 1200), shape);
            double sd = NumJ.sqdist(av.get(0).a, bv.get(0).a), ma = NumJ.sumSqMulAdd(av.get(0).a, bv.get(0).a, cv.get(0).a);
            for (int i = 0; i < av.size(); i++) {
                V a = av.get(i), b = bv.get((i + 1) % av.size()), c = cv.get((i + 2) % av.size());
                allPaths(() -> {
                    assertBits("sqdist " + a.name + "/" + b.name, sd, NumJ.sqdist(a.a, b.a));
                    assertBits("sumSqMulAdd " + a.name, ma, NumJ.sumSqMulAdd(a.a, b.a, c.a));
                });
            }
            // row kernels on row-stepped views
            F64Array big = F64Array.copyOf(Data.splitmix(34, 60 * 25), 60, 25);
            F64Array x = big.slice("::2, 2:22");                    // [30, 20], ld = 25
            F64Array q = F64Array.copyOf(Data.splitmix(35, 20), 20);
            try (F64Array xc = x.copy(); F64Array o1 = F64Array.allocate(30); F64Array o2 = F64Array.allocate(30);
                 F64Array y1 = F64Array.allocate(30, 20); F64Array n1 = F64Array.allocate(30); F64Array n2 = F64Array.allocate(30)) {
                NumJ.sqdistRows(q, xc, o1);
                NumJ.sqdistRows(q, x, o2);
                for (int r = 0; r < 30; r++) assertBits("sqdistRows ld", o1.get(r), o2.get(r));
                NumJ.normalizeRows(xc, y1, n1);
                F64Array ybig = F64Array.allocate(ar, 31, 40), y2 = ybig.slice("1:, 5:25");
                NumJ.normalizeRows(x, y2, n2);
                for (int r = 0; r < 30; r++) {
                    assertBits("norms ld", n1.get(r), n2.get(r));
                    for (int c = 0; c < 20; c++) assertBits("normalize ld", y1.get(r, c), y2.get(r, c));
                }
                assertBits("gap untouched", 0.0, ybig.get(1, 4));
                NumJ.normalizeRowsInPlace(x, null);
                for (int r = 0; r < 30; r++) for (int c = 0; c < 20; c++) assertBits("in place ld", y1.get(r, c), x.get(r, c));
                throwsIAE("column-strided x", () -> NumJ.sqdistRows(F64Array.allocate(30), xc.transpose(), F64Array.allocate(20)));
                throwsIAE("reversed rows", () -> NumJ.normalizeRows(xc.slice("::-1"), y1, null));
                throwsIAE("strided q", () -> NumJ.sqdistRows(big.slice("0:20, 0"), x, o1));
            }
            big.close();
            q.close();
        }
    }

    static void expr() {
        try (F64Array a = F64Array.copyOf(Data.splitmix(41, 5000), 50, 100); F64Array b = F64Array.copyOf(Data.splitmix(42, 5000), 50, 100);
             F64Array c = F64Array.copyOf(Data.splitmix(43, 5000), 50, 100); F64Array row = F64Array.copyOf(Data.splitmix(44, 100), 100)) {
            Expr ea = Expr.of(a), eb = Expr.of(b), ec = Expr.of(c);
            Expr.Reduction muladd = Expr.sum(ea.mul(eb).add(ec).square());
            check("muladd recognised", "sumSqMulAdd".equals(muladd.fusedKernel()));
            assertBits("muladd fused == kernel", NumJ.sumSqMulAdd(a, b, c), muladd.evaluate());
            assertBits("muladd fused == unfused", muladd.evaluateUnfused(), muladd.evaluate());
            check("c + a*b recognised", "sumSqMulAdd".equals(Expr.sum(ec.add(ea.mul(eb)).pow(2)).fusedKernel()));
            Expr d = ea.sub(eb);
            Expr.Reduction sq = Expr.sum(d.mul(d));
            check("(a-b)*(a-b) recognised", "sqdist".equals(sq.fusedKernel()));
            assertBits("sqdist fused == unfused", sq.evaluateUnfused(), sq.evaluate());
            Expr.Reduction plain = Expr.sum(Expr.of(a.transpose()));
            check("sum recognised", "sum".equals(plain.fusedKernel()));
            assertBits("sum fused", NumJ.sum(a.transpose()), plain.evaluate());
            Expr.Reduction other = Expr.sum(ea.mul(eb).sub(ec).square());
            check("unsupported pattern", other.fusedKernel() == null);
            throwsType("unsupported evaluate", UnsupportedOperationException.class, other::evaluate);
            check("unfused works", Double.isFinite(other.evaluateUnfused()));
            Expr.Reduction bc = Expr.sum(ea.sub(Expr.of(row)).square());
            check("broadcast operands are not fused", bc.fusedKernel() == null);
            double want = 0;
            try (F64Array t = NumJ.subtract(a, row); F64Array t2 = NumJ.multiply(t, t)) {
                want = NumJ.sum(t2);
            }
            assertBits("broadcast unfused", want, bc.evaluateUnfused());
            throwsType("pow(3)", UnsupportedOperationException.class, () -> ea.pow(3));
        }
    }

    // ================================================================== lifetime, threads, limits

    static void lifetime() {
        F64Array a = F64Array.copyOf(Data.splitmix(51, 2000), 40, 50);
        F64Array v = a.slice("::2").transpose();
        F64Array small = F64Array.of(1, 2, 3);
        F64Array sv = small.slice("::-1");
        a.close();
        small.close();
        check("views report freed", !v.isAlive() && !sv.isAlive());
        throwsType("get on view", IllegalStateException.class, () -> v.get(0));
        throwsType("add native path", IllegalStateException.class, () -> NumJ.add(v, 1.0));
        throwsType("add java path", IllegalStateException.class, () -> NumJ.add(sv, sv));
        throwsType("sum native", IllegalStateException.class, () -> NumJ.sum(v));
        throwsType("sum axis", IllegalStateException.class, () -> NumJ.sum(v, 0, false));
        throwsType("copy", IllegalStateException.class, v::copy);
        throwsType("sqdist", IllegalStateException.class, () -> NumJ.sqdist(v, v));
        try (F64Array r = NumJ.add(F64Array.of(1), 1.0)) {
            check("results are owners", r.isOwner());
        }
    }

    static void confinement() throws InterruptedException {
        AtomicReference<Throwable> err = new AtomicReference<>();
        try (F64Array small = F64Array.of(1, 2, 3); F64Array large = F64Array.allocate(100_000);
             F64Array largeT = F64Array.allocate(300, 300)) {
            Thread t = new Thread(() -> {
                try {
                    for (Runnable r : new Runnable[] {
                            () -> NumJ.add(small, small), () -> NumJ.add(large, large), () -> NumJ.add(largeT.transpose(), 1.0),
                            () -> NumJ.sum(small), () -> NumJ.sum(large), () -> NumJ.sum(largeT, 0, false),
                            () -> NumJ.sqdist(large, large), () -> large.copy(), () -> small.get(0)}) {
                        try {
                            r.run();
                            err.set(new AssertionError("confined array used from another thread"));
                        } catch (WrongThreadException expected) {
                            // ok
                        }
                    }
                } catch (Throwable e) {
                    err.set(e);
                }
            });
            t.start();
            t.join();
        }
        if (err.get() != null) throw new AssertionError(err.get());
    }

    static void maxDims() {
        long[] shape = new long[64];
        Arrays.fill(shape, 1);
        shape[0] = 2;
        shape[63] = 3;
        try (F64Array a = F64Array.copyOf(new double[] {1, 2, 3, 4, 5, 6}, shape)) {
            assertBits("sum 64-d", 21, NumJ.sum(a));
            try (F64Array r = NumJ.sum(a, new int[] {63}, true); F64Array e = NumJ.add(a, a.transpose().transpose())) {
                check("keepdims 64-d", r.ndim() == 64 && r.shape(0) == 2 && r.shape(63) == 1);
                assertBits("64-d axis sum", 15, r.get(1));
                assertBits("64-d add", 12, e.get(5));
            }
            throwsIAE("65th axis via newaxis", () -> a.slice("None"));
        }
    }
}
