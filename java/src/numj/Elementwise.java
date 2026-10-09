package numj;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * Elementwise engine: broadcasting, output validation, overlap handling and path selection for binary
 * operations and copies. Package-private; the public API is {@link NumJ} and {@link F64Array}.
 *
 * <p>Path selection (after {@link Plan} has merged axes):
 * <ul>
 *   <li>at most {@link #javaMaxElements} elements: a Java loop (no native call; avoids the ~8-12 ns downcall);</li>
 *   <li>every operand one unit-stride run: a contiguous Fortran kernel;</li>
 *   <li>otherwise: the n-d strided Fortran kernel (unit-stride and broadcast rows still use vector loops).</li>
 * </ul>
 * All paths produce identical bits: each element is one correctly rounded IEEE operation.
 *
 * <p>Overlap rules: an input that addresses exactly the output's elements in the same order is used in place
 * (Fortran receives it as the single in-out argument). Any other overlap between an input and the output is
 * resolved by copying that input first, so the result is always as if no memory were shared (NumPy's
 * guarantee). Inputs never get copied otherwise.
 */
final class Elementwise {
    private Elementwise() {}

    static final int ADD = 1, SUB = 2, MUL = 3, DIV = 4, RSUB = 5, RDIV = 6;
    private static final int COPY = 0;

    /**
     * Strided (planned) operations with at most this many elements use the Java loop. Measured crossover: the Java
     * loop wins up to ~64 elements, where the native path's planning and descriptor cost dominates (results/nd).
     * Mutable for tests and benchmarks only.
     */
    static long javaMaxElements = Long.getLong("numj.ew.javaMaxElements", 64);
    /** Same for the contiguous fast path: the Java loop wins only below ~4 elements (results/nd). */
    static long contigJavaMaxElements = Long.getLong("numj.ew.contigJavaMaxElements", 4);
    /** Smallest element count for which more than one native thread is used (measured: <= 1.3x below this). */
    static long parallelMinElements = Long.getLong("numj.ew.parallelMinElements", 1L << 23);

    private static final ValueLayout.OfDouble F64 = ValueLayout.JAVA_DOUBLE;
    private static final int NONE = 0, IDENTICAL = 1, OVERLAP = 2;

    static int reversed(int op) {
        return switch (op) {
            case SUB -> RSUB;
            case DIV -> RDIV;
            case RSUB -> SUB;
            case RDIV -> DIV;
            default -> op;
        };
    }

    static double apply(int op, double x, double y) {
        return switch (op) {
            case ADD -> x + y;
            case SUB -> x - y;
            case MUL -> x * y;
            case DIV -> x / y;
            case RSUB -> y - x;
            case RDIV -> y / x;
            default -> x;   // COPY
        };
    }

    // ================================================================== public entry points

    /** {@code out = a op b} with broadcasting; allocates the result if {@code out} is null. */
    static F64Array binary(int op, F64Array a, F64Array b, F64Array out) {
        if (a.cContig && b.cContig && (out == null || out.cContig) && Arrays.equals(a.layout.shape, b.layout.shape)
                && (out == null || Arrays.equals(out.layout.shape, a.layout.shape))) {
            F64Array r = contiguousBinary(op, a, b, out);
            if (r != null) return r;
        }
        a.requireAlive();
        b.requireAlive();
        long[] shape = Layout.broadcastShapes(a.layout.shape, b.layout.shape);
        if (out == null) {
            F64Array r = F64Array.result(shape);
            try {
                outOfPlace(op, r, a, b);
            } catch (RuntimeException | Error e) {
                r.close();
                throw e;
            }
            return r;
        }
        checkOut(out, shape);
        F64Array ta = null, tb = null;
        try {
            int ra = role(a, out), rb = role(b, out);
            if (ra == OVERLAP) { ta = a.copy(); a = ta; ra = NONE; }
            if (rb == OVERLAP) { tb = b.copy(); b = tb; rb = NONE; }
            if (out.size() == 0) return out;
            if (ra == IDENTICAL && rb == IDENTICAL) self(op, out);
            else if (ra == IDENTICAL) inPlace(op, out, b);
            else if (rb == IDENTICAL) inPlace(reversed(op), out, a);
            else outOfPlace(op, out, a, b);
        } finally {
            if (ta != null) ta.close();
            if (tb != null) tb.close();
        }
        return out;
    }

    /** {@code out = a op s}; for {@code s op a} pass {@code reversed(op)}. */
    static F64Array scalar(int op, F64Array a, double s, F64Array out) {
        if (a.cContig && (out == null || (out.cContig && Arrays.equals(out.layout.shape, a.layout.shape)))) {
            F64Array r = contiguousScalar(op, a, s, out);
            if (r != null) return r;
        }
        a.requireAlive();
        if (out == null) {
            F64Array r = F64Array.result(a.layout.shape);
            try {
                outOfPlaceScalar(op, r, a, s);
            } catch (RuntimeException | Error e) {
                r.close();
                throw e;
            }
            return r;
        }
        checkOut(out, a.layout.shape);
        F64Array ta = null;
        try {
            int ra = role(a, out);
            if (ra == OVERLAP) { ta = a.copy(); a = ta; ra = NONE; }
            if (out.size() == 0) return out;
            if (ra == IDENTICAL) inPlaceScalar(op, out, s);
            else outOfPlaceScalar(op, out, a, s);
        } finally {
            if (ta != null) ta.close();
        }
        return out;
    }

    /** Bitwise copy {@code dst[...] = src[...]}; shapes must be equal ({@code src} may be a broadcast view). */
    static void copy(F64Array src, F64Array dst) {
        src.requireAlive();
        dst.requireAlive();
        dst.requireWritable("destination");
        if (!src.sameShape(dst))
            throw new IllegalArgumentException("shape mismatch: " + src.shapeString() + " vs " + dst.shapeString());
        if (dst.size() == 0) return;
        int r = role(src, dst);
        if (r == IDENTICAL) return;
        F64Array tmp = null;
        try {
            if (r == OVERLAP) { tmp = src.copy(); src = tmp; }
            Plan p = Plan.of(dst.layout.shape, new long[][] {dst.layout.strides, src.layout.strides},
                    new long[] {dst.layout.offset, src.layout.offset}, true);
            long n = p.size();
            if (p.contiguous()) {
                MemorySegment.copy(src.base, p.first[1], dst.base, p.first[0], n * F64Array.ITEM);
            } else if (n <= javaMaxElements) {
                javaLoop(COPY, p, new MemorySegment[] {dst.base, src.base}, 1, -1, 0);
            } else {
                MemorySegment sc = Plan.scratch();
                int fnd = p.writeDesc(sc, 2);
                Native.COPY_ND.invokeExact(fnd, sc, seg(dst, p, 0), elemOff(p, 0), seg(src, p, 1), elemOff(p, 1),
                        NumJ.threadsFor(n, parallelMinElements));
            }
        } catch (Throwable e) {
            throw Native.rethrow(e);
        } finally {
            if (tmp != null) tmp.close();
        }
    }

    // ================================================================== fast path: all operands C-contiguous, same shape

    /**
     * {@code out = a op b} when a, b (and out, if given) are C-contiguous with one shape. No plan, no descriptor,
     * no slice allocation. Returns null to fall back to the general path (partial overlap needs a copy there).
     * For same-shape C-contiguous arrays, "identical layout" is simply "same first address".
     */
    private static F64Array contiguousBinary(int op, F64Array a, F64Array b, F64Array out) {
        a.requireAlive();
        b.requireAlive();
        if (out == null) {
            F64Array r = F64Array.result(a.layout.shape);
            try {
                contig(op, r.layout.size, r.contig, a.contig, b.contig, 0);
            } catch (RuntimeException | Error e) {
                r.close();
                throw e;
            }
            return r;
        }
        out.requireAlive();
        out.requireWritable("out");
        long n = out.layout.size;
        if (n == 0) return out;
        boolean ia = a.overlaps(out), ib = b.overlaps(out);
        if ((ia && a.addrLo != out.addrLo) || (ib && b.addrLo != out.addrLo)) return null;
        if (ia && ib) contigInPlace(op, n, out.contig, null, 0, true);
        else if (ia) contigInPlace(op, n, out.contig, b.contig, 0, false);
        else if (ib) contigInPlace(reversed(op), n, out.contig, a.contig, 0, false);
        else contig(op, n, out.contig, a.contig, b.contig, 0);
        return out;
    }

    private static F64Array contiguousScalar(int op, F64Array a, double s, F64Array out) {
        a.requireAlive();
        if (out == null) {
            F64Array r = F64Array.result(a.layout.shape);
            try {
                contig(op, r.layout.size, r.contig, a.contig, null, s);
            } catch (RuntimeException | Error e) {
                r.close();
                throw e;
            }
            return r;
        }
        out.requireAlive();
        out.requireWritable("out");
        long n = out.layout.size;
        if (n == 0) return out;
        boolean ia = a.overlaps(out);
        if (ia && a.addrLo != out.addrLo) return null;
        if (ia) contigInPlace(op, n, out.contig, null, s, false);
        else contig(op, n, out.contig, a.contig, null, s);
        return out;
    }

    /** o = x op (y or s); o disjoint from x and y. */
    private static void contig(int op, long n, MemorySegment o, MemorySegment x, MemorySegment y, double s) {
        if (n == 0) return;
        try {
            if (n <= Math.min(javaMaxElements, contigJavaMaxElements)) {
                for (long i = 0, e = n * F64Array.ITEM; i < e; i += F64Array.ITEM)
                    o.set(F64, i, apply(op, x.get(F64, i), y == null ? s : y.get(F64, i)));
            } else if (y != null) {
                Native.EW_CONTIG.invokeExact(op, n, x, y, o, NumJ.threadsFor(n, parallelMinElements));
            } else {
                Native.EW_CONTIG_S.invokeExact(op, n, x, s, o, NumJ.threadsFor(n, parallelMinElements));
            }
        } catch (Throwable e) {
            throw Native.rethrow(e);
        }
    }

    /** x = x op (y or s), or x = x op x when {@code self}; y disjoint from x. */
    private static void contigInPlace(int op, long n, MemorySegment x, MemorySegment y, double s, boolean self) {
        try {
            if (n <= Math.min(javaMaxElements, contigJavaMaxElements)) {
                for (long i = 0, e = n * F64Array.ITEM; i < e; i += F64Array.ITEM) {
                    double u = x.get(F64, i);
                    x.set(F64, i, apply(op, u, self ? u : y == null ? s : y.get(F64, i)));
                }
            } else if (self) {
                Native.EW_CONTIG_SELF.invokeExact(op, n, x, NumJ.threadsFor(n, parallelMinElements));
            } else if (y != null) {
                Native.EW_CONTIG_IP.invokeExact(op, n, x, y, NumJ.threadsFor(n, parallelMinElements));
            } else {
                Native.EW_CONTIG_IP_S.invokeExact(op, n, x, s, NumJ.threadsFor(n, parallelMinElements));
            }
        } catch (Throwable e) {
            throw Native.rethrow(e);
        }
    }

    // ================================================================== validation

    static void checkOut(F64Array out, long[] shape) {
        out.requireAlive();
        if (!java.util.Arrays.equals(out.layout.shape, shape))
            throw new IllegalArgumentException("out has shape " + out.shapeString() + ", result has shape "
                    + Layout.shapeString(shape));
        out.requireWritable("out");
    }

    /** How input {@code x} relates to output {@code out} (x already validated as broadcastable to out). */
    private static int role(F64Array x, F64Array out) {
        if (!x.overlaps(out)) return NONE;
        return x.sameLayout(out) ? IDENTICAL : OVERLAP;
    }

    // ================================================================== execution

    private static void outOfPlace(int op, F64Array out, F64Array a, F64Array b) {
        long[] shape = out.layout.shape;
        if (out.size() == 0) return;
        Plan p = Plan.of(shape, new long[][] {out.layout.strides, a.layout.broadcastStrides(shape),
                b.layout.broadcastStrides(shape)}, new long[] {out.layout.offset, a.layout.offset, b.layout.offset}, true);
        long n = p.size();
        try {
            if (n <= javaMaxElements) {
                javaLoop(op, p, new MemorySegment[] {out.base, a.base, b.base}, 1, 2, 0);
            } else if (p.contiguous()) {
                Native.EW_CONTIG.invokeExact(op, n, seg(a, p, 1), seg(b, p, 2), seg(out, p, 0),
                        NumJ.threadsFor(n, parallelMinElements));
            } else {
                MemorySegment sc = Plan.scratch();
                int fnd = p.writeDesc(sc, 3);
                Native.EW_ND.invokeExact(op, fnd, sc, seg(out, p, 0), elemOff(p, 0), seg(a, p, 1), elemOff(p, 1),
                        seg(b, p, 2), elemOff(p, 2), NumJ.threadsFor(n, parallelMinElements));
            }
        } catch (Throwable e) {
            throw Native.rethrow(e);
        }
    }

    private static void outOfPlaceScalar(int op, F64Array out, F64Array a, double s) {
        long[] shape = out.layout.shape;
        if (out.size() == 0) return;
        Plan p = Plan.of(shape, new long[][] {out.layout.strides, a.layout.strides},
                new long[] {out.layout.offset, a.layout.offset}, true);
        long n = p.size();
        try {
            if (n <= javaMaxElements) {
                javaLoop(op, p, new MemorySegment[] {out.base, a.base}, 1, -1, s);
            } else if (p.contiguous()) {
                Native.EW_CONTIG_S.invokeExact(op, n, seg(a, p, 1), s, seg(out, p, 0), NumJ.threadsFor(n, parallelMinElements));
            } else {
                // the scalar becomes a stride-0 operand stored in the scratch segment
                MemorySegment sc = Plan.scratch();
                int fnd = p.writeDesc(sc, 3);
                sc.set(F64, Plan.SCALAR_OFFSET, s);
                Native.EW_ND.invokeExact(op, fnd, sc, seg(out, p, 0), elemOff(p, 0), seg(a, p, 1), elemOff(p, 1),
                        sc.asSlice(Plan.SCALAR_OFFSET, 8), 0L, NumJ.threadsFor(n, parallelMinElements));
            }
        } catch (Throwable e) {
            throw Native.rethrow(e);
        }
    }

    /** {@code x = x op y}; y is disjoint from x (or a broadcast view whose memory is disjoint). */
    private static void inPlace(int op, F64Array x, F64Array y) {
        long[] shape = x.layout.shape;
        Plan p = Plan.of(shape, new long[][] {x.layout.strides, y.layout.broadcastStrides(shape)},
                new long[] {x.layout.offset, y.layout.offset}, true);
        long n = p.size();
        try {
            if (n <= javaMaxElements) {
                javaLoop(op, p, new MemorySegment[] {x.base, y.base}, 0, 1, 0);
            } else if (p.contiguous()) {
                Native.EW_CONTIG_IP.invokeExact(op, n, seg(x, p, 0), seg(y, p, 1), NumJ.threadsFor(n, parallelMinElements));
            } else {
                MemorySegment sc = Plan.scratch();
                int fnd = p.writeDesc(sc, 2);
                Native.EW_ND_IP.invokeExact(op, fnd, sc, seg(x, p, 0), elemOff(p, 0), seg(y, p, 1), elemOff(p, 1),
                        NumJ.threadsFor(n, parallelMinElements));
            }
        } catch (Throwable e) {
            throw Native.rethrow(e);
        }
    }

    private static void inPlaceScalar(int op, F64Array x, double s) {
        Plan p = Plan.of(x.layout.shape, new long[][] {x.layout.strides}, new long[] {x.layout.offset}, true);
        long n = p.size();
        try {
            if (n <= javaMaxElements) {
                javaLoop(op, p, new MemorySegment[] {x.base}, 0, -1, s);
            } else if (p.contiguous()) {
                Native.EW_CONTIG_IP_S.invokeExact(op, n, seg(x, p, 0), s, NumJ.threadsFor(n, parallelMinElements));
            } else {
                MemorySegment sc = Plan.scratch();
                int fnd = p.writeDesc(sc, 2);   // the scalar is operand y with stride 0
                sc.set(F64, Plan.SCALAR_OFFSET, s);
                Native.EW_ND_IP.invokeExact(op, fnd, sc, seg(x, p, 0), elemOff(p, 0), sc.asSlice(Plan.SCALAR_OFFSET, 8), 0L,
                        NumJ.threadsFor(n, parallelMinElements));
            }
        } catch (Throwable e) {
            throw Native.rethrow(e);
        }
    }

    /** {@code x = x op x}. */
    private static void self(int op, F64Array x) {
        Plan p = Plan.of(x.layout.shape, new long[][] {x.layout.strides}, new long[] {x.layout.offset}, true);
        long n = p.size();
        try {
            if (n <= javaMaxElements) {
                javaLoop(op, p, new MemorySegment[] {x.base}, 0, 0, 0);
            } else if (p.contiguous()) {
                Native.EW_CONTIG_SELF.invokeExact(op, n, seg(x, p, 0), NumJ.threadsFor(n, parallelMinElements));
            } else {
                MemorySegment sc = Plan.scratch();
                int fnd = p.writeDesc(sc, 1);
                Native.EW_ND_SELF.invokeExact(op, fnd, sc, seg(x, p, 0), elemOff(p, 0), NumJ.threadsFor(n, parallelMinElements));
            }
        } catch (Throwable e) {
            throw Native.rethrow(e);
        }
    }

    // ================================================================== helpers

    /** Segment handed to native code for operand k: from its lowest to its highest addressed byte. */
    private static MemorySegment seg(F64Array a, Plan p, int k) {
        long lo = p.lo(k);
        return a.base.asSlice(lo, p.hi(k) - lo);
    }

    /** Element offset of operand k's first logical element from the start of {@link #seg}. */
    private static long elemOff(Plan p, int k) {
        return (p.first[k] - p.lo(k)) / F64Array.ITEM;
    }

    /**
     * Java loop over a plan: operand 0 is written with {@code apply(op, in1, in2)}, where in1/in2 are read from
     * operands {@code i1}/{@code i2}, or are the scalar {@code s} when the index is negative.
     */
    private static void javaLoop(int op, Plan p, MemorySegment[] seg, int i1, int i2, double s) {
        int nd = p.nd, m = seg.length;
        long[] off = p.first.clone();
        if (nd == 0) {
            double x = i1 >= 0 ? seg[i1].get(F64, off[i1]) : s;
            double y = i2 >= 0 ? seg[i2].get(F64, off[i2]) : s;
            seg[0].set(F64, off[0], apply(op, x, y));
            return;
        }
        long[] idx = new long[nd];
        long inner = p.shape[nd - 1];
        long[] step = new long[m];
        for (int k = 0; k < m; k++) step[k] = p.st[k][nd - 1];
        while (true) {
            long o0 = off[0];
            long o1 = i1 >= 0 ? off[i1] : 0, o2 = i2 >= 0 ? off[i2] : 0;
            long s0 = step[0], s1 = i1 >= 0 ? step[i1] : 0, s2 = i2 >= 0 ? step[i2] : 0;
            for (long j = 0; j < inner; j++) {
                double x = i1 >= 0 ? seg[i1].get(F64, o1) : s;
                double y = i2 >= 0 ? seg[i2].get(F64, o2) : s;
                seg[0].set(F64, o0, apply(op, x, y));
                o0 += s0;
                o1 += s1;
                o2 += s2;
            }
            int d = nd - 2;
            while (d >= 0) {
                idx[d]++;
                for (int k = 0; k < m; k++) off[k] += p.st[k][d];
                if (idx[d] < p.shape[d]) break;
                for (int k = 0; k < m; k++) off[k] -= p.st[k][d] * p.shape[d];
                idx[d] = 0;
                d--;
            }
            if (d < 0) return;
        }
    }
}
