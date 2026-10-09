package numj;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * Reduction engine (sum, mean, and the strided paths of the fused reductions). Package-private.
 *
 * <p>Summation order: each output sums a logical sequence (its reduced elements in C order of the reduced axes)
 * with the blocked algorithm documented in {@link NumJ}. The Java loop (tiny inputs) and every Fortran path
 * implement that same order, so results are bitwise identical across memory layouts, paths and thread counts.
 */
final class Reduce {
    private Reduce() {}

    static final int K_SQDIST = 1, K_MULADD = 2, K_SUM = 4;   // Fortran block-kernel selectors

    /** Total elements (outputs x reduced) up to this use the Java loop. Mutable for tests and benchmarks only. */
    static long javaMaxElements = Long.getLong("numj.reduce.javaMaxElements", 0);   // measured: native wins at every size
    /** Smallest element count for which more than one native thread is used. */
    static long parallelMinElements = Long.getLong("numj.reduce.parallelMinElements", NumJ.PARALLEL_MIN_ELEMENTS);

    private static final ValueLayout.OfDouble F64 = ValueLayout.JAVA_DOUBLE;

    // ================================================================== whole-array reductions

    /** Sum of all elements of {@code x} (C order sequence). */
    static double sumAll(F64Array x, SumOrder order) {
        x.requireAlive();
        if (x.size() == 0) return 0.0;
        return seq(K_SUM, x, x, x, javaMaxElements, order == SumOrder.MEMORY);
    }

    /**
     * Fused reductions on arrays of identical shape, any layout: {@code K_SQDIST} sums (a-b)^2,
     * {@code K_MULADD} sums (a*b+c)^2, {@code K_SUM} sums a. Same bits as the contiguous kernels.
     */
    static double seq(int op, F64Array a, F64Array b, F64Array c, long javaMax) {
        return seq(op, a, b, c, javaMax, false);
    }

    /**
     * {@code memoryOrder}: sum in memory order (axes reversed where operand 0's stride is negative and ordered
     * by decreasing stride, as for elementwise plans) instead of logical C order; see {@link SumOrder}.
     */
    static double seq(int op, F64Array a, F64Array b, F64Array c, long javaMax, boolean memoryOrder) {
        long[] shape = a.layout.shape;
        if (a.size() == 0) return 0.0;
        if (a.cContig && b.cContig && c.cContig) {            // fast path: no plan, cached segments
            long n = a.layout.size;
            try {
                if (n <= javaMax && n <= NumJ.BLOCK) return javaContig(op, a.contig, b.contig, c.contig, n);
                int t = NumJ.threadsFor(n, parallelMinElements);
                return switch (op) {
                    case K_SUM -> (double) Native.SUM.invokeExact(a.contig, n, t);
                    case K_SQDIST -> (double) Native.SQDIST.invokeExact(a.contig, b.contig, n, t);
                    default -> (double) Native.MULADD.invokeExact(a.contig, b.contig, c.contig, n, t);
                };
            } catch (Throwable e) {
                throw Native.rethrow(e);
            }
        }
        Plan p = Plan.of(shape, new long[][] {a.layout.strides, b.layout.strides, c.layout.strides},
                new long[] {a.layout.offset, b.layout.offset, c.layout.offset}, memoryOrder);
        long n = p.size();
        try {
            if (n <= javaMax)
                return javaSeq(op, p.nd, p.shape, p.st, new MemorySegment[] {a.base, b.base, c.base}, p.first);
            int t = NumJ.threadsFor(n, parallelMinElements);
            if (p.contiguous()) {
                return switch (op) {
                    case K_SUM -> (double) Native.SUM.invokeExact(seg(a, p, 0), n, t);
                    case K_SQDIST -> (double) Native.SQDIST.invokeExact(seg(a, p, 0), seg(b, p, 1), n, t);
                    default -> (double) Native.MULADD.invokeExact(seg(a, p, 0), seg(b, p, 1), seg(c, p, 2), n, t);
                };
            }
            MemorySegment sc = Plan.scratch();
            int fnd = p.writeDesc(sc, 3);
            return (double) Native.REDUCE_SEQ.invokeExact(op, fnd, sc, seg(a, p, 0), off(p, 0), seg(b, p, 1), off(p, 1),
                    seg(c, p, 2), off(p, 2));
        } catch (Throwable e) {
            throw Native.rethrow(e);
        }
    }

    // ================================================================== axis reductions

    /** NumPy {@code sum}/{@code mean} with {@code axis} ({@code null} = all), {@code keepdims} and {@code out}. */
    static F64Array sum(F64Array x, int[] axes, boolean keepdims, F64Array out, boolean mean) {
        return sum(x, axes, keepdims, out, mean, SumOrder.LOGICAL);
    }

    static F64Array sum(F64Array x, int[] axes, boolean keepdims, F64Array out, boolean mean, SumOrder order) {
        x.requireAlive();
        int nd = x.ndim();
        boolean[] red = new boolean[nd];
        if (axes == null) {
            Arrays.fill(red, true);
        } else {
            for (int ax : axes) {
                int a = Layout.normalizeAxis(ax, nd);
                if (red[a]) throw new IllegalArgumentException("duplicate value in 'axis': " + Arrays.toString(axes));
                red[a] = true;
            }
        }
        long[] xs = x.layout.shape;
        int rn = 0;
        long K = 1;
        for (int d = 0; d < nd; d++) {
            if (red[d]) K *= xs[d];
            else rn++;
        }
        long[] rshape = new long[keepdims ? nd : rn];
        for (int d = 0, j = 0; d < nd; d++) {
            if (keepdims) rshape[d] = red[d] ? 1 : xs[d];
            else if (!red[d]) rshape[j++] = xs[d];
        }
        boolean fresh = out == null;
        if (fresh) out = F64Array.result(rshape);
        else Elementwise.checkOut(out, rshape);
        F64Array tmp = null;
        try {
            if (!fresh && x.overlaps(out)) {
                tmp = x.copy();
                x = tmp;
            }
            if (out.size() == 0) return out;
            if (K == 0) {
                out.fill(mean ? Double.NaN : 0.0);
                return out;
            }
            reduceInto(x, red, keepdims, out, K, order == SumOrder.MEMORY);
            if (mean) Elementwise.scalar(Elementwise.DIV, out, (double) K, out);
            return out;
        } catch (RuntimeException | Error e) {
            if (fresh) out.close();
            throw e;
        } finally {
            if (tmp != null) tmp.close();
        }
    }

    /** out = sum over the reduced axes; out.size() > 0 and every reduced extent > 0. */
    private static void reduceInto(F64Array x, boolean[] red, boolean keepdims, F64Array out, long K, boolean memoryOrder) {
        int nd = x.ndim();
        long[] xs = x.layout.shape, xst = x.layout.strides, ost = out.layout.strides;
        long[] ksh = new long[nd], kx = new long[nd], ko = new long[nd], rsh = new long[nd], rx = new long[nd];
        int nk = 0, nr = 0;
        for (int d = 0, od = 0; d < nd; d++) {
            if (red[d]) {
                if (keepdims) od++;
                if (xs[d] != 1) { rsh[nr] = xs[d]; rx[nr] = xst[d]; nr++; }
            } else {
                if (xs[d] != 1) { ksh[nk] = xs[d]; kx[nk] = xst[d]; ko[nk] = ost[od]; nk++; }
                od++;
            }
        }
        if (nr == 0) {   // nothing to sum: each output is 0.0 + x (NumPy adds the +0.0 identity)
            Elementwise.scalar(Elementwise.ADD, x.reshape(out.layout.shape), 0.0, out);
            return;
        }
        // kept axes: any order is fine (outputs are independent); sort by decreasing |x stride| so the
        // unit-stride axis (if any) is innermost, then merge axes contiguous in both x and out.
        for (int i = 1; i < nk; i++) {
            long a = ksh[i], b = kx[i], c = ko[i];
            int j = i - 1;
            while (j >= 0 && Math.abs(kx[j]) < Math.abs(b)) { ksh[j + 1] = ksh[j]; kx[j + 1] = kx[j]; ko[j + 1] = ko[j]; j--; }
            ksh[j + 1] = a; kx[j + 1] = b; ko[j + 1] = c;
        }
        nk = merge(nk, ksh, kx, ko);
        // reduced axes: logical order is the summation order; merge only adjacent contiguous axes.
        long xfirst = x.layout.offset;   // byte offset of the first element of the summed sequence
        if (memoryOrder) {
            // memory order: walk reversed reduced axes forwards and order reduced axes by decreasing |stride|
            for (int i = 0; i < nr; i++) {
                if (rx[i] < 0) {
                    xfirst += rx[i] * (rsh[i] - 1);
                    rx[i] = -rx[i];
                }
            }
            for (int i = 1; i < nr; i++) {
                long a = rsh[i], b = rx[i];
                int j = i - 1;
                while (j >= 0 && rx[j] < b) { rsh[j + 1] = rsh[j]; rx[j + 1] = rx[j]; j--; }
                rsh[j + 1] = a; rx[j + 1] = b;
            }
        }
        nr = merge(nr, rsh, rx, null);

        long xlo = xfirst, xhi = xfirst, olo = out.layout.offset, ohi = out.layout.offset;
        for (int i = 0; i < nk; i++) {
            long e = ksh[i] - 1;
            if (kx[i] < 0) xlo += kx[i] * e; else xhi += kx[i] * e;
            if (ko[i] < 0) olo += ko[i] * e; else ohi += ko[i] * e;
        }
        for (int i = 0; i < nr; i++) {
            long e = rsh[i] - 1;
            if (rx[i] < 0) xlo += rx[i] * e; else xhi += rx[i] * e;
        }
        long nout = out.size();
        try {
            if (nout * K <= javaMaxElements) {
                javaSum(nk, ksh, kx, ko, nr, rsh, rx, x.base, xfirst, out.base, out.layout.offset);
                return;
            }
            MemorySegment sc = Plan.scratch();
            long[] desc = new long[3 * nk + 2 * nr];
            for (int i = 0; i < nk; i++) {          // innermost first
                int s = nk - 1 - i;
                desc[i] = ksh[s];
                desc[nk + i] = kx[s] / F64Array.ITEM;
                desc[2 * nk + i] = ko[s] / F64Array.ITEM;
            }
            for (int i = 0; i < nr; i++) {
                int s = nr - 1 - i;
                desc[3 * nk + i] = rsh[s];
                desc[3 * nk + nr + i] = rx[s] / F64Array.ITEM;
            }
            MemorySegment.copy(desc, 0, sc, ValueLayout.JAVA_LONG, 0, desc.length);
            Native.SUM_ND.invokeExact(nk, nr, sc, x.base.asSlice(xlo, xhi + F64Array.ITEM - xlo),
                    (xfirst - xlo) / F64Array.ITEM, out.base.asSlice(olo, ohi + F64Array.ITEM - olo),
                    (out.layout.offset - olo) / F64Array.ITEM, NumJ.threadsFor(nout * K, parallelMinElements));
        } catch (Throwable e) {
            throw Native.rethrow(e);
        }
    }

    /** Merges adjacent axes (C order) that are contiguous in every stride array given; returns the new count. */
    private static int merge(int n, long[] sh, long[] s1, long[] s2) {
        if (n == 0) return 0;
        int j = 0;
        for (int i = 1; i < n; i++) {
            boolean m = s1[j] == s1[i] * sh[i] && (s2 == null || s2[j] == s2[i] * sh[i]);
            if (m) {
                sh[j] *= sh[i];
                s1[j] = s1[i];
                if (s2 != null) s2[j] = s2[i];
            } else {
                j++;
                sh[j] = sh[i];
                s1[j] = s1[i];
                if (s2 != null) s2[j] = s2[i];
            }
        }
        return j + 1;
    }

    // ================================================================== Java implementations (tiny inputs)

    private static void javaSum(int nk, long[] ksh, long[] kx, long[] ko, int nr, long[] rsh, long[] rx,
                                MemorySegment xs, long xoff, MemorySegment os, long ooff) {
        long[] idx = new long[nk];
        long px = xoff, po = ooff;
        long[][] st = {Arrays.copyOf(rx, nr), new long[nr], new long[nr]};
        long[] rshape = Arrays.copyOf(rsh, nr);
        MemorySegment[] segs = {xs, xs, xs};
        while (true) {
            os.set(F64, po, javaSeq(K_SUM, nr, rshape, st, segs, new long[] {px, px, px}));
            int d = nk - 1;
            while (d >= 0) {
                idx[d]++;
                px += kx[d];
                po += ko[d];
                if (idx[d] < ksh[d]) break;
                px -= kx[d] * ksh[d];
                po -= ko[d] * ksh[d];
                idx[d] = 0;
                d--;
            }
            if (d < 0) return;
        }
    }

    /**
     * The blocked summation of numj_kernels.f90 over a strided sequence (C order, axis 0 outermost):
     * 4096-element blocks, element j of a block into lane j mod 16 (lanes start at +0.0), lanes combined by the
     * fixed tree, block results combined pairwise.
     */
    static double javaSeq(int op, int nd, long[] shape, long[][] st, MemorySegment[] seg, long[] first) {
        long n = 1;
        for (int d = 0; d < nd; d++) n *= shape[d];
        if (n == 0) return 0.0;
        double[] acc = new double[NumJ.LANES];
        double[] part = new double[(int) Math.max(1, (n + NumJ.BLOCK - 1) / NumJ.BLOCK)];
        int nb = 0, lane = 0, pos = 0;
        long pa = first[0], pb = first[1], pc = first[2];
        long[] idx = new long[nd];
        long inner = nd == 0 ? 1 : shape[nd - 1];
        long sa = nd == 0 ? 0 : st[0][nd - 1], sb = nd == 0 ? 0 : st[1][nd - 1], sc = nd == 0 ? 0 : st[2][nd - 1];
        while (true) {
            long a = pa, b = pb, c = pc;
            for (long j = 0; j < inner; j++) {
                double v;
                switch (op) {
                    case K_SUM -> v = seg[0].get(F64, a);
                    case K_SQDIST -> {
                        double t = seg[0].get(F64, a) - seg[1].get(F64, b);
                        v = t * t;
                    }
                    default -> {
                        double t = seg[0].get(F64, a) * seg[1].get(F64, b) + seg[2].get(F64, c);
                        v = t * t;
                    }
                }
                acc[lane] += v;
                lane = (lane + 1) & (NumJ.LANES - 1);
                if (++pos == NumJ.BLOCK) {
                    part[nb++] = laneTree(acc);
                    Arrays.fill(acc, 0.0);
                    lane = 0;
                    pos = 0;
                }
                a += sa;
                b += sb;
                c += sc;
            }
            int d = nd - 2;
            while (d >= 0) {
                idx[d]++;
                pa += st[0][d];
                pb += st[1][d];
                pc += st[2][d];
                if (idx[d] < shape[d]) break;
                pa -= st[0][d] * shape[d];
                pb -= st[1][d] * shape[d];
                pc -= st[2][d] * shape[d];
                idx[d] = 0;
                d--;
            }
            if (d < 0) break;
        }
        if (pos > 0) part[nb++] = laneTree(acc);
        return nb == 1 ? part[0] : pairwise(part, nb);
    }

    /** One block (n <= BLOCK) of contiguous elements: lanes then lane tree; same bits as the Fortran block kernels. */
    private static double javaContig(int op, MemorySegment a, MemorySegment b, MemorySegment c, long n) {
        double[] acc = new double[NumJ.LANES];
        for (long i = 0; i < n; i++) {
            long o = i * F64Array.ITEM;
            double v;
            switch (op) {
                case K_SUM -> v = a.get(F64, o);
                case K_SQDIST -> {
                    double t = a.get(F64, o) - b.get(F64, o);
                    v = t * t;
                }
                default -> {
                    double t = a.get(F64, o) * b.get(F64, o) + c.get(F64, o);
                    v = t * t;
                }
            }
            acc[(int) (i & (NumJ.LANES - 1))] += v;
        }
        return laneTree(acc);
    }

    /** Fortran lane_tree: (1:8)+(9:16), (1:4)+(5:8), (1:2)+(3:4), 1+2. */
    static double laneTree(double[] acc) {
        double[] t = new double[8];
        for (int i = 0; i < 8; i++) t[i] = acc[i] + acc[i + 8];
        for (int i = 0; i < 4; i++) t[i] = t[i] + t[i + 4];
        for (int i = 0; i < 2; i++) t[i] = t[i] + t[i + 2];
        return t[0] + t[1];
    }

    /** Fortran pairwise: repeatedly p[i] = p[2i] + p[2i+1], carrying an odd last element. */
    static double pairwise(double[] p, int n) {
        int len = n;
        while (len > 1) {
            int half = len / 2;
            for (int i = 0; i < half; i++) p[i] = p[2 * i] + p[2 * i + 1];
            if ((len & 1) == 1) {
                p[half] = p[len - 1];
                len = half + 1;
            } else {
                len = half;
            }
        }
        return p[0];
    }

    // ================================================================== helpers

    private static MemorySegment seg(F64Array a, Plan p, int k) {
        long lo = p.lo(k);
        return a.base.asSlice(lo, p.hi(k) - lo);
    }

    private static long off(Plan p, int k) {
        return (p.first[k] - p.lo(k)) / F64Array.ITEM;
    }
}
