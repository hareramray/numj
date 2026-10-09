package numj;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Iteration plan for k operands that share one logical shape: axes of extent 1 are dropped, and adjacent axes
 * that are contiguous with respect to each other in <em>every</em> operand are merged. For elementwise work the
 * axes may also be reversed and reordered (by operand 0's strides) because the order of independent element
 * operations does not affect results; reductions keep the logical order because it defines the summation order.
 *
 * <p>Axes are stored in C order (index 0 outermost). Strides are in bytes; {@code first[k]} is the byte offset
 * (within operand k's base segment) of the first logical element.
 */
final class Plan {
    final int nd;
    final long[] shape;
    final long[][] st;
    final long[] first;

    private Plan(int nd, long[] shape, long[][] st, long[] first) {
        this.nd = nd;
        this.shape = shape;
        this.st = st;
        this.first = first;
    }

    /** Builds a plan. {@code shape} must have no zero extents (callers return early for empty work). */
    static Plan of(long[] shape, long[][] strides, long[] offsets, boolean reorder) {
        int m = strides.length;
        int n0 = shape.length;
        long[] sh = new long[n0];
        long[][] st = new long[m][n0];
        long[] first = offsets.clone();
        int nd = 0;
        for (int d = 0; d < n0; d++) {
            if (shape[d] == 1) continue;
            sh[nd] = shape[d];
            for (int k = 0; k < m; k++) st[k][nd] = strides[k][d];
            nd++;
        }
        if (reorder && nd > 0) {
            // make operand 0's strides positive (reverse those axes in every operand) ...
            for (int d = 0; d < nd; d++) {
                if (st[0][d] < 0) {
                    for (int k = 0; k < m; k++) {
                        first[k] += st[k][d] * (sh[d] - 1);
                        st[k][d] = -st[k][d];
                    }
                }
            }
            // ... and order axes by decreasing operand-0 stride (stable insertion sort; nd <= 64)
            for (int i = 1; i < nd; i++) {
                long key = st[0][i], ext = sh[i];
                long[] col = new long[m];
                for (int k = 0; k < m; k++) col[k] = st[k][i];
                int j = i - 1;
                while (j >= 0 && st[0][j] < key) {
                    sh[j + 1] = sh[j];
                    for (int k = 0; k < m; k++) st[k][j + 1] = st[k][j];
                    j--;
                }
                sh[j + 1] = ext;
                for (int k = 0; k < m; k++) st[k][j + 1] = col[k];
            }
        }
        // merge axis i (inner) into the previous kept axis j (outer) when contiguous in every operand
        int j = 0;
        for (int i = 1; i < nd; i++) {
            boolean merge = true;
            for (int k = 0; k < m && merge; k++) merge = st[k][j] == st[k][i] * sh[i];
            if (merge) {
                sh[j] *= sh[i];
                for (int k = 0; k < m; k++) st[k][j] = st[k][i];
            } else {
                j++;
                sh[j] = sh[i];
                for (int k = 0; k < m; k++) st[k][j] = st[k][i];
            }
        }
        if (nd > 0) nd = j + 1;
        return new Plan(nd, sh, st, first);
    }

    long size() {
        long n = 1;
        for (int d = 0; d < nd; d++) n *= shape[d];
        return n;
    }

    /** True if every operand is a single run of unit-stride elements (or a single element). */
    boolean contiguous() {
        if (nd == 0) return true;
        if (nd > 1) return false;
        for (long[] s : st) if (s[0] != F64Array.ITEM) return false;
        return true;
    }

    /** Lowest byte offset touched by operand k. */
    long lo(int k) {
        long lo = first[k];
        for (int d = 0; d < nd; d++) if (st[k][d] < 0) lo += st[k][d] * (shape[d] - 1);
        return lo;
    }

    /** One past the highest byte touched by operand k. */
    long hi(int k) {
        long hi = first[k];
        for (int d = 0; d < nd; d++) if (st[k][d] > 0) hi += st[k][d] * (shape[d] - 1);
        return hi + F64Array.ITEM;
    }

    /**
     * Writes the Fortran descriptor {@code desc(nd, 1 + operands)} (extents, then element strides per operand;
     * dim 1 = innermost) into the scratch segment, using at least one dimension. Columns for operands beyond the
     * plan's own (a scalar operand) get stride 0. Returns the Fortran nd.
     */
    int writeDesc(MemorySegment scratch, int operands) {
        int fnd = Math.max(nd, 1);
        for (int d = 0; d < fnd; d++) {
            int src = nd - 1 - d;
            scratch.setAtIndex(ValueLayout.JAVA_LONG, d, nd == 0 ? 1 : shape[src]);
            for (int k = 0; k < operands; k++)
                scratch.setAtIndex(ValueLayout.JAVA_LONG, (long) (k + 1) * fnd + d,
                        nd == 0 || k >= st.length ? 0 : st[k][src] / F64Array.ITEM);
        }
        return fnd;
    }

    // ------------------------------------------------------------------ per-thread native scratch

    /** Bytes reserved for descriptors: 64 dims x 5 columns x 8 bytes, plus 8 for a scalar operand. */
    static final long SCRATCH_BYTES = 64 * 5 * 8 + 64;
    static final long SCALAR_OFFSET = 64 * 5 * 8;

    private static final ThreadLocal<MemorySegment> SCRATCH =
            ThreadLocal.withInitial(() -> Arena.ofAuto().allocate(SCRATCH_BYTES, 64));

    /** Native scratch for descriptors, private to the calling thread (only used for the duration of one call). */
    static MemorySegment scratch() {
        return SCRATCH.get();
    }
}
