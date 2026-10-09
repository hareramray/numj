package numj;

import java.util.Arrays;

/**
 * Immutable strided layout: shape, byte strides and byte offset of element {@code [0, ..., 0]} from the start of
 * the underlying buffer. All rules that do not depend on the element type live here (validation, slicing,
 * axis permutation, reshape-as-view, broadcasting, contiguity). Package-private; {@link NDArray} exposes it.
 *
 * <p>Invariants (established by every factory and preserved by every transformation):
 * <ul>
 *   <li>{@code 0 <= ndim <= MAX_NDIM}; every extent {@code >= 0}; the product of the non-zero extents times the
 *       item size fits in a {@code long} (so every stride and offset computed below is exact);</li>
 *   <li>for non-empty layouts every addressed byte lies in {@code [spanLo(), spanHi())}; callers check that range
 *       against the buffer once, when the layout is attached to memory.</li>
 * </ul>
 */
final class Layout {
    /** Maximum number of dimensions (NumPy 2.x NPY_MAXDIMS). */
    static final int MAX_NDIM = 64;

    final long[] shape;
    final long[] strides;
    final long offset;
    final long size;
    final int itemSize;

    private Layout(long[] shape, long[] strides, long offset, int itemSize) {
        this.shape = shape;
        this.strides = strides;
        this.offset = offset;
        this.itemSize = itemSize;
        long n = 1;
        for (long d : shape) n *= d;   // cannot overflow: checked by checkShape or derived from a checked layout
        this.size = n;
    }

    int ndim() { return shape.length; }

    // ------------------------------------------------------------------ construction

    /** Validates a shape: rank, non-negative extents, and that the byte size fits in a long. Returns a copy. */
    static long[] checkShape(long[] shape, int itemSize) {
        if (shape.length > MAX_NDIM)
            throw new IllegalArgumentException("too many dimensions: " + shape.length + " > " + MAX_NDIM);
        long bytes = itemSize;
        for (long d : shape) {
            if (d < 0) throw new IllegalArgumentException("negative dimensions are not allowed: " + Arrays.toString(shape));
            if (d == 0) continue;
            try {
                bytes = Math.multiplyExact(bytes, d);
            } catch (ArithmeticException e) {
                throw new IllegalArgumentException("array is too big: shape " + Arrays.toString(shape)
                        + " exceeds 2^63-1 bytes");
            }
        }
        return shape.clone();
    }

    /** C-order (row-major) layout at the given byte offset. Extents of 0 count as 1 for strides (as in NumPy). */
    static Layout cOrder(long[] shape, int itemSize, long offset) {
        long[] sh = checkShape(shape, itemSize);
        return new Layout(sh, cStrides(sh, itemSize), offset, itemSize);
    }

    static long[] cStrides(long[] shape, int itemSize) {
        long[] st = new long[shape.length];
        long acc = itemSize;
        for (int i = shape.length - 1; i >= 0; i--) {
            st[i] = acc;
            acc *= Math.max(shape[i], 1);
        }
        return st;
    }

    /** Arbitrary layout from caller-supplied values (validated except for the buffer range). */
    static Layout of(long[] shape, long[] strides, long offset, int itemSize) {
        long[] sh = checkShape(shape, itemSize);
        if (strides.length != sh.length)
            throw new IllegalArgumentException("strides has " + strides.length + " entries, shape has " + sh.length);
        for (long s : strides)
            if (s % itemSize != 0)
                throw new IllegalArgumentException("strides must be multiples of the item size " + itemSize + ": "
                        + Arrays.toString(strides));
        if (offset % itemSize != 0) throw new IllegalArgumentException("offset must be a multiple of " + itemSize);
        Layout l = new Layout(sh, strides.clone(), offset, itemSize);
        if (l.size > 0) {
            // span computation must not overflow
            try {
                long lo = offset, hi = offset;
                for (int i = 0; i < sh.length; i++) {
                    long ext = Math.multiplyExact(strides[i], sh[i] - 1);
                    if (ext < 0) lo = Math.addExact(lo, ext); else hi = Math.addExact(hi, ext);
                }
                Math.addExact(hi, itemSize);
                if (lo < 0) throw new IllegalArgumentException("layout addresses bytes before the buffer start");
            } catch (ArithmeticException e) {
                throw new IllegalArgumentException("strides overflow: " + Arrays.toString(strides));
            }
        }
        return l;
    }

    // ------------------------------------------------------------------ queries

    /** Lowest byte offset touched (inclusive); equals {@link #offset} when empty. */
    long spanLo() {
        if (size == 0) return offset;
        long lo = offset;
        for (int i = 0; i < shape.length; i++) if (strides[i] < 0) lo += strides[i] * (shape[i] - 1);
        return lo;
    }

    /** One past the highest byte touched; equals {@link #offset} when empty. */
    long spanHi() {
        if (size == 0) return offset;
        long hi = offset;
        for (int i = 0; i < shape.length; i++) if (strides[i] > 0) hi += strides[i] * (shape[i] - 1);
        return hi + itemSize;
    }

    /** NumPy's C_CONTIGUOUS flag (relaxed strides: extents of 1 are ignored; empty arrays are contiguous). */
    boolean isC() {
        if (size == 0) return true;
        long expect = itemSize;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (shape[i] != 1) {
                if (strides[i] != expect) return false;
                expect *= shape[i];
            }
        }
        return true;
    }

    /** NumPy's F_CONTIGUOUS flag. */
    boolean isF() {
        if (size == 0) return true;
        long expect = itemSize;
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] != 1) {
                if (strides[i] != expect) return false;
                expect *= shape[i];
            }
        }
        return true;
    }

    /**
     * True if no two distinct indices can address overlapping bytes. This is a sufficient test (sort the axes of
     * extent > 1 by |stride|; each stride must step over everything the smaller axes can reach). Layouts built by
     * this library from such a layout always pass; it is only consulted for caller-supplied strides.
     */
    boolean provablyNoSelfOverlap() {
        if (size <= 1) return true;
        int k = 0;
        long[] abs = new long[shape.length];
        long[] ext = new long[shape.length];
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] > 1) {
                abs[k] = Math.abs(strides[i]);
                ext[k] = shape[i] - 1;
                k++;
            }
        }
        // insertion sort by |stride|
        for (int i = 1; i < k; i++) {
            long a = abs[i], e = ext[i];
            int j = i - 1;
            while (j >= 0 && abs[j] > a) { abs[j + 1] = abs[j]; ext[j + 1] = ext[j]; j--; }
            abs[j + 1] = a;
            ext[j + 1] = e;
        }
        long reach = itemSize;   // bytes covered by the axes processed so far, starting from one element
        for (int i = 0; i < k; i++) {
            if (abs[i] < reach) return false;
            reach = abs[i] * ext[i] + reach;
        }
        return true;
    }

    /** Byte offset of a full multi-index (bounds-checked; negative indices are NOT wrapped). */
    long offsetOf(long[] idx) {
        if (idx.length != shape.length)
            throw new IllegalArgumentException("expected " + shape.length + " indices, got " + idx.length);
        long off = offset;
        for (int i = 0; i < idx.length; i++) off += java.util.Objects.checkIndex(idx[i], shape[i]) * strides[i];
        return off;
    }

    /** Byte offset of the element at C-order flat index {@code i} (bounds-checked). */
    long offsetOfFlat(long i) {
        java.util.Objects.checkIndex(i, size);
        long off = offset;
        for (int k = shape.length - 1; k >= 0; k--) {
            long d = shape[k];
            off += (i % d) * strides[k];
            i /= d;
        }
        return off;
    }

    // ------------------------------------------------------------------ transformations (all return views)

    /** NumPy basic indexing. See {@link Ix}. */
    Layout index(Ix[] items) {
        int consuming = 0, ellipses = 0;
        for (Ix it : items) {
            if (it.kind == Ix.Kind.INDEX || it.kind == Ix.Kind.SLICE) consuming++;
            else if (it.kind == Ix.Kind.ELLIPSIS) ellipses++;
        }
        if (ellipses > 1) throw new IllegalArgumentException("an index can only have a single ellipsis ('...')");
        if (consuming > shape.length)
            throw new IllegalArgumentException("too many indices: array is " + shape.length + "-dimensional, but "
                    + consuming + " were indexed");
        long[] nsh = new long[MAX_NDIM + 1];
        long[] nst = new long[MAX_NDIM + 1];
        int n = 0, ax = 0;
        long off = offset;
        boolean sawEllipsis = false;
        for (Ix it : items) {
            switch (it.kind) {
                case INDEX -> {
                    long d = shape[ax];
                    long i = it.index < 0 ? it.index + d : it.index;
                    if (i < 0 || i >= d)
                        throw new IndexOutOfBoundsException("index " + it.index + " is out of bounds for axis " + ax
                                + " with size " + d);
                    off += i * strides[ax];
                    ax++;
                }
                case SLICE -> {
                    long d = shape[ax];
                    long[] sl = adjustSlice(it, d);   // {start, length, step}
                    if (n >= MAX_NDIM) throw tooMany();
                    nsh[n] = sl[1];
                    nst[n] = sl[1] > 1 ? Math.multiplyExact(strides[ax], sl[2]) : strides[ax];
                    if (sl[1] > 0) off += sl[0] * strides[ax];
                    n++;
                    ax++;
                }
                case NEW_AXIS -> {
                    if (n >= MAX_NDIM) throw tooMany();
                    nsh[n] = 1;
                    nst[n] = 0;
                    n++;
                }
                case ELLIPSIS -> {
                    int fill = shape.length - consuming;
                    for (int k = 0; k < fill; k++) {
                        if (n >= MAX_NDIM) throw tooMany();
                        nsh[n] = shape[ax];
                        nst[n] = strides[ax];
                        n++;
                        ax++;
                    }
                    sawEllipsis = true;
                }
            }
        }
        if (!sawEllipsis) {
            while (ax < shape.length) {
                if (n >= MAX_NDIM) throw tooMany();
                nsh[n] = shape[ax];
                nst[n] = strides[ax];
                n++;
                ax++;
            }
        }
        return new Layout(Arrays.copyOf(nsh, n), Arrays.copyOf(nst, n), off, itemSize);
    }

    private static IllegalArgumentException tooMany() {
        return new IllegalArgumentException("result would have more than " + MAX_NDIM + " dimensions");
    }

    /** Python's PySlice_Unpack + PySlice_AdjustIndices. Returns {start, length, step}. */
    static long[] adjustSlice(Ix it, long n) {
        long step = it.step;
        if (step == 0) throw new IllegalArgumentException("slice step cannot be zero");
        if (step < -Long.MAX_VALUE) step = -Long.MAX_VALUE;
        long start = it.hasStart ? it.start : (step < 0 ? Long.MAX_VALUE : 0);
        long stop = it.hasStop ? it.stop : (step < 0 ? Long.MIN_VALUE : Long.MAX_VALUE);
        if (start < 0) {
            start += n;
            if (start < 0) start = step < 0 ? -1 : 0;
        } else if (start >= n) {
            start = step < 0 ? n - 1 : n;
        }
        if (stop < 0) {
            stop += n;
            if (stop < 0) stop = step < 0 ? -1 : 0;
        } else if (stop >= n) {
            stop = step < 0 ? n - 1 : n;
        }
        long len;
        if (step < 0) len = stop < start ? (start - stop - 1) / (-step) + 1 : 0;
        else len = start < stop ? (stop - start - 1) / step + 1 : 0;
        return new long[] {start, len, step};
    }

    Layout permute(int[] axes) {
        int nd = shape.length;
        if (axes.length != nd)
            throw new IllegalArgumentException("axes don't match array: got " + axes.length + " axes for a "
                    + nd + "-dimensional array");
        long[] sh = new long[nd], st = new long[nd];
        boolean[] seen = new boolean[nd];
        for (int i = 0; i < nd; i++) {
            int a = normalizeAxis(axes[i], nd);
            if (seen[a]) throw new IllegalArgumentException("repeated axis in transpose: " + Arrays.toString(axes));
            seen[a] = true;
            sh[i] = shape[a];
            st[i] = strides[a];
        }
        return new Layout(sh, st, offset, itemSize);
    }

    Layout transpose() {
        int nd = shape.length;
        int[] axes = new int[nd];
        for (int i = 0; i < nd; i++) axes[i] = nd - 1 - i;
        return permute(axes);
    }

    /** Resolves a single {@code -1} and checks the element count. */
    static long[] resolveReshape(long[] newShape, long size, int itemSize) {
        long[] sh = newShape.clone();
        int unknown = -1;
        long known = 1;
        for (int i = 0; i < sh.length; i++) {
            if (sh[i] == -1) {
                if (unknown >= 0) throw new IllegalArgumentException("can only specify one unknown dimension");
                unknown = i;
            } else if (sh[i] < 0) {
                throw new IllegalArgumentException("negative dimensions are not allowed: " + Arrays.toString(newShape));
            } else {
                try {
                    known = Math.multiplyExact(known, sh[i]);
                } catch (ArithmeticException e) {
                    throw new IllegalArgumentException("array is too big: " + Arrays.toString(newShape));
                }
            }
        }
        if (unknown >= 0) {
            if (known == 0 || size % known != 0)
                throw new IllegalArgumentException("cannot reshape array of size " + size + " into shape "
                        + shapeString(newShape));
            sh[unknown] = size / known;
        } else if (known != size) {
            throw new IllegalArgumentException("cannot reshape array of size " + size + " into shape "
                    + shapeString(newShape));
        }
        return checkShape(sh, itemSize);
    }

    /**
     * Reshape (C order) without copying, or {@code null} if this layout cannot express the new shape as a view.
     * C-contiguous layouts always succeed. Otherwise NumPy's {@code _attempt_nocopy_reshape} rule is applied:
     * each group of old axes that is merged or split must be contiguous with respect to each other.
     */
    Layout reshapeView(long[] resolved) {
        if (isC()) return new Layout(resolved.clone(), cStrides(resolved, itemSize), offset, itemSize);
        // drop length-1 axes from the old layout
        int on = 0;
        long[] od = new long[shape.length], os = new long[shape.length];
        for (int i = 0; i < shape.length; i++) if (shape[i] != 1) { od[on] = shape[i]; os[on] = strides[i]; on++; }
        int nn = resolved.length;
        long[] ns = new long[nn];
        int oi = 0, oj = 1, ni = 0, nj = 1;
        while (ni < nn && oi < on) {
            long np = resolved[ni], op = od[oi];
            while (np != op) {
                if (np < op) np *= resolved[nj++];
                else op *= od[oj++];
            }
            for (int ok = oi; ok < oj - 1; ok++) if (os[ok] != od[ok + 1] * os[ok + 1]) return null;
            ns[nj - 1] = os[oj - 1];
            for (int nk = nj - 1; nk > ni; nk--) ns[nk - 1] = ns[nk] * resolved[nk];
            ni = nj++;
            oi = oj++;
        }
        long last = ni >= 1 ? ns[ni - 1] : itemSize;
        for (int nk = ni; nk < nn; nk++) ns[nk] = last;
        return new Layout(resolved.clone(), ns, offset, itemSize);
    }

    /** Read-only broadcast view onto {@code target} (stride 0 on broadcast axes). */
    Layout broadcastTo(long[] target) {
        long[] t = checkShape(target, itemSize);
        int nd = t.length, d = nd - shape.length;
        if (d < 0) throw new IllegalArgumentException("cannot broadcast shape " + shapeString(shape) + " to "
                + shapeString(t) + ": target has fewer dimensions");
        long[] st = new long[nd];
        for (int i = 0; i < nd; i++) {
            if (i < d) continue;
            long s = shape[i - d];
            if (s == t[i]) st[i] = strides[i - d];
            else if (s == 1) st[i] = 0;
            else throw new IllegalArgumentException("cannot broadcast shape " + shapeString(shape) + " to "
                        + shapeString(t));
        }
        return new Layout(t, st, offset, itemSize);
    }

    /** Strides of this layout as seen when broadcast to {@code target} (already known to be compatible). */
    long[] broadcastStrides(long[] target) {
        int nd = target.length, d = nd - shape.length;
        long[] st = new long[nd];
        for (int i = d; i < nd; i++) st[i] = shape[i - d] == target[i] ? strides[i - d] : 0;
        return st;
    }

    // ------------------------------------------------------------------ static helpers

    /** NumPy broadcasting of several shapes; {@link IllegalArgumentException} names the first conflict. */
    static long[] broadcastShapes(long[]... shapes) {
        int nd = 0;
        for (long[] s : shapes) nd = Math.max(nd, s.length);
        long[] out = new long[nd];
        Arrays.fill(out, 1);
        for (int k = 0; k < shapes.length; k++) {
            long[] s = shapes[k];
            int d = nd - s.length;
            for (int i = 0; i < s.length; i++) {
                long a = out[d + i], b = s[i];
                if (a == b || b == 1) continue;
                if (a == 1) { out[d + i] = b; continue; }
                throw new IllegalArgumentException("shape mismatch: objects cannot be broadcast to a single shape; "
                        + "mismatch between " + shapeString(Arrays.copyOfRange(out, d, nd)) + " and arg " + k
                        + " with shape " + shapeString(s));
            }
        }
        return out;
    }

    /** Maps {@code axis} in {@code [-ndim, ndim)} to {@code [0, ndim)}. */
    static int normalizeAxis(int axis, int ndim) {
        if (axis < -ndim || axis >= ndim)
            throw new IllegalArgumentException("axis " + axis + " is out of bounds for array of dimension " + ndim);
        return axis < 0 ? axis + ndim : axis;
    }

    static String shapeString(long[] shape) {
        StringBuilder b = new StringBuilder("[");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) b.append(", ");
            b.append(shape[i]);
        }
        return b.append(']').toString();
    }

    @Override
    public String toString() {
        return "Layout" + shapeString(shape) + " strides=" + Arrays.toString(strides) + " offset=" + offset;
    }
}
