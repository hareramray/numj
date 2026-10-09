package numj;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * An n-dimensional, strided array of IEEE-754 {@code float64} values in native (off-heap) memory.
 *
 * <p>See {@link NDArray} for the layout, ownership, lifetime, thread and writability rules shared by all arrays.
 *
 * <h2>Creation</h2>
 * {@link #allocate(long...)} (zero-filled), {@link #full}, {@link #ones}, {@link #scalar}, {@link #arange},
 * {@link #of}, {@link #copyOf(double[], long...)} create C-ordered owners; variants taking an {@link Arena}
 * allocate into a caller-managed arena; {@link #wrap} views caller-managed memory. Shapes are validated:
 * negative extents, more than {@value Layout#MAX_NDIM} dimensions, or a byte size above {@code 2^63-1} throw
 * {@link IllegalArgumentException}.
 *
 * <h2>Views (never copy)</h2>
 * {@link #slice(Ix...)}, {@link #row}, {@link #transpose()}, {@link #permute}, {@link #swapAxes},
 * {@link #reshape(long...)}, {@link #flattenView()}, {@link #broadcastTo}, {@link #asReadOnly()}.
 * Views share memory and inherit writability; {@link #broadcastTo} views are always read-only.
 *
 * <h2>Copies (always copy, return a C-ordered owner)</h2>
 * {@link #copy()}, {@link #reshapeCopy}, {@link #flattenCopy()}, and the heap transfers {@link #toArray()} /
 * {@link #copyTo(double[])}.
 *
 * <h2>Element access</h2>
 * All accessors are bounds-checked ({@link IndexOutOfBoundsException}); multi-indices are not wrapped (use
 * {@link #slice} for negative indices). {@link #get(long)} / {@link #set(long, double)} use a <em>flat</em>
 * row-major index for every rank (NumPy {@code a.flat[i]}).
 */
public final class F64Array extends NDArray {
    static final long ALIGNMENT = 64;
    static final int ITEM = Double.BYTES;
    private static final ValueLayout.OfDouble F64 = ValueLayout.JAVA_DOUBLE;

    F64Array(Arena arena, MemorySegment base, Layout layout, boolean writable) {
        super(arena, base, layout, writable);
    }

    @Override
    public DType dtype() { return DType.FLOAT64; }

    // ================================================================== creation

    /** New zero-filled rank-1 array of length {@code n}, confined to the calling thread. */
    public static F64Array allocate(long n) {
        return owner(Arena.ofConfined(), new long[] {n});
    }

    /** New zero-filled rank-2 array {@code [rows, cols]}, confined to the calling thread. */
    public static F64Array allocate(long rows, long cols) {
        return owner(Arena.ofConfined(), new long[] {rows, cols});
    }

    /** New zero-filled C-ordered array of any shape ({@code allocate()} is a 0-d scalar), confined to this thread. */
    public static F64Array allocate(long... shape) {
        return owner(Arena.ofConfined(), shape);
    }

    /** Same as {@link #allocate(long...)} (NumPy {@code zeros}). */
    public static F64Array zeros(long... shape) {
        return allocate(shape);
    }

    /**
     * New zero-filled array inside a caller-managed arena. The result is not an owner: it is freed when
     * {@code arena} is closed (its {@link #close()} does nothing), and follows the arena's thread rules.
     */
    public static F64Array allocate(Arena arena, long... shape) {
        Objects.requireNonNull(arena, "arena");
        long[] sh = Layout.checkShape(shape, ITEM);
        MemorySegment s = arena.allocate(bytes(sh), ALIGNMENT);
        return new F64Array(null, s, Layout.cOrder(sh, ITEM, 0), true);
    }

    /** Like {@link #allocate(long)} but usable from any thread. */
    public static F64Array allocateShared(long n) {
        return owner(Arena.ofShared(), new long[] {n});
    }

    /** Like {@link #allocate(long, long)} but usable from any thread. */
    public static F64Array allocateShared(long rows, long cols) {
        return owner(Arena.ofShared(), new long[] {rows, cols});
    }

    /** Like {@link #allocate(long...)} but usable from any thread. */
    public static F64Array allocateShared(long... shape) {
        return owner(Arena.ofShared(), shape);
    }

    /** New confined array with every element equal to {@code value}. */
    public static F64Array full(double value, long... shape) {
        F64Array a = allocate(shape);
        a.fill(value);
        return a;
    }

    public static F64Array ones(long... shape) {
        return full(1.0, shape);
    }

    /** New confined 0-dimensional array holding {@code value}. */
    public static F64Array scalar(double value) {
        F64Array a = allocate(new long[0]);
        a.base.set(F64, 0, value);
        return a;
    }

    /** New confined rank-1 array {@code [0, 1, ..., n-1]}. */
    public static F64Array arange(long n) {
        F64Array a = allocate(n);
        for (long i = 0; i < n; i++) a.base.setAtIndex(F64, i, i);
        return a;
    }

    /** New confined rank-1 array holding a copy of {@code values}. */
    public static F64Array of(double... values) {
        F64Array a = allocate(values.length);
        a.copyFrom(values);
        return a;
    }

    /** New confined rank-2 array holding a copy of {@code data}, interpreted row-major. */
    public static F64Array copyOf(double[] data, long rows, long cols) {
        return copyOf(data, new long[] {rows, cols});
    }

    /** New confined C-ordered array of the given shape holding a copy of {@code data}. */
    public static F64Array copyOf(double[] data, long... shape) {
        F64Array a = allocate(shape);
        try {
            a.copyFrom(data);
        } catch (RuntimeException e) {
            a.close();
            throw e;
        }
        return a;
    }

    /**
     * A rank-2 view over memory the caller manages. The segment must be native, 8-byte aligned and exactly
     * {@code rows * cols * 8} bytes. The view never frees the memory; it is read-only if the segment is.
     */
    public static F64Array wrap(MemorySegment segment, long rows, long cols) {
        return wrap(segment, new long[] {rows, cols});
    }

    /** C-ordered view over a native, 8-byte-aligned segment of exactly {@code size * 8} bytes. */
    public static F64Array wrap(MemorySegment segment, long... shape) {
        checkWrapSegment(segment);
        long[] sh = Layout.checkShape(shape, ITEM);
        if (segment.byteSize() != bytes(sh))
            throw new IllegalArgumentException("segment is " + segment.byteSize() + " bytes, shape "
                    + Layout.shapeString(sh) + " needs " + bytes(sh));
        return new F64Array(null, segment, Layout.cOrder(sh, ITEM, 0), !segment.isReadOnly());
    }

    /**
     * Strided view over a native, 8-byte-aligned segment (NumPy {@code as_strided}, but checked): offset and
     * strides must be multiples of 8 and every addressed element must lie inside the segment. The view is
     * writable only if the segment is writable <em>and</em> the layout provably never addresses the same
     * element twice; otherwise it is read-only, so that writes can never be ambiguous.
     */
    public static F64Array wrap(MemorySegment segment, long byteOffset, long[] shape, long[] byteStrides) {
        checkWrapSegment(segment);
        Layout l = Layout.of(shape, byteStrides, byteOffset, ITEM);
        if (l.size > 0 && (l.spanLo() < 0 || l.spanHi() > segment.byteSize()))
            throw new IllegalArgumentException("layout addresses bytes [" + l.spanLo() + ", " + l.spanHi()
                    + ") outside the " + segment.byteSize() + "-byte segment");
        return new F64Array(null, segment, l, !segment.isReadOnly() && l.provablyNoSelfOverlap());
    }

    private static void checkWrapSegment(MemorySegment segment) {
        Objects.requireNonNull(segment, "segment");
        if (!segment.isNative()) throw new IllegalArgumentException("segment must be native (off-heap)");
        if (segment.address() % ITEM != 0) throw new IllegalArgumentException("segment must be 8-byte aligned");
    }

    static F64Array owner(Arena arena, long[] shape) {
        try {
            long[] sh = Layout.checkShape(shape, ITEM);
            MemorySegment s = arena.allocate(bytes(sh), ALIGNMENT);
            return new F64Array(arena, s, Layout.cOrder(sh, ITEM, 0), true);
        } catch (RuntimeException | Error e) {
            arena.close();
            throw e;
        }
    }

    /**
     * New confined owner whose contents the caller will overwrite completely (used for operation results).
     * By default the memory comes from the C runtime's {@code malloc} without zero-filling (every caller overwrites
     * all elements): measured 17-22% faster for allocating calls of 10^5-10^7 elements. {@code -Dnumj.resultAlloc=arena}
     * restores zero-filled {@link Arena#allocate}. Factories such as {@link #allocate} always zero-fill.
     */
    static F64Array result(long[] shape) {
        if (!MALLOC_RESULTS) return owner(Arena.ofConfined(), shape);
        long[] sh = Layout.checkShape(shape, ITEM);
        Arena arena = Arena.ofConfined();
        try {
            return new F64Array(arena, RawAlloc.allocate(arena, bytes(sh), ALIGNMENT), Layout.cOrder(sh, ITEM, 0), true);
        } catch (RuntimeException | Error e) {
            arena.close();
            throw e;
        }
    }

    private static final boolean MALLOC_RESULTS = !"arena".equals(System.getProperty("numj.resultAlloc", "malloc"));

    private static long bytes(long[] shape) {
        long n = ITEM;
        for (long d : shape) n *= d;   // validated by checkShape
        return n;
    }

    // ================================================================== shape (rank-1/2 compatibility)

    /** Rows of a rank-2 array, 1 for rank 1. Other ranks throw {@link IllegalStateException}. */
    public long rows() {
        return switch (ndim()) {
            case 1 -> 1;
            case 2 -> layout.shape[0];
            default -> throw new IllegalStateException("rows() needs rank 1 or 2, array is " + shapeString());
        };
    }

    /** Columns of a rank-2 array, or the length of a rank-1 array. Other ranks throw {@link IllegalStateException}. */
    public long cols() {
        return switch (ndim()) {
            case 1 -> layout.shape[0];
            case 2 -> layout.shape[1];
            default -> throw new IllegalStateException("cols() needs rank 1 or 2, array is " + shapeString());
        };
    }

    /** True if both arrays have the same shape. */
    public boolean sameShape(F64Array o) {
        return super.sameShape(o);
    }

    // ================================================================== views

    F64Array view(Layout l) {
        return new F64Array(null, base, l, writable);
    }

    /** Basic indexing view (NumPy {@code a[...]}); see {@link Ix}. */
    public F64Array slice(Ix... index) {
        return view(layout.index(index));
    }

    /** Basic indexing view from a NumPy-syntax string, e.g. {@code a.slice("::-1, 1:3, None")}. */
    public F64Array slice(String numpyIndex) {
        return slice(Ix.parse(numpyIndex));
    }

    /** Rank-1 view of row {@code r} of a rank-2 array. */
    public F64Array row(long r) {
        requireRank(2);
        Objects.checkIndex(r, layout.shape[0]);
        return slice(Ix.at(r));
    }

    /** View with the axes reversed (NumPy {@code a.T}). */
    public F64Array transpose() {
        return view(layout.transpose());
    }

    /** View with axes permuted: result axis {@code i} is input axis {@code axes[i]} (NumPy {@code transpose(axes)}). */
    public F64Array permute(int... axes) {
        return view(layout.permute(axes));
    }

    /** View with two axes exchanged. */
    public F64Array swapAxes(int a, int b) {
        int nd = ndim();
        int x = Layout.normalizeAxis(a, nd), y = Layout.normalizeAxis(b, nd);
        int[] axes = new int[nd];
        for (int i = 0; i < nd; i++) axes[i] = i;
        axes[x] = y;
        axes[y] = x;
        return permute(axes);
    }

    /** Rank-2 reshape view; same rules as {@link #reshape(long...)}. */
    public F64Array reshape(long newRows, long newCols) {
        return reshape(new long[] {newRows, newCols});
    }

    /**
     * Reshape (C order) <em>as a view</em>; never copies. One extent may be {@code -1} (inferred).
     * A view is possible when the array is C-contiguous, or when every group of axes that is merged or split is
     * itself contiguous (NumPy's no-copy rule). Otherwise this throws {@link IllegalArgumentException}; use
     * {@link #reshapeCopy} or check {@link #canReshapeView} first. (NumPy's {@code reshape} silently copies in
     * that case; numj does not, because a copy is a new owner that must be closed.)
     */
    public F64Array reshape(long... shape) {
        long[] r = Layout.resolveReshape(shape, size(), ITEM);
        Layout l = layout.reshapeView(r);
        if (l == null)
            throw new IllegalArgumentException("cannot reshape " + shapeString() + " to " + Layout.shapeString(r)
                    + " without copying (array is not contiguous in the required way); use reshapeCopy");
        return view(l);
    }

    /** True if {@link #reshape(long...)} would succeed (the element count matches and a view is possible). */
    public boolean canReshapeView(long... shape) {
        try {
            return layout.reshapeView(Layout.resolveReshape(shape, size(), ITEM)) != null;
        } catch (IllegalArgumentException e) {
            return false;
        }
    }

    /** Reshape into a new C-ordered owner (always copies, like NumPy {@code reshape(..., copy=True)}). */
    public F64Array reshapeCopy(long... shape) {
        long[] r = Layout.resolveReshape(shape, size(), ITEM);
        F64Array c = copy();
        return new F64Array(c.arena, c.base, Layout.cOrder(r, ITEM, 0), true);
    }

    /**
     * Rank-1 view of all elements in row-major order.
     *
     * @deprecated NumPy's {@code flatten()} returns a copy; this method has returned a view since numj 0.1 and still
     *     does, so existing code keeps working. It will return a copy in a future release. Use {@link #flattenView()}
     *     for a view (identical behaviour to this method) or {@link #flattenCopy()} for NumPy semantics.
     *     For arrays that cannot be flattened without copying it throws {@link IllegalArgumentException}.
     */
    @Deprecated(since = "0.2")
    public F64Array flatten() {
        return flattenView();
    }

    /** Rank-1 view of all elements in row-major order; {@link IllegalArgumentException} if a view is impossible. */
    public F64Array flattenView() {
        return reshape(size());
    }

    /** Rank-1 C-ordered copy (NumPy {@code flatten()}); a new owner. */
    public F64Array flattenCopy() {
        return reshapeCopy(size());
    }

    /** Read-only broadcast view (NumPy {@code broadcast_to}); broadcast axes have stride 0. */
    public F64Array broadcastTo(long... shape) {
        return new F64Array(null, base, layout.broadcastTo(shape), false);
    }

    /** Read-only view of the same elements. */
    public F64Array asReadOnly() {
        return new F64Array(null, base, layout, false);
    }

    /** New C-ordered owner with a copy of the elements (NumPy {@code a.copy()}, order 'C'). */
    public F64Array copy() {
        requireAlive();
        F64Array c = result(layout.shape);
        try {
            Elementwise.copy(this, c);
        } catch (RuntimeException | Error e) {
            c.close();
            throw e;
        }
        return c;
    }

    // ================================================================== element access (bounds-checked)

    /** Element at flat (row-major) index {@code i}; valid for every rank. */
    public double get(long i) {
        return base.get(F64, cContig ? layout.offset + Objects.checkIndex(i, layout.size) * ITEM : layout.offsetOfFlat(i));
    }

    public void set(long i, double v) {
        requireWritable("array");
        base.set(F64, cContig ? layout.offset + Objects.checkIndex(i, layout.size) * ITEM : layout.offsetOfFlat(i), v);
    }

    /** Element {@code (r, c)} of a rank-2 array. */
    public double get(long r, long c) {
        requireRank(2);
        return base.get(F64, layout.offset + Objects.checkIndex(r, layout.shape[0]) * layout.strides[0]
                + Objects.checkIndex(c, layout.shape[1]) * layout.strides[1]);
    }

    public void set(long r, long c, double v) {
        requireRank(2);
        requireWritable("array");
        base.set(F64, layout.offset + Objects.checkIndex(r, layout.shape[0]) * layout.strides[0]
                + Objects.checkIndex(c, layout.shape[1]) * layout.strides[1], v);
    }

    /** Element at a full multi-index ({@code get()} reads a 0-d array). */
    public double get(long... index) {
        return base.get(F64, layout.offsetOf(index));
    }

    /** Sets the element at a full multi-index. */
    public void set(long[] index, double v) {
        requireWritable("array");
        base.set(F64, layout.offsetOf(index), v);
    }

    /** The only element of a size-1 array (NumPy {@code item()}). */
    public double item() {
        if (layout.size != 1) throw new IllegalArgumentException("item() needs an array of size 1, got " + shapeString());
        return base.get(F64, layout.offset);
    }

    public void fill(double v) {
        requireWritable("array");
        if (cContig) {
            MemorySegment s = base.asSlice(layout.offset, layout.size * ITEM);
            if (Double.doubleToRawLongBits(v) == 0L) s.fill((byte) 0);
            else for (long i = 0, n = layout.size; i < n; i++) s.setAtIndex(F64, i, v);
        } else {
            for (long i = 0, n = layout.size; i < n; i++) base.set(F64, layout.offsetOfFlat(i), v);
        }
    }

    // ================================================================== bulk transfers (row-major element order)

    /** Copies all of {@code src} into this array in row-major order; {@code src.length} must equal {@link #size()}. */
    public void copyFrom(double[] src) {
        if (src.length != size()) throw new IllegalArgumentException("source length " + src.length + " != size " + size());
        copyFrom(src, 0, 0, src.length);
    }

    /** Copies {@code len} elements from {@code src[srcPos..]} to flat (row-major) indices {@code dstIndex..}. */
    public void copyFrom(double[] src, int srcPos, long dstIndex, int len) {
        Objects.checkFromIndexSize(srcPos, len, src.length);
        Objects.checkFromIndexSize(dstIndex, len, size());
        requireWritable("array");
        if (cContig) MemorySegment.copy(src, srcPos, base, F64, layout.offset + dstIndex * ITEM, len);
        else for (int k = 0; k < len; k++) base.set(F64, layout.offsetOfFlat(dstIndex + k), src[srcPos + k]);
    }

    /** Copies all elements, row-major, into {@code dst}; {@code dst.length} must equal {@link #size()}. */
    public void copyTo(double[] dst) {
        if (dst.length != size()) throw new IllegalArgumentException("destination length " + dst.length + " != size " + size());
        if (cContig) MemorySegment.copy(base, F64, layout.offset, dst, 0, dst.length);
        else for (int k = 0; k < dst.length; k++) dst[k] = base.get(F64, layout.offsetOfFlat(k));
    }

    /** New heap array with the elements in row-major order (size must be at most {@code Integer.MAX_VALUE - 8}). */
    public double[] toArray() {
        if (size() > Integer.MAX_VALUE - 8) throw new IllegalStateException("array too large for a Java array: " + size());
        double[] d = new double[(int) size()];
        copyTo(d);
        return d;
    }

    /**
     * Assigns {@code src}, broadcast to this array's shape (NumPy {@code np.copyto(self, src)}). If {@code src}
     * overlaps this array the result is as if {@code src} had been copied first.
     */
    public void copyFrom(F64Array src) {
        requireWritable("array");
        Elementwise.copy(src.broadcastTo(layout.shape), this);
    }

    // ================================================================== internals

    private void requireRank(int r) {
        if (ndim() != r) throw new IllegalArgumentException("expected rank " + r + ", array is " + shapeString());
    }
}
