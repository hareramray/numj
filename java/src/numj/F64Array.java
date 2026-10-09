package numj;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * A contiguous, row-major (C order) array of IEEE-754 {@code float64} values in native (off-heap) memory.
 *
 * <h2>Shape and layout</h2>
 * Rank 1 ({@code [n]}) or rank 2 ({@code [rows, cols]}). Element {@code (r, c)} lives at flat index
 * {@code r * cols + c}; rows are contiguous. There are no strides: every array is dense, which is what
 * lets the native kernels run without copies.
 *
 * <h2>Ownership and lifetime</h2>
 * <ul>
 *   <li>An <b>owner</b> is created by {@link #allocate}, {@link #allocateShared}, {@link #of} or
 *       {@link #copyOf}. It owns a private {@link Arena}; {@link #close()} frees the memory immediately
 *       and deterministically. Use try-with-resources.</li>
 *   <li>A <b>view</b> is created by {@link #row}, {@link #reshape} or {@link #wrap}. It shares memory with
 *       its parent (or with a caller-managed arena) and never frees anything; {@link #close()} on a view is a
 *       no-op. A view becomes unusable as soon as the underlying memory is freed.</li>
 *   <li>Any access after the memory is freed throws {@link IllegalStateException}; native kernels are never
 *       handed freed memory (the FFM runtime keeps the arena alive for the duration of each native call).</li>
 * </ul>
 *
 * <h2>Threads</h2>
 * {@link #allocate} uses a confined arena: only the creating thread may read, write, pass to kernels, or
 * close it (other threads get {@link WrongThreadException}). {@link #allocateShared} may be used from any
 * thread; the caller is responsible for not closing it while another thread is using it.
 *
 * <h2>Bounds</h2>
 * Every element accessor checks indices and throws {@link IndexOutOfBoundsException}. Native memory is
 * 64-byte aligned and zero-initialised on allocation.
 */
public final class F64Array implements AutoCloseable {
    static final long ALIGNMENT = 64;
    private static final ValueLayout.OfDouble F64 = ValueLayout.JAVA_DOUBLE;

    private final Arena arena;          // non-null only for owners
    private final MemorySegment seg;
    private final int rank;
    private final long rows, cols;      // rank 1: rows == 1, cols == n

    private F64Array(Arena arena, MemorySegment seg, int rank, long rows, long cols) {
        this.arena = arena;
        this.seg = seg;
        this.rank = rank;
        this.rows = rows;
        this.cols = cols;
    }

    // ---------------------------------------------------------------- creation

    /** New zero-filled rank-1 array of length {@code n}, confined to the calling thread. */
    public static F64Array allocate(long n) {
        return owner(Arena.ofConfined(), 1, 1, checkDim(n, "n"));
    }

    /** New zero-filled rank-2 array {@code [rows, cols]}, confined to the calling thread. */
    public static F64Array allocate(long rows, long cols) {
        return owner(Arena.ofConfined(), 2, checkDim(rows, "rows"), checkDim(cols, "cols"));
    }

    /** Like {@link #allocate(long, long)} but usable from any thread. */
    public static F64Array allocateShared(long rows, long cols) {
        return owner(Arena.ofShared(), 2, checkDim(rows, "rows"), checkDim(cols, "cols"));
    }

    /** Like {@link #allocate(long)} but usable from any thread. */
    public static F64Array allocateShared(long n) {
        return owner(Arena.ofShared(), 1, 1, checkDim(n, "n"));
    }

    /** New confined rank-1 array holding a copy of {@code values}. */
    public static F64Array of(double... values) {
        F64Array a = allocate(values.length);
        a.copyFrom(values);
        return a;
    }

    /** New confined rank-2 array holding a copy of {@code data}, interpreted row-major. */
    public static F64Array copyOf(double[] data, long rows, long cols) {
        F64Array a = allocate(rows, cols);
        a.copyFrom(data);
        return a;
    }

    /**
     * A view over memory the caller manages (for interop with other FFM code). The segment must be native,
     * 8-byte aligned and exactly {@code rows * cols * 8} bytes. The view never frees the memory.
     */
    public static F64Array wrap(MemorySegment segment, long rows, long cols) {
        Objects.requireNonNull(segment, "segment");
        long r = checkDim(rows, "rows"), c = checkDim(cols, "cols");
        if (!segment.isNative()) throw new IllegalArgumentException("segment must be native (off-heap)");
        if (segment.address() % Double.BYTES != 0) throw new IllegalArgumentException("segment must be 8-byte aligned");
        if (segment.byteSize() != bytes(r, c))
            throw new IllegalArgumentException("segment is " + segment.byteSize() + " bytes, shape needs " + bytes(r, c));
        return new F64Array(null, segment, 2, r, c);
    }

    private static F64Array owner(Arena arena, int rank, long rows, long cols) {
        try {
            MemorySegment s = arena.allocate(bytes(rows, cols), ALIGNMENT);
            return new F64Array(arena, s, rank, rows, cols);
        } catch (RuntimeException | Error e) {
            arena.close();
            throw e;
        }
    }

    private static long checkDim(long d, String name) {
        if (d < 0) throw new IllegalArgumentException(name + " must be >= 0, got " + d);
        return d;
    }

    private static long bytes(long rows, long cols) {
        return Math.multiplyExact(Math.multiplyExact(rows, cols), (long) Double.BYTES);
    }

    // ---------------------------------------------------------------- shape

    public int rank() { return rank; }
    /** Total number of elements. */
    public long size() { return rows * cols; }
    /** Rows of a rank-2 array (1 for rank 1). */
    public long rows() { return rows; }
    /** Columns of a rank-2 array, or the length of a rank-1 array. */
    public long cols() { return cols; }
    public long[] shape() { return rank == 1 ? new long[] {cols} : new long[] {rows, cols}; }

    /** True if both arrays have the same rank and extents. */
    public boolean sameShape(F64Array o) {
        return rank == o.rank && rows == o.rows && cols == o.cols;
    }

    public String shapeString() {
        return rank == 1 ? "[" + cols + "]" : "[" + rows + ", " + cols + "]";
    }

    // ---------------------------------------------------------------- views

    /** Rank-1 view of row {@code r} of a rank-2 array (shares memory). */
    public F64Array row(long r) {
        requireRank(2);
        Objects.checkIndex(r, rows);
        return new F64Array(null, seg.asSlice(r * cols * Double.BYTES, cols * Double.BYTES), 1, 1, cols);
    }

    /** View with a different rank-2 shape and the same number of elements (shares memory). */
    public F64Array reshape(long newRows, long newCols) {
        if (Math.multiplyExact(checkDim(newRows, "rows"), checkDim(newCols, "cols")) != size())
            throw new IllegalArgumentException("cannot reshape " + shapeString() + " to [" + newRows + ", " + newCols + "]");
        return new F64Array(null, seg, 2, newRows, newCols);
    }

    /** Rank-1 view of all elements in row-major order (shares memory). */
    public F64Array flatten() {
        return new F64Array(null, seg, 1, 1, size());
    }

    // ---------------------------------------------------------------- element access (bounds-checked)

    /** Element at flat (row-major) index {@code i}. */
    public double get(long i) {
        return seg.getAtIndex(F64, Objects.checkIndex(i, size()));
    }

    public void set(long i, double v) {
        seg.setAtIndex(F64, Objects.checkIndex(i, size()), v);
    }

    /** Element {@code (r, c)} of a rank-2 array. */
    public double get(long r, long c) {
        requireRank(2);
        return seg.getAtIndex(F64, Objects.checkIndex(r, rows) * cols + Objects.checkIndex(c, cols));
    }

    public void set(long r, long c, double v) {
        requireRank(2);
        seg.setAtIndex(F64, Objects.checkIndex(r, rows) * cols + Objects.checkIndex(c, cols), v);
    }

    public void fill(double v) {
        if (v == 0.0 && Double.doubleToRawLongBits(v) == 0L) seg.fill((byte) 0);
        else for (long i = 0, n = size(); i < n; i++) seg.setAtIndex(F64, i, v);
    }

    // ---------------------------------------------------------------- bulk copies (one memcpy each)

    /** Copies all of {@code src} into this array; {@code src.length} must equal {@link #size()}. */
    public void copyFrom(double[] src) {
        if (src.length != size()) throw new IllegalArgumentException("source length " + src.length + " != size " + size());
        MemorySegment.copy(src, 0, seg, F64, 0, src.length);
    }

    /** Copies {@code len} elements from {@code src[srcPos..]} to flat index {@code dstIndex..}. */
    public void copyFrom(double[] src, int srcPos, long dstIndex, int len) {
        Objects.checkFromIndexSize(srcPos, len, src.length);
        Objects.checkFromIndexSize(dstIndex, len, size());
        MemorySegment.copy(src, srcPos, seg, F64, dstIndex * Double.BYTES, len);
    }

    /** Copies all elements into {@code dst}; {@code dst.length} must equal {@link #size()}. */
    public void copyTo(double[] dst) {
        if (dst.length != size()) throw new IllegalArgumentException("destination length " + dst.length + " != size " + size());
        MemorySegment.copy(seg, F64, 0, dst, 0, dst.length);
    }

    /** New heap array with a copy of the contents (size must fit in an int). */
    public double[] toArray() {
        return seg.toArray(F64);
    }

    // ---------------------------------------------------------------- lifetime & interop

    /** The backing native segment (shares memory; subject to the same lifetime rules). */
    public MemorySegment segment() { return seg; }

    /** True if this array owns (and {@link #close()} frees) its memory. */
    public boolean isOwner() { return arena != null; }

    /** False once the underlying memory has been freed. */
    public boolean isAlive() { return seg.scope().isAlive(); }

    /** True if any byte of this array is also part of {@code other}. */
    public boolean overlaps(F64Array other) {
        return seg.asOverlappingSlice(other.seg).isPresent();
    }

    /** True if both arrays cover exactly the same bytes. */
    public boolean sameMemory(F64Array other) {
        return seg.address() == other.seg.address() && seg.byteSize() == other.seg.byteSize();
    }

    /** Frees the memory if this is an owner (idempotent); no-op for views. */
    @Override
    public void close() {
        if (arena != null && arena.scope().isAlive()) arena.close();
    }

    void requireAlive() {
        if (!isAlive()) throw new IllegalStateException("F64Array memory has been freed");
    }

    private void requireRank(int r) {
        if (rank != r) throw new IllegalArgumentException("expected rank " + r + ", array is " + shapeString());
    }

    @Override
    public String toString() {
        return "F64Array" + shapeString() + (isOwner() ? "" : " (view)") + (isAlive() ? "" : " (freed)");
    }
}
