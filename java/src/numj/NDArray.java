package numj;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Element-type-independent part of a strided n-dimensional array in native memory: shape, byte strides,
 * byte offset, flags, ownership and lifetime. {@link F64Array} is the only implementation so far; later element
 * types (see {@link DType}) are added as further permitted subclasses.
 *
 * <h2>Layout</h2>
 * Element {@code [i0, ..., ik]} lives at byte {@code byteOffset() + sum(i_j * strides()[j])} of
 * {@link #baseSegment()}. Strides are in bytes, may be negative (reversed views) or zero (broadcast views), and
 * are always multiples of the item size. Zero-dimensional arrays have shape {@code []} and one element.
 *
 * <h2>Ownership and lifetime</h2>
 * <ul>
 *   <li><b>Owner</b> ({@link #isOwner()}): created by an allocating factory or operation without an explicit
 *       arena. It owns a private confined or shared {@link Arena}; {@link #close()} frees the memory at once.
 *       Every operation that returns a new array without an {@code out} argument returns an owner, which the
 *       caller must close (try-with-resources).</li>
 *   <li><b>Arena-managed</b>: allocated in a caller-supplied {@link Arena}; freed when that arena is closed;
 *       {@link #close()} is a no-op.</li>
 *   <li><b>View</b>: shares memory with another array or a caller-managed segment; never frees anything;
 *       {@link #close()} is a no-op. A view is valid exactly as long as the memory it points to.</li>
 * </ul>
 * Using an array (or any view of it) after its memory is freed throws {@link IllegalStateException}. The FFM runtime
 * keeps memory alive for the duration of every native call, so native code never sees freed memory.
 *
 * <h2>Threads</h2>
 * Confined arrays (the default) may only be used by the thread that created them; other threads get
 * {@link WrongThreadException}, both from element access and from native kernels. Shared arrays
 * ({@code allocateShared}) may be used from any thread; the caller must not close them while another thread
 * is using them, and concurrent writes to the same elements are a data race (no locking is done).
 *
 * <h2>Writability</h2>
 * Arrays are writable unless created read-only ({@code asReadOnly()}, {@code broadcastTo()}, wrapping a
 * read-only segment, or a caller-supplied layout that might address the same element twice). Writing to a
 * read-only array, or passing it as an output, throws {@link ReadOnlyArrayException}. Every writable array is
 * guaranteed to be free of internal overlap (no two indices address the same bytes), so writes are never
 * ambiguous.
 */
public abstract sealed class NDArray implements AutoCloseable permits F64Array {
    final Arena arena;          // non-null only for owners
    final MemorySegment base;   // the whole underlying buffer (read-only segment when !writable)
    final Layout layout;
    final boolean writable;
    final boolean cContig;
    final long addrLo, addrHi;  // absolute address range spanned by the elements (equal when empty)
    final MemorySegment contig; // exactly the elements, for C-contiguous arrays (null otherwise)

    NDArray(Arena arena, MemorySegment base, Layout layout, boolean writable) {
        this.arena = arena;
        this.base = writable ? base : base.asReadOnly();
        this.layout = layout;
        this.writable = writable;
        this.cContig = layout.isC();
        long lo = layout.spanLo(), hi = layout.spanHi();
        this.addrLo = base.address() + lo;
        this.addrHi = base.address() + hi;
        this.contig = !cContig ? null
                : layout.size == 0 ? this.base.asSlice(0, 0) : this.base.asSlice(layout.offset, hi - lo);
    }

    /** Element type. */
    public abstract DType dtype();

    // ------------------------------------------------------------------ shape

    /** Number of dimensions (0 for a scalar array). */
    public int ndim() { return layout.ndim(); }

    /** Same as {@link #ndim()} (kept from numj 0.1). */
    public int rank() { return layout.ndim(); }

    /** Copy of the shape. */
    public long[] shape() { return layout.shape.clone(); }

    /** Extent of one axis ({@code axis} may be negative). */
    public long shape(int axis) { return layout.shape[Layout.normalizeAxis(axis, layout.ndim())]; }

    /** Copy of the byte strides. */
    public long[] strides() { return layout.strides.clone(); }

    /** Byte offset of element {@code [0, ..., 0]} within {@link #baseSegment()}. */
    public long byteOffset() { return layout.offset; }

    /** Total number of elements (1 for a scalar array, 0 if any extent is 0). */
    public long size() { return layout.size; }

    /** True if both arrays have the same shape. */
    public boolean sameShape(NDArray o) { return java.util.Arrays.equals(layout.shape, o.layout.shape); }

    /** Shape as {@code "[2, 3]"} ({@code "[]"} for a scalar array). */
    public String shapeString() { return Layout.shapeString(layout.shape); }

    // ------------------------------------------------------------------ flags

    /** Row-major contiguous (NumPy {@code C_CONTIGUOUS}: axes of length 1 are ignored, empty arrays qualify). */
    public boolean isCContiguous() { return cContig; }

    /** Column-major contiguous (NumPy {@code F_CONTIGUOUS}). */
    public boolean isFContiguous() { return layout.isF(); }

    public boolean isWritable() { return writable; }

    /** True if this array owns (and {@link #close()} frees) its memory. */
    public boolean isOwner() { return arena != null; }

    /** False once the underlying memory has been freed. */
    public boolean isAlive() { return base.scope().isAlive(); }

    // ------------------------------------------------------------------ memory

    /**
     * The bytes spanned by this array's elements, from the lowest to the highest addressed element (empty for
     * an empty array). For a C-contiguous array these are exactly its elements in row-major order. The segment
     * is read-only if the array is. It follows the same lifetime rules as the array.
     */
    public MemorySegment segment() {
        if (contig != null) return contig;
        long lo = layout.spanLo();
        return base.asSlice(lo, layout.spanHi() - lo);
    }

    /** The whole underlying buffer that {@link #byteOffset()} and {@link #strides()} refer to. */
    public MemorySegment baseSegment() { return base; }

    /**
     * Conservative overlap test (like NumPy's {@code may_share_memory}): true if the address ranges spanned by
     * the two arrays intersect. Interleaved views such as {@code x[0::2]} and {@code x[1::2]} report true.
     */
    public boolean overlaps(NDArray other) {
        if (layout.size == 0 || other.layout.size == 0) return false;
        return addrLo < other.addrHi && other.addrLo < addrHi;
    }

    /** True if both arrays span exactly the same bytes. */
    public boolean sameMemory(NDArray other) {
        return base.address() + layout.spanLo() == other.base.address() + other.layout.spanLo()
                && layout.spanHi() - layout.spanLo() == other.layout.spanHi() - other.layout.spanLo();
    }

    /** True if both arrays address exactly the same elements in the same index order. */
    boolean sameLayout(NDArray other) {
        if (!sameShape(other)) return false;
        if (base.address() + layout.offset != other.base.address() + other.layout.offset) return false;
        for (int i = 0; i < layout.ndim(); i++)
            if (layout.shape[i] > 1 && layout.strides[i] != other.layout.strides[i]) return false;
        return true;
    }

    /** Frees the memory if this is an owner (idempotent); no-op for views and arena-managed arrays. */
    @Override
    public void close() {
        if (arena != null && arena.scope().isAlive()) arena.close();
    }

    void requireAlive() {
        if (!base.scope().isAlive()) throw new IllegalStateException("array memory has been freed");
    }

    void requireWritable(String what) {
        if (!writable) throw new ReadOnlyArrayException(what + " is read-only");
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + shapeString() + (isOwner() ? "" : " (view)") + (writable ? "" : " (read-only)")
                + (isAlive() ? "" : " (freed)");
    }
}
