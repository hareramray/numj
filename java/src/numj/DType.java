package numj;

/**
 * Element type of an {@link NDArray}.
 *
 * <p>Only {@link #FLOAT64} is implemented. The type exists so that shape, stride and lifetime logic
 * ({@code Layout}, {@link NDArray}) is written once in terms of {@link #itemSize()}, and later element types
 * (float32, signed/unsigned integers, bool, complex64/complex128) can be added as new constants with their own
 * typed {@code NDArray} subclass, kernels and promotion rules. See {@code docs/ROADMAP.md}.
 */
public enum DType {
    /** IEEE-754 binary64, little-endian; NumPy {@code float64} / {@code '<f8'}. */
    FLOAT64("float64", 8, 'f');

    private final String numpyName;
    private final int itemSize;
    private final char kind;

    DType(String numpyName, int itemSize, char kind) {
        this.numpyName = numpyName;
        this.itemSize = itemSize;
        this.kind = kind;
    }

    /** Bytes per element. */
    public int itemSize() { return itemSize; }

    /** NumPy's name for the equivalent dtype. */
    public String numpyName() { return numpyName; }

    /** NumPy kind character: {@code 'f'} floating, later {@code 'i'}, {@code 'u'}, {@code 'b'}, {@code 'c'}. */
    public char kind() { return kind; }
}
