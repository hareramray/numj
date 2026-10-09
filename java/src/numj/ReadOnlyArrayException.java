package numj;

/**
 * Thrown when writing to a read-only array or passing one as an output (NumPy raises
 * {@code ValueError: assignment destination is read-only}).
 */
public final class ReadOnlyArrayException extends UnsupportedOperationException {
    private static final long serialVersionUID = 1L;

    public ReadOnlyArrayException(String message) {
        super(message);
    }
}
