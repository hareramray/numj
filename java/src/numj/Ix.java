package numj;

import java.util.ArrayList;
import java.util.List;

/**
 * One item of a basic (view-producing) index, the Java counterpart of NumPy's {@code a[i, start:stop:step, None, ...]}.
 * Used with {@link F64Array#slice(Ix...)}.
 *
 * <ul>
 *   <li>{@link #at(long)}: integer index; removes the axis. Negative values count from the end.
 *       Out of range throws {@link IndexOutOfBoundsException}.</li>
 *   <li>{@link #range}, {@link #from}, {@link #to}, {@link #step}, {@link #all()}: slices with Python semantics.
 *       Bounds are clamped (never an error); a step of 0 throws {@link IllegalArgumentException};
 *       a negative step walks backwards (e.g. {@code Ix.step(-1)} reverses an axis).</li>
 *   <li>{@link #newAxis()}: inserts an axis of length 1 (NumPy {@code None}).</li>
 *   <li>{@link #ellipsis()}: expands to as many {@link #all()} as needed (at most one per index).</li>
 * </ul>
 * Axes not covered by the index are kept whole, as in NumPy.
 */
public final class Ix {
    enum Kind { INDEX, SLICE, NEW_AXIS, ELLIPSIS }

    final Kind kind;
    final long index;                 // INDEX
    final boolean hasStart, hasStop;  // SLICE
    final long start, stop, step;     // SLICE

    private Ix(Kind kind, long index, boolean hasStart, long start, boolean hasStop, long stop, long step) {
        this.kind = kind;
        this.index = index;
        this.hasStart = hasStart;
        this.start = start;
        this.hasStop = hasStop;
        this.stop = stop;
        this.step = step;
    }

    private static final Ix ALL = new Ix(Kind.SLICE, 0, false, 0, false, 0, 1);
    private static final Ix NEW_AXIS = new Ix(Kind.NEW_AXIS, 0, false, 0, false, 0, 1);
    private static final Ix ELLIPSIS = new Ix(Kind.ELLIPSIS, 0, false, 0, false, 0, 1);

    /** Integer index {@code i} (axis removed). */
    public static Ix at(long i) { return new Ix(Kind.INDEX, i, false, 0, false, 0, 1); }

    /** {@code :} */
    public static Ix all() { return ALL; }

    /** {@code start:stop} */
    public static Ix range(long start, long stop) { return new Ix(Kind.SLICE, 0, true, start, true, stop, 1); }

    /** {@code start:stop:step} */
    public static Ix range(long start, long stop, long step) { return new Ix(Kind.SLICE, 0, true, start, true, stop, step); }

    /** {@code start:} */
    public static Ix from(long start) { return new Ix(Kind.SLICE, 0, true, start, false, 0, 1); }

    /** {@code :stop} */
    public static Ix to(long stop) { return new Ix(Kind.SLICE, 0, false, 0, true, stop, 1); }

    /** {@code ::step} ({@code step(-1)} reverses the axis). */
    public static Ix step(long step) { return new Ix(Kind.SLICE, 0, false, 0, false, 0, step); }

    /** General slice; {@code null} means "omitted" exactly as in Python ({@code step} null means 1). */
    public static Ix slice(Long start, Long stop, Long step) {
        return new Ix(Kind.SLICE, 0, start != null, start == null ? 0 : start, stop != null, stop == null ? 0 : stop,
                step == null ? 1 : step);
    }

    /** {@code None} / {@code np.newaxis}. */
    public static Ix newAxis() { return NEW_AXIS; }

    /** {@code ...} */
    public static Ix ellipsis() { return ELLIPSIS; }

    /**
     * Parses a NumPy basic-index expression such as {@code "::-1, 2, None, ..., 1:5:2"} (whitespace ignored).
     * Supports integers, slices, {@code None}/{@code newaxis} and {@code ...}; an empty string is the empty index.
     */
    public static Ix[] parse(String expr) {
        String s = expr.strip();
        if (s.isEmpty()) return new Ix[0];
        List<Ix> out = new ArrayList<>();
        for (String raw : s.split(",", -1)) {
            String t = raw.strip();
            if (t.isEmpty()) throw new IllegalArgumentException("empty index item in \"" + expr + "\"");
            if (t.equals("...")) out.add(ELLIPSIS);
            else if (t.equals("None") || t.equals("newaxis")) out.add(NEW_AXIS);
            else if (t.indexOf(':') < 0) out.add(at(parseLong(t, expr)));
            else {
                String[] p = t.split(":", -1);
                if (p.length > 3) throw new IllegalArgumentException("bad slice \"" + t + "\"");
                Long a = p[0].isBlank() ? null : parseLong(p[0], expr);
                Long b = p[1].isBlank() ? null : parseLong(p[1], expr);
                Long c = p.length < 3 || p[2].isBlank() ? null : parseLong(p[2], expr);
                out.add(slice(a, b, c));
            }
        }
        return out.toArray(new Ix[0]);
    }

    private static long parseLong(String t, String expr) {
        try {
            return Long.parseLong(t.strip());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("bad index \"" + t.strip() + "\" in \"" + expr + "\"");
        }
    }

    @Override
    public String toString() {
        return switch (kind) {
            case INDEX -> Long.toString(index);
            case NEW_AXIS -> "None";
            case ELLIPSIS -> "...";
            case SLICE -> (hasStart ? Long.toString(start) : "") + ":" + (hasStop ? Long.toString(stop) : "")
                    + (step != 1 ? ":" + step : "");
        };
    }
}
