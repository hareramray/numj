package numj;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/** Minimal shared test harness (no framework): named tests, assertions, failure summary. */
final class T {
    static final double U = 0x1p-53;
    static int passed, failed;
    static final List<String> failures = new ArrayList<>();

    private T() {}

    interface Body { void run() throws Exception; }

    static void test(String name, Body body) {
        try {
            body.run();
            passed++;
            System.out.println("PASS  " + name);
        } catch (Throwable t) {
            failed++;
            failures.add(name + ": " + t);
            System.out.println("FAIL  " + name + "\n      " + t);
            StackTraceElement[] st = t.getStackTrace();
            for (int i = 0; i < Math.min(6, st.length); i++) System.out.println("        at " + st[i]);
        }
    }

    static int finish() {
        System.out.printf("%n%d passed, %d failed%n", passed, failed);
        failures.forEach(f -> System.out.println("  FAILED: " + f));
        return failed == 0 ? 0 : 1;
    }

    static void check(String what, boolean ok) {
        if (!ok) throw new AssertionError(what);
    }

    static void assertBits(String what, double expected, double got) {
        if (Double.doubleToRawLongBits(expected) != Double.doubleToRawLongBits(got)
                && !(Double.isNaN(expected) && Double.isNaN(got)))
            throw new AssertionError(what + ": expected " + expected + " got " + got);
    }

    static void assertShape(String what, long[] expected, long[] got) {
        if (!java.util.Arrays.equals(expected, got))
            throw new AssertionError(what + ": expected shape " + java.util.Arrays.toString(expected) + " got "
                    + java.util.Arrays.toString(got));
    }

    static <X extends Throwable> X throwsType(String what, Class<X> type, Runnable r) {
        try {
            r.run();
        } catch (Throwable t) {
            if (type.isInstance(t)) return type.cast(t);
            throw new AssertionError(what + ": expected " + type.getSimpleName() + " got " + t, t);
        }
        throw new AssertionError(what + ": expected " + type.getSimpleName() + ", nothing thrown");
    }

    static void throwsIAE(String what, Runnable r) {
        throwsType(what, IllegalArgumentException.class, r);
    }

    /** Visits every multi-index of {@code shape} in C order (the array passed to {@code f} is reused). */
    static void forEach(long[] shape, Consumer<long[]> f) {
        for (long d : shape) if (d == 0) return;
        long[] idx = new long[shape.length];
        while (true) {
            f.accept(idx);
            int d = shape.length - 1;
            while (d >= 0) {
                if (++idx[d] < shape[d]) break;
                idx[d] = 0;
                d--;
            }
            if (d < 0) return;
        }
    }

    static long size(long[] shape) {
        long n = 1;
        for (long d : shape) n *= d;
        return n;
    }

    /** Values in C order, read element by element through the multi-index accessor. */
    static double[] values(F64Array a) {
        double[] v = new double[(int) a.size()];
        int[] k = {0};
        forEach(a.shape(), idx -> v[k[0]++] = a.get(idx.clone()));
        return v;
    }

    /** Index into an input of shape {@code in} for output index {@code out} under broadcasting. */
    static long[] bcastIndex(long[] out, long[] in) {
        long[] r = new long[in.length];
        int d = out.length - in.length;
        for (int i = 0; i < in.length; i++) r[i] = in[i] == 1 ? 0 : out[d + i];
        return r;
    }

    static long flat(long[] idx, long[] shape) {
        long f = 0;
        for (int i = 0; i < shape.length; i++) f = f * shape[i] + idx[i];
        return f;
    }

    static double gamma(long k) {
        return k * U / (1 - k * U);
    }
}
