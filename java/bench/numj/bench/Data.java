package numj.bench;

/**
 * Deterministic benchmark inputs, bit-identical to bench/numpy_bench.py's {@code splitmix()}:
 * element i (0-based) = 2 * ((splitmix64(seed + (i+1) * 0x9E3779B97F4A7C15) >>> 11) * 2^-53) - 1.
 */
public final class Data {
    private Data() {}

    public static double[] splitmix(long seed, int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            long z = seed + (i + 1L) * 0x9E3779B97F4A7C15L;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            z = z ^ (z >>> 31);
            out[i] = ((z >>> 11) * 0x1p-53) * 2.0 - 1.0;
        }
        return out;
    }

    /** Order-sensitive fingerprint used to check that Java and Python generated identical data. */
    public static long fingerprint(double[] x) {
        long h = 0xcbf29ce484222325L;
        for (double v : x) h = (h ^ Double.doubleToRawLongBits(v)) * 0x100000001b3L;
        return h;
    }
}
