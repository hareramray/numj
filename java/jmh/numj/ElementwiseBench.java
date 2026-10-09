package numj;

import numj.bench.Data;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.TimeUnit;

/**
 * Elementwise addition, single-threaded, same input bits as bench/numpy_nd_bench.py. Layouts:
 * <ul>
 *   <li>{@code contig}: {@code a + b}, both [n];</li>
 *   <li>{@code transposed}: {@code A.T + B.T} into a C-ordered [r, c] output (A, B are [c, r]);</li>
 *   <li>{@code stepped}: {@code a[::2] + b[::2]} into [n];</li>
 *   <li>{@code bcast}: {@code M + row}, M [r, c], row [c].</li>
 * </ul>
 * {@code numj}: output buffer reused; {@code numjAlloc}: fresh result array each call (allocation, zero-fill and
 * free included); {@code java}: straightforward loops over {@code double[]} with the same access pattern.
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 200, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 5, time = 200, timeUnit = TimeUnit.MILLISECONDS)
@Fork(value = 3, jvmArgsAppend = {"--enable-native-access=ALL-UNNAMED", "-Xms3g", "-Xmx3g", "-XX:+AlwaysPreTouch"})
public class ElementwiseBench {
    @Param({"16", "1000", "100000", "10000000"})
    public int n;
    @Param({"contig", "transposed", "stepped", "bcast"})
    public String layout;

    F64Array baseA, baseB, a, b, out;
    double[] ha, hb, ho;
    int r, c;

    static int[] dims(int n) {
        return switch (n) {
            case 16 -> new int[] {4, 4};
            case 1000 -> new int[] {25, 40};
            case 100000 -> new int[] {250, 400};
            default -> new int[] {2500, 4000};
        };
    }

    @Setup
    public void setup() {
        int[] d = dims(n);
        r = d[0];
        c = d[1];
        switch (layout) {
            case "contig" -> {
                ha = Data.splitmix(1, n);
                hb = Data.splitmix(2, n);
                baseA = F64Array.of(ha);
                baseB = F64Array.of(hb);
                a = baseA;
                b = baseB;
                out = F64Array.allocate(n);
            }
            case "transposed" -> {
                ha = Data.splitmix(1, n);                    // [c, r] row-major
                hb = Data.splitmix(2, n);
                baseA = F64Array.copyOf(ha, c, r);
                baseB = F64Array.copyOf(hb, c, r);
                a = baseA.transpose();
                b = baseB.transpose();
                out = F64Array.allocate(r, c);
            }
            case "stepped" -> {
                ha = Data.splitmix(1, 2 * n);
                hb = Data.splitmix(2, 2 * n);
                baseA = F64Array.of(ha);
                baseB = F64Array.of(hb);
                a = baseA.slice("::2");
                b = baseB.slice("::2");
                out = F64Array.allocate(n);
            }
            default -> {
                ha = Data.splitmix(1, n);
                hb = Data.splitmix(2, c);
                baseA = F64Array.copyOf(ha, r, c);
                baseB = F64Array.of(hb);
                a = baseA;
                b = baseB;
                out = F64Array.allocate(r, c);
            }
        }
        ho = new double[n];
    }

    @TearDown
    public void tearDown() {
        baseA.close();
        baseB.close();
        out.close();
    }

    @Benchmark
    public F64Array numj() {
        return NumJ.add(a, b, out);
    }

    @Benchmark
    public void numjAlloc(Blackhole bh) {
        try (F64Array res = NumJ.add(a, b)) {
            bh.consume(res);
        }
    }

    @Benchmark
    public double[] java() {
        double[] x = ha, y = hb, o = ho;
        switch (layout) {
            case "contig" -> {
                for (int i = 0; i < o.length; i++) o[i] = x[i] + y[i];
            }
            case "transposed" -> {
                int rr = r, cc = c;
                for (int i = 0; i < rr; i++)
                    for (int j = 0; j < cc; j++) o[i * cc + j] = x[j * rr + i] + y[j * rr + i];
            }
            case "stepped" -> {
                for (int i = 0; i < o.length; i++) o[i] = x[2 * i] + y[2 * i];
            }
            default -> {
                int rr = r, cc = c;
                for (int i = 0; i < rr; i++)
                    for (int j = 0; j < cc; j++) o[i * cc + j] = x[i * cc + j] + y[j];
            }
        }
        return o;
    }
}
