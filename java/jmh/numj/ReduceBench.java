package numj;

import numj.bench.Data;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

/**
 * Sums of a [r, c] C-ordered array (same shapes and input bits as bench/numpy_nd_bench.py), single-threaded.
 * {@code all}: every element; {@code axis0}: column sums [c]; {@code axis1}: row sums [r];
 * {@code allT}: every element of the transposed view (numj sums in the view's logical order, so this measures the
 * cost of layout-independent results). Outputs are reused.
 * {@code java}: straightforward loops in memory order (sequential accumulation: different rounding, see report).
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 200, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 5, time = 200, timeUnit = TimeUnit.MILLISECONDS)
@Fork(value = 3, jvmArgsAppend = {"--enable-native-access=ALL-UNNAMED", "-Xms2g", "-Xmx2g", "-XX:+AlwaysPreTouch"})
public class ReduceBench {
    @Param({"1000", "100000", "10000000"})
    public int n;
    @Param({"all", "axis0", "axis1", "allT"})
    public String kind;

    F64Array A, out0, out1;
    double[] h, h0, h1;
    int r, c;
    static final int[] AX0 = {0}, AX1 = {1};

    @Setup
    public void setup() {
        int[] d = ElementwiseBench.dims(n);
        r = d[0];
        c = d[1];
        h = Data.splitmix(1, n);
        A = F64Array.copyOf(h, r, c);
        out0 = F64Array.allocate(c);
        out1 = F64Array.allocate(r);
        h0 = new double[c];
        h1 = new double[r];
    }

    @TearDown
    public void tearDown() {
        A.close();
        out0.close();
        out1.close();
    }

    @Benchmark
    public Object numj() {
        return switch (kind) {
            case "all" -> NumJ.sum(A);
            case "axis0" -> NumJ.sum(A, AX0, false, out0);
            case "axis1" -> NumJ.sum(A, AX1, false, out1);
            default -> NumJ.sum(A.transpose());
        };
    }

    @Benchmark
    public Object java() {
        double[] x = h;
        int rr = r, cc = c;
        switch (kind) {
            case "axis0" -> {
                double[] o = h0;
                java.util.Arrays.fill(o, 0.0);
                for (int i = 0; i < rr; i++)
                    for (int j = 0; j < cc; j++) o[j] += x[i * cc + j];
                return o;
            }
            case "axis1" -> {
                double[] o = h1;
                for (int i = 0; i < rr; i++) {
                    double s = 0;
                    for (int j = 0; j < cc; j++) s += x[i * cc + j];
                    o[i] = s;
                }
                return o;
            }
            default -> {
                double s = 0;
                for (double v : x) s += v;
                return s;
            }
        }
    }
}
