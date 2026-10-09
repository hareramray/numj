package numj;

import numj.bench.Data;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

/**
 * Short rows, 10^6 elements in rows of k: {@code sumRows} sums each row of a C-ordered [10^6/k, k] array into a
 * reused output; {@code addRows} adds two [R, k] views that cannot be merged into one run (they are the first k
 * columns of [R, k+1] arrays), so the native kernel walks R rows of k elements. {@code java*} are plain loops over
 * the same layouts. Same shapes and data as the "short" suite of bench/numpy_nd_bench.py.
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 200, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 5, time = 200, timeUnit = TimeUnit.MILLISECONDS)
@Fork(value = 3, jvmArgsAppend = {"--enable-native-access=ALL-UNNAMED", "-Xms2g", "-Xmx2g", "-XX:+AlwaysPreTouch"})
public class ShortRowBench {
    static final int N = 1_000_000;
    static final int[] AX1 = {1};

    @Param({"2", "4", "8", "16", "64"})
    public int k;

    F64Array M, sums, X, Y, xv, yv, out;
    double[] hm, hs, hx, hy, ho;
    int rows;

    @Setup
    public void setup() {
        rows = N / k;
        hm = Data.splitmix(1, N);
        M = F64Array.copyOf(hm, rows, k);
        sums = F64Array.allocate(rows);
        hs = new double[rows];
        hx = Data.splitmix(2, rows * (k + 1));
        hy = Data.splitmix(3, rows * (k + 1));
        X = F64Array.copyOf(hx, rows, k + 1);
        Y = F64Array.copyOf(hy, rows, k + 1);
        xv = X.slice(Ix.all(), Ix.to(k));
        yv = Y.slice(Ix.all(), Ix.to(k));
        out = F64Array.allocate(rows, k);
        ho = new double[rows * k];
    }

    @TearDown
    public void tearDown() {
        M.close();
        sums.close();
        X.close();
        Y.close();
        out.close();
    }

    @Benchmark
    public F64Array sumRows() {
        return NumJ.sum(M, AX1, false, sums);
    }

    @Benchmark
    public double[] javaSumRows() {
        double[] x = hm, o = hs;
        int kk = k;
        for (int r = 0; r < o.length; r++) {
            double s = 0;
            for (int j = 0; j < kk; j++) s += x[r * kk + j];
            o[r] = s;
        }
        return o;
    }

    @Benchmark
    public F64Array addRows() {
        return NumJ.add(xv, yv, out);
    }

    @Benchmark
    public double[] javaAddRows() {
        double[] x = hx, y = hy, o = ho;
        int kk = k, ld = k + 1;
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < kk; j++) o[r * kk + j] = x[r * ld + j] + y[r * ld + j];
        return o;
    }
}
