package numj;

import numj.bench.Data;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

/**
 * Native OpenMP scaling, to set per-operation thresholds. The parallel-size thresholds are disabled here
 * (every call uses {@code threads} threads) so the raw scaling is visible. Contiguous {@code add} with output reuse,
 * contiguous full {@code sum}, and column sums ({@code sum axis 0}) of a [n/1000, 1000] array.
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 4, time = 200, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 5, time = 200, timeUnit = TimeUnit.MILLISECONDS)
@Fork(value = 2, jvmArgsAppend = {"--enable-native-access=ALL-UNNAMED", "-Xms2g", "-Xmx2g", "-XX:+AlwaysPreTouch",
    "-Dnumj.ew.parallelMinElements=0", "-Dnumj.reduce.parallelMinElements=0"})
public class ThreadsBench {
    @Param({"100000", "1000000", "4000000", "16000000"})
    public int n;
    @Param({"1", "2", "4", "8"})
    public int threads;

    F64Array a, b, out, m, cols;
    static final int[] AX0 = {0};

    @Setup
    public void setup() {
        a = F64Array.of(Data.splitmix(1, n));
        b = F64Array.of(Data.splitmix(2, n));
        out = F64Array.allocate(n);
        m = a.reshape(n / 1000, 1000);
        cols = F64Array.allocate(1000);
        NumJ.setThreads(threads);
    }

    @TearDown
    public void tearDown() {
        NumJ.setThreads(1);
        a.close();
        b.close();
        out.close();
        cols.close();
    }

    @Benchmark
    public F64Array add() {
        return NumJ.add(a, b, out);
    }

    @Benchmark
    public double sum() {
        return NumJ.sum(a);
    }

    @Benchmark
    public F64Array sumAxis0() {
        return NumJ.sum(m, AX0, false, cols);
    }
}
