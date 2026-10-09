package numj;

import numj.bench.Data;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

/**
 * Tiny inputs: the Java loop path versus a (regular) native downcall, to choose the crossover thresholds
 * ({@code Elementwise.javaMaxElements}, {@code Reduce.javaMaxElements}, {@code NumJ.fusedJavaMaxElements}).
 * Both paths produce identical bits; only the cost differs.
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 4, time = 150, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 5, time = 150, timeUnit = TimeUnit.MILLISECONDS)
@Fork(value = 2, jvmArgsAppend = {"--enable-native-access=ALL-UNNAMED", "-Xms1g", "-Xmx1g"})
public class SmallBench {
    @Param({"1", "4", "16", "32", "64", "128", "256", "1024"})
    public int n;
    @Param({"java", "native"})
    public String path;

    F64Array a, b, out, bt, at;

    @Setup
    public void setup() {
        long t = path.equals("java") ? Long.MAX_VALUE : -1;
        Elementwise.javaMaxElements = t;
        Elementwise.contigJavaMaxElements = t;
        Reduce.javaMaxElements = t;
        NumJ.fusedJavaMaxElements = t;
        a = F64Array.of(Data.splitmix(1, n));
        b = F64Array.of(Data.splitmix(2, n));
        out = F64Array.allocate(n);
        // a short strided case: column of a [n, 2] array
        at = F64Array.copyOf(Data.splitmix(3, 2 * n), n, 2);
        bt = at.slice(":, 1");
    }

    @TearDown
    public void tearDown() {
        a.close();
        b.close();
        out.close();
        at.close();
    }

    @Benchmark
    public F64Array add() {
        return NumJ.add(a, b, out);
    }

    @Benchmark
    public F64Array addStrided() {
        return NumJ.add(bt, b, out);
    }

    @Benchmark
    public double sum() {
        return NumJ.sum(a);
    }

    @Benchmark
    public double sqdist() {
        return NumJ.sqdist(a, b);
    }
}
