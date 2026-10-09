package numj;

import numj.bench.Data;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

/**
 * Regular versus critical downcalls for the fused {@code sqdist} kernel (the Java path is disabled so every call
 * goes native). {@link Critical} runs the same code in forks started with {@code -Dnumj.criticalMaxElements}.
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 4, time = 150, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 5, time = 150, timeUnit = TimeUnit.MILLISECONDS)
@Fork(value = 2, jvmArgsAppend = {"--enable-native-access=ALL-UNNAMED", "-Xms1g", "-Xmx1g"})
public class DowncallBench {
    @Param({"0", "16", "1000", "65536"})
    public int n;

    F64Array a, b;

    @Setup
    public void setup() {
        NumJ.fusedJavaMaxElements = -1;
        a = F64Array.of(Data.splitmix(1, n));
        b = F64Array.of(Data.splitmix(2, n));
    }

    @TearDown
    public void tearDown() {
        a.close();
        b.close();
    }

    @Benchmark
    public double regular() {
        return NumJ.sqdist(a, b);
    }

    @State(Scope.Thread)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    @Warmup(iterations = 4, time = 150, timeUnit = TimeUnit.MILLISECONDS)
    @Measurement(iterations = 5, time = 150, timeUnit = TimeUnit.MILLISECONDS)
    // nested benchmark classes do not inherit the enclosing class's annotations
    @Fork(value = 2, jvmArgsAppend = {"--enable-native-access=ALL-UNNAMED", "-Xms1g", "-Xmx1g",
        "-Dnumj.criticalMaxElements=1000000"})
    public static class Critical {
        @Param({"0", "16", "1000", "65536"})
        public int n;

        F64Array a, b;

        @Setup
        public void setup() {
            NumJ.fusedJavaMaxElements = -1;
            a = F64Array.of(Data.splitmix(1, n));
            b = F64Array.of(Data.splitmix(2, n));
        }

        @TearDown
        public void tearDown() {
            a.close();
            b.close();
        }

        @Benchmark
        public double critical() {
            return NumJ.sqdist(a, b);
        }
    }
}
