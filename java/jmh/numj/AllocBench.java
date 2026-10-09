package numj;

import numj.bench.Data;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.TimeUnit;

/**
 * Allocation costs. {@code allocClose}: a zero-filled owner (confined arena) allocated and freed;
 * {@code addAlloc}: {@code NumJ.add(a, b)} with a fresh result; {@code addReuse}: the same into a reused buffer.
 * {@link Malloc} repeats {@code addAlloc} with results taken from non-zeroing {@code malloc}
 * ({@code -Dnumj.resultAlloc=malloc}), the candidate improvement being evaluated.
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 200, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 5, time = 200, timeUnit = TimeUnit.MILLISECONDS)
@Fork(value = 3, jvmArgsAppend = {"--enable-native-access=ALL-UNNAMED", "-Xms1g", "-Xmx1g"})
public class AllocBench {
    @Param({"16", "1000", "100000", "10000000"})
    public int n;

    F64Array a, b, out;

    @Setup
    public void setup() {
        a = F64Array.of(Data.splitmix(1, n));
        b = F64Array.of(Data.splitmix(2, n));
        out = F64Array.allocate(n);
    }

    @TearDown
    public void tearDown() {
        a.close();
        b.close();
        out.close();
    }

    @Benchmark
    public void allocClose(Blackhole bh) {
        try (F64Array x = F64Array.allocate(n)) {
            bh.consume(x);
        }
    }

    @Benchmark
    public void addAlloc(Blackhole bh) {
        try (F64Array r = NumJ.add(a, b)) {
            bh.consume(r);
        }
    }

    @Benchmark
    public F64Array addReuse() {
        return NumJ.add(a, b, out);
    }

    @State(Scope.Thread)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    @Warmup(iterations = 5, time = 200, timeUnit = TimeUnit.MILLISECONDS)
    @Measurement(iterations = 5, time = 200, timeUnit = TimeUnit.MILLISECONDS)
    // nested benchmark classes do not inherit the enclosing class's annotations
    @Fork(value = 3, jvmArgsAppend = {"--enable-native-access=ALL-UNNAMED", "-Xms1g", "-Xmx1g", "-Dnumj.resultAlloc=malloc"})
    public static class Malloc {
        @Param({"16", "1000", "100000", "10000000"})
        public int n;

        F64Array a, b;

        @Setup
        public void setup() {
            a = F64Array.of(Data.splitmix(1, n));
            b = F64Array.of(Data.splitmix(2, n));
        }

        @TearDown
        public void tearDown() {
            a.close();
            b.close();
        }

        @Benchmark
        public void addAllocMalloc(Blackhole bh) {
            try (F64Array r = NumJ.add(a, b)) {
                bh.consume(r);
            }
        }
    }
}
