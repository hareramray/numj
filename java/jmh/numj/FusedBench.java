package numj;

import numj.bench.Data;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

/**
 * {@code sum((a*b + c)**2)}: the explicit fused pattern ({@link Expr} -> {@code sumSqMulAdd}, one pass), the same
 * expression evaluated operation by operation with fresh temporaries ({@code unfusedAlloc}) and with one reused
 * temporary ({@code unfusedReuse}), and a plain Java loop. {@code transposed} runs the numj variants on transposed
 * views of the same logical values (general strided path; same bits).
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 200, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 5, time = 200, timeUnit = TimeUnit.MILLISECONDS)
@Fork(value = 3, jvmArgsAppend = {"--enable-native-access=ALL-UNNAMED", "-Xms2g", "-Xmx2g", "-XX:+AlwaysPreTouch"})
public class FusedBench {
    @Param({"1000", "100000", "10000000"})
    public int n;
    @Param({"contig", "transposed"})
    public String layout;

    F64Array A, B, C, a, b, c, t;
    double[] ha, hb, hc;
    Expr.Reduction expr;

    @Setup
    public void setup() {
        int[] d = ElementwiseBench.dims(n);
        ha = Data.splitmix(1, n);
        hb = Data.splitmix(2, n);
        hc = Data.splitmix(3, n);
        A = F64Array.copyOf(ha, d[0], d[1]);
        B = F64Array.copyOf(hb, d[0], d[1]);
        C = F64Array.copyOf(hc, d[0], d[1]);
        boolean tr = layout.equals("transposed");
        a = tr ? A.transpose() : A;
        b = tr ? B.transpose() : B;
        c = tr ? C.transpose() : C;
        t = F64Array.allocate(a.shape());
        expr = Expr.sum(Expr.of(a).mul(Expr.of(b)).add(Expr.of(c)).square());
        if (!"sumSqMulAdd".equals(expr.fusedKernel())) throw new IllegalStateException("pattern not recognised");
    }

    @TearDown
    public void tearDown() {
        A.close();
        B.close();
        C.close();
        t.close();
    }

    @Benchmark
    public double fused() {
        return expr.evaluate();
    }

    @Benchmark
    public double unfusedAlloc() {
        return expr.evaluateUnfused();
    }

    @Benchmark
    public double unfusedReuse() {
        NumJ.multiply(a, b, t);
        NumJ.add(t, c, t);
        NumJ.multiply(t, t, t);
        return NumJ.sum(t);
    }

    @Benchmark
    public double java() {
        double s = 0;
        double[] x = ha, y = hb, z = hc;
        for (int i = 0; i < x.length; i++) {
            double v = x[i] * y[i] + z[i];
            s += v * v;
        }
        return s;
    }
}
