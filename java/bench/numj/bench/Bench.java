package numj.bench;

import numj.F64Array;
import numj.NumJ;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.function.DoubleSupplier;

/**
 * Benchmark harness (no JMH, to keep zero dependencies). Protocol per case, identical to bench/numpy_bench.py:
 * <ol>
 *   <li>warm up for {@code --warmup-ms} (JIT reaches C2 well before 1.5 s for these loops);</li>
 *   <li>calibrate reps K so one sample lasts >= {@code --sample-ms};</li>
 *   <li>take {@code --samples} samples; per-op time = sample time / K;</li>
 *   <li>report median, min, p10, p90, mean, stdev. Every result is folded into a sink that is printed,
 *       so the JIT cannot eliminate the work.</li>
 * </ol>
 * Modes: {@code compute} = data already in the implementation's container, outputs preallocated;
 * {@code e2e} = starts from Java double[] and includes native allocation, copy-in, compute, copy-out, free.
 *
 * Usage: Bench --suite main|profiles|threads|overhead --out results/x.csv [--profile name] [--quick]
 */
public final class Bench {
    static double sink;
    static long warmupMs = 1500, sampleMs = 20;
    static int samples = 21;
    static String suite = "main", profile = "numj", out = null;

    record Row(String suite, String kernel, String shape, long n, String impl, String mode, int threads, String profile,
               double[] perOpNs, long reps, double result) {}

    static final List<Row> rows = new ArrayList<>();

    public static void main(String[] args) throws IOException {
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--suite" -> suite = args[++i];
                case "--out" -> out = args[++i];
                case "--profile" -> profile = args[++i];
                case "--samples" -> samples = Integer.parseInt(args[++i]);
                case "--warmup-ms" -> warmupMs = Long.parseLong(args[++i]);
                case "--sample-ms" -> sampleMs = Long.parseLong(args[++i]);
                case "--quick" -> { warmupMs = 200; samples = 5; sampleMs = 5; }
                default -> throw new IllegalArgumentException("unknown arg " + args[i]);
            }
        }
        System.out.println("# java " + System.getProperty("java.vm.version") + " | " + System.getProperty("java.vm.name"));
        System.out.println("# native " + NumJ.libraryPath());
        System.out.println("# build " + NumJ.buildInfo());
        System.out.println("# criticalMaxElements=" + NumJ.CRITICAL_MAX_ELEMENTS + " availableProcessors=" + Runtime.getRuntime().availableProcessors());
        switch (suite) {
            case "main" -> mainSuite(true);
            case "profiles" -> mainSuite(false);
            case "threads" -> threadsSuite();
            case "overhead" -> overheadSuite();
            case "fingerprints" -> fingerprints();
            default -> throw new IllegalArgumentException("unknown suite " + suite);
        }
        if (out != null) writeCsv(Path.of(out));
        System.out.println("# sink " + sink);
    }

    // ------------------------------------------------------------------ suites

    static final int[] VEC_SIZES = {16, 1_000, 100_000, 10_000_000};
    static final int[][] NORM_SHAPES = {{4, 4}, {100, 10}, {1_000, 100}, {100_000, 100}};
    static final int[][] ROWS_SHAPES = {{1_000, 16}, {10_000, 100}, {100_000, 100}};

    /** withBaselines=false: numj compute-only only (used to compare native build profiles). */
    static void mainSuite(boolean withBaselines) {
        for (int n : VEC_SIZES) {
            double[] a = Data.splitmix(1, n), b = Data.splitmix(2, n), c = Data.splitmix(3, n);
            String shape = "[" + n + "]";
            try (F64Array A = F64Array.of(a); F64Array B = F64Array.of(b); F64Array C = F64Array.of(c)) {
                run("sqdist", shape, n, "numj", "compute", 1, () -> NumJ.sqdist(A, B));
                run("sumsq_muladd", shape, n, "numj", "compute", 1, () -> NumJ.sumSqMulAdd(A, B, C));
            }
            if (!withBaselines) continue;
            run("sqdist", shape, n, "numj", "e2e", 1, () -> {
                try (F64Array A = F64Array.of(a); F64Array B = F64Array.of(b)) { return NumJ.sqdist(A, B); }
            });
            run("sqdist", shape, n, "java-naive", "compute", 1, () -> JavaKernels.sqdistNaive(a, b));
            run("sqdist", shape, n, "java-blocked", "compute", 1, () -> JavaKernels.sqdistBlocked(a, b));
            run("sumsq_muladd", shape, n, "numj", "e2e", 1, () -> {
                try (F64Array A = F64Array.of(a); F64Array B = F64Array.of(b); F64Array C = F64Array.of(c)) {
                    return NumJ.sumSqMulAdd(A, B, C);
                }
            });
            run("sumsq_muladd", shape, n, "java-naive", "compute", 1, () -> JavaKernels.sumSqMulAddNaive(a, b, c));
            run("sumsq_muladd", shape, n, "java-blocked", "compute", 1, () -> JavaKernels.sumSqMulAddBlocked(a, b, c));
        }

        for (int[] s : NORM_SHAPES) {
            int r = s[0], k = s[1], n = r * k;
            String shape = "[" + r + "," + k + "]";
            double[] x = Data.splitmix(4, n), y = new double[n];
            try (F64Array X = F64Array.copyOf(x, r, k); F64Array Y = F64Array.allocate(r, k)) {
                run("normalize_rows", shape, n, "numj", "compute", 1, () -> { NumJ.normalizeRows(X, Y, null); return Y.get(n - 1); });
                if (withBaselines) {
                    run("normalize_rows", shape, n, "numj", "e2e", 1, () -> {
                        try (F64Array X2 = F64Array.copyOf(x, r, k); F64Array Y2 = F64Array.allocate(r, k)) {
                            NumJ.normalizeRows(X2, Y2, null);
                            return Y2.toArray()[n - 1];
                        }
                    });
                }
            }
            if (!withBaselines) continue;
            run("normalize_rows", shape, n, "java-naive", "compute", 1, () -> { JavaKernels.normalizeRowsNaive(x, y, r, k); return y[n - 1]; });
            run("normalize_rows", shape, n, "java-blocked", "compute", 1, () -> { JavaKernels.normalizeRowsBlocked(x, y, r, k, null); return y[n - 1]; });
        }

        for (int[] s : ROWS_SHAPES) {
            int r = s[0], k = s[1], n = r * k;
            String shape = "[" + r + "," + k + "]";
            double[] x = Data.splitmix(4, n), q = Data.splitmix(5, k), o = new double[r];
            try (F64Array X = F64Array.copyOf(x, r, k); F64Array Q = F64Array.of(q); F64Array O = F64Array.allocate(r)) {
                run("sqdist_rows", shape, n, "numj", "compute", 1, () -> { NumJ.sqdistRows(Q, X, O); return O.get(r - 1); });
                if (withBaselines) {
                    // Unbatched: one native call per row (row views created up front), to show call overhead.
                    F64Array[] views = new F64Array[r];
                    for (int i = 0; i < r; i++) views[i] = X.row(i);
                    run("sqdist_rows", shape, n, "numj-per-row-calls", "compute", 1, () -> {
                        double acc = 0;
                        for (F64Array v : views) acc = NumJ.sqdist(v, Q);
                        return acc;
                    });
                    run("sqdist_rows", shape, n, "numj", "e2e", 1, () -> {
                        try (F64Array X2 = F64Array.copyOf(x, r, k); F64Array Q2 = F64Array.of(q); F64Array O2 = F64Array.allocate(r)) {
                            NumJ.sqdistRows(Q2, X2, O2);
                            return O2.toArray()[r - 1];
                        }
                    });
                }
            }
            if (!withBaselines) continue;
            run("sqdist_rows", shape, n, "java-naive", "compute", 1, () -> { JavaKernels.sqdistRowsNaive(q, x, r, k, o); return o[r - 1]; });
            run("sqdist_rows", shape, n, "java-blocked", "compute", 1, () -> { JavaKernels.sqdistRowsBlocked(q, x, r, k, o); return o[r - 1]; });
        }
    }

    /** Prints input-data fingerprints; bench/report.py checks them against the NumPy side. */
    static void fingerprints() {
        long[][] cases = {{1, 1000}, {2, 1000}, {3, 1000}, {4, 10_000}, {5, 100}};
        for (long[] c : cases)
            System.out.printf("fingerprint splitmix(%d,%d) %016x%n", c[0], c[1], Data.fingerprint(Data.splitmix(c[0], (int) c[1])));
    }

    /** numj compute-only on large inputs with 1..8 native threads. */
    static void threadsSuite() {
        int[] sizes = {1 << 18, 1_000_000, 10_000_000, 50_000_000};
        int[] threadCounts = {1, 2, 4, 8};
        for (int n : sizes) {
            String shape = "[" + n + "]";
            try (F64Array A = F64Array.of(Data.splitmix(1, n)); F64Array B = F64Array.of(Data.splitmix(2, n));
                 F64Array C = F64Array.of(Data.splitmix(3, n))) {
                for (int t : threadCounts) {
                    NumJ.setThreads(t);
                    run("sqdist", shape, n, "numj", "compute", t, () -> NumJ.sqdist(A, B));
                    run("sumsq_muladd", shape, n, "numj", "compute", t, () -> NumJ.sumSqMulAdd(A, B, C));
                }
            }
            int r = n / 100;
            try (F64Array X = F64Array.copyOf(Data.splitmix(4, r * 100), r, 100); F64Array Y = F64Array.allocate(r, 100)) {
                for (int t : threadCounts) {
                    NumJ.setThreads(t);
                    run("normalize_rows", "[" + r + ",100]", (long) r * 100, "numj", "compute", t,
                            () -> { NumJ.normalizeRows(X, Y, null); return Y.get(0); });
                }
            }
        }
        NumJ.setThreads(1);
    }

    /** Fixed per-call cost: tiny inputs, run once with critical downcalls and once without (see run_bench.ps1). */
    static void overheadSuite() {
        String mode = NumJ.CRITICAL_MAX_ELEMENTS >= 0 ? "critical" : "regular";
        for (int n : new int[] {0, 1, 16, 17, 256}) {
            try (F64Array A = F64Array.of(Data.splitmix(1, n)); F64Array B = F64Array.of(Data.splitmix(2, n))) {
                run("sqdist", "[" + n + "]", n, "numj-" + mode, "compute", 1, () -> NumJ.sqdist(A, B));
            }
        }
    }

    // ------------------------------------------------------------------ measurement

    static void run(String kernel, String shape, long n, String impl, String mode, int threads, DoubleSupplier op) {
        long warmEnd = System.nanoTime() + warmupMs * 1_000_000L;
        double s = 0;
        do { s += op.getAsDouble(); } while (System.nanoTime() < warmEnd);

        long reps = 1;
        while (true) {
            long t0 = System.nanoTime();
            for (long i = 0; i < reps; i++) s += op.getAsDouble();
            long dt = System.nanoTime() - t0;
            if (dt >= sampleMs * 1_000_000L) break;
            reps = Math.max(reps * 2, (long) Math.ceil(reps * (sampleMs * 1_000_000.0 / Math.max(dt, 1)) * 1.1));
        }
        double[] perOp = new double[samples];
        for (int k = 0; k < samples; k++) {
            long t0 = System.nanoTime();
            for (long i = 0; i < reps; i++) s += op.getAsDouble();
            perOp[k] = (System.nanoTime() - t0) / (double) reps;
        }
        double result = op.getAsDouble();
        sink += s;
        Row row = new Row(suite, kernel, shape, n, impl, mode, threads, profile, perOp, reps, result);
        rows.add(row);
        double[] st = stats(perOp);
        System.out.printf(Locale.ROOT, "%-15s %-16s %-22s %-8s t=%d  median %12s  p10 %12s  p90 %12s  cv %5.1f%%%n",
                kernel, shape, impl, mode, threads, fmt(st[0]), fmt(st[2]), fmt(st[3]), 100 * st[5] / st[4]);
    }

    /** median, min, p10, p90, mean, stdev */
    static double[] stats(double[] v) {
        double[] s = v.clone();
        Arrays.sort(s);
        double mean = Arrays.stream(s).average().orElse(Double.NaN);
        double var = Arrays.stream(s).map(x -> (x - mean) * (x - mean)).sum() / Math.max(1, s.length - 1);
        return new double[] {pct(s, 0.5), s[0], pct(s, 0.1), pct(s, 0.9), mean, Math.sqrt(var)};
    }

    static double pct(double[] sorted, double p) {
        double idx = p * (sorted.length - 1);
        int lo = (int) Math.floor(idx), hi = (int) Math.ceil(idx);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
    }

    static String fmt(double ns) {
        if (ns < 1e3) return String.format(Locale.ROOT, "%.1f ns", ns);
        if (ns < 1e6) return String.format(Locale.ROOT, "%.2f us", ns / 1e3);
        return String.format(Locale.ROOT, "%.3f ms", ns / 1e6);
    }

    static void writeCsv(Path path) throws IOException {
        Files.createDirectories(path.toAbsolutePath().getParent());
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(path))) {
            w.println("suite,kernel,shape,n,impl,mode,threads,profile,median_ns,min_ns,p10_ns,p90_ns,mean_ns,stdev_ns,samples,reps,result");
            for (Row r : rows) {
                double[] st = stats(r.perOpNs());
                w.printf(Locale.ROOT, "%s,%s,\"%s\",%d,%s,%s,%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%.17g%n",
                        r.suite(), r.kernel(), r.shape(), r.n(), r.impl(), r.mode(), r.threads(), r.profile(),
                        st[0], st[1], st[2], st[3], st[4], st[5], r.perOpNs().length, r.reps(), r.result());
            }
        }
        System.out.println("# wrote " + path);
    }
}
