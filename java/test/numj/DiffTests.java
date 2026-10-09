package numj;

import numj.bench.Data;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static numj.T.*;

/**
 * Differential tests against NumPy: replays build/difftest/cases.json (written by difftest/gen_numpy_cases.py) and
 * compares numj's outcome with NumPy's. See the generator's docstring for the comparison rules. Each case runs with
 * the default path selection, with the native path forced, and with the Java path forced.
 */
public final class DiffTests {
    static final double[] SPECIALS = {0.0, -0.0, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NaN,
        Double.MIN_VALUE, -Double.MIN_VALUE, Double.MIN_NORMAL, Double.MAX_VALUE, -Double.MAX_VALUE, 1.0, -1.0, 3.0, 0.1};

    static Path dir;
    static double maxRatio;          // largest |numj - NumPy| / tolerance over all reduction outputs
    static long reductionOutputs, exactOutputs;

    public static void main(String[] args) throws IOException {
        dir = Path.of(args.length > 0 ? args[0] : "build/difftest");
        @SuppressWarnings("unchecked")
        Map<String, Object> meta = (Map<String, Object>) Json.parse(Files.readString(dir.resolve("cases.json")));
        System.out.println("numj library: " + NumJ.libraryPath());
        System.out.println("reference: NumPy " + meta.get("numpy") + " on Python " + meta.get("python"));
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> cases = (List<Map<String, Object>>) meta.get("cases");
        Map<String, int[]> counts = new LinkedHashMap<>();
        for (Map<String, Object> c : cases) {
            String tag = ((List<?>) c.get("tags")).isEmpty() ? "other" : (String) ((List<?>) c.get("tags")).get(0);
            int before = failed;
            test("[" + tag + "] " + c.get("id"), () -> {
                for (long jm : new long[] {-2, -1, Long.MAX_VALUE}) runCase(c, jm);
            });
            counts.computeIfAbsent(tag, k -> new int[2])[failed > before ? 1 : 0]++;
        }
        counts.forEach((k, v) -> System.out.printf("  %-12s %4d passed %4d failed%n", k, v[0], v[1]));
        System.out.printf("  reductions: %d finite outputs compared, %d bitwise equal to NumPy, max |diff|/tolerance = %.3g%n",
                reductionOutputs, exactOutputs, maxRatio);
        System.exit(finish());
    }

    // ------------------------------------------------------------------ replay

    @SuppressWarnings("unchecked")
    static void runCase(Map<String, Object> c, long javaMax) {
        long saveE = Elementwise.javaMaxElements, saveC = Elementwise.contigJavaMaxElements, saveR = Reduce.javaMaxElements;
        if (javaMax != -2) {
            Elementwise.javaMaxElements = javaMax;
            Elementwise.contigJavaMaxElements = javaMax;
            Reduce.javaMaxElements = javaMax;
        }
        String mode = javaMax == -2 ? "default" : javaMax == -1 ? "native" : "java";
        Map<String, Object> exp = (Map<String, Object>) c.get("expect");
        Map<String, Object> op = (Map<String, Object>) c.get("op");
        Map<String, F64Array> bases = new LinkedHashMap<>();
        try {
            for (Map.Entry<String, Object> e : ((Map<String, Object>) c.get("bases")).entrySet())
                bases.put(e.getKey(), makeBase((Map<String, Object>) e.getValue()));
            if (Boolean.TRUE.equals(op.get("negzero"))) bases.get("A").fill(-0.0);
            Map<String, F64Array> arrays = new LinkedHashMap<>();
            Object result;
            try {
                for (Map.Entry<String, Object> e : ((Map<String, Object>) c.get("inputs")).entrySet()) {
                    Map<String, Object> spec = (Map<String, Object>) e.getValue();
                    F64Array x = bases.get((String) spec.get("base"));
                    for (Object v : (List<Object>) spec.getOrDefault("view", List.of())) x = applyView(x, (List<Object>) v);
                    arrays.put(e.getKey(), x);
                }
                result = runOp(op, arrays);
            } catch (RuntimeException ex) {
                if ("error".equals(exp.get("status"))) {
                    String want = (String) exp.get("java");
                    if (!matches(ex, want))
                        throw new AssertionError(mode + ": expected " + want + " (NumPy " + exp.get("numpy") + ": "
                                + exp.get("message") + "), got " + ex);
                    return;
                }
                throw new AssertionError(mode + ": unexpected " + ex + " (NumPy succeeded)", ex);
            }
            if ("error".equals(exp.get("status"))) {
                if (result instanceof F64Array r && r.isOwner()) r.close();
                throw new AssertionError(mode + ": expected " + exp.get("java") + " (NumPy " + exp.get("numpy") + ": "
                        + exp.get("message") + "), numj succeeded");
            }
            F64Array r = (F64Array) result;
            try {
                compare(c, op, exp, r, arrays, bases, mode);
            } finally {
                if (r.isOwner()) r.close();
            }
        } finally {
            Elementwise.javaMaxElements = saveE;
            Elementwise.contigJavaMaxElements = saveC;
            Reduce.javaMaxElements = saveR;
            for (F64Array b : bases.values()) b.close();
        }
    }

    static boolean matches(RuntimeException ex, String javaName) {
        return switch (javaName) {
            case "IllegalArgumentException" -> ex instanceof IllegalArgumentException;
            case "IndexOutOfBoundsException" -> ex instanceof IndexOutOfBoundsException;
            case "ReadOnlyArrayException" -> ex instanceof ReadOnlyArrayException;
            default -> false;
        };
    }

    @SuppressWarnings("unchecked")
    static F64Array makeBase(Map<String, Object> spec) {
        long seed = ((Number) spec.get("seed")).longValue();
        long[] shape = longs((List<Object>) spec.get("shape"));
        int n = (int) size(shape);
        double[] d = Data.splitmix(seed, n);
        if (Boolean.TRUE.equals(spec.get("special")))
            for (int k = 0, pos = 0; pos < n; k++, pos += 3) d[pos] = SPECIALS[(int) ((k + seed) % SPECIALS.length)];
        return F64Array.copyOf(d, shape);
    }

    @SuppressWarnings("unchecked")
    static F64Array applyView(F64Array x, List<Object> v) {
        String kind = (String) v.get(0);
        return switch (kind) {
            case "slice" -> x.slice((String) v.get(1));
            case "T" -> x.transpose();
            case "permute" -> x.permute(ints((List<Object>) v.get(1)));
            case "swap" -> x.swapAxes(((Number) v.get(1)).intValue(), ((Number) v.get(2)).intValue());
            case "reshape" -> x.reshape(longs((List<Object>) v.get(1)));
            case "bcast" -> x.broadcastTo(longs((List<Object>) v.get(1)));
            default -> throw new IllegalStateException("unknown view op " + kind);
        };
    }

    @SuppressWarnings("unchecked")
    static Object runOp(Map<String, Object> op, Map<String, F64Array> arrays) {
        String kind = (String) op.get("kind");
        F64Array out = op.get("out") == null ? null : arrays.get((String) op.get("out"));
        switch (kind) {
            case "view":
                return arrays.get((String) op.get("of"));
            case "copy":
                return arrays.get((String) op.get("of")).copy();
            case "flattenCopy":
                return arrays.get((String) op.get("of")).flattenCopy();
            case "reshapeCopy":
                return arrays.get((String) op.get("of")).reshapeCopy(longs((List<Object>) op.get("shape")));
            case "sum":
            case "mean": {
                F64Array x = arrays.get((String) op.get("of"));
                int[] axes = op.get("axes") == null ? null : ints((List<Object>) op.get("axes"));
                boolean keep = Boolean.TRUE.equals(op.get("keepdims"));
                boolean mean = kind.equals("mean");
                if (out == null) return mean ? NumJ.mean(x, axes, keep) : NumJ.sum(x, axes, keep);
                return mean ? NumJ.mean(x, axes, keep, out) : NumJ.sum(x, axes, keep, out);
            }
            default: {
                Object a = op.get("a"), b = op.get("b");
                int code = switch (kind) {
                    case "add" -> Elementwise.ADD;
                    case "subtract" -> Elementwise.SUB;
                    case "multiply" -> Elementwise.MUL;
                    default -> Elementwise.DIV;
                };
                if (a instanceof Map<?, ?> sa) {
                    double s = num(sa.get("scalar"));
                    return Elementwise.scalar(Elementwise.reversed(code), arrays.get((String) b), s, out);
                }
                if (b instanceof Map<?, ?> sb) return Elementwise.scalar(code, arrays.get((String) a), num(sb.get("scalar")), out);
                return Elementwise.binary(code, arrays.get((String) a), arrays.get((String) b), out);
            }
        }
    }

    // ------------------------------------------------------------------ comparison

    @SuppressWarnings("unchecked")
    static void compare(Map<String, Object> c, Map<String, Object> op, Map<String, Object> exp, F64Array r,
                        Map<String, F64Array> arrays, Map<String, F64Array> bases, String mode) {
        String kind = (String) op.get("kind");
        long[] wantShape = longs((List<Object>) exp.get("shape"));
        assertShape(mode + " shape", wantShape, r.shape());
        Npy want = Npy.read(dir.resolve((String) exp.get("file")));
        assertShape(mode + " npy shape", wantShape, want.shape);
        double[] got = values(r);
        if (kind.equals("sum") || kind.equals("mean")) {
            Npy abs = Npy.read(dir.resolve((String) exp.get("absfile")));
            long n = ((Number) exp.get("count")).longValue();
            double g = gamma(NumJ.summationDepth(n)) + gamma(Math.max(n - 1, 0));
            for (int i = 0; i < got.length; i++) {
                double w = want.data[i], v = got[i];
                if (!Double.isFinite(w) || !Double.isFinite(v) || n <= 1) {
                    if (n <= 1 || !Double.isFinite(w)) assertBits(mode + " [" + i + "]", w, v);
                    else throw new AssertionError(mode + " [" + i + "]: NumPy " + w + " numj " + v);
                    continue;
                }
                double tol = g * abs.data[i] * (1 + 4 * U);
                if (kind.equals("mean")) tol = tol / n + 2 * U * Math.abs(w);
                reductionOutputs++;
                if (w == v) exactOutputs++;
                if (tol > 0) maxRatio = Math.max(maxRatio, Math.abs(w - v) / tol);
                if (Math.abs(w - v) > tol)
                    throw new AssertionError(mode + " [" + i + "]: NumPy " + w + " numj " + v + " |diff| " + Math.abs(w - v)
                            + " > tol " + tol);
            }
        } else {
            for (int i = 0; i < got.length; i++) assertBits(mode + " value [" + i + "]", want.data[i], got[i]);
        }
        if (kind.equals("view")) {
            long[] ws = longs((List<Object>) exp.get("strides")), gs = r.strides();
            for (int d = 0; d < ws.length; d++)
                if (wantShape[d] > 1 && ws[d] != gs[d])
                    throw new AssertionError(mode + " strides: NumPy " + Arrays.toString(ws) + " numj " + Arrays.toString(gs));
            check(mode + " C flag", r.isCContiguous() == (Boolean) exp.get("c"));
            check(mode + " F flag", r.isFContiguous() == (Boolean) exp.get("f"));
            check(mode + " writeable", r.isWritable() == (Boolean) exp.get("writeable"));
        }
        if (exp.containsKey("shares")) {
            boolean npShares = (Boolean) exp.get("shares");
            F64Array src = kind.equals("view")
                    ? bases.get((String) ((Map<String, Object>) ((Map<String, Object>) c.get("inputs")).get(op.get("of"))).get("base"))
                    : arrays.get((String) op.get("of"));
            // documented difference: NumPy returns a scalar copy for an all-integer index, numj a 0-d view
            if (kind.equals("view") && r.size() > 0 && !Boolean.TRUE.equals(exp.get("numpy_scalar")))
                check(mode + " view/copy like NumPy", (!r.isOwner() && r.overlaps(src)) == npShares);
            if (kind.endsWith("opy") || kind.equals("flattenCopy"))
                check(mode + " copy does not share memory", r.isOwner() && !r.overlaps(src) && !npShares);
        }
        if (exp.containsKey("basefile")) {
            Npy b = Npy.read(dir.resolve((String) exp.get("basefile")));
            double[] bv = bases.get((String) exp.get("outbase")).toArray();
            // reductions: elements written through out were compared with the tolerance above; everything else
            // in the buffer must be bitwise unchanged (and for elementwise ops, everything is bitwise)
            java.util.Set<Long> written = new java.util.HashSet<>();
            if (kind.equals("sum") || kind.equals("mean")) {
                F64Array o = arrays.get((String) op.get("out"));
                forEach(o.shape(), idx -> {
                    long off = o.byteOffset();
                    for (int d = 0; d < idx.length; d++) off += idx[d] * o.strides()[d];
                    written.add(off / 8);
                });
            }
            for (int i = 0; i < bv.length; i++)
                if (!written.contains((long) i)) assertBits(mode + " memory after out= [" + i + "]", b.data[i], bv[i]);
        }
    }

    // ------------------------------------------------------------------ helpers

    static long[] longs(List<Object> l) {
        long[] r = new long[l.size()];
        for (int i = 0; i < r.length; i++) r[i] = ((Number) l.get(i)).longValue();
        return r;
    }

    static int[] ints(List<Object> l) {
        int[] r = new int[l.size()];
        for (int i = 0; i < r.length; i++) r[i] = ((Number) l.get(i)).intValue();
        return r;
    }

    static double num(Object o) {
        if (o instanceof Number n) return n.doubleValue();
        return switch ((String) o) {   // json.dump writes NaN / Infinity / -Infinity as bare tokens
            case "NaN" -> Double.NaN;
            case "Infinity" -> Double.POSITIVE_INFINITY;
            case "-Infinity" -> Double.NEGATIVE_INFINITY;
            default -> throw new IllegalArgumentException(String.valueOf(o));
        };
    }

    /** NumPy .npy reader for little-endian float64 C-order arrays (format versions 1-3). */
    record Npy(long[] shape, double[] data) {
        static Npy read(Path p) {
            try {
                byte[] all = Files.readAllBytes(p);
                if (all[0] != (byte) 0x93 || all[1] != 'N') throw new IOException("not an npy file: " + p);
                int major = all[6];
                ByteBuffer bb = ByteBuffer.wrap(all).order(ByteOrder.LITTLE_ENDIAN);
                int hlen, start;
                if (major == 1) { hlen = Short.toUnsignedInt(bb.getShort(8)); start = 10; }
                else { hlen = bb.getInt(8); start = 12; }
                String h = new String(all, start, hlen, StandardCharsets.ISO_8859_1);
                if (!h.contains("'descr': '<f8'") || !h.contains("'fortran_order': False"))
                    throw new IOException("unsupported npy header " + h);
                String sh = h.substring(h.indexOf("'shape': (") + 10);
                sh = sh.substring(0, sh.indexOf(')'));
                List<Long> dims = new ArrayList<>();
                for (String t : sh.split(",")) if (!t.isBlank()) dims.add(Long.parseLong(t.strip()));
                long[] shape = dims.stream().mapToLong(Long::longValue).toArray();
                double[] data = new double[(int) size(shape)];
                bb.position(start + hlen);
                for (int i = 0; i < data.length; i++) data[i] = bb.getDouble();
                return new Npy(shape, data);
            } catch (IOException e) {
                throw new IllegalStateException(e);
            }
        }
    }

    /** Minimal JSON reader (objects, arrays, strings, numbers, true/false/null, NaN/Infinity tokens). */
    static final class Json {
        private final String s;
        private int i;

        private Json(String s) { this.s = s; }

        static Object parse(String s) {
            Json j = new Json(s);
            Object v = j.value();
            j.ws();
            if (j.i != s.length()) throw new IllegalArgumentException("trailing data at " + j.i);
            return v;
        }

        private void ws() { while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++; }

        private Object value() {
            ws();
            char ch = s.charAt(i);
            if (ch == '{') {
                i++;
                Map<String, Object> m = new LinkedHashMap<>();
                ws();
                if (s.charAt(i) == '}') { i++; return m; }
                while (true) {
                    ws();
                    String k = (String) value();
                    ws();
                    i++;   // ':'
                    m.put(k, value());
                    ws();
                    if (s.charAt(i++) == '}') return m;
                }
            }
            if (ch == '[') {
                i++;
                List<Object> l = new ArrayList<>();
                ws();
                if (s.charAt(i) == ']') { i++; return l; }
                while (true) {
                    l.add(value());
                    ws();
                    if (s.charAt(i++) == ']') return l;
                }
            }
            if (ch == '"') {
                StringBuilder b = new StringBuilder();
                i++;
                while (s.charAt(i) != '"') {
                    char c = s.charAt(i++);
                    if (c == '\\') {
                        char e = s.charAt(i++);
                        switch (e) {
                            case 'n' -> b.append('\n');
                            case 't' -> b.append('\t');
                            case 'u' -> { b.append((char) Integer.parseInt(s.substring(i, i + 4), 16)); i += 4; }
                            default -> b.append(e);
                        }
                    } else b.append(c);
                }
                i++;
                return b.toString();
            }
            for (String lit : new String[] {"true", "false", "null", "NaN", "-Infinity", "Infinity"}) {
                if (s.startsWith(lit, i)) {
                    i += lit.length();
                    return switch (lit) {
                        case "true" -> Boolean.TRUE;
                        case "false" -> Boolean.FALSE;
                        case "null" -> null;
                        default -> lit;
                    };
                }
            }
            int j = i;
            while (i < s.length() && "+-0123456789.eE".indexOf(s.charAt(i)) >= 0) i++;
            String t = s.substring(j, i);
            if (t.contains(".") || t.contains("e") || t.contains("E")) return Double.parseDouble(t);
            return Long.parseLong(t);
        }
    }
}
