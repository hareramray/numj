package numj;

import java.util.List;

/**
 * Explicit expression trees for <b>pattern-based fusion</b>.
 *
 * <p>Ordinary method calls ({@code NumJ.add(...)}, {@code NumJ.multiply(...)}) are always evaluated eagerly, one
 * operation at a time, each producing a full temporary array. Nothing is fused automatically. To evaluate a
 * reduction in a single pass, build it explicitly with this class:
 * <pre>{@code
 * Expr.Reduction r = Expr.sum(Expr.of(a).mul(Expr.of(b)).add(Expr.of(c)).square());   // sum((a*b + c)**2)
 * r.fusedKernel();      // "sumSqMulAdd"
 * double s = r.evaluate();
 * }</pre>
 * {@link Reduction#evaluate()} recognises a fixed list of patterns ({@link #SUPPORTED_PATTERNS}) and maps each
 * to a precompiled Fortran kernel. It is <em>not</em> an expression compiler: any other expression makes
 * {@code evaluate()} throw {@link UnsupportedOperationException}, and {@link Reduction#evaluateUnfused()} must be
 * called explicitly to evaluate it step by step with temporaries. Matching is exact: operands must be the same
 * {@link F64Array} objects in the pattern positions and have identical shapes (the fused kernels do not broadcast).
 * Only rewrites that are bitwise exact in IEEE arithmetic are applied ({@code x*x} for {@code x**2}, and
 * commutativity of a single {@code +} or {@code *}); the fused and unfused results are therefore bitwise identical.
 */
public abstract sealed class Expr permits Expr.Leaf, Expr.Binary, Expr.Square {

    /** Patterns {@link Reduction#evaluate()} maps to fused kernels. */
    public static final List<String> SUPPORTED_PATTERNS = List.of(
            "sum(a)                -> NumJ.sum",
            "sum((a - b)**2)       -> NumJ.sqdist",
            "sum((a * b + c)**2)   -> NumJ.sumSqMulAdd   (also c + a*b, b*a + c)");

    Expr() {}

    /** Leaf referring to an existing array (not copied, not closed). */
    public static Expr of(F64Array a) {
        return new Leaf(a);
    }

    public Expr add(Expr o) { return new Binary('+', this, o); }
    public Expr sub(Expr o) { return new Binary('-', this, o); }
    public Expr mul(Expr o) { return new Binary('*', this, o); }
    public Expr div(Expr o) { return new Binary('/', this, o); }
    public Expr add(F64Array o) { return add(of(o)); }
    public Expr sub(F64Array o) { return sub(of(o)); }
    public Expr mul(F64Array o) { return mul(of(o)); }
    public Expr div(F64Array o) { return div(of(o)); }

    /** {@code this ** 2}, evaluated as {@code x * x} (exactly what NumPy does for a power of 2). */
    public Expr square() { return new Square(this); }

    /** {@code this ** p}; only {@code p == 2} is supported. */
    public Expr pow(int p) {
        if (p != 2) throw new UnsupportedOperationException("only pow(2) is supported, got " + p);
        return square();
    }

    /** Sum of all elements of {@code e}. */
    public static Reduction sum(Expr e) {
        return new Reduction(e);
    }

    // ------------------------------------------------------------------ nodes

    public static final class Leaf extends Expr {
        final F64Array array;

        Leaf(F64Array array) {
            this.array = java.util.Objects.requireNonNull(array, "array");
        }

        @Override
        public String toString() { return "a" + Integer.toHexString(System.identityHashCode(array)); }
    }

    public static final class Binary extends Expr {
        final char op;
        final Expr left, right;

        Binary(char op, Expr left, Expr right) {
            this.op = op;
            this.left = java.util.Objects.requireNonNull(left);
            this.right = java.util.Objects.requireNonNull(right);
        }

        @Override
        public String toString() { return "(" + left + " " + op + " " + right + ")"; }
    }

    public static final class Square extends Expr {
        final Expr base;

        Square(Expr base) {
            this.base = java.util.Objects.requireNonNull(base);
        }

        @Override
        public String toString() { return base + "**2"; }
    }

    // ------------------------------------------------------------------ reductions

    /** A full reduction over an expression. */
    public static final class Reduction {
        private final Expr body;

        Reduction(Expr body) {
            this.body = body;
        }

        /** Name of the fused kernel {@link #evaluate()} would use, or {@code null} if no pattern matches. */
        public String fusedKernel() {
            Match m = match();
            return m == null ? null : m.kernel;
        }

        /** Evaluates with a fused kernel; throws {@link UnsupportedOperationException} if no pattern matches. */
        public double evaluate() {
            Match m = match();
            if (m == null)
                throw new UnsupportedOperationException("no fused kernel for sum(" + body + "); supported: "
                        + SUPPORTED_PATTERNS + ". Use evaluateUnfused() to evaluate it with temporaries.");
            return switch (m.kernel) {
                case "sum" -> NumJ.sum(m.a);
                case "sqdist" -> NumJ.sqdist(m.a, m.b);
                default -> NumJ.sumSqMulAdd(m.a, m.b, m.c);
            };
        }

        /** Evaluates operation by operation (broadcasting allowed), allocating and freeing temporaries. */
        public double evaluateUnfused() {
            try (Value v = eval(body)) {
                return NumJ.sum(v.array);
            }
        }

        @Override
        public String toString() { return "sum(" + body + ")"; }

        private record Match(String kernel, F64Array a, F64Array b, F64Array c) {}

        private Match match() {
            if (body instanceof Leaf l) return new Match("sum", l.array, null, null);
            Expr sq = squared(body);
            if (sq == null) return null;
            if (sq instanceof Binary d && d.op == '-' && d.left instanceof Leaf a && d.right instanceof Leaf b
                    && a.array.sameShape(b.array))
                return new Match("sqdist", a.array, b.array, null);
            if (sq instanceof Binary s && s.op == '+') {
                Match m = mulAdd(s.left, s.right);
                return m != null ? m : mulAdd(s.right, s.left);
            }
            return null;
        }

        /** {@code e} as {@code x**2} or {@code x*x} (same subtree object) -> {@code x}. */
        private static Expr squared(Expr e) {
            if (e instanceof Square s) return s.base;
            if (e instanceof Binary b && b.op == '*' && b.left == b.right) return b.left;
            return null;
        }

        private static Match mulAdd(Expr prod, Expr add) {
            if (prod instanceof Binary p && p.op == '*' && p.left instanceof Leaf a && p.right instanceof Leaf b
                    && add instanceof Leaf c && a.array.sameShape(b.array) && a.array.sameShape(c.array))
                return new Match("sumSqMulAdd", a.array, b.array, c.array);
            return null;
        }

        /** Evaluated subexpression; closes the array only if it is a temporary. */
        private record Value(F64Array array, boolean temp) implements AutoCloseable {
            @Override
            public void close() {
                if (temp) array.close();
            }
        }

        private static Value eval(Expr e) {
            if (e instanceof Leaf l) return new Value(l.array, false);
            if (e instanceof Square s) {
                try (Value x = eval(s.base)) {
                    return new Value(NumJ.multiply(x.array, x.array), true);
                }
            }
            Binary b = (Binary) e;
            try (Value x = eval(b.left); Value y = eval(b.right)) {
                F64Array r = switch (b.op) {
                    case '+' -> NumJ.add(x.array, y.array);
                    case '-' -> NumJ.subtract(x.array, y.array);
                    case '*' -> NumJ.multiply(x.array, y.array);
                    default -> NumJ.divide(x.array, y.array);
                };
                return new Value(r, true);
            }
        }
    }
}
