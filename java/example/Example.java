import numj.Expr;
import numj.F64Array;
import numj.Ix;
import numj.NumJ;

import java.lang.foreign.Arena;
import java.util.Arrays;

/**
 * numj usage tour. Run with:
 *   java --enable-native-access=ALL-UNNAMED -cp "build\classes;build\example-classes" Example
 */
public class Example {
    public static void main(String[] args) {
        fusedKernels();
        multidimensional();
        reusableBuffersAndArenas();
        explicitFusion();
    }

    /** The numj 0.1 kernels: fused, one pass, no temporaries. */
    static void fusedKernels() {
        System.out.println("== fused kernels");
        try (F64Array a = F64Array.of(1, 2, 3, 4);
             F64Array b = F64Array.of(0, 2, 0, 4);
             F64Array c = F64Array.of(1, 1, 1, 1)) {
            System.out.println("sqdist(a, b)          = " + NumJ.sqdist(a, b));            // 1 + 0 + 9 + 0 = 10
            System.out.println("sum((a*b + c)^2)      = " + NumJ.sumSqMulAdd(a, b, c));    // 1 + 25 + 1 + 289 = 316
        }
        try (F64Array x = F64Array.copyOf(new double[] {3, 4, 0, 0, 1, 1}, 3, 2);
             F64Array norms = F64Array.allocate(3)) {
            NumJ.normalizeRowsInPlace(x, norms);
            for (int r = 0; r < x.rows(); r++)
                System.out.printf("row %d -> [%.4f, %.4f]  norm %.4f%n", r, x.get(r, 0), x.get(r, 1), norms.get(r));
        }
    }

    /** Views (no copies), broadcasting and axis reductions. */
    static void multidimensional() {
        System.out.println("== n-d arrays");
        try (F64Array x = F64Array.arange(24).reshapeCopy(2, 3, 4)) {        // [2, 3, 4], C order, owner
            F64Array firstRows = x.slice("::-1, 0, 1::2");                      // views: reversed, indexed, stepped
            System.out.println("x[::-1, 0, 1::2]      = " + Arrays.toString(firstRows.toArray())
                    + "  shape " + firstRows.shapeString() + "  strides " + Arrays.toString(firstRows.strides()));
            F64Array t = x.permute(2, 0, 1);                                     // [4, 2, 3] view
            System.out.println("permute(2,0,1) shape  = " + t.shapeString() + "  C-contiguous? " + t.isCContiguous());
            System.out.println("reshape(6, 4) view?   = " + x.canReshapeView(6, 4) + "; of the permuted view? "
                    + t.canReshapeView(24));
            try (F64Array flat = t.reshapeCopy(24)) {                            // explicit copy when a view is impossible
                System.out.println("t.reshapeCopy(24)[:6] = " + Arrays.toString(flat.slice(Ix.to(6)).toArray()));
            }

            try (F64Array row = F64Array.of(100, 200, 300, 400);
                 F64Array y = NumJ.add(x, row);                                  // [2,3,4] + [4] broadcast
                 F64Array z = NumJ.multiply(2.0, x.slice(Ix.all(), Ix.all(), Ix.newAxis(), Ix.at(0)))) {   // [2,3,1]
                System.out.println("(x + row)[1, 2]       = " + Arrays.toString(y.slice("1, 2").toArray()));
                System.out.println("2 * x[:, :, None, 0]  shape " + z.shapeString());
            }

            try (F64Array s = NumJ.sum(x, new int[] {0, 2}, false);           // over axes 0 and 2 -> [3]
                 F64Array m = NumJ.mean(x, -1, true)) {                           // last axis, keepdims -> [2, 3, 1]
                System.out.println("sum(x, axis=(0,2))    = " + Arrays.toString(s.toArray()));
                System.out.println("mean(x, -1, keepdims) shape " + m.shapeString() + " first " + m.get(0));
            }
            System.out.println("sum(x), sum(x.T)      = " + NumJ.sum(x) + ", " + NumJ.sum(x.transpose()));
        }
    }

    /** Output reuse (no allocation per call) and arena-scoped temporaries. */
    static void reusableBuffersAndArenas() {
        System.out.println("== reusable buffers");
        try (Arena arena = Arena.ofConfined()) {                                  // everything below is freed with the arena
            F64Array acc = F64Array.allocate(arena, 1000);
            F64Array step = F64Array.allocate(arena, 1000);
            step.fill(0.5);
            for (int i = 0; i < 10; i++) NumJ.add(acc, step, acc);               // in place: out is the first input
            NumJ.multiply(acc, acc.slice("::-1"), acc);                           // overlapping, reversed: as if copied
            System.out.println("acc[0] after 10 x 0.5 and squaring = " + acc.get(0));
        }
    }

    /** Fusion is explicit: build the expression, then evaluate it with a matching compiled kernel. */
    static void explicitFusion() {
        System.out.println("== explicit fusion");
        try (F64Array a = F64Array.copyOf(new double[] {1, 2, 3, 4, 5, 6}, 2, 3);
             F64Array b = F64Array.copyOf(new double[] {6, 5, 4, 3, 2, 1}, 2, 3);
             F64Array c = F64Array.ones(2, 3)) {
            Expr.Reduction r = Expr.sum(Expr.of(a).mul(Expr.of(b)).add(Expr.of(c)).square());
            System.out.println(r + " -> kernel " + r.fusedKernel() + " = " + r.evaluate()
                    + " (unfused: " + r.evaluateUnfused() + ")");
            Expr.Reduction other = Expr.sum(Expr.of(a).div(Expr.of(b)));
            System.out.println("sum(a / b) has a fused kernel? " + (other.fusedKernel() != null)
                    + "; evaluated step by step = " + other.evaluateUnfused());
        }
    }
}
