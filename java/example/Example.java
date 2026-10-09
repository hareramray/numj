import numj.F64Array;
import numj.NumJ;

/** Minimal numj usage. Run with: java --enable-native-access=ALL-UNNAMED -cp build/classes;build/example-classes Example */
public class Example {
    public static void main(String[] args) {
        // Arrays live in native memory owned by the try-with-resources block; close() frees them.
        try (F64Array a = F64Array.of(1, 2, 3, 4);
             F64Array b = F64Array.of(0, 2, 0, 4);
             F64Array c = F64Array.of(1, 1, 1, 1)) {

            System.out.println("sqdist(a, b)          = " + NumJ.sqdist(a, b));            // 1 + 0 + 9 + 0 = 10
            System.out.println("sum((a*b + c)^2)      = " + NumJ.sumSqMulAdd(a, b, c));    // 1 + 25 + 1 + 289 = 316

            // 2-D, row-major. Fill once, then run kernels on the same native memory without copying.
            try (F64Array x = F64Array.copyOf(new double[] {3, 4, 0, 0, 1, 1}, 3, 2);
                 F64Array norms = F64Array.allocate(3)) {
                NumJ.normalizeRowsInPlace(x, norms);
                for (int r = 0; r < x.rows(); r++)
                    System.out.printf("row %d -> [%.4f, %.4f]  norm %.4f%n", r, x.get(r, 0), x.get(r, 1), norms.get(r));
            }
        }
    }
}
