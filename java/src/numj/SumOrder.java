package numj;

/**
 * Which sequence a reduction sums (the blocked algorithm and its error bound are the same either way).
 *
 * <ul>
 *   <li>{@link #LOGICAL} (default): the elements in C order of the array's <em>logical</em> indices. The result
 *       depends only on the values and the shape, so a view and a contiguous copy of the same values give
 *       identical bits. For views whose logical order walks memory with a large stride (e.g. the full sum of a
 *       transposed array) this reads memory in a cache-unfriendly order and is slower.</li>
 *   <li>{@link #MEMORY} (opt-in): the elements in the order they are laid out in memory (reduced axes ordered by
 *       decreasing |stride|, reversed axes walked forwards). A transposed or reversed view then sums at the
 *       speed of a contiguous array. The result is deterministic for a given layout (shape, strides) and
 *       independent of the thread count, but {@code sum(a.T)} and {@code sum(a.T.copy())} may differ in the last
 *       bits. For C-contiguous arrays both orders are the same sequence and give identical results.</li>
 * </ul>
 */
public enum SumOrder {
    LOGICAL,
    MEMORY
}
