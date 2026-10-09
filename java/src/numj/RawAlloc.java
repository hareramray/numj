package numj;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Uninitialised native memory from the C runtime ({@code malloc}/{@code free}), attached to an arena so that closing
 * the arena frees it. Used for operation results unless {@code -Dnumj.resultAlloc=arena} (see docs/PERFORMANCE.md).
 * The memory is not zeroed: callers must overwrite all of it before it is read.
 */
@SuppressWarnings("restricted")
final class RawAlloc {
    private RawAlloc() {}

    private static final MethodHandle MALLOC, FREE;

    static {
        Linker l = Linker.nativeLinker();
        MALLOC = l.downcallHandle(l.defaultLookup().find("malloc").orElseThrow(), FunctionDescriptor.of(ADDRESS, JAVA_LONG));
        FREE = l.downcallHandle(l.defaultLookup().find("free").orElseThrow(), FunctionDescriptor.ofVoid(ADDRESS));
    }

    static MemorySegment allocate(Arena arena, long bytes, long alignment) {
        try {
            MemorySegment raw = (MemorySegment) MALLOC.invokeExact(Math.max(bytes, 1) + alignment - 1);
            if (raw.address() == 0) throw new OutOfMemoryError("malloc(" + bytes + ") failed");
            long aligned = (raw.address() + alignment - 1) & -alignment;
            return MemorySegment.ofAddress(aligned).reinterpret(bytes, arena, s -> free(raw));
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    private static void free(MemorySegment raw) {
        try {
            FREE.invokeExact(raw);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }
}
