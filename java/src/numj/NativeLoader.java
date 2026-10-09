package numj;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.SymbolLookup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Locale;

import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * Locates the native library. Resolution order:
 * <ol>
 *   <li>system property {@code numj.library} or environment variable {@code NUMJ_LIBRARY}: an explicit file;</li>
 *   <li>a classpath resource {@code /numj/native/<os>-<arch>/<cpu-level>/<file>}, as shipped in a
 *       platform-specific natives jar; it is copied to a private temporary directory and loaded from there;</li>
 *   <li>{@code build/native/<file>} relative to the working directory (development builds).</li>
 * </ol>
 * The CPU level is {@code x86-64-v3} (AVX2/FMA build) when the CPU reports AVX2, otherwise {@code baseline}
 * (SSE2 build, runs on every x86-64 CPU). Override with {@code -Dnumj.cpu=baseline} or
 * {@code -Dnumj.cpu=x86-64-v3}. Detection: Windows {@code IsProcessorFeaturePresent(PF_AVX2)}, Linux
 * {@code /proc/cpuinfo} (avx2, fma, bmi1, bmi2, movbe, f16c, abm); anything else uses {@code baseline}.
 */
@SuppressWarnings("restricted")
final class NativeLoader {
    private NativeLoader() {}

    static final String OS = os();
    static final String ARCH = arch();
    static final String CPU_LEVEL = cpuLevel();

    static Path resolve() {
        String p = System.getProperty("numj.library");
        if (p == null || p.isBlank()) p = System.getenv("NUMJ_LIBRARY");
        if (p != null && !p.isBlank()) return Path.of(p).toAbsolutePath();

        String file = OS.equals("windows") ? "numj.dll" : OS.equals("macos") ? "libnumj.dylib" : "libnumj.so";
        for (String level : levels()) {
            String res = "/numj/native/" + OS + "-" + ARCH + "/" + level + "/" + file;
            try (InputStream in = NativeLoader.class.getResourceAsStream(res)) {
                if (in == null) continue;
                Path dir = Files.createTempDirectory("numj-native-");
                Path lib = dir.resolve(file);
                Files.copy(in, lib, StandardCopyOption.REPLACE_EXISTING);
                lib.toFile().deleteOnExit();
                dir.toFile().deleteOnExit();
                return lib;
            } catch (IOException e) {
                throw new UnsatisfiedLinkError("cannot extract " + res + ": " + e);
            }
        }
        // development layout: build/native/numj.dll (x86-64-v3) and numj-sse2.dll (baseline)
        String devFile = CPU_LEVEL.equals("x86-64-v3") ? file : file.replace("numj", "numj-sse2");
        for (String f : List.of(devFile, file)) {
            Path def = Path.of("build", "native", f).toAbsolutePath();
            if (Files.exists(def)) return def;
        }
        throw new UnsatisfiedLinkError("numj native library not found (no numj.library property, no " + OS + "-" + ARCH
                + " natives jar on the classpath, no build/native/" + file + "); see docs/PACKAGING.md");
    }

    private static List<String> levels() {
        return CPU_LEVEL.equals("x86-64-v3") ? List.of("x86-64-v3", "baseline") : List.of("baseline");
    }

    private static String os() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        return os.contains("win") ? "windows" : os.contains("mac") ? "macos" : "linux";
    }

    private static String arch() {
        String a = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        return a.equals("amd64") || a.equals("x86_64") ? "x86_64" : a.equals("aarch64") || a.equals("arm64") ? "aarch64" : a;
    }

    private static String cpuLevel() {
        String forced = System.getProperty("numj.cpu");
        if (forced != null && !forced.isBlank()) return forced.strip();
        if (!ARCH.equals("x86_64")) return "baseline";
        try {
            if (OS.equals("windows")) {
                try (Arena a = Arena.ofConfined()) {
                    SymbolLookup k32 = SymbolLookup.libraryLookup("kernel32", a);
                    var h = Linker.nativeLinker().downcallHandle(k32.find("IsProcessorFeaturePresent").orElseThrow(),
                            FunctionDescriptor.of(JAVA_INT, JAVA_INT));
                    final int PF_AVX2_INSTRUCTIONS_AVAILABLE = 40;
                    return (int) h.invokeExact(PF_AVX2_INSTRUCTIONS_AVAILABLE) != 0 ? "x86-64-v3" : "baseline";
                }
            }
            if (OS.equals("linux")) {
                for (String line : Files.readAllLines(Path.of("/proc/cpuinfo"))) {
                    if (!line.startsWith("flags")) continue;
                    List<String> f = List.of(line.substring(line.indexOf(':') + 1).trim().split("\\s+"));
                    return f.containsAll(List.of("avx2", "fma", "bmi1", "bmi2", "movbe", "f16c", "abm"))
                            ? "x86-64-v3" : "baseline";
                }
            }
        } catch (Throwable t) {
            // fall through: detection failure means the safe choice
        }
        return "baseline";
    }
}
