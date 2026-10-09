package numj;

/**
 * Prints which native library numj loaded and why (diagnostics):
 * {@code java --enable-native-access=ALL-UNNAMED -cp <numj jars> numj.NativeInfo}.
 */
public final class NativeInfo {
    private NativeInfo() {}

    public static void main(String[] args) {
        System.out.println("library " + NumJ.libraryPath() + " | platform " + NativeLoader.OS + "-" + NativeLoader.ARCH
                + " | cpu level " + NativeLoader.CPU_LEVEL + " | build " + NumJ.buildInfo().replaceAll(" -J .*", ""));
    }
}
