package lowrank;

import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class CompressionResult {
    private final CompressionMode mode;
    private final double requestedThreshold;
    private final double targetDeletedFraction;
    private final double effectiveThreshold;
    private final int totalThinDirections;
    private final int deletedDirections;
    private final double actualDeletedFraction;
    private final double deviation;
    private final Map<String, CompressedMatrix> matrices;

    CompressionResult(CompressionMode mode, double requestedThreshold, double targetDeletedFraction,
                      double effectiveThreshold, int totalThinDirections, int deletedDirections,
                      List<CompressedMatrix> matrices) {
        this.mode = Objects.requireNonNull(mode, "mode");
        this.requestedThreshold = requestedThreshold;
        this.targetDeletedFraction = targetDeletedFraction;
        this.effectiveThreshold = effectiveThreshold;
        this.totalThinDirections = totalThinDirections;
        this.deletedDirections = deletedDirections;
        this.actualDeletedFraction = (double) deletedDirections / totalThinDirections;
        this.deviation = mode == CompressionMode.TARGET_DELETION
                ? Math.abs(this.actualDeletedFraction - targetDeletedFraction)
                : 0.0;
        Map<String, CompressedMatrix> ordered = new LinkedHashMap<>();
        for (CompressedMatrix matrix : matrices) {
            ordered.put(matrix.name(), matrix);
        }
        this.matrices = Collections.unmodifiableMap(ordered);
    }

    public static CompressionResult compressByThreshold(double threshold, NamedMatrix... inputs) {
        return CompressionEngine.compressByThreshold(threshold, List.of(inputs));
    }

    public static CompressionResult compressByThreshold(double threshold,
                                                        List<NamedMatrix> inputs) {
        return CompressionEngine.compressByThreshold(threshold, inputs);
    }

    public static CompressionResult compressByTargetFraction(double targetDeletedFraction,
                                                             NamedMatrix... inputs) {
        return CompressionEngine.compressByTargetFraction(targetDeletedFraction, List.of(inputs));
    }

    public static CompressionResult compressByTargetFraction(double targetDeletedFraction,
                                                             List<NamedMatrix> inputs) {
        return CompressionEngine.compressByTargetFraction(targetDeletedFraction, inputs);
    }

    public static void save(Path file, CompressionResult result) {
        ResultStore.save(file, result);
    }

    public static CompressionResult load(Path file) {
        return ResultStore.load(file);
    }

    public CompressionMode mode() {
        return mode;
    }

    public double requestedThreshold() {
        return requestedThreshold;
    }

    public double targetDeletedFraction() {
        return targetDeletedFraction;
    }

    public double effectiveThreshold() {
        return effectiveThreshold;
    }
    public int totalThinDirections() {
        return totalThinDirections;
    }

    public int deletedDirections() {
        return deletedDirections;
    }

    public double actualDeletedFraction() {
        return actualDeletedFraction;
    }

    public double deviation() {
        return deviation;
    }

    public Map<String, CompressedMatrix> matrices() {
        return matrices;
    }

    public CompressedMatrix matrix(String name) {
        CompressedMatrix matrix = matrices.get(name);
        if (matrix == null) {
            throw new IllegalArgumentException("unknown matrix name: " + name);
        }
        return matrix;
    }

    public Matrix reconstruct(String name) {
        return matrix(name).reconstruct();
    }

    public Matrix multiply(String leftName, String rightName) {
        return matrix(leftName).reconstruct().multiply(matrix(rightName).reconstruct());
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("mode=").append(mode)
                .append(", effectiveThreshold=").append(effectiveThreshold)
                .append(", deleted=").append(deletedDirections).append('/').append(totalThinDirections)
                .append(", actualFraction=").append(actualDeletedFraction);
        if (mode == CompressionMode.TARGET_DELETION) {
            builder.append(", target=").append(targetDeletedFraction)
                    .append(", deviation=").append(deviation);
        }
        builder.append(System.lineSeparator());
        matrices.values().forEach(matrix -> builder.append("  ").append(matrix).append(System.lineSeparator()));
        return builder.toString();
    }
}
