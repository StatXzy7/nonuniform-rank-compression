package lowrank;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

final class CompressionEngine {
    static final double SPECTRAL_EQUALITY_EPS = 1.0e-12;

    private CompressionEngine() {
    }

    static CompressionResult compressByThreshold(double threshold, List<NamedMatrix> inputs) {
        validateUnitInterval(threshold, "threshold");
        List<Spec> specs = validate(inputs);
        return build(CompressionMode.THRESHOLD, threshold, Double.NaN, threshold, specs);
    }

    static CompressionResult compressByTargetFraction(double target, List<NamedMatrix> inputs) {
        validateUnitInterval(target, "target deletion fraction");
        List<Spec> specs = validate(inputs);
        double bestThreshold = 0.0;
        int bestDeleted = countDeleted(specs, 0.0, false);
        double bestDifference = Math.abs((double) bestDeleted / totalDirections(specs) - target);

        for (double candidate : candidateThresholds(specs)) {
            int deleted = countDeleted(specs, candidate, false);
            double actual = (double) deleted / totalDirections(specs);
            double difference = Math.abs(actual - target);
            if (difference < bestDifference) {
                bestDifference = difference;
                bestDeleted = deleted;
                bestThreshold = candidate;
            }
        }
        return build(CompressionMode.TARGET_DELETION, Double.NaN, target, bestThreshold, specs);
    }

    private static CompressionResult build(CompressionMode mode, double requestedThreshold,
                                           double target, double threshold, List<Spec> specs) {
        List<CompressedMatrix> compressed = new ArrayList<>();
        int totalDeleted = 0;
        int totalDirections = 0;
        for (Spec spec : specs) {
            ThinSvd svd = spec.svd();
            boolean[] retained = new boolean[svd.positiveRank()];
            int deletedPositive = 0;
            for (int i = 0; i < svd.positiveRank(); i++) {
                double normalized = spec.normalized()[i];
                boolean equalsThreshold = Math.abs(normalized - threshold) <= SPECTRAL_EQUALITY_EPS;
                retained[i] = mode == CompressionMode.THRESHOLD
                        ? normalized > threshold || equalsThreshold
                        : normalized > threshold;
                if (!retained[i]) {
                    deletedPositive++;
                }
            }
            int rank = svd.positiveRank() - deletedPositive;
            Matrix truncated = reconstruct(spec.source().matrix(), svd, retained);
            int factorElements = rank * (spec.source().matrix().rows() + spec.source().matrix().columns());
            boolean factorHasGain = factorElements < spec.source().matrix().elementCount();
            boolean useDenseOriginal = !factorHasGain && deletedPositive == 0 && rank > 0;
            boolean useFactors = factorHasGain;
            CompressedMatrix matrix = useFactors
                    ? CompressedMatrix.factors(spec.source(), rank, svd.zeroDirections(),
                    deletedPositive, svd, retained, truncated)
                    : CompressedMatrix.dense(spec.source(), rank, svd.zeroDirections(),
                    deletedPositive, useDenseOriginal ? spec.source().matrix() : truncated);
            compressed.add(matrix);
            totalDeleted += deletedPositive + svd.zeroDirections();
            totalDirections += svd.thinDirections();
        }
        return new CompressionResult(mode, requestedThreshold, target, threshold,
                totalDirections, totalDeleted, compressed);
    }

    private static Matrix reconstruct(Matrix original, ThinSvd svd, boolean[] retained) {
        double[] values = new double[original.elementCount()];
        for (int component = 0; component < svd.positiveRank(); component++) {
            if (!retained[component]) {
                continue;
            }
            double singularValue = svd.singularValue(component);
            for (int i = 0; i < original.rows(); i++) {
                double left = svd.leftVectors()[i][component] * singularValue;
                for (int j = 0; j < original.columns(); j++) {
                    values[i * original.columns() + j] += left * svd.rightVectors()[j][component];
                }
            }
        }
        return new Matrix(original.rows(), original.columns(), values);
    }

    private static List<Double> candidateThresholds(List<Spec> specs) {
        List<Double> normalizedValues = new ArrayList<>();
        for (Spec spec : specs) {
            for (double value : spec.normalized()) {
                normalizedValues.add(value);
            }
        }
        normalizedValues.sort(Double::compare);
        List<double[]> clusters = new ArrayList<>();
        for (double value : normalizedValues) {
            if (clusters.isEmpty() || value - clusters.get(clusters.size() - 1)[1] > SPECTRAL_EQUALITY_EPS) {
                clusters.add(new double[] {value, value});
            } else {
                clusters.get(clusters.size() - 1)[1] = value;
            }
        }
        List<Double> thresholds = new ArrayList<>();
        for (int i = 0; i + 1 < clusters.size(); i++) {
            thresholds.add((clusters.get(i)[1] + clusters.get(i + 1)[0]) / 2.0);
        }
        thresholds.add(1.0);
        return thresholds;
    }

    private static int countDeleted(List<Spec> specs, double threshold, boolean unused) {
        int deleted = 0;
        for (Spec spec : specs) {
            deleted += spec.svd().zeroDirections();
            for (double normalized : spec.normalized()) {
                if (normalized < threshold) {
                    deleted++;
                }
            }
        }
        return deleted;
    }

    private static int totalDirections(List<Spec> specs) {
        int total = 0;
        for (Spec spec : specs) {
            total += spec.svd().thinDirections();
        }
        return total;
    }

    private static List<Spec> validate(List<NamedMatrix> inputs) {
        Objects.requireNonNull(inputs, "inputs");
        if (inputs.isEmpty()) {
            throw new IllegalArgumentException("input batch must not be empty");
        }
        Set<String> names = new HashSet<>();
        List<Spec> specs = new ArrayList<>();
        for (NamedMatrix input : inputs) {
            Objects.requireNonNull(input, "named matrix");
            if (!names.add(input.name())) {
                throw new IllegalArgumentException("duplicate matrix name: " + input.name());
            }
            ThinSvd svd = Svd.compute(input.matrix());
            double[] normalized = new double[svd.positiveRank()];
            if (svd.positiveRank() > 0) {
                double maximum = svd.singularValue(0);
                for (int i = 0; i < normalized.length; i++) {
                    normalized[i] = svd.singularValue(i) / maximum;
                }
            }
            specs.add(new Spec(input, svd, normalized));
        }
        return specs;
    }

    private static void validateUnitInterval(double value, String label) {
        if (!Double.isFinite(value) || value < 0.0 || value > 1.0) {
            throw new IllegalArgumentException(label + " must be finite and within [0, 1]");
        }
    }

    private record Spec(NamedMatrix source, ThinSvd svd, double[] normalized) {
    }
}
