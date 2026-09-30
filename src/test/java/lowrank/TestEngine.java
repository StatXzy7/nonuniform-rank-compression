package lowrank;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public final class TestEngine {
    private int passed;

    public static void main(String[] args) throws Exception {
        new TestEngine().run();
    }

    private void run() throws Exception {
        differentSpectraProduceDifferentRanks();
        thresholdEqualityKeepsDirectionWithoutSplitting();
        discreteBudgetTiePrefersLessDeletion();
        repeatedSpectralValuesAreNotSplitForBudget();
        zeroMatrixIsRankZeroAndDeletesAllDirections();
        reconstructionErrorMatchesDroppedEnergy();
        signAndBasisIndependent();
        jacobiSvdReconstructsRandomShapes();
        saveLoadPreservesReconstructionAndMultiplication();
        corruptFileIsRejectedAndCompleteFileIsNotReplaced();
        invalidBatchesAreRejected();

        System.out.println("PASS: " + passed + " automated checks");
    }

    private void differentSpectraProduceDifferentRanks() {
        NamedMatrix rankThree = new NamedMatrix("rank_three", new Matrix(3, 3,
                1.0, 0.0, 0.0,
                0.0, 0.5, 0.0,
                0.0, 0.0, 0.25));
        NamedMatrix rankTwo = new NamedMatrix("rank_two", new Matrix(3, 3,
                1.0, 0.0, 0.0,
                0.0, 0.5, 0.0,
                0.0, 0.0, 0.0));
        NamedMatrix rankOne = new NamedMatrix("rank_one", new Matrix(3, 3,
                1.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0));

        CompressionResult result = CompressionResult.compressByThreshold(
                0.3, rankThree, rankTwo, rankOne);
        check(result.matrix("rank_three").retainedRank() == 2, "0.5 direction survives but 0.25 is removed");
        check(result.matrix("rank_two").retainedRank() == 2, "zero directions do not affect nonzero rank");
        check(result.matrix("rank_one").retainedRank() == 1, "rank-one matrix remains rank one");
        check(result.deletedDirections() == 4, "global deleted count uses all thin directions");
        check(result.actualDeletedFraction() == 4.0 / 9.0, "zero directions are included in denominator");
    }

    private void thresholdEqualityKeepsDirectionWithoutSplitting() {
        NamedMatrix equalPair = new NamedMatrix("equal_pair", new Matrix(2, 2,
                1.0, 0.0,
                0.0, 0.5));
        CompressionResult result = CompressionResult.compressByThreshold(0.5, equalPair);

        check(result.matrix("equal_pair").retainedRank() == 2, "nonzero singular value equal to threshold is kept");
        check(result.deletedDirections() == 0, "keeping one equal value keeps every equal value");
        check(result.requestedThreshold() == 0.5 && result.effectiveThreshold() == 0.5,
                "direct threshold mode reports the requested threshold as effective");
    }

    private void discreteBudgetTiePrefersLessDeletion() {
        NamedMatrix first = new NamedMatrix("a", new Matrix(2, 2,
                1.0, 0.0,
                0.0, 0.75));
        NamedMatrix second = new NamedMatrix("b", new Matrix(2, 2,
                1.0, 0.0,
                0.0, 0.25));

        CompressionResult result = CompressionResult.compressByTargetFraction(
                0.375, first, second);

        check(result.deletedDirections() == 1 || result.deletedDirections() == 2,
                "reachable fractions around the target are 1/4 and 2/4");
        check(result.deletedDirections() == 1, "equidistant discrete budgets choose fewer deletions");
        check(result.effectiveThreshold() == 0.5, "selected threshold lies between grouped spectral values");
        check(result.targetDeletedFraction() == 0.375, "target fraction is reported");
        check(result.actualDeletedFraction() == 0.25 && result.deviation() == 0.125,
                "deviation is reported for the nearest reachable result");
    }

    private void zeroMatrixIsRankZeroAndDeletesAllDirections() {
        NamedMatrix zero = new NamedMatrix("zero", Matrix.zeros(2, 3));
        CompressionResult result = CompressionResult.compressByThreshold(0.0, zero);
        CompressedMatrix compressed = result.matrix("zero");

        check(compressed.retainedRank() == 0, "all-zero matrix is always rank zero");
        check(compressed.zeroDirections() == 2, "thin zero directions are counted");
        check(compressed.deletedDirections() == 2, "all-zero matrix contributes deleted directions");
        check(compressed.relativeReconstructionError() == 0.0, "zero matrix has no division by zero");
        check(compressed.absoluteReconstructionError() == 0.0, "zero reconstruction is exact");
        check(compressed.reconstruct().allZero(), "zero reconstruction contains only zeros");
        check(compressed.factorStored() && compressed.storedElementCount() == 0,
                "rank-zero factors have zero stored entries");
    }

    private void repeatedSpectralValuesAreNotSplitForBudget() {
        NamedMatrix matrix = new NamedMatrix("repeated", new Matrix(3, 3,
                4.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0));
        CompressionResult result = CompressionResult.compressByTargetFraction(0.5, matrix);

        check(result.matrix("repeated").retainedRank() == 1,
                "equal 0.25 directions are removed together even if rank two is closer to a budget");
        check(result.deletedDirections() == 2, "reachable deletion keeps equal spectrum directions together");
        check(result.actualDeletedFraction() == 2.0 / 3.0, "global thin denominator remains three");
    }

    private void reconstructionErrorMatchesDroppedEnergy() {
        NamedMatrix matrix = new NamedMatrix("energy", new Matrix(2, 2,
                3.0, 0.0,
                0.0, 4.0));
        CompressionResult result = CompressionResult.compressByThreshold(
                0.8, matrix);
        CompressedMatrix compressed = result.matrix("energy");

        check(compressed.retainedRank() == 1, "the normalized 0.75 direction is removed");
        check(close(compressed.absoluteReconstructionError(), 3.0, 1.0e-12),
                "absolute error equals the dropped singular value");
        check(close(compressed.relativeReconstructionError(), 0.6, 1.0e-12),
                "relative error is dropped Frobenius energy over original energy");
        check(compressed.factorElementCount() == 4, "rank-one 2x2 factors contain four values");
        check(!compressed.factorStored(), "equal-size factors do not produce a storage gain");
        check(compressed.storedElementCount() == 4, "no-gain result uses dense fallback storage");
        check(compressed.storageSavings() == 0.0, "real storage gain is separate from spectral truncation");
    }

    private void signAndBasisIndependent() {
        NamedMatrix original = new NamedMatrix("original", new Matrix(2, 2,
                1.0, 0.0,
                0.0, 0.5));
        Matrix negative = new Matrix(2, 2,
                -1.0, 0.0,
                0.0, -0.5);
        Matrix swapped = new Matrix(2, 2,
                0.5, 0.0,
                0.0, 1.0);
        CompressionResult result = CompressionResult.compressByThreshold(
                0.75, original,
                new NamedMatrix("negative", negative),
                new NamedMatrix("swapped", swapped));

        for (String name : List.of("original", "negative", "swapped")) {
            check(result.matrix(name).retainedRank() == 1, name + " has one retained direction");
        }
    }

    private void jacobiSvdReconstructsRandomShapes() {
        java.util.Random random = new java.util.Random(20261001L);
        int[][] shapes = {{1, 1}, {1, 12}, {12, 1}, {3, 7}, {8, 5}, {12, 12}};
        for (int[] shape : shapes) {
            double[] values = new double[shape[0] * shape[1]];
            for (int i = 0; i < values.length; i++) {
                values[i] = random.nextDouble() * 20.0 - 10.0;
            }
            Matrix matrix = new Matrix(shape[0], shape[1], values);
            ThinSvd svd = Svd.compute(matrix);
            check(svd.positiveRank() == Math.min(shape[0], shape[1]),
                    "random full-rank matrix has the expected thin rank");
            double[] reconstructedValues = new double[values.length];
            for (int component = 0; component < svd.positiveRank(); component++) {
                double singularValue = svd.singularValue(component);
                for (int i = 0; i < shape[0]; i++) {
                    for (int j = 0; j < shape[1]; j++) {
                        reconstructedValues[i * shape[1] + j] +=
                                svd.leftVectors()[i][component] * singularValue
                                        * svd.rightVectors()[j][component];
                    }
                }
            }
            Matrix reconstructed = new Matrix(shape[0], shape[1], reconstructedValues);
            double tolerance = 1.0e-10 * Math.max(1.0, matrix.frobeniusNorm());
            check(reconstructed.approximatelyEquals(matrix, tolerance),
                    "Jacobi SVD reconstructs " + shape[0] + "x" + shape[1]);
        }
    }

    private void saveLoadPreservesReconstructionAndMultiplication() throws Exception {
        NamedMatrix left = new NamedMatrix("left", new Matrix(3, 2,
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0));
        NamedMatrix right = new NamedMatrix("right", new Matrix(2, 2,
                1.0, 0.0,
                0.0, 2.0));
        CompressionResult before = CompressionResult.compressByThreshold(0.0, left, right);

        Path file = Files.createTempFile("nrcb-valid-", ".bin");
        CompressionResult.save(file, before);
        CompressionResult after = CompressionResult.load(file);

        Path secondFile = Files.createTempFile("nrcb-roundtrip-", ".bin");
        CompressionResult.save(secondFile, after);
        CompressionResult roundTrip = CompressionResult.load(secondFile);
        for (String name : List.of("left", "right")) {
            checkExact(before.reconstruct(name), after.reconstruct(name),
                    name + " reconstruction is identical after reload");
            checkExact(before.reconstruct(name), roundTrip.reconstruct(name),
                    name + " reconstruction survives saving a loaded result");
        }
        checkExact(before.multiply("left", "right"), after.multiply("left", "right"),
                "multiplication is identical after reload");
        Matrix expected = new Matrix(3, 2,
                1.0, 4.0,
                3.0, 8.0,
                5.0, 12.0);
        check(after.multiply("left", "right").approximatelyEquals(expected, 1.0e-12),
                "loaded reconstructed factors multiply correctly");
    }

    private void corruptFileIsRejectedAndCompleteFileIsNotReplaced() throws Exception {
        NamedMatrix matrix = new NamedMatrix("matrix", new Matrix(2, 2,
                1.0, 2.0,
                3.0, 4.0));
        CompressionResult result = CompressionResult.compressByThreshold(0.0, matrix);
        Path file = Files.createTempFile("nrcb-corrupt-", ".bin");
        CompressionResult.save(file, result);
        byte[] complete = Files.readAllBytes(file);

        byte[] damaged = complete.clone();
        damaged[damaged.length - 1] ^= 1;
        Path damagedFile = Files.createTempFile("nrcb-damaged-", ".bin");
        Files.write(damagedFile, damaged);
        expectIllegalArgument(() -> CompressionResult.load(damagedFile), "checksum-damaged file is rejected");

        expectIllegalArgument(() -> CompressionResult.save(file, CompressionResult.compressByThreshold(
                Double.NaN, matrix)), "failed batch is rejected");
        byte[] stillComplete = Files.readAllBytes(file);
        check(java.util.Arrays.equals(complete, stillComplete),
                "failed save never replaces an existing complete result");

        Path directory = Files.createTempDirectory("nrcb-directory-");
        expectIllegalArgument(() -> CompressionResult.save(directory, result),
                "I/O failure during atomic save is reported");
        check(java.util.Arrays.equals(complete, Files.readAllBytes(file)),
                "I/O failure leaves the previously complete result intact");

        CompressionResult rankDeficient = CompressionResult.compressByThreshold(0.0,
                new NamedMatrix("rank_deficient", new Matrix(2, 3,
                        1.0, 2.0, 3.0,
                        2.0, 4.0, 6.0)));
        CompressedMatrix factorGain = rankDeficient.matrix("rank_deficient");
        check(factorGain.factorStored() && factorGain.retainedRank() == 1,
                "zero-direction deletion uses factors when they reduce storage");
        check(factorGain.storedElementCount() == 5 && factorGain.originalElementCount() == 6,
                "factor storage gain is reported separately from spectral deletion");

        CompressionResult noGain = CompressionResult.compressByThreshold(0.0,
                new NamedMatrix("rank_deficient_square", new Matrix(2, 2,
                        1.0, 2.0,
                        2.0, 4.0)));
        CompressedMatrix denseOriginal = noGain.matrix("rank_deficient_square");
        check(!denseOriginal.factorStored() && denseOriginal.retainedRank() == 1,
                "only zero-direction deletion with no factor gain uses dense storage");
        check(denseOriginal.reconstruct().approximatelyEquals(new Matrix(2, 2,
                        1.0, 2.0,
                        2.0, 4.0), 0.0),
                "dense zero-direction fallback stores the original matrix exactly");
    }

    private void invalidBatchesAreRejected() throws Exception {
        NamedMatrix matrix = new NamedMatrix("matrix", new Matrix(1, 1, 1.0));
        expectIllegalArgument(() -> new Matrix(0, 1), "empty matrix rows are rejected");
        expectIllegalArgument(() -> new Matrix(1, 13, new double[13]),
                "dimensions greater than twelve are rejected");
        expectIllegalArgument(() -> new Matrix(1, 1, Double.NaN), "non-finite values are rejected");
        expectIllegalArgument(() -> CompressionResult.compressByThreshold(-0.1, matrix),
                "negative threshold is rejected");
        expectIllegalArgument(() -> CompressionResult.compressByThreshold(1.1, matrix),
                "threshold above one is rejected");
        expectIllegalArgument(() -> CompressionResult.compressByTargetFraction(Double.POSITIVE_INFINITY, matrix),
                "non-finite target fraction is rejected");
        expectIllegalArgument(() -> CompressionResult.compressByThreshold(0.1),
                "empty batch is rejected");
        expectIllegalArgument(() -> CompressionResult.compressByThreshold(
                0.1, matrix, new NamedMatrix("matrix", Matrix.zeros(1, 1))),
                "duplicate names are rejected");
    }

    private void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
        passed++;
    }

    private void checkExact(Matrix expected, Matrix actual, String message) {
        double[] expectedValues = expected.toRowMajor();
        double[] actualValues = actual.toRowMajor();
        check(expected.rows() == actual.rows() && expected.columns() == actual.columns()
                && java.util.Arrays.equals(expectedValues, actualValues), message);
    }

    private boolean close(double expected, double actual, double tolerance) {
        return Math.abs(expected - actual) <= tolerance;
    }

    private void expectIllegalArgument(ThrowingRunnable runnable, String message) throws Exception {
        try {
            runnable.run();
        } catch (IllegalArgumentException expected) {
            passed++;
            return;
        }
        throw new AssertionError(message);
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}
