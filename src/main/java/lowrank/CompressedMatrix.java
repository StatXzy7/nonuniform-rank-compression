package lowrank;

import java.util.Objects;

public final class CompressedMatrix {
    private final String name;
    private final int rows;
    private final int columns;
    private final int rank;
    private final int zeroDirections;
    private final int deletedPositiveDirections;
    private final int factorElementCount;
    private final int storedElementCount;
    private final int originalElementCount;
    private final boolean factorStored;
    private final double[] leftFactor;
    private final double[] rightFactor;
    private final Matrix denseMatrix;
    private final double absoluteError;
    private final double relativeError;

    CompressedMatrix(String name, int rows, int columns, int rank, int zeroDirections,
                     int deletedPositiveDirections,
                     boolean factorStored, double[] leftFactor, double[] rightFactor,
                     Matrix denseMatrix, double absoluteError, double relativeError) {
        this.name = name;
        this.rows = rows;
        this.columns = columns;
        this.rank = rank;
        this.zeroDirections = zeroDirections;
        this.deletedPositiveDirections = deletedPositiveDirections;
        this.originalElementCount = rows * columns;
        this.factorElementCount = rank * (rows + columns);
        this.storedElementCount = factorStored ? factorElementCount : originalElementCount;
        this.factorStored = factorStored;
        this.leftFactor = leftFactor == null ? new double[0] : leftFactor.clone();
        this.rightFactor = rightFactor == null ? new double[0] : rightFactor.clone();
        this.denseMatrix = denseMatrix;
        this.absoluteError = absoluteError;
        this.relativeError = relativeError;
    }

    static CompressedMatrix factors(NamedMatrix source, int rank, int zeroDirections,
                                    int deletedPositiveDirections, ThinSvd svd,
                                    boolean[] retained, Matrix truncated) {
        Matrix original = source.matrix();
        double[] left = new double[original.rows() * rank];
        double[] right = new double[rank * original.columns()];
        int write = 0;
        for (int j = 0; j < svd.positiveRank(); j++) {
            if (!retained[j]) {
                continue;
            }
            double singularValue = svd.singularValue(j);
            for (int i = 0; i < original.rows(); i++) {
                left[i * rank + write] = svd.leftVectors()[i][j] * singularValue;
            }
            for (int jRow = 0; jRow < original.columns(); jRow++) {
                right[write * original.columns() + jRow] = svd.rightVectors()[jRow][j];
            }
            write++;
        }
        double[] errors = errors(original, truncated);
        return new CompressedMatrix(source.name(), original.rows(), original.columns(),
                rank, zeroDirections, deletedPositiveDirections,
                true, left, right, null, errors[0], errors[1]);
    }

    static CompressedMatrix dense(NamedMatrix source, int rank, int zeroDirections,
                                  int deletedPositiveDirections, Matrix truncated) {
        double[] errors = errors(source.matrix(), truncated);
        return new CompressedMatrix(source.name(), source.matrix().rows(), source.matrix().columns(),
                rank, zeroDirections, deletedPositiveDirections,
                false, null, null, truncated, errors[0], errors[1]);
    }

    private static double[] errors(Matrix original, Matrix reconstructed) {
        double sum = 0.0;
        for (int i = 0; i < original.rows(); i++) {
            for (int j = 0; j < original.columns(); j++) {
                double difference = original.get(i, j) - reconstructed.get(i, j);
                sum += difference * difference;
            }
        }
        double originalNorm = original.frobeniusNorm();
        return new double[] {Math.sqrt(sum), originalNorm == 0.0 ? 0.0 : Math.sqrt(sum) / originalNorm};
    }

    public String name() {
        return name;
    }

    public int rows() {
        return rows;
    }

    public int columns() {
        return columns;
    }

    public int retainedRank() {
        return rank;
    }

    public int thinDirections() {
        return rank + deletedPositiveDirections + zeroDirections;
    }

    public int deletedDirections() {
        return deletedPositiveDirections + zeroDirections;
    }

    public int zeroDirections() {
        return zeroDirections;
    }

    public int deletedPositiveDirections() {
        return deletedPositiveDirections;
    }

    public int factorElementCount() {
        return factorElementCount;
    }

    public int storedElementCount() {
        return storedElementCount;
    }

    public int originalElementCount() {
        return originalElementCount;
    }

    public boolean factorStored() {
        return factorStored;
    }

    public double absoluteReconstructionError() {
        return absoluteError;
    }

    public double relativeReconstructionError() {
        return relativeError;
    }

    public double storageSavings() {
        return (double) (originalElementCount - storedElementCount) / originalElementCount;
    }

    public Matrix reconstruct() {
        if (!factorStored) {
            return denseMatrix;
        }
        if (rank == 0) {
            return Matrix.zeros(rows, columns);
        }
        Matrix left = new Matrix(rows, rank, leftFactor);
        Matrix right = new Matrix(rank, columns, rightFactor);
        return left.multiply(right);
    }

    double[] leftFactor() {
        return leftFactor.clone();
    }

    double[] rightFactor() {
        return rightFactor.clone();
    }

    Matrix denseMatrix() {
        return denseMatrix;
    }

    @Override
    public String toString() {
        return name() + ": rank=" + rank + ", deleted=" + deletedDirections() + '/' + thinDirections()
                + ", relativeError=" + relativeError + ", factorElements=" + factorElementCount
                + ", storedElements=" + storedElementCount + ", originalElements=" + originalElementCount
                + ", storage=" + (factorStored ? "factors" : "dense-fallback");
    }
}
