
import lowrank.CompressedMatrix;
import lowrank.CompressionResult;
import lowrank.Matrix;
import lowrank.NamedMatrix;

public final class Example {
    private Example() {
    }

    public static void main(String[] args) {
        NamedMatrix lowRank = new NamedMatrix("low_rank", new Matrix(3, 3,
                1.0, 2.0, 3.0,
                2.0, 4.0, 6.0,
                0.0, 0.0, 0.0));
        NamedMatrix smallDirection = new NamedMatrix("small_direction", new Matrix(2, 2,
                1.0, 0.0,
                0.0, 0.05));
        NamedMatrix zero = new NamedMatrix("zero", Matrix.zeros(2, 3));

        CompressionResult thresholdResult = CompressionResult.compressByThreshold(
                0.1, lowRank, smallDirection, zero);
        System.out.println("Threshold batch:");
        print(thresholdResult);

        CompressionResult targetResult = CompressionResult.compressByTargetFraction(
                0.5, lowRank, smallDirection, zero);
        System.out.println("Target-deletion batch:");
        print(targetResult);
    }

    private static void print(CompressionResult result) {
        System.out.println(result);
        for (CompressedMatrix matrix : result.matrices().values()) {
            Matrix reconstructed = matrix.reconstruct();
            System.out.println("  reconstructed " + matrix.name() + " = " + reconstructed);
        }
    }
}
