package lowrank;

public final class ThinSvd {
    private final double[] positiveSingularValues;
    private final double[][] leftVectors;
    private final double[][] rightVectors;
    private final int zeroDirections;

    ThinSvd(double[] positiveSingularValues, double[][] leftVectors,
            double[][] rightVectors, int zeroDirections) {
        this.positiveSingularValues = positiveSingularValues;
        this.leftVectors = leftVectors;
        this.rightVectors = rightVectors;
        this.zeroDirections = zeroDirections;
    }

    public int positiveRank() {
        return positiveSingularValues.length;
    }

    public int thinDirections() {
        return positiveSingularValues.length + zeroDirections;
    }

    public double singularValue(int index) {
        return positiveSingularValues[index];
    }

    public double maxSingularValue() {
        return positiveRank() == 0 ? 0.0 : positiveSingularValues[0];
    }

    public double[] positiveSingularValues() {
        return positiveSingularValues.clone();
    }

    double[][] leftVectors() {
        return leftVectors;
    }

    double[][] rightVectors() {
        return rightVectors;
    }

    int zeroDirections() {
        return zeroDirections;
    }
}
