package lowrank;

final class Svd {
    private static final int MAX_SWEEPS = 100;

    private Svd() {
    }

    static ThinSvd compute(Matrix matrix) {
        int rows = matrix.rows();
        int columns = matrix.columns();
        int thin = Math.min(rows, columns);
        double scale = matrix.frobeniusNorm();
        double jacobiTolerance = 1.0e-14 * Math.max(1.0, scale);
        double[][] v = identity(columns);
        double[][] right = columns(matrix.internalValues(), rows, columns);

        for (int sweep = 0; sweep < MAX_SWEEPS; sweep++) {
            double offDiagonal = 0.0;
            for (int p = 0; p < columns - 1; p++) {
                for (int q = p + 1; q < columns; q++) {
                    double alpha = dot(right[p], right[p]);
                    double beta = dot(right[q], right[q]);
                    double gamma = dot(right[p], right[q]);
                    offDiagonal = Math.max(offDiagonal, Math.abs(gamma));
                    if (Math.abs(gamma) <= 1.0e-15 * Math.max(Math.max(alpha, beta), 1.0)) {
                        continue;
                    }

                    double zeta = (beta - alpha) / (2.0 * gamma);
                    double tangent = zeta == 0.0 ? 1.0 : Math.signum(zeta) /
                            (Math.abs(zeta) + Math.sqrt(1.0 + zeta * zeta));
                    double cosine = 1.0 / Math.sqrt(1.0 + tangent * tangent);
                    double sine = tangent * cosine;
                    rotateColumns(right, p, q, cosine, sine);
                    rotateColumns(v, p, q, cosine, sine);
                }
            }
            if (offDiagonal <= jacobiTolerance) {
                break;
            }
        }

        double[] norms = new double[columns];
        for (int j = 0; j < columns; j++) {
            norms[j] = norm(right[j]);
        }
        Integer[] order = sortedDescending(norms);
        double numericalZero = scale == 0.0
                ? 0.0
                : 8.0 * Math.ulp(scale) * Math.max(1.0, thin);

        int rank = 0;
        for (double norm : norms) {
            if (norm > numericalZero) {
                rank++;
            }
        }

        double[] values = new double[rank];
        double[][] u = new double[rows][rank];
        double[][] vt = new double[columns][rank];
        int write = 0;
        for (Integer column : order) {
            if (write == rank) {
                break;
            }
            if (norms[column] <= numericalZero) {
                continue;
            }
            values[write] = norms[column];
            for (int i = 0; i < rows; i++) {
                u[i][write] = right[column][i] / norms[column];
            }
            for (int i = 0; i < columns; i++) {
                vt[i][write] = v[column][i];
            }
            write++;
        }

        return new ThinSvd(values, u, vt, thin - rank);
    }

    private static double[][] identity(int size) {
        double[][] result = new double[size][size];
        for (int i = 0; i < size; i++) {
            result[i][i] = 1.0;
        }
        return result;
    }

    private static double[][] columns(double[] rowMajor, int rows, int columns) {
        double[][] result = new double[columns][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result[j][i] = rowMajor[i * columns + j];
            }
        }
        return result;
    }

    private static void rotateColumns(double[][] matrix, int p, int q,
                                      double cosine, double sine) {
        for (int i = 0; i < matrix[p].length; i++) {
            double x = matrix[p][i];
            double y = matrix[q][i];
            matrix[p][i] = cosine * x - sine * y;
            matrix[q][i] = sine * x + cosine * y;
        }
    }

    private static double dot(double[] x, double[] y) {
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            sum += x[i] * y[i];
        }
        return sum;
    }

    private static double norm(double[] vector) {
        return Math.sqrt(dot(vector, vector));
    }

    private static Integer[] sortedDescending(double[] values) {
        Integer[] indices = new Integer[values.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        java.util.Arrays.sort(indices, (a, b) -> Double.compare(values[b], values[a]));
        return indices;
    }
}
