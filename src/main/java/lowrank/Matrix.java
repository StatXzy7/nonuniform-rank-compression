package lowrank;

import java.util.Arrays;
import java.util.Objects;

public final class Matrix {
    private final int rows;
    private final int columns;
    private final double[] values;

    public Matrix(int rows, int columns, double... values) {
        if (rows <= 0 || columns <= 0) {
            throw new IllegalArgumentException("matrix dimensions must be positive");
        }
        if (rows > 12 || columns > 12) {
            throw new IllegalArgumentException("matrix dimensions must not exceed 12");
        }
        Objects.requireNonNull(values, "values");
        if (values.length != rows * columns) {
            throw new IllegalArgumentException("values length does not match dimensions");
        }
        for (double value : values) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException("matrix values must be finite");
            }
        }
        this.rows = rows;
        this.columns = columns;
        this.values = values.clone();
    }

    public static Matrix zeros(int rows, int columns) {
        return new Matrix(rows, columns, new double[rows * columns]);
    }

    public int rows() {
        return rows;
    }

    public int columns() {
        return columns;
    }

    public int elementCount() {
        return rows * columns;
    }

    public double get(int row, int column) {
        if (row < 0 || row >= rows || column < 0 || column >= columns) {
            throw new IndexOutOfBoundsException("matrix index is out of bounds");
        }
        return values[row * columns + column];
    }

    public double[] toRowMajor() {
        return values.clone();
    }

    public Matrix multiply(Matrix right) {
        Objects.requireNonNull(right, "right");
        if (columns != right.rows) {
            throw new IllegalArgumentException("matrices have incompatible shapes for multiplication");
        }
        if (right.columns > 12) {
            throw new IllegalArgumentException("matrix dimensions must not exceed 12");
        }
        double[] result = new double[rows * right.columns];
        for (int i = 0; i < rows; i++) {
            for (int k = 0; k < columns; k++) {
                double leftValue = values[i * columns + k];
                for (int j = 0; j < right.columns; j++) {
                    result[i * right.columns + j] += leftValue * right.get(k, j);
                }
            }
        }
        return new Matrix(rows, right.columns, result);
    }

    public boolean approximatelyEquals(Matrix other, double tolerance) {
        if (rows != other.rows || columns != other.columns) {
            return false;
        }
        for (int i = 0; i < values.length; i++) {
            if (Math.abs(values[i] - other.values[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    public boolean allZero() {
        for (double value : values) {
            if (value != 0.0) {
                return false;
            }
        }
        return true;
    }

    double frobeniusNorm() {
        double sum = 0.0;
        for (double value : values) {
            sum += value * value;
        }
        return Math.sqrt(sum);
    }

    double[] internalValues() {
        return values;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append(rows).append('x').append(columns).append('[');
        for (int i = 0; i < rows; i++) {
            if (i > 0) {
                builder.append(';');
            }
            for (int j = 0; j < columns; j++) {
                if (j > 0) {
                    builder.append(',');
                }
                builder.append(values[i * columns + j]);
            }
        }
        return builder.append(']').toString();
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof Matrix)) {
            return false;
        }
        Matrix matrix = (Matrix) other;
        return rows == matrix.rows && columns == matrix.columns && Arrays.equals(values, matrix.values);
    }

    @Override
    public int hashCode() {
        return Objects.hash(rows, columns, Arrays.hashCode(values));
    }
}
